import os
import numpy as np
import pandas as pd
import rasterio
import h5py
from glob import glob
from config import FEATURE_NAMES

"""
Methods to load data and calculate features for classic ML and deep learning models.

build_df -> returns a pandas dataFrame for tabular data use

"""

def _to_db(x, eps=1e-12):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(np.maximum(x, eps))

def _ratio_db(num, den, eps=1e-12):
    den_s = np.maximum(np.abs(den), eps)
    return 10.0 * np.log10(np.maximum(num, eps) / den_s)


def list_aoi_dirs(data_dir):
    """
    Return a sorted list of AOI directories under data_dir.
    """
    return sorted(
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )

def load_stack(aoi_dir):
    """
    Read and stack raster bands for one AOI directory
    Returns:
    stack : np.ndarray of shape (H, W, 27)
            Channels in this exact order:
            FEATURE_NAMES (26) + SD label as the last channel.
    """
    arrs = []

    # Read SAR bands (by fixed index order)
    sar_path = glob(os.path.join(aoi_dir, "*SAR.tif"))[0]
    with rasterio.open(sar_path) as src:
        # linear inputs
        Sigma_VH_lin = src.read(1).astype(np.float32)   # Sigma0_VH
        Gamma_VH_lin = src.read(2).astype(np.float32)   # Gamma0_VH
        Beta_VH_lin  = src.read(3).astype(np.float32)   # Beta0_VH
        Sigma_VV_lin = src.read(4).astype(np.float32)   # Sigma0_VV
        Gamma_VV_lin = src.read(5).astype(np.float32)   # Gamma0_VV
        Beta_VV_lin  = src.read(6).astype(np.float32)   # Beta0_VV
        Gamma_VH2_lin = src.read(7).astype(np.float32)  # Gamma0_VH_2 (RTC)
        Gamma_VV2_lin = src.read(8).astype(np.float32)  # Gamma0_VV_2 (RTC)
        LIA  = src.read(9).astype(np.float32)           # localIncidenceAngle
        IAFE = src.read(10).astype(np.float32)          # incidenceAngleFromEllipsoid

    # Base backscatter in dB
    Sigma_VH = _to_db(Sigma_VH_lin)
    Sigma_VV = _to_db(Sigma_VV_lin)
    Gamma_VH = _to_db(Gamma_VH_lin)
    Gamma_VV = _to_db(Gamma_VV_lin)
    Beta_VH  = _to_db(Beta_VH_lin)
    Beta_VV  = _to_db(Beta_VV_lin)
    Gamma_VH_RTC = _to_db(Gamma_VH2_lin)
    Gamma_VV_RTC = _to_db(Gamma_VV2_lin)

    # Linear sums/differences (linear)
    Sigma_sum      = Sigma_VH_lin + Sigma_VV_lin
    Gamma_sum      = Gamma_VH_lin + Gamma_VV_lin
    Beta_sum       = Beta_VH_lin  + Beta_VV_lin
    Gamma_RTC_sum  = Gamma_VH2_lin + Gamma_VV2_lin

    Sigma_diff     = Sigma_VH_lin - Sigma_VV_lin
    Gamma_diff     = Gamma_VH_lin - Gamma_VV_lin
    Beta_diff      = Beta_VH_lin  - Beta_VV_lin
    Gamma_RTC_diff = Gamma_VH2_lin - Gamma_VV2_lin

    # Ratios in dB
    Sigma_ratio     = _ratio_db(Sigma_VH_lin, Sigma_VV_lin)
    Gamma_ratio     = _ratio_db(Gamma_VH_lin, Gamma_VV_lin)
    Beta_ratio      = _ratio_db(Beta_VH_lin,  Beta_VV_lin)
    Gamma_RTC_ratio = _ratio_db(Gamma_VH2_lin, Gamma_VV2_lin)

    # Append in the exact FEATURE_NAMES order
    arrs.extend([
        Sigma_VH, Sigma_VV,
        Gamma_VH, Gamma_VV,
        Beta_VH,  Beta_VV,
        Gamma_VH_RTC, Gamma_VV_RTC,
        Sigma_sum, Gamma_sum, Beta_sum, Gamma_RTC_sum,
        Sigma_diff, Gamma_diff, Beta_diff, Gamma_RTC_diff,
        Sigma_ratio, Gamma_ratio, Beta_ratio, Gamma_RTC_ratio,
        LIA, IAFE,
    ])

    # Read DEM, Aspect, Slope, SD
    for suf in ("DEM", "Slope", "Aspect", "SD"):
        tif_path = glob(os.path.join(aoi_dir, f"*{suf}.tif"))[0]
        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype(np.float32)
        if suf == "Aspect":
            rad = np.deg2rad(arr)
            arrs.append(np.sin(rad))
            arrs.append(np.cos(rad))
        else:
            arrs.append(arr)
    
    # Crop everything to the smallest H,W and stack
    H_min = min(a.shape[0] for a in arrs)
    W_min = min(a.shape[1] for a in arrs)
    arrs = [a[:H_min, :W_min] for a in arrs]

    # stack: FEATURE_NAMES (26) + SD (label) as last channel
    return np.stack(arrs, axis=-1)


def build_df(data_dir, drop_invalid=True, upper_threshold=3, selected_features=None):
    """
    Build and return a pandas DataFrame containing all pixels from all AOIs.

    Parameters
    data_dir : str
        Path to directory containing AOI subfolders with TIF files.

    drop_invalid : bool, optional (default: True)
        If True, drop rows where SD is NaN or SD < 0

    upper_threshold : float or None, optional (default: 3)
        If provided, rows with SD greater than this value are removed.
    
    selected_features : list[str] or None
        If provided, only these feature channels (by name) are written to
        each group's 'features' dataset. Names should be among FEATURE_NAMES.

    Returns:
    df : pandas.DataFrame
        DataFrame containing all pixels with columns in this order:
        ['aoi_name','row','col'] + FEATURE_NAMES + ['SD']
    """
    aoi_dirs = list_aoi_dirs(data_dir)
    dfs = []
    for aoi_path in aoi_dirs:
        name = os.path.basename(aoi_path)
        stack = load_stack(aoi_path)  # (H, W, 27) -> 26 features + SD
        H, W, _ = stack.shape
        rows, cols = np.indices((H, W))

        # map stack to columns following FEATURE_NAMES order
        data = {
            'aoi_name': [name] * (H * W),
            'row': rows.ravel(),
            'col': cols.ravel(),
        }
        for i, fname in enumerate(FEATURE_NAMES):
            data[fname] = stack[:, :, i].ravel()
        data['SD'] = stack[:, :, len(FEATURE_NAMES)].ravel()

        df = pd.DataFrame(data)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    if drop_invalid:
        valid = df['SD'].notna() & (df['SD'] >= 0)
        if upper_threshold is not None:
            valid &= (df['SD'] <= upper_threshold)
        df = df.loc[valid].reset_index(drop=True)

    if selected_features is not None:
        keep = ["aoi_name", "row", "col", "SD"] + list(selected_features)
        missing = [c for c in selected_features if c not in df.columns]
        if missing:
            raise KeyError(f"Selected features not found in DataFrame: {missing}")
        df = df[keep].copy()

    return df



def build_h5(data_dir, out_dir, write_mask=True, upper_threshold=3, selected_features=None):
    """
    Build and write a single HDF5 at out_dir/data.h5 with one group per AOI:

    For each AOI group the following datasets are written:
      - '<aoi_name>/features': dataset of shape (H, W, 7) 
      - '<aoi_name>/label':    dataset of shape (H, W, 1) 
      - '<aoi_name>/mask':     dataset of shape (H, W)

    Parameters
    ----------
    data_dir : str
        Directory of AOI subfolders containing TIF stacks.

    out_dir : str
        Directory where data.h5 will be written.

    write_mask : bool, optional (default: True)
        When True, writes the '<aoi_name>/mask', [1=valid, 0=invalid]
        Valid if SD is not NaN and SD >= 0 (and <= upper_threshold if set).

    upper_threshold : float or None, optional (default: 3)
    If provided, mask marks pixels with SD > upper_threshold as invalid (0).

    selected_features : list[str] or None, optional (default: None)
        If provided, only these feature channels (by name) are written to
        each group's 'features' dataset. Names must be among FEATURE_NAMES.
        The order you pass here is preserved in the written tensor.

    """
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # List all AOI directories
    aoi_dirs = list_aoi_dirs(data_dir)
    if not aoi_dirs:
        raise FileNotFoundError(f"No AOI directories found in {data_dir}")

    # Path to the single HDF5 file
    file_path = os.path.join(out_dir, 'dataframe.h5')

    # Write all AOIs into one HDF5
    with h5py.File(file_path, 'w') as hf:
        for aoi_path in aoi_dirs:
            name = os.path.basename(aoi_path)
            stack = load_stack(aoi_path).astype('float32')  # (H, W, 8)

            feats   = stack[..., :-1]        # (H, W, 7)
            label2d = stack[..., -1]         # (H, W)

            # Optionally subset features by NAME
            if selected_features is not None:
                # verify names and map to indices in FEATURE_NAMES
                missing = [n for n in selected_features if n not in FEATURE_NAMES]
                if missing:
                    raise KeyError(f"Unknown feature(s) in selected_features: {missing}")
                idxs = [FEATURE_NAMES.index(n) for n in selected_features]
                feats = feats[..., idxs]     # (H, W, C_selected)
                feat_names_to_write = list(selected_features) 
            else:
                feat_names_to_write = FEATURE_NAMES

            # Mask invalid SD (NaN or < 0) and optional upper threshold
            valid = (~np.isnan(label2d)) & (label2d >= 0.0)
            if upper_threshold is not None:
                valid &= (label2d <= float(upper_threshold))

            label = label2d[..., np.newaxis] # (H, W, 1)

            grp = hf.create_group(name)
            grp.create_dataset('features', data=feats,  compression='gzip')
            grp.create_dataset('label',    data=label, compression='gzip')
            if write_mask:
                grp.create_dataset('mask', data=valid.astype('uint8'), compression='gzip')

    print(f"Wrote single HDF5 with {len(aoi_dirs)} AOI(s) at {file_path}")


