import os
import numpy as np
import pandas as pd
import rasterio
import h5py
from glob import glob
from SnowDepth.config import FEATURE_NAMES


"""

Methods to load data and calculate features for classic ML and deep learning models.
build_df -> returns a pandas dataFrame for tabular data use
build_h5 -> writes an h5 file with one group per AOI


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
    Read and stack raster bands for one AOI directory.

    Returns:
    stack : np.ndarray of shape (H, W, 28)
            Channels in this exact order:
            FEATURE_NAMES (27) + SD label as the last channel
    """

    arrs = []
    
    # Read SAR bands (by fixed index order)
    if not glob(os.path.join(aoi_dir, "*SAR.tif")):
        raise FileNotFoundError(f"No SAR file found in {aoi_dir}")
    sar_path = glob(os.path.join(aoi_dir, "*SAR.tif"))[0]

    with rasterio.open(sar_path) as src:
        # linear inputs
        Sigma_VH_lin = src.read(1).astype(np.float32)   # Sigma0_VH
        Gamma_VH_lin = src.read(2).astype(np.float32)   # Gamma0_VH
        Beta_VH_lin  = src.read(3).astype(np.float32)   # Beta0_VH
        Sigma_VV_lin = src.read(4).astype(np.float32)   # Sigma0_VV
        Gamma_VV_lin = src.read(5).astype(np.float32)   # Gamma0_VV
        Beta_VV_lin  = src.read(6).astype(np.float32)   # Beta0_VV
        Gamma_VH_RTC_lin = src.read(7).astype(np.float32)  # Gamma0_VH (RTC)
        Gamma_VV_RTC_lin = src.read(8).astype(np.float32)  # Gamma0_VV (RTC)
        LIA  = src.read(9).astype(np.float32)           # localIncidenceAngle
        IAFE = src.read(10).astype(np.float32)          # incidenceAngleFromEllipsoid

    # Base backscatter in dB
    Sigma_VH = _to_db(Sigma_VH_lin)
    Sigma_VV = _to_db(Sigma_VV_lin)
    Gamma_VH = _to_db(Gamma_VH_lin)
    Gamma_VV = _to_db(Gamma_VV_lin)
    Beta_VH  = _to_db(Beta_VH_lin)
    Beta_VV  = _to_db(Beta_VV_lin)
    Gamma_VH_RTC = _to_db(Gamma_VH_RTC_lin)
    Gamma_VV_RTC = _to_db(Gamma_VV_RTC_lin)

    # Linear sums/differences (linear)
    Sigma_sum      = Sigma_VH_lin + Sigma_VV_lin
    Gamma_sum      = Gamma_VH_lin + Gamma_VV_lin
    Beta_sum       = Beta_VH_lin  + Beta_VV_lin
    Gamma_RTC_sum  = Gamma_VH_RTC_lin + Gamma_VV_RTC_lin
    Sigma_diff     = Sigma_VH_lin - Sigma_VV_lin
    Gamma_diff     = Gamma_VH_lin - Gamma_VV_lin
    Beta_diff      = Beta_VH_lin  - Beta_VV_lin
    Gamma_RTC_diff = Gamma_VH_RTC_lin - Gamma_VV_RTC_lin

    # Ratios in dB
    Sigma_ratio     = _ratio_db(Sigma_VH_lin, Sigma_VV_lin)
    Gamma_ratio     = _ratio_db(Gamma_VH_lin, Gamma_VV_lin)
    Beta_ratio      = _ratio_db(Beta_VH_lin,  Beta_VV_lin)
    Gamma_RTC_ratio = _ratio_db(Gamma_VH_RTC_lin, Gamma_VV_RTC_lin)

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

    # Read DEM, Aspect, Slope, VH, SD
    shape_info = []  # store (filename, original shape)
    for suf in ("DEM", "Slope", "Aspect", "VH", "SD"):
        tif_path = glob(os.path.join(aoi_dir, f"*{suf}.tif"))[0]
        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype(np.float32)
            shape_info.append((os.path.basename(tif_path), src.shape))
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

    # Had an issue with Veg_height so drop pixel if any of features is nan
    stack = np.stack(arrs, axis=-1)
    nan_any = np.any(np.isnan(stack), axis=-1)
    if nan_any.any():
        stack[nan_any] = np.nan

    # stack: FEATURE_NAMES (27) + SD (label) as last channel
    return stack


def build_df(
    data_dir, 
    drop_invalid=True, 
    upper_threshold=None,
    selected_features=None
    ):
    """
    Build and return a pandas DataFrame containing all pixels from all AOIs.

    Returns:
    df : pandas.DataFrame
        DataFrame containing all pixels with columns in this order:
        ['aoi_name','row','col'] + FEATURE_NAMES + ['SD']
    """
    aoi_dirs = list_aoi_dirs(data_dir)
    dfs = []

    for aoi_path in aoi_dirs:
        name = os.path.basename(aoi_path)
        stack = load_stack(aoi_path)  # (H, W, 27 features + SD)
        H, W, _ = stack.shape
        rows, cols = np.indices((H, W))

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

    # Keeps NaNs as NaN, sets negatives to 0, caps to upper_threshold if provided
    df['SD'] = df['SD'].clip(lower=0, upper=upper_threshold)

    if drop_invalid:
        # Only drop rows where SD is NaN; keep zeros and capped values
        df = df.loc[df['SD'].notna()].reset_index(drop=True)

    if selected_features is not None:
        keep = ["aoi_name", "row", "col", "SD"] + list(selected_features)
        missing = [c for c in selected_features if c not in df.columns]
        if missing:
            raise KeyError(f"Selected features not found in DataFrame: {missing}")
        df = df[keep].copy()

    return df



def build_h5(
    data_dir,
    out_dir,
    write_mask=True,
    upper_threshold=None,
    selected_features=None,
    compression="gzip",
    chunks=True,
    dtype="float32",
    out_name="data.h5",
):
    """
    Build a single HDF5 at out_dir/out_name with one group per AOI.

    For each AOI group:
      - '<aoi>/features': (H, W, C)
      - '<aoi>/label':    (H, W, 1)  SD (clamped: negatives->0, optionally capped)
      - '<aoi>/mask':     (H, W)     [1=valid (SD not NaN), 0=invalid]  (optional)
    """
    os.makedirs(out_dir, exist_ok=True)
    aoi_dirs = list_aoi_dirs(data_dir)

    if not aoi_dirs:
        raise FileNotFoundError(f"No AOI directories found in {data_dir}")

    # Map selected feature names -> indices in FEATURE_NAMES
    if selected_features is not None:
        feat_idxs = [FEATURE_NAMES.index(n) for n in selected_features]
        feat_names_to_write = list(selected_features)
    else:
        feat_idxs = list(range(len(FEATURE_NAMES)))
        feat_names_to_write = list(FEATURE_NAMES)

    file_path = os.path.join(out_dir, out_name)
    with h5py.File(file_path, "w") as hf:
        # Store global metadata
        hf.attrs["feature_names"] = np.array(feat_names_to_write, dtype="S")
        hf.attrs["upper_threshold"] = -1 if upper_threshold is None else float(upper_threshold)
        hf.attrs["write_mask"] = bool(write_mask)
        hf.attrs["sd_clamping"] = np.string_("applied: lower=0, upper=upper_threshold")

        for aoi_path in aoi_dirs:
            name = os.path.basename(aoi_path)
            stack = load_stack(aoi_path).astype(dtype)  # (H, W, len(FEATURE_NAMES)+1)
            feats_all = stack[..., :-1]
            label2d   = stack[..., -1]

            # --- NEW: clamp SD in-place; preserve NaNs ---
            # negatives -> 0; optionally cap to upper_threshold
            label2d = np.clip(label2d, a_min=0, a_max=upper_threshold) if upper_threshold is not None \
                      else np.where(np.isnan(label2d), np.nan, np.maximum(label2d, 0))

            # Subset features
            feats = feats_all[..., feat_idxs]
            label = label2d[..., np.newaxis]

            # Mask = SD not NaN (no threshold-based dropping)
            valid = (~np.isnan(label2d))

            grp = hf.create_group(name)
            grp.create_dataset("features", data=feats, compression=compression, chunks=chunks)
            grp.create_dataset("label", data=label.astype(dtype), compression=compression, chunks=chunks)

            if write_mask:
                grp.create_dataset("mask", data=valid.astype("uint8"), compression=compression, chunks=chunks)

            # Per-AOI metadata
            grp.attrs["aoi_name"] = name
            grp.attrs["shape"] = feats.shape
            grp.attrs["feature_names"] = np.array(feat_names_to_write, dtype="S")

    print(f"Wrote HDF5 with {len(aoi_dirs)} AOI(s) at {file_path}")
    print(f"features: {feat_names_to_write}")