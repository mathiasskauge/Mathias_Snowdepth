import os
import numpy as np
import pandas as pd
import rasterio
import h5py
from glob import glob

"""
Methods to load data for classical and deep learning models.

build_df -> returns a pandas dataFrame for tabular data use

create_H5 -> returns an H5 file (features/label/mask) for DL models

"""


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
    Read and stack raster bands for one AOI directory:
      - SAR bands 3,4,5 → VH_dB, VV_dB, CrossPolRatio_dB
      - DEM -> Elevation
      - Slope
      - Aspect → sin(Aspect), cos(Aspect)
      - SD

    Returns
    -------
    stack : np.ndarray of shape (H, W, 8)
    """
    arrs = []

    # Read SAR bands
    sar_path = glob(os.path.join(aoi_dir, "*SAR.tif"))[0]
    with rasterio.open(sar_path) as src:
        for arr in src.read([3, 4, 5]):
            arrs.append(arr)

    # Read DEM, Aspect, Slope, SD
    for suf in ("DEM", "Slope", "Aspect", "SD"):
        tif_path = glob(os.path.join(aoi_dir, f"*{suf}.tif"))[0]
        with rasterio.open(tif_path) as src:
            arr = src.read(1)
        if suf == "Aspect":
            rad = np.deg2rad(arr)
            arrs.append(np.sin(rad))
            arrs.append(np.cos(rad))
        else:
            arrs.append(arr)

    return np.stack(arrs, axis=-1)
 

def build_df(data_dir, drop_invalid=True, upper_threshold=3):
    """
    Build and return a pandas DataFrame containing all pixels from all AOIs.

    Parameters
    ----------
    data_dir : str
        Path to directory containing AOI subfolders with TIF files.

    drop_invalid : bool, optional (default: True)
        If True, drop rows where SD is NaN or SD < 0

    upper_threshold : float or None, optional (default: 3)
        If provided, rows with SD greater than this value are removed.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing all pixels with columns in this order:
        ['aoi_name', 'row', 'col', 'VH_dB', 'VV_dB', 'CrossPolRatio_dB','Elevation', 'Slope', 'sin_Aspect', 'cos_Aspect', 'SD']
    """
    aoi_dirs = list_aoi_dirs(data_dir)
    dfs = []
    for aoi_path in aoi_dirs:
        name = os.path.basename(aoi_path)
        stack = load_stack(aoi_path)
        H, W, _ = stack.shape
        rows, cols = np.indices((H, W))
        df = pd.DataFrame({
            'aoi_name':          [name] * (H * W),
            'row':               rows.ravel(),
            'col':               cols.ravel(),
            'VH_dB':             stack[:, :, 0].ravel(),
            'VV_dB':             stack[:, :, 1].ravel(),
            'CrossPolRatio_dB':  stack[:, :, 2].ravel(),
            'Elevation':         stack[:, :, 3].ravel(),
            'Slope':             stack[:, :, 4].ravel(),
            'sin_Aspect':        stack[:, :, 5].ravel(),
            'cos_Aspect':        stack[:, :, 6].ravel(),
            'SD':                stack[:, :, 7].ravel(),
        })
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    if drop_invalid:
        valid = df['SD'].notna() & (df['SD'] >= 0)
        if upper_threshold is not None:
            valid &= (df['SD'] <= upper_threshold)
        df = df.loc[valid].reset_index(drop=True)

    return df



def create_h5(data_dir, out_dir, write_mask=True, upper_threshold=3):
    """
    Create a single HDF5 at out_dir/data.h5 with one group per AOI:

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

    """
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # List all AOI directories
    aoi_dirs = list_aoi_dirs(data_dir)
    if not aoi_dirs:
        raise FileNotFoundError(f"No AOI directories found in {data_dir}")

    # Path to the single HDF5 file
    file_path = os.path.join(out_dir, 'data.h5')

    # Write all AOIs into one HDF5
    with h5py.File(file_path, 'w') as hf:
        for aoi_path in aoi_dirs:
            name = os.path.basename(aoi_path)
            stack = load_stack(aoi_path).astype('float32')  # (H, W, 8)

            feats   = stack[..., :-1]        # (H, W, 7)
            label2d = stack[..., -1]         # (H, W)

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


