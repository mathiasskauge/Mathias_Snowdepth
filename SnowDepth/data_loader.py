import os
import numpy as np
import pandas as pd
import rasterio
import h5py
from glob import glob


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
 

def build_dataframe(data_dir):
    """
    Build and return a pandas DataFrame containing all pixels from all AOIs.

    Parameters
    ----------
    data_dir : str
        Path to directory containing AOI subfolders with TIF files.

    Returns
    -------
    full_df : pandas.DataFrame
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
            'aoi_name':        [name] * (H * W),
            'row':             rows.ravel(),
            'col':             cols.ravel(),
            'VH_dB':           stack[:,:,0].ravel(),
            'VV_dB':           stack[:,:,1].ravel(),
            'CrossPolRatio_dB':stack[:,:,2].ravel(),
            'Elevation':       stack[:,:,3].ravel(),
            'Slope':           stack[:,:,6].ravel(),
            'sin_Aspect':      stack[:,:,4].ravel(),
            'cos_Aspect':      stack[:,:,5].ravel(),
            'SD':              stack[:,:,7].ravel(),
        })
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df


def create_h5s(data_dir, out_dir):
    """
    Create a HDF5 file containing data from all AOIs.

    The file is created at out_dir/data.h5 and contains one group per AOI:
      - '<aoi_name>/features': dataset of shape (H, W, C) float32
      - '<aoi_name>/label':    dataset of shape (H, W, 1) float32

    Parameters
    ----------
    data_dir : str
        Directory of AOI subfolders containing TIF stacks.
    out_dir : str
        Directory where data.h5 will be written.
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
            stack = load_stack(aoi_path).astype('float32')  # (H, W, C+1)
            feats = stack[..., :-1]                          # (H, W, C)
            label = stack[..., -1][..., np.newaxis]         # (H, W, 1)

            grp = hf.create_group(name)
            grp.create_dataset('features', data=feats, compression='gzip')
            grp.create_dataset('label', data=label, compression='gzip')

    print(f"Wrote single HDF5 with {len(aoi_dirs)} AOI(s) at {file_path}")