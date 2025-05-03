import os
import h5py
import numpy as np
import rasterio
from glob import glob

def create_h5(data_dir, out_dir, test_aoi):
    """
    Create two HDF5 files under out_dir:
      - train.h5: Data from AOIs except test_aoi
      - test.h5 : Data from test_aoi

    Each HDF5 has:
      /data   : shape (N, 7) float32, feature channels: [VH_dB, VV_dB, CrossPolRatio_dB, DEM, sin(Aspect), cos(Aspect), Slope]
      /labels : shape (N,) float32, target snow depth (SD)
      /aoi    : shape (N,) string, AOI identifier for each sample
    """
    os.makedirs(out_dir, exist_ok=True)

    # Locate AOI directories
    train_dirs, test_dir = find_aoi_dirs(data_dir, test_aoi)

    # Sample and flatten train data
    X_parts, y_parts, aoi_parts = [], [], []
    for aoi in train_dirs:
        aoi_name = os.path.basename(aoi)
        stack = load_stack(aoi)
        feats = stack[..., :-1]      # Extract feature channels
        sd = stack[..., -1]          # Extract snow-depth channel

        # Flatten dimensions
        flat_feats = feats.reshape(-1, feats.shape[-1]).astype(np.float32)
        flat_sd    = sd.reshape(-1).astype(np.float32)
        aoi_ids    = np.full(flat_sd.shape, aoi_name, dtype=h5py.string_dtype())

        X_parts.append(flat_feats)
        y_parts.append(flat_sd)
        aoi_parts.append(aoi_ids)

    # Concatenate training arrays
    X_train = np.vstack(X_parts)
    y_train = np.concatenate(y_parts)
    aoi_train_ids = np.concatenate(aoi_parts)

    # Sample and flatten test data
    stack_t = load_stack(test_dir)
    feats_t = stack_t[..., :-1]
    sd_t    = stack_t[..., -1]

    # Flatten dimensions
    X_test = feats_t.reshape(-1, feats_t.shape[-1]).astype(np.float32)
    y_test    = sd_t.reshape(-1).astype(np.float32)
    aoi_test_id = np.full(y_test.shape, os.path.basename(test_dir), dtype=h5py.string_dtype())

    # Write train.h5
    train_path = os.path.join(out_dir, "train.h5")
    with h5py.File(train_path, "w") as f:
        f.create_dataset("data",   data=X_train, compression="gzip")
        f.create_dataset("labels", data=y_train)
        f.create_dataset("aoi",    data=aoi_train_ids)

    # Write test.h5
    test_path = os.path.join(out_dir, "test.h5")
    with h5py.File(test_path, "w") as f:
        f.create_dataset("data",   data=X_test, compression="gzip")
        f.create_dataset("labels", data=y_test)
        f.create_dataset("aoi",    data=aoi_test_id)

    print(f"Wrote train set: {train_path} → data {X_train.shape}, labels {y_train.shape}, aoi {aoi_train_ids.shape}")
    print(f"Wrote test set : {test_path}  → data {X_test.shape}, labels {y_test.shape}, aoi {aoi_test_id.shape}")



def create_dl_h5(data_dir, out_dir, test_aoi,
                 patch_size=64, patches_per_aoi=1000, random_state=42):
    """
    Create two HDF5 files under out_dir for DL patch training:
      - dl_train.h5: patches from train AOIs
      - dl_test.h5 : patches from test AOI

    Each HDF5 has:
      /patches : shape (N, patch_size, patch_size, 7) float32, feature channels
      /labels  : shape (N, patch_size, patch_size) float32, target SD values
      /aoi     : shape (N,) string, AOI identifier for each patch
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(random_state)
    train_dirs, test_dir = find_aoi_dirs(data_dir, test_aoi)

    # Sample patches from a single AOI
    def sample_patches(stack, aoi_name):
        H, W, C = stack.shape
        patches, labels, aois = [], [], []
        for _ in range(patches_per_aoi):
            i = rng.randint(0, H - patch_size + 1)
            j = rng.randint(0, W - patch_size + 1)
            patch = stack[i:i+patch_size, j:j+patch_size, :].astype(np.float32)
            patches.append(patch[..., :-1])  # feature channels
            labels.append(patch[..., -1])    # SD layer
            aois.append(aoi_name)
        return (np.stack(patches),
                np.stack(labels),
                np.array(aois, dtype=h5py.string_dtype()))

    # Sample train patches
    X_tr, y_tr, aoi_tr = [], [], []
    for aoi in train_dirs:
        name = os.path.basename(aoi)
        stack = load_stack(aoi)
        Xp, yp, ap = sample_patches(stack, name)
        X_tr.append(Xp)
        y_tr.append(yp)
        aoi_tr.append(ap)
    X_train = np.concatenate(X_tr, axis=0)
    y_train = np.concatenate(y_tr, axis=0)
    aoi_train_ids = np.concatenate(aoi_tr, axis=0)

    # Sample test patches
    name_t = os.path.basename(test_dir)
    stack_t = load_stack(test_dir)
    X_test, y_test, aoi_test = sample_patches(stack_t, name_t)

    # Write dl_train.h5
    dl_train_path = os.path.join(out_dir, "dl_train.h5")
    with h5py.File(dl_train_path, "w") as f:
        f.create_dataset("patches", data=X_train, compression="gzip")
        f.create_dataset("labels",  data=y_train)
        f.create_dataset("aoi",     data=aoi_train_ids)
    # Write dl_test.h5
    dl_test_path = os.path.join(out_dir, "dl_test.h5")
    with h5py.File(dl_test_path, "w") as f:
        f.create_dataset("patches", data=X_test, compression="gzip")
        f.create_dataset("labels",  data=y_test)
        f.create_dataset("aoi",     data=aoi_test)

    print(f"Wrote DL train: {dl_train_path} → patches {X_train.shape}, labels {y_train.shape}, aoi {aoi_train_ids.shape}")
    print(f"Wrote DL test : {dl_test_path}  → patches {X_test.shape}, labels {y_test.shape}")



def find_aoi_dirs(data_dir, test_aoi):
    """
    Return two lists of AOI directories:
      - train_dirs: all AOIs except test_aoi
      - test_dir:   the single test AOI directory
    """
    all_dirs = sorted(
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    train_dirs = [d for d in all_dirs if os.path.basename(d) != test_aoi]
    test_dirs  = [d for d in all_dirs if os.path.basename(d) == test_aoi]
    if not test_dirs:
        raise FileNotFoundError(f"Test AOI '{test_aoi}' not found under {data_dir}")
    return train_dirs, test_dirs[0]



def load_stack(aoi_dir):
    """
    Read and stack raster bands for one AOI:
      - SAR bands 3,4,5 → VH_dB, VV_dB, CrossPolRatio_dB
      - DEM
      - Aspect → sin(Aspect), cos(Aspect)
      - Slope
      - SD

    All arrays are cropped to the smallest common H×W.

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
    for suf in ("DEM", "Aspect", "Slope", "SD"):
        tif_path = glob(os.path.join(aoi_dir, f"*{suf}.tif"))[0]
        with rasterio.open(tif_path) as src:
            arr = src.read(1)
        if suf == "Aspect":
            rad = np.deg2rad(arr)
            arrs.append(np.sin(rad))
            arrs.append(np.cos(rad))
        else:
            arrs.append(arr)

    # Crop to smallest common shape
    Hmin = min(a.shape[0] for a in arrs)
    Wmin = min(a.shape[1] for a in arrs)
    arrs = [a[:Hmin, :Wmin] for a in arrs]

    # Stack arrays to (H, W, 8)
    stack = np.stack(arrs, axis=-1)
    return stack
