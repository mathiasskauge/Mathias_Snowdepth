import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

"""
Implements splitting strategies for different models

"""

def ML_split(dev_df, seed, pxs_per_aoi=10000):

    dev_df = dev_df.copy()
    
    # Assign SD quartiles within each AOI
    dev_df['sd_quartile'] = (
        dev_df
        .groupby('aoi_name')['SD']
        .transform(lambda x: pd.qcut(x, 4, labels=False, duplicates='drop'))
    )

    sss = StratifiedShuffleSplit(n_splits=1, train_size=pxs_per_aoi, random_state=seed)

    samples = []

    for aoi, group in dev_df.groupby('aoi_name'):
        # Stratified train sampling
        sample_idx, _ = next(sss.split(group, group['sd_quartile']))
        df_samples = group.iloc[sample_idx].copy()
        samples.append(df_samples)

    df_sampled = pd.concat(samples, ignore_index=True)

    # Determine feature columns 
    feature_cols = [c for c in df_sampled.columns if c not in ('aoi_name', 'row', 'col', 'sd_quartile', 'SD')]

    # Build development arrays
    X_dev = df_sampled[feature_cols].values
    y_dev = df_sampled['SD'].values
    groups = df_sampled['aoi_name'].values

    print(f"Total samples: {len(df_sampled)} across {df_sampled['aoi_name'].nunique()} AOIs")
    print(f"Features used: {feature_cols}")
    print(f"X_dev shape: {X_dev.shape}")

    return X_dev, y_dev, groups


def DL_split(
    h5_path,
    holdout_aoi,
    val_fraction=0.10,
    seed=18,
    patch_size=128,
    stride=None,
    min_valid_frac=0.0
):
    """
    - Split HDF5 dataset into train/val from dev AOIs (all except holdout_aoi),
    - Tile and return the holdout AOI for inference/evaluation.

    Returns
    (X_train, y_train, m_train), (X_val, y_val, m_val) (X_hold, y_hold, m_hold)
        X: (N, patch_size, patch_size, C)
        y: (N, patch_size, patch_size, 1)
        m: (N, patch_size, patch_size, 1) -> mask (1=valid, 0=invalid)
    """
    rng = np.random.RandomState(seed)
    if stride is None:
        stride = patch_size

    def _tile_one(feats, label, mask, ps, st, min_frac):
        H, W, C = feats.shape
        xs, ys, ms = [], [], []
        for r in range(0, H - ps + 1, st):
            for c in range(0, W - ps + 1, st):
                m = mask[r:r+ps, c:c+ps]
                if m.size == 0:
                    continue
                if m.mean() < min_frac:
                    continue
                xs.append(feats[r:r+ps, c:c+ps, :])
                ys.append(label[r:r+ps, c:c+ps, :])
                ms.append(m[..., None]) 
        if xs:
            return np.stack(xs), np.stack(ys), np.stack(ms)
        # empty fallback with correct dims
        return (
            np.empty((0, ps, ps, feats.shape[-1]), feats.dtype),
            np.empty((0, ps, ps, 1), label.dtype),
            np.empty((0, ps, ps, 1), np.uint8),
        )

    def _mask_from_label(label_3d):
        sd2d = label_3d[..., 0]
        return ((~np.isnan(sd2d)) & (sd2d >= 0)).astype(np.uint8)

    with h5py.File(h5_path, 'r') as hf:
        aoi_names = list(hf.keys())
        if holdout_aoi not in aoi_names:
            raise KeyError(f"Holdout AOI '{holdout_aoi}' not found in {h5_path}")

        dev_names  = [n for n in aoi_names if n != holdout_aoi]

        # Load & tile all DEV AOIs, then split patches into train/val 
        X_dev_list, y_dev_list, m_dev_list = [], [], []
        for name in dev_names:
            grp = hf[name]
            feats  = grp['features'][...]                  
            label  = grp['label'][...]                     
            if 'mask' in grp:
                mask = grp['mask'][...].astype(np.uint8)   
            else:
                mask = _mask_from_label(label)
            x, y, m = _tile_one(feats, label, mask, patch_size, stride, min_valid_frac)
            if x.shape[0] > 0:
                X_dev_list.append(x); y_dev_list.append(y); m_dev_list.append(m)

        if X_dev_list:
            X_dev = np.concatenate(X_dev_list, axis=0)
            y_dev = np.concatenate(y_dev_list, axis=0)
            m_dev = np.concatenate(m_dev_list, axis=0)
        else:
            any_name = dev_names[0]
            C = hf[any_name]['features'].shape[-1]
            X_dev = np.empty((0, patch_size, patch_size, C), dtype=np.float32)
            y_dev = np.empty((0, patch_size, patch_size, 1), dtype=np.float32)
            m_dev = np.empty((0, patch_size, patch_size, 1), dtype=np.uint8)

        # Random patch-level split across DEV patches
        n_dev = X_dev.shape[0]
        idx = np.arange(n_dev)
        rng.shuffle(idx)
        n_val = int(np.round(n_dev * val_fraction))
        val_idx = idx[:n_val]
        trn_idx = idx[n_val:]

        X_train, y_train, m_train = X_dev[trn_idx], y_dev[trn_idx], m_dev[trn_idx]
        X_val,   y_val,   m_val   = X_dev[val_idx], y_dev[val_idx], m_dev[val_idx]

        # Tile & return HOLDOUT AOI
        grpH = hf[holdout_aoi]
        featsH = grpH['features'][...]
        labelH = grpH['label'][...]
        if 'mask' in grpH:
            maskH = grpH['mask'][...].astype(np.uint8)
        else:
            maskH = _mask_from_label(labelH)

        X_hold, y_hold, m_hold = _tile_one(featsH, labelH, maskH, patch_size, stride, min_valid_frac)

    return (X_train, y_train, m_train), (X_val, y_val, m_val), (X_hold, y_hold, m_hold)


def DL_split(
    h5_path,
    holdout_aoi,
    val_fraction=0.10,
    seed=18,
    patch_size=128,
    stride=None,
    min_valid_frac=0.0
):
    """
    - Split HDF5 dataset into train/val from dev AOIs (all except holdout_aoi)
    - Tile and return the holdout AOI for inference/evaluation.

    Returns
    (X_train, y_train, m_train), (X_val, y_val, m_val) (X_hold, y_hold, m_hold)
        X: (N, patch_size, patch_size, C)
        y: (N, patch_size, patch_size, 1)
        m: (N, patch_size, patch_size, 1) -> mask (1=valid, 0=invalid)
    """
    rng = np.random.RandomState(seed)
    if stride is None:
        stride = patch_size

    def _tile_one(feats, label, mask, ps, st, min_frac):
        H, W, C = feats.shape
        xs, ys, ms = [], [], []
        for r in range(0, H - ps + 1, st):
            for c in range(0, W - ps + 1, st):
                m = mask[r:r+ps, c:c+ps]
                if m.size == 0:
                    continue
                if m.mean() < min_frac:
                    continue
                xs.append(feats[r:r+ps, c:c+ps, :])
                ys.append(label[r:r+ps, c:c+ps, :])
                ms.append(m[..., None])   # (ps, ps, 1)
        if xs:
            return np.stack(xs), np.stack(ys), np.stack(ms)
        # empty fallback with correct dims
        return (
            np.empty((0, ps, ps, feats.shape[-1]), feats.dtype),
            np.empty((0, ps, ps, 1), label.dtype),
            np.empty((0, ps, ps, 1), np.uint8),
        )

    def _mask_from_label(label_3d):
        sd2d = label_3d[..., 0]
        return ((~np.isnan(sd2d)) & (sd2d >= 0)).astype(np.uint8)

    with h5py.File(h5_path, 'r') as hf:
        aoi_names = list(hf.keys())
        if holdout_aoi not in aoi_names:
            raise KeyError(f"Holdout AOI '{holdout_aoi}' not found in {h5_path}")

        dev_names  = [n for n in aoi_names if n != holdout_aoi]

        # --- Load & tile all DEV AOIs, then split patches into train/val ---
        X_dev_list, y_dev_list, m_dev_list = [], [], []
        for name in dev_names:
            grp = hf[name]
            feats  = grp['features'][...]                    # (H, W, C)
            label  = grp['label'][...]                      # (H, W, 1)
            if 'mask' in grp:
                mask = grp['mask'][...].astype(np.uint8)    # (H, W)
            else:
                mask = _mask_from_label(label)
            x, y, m = _tile_one(feats, label, mask, patch_size, stride, min_valid_frac)
            if x.shape[0] > 0:
                X_dev_list.append(x); y_dev_list.append(y); m_dev_list.append(m)

        if X_dev_list:
            X_dev = np.concatenate(X_dev_list, axis=0)
            y_dev = np.concatenate(y_dev_list, axis=0)
            m_dev = np.concatenate(m_dev_list, axis=0)
        else:
            # no patches met criteria
            # create empty placeholders with C inferred from one sample read above
            # (re-open one group to infer C safely if needed)
            any_name = dev_names[0]
            C = hf[any_name]['features'].shape[-1]
            X_dev = np.empty((0, patch_size, patch_size, C), dtype=np.float32)
            y_dev = np.empty((0, patch_size, patch_size, 1), dtype=np.float32)
            m_dev = np.empty((0, patch_size, patch_size, 1), dtype=np.uint8)

        # Random patch-level split across DEV patches
        n_dev = X_dev.shape[0]
        idx = np.arange(n_dev)
        rng.shuffle(idx)
        n_val = int(np.round(n_dev * val_fraction))
        val_idx = idx[:n_val]
        trn_idx = idx[n_val:]

        X_train, y_train, m_train = X_dev[trn_idx], y_dev[trn_idx], m_dev[trn_idx]
        X_val,   y_val,   m_val   = X_dev[val_idx], y_dev[val_idx], m_dev[val_idx]

        # --- Tile & return HOLDOUT AOI as well ---
        grpH = hf[holdout_aoi]
        featsH = grpH['features'][...]
        labelH = grpH['label'][...]
        if 'mask' in grpH:
            maskH = grpH['mask'][...].astype(np.uint8)
        else:
            maskH = _mask_from_label(labelH)

        X_hold, y_hold, m_hold = _tile_one(featsH, labelH, maskH, patch_size, stride, min_valid_frac)

    return (X_train, y_train, m_train), (X_val, y_val, m_val), (X_hold, y_hold, m_hold)
