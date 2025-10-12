import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

"""
Implements splitting strategies for different models

"""

def ML_split(dev_df, hold_df, seed, pxs_per_aoi=1000):

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

    # Build hold‐out arrays using the same feature columns 
    X_hold = hold_df[feature_cols].values
    y_hold = hold_df['SD'].values
    print(f"X_hold shape: {X_hold.shape}")

    return X_dev, y_dev, groups, X_hold, y_hold


def unet_split(h5_path, holdout_aoi, val_fraction=0.3, seed=18, patch_size=256, stride=None, min_valid_frac=0.0):
    """
    Split HDF5 dataset into train/val/test for UNet training, with tiling.

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_hold, y_hold)
        Each X: (N, patch_size, patch_size, C)
              y: (N, patch_size, patch_size, 1)
    """
    rng = np.random.RandomState(seed)
    if stride is None:
        stride = patch_size

    def _tile_one(feats, label, mask, ps, st, min_frac):
        H, W, C = feats.shape
        xs, ys = [], []
        for r in range(0, H - ps + 1, st):
            for c in range(0, W - ps + 1, st):
                m = mask[r:r+ps, c:c+ps]
                if m.size == 0:
                    continue
                valid_frac = m.mean()  # mask is 1/0
                if valid_frac < min_frac:
                    continue
                xs.append(feats[r:r+ps, c:c+ps, :])
                ys.append(label[r:r+ps, c:c+ps, :])
        if xs:
            return np.stack(xs), np.stack(ys)
        else:
            # Return empty arrays with correct last dims if nothing qualified
            return (np.empty((0, ps, ps, feats.shape[-1]), dtype=feats.dtype),
                    np.empty((0, ps, ps, 1), dtype=label.dtype))

    with h5py.File(h5_path, 'r') as hf:
        aoi_names = list(hf.keys())
        if holdout_aoi not in aoi_names:
            raise KeyError(f"Holdout AOI '{holdout_aoi}' not found in {h5_path}")

        # dev vs test sets by AOI
        dev_names  = [n for n in aoi_names if n != holdout_aoi]
        test_names = [holdout_aoi]

        rng.shuffle(dev_names)
        n_val = int(len(dev_names) * val_fraction)
        val_names   = dev_names[:n_val]
        train_names = dev_names[n_val:]

        def _load(names):
            Xs, Ys = [], []
            for name in names:
                grp = hf[name]
                feats  = grp['features'][...]          # (H, W, C)
                label  = grp['label'][...]             # (H, W, 1)
                if 'mask' in grp:
                    mask = grp['mask'][...].astype(np.uint8)
                else:
                    # build mask from SD only (not NaN and >=0)
                    sd2d = label[..., 0]
                    mask = ((~np.isnan(sd2d)) & (sd2d >= 0)).astype(np.uint8)

                x, y = _tile_one(feats, label, mask, patch_size, stride, min_valid_frac)
                if x.shape[0] > 0:
                    Xs.append(x)
                    Ys.append(y)

            if not Xs:
                # no patches extracted
                return (np.empty((0, patch_size, patch_size, feats.shape[-1]), dtype=np.float32),
                        np.empty((0, patch_size, patch_size, 1), dtype=np.float32))
            return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)

        X_train, y_train = _load(train_names)
        X_val,   y_val   = _load(val_names)
        X_hold,  y_hold  = _load(test_names)

    return (X_train, y_train), (X_val, y_val), (X_hold, y_hold)
