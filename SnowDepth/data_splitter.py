import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KDTree

"""
Implements splitting strategies for different models

"""

def RF_split(dev_df, hold_df, seed, pxs_per_aoi=3000):
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


def duggal_RF_split(dev_df, hold_df, seed, pxs_per_aoi=3000, val_size=0.3, features=None):
    """
      - Stratified sample "pxs_per_aoi" by SD quartiles
      - From each AOI, draw 70% of "pxs_per_aoi" -> train
      - From remaining, draw 30% of "pxs_per_aoi" -> val
      - Val points must be ≥100 m (10 pixels) away from any train point
      - Optionally select a subset of feature columns
    """

    # Assign quartile labels within each AOI
    dev_df['sd_quartile'] = (
    dev_df
      .groupby('aoi_name')['SD']
      .transform(lambda x: pd.qcut(x, 4, labels=False, duplicates='drop')))

    # Prepare stratified splitter: 
    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=int(pxs_per_aoi * (1 - val_size)),
        random_state=seed
    )

    train_parts = []
    val_parts = []

    # Loop per AOI
    for aoi, group in dev_df.groupby('aoi_name'):
        # Stratified train sampling
        train_idx, _ = next(sss.split(group, group['sd_quartile']))
        df_train = group.iloc[train_idx].copy()

        # Build validation pool
        pool = group.drop(df_train.index).reset_index(drop=True)

        # Spatial buffer: drop any pool points closer than 10 pixels to any train point
        tree = KDTree(df_train[['row', 'col']].values)
        dists, _ = tree.query(pool[['row', 'col']].values, k=1)
        eligible = pool.iloc[np.where(dists[:, 0] >= 10)[0]].copy()

        # Sample validation points (use all if fewer eligeble than requested)
        val_count = int(pxs_per_aoi * val_size)
        if len(eligible) < val_count:
            print(f"AOI {aoi}: only {len(eligible)} eligible val samples, wanted {val_count}; using all {len(eligible)} for validation")
            df_val = eligible.copy()
        else:
            df_val = eligible.sample(n=val_count, random_state=seed)

        train_parts.append(df_train)
        val_parts.append(df_val)

    # Concatenate across AOIs
    df_train = pd.concat(train_parts, ignore_index=True)
    df_val = pd.concat(val_parts, ignore_index=True)

    # Determine feature columns
    all_features = [c for c in df_train.columns if c not in ('aoi_name', 'row', 'col', 'sd_quartile', 'SD')]
    if features is None:
        sel_cols = all_features
    else:
        if all(isinstance(f, int) for f in features):
            sel_cols = [all_features[i] for i in features]
        else:
            missing = [f for f in features if f not in df_train.columns]
            if missing:
                raise KeyError(f"Features not found: {missing}")
            sel_cols = list(features)

    # Build development arrays
    X_train = df_train[sel_cols].values
    y_train = df_train['SD'].values
    X_val = df_val[sel_cols].values
    y_val = df_val['SD'].values

    # Build hold‐out arrays using the same feature columns 
    X_hold = hold_df[sel_cols].values
    y_hold = hold_df['SD'].values

    # Report
    print(f"Features used for modeling: {sel_cols}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_hold shape: {X_hold.shape}")

    return X_train, y_train, X_val, y_val, X_hold, y_hold



def dl_unet_split(h5_path, holdout_aoi, val_fraction=0.3, seed=18):
    """
    Split HDF5 dataset into train/val/test numpy arrays for UNet training.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file containing one group per AOI:
          group:
            'features' → (H, W, C) float32
            'label'    → (H, W, 1) float32
    holdout_aoi : str
        Name of the AOI group to reserve for test.
    val_fraction : float, optional
        Fraction of the *remaining* AOIs to reserve for validation (default=0.3).
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_hold, y_hold)
        Tuples of numpy arrays:
          - X: shape (N_images, H, W, C)
          - y: shape (N_images, H, W, 1)
    """
    rng = np.random.RandomState(seed)

    with h5py.File(h5_path, 'r') as hf:
        aoi_names = list(hf.keys())
        if holdout_aoi not in aoi_names:
            raise KeyError(f"Holdout AOI '{holdout_aoi}' not found in {h5_path}")

        # carve out test AOI
        dev_names  = [n for n in aoi_names if n != holdout_aoi]
        test_names = [holdout_aoi]

        # shuffle & split dev AOIs into train / val
        rng.shuffle(dev_names)
        n_val = int(len(dev_names) * val_fraction)
        val_names   = dev_names[:n_val]
        train_names = dev_names[n_val:]

        # loader helper
        def _load(names):
            X_list, y_list = [], []
            for name in names:
                grp = hf[name]
                X_list.append(grp['features'][...])
                y_list.append(grp['label'][...])
            return np.stack(X_list), np.stack(y_list)

        X_train, y_train = _load(train_names)
        X_val,   y_val   = _load(val_names)
        X_hold,  y_hold  = _load(test_names)

    return (X_train, y_train), (X_val, y_val), (X_hold, y_hold)