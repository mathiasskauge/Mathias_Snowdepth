import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KDTree


def duggal_RF_split(df, holdout_aoi, seed, pxs_per_aoi=3000, val_size=0.3, features=None):
    """
      - Filter out nan and negative SD values
      - Stratified sample "pxs_per_aoi" by SD quartiles
      - From each AOI, draw 70% of "pxs_per_aoi" -> train
      - From remaining, draw 30% of "pxs_per_aoi" -> val
      - Val points must be ≥100 m (10 pixels) from any train point
      - Concatenate across AOIs
      - Optionally select a subset of feature columns
    """

    # Separate test set (holdout AOI)
    mask = df['aoi_name'] == holdout_aoi
    df_dev  = df[~mask].reset_index(drop=True)
    df_test = df[mask].reset_index(drop=True)

    # Filter invalid snow-depth
    df_dev = df_dev.loc[(df_dev['SD'] >= 0) & (df_dev['SD'].notna())].copy()
    df_test = df_test.loc[(df_test['SD'] >= 0) & (df_test['SD'].notna())].copy()

    # Assign quartile labels within each AOI
    df_dev['sd_quartile'] = (
    df_dev
      .groupby('aoi_name')['SD']
      .transform(lambda x: pd.qcut(x, 4, labels=False, duplicates='drop')))

    # Prepare stratified splitter: 
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=int(pxs_per_aoi * (1 - val_size)),
        random_state=seed
    )

    train_parts = []
    val_parts = []

    # Loop per AOI
    for aoi, group in df_dev.groupby('aoi_name'):
        # Stratified train sampling
        train_idx, _ = next(splitter.split(group, group['sd_quartile']))
        df_train = group.iloc[train_idx].copy()

        # Build validation pool
        pool = group.drop(df_train.index).reset_index(drop=True)

        # Spatial buffer: drop any pool points closer than 10 pixels to any train point
        tree = KDTree(df_train[['row', 'col']].values)
        dists, _ = tree.query(pool[['row', 'col']].values, k=1)
        eligible = pool.iloc[np.where(dists[:, 0] >= 10)[0]].copy()

        # Sample validation points (use all if fewer than requested)
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

    # Extract arrays
    X_train = df_train[sel_cols].values
    y_train = df_train['SD'].values
    X_val = df_val[sel_cols].values
    y_val = df_val['SD'].values
    X_test = df_test[sel_cols].values
    y_test = df_test['SD'].values

    # Report
    print(f"Features used for modeling: {sel_cols}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

