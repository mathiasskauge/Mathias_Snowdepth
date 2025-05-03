import numpy as np
import h5py
from sklearn.model_selection import train_test_split

def split_data(strategy, train_h5, seed, features=None):
    """
    Dispatch to different splitting strategies.

    Parameters
    ----------
    strategy : str
        Which split style to use.

    train_h5 : str
        Path to train.h5 containing /data, /labels, /aoi.

    seed : int 

    features : list of int (optional)
        Indices of feature columns to select from data. If None, use all features.

    Returns
    -------
    train : tuple (X_train, y_train)
    val   : tuple (X_val,   y_val)
    """
    if strategy == "simple_rf":
        return simple_RF_split(train_h5, seed, features=features)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")



def simple_RF_split(train_h5, seed, features=None):
    """
    RF-style split:
      - Sample n_samples_per_aoi pixels from each AOI in train_h5
      - Split each AOI’s samples into 70% train / 30% val 
      - Concatenate across AOIs
      - Optionally select a subset of feature columns
    """
    # Read HDF5 contents
    with h5py.File(train_h5, "r") as f:
        data   = f["data"][:]       # All feature vectors 
        labels = f["labels"][:]     # All targets 
        aoi    = f["aoi"][:]        # All AOI IDs     
        # Decode byte-strings if necessary
        if aoi.dtype.kind == "S":
            aoi = aoi.astype(str)

    # Number of samples per AOI
    n_samples_per_aoi = 4000

    # Size of validation set
    val_size = 0.3

    # If feature subset requested, apply it
    if features is not None:
        data = data[:, features]

    # Prepare lists for train/val indices
    train_idx_list = []
    val_idx_list   = []

    # Process each AOI separately
    for aoi_name in np.unique(aoi):
        # Get all indices for this AOI
        idx_all = np.flatnonzero(aoi == aoi_name)

        # Filter out invalid SD (<=0 or NaN)
        valid_mask = (labels > 0) & ~np.isnan(labels)
        idx_valid  = idx_all[valid_mask[idx_all]]

        if idx_valid.size < n_samples_per_aoi:
            raise ValueError(
                f"AOI '{aoi_name}' has only {idx_valid.size} valid samples (need {n_samples_per_aoi})"
            )
        if idx_all.size < n_samples_per_aoi:
            raise ValueError(f"AOI '{aoi_name}' has only {idx_all.size} samples (need {n_samples_per_aoi})")

        # Initialize random generator
        rng = np.random.RandomState(seed)

        # Randomly sample without replacement
        chosen = rng.choice(idx_valid, size=n_samples_per_aoi, replace=False)

        # Split sampled indices into train/val
        tr_idx, val_idx = train_test_split(chosen, test_size=val_size, random_state=seed)
        train_idx_list.append(tr_idx)
        val_idx_list.append(val_idx)

    # Concatenate indices from all AOIs
    train_idx = np.concatenate(train_idx_list)
    val_idx   = np.concatenate(val_idx_list)

    # Extract train and validation sets
    X_train = data[train_idx]
    y_train = labels[train_idx]
    X_val   = data[val_idx]
    y_val   = labels[val_idx]

    # Determine feature names and which ones were used
    all_feature_names = [
        'VH_dB','VV_dB','CrossPolRatio_dB', 'DEM', 'sin(Aspect)','cos(Aspect)', 'Slope']
    if features is None:
        used_names = all_feature_names
    else:
        used_names = [all_feature_names[i] for i in features]

    # Print split summary
    print(f"RF split: train samples = {X_train.shape[0]}, val samples = {X_val.shape[0]}, features = {used_names}")

    return (X_train, y_train), (X_val, y_val)
