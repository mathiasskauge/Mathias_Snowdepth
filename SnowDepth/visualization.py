# main.py
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import tensorflow as tf

# --- Project imports ---
from SnowDepth import data_loader as DL
from SnowDepth import data_splitter as DS
from SnowDepth import architecture as ARCH
from SnowDepth.config import FEATURE_NAMES  # single source of truth for names


# -----------------------
# CONFIG
# -----------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "tif_files"
H5_DIR   = ROOT / "data" / "h5_dir"
MODELS_DIR = ROOT / "models"

# Holdout AOI name
HOLDOUT_AOI = "ID_BS"

# Tile params used in training UNet/Transformer
PATCH_SIZE = 128
STRIDE     = 64
MIN_VALID_FRAC = 0.80

# Winners (edit these to match your CV summaries)
WINNERS = {
    # Classic ML (will be retrained on all dev AOIs before scoring on holdout)
    "RF": {
        "feature_set": "HSIC",
        "params": {          # paste from your best RF printout
            "n_estimators": 300,
            "min_samples_leaf": 50,
            "max_samples": 0.5,
            "max_features": 3,
            "max_depth": 18,
            "random_state": 18,
            "n_jobs": -1,
            "bootstrap": True,
        },
    },
    "XGB": {
        "feature_set": "MI",
        "params": {         # paste from your best XGB printout
            "subsample": 1.0,
            "reg_lambda": 10,
            "reg_alpha": 1,
            "n_estimators": 1000,
            "min_child_weight": 2,
            "max_depth": 6,
            "max_bin": 256,
            "learning_rate": 0.1,
            "colsample_bytree": 0.6,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "eval_metric": "rmse",
            "n_jobs": 1,
            "random_state": 18,
        },
    },

    # Deep models (weights already trained & saved)
    "UNET": {
        "feature_set": "HSIC",  # change if another won
        "weights": MODELS_DIR / "UNet_weights" / "unet_best_HSIC.weights.h5",
    },
    "TRANS": {
        "feature_set": "HSIC",  # change if another won
        "weights": MODELS_DIR / "Transformer_weights" / "transformer_best_HSIC.weights.h5",
        # Transformer model settings (must match training)
        "model_kwargs": dict(patch_size=16, d_model=256, depth=4, num_heads=4, mlp_dim=512, dropout=0.0),
    },
}


# -----------------------
# Utilities
# -----------------------
def metric_rmse_mae(y_true, y_pred, mask=None):
    """
    Compute RMSE/MAE. If mask is provided (0/1, same shape as y), metrics are computed
    only where mask==1.
    """
    if mask is not None:
        m = mask.astype(bool)
        y_true = y_true[m]
        y_pred = y_pred[m]
    rmse = root_mean_squared_error(y_true, y_pred, squared=False)
    mae  = mean_absolute_error(y_true, y_pred)
    return rmse, mae


def build_algo_dfs(dev_df, hold_df, selected_features):
    base_cols = ["aoi_name", "row", "col", "SD"]
    cols = base_cols + list(selected_features)
    # safety
    missing = [c for c in cols if c not in dev_df.columns]
    if missing:
        raise KeyError(f"Missing columns in dev_df: {missing}")
    return dev_df[cols].copy(), hold_df[cols].copy()


def load_h5_paths_for(feature_set):
    """Return the H5 path for a given feature set name (HSIC|PCC|MI)."""
    return H5_DIR / feature_set / f"data_{feature_set}.h5"


def dl_normalize_from_dev(h5_path, holdout_aoi, patch_size, stride, min_valid_frac, seed=18):
    """
    Tile all dev AOIs to compute mean/std on features (train-time stats),
    and return (mean, std) to normalize holdout tiles consistently.
    """
    (X_dev, _, _), (_, _, _) = DS.unet_split(
        h5_path=str(h5_path),
        holdout_aoi=holdout_aoi,
        val_fraction=0.0,     # all dev tiles
        seed=seed,
        patch_size=patch_size,
        stride=stride,
        min_valid_frac=min_valid_frac
    )
    mean = X_dev.mean(axis=(0, 1, 2), keepdims=True)
    std  = X_dev.std(axis=(0, 1, 2), keepdims=True) + 1e-6
    return mean.astype("float32"), std.astype("float32")


def normalize_with_stats(x, mean, std):
    return (x - mean) / std


# -----------------------
# Classic ML evaluation (RF, XGB)
# -----------------------
def eval_tabular_model(model_name, model_obj, feature_set, df):
    print(f"\n=== {model_name}: evaluating on holdout AOI ({HOLDOUT_AOI}) with feature set {feature_set} ===")

    # Limit DataFrame to selected features
    # We'll read the features to use from the H5 feature list stored as attrs
    # but since you already know the set, just rebuild from CSV by names.
    # Build full DF once, then slice below.
    dev_df = df[df["aoi_name"] != HOLDOUT_AOI].copy()
    hold_df = df[df["aoi_name"] == HOLDOUT_AOI].copy()

    # Selected features come from the H5 header (already written) or from your earlier dict.
    # Here we load from the H5 attributes (single source of truth).
    h5_path = load_h5_paths_for(feature_set)
    with h5py.File(h5_path, "r") as hf:
        feat_names = [n.decode("utf-8") for n in hf.attrs["feature_names"]]

    dev_df_fs, hold_df_fs = build_algo_dfs(dev_df, hold_df, feat_names)

    # Split to arrays (sample dev per AOI for balance; then fit on that sample)
    X_dev, y_dev, groups, X_hold, y_hold = DS.ML_split(
        dev_df=dev_df_fs,
        hold_df=hold_df_fs,
        seed=18,
        pxs_per_aoi=10000,
        return_holdout=True
    )

    # Fit on sampled dev; you can also re-fit on all dev afterwards if desired
    model_obj.fit(X_dev, y_dev)
    y_pred_hold = model_obj.predict(X_hold)

    rmse, mae = metric_rmse_mae(y_hold, y_pred_hold)
    print(f"{model_name} — Holdout RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return {"rmse": rmse, "mae": mae, "y_true": y_hold, "y_pred": y_pred_hold}


# -----------------------
# DL models evaluation (UNet, Transformer)
# -----------------------
def eval_dl_model(model_name, build_model_fn, weights_path, feature_set, seed=18):
    print(f"\n=== {model_name}: evaluating on holdout AOI ({HOLDOUT_AOI}) with feature set {feature_set} ===")

    h5_path = load_h5_paths_for(feature_set)

    # Compute train-time normalization stats from dev tiles
    mean, std = dl_normalize_from_dev(
        h5_path=h5_path,
        holdout_aoi=HOLDOUT_AOI,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        min_valid_frac=MIN_VALID_FRAC,
        seed=seed
    )

    # Load dev just to get shapes (not used), and holdout tiles for inference
    # (We only need holdout tiles/masks/labels here)
    (_, _, _), (_, _, _), (X_hold, y_hold, m_hold) = DS.unet_split(
        h5_path=str(h5_path),
        holdout_aoi=HOLDOUT_AOI,
        val_fraction=0.0,           # not used; we only extract hold here
        seed=seed,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        min_valid_frac=MIN_VALID_FRAC,
        return_holdout=True
    )

    # Normalize holdout with dev stats
    X_hold_n = normalize_with_stats(X_hold, mean, std).astype("float32")

    # Prepare labels & per-pixel weights
    y_hold_f, w_hold = ARCH.fill_nan_and_mask(y_hold)
    w_hold_4d = w_hold[..., None].astype("float32")

    # Build and load model
    model = build_model_fn(input_shape=X_hold_n.shape[1:])
    model.load_weights(str(weights_path))

    # Predict and score (mask-aware)
    y_pred = model.predict(X_hold_n, batch_size=4, verbose=1)
    rmse, mae = metric_rmse_mae(y_hold_f, y_pred, mask=w_hold_4d)
    print(f"{model_name} — Holdout RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return {"rmse": rmse, "mae": mae, "y_true": y_hold_f, "y_pred": y_pred, "mask": w_hold_4d}


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    import h5py

    # Build full DF once for tabular models
    print("Loading full DataFrame...")
    df = DL.build_df(str(DATA_DIR), drop_invalid=True, upper_threshold=3)

    results = {}

    # ----- Random Forest -----
    rf_cfg = WINNERS["RF"]
    rf = RandomForestRegressor(**rf_cfg["params"])
    results["RF"] = eval_tabular_model("RF", rf, rf_cfg["feature_set"], df)

    # ----- XGBoost -----
    xgb_cfg = WINNERS["XGB"]
    xgb = XGBRegressor(**xgb_cfg["params"])
    results["XGB"] = eval_tabular_model("XGB", xgb, xgb_cfg["feature_set"], df)

    # ----- UNet -----
    unet_cfg = WINNERS["UNET"]
    def build_unet(input_shape):
        return ARCH.unet(input_shape=input_shape, base_filters=32)
    results["UNET"] = eval_dl_model(
        "UNet",
        build_unet,
        weights_path=unet_cfg["weights"],
        feature_set=unet_cfg["feature_set"],
        seed=18
    )

    # ----- Transformer -----
    trans_cfg = WINNERS["TRANS"]
    def build_trans(input_shape):
        return ARCH.transformer_seg_model(
            input_shape=input_shape,
            **trans_cfg.get("model_kwargs", {})
        )
    results["TRANS"] = eval_dl_model(
        "Transformer",
        build_trans,
        weights_path=trans_cfg["weights"],
        feature_set=trans_cfg["feature_set"],
        seed=18
    )

    # Summary
    print("\n=== Holdout summary (RMSE ↓, MAE ↓) ===")
    for k, v in results.items():
        print(f"{k:11s} | RMSE: {v['rmse']:.4f} | MAE: {v['mae']:.4f}")

    # Save a small JSON with metrics
    out_json = ROOT / "holdout_results.json"
    with open(out_json, "w") as f:
        json.dump({k: {"rmse": v["rmse"], "mae": v["mae"]} for k, v in results.items()}, f, indent=2)
    print(f"\nWrote metrics to: {out_json}")
