# main.py
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from SnowDepth import data_loader as DL
from SnowDepth import data_splitter as DS
from SnowDepth import architecture as ARCH

# --- CONFIG ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "tif_files"
H5_DIR = ROOT / "data" / "h5_dir"
MODELS_DIR = ROOT / "models"
HOLDOUT_AOI = "ID_BS"
SEED = 18

# Best models & feature sets
RF_best = {"feature_set": "HSIC",
           "params": {"n_estimators": 300, "min_samples_leaf": 50,
                      "max_samples": 0.5, "max_features": 3,
                      "max_depth": 18, "random_state": SEED, "n_jobs": -1}}

XGB_best = {"feature_set": "MI",
            "params": {"subsample": 1.0, "reg_lambda": 10, "reg_alpha": 1,
                       "n_estimators": 1000, "min_child_weight": 2,
                       "max_depth": 6, "max_bin": 256, "learning_rate": 0.1,
                       "colsample_bytree": 0.6, "random_state": SEED}}

UNET_best = {"feature_set": "HSIC",
             "weights": MODELS_DIR / "UNet_weights" / "unet_best_HSIC.weights.h5"}

TRANS_best = {"feature_set": "HSIC",
              "weights": MODELS_DIR / "Transformer_weights" / "transformer_best_HSIC.weights.h5"}


# --- HELPERS ---
def rmse_mae(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred, squared=False),
            mean_absolute_error(y_true, y_pred))


# --- LOAD DATA ---
print("Loading dataframe...")
df = DL.build_df(str(DATA_DIR), drop_invalid=True, upper_threshold=3)
dev_df = df[df["aoi_name"] != HOLDOUT_AOI].copy()
hold_df = df[df["aoi_name"] == HOLDOUT_AOI].copy()


# --- RF ---
print("\n🟢 Random Forest (HSIC)")
rf = RandomForestRegressor(**RF_best["params"])
X_dev, y_dev, groups, X_hold, y_hold = DS.ML_split(dev_df, hold_df, SEED)
rf.fit(X_dev, y_dev)
y_pred = rf.predict(X_hold)
rmse, mae = rmse_mae(y_hold, y_pred)
print(f"RF Holdout RMSE={rmse:.3f}, MAE={mae:.3f}")


# --- XGBoost ---
print("\n🟣 XGBoost (MI)")
xgb = XGBRegressor(**XGB_best["params"])
X_dev, y_dev, groups, X_hold, y_hold = DS.ML_split(dev_df, hold_df, SEED)
xgb.fit(X_dev, y_dev)
y_pred = xgb.predict(X_hold)
rmse, mae = rmse_mae(y_hold, y_pred)
print(f"XGB Holdout RMSE={rmse:.3f}, MAE={mae:.3f}")


# --- UNet ---
print("\n🔵 UNet (HSIC)")
h5_path = H5_DIR / "HSIC" / "data_HSIC.h5"
(_, _, _), (_, _, _), (X_hold, y_hold, m_hold) = DS.unet_split(
    h5_path=str(h5_path),
    holdout_aoi=HOLDOUT_AOI,
    val_fraction=0.0,
    patch_size=128,
    stride=64,
    min_valid_frac=0.8
)
y_hold_f, w_hold = ARCH.fill_nan_and_mask(y_hold)
model_unet = ARCH.unet(input_shape=X_hold.shape[1:], base_filters=32)
model_unet.load_weights(str(UNET_best["weights"]))
y_pred = model_unet.predict(X_hold, verbose=0)
mask = w_hold[..., None].astype(bool)
rmse, mae = rmse_mae(y_hold_f[mask], y_pred[mask])
print(f"UNet Holdout RMSE={rmse:.3f}, MAE={mae:.3f}")


# --- Transformer ---
print("\n🟠 Transformer (HSIC)")
model_trans = ARCH.transformer_seg_model(
    input_shape=X_hold.shape[1:],
    patch_size=16, d_model=256, depth=4, num_heads=4, mlp_dim=512, dropout=0.0
)
model_trans.load_weights(str(TRANS_best["weights"]))
y_pred = model_trans.predict(X_hold, verbose=0)
rmse, mae = rmse_mae(y_hold_f[mask], y_pred[mask])
print(f"Transformer Holdout RMSE={rmse:.3f}, MAE={mae:.3f}")

print("\n✅ Done.")
