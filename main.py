
import torch
import numpy as np
from SnowDepth import DS, ARCH
from SnowDepth.models import TransformerSD  # adjust to your actual model name

# ----------------------------
# CONFIG
# ----------------------------
h5_path = "data/processed/HSIC/data.h5"
holdout_aoi = "ID_BS"      # example AOI
checkpoint_path = "checkpoints/transformer_best_HSIC.h5"
seed = 18
patch_size = 128
stride = 64
min_valid_frac = 0.80
device = "cuda" if torch.cuda.is_available() else "cpu"

(_, _, _), (_, _, _), (X_hold, y_hold, m_hold) = DS.DL_split(
    h5_path=h5_path,
    holdout_aoi=holdout_aoi,
    val_fraction=0.10,
    seed=seed,
    patch_size=patch_size,
    stride=stride,
    min_valid_frac=min_valid_frac
)

# Normalize and clean
X_hold_n = ARCH.zscore_from_train_ref(X_hold)  # or zscore_from_train if you saved stats
y_hold_f, w_hold = ARCH.fill_nan_and_mask(y_hold)

X_hold_n = X_hold_n.astype("float32")
y_hold_f = y_hold_f.astype("float32")

print(f"Holdout AOI: {holdout_aoi}, tiles: {X_hold_n.shape[0]}")

# ----------------------------
# LOAD MODEL
# ----------------------------
# Instantiate model (adjust args to your training setup)
C = X_hold_n.shape[-1]
model = TransformerSD(in_channels=C, embed_dim=128, num_layers=4).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ----------------------------
# INFERENCE
# ----------------------------
preds = []
with torch.no_grad():
    for i in range(0, len(X_hold_n)):
        x = torch.from_numpy(X_hold_n[i:i+1]).permute(0, 3, 1, 2).to(device)  # NCHW
        y_pred = model(x)
        y_pred = y_pred.permute(0, 2, 3, 1).cpu().numpy()  # NHWC
        preds.append(y_pred[0])

y_pred_full = np.stack(preds, axis=0)

print("Inference complete.")
print("Pred shape:", y_pred_full.shape)

# ----------------------------
# EVALUATE
# ----------------------------
mae = np.mean(np.abs(y_pred_full[m_hold == 1] - y_hold_f[m_hold == 1]))
rmse = np.sqrt(np.mean((y_pred_full[m_hold == 1] - y_hold_f[m_hold == 1])**2))
print(f"Holdout MAE: {mae:.3f}, RMSE: {rmse:.3f}")

# ----------------------------
# SAVE OUTPUT
# ----------------------------
np.savez(f"outputs/preds_{holdout_aoi}_HSIC.npz",
         pred=y_pred_full, gt=y_hold_f, mask=m_hold)

print(f"Saved predictions to outputs/preds_{holdout_aoi}_HSIC.npz")
