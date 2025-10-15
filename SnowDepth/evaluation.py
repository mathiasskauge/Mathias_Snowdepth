import numpy as np

def nan_safe_normalize(x, mu, sigma):
    """Normalize with NaN/Inf protection"""
    x = np.nan_to_num(x, nan=mu, posinf=mu, neginf=mu).astype("float32")
    return ((x - mu) / sigma).astype("float32")

def valid_idx(y_true, y_pred, mask):
    t = np.squeeze(np.asarray(y_true, dtype=float))
    p = np.squeeze(np.asarray(y_pred, dtype=float))
    m = np.squeeze(np.asarray(mask, dtype=float))
    # broadcast 2D mask to 3D if needed
    if m.ndim == 2 and t.ndim == 3 and t.shape[-1] == 1: m = m[..., None]
    if m.ndim == 2 and p.ndim == 3 and p.shape[-1] == 1: m = m[..., None]
    vm = (m > 0)
    if vm.ndim == 3 and vm.shape[-1] == 1: vm = vm[..., 0]
    return vm & np.isfinite(np.squeeze(t)) & np.isfinite(np.squeeze(p))

def mae_rmse_r2(y_true, y_pred, mask):
    """Compute MAE/RMSE/R² on valid finite pixels """
    idx = valid_idx(y_true, y_pred, mask)
    if not np.any(idx):
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "count": 0.0}
    t = np.squeeze(y_true)[idx].astype(float)
    p = np.squeeze(y_pred)[idx].astype(float)
    mae  = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t)**2)))
    mu   = float(np.mean(t))
    ss_res, ss_tot = float(np.sum((p - t)**2)), float(np.sum((t - mu)**2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res/ss_tot
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def evaluate_tiles(X_hold, y_hold, m_hold, *, model, mu, sigma, batch_size=4):
    """
    Evaluate metrics on holdout tiles.
    """
    Xn = nan_safe_normalize(X_hold, mu, sigma)
    pred = model.predict(Xn, batch_size=batch_size, verbose=0)
    return mae_rmse_r2(y_hold, pred, m_hold)
