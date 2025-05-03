import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(y_true, y_pred):
    """Return dict with MAE, RMSE, R2."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def evaluate(model, X, y):
    """
    Wrapper to predict on X, compare to y, and print/return metrics.
    """
    y_pred = model.predict(X)
    return compute_metrics(y, y_pred)