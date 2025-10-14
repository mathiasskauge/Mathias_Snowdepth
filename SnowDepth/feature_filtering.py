import numpy as np
import pandas as pd
from pyHSICLasso import HSICLasso
from sklearn.feature_selection import mutual_info_regression
from SnowDepth.config import SEED


def hsic_lasso_select(df, feature_cols, top_k: int = 10):
    """
    HSIC-Lasso feature selection using `pyHSICLasso`

    Returns:
    selected_features : List[str]
    weights_df : DataFrame with ['feature','score'] sorted by score desc
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ("aoi_name", "row", "col", 'SD')]

    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df['SD'].to_numpy(dtype=np.float64)

    hsic = HSICLasso()
    hsic.input(X, y)
    hsic.regression(int(top_k))

    idx = list(hsic.get_index())[:top_k]
    scores = np.asarray(hsic.get_index_score())[:top_k]

    selected = [feature_cols[i] for i in idx]
    weights_df = (pd.DataFrame({"feature": selected, "score": scores}).sort_values("score", ascending=False).reset_index(drop=True))
    return selected, weights_df


def pcc_scores(df, feature_cols):
    """
    Compute Pearson correlation between target and each feature.

    Returns a DataFrame sorted by |corr| descending with columns:
      ['feature','pcc','abs_pcc']
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ("aoi_name", "row", "col", "SD")]

    X = df[feature_cols]
    y = df["SD"]
    r = X.corrwith(y, method="pearson")
    return (
        pd.DataFrame({"feature": r.index, "pcc": r.values})
        .assign(abs_pcc=lambda d: d["pcc"].abs())
        .sort_values("abs_pcc", ascending=False)
        .reset_index(drop=True)
    )


def pcc_select(df, feature_cols, top_k: int = 10, max_intercorr: float = 0.90, min_abs_corr: float = 0.0):
    """
    Pearson-correlation-based feature filtering 

    Returns:
      selected_features, ranking_df, inter_corr_df (for the candidate pool).
    """
    rank = pcc_scores(df, feature_cols=feature_cols)

    cand = rank[rank["abs_pcc"] >= float(min_abs_corr)].copy()
    cand = cand.head(int(top_k))
    cand_feats = cand["feature"].tolist()

    inter = df[cand_feats].replace([np.inf, -np.inf], np.nan).corr(method="pearson")

    selected = []
    for f in cand_feats:
        if all(np.abs(inter.loc[f, s]) < max_intercorr for s in selected):
            selected.append(f)

    return selected, rank, inter


def mi_scores(df, feature_cols, n_neighbors=5):
    """
    Mutual Information (nonparametric) between each feature and SD.
    Returns a DataFrame sorted by MI desc with columns: ['feature','mi']
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ("aoi_name", "row", "col", "SD")]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df["SD"]

    # Drop rows with any NaN in X or y
    valid = X.notna().all(axis=1) & y.notna()
    Xv = X.loc[valid].to_numpy(dtype=float)
    yv = y.loc[valid].to_numpy(dtype=float)

    mi = mutual_info_regression(Xv, yv, n_neighbors=int(n_neighbors), random_state=SEED)
    return (
        pd.DataFrame({"feature": feature_cols, "mi": mi})
        .sort_values("mi", ascending=False)
        .reset_index(drop=True)
    )

def mi_select(df, feature_cols, top_k: int = 10, max_intercorr: float = 0.90, n_neighbors=5):
    """
    MI-based feature filtering: rank by MI, then prune by inter-feature correlation.
    Returns: selected_features, ranking_df, inter_corr_df (for candidate pool)
    """
    rank = mi_scores(df, feature_cols, n_neighbors=n_neighbors, random_state=SEED)
    cand = rank.head(int(top_k)).copy()
    cand_feats = cand["feature"].tolist()

    inter = (
        df[cand_feats]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(axis=0, how="any")
        .corr(method="pearson")
    )

    selected = []
    for f in cand_feats:
        if not selected: 
            selected.append(f); continue
        if all(np.abs(inter.loc[f, s]) < max_intercorr for s in selected if f in inter.index and s in inter.columns):
            selected.append(f)

    return selected, rank, inter