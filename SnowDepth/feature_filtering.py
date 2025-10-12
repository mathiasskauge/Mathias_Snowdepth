from __future__ import annotations
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
from pyHSIClasso import HSICLasso


# This order MUST match the feature order produced by data_loader.load_stack()
DEFAULT_FEATURE_NAMES: List[str] = [
    # Base backscatter in dB
    "Sigma_VH", "Sigma_VV",
    "Gamma_VH", "Gamma_VV",
    "Beta_VH",  "Beta_VV",
    "Gamma_VH_RTC", "Gamma_VV_RTC",
    # Linear sums/differences
    "Sigma_sum", "Gamma_sum", "Beta_sum", "Gamma_RTC_sum",
    "Sigma_diff", "Gamma_diff", "Beta_diff", "Gamma_RTC_diff",
    # Ratios (in dB)
    "Sigma_ratio", "Gamma_ratio", "Beta_ratio", "Gamma_RTC_ratio",
    # Angles
    "LIA", "IAFE",
    # Topography
    "Elevation", "Slope", "sin_Aspect", "cos_Aspect",
]


# -------------------------- FFPCC (Pearson) --------------------------

def pcc_scores(
    df: pd.DataFrame,
    feature_cols: Optional[Iterable[str]] = None,
    target_col: str = "SD",
) -> pd.DataFrame:
    """
    Compute Pearson correlation between target and each feature.

    Returns a DataFrame sorted by |corr| descending with columns:
      ['feature','pcc','abs_pcc']
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ("aoi_name", "row", "col", target_col)]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df[target_col]
    r = X.corrwith(y, method="pearson")
    return (
        pd.DataFrame({"feature": r.index, "pcc": r.values})
        .assign(abs_pcc=lambda d: d["pcc"].abs())
        .sort_values("abs_pcc", ascending=False)
        .reset_index(drop=True)
    )


def select_ffpcc(
    df: pd.DataFrame,
    feature_cols: Optional[Iterable[str]] = None,
    target_col: str = "SD",
    top_k: Optional[int] = None,
    max_intercorr: float = 0.90,
    min_abs_corr: float = 0.0,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """
    Pearson-correlation-based feature filtering (FFPCC).

    Steps:
      1) Rank features by |corr(target, feature)|.
      2) Keep top_k (if given) AND those with |corr| >= min_abs_corr.
      3) Greedy prune to ensure pairwise |corr| < max_intercorr among selected.

    Returns:
      selected_features, ranking_df, inter_corr_df (for the candidate pool).
    """
    rank = pcc_scores(df, feature_cols=feature_cols, target_col=target_col)

    cand = rank[rank["abs_pcc"] >= float(min_abs_corr)].copy()
    if top_k is not None:
        cand = cand.head(int(top_k))
    cand_feats = cand["feature"].tolist()
    if not cand_feats:
        return [], rank, pd.DataFrame()

    inter = df[cand_feats].replace([np.inf, -np.inf], np.nan).corr(method="pearson")

    selected: List[str] = []
    for f in cand_feats:
        if all(np.abs(inter.loc[f, s]) < max_intercorr for s in selected):
            selected.append(f)

    return selected, rank, inter


# -------------------------- HSIC-Lasso (library) --------------------------

def hsic_lasso_select(
    df: pd.DataFrame,
    feature_cols: Optional[Iterable[str]] = None,
    target_col: str = "SD",
    top_k: int = 10,
    random_state: int = 18,
) -> Tuple[List[str], pd.DataFrame]:
    """
    HSIC-Lasso feature selection for regression using `pyHSICLasso`.

    Parameters
    ----------
    df : DataFrame with features + target
    feature_cols : columns to consider (default: all non-meta, non-target)
    target_col : name of target column (default 'SD')
    top_k : number of features to select
    random_state : seed for reproducibility

    Returns
    -------
    selected_features : List[str]
    weights_df : DataFrame with columns ['feature','score'] sorted by score desc
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ("aoi_name", "row", "col", target_col)]

    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df[target_col].to_numpy(dtype=np.float64)

    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid]
    y = y[valid]

    hsic = HSICLasso(random_state=random_state)
    hsic.input(X, y, featname=list(feature_cols))
    hsic.regression(numFeat=int(top_k))   # library chooses lambda internally

    idx = hsic.get_index()                # indices of selected features (desc)
    rel = hsic.get_relevance()            # relevance scores (aligned with idx)

    idx = list(idx[:top_k])
    selected = [feature_cols[i] for i in idx]
    scores = rel[:len(selected)]

    weights_df = (
        pd.DataFrame({"feature": selected, "score": scores})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    return selected, weights_df
