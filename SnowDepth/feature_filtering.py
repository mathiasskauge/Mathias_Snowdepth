# SnowDepth/feature_filtering.py
"""
Feature filtering utilities (shared by RF/XGB/UNet).

Implements FFPCC (Pearson-correlation-based) feature selection inspired by
Yu et al. (2024): rank features by |corr(SD, X)|, keep top-K, then remove
highly inter-correlated ones (|corr(X_i, X_j)| >= M).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, List, Tuple, Optional

# This order MUST match the feature order produced by data_loader.load_stack()
DEFAULT_FEATURE_NAMES: List[str] = [
    "VH_dB", "VV_dB", "CrossPolRatio_dB", "Elevation", "Slope", "sin_Aspect", "cos_Aspect"
]


def pcc_scores(df, feature_cols, target_col = "SD"):
    """
    Compute Pearson correlation between target and each feature.

    Returns a DataFrame sorted by |corr| descending with columns:
      ['feature','pcc','abs_pcc']
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ("aoi_name", "row", "col", target_col)]
    corrs = []
    y = df[target_col].values
    for f in feature_cols:
        x = df[f].values
        # robust to NaNs
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            r = np.nan
        else:
            r = np.corrcoef(x[mask], y[mask])[0, 1]
        corrs.append((f, r, np.abs(r) if np.isfinite(r) else -np.inf))
    out = pd.DataFrame(corrs, columns=["feature", "pcc", "abs_pcc"]).sort_values("abs_pcc", ascending=False)
    return out.reset_index(drop=True)


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

    # step 1/2: candidate pool
    cand = rank[rank["abs_pcc"] >= float(min_abs_corr)].copy()
    if top_k is not None:
        cand = cand.head(int(top_k)).copy()

    cand_feats = cand["feature"].tolist()
    if len(cand_feats) == 0:
        return [], rank, pd.DataFrame()

    # pairwise inter-feature correlation (on candidate set)
    inter = df[cand_feats].corr(method="pearson")

    # step 3: greedy prune by max_intercorr threshold
    selected: List[str] = []
    for f in cand_feats:
        ok = True
        for s in selected:
            if np.abs(inter.loc[f, s]) >= max_intercorr:
                ok = False
                break
        if ok:
            selected.append(f)

    return selected, rank, inter



# Balanced-subsample HSIC-Lasso (memory-safe)
from typing import Optional, Iterable, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

def _balanced_sample_indices(
    df: pd.DataFrame,
    target_col: str,
    by_aoi: bool = True,
    n_quantiles: int = 4,
    max_samples: int = 6000,
    seed: int = 18,
) -> np.ndarray:
    """Draw a balanced subsample across AOIs and SD quantiles."""
    rng = np.random.RandomState(seed)
    if max_samples is None or max_samples >= len(df):
        return df.index.to_numpy()

    if by_aoi and "aoi_name" in df.columns:
        groups = list(df["aoi_name"].unique())
        per_group = max(1, max_samples // max(len(groups), 1))
        idxs = []
        for g in groups:
            sub = df[df["aoi_name"] == g].copy()
            # quantiles within AOI
            sub["q"] = pd.qcut(sub[target_col], q=n_quantiles, labels=False, duplicates="drop")
            per_q = max(1, per_group // max(sub["q"].nunique(), 1))
            take = (
                sub.groupby("q", group_keys=False)
                   .apply(lambda s: s.sample(n=min(per_q, len(s)), random_state=seed))
            )
            idxs.append(take.index.to_numpy())
        idx = np.concatenate(idxs)
        # If we undershot, top up at random
        if idx.size < max_samples:
            need = max_samples - idx.size
            rest = df.index.difference(idx)
            extra = rng.choice(rest, size=min(need, rest.size), replace=False)
            idx = np.concatenate([idx, extra])
        return np.sort(idx)
    else:
        # global quantiles
        tmp = df.copy()
        tmp["q"] = pd.qcut(tmp[target_col], q=n_quantiles, labels=False, duplicates="drop")
        per_q = max(1, max_samples // max(tmp["q"].nunique(), 1))
        take = (
            tmp.groupby("q", group_keys=False)
               .apply(lambda s: s.sample(n=min(per_q, len(s)), random_state=seed))
        )
        return np.sort(take.index.to_numpy())


def _gaussian_kernel_1d(x: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    """Gaussian kernel on a 1-D vector (n,). Uses median heuristic if sigma=None."""
    x = x.reshape(-1, 1).astype(np.float32)
    d2 = (x - x.T) ** 2  # (n, n) ~ O(n^2) memory — keep n small via subsampling
    if sigma is None:
        # median heuristic (robust; avoid zero)
        nonzero = d2[d2 > 0]
        if nonzero.size == 0:
            sigma = float(np.std(x) + 1e-12)
        else:
            sigma = float(np.median(np.sqrt(nonzero)))
            if sigma <= 0:
                sigma = float(np.std(x) + 1e-12)
    K = np.exp(-d2 / (2.0 * sigma ** 2))
    return K

def _center_kernel(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n, dtype=K.dtype) - (1.0 / n) * np.ones((n, n), dtype=K.dtype)
    return H @ K @ H

def _hsic_inner(Ac: np.ndarray, Bc: np.ndarray) -> float:
    return float(np.sum(Ac * Bc))

def hsic_lasso_select(
    df: pd.DataFrame,
    feature_cols: Optional[Iterable[str]] = None,
    target_col: str = "SD",
    alpha: float = 0.01,
    top_k: Optional[int] = None,
    standardize: bool = True,
    max_samples: int = 6000,   # << NEW: cap n to keep kernels feasible (6000^2 ~ 36M elems)
    seed: int = 18,
) -> Tuple[List[str], pd.DataFrame]:
    """
    HSIC-Lasso with balanced subsampling to avoid n^2 memory blow-ups.
    Returns (selected_features, weights_df).
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ("aoi_name","row","col", target_col)]

    # balanced subsample indices (across AOIs & SD-quantiles)
    idx = _balanced_sample_indices(df, target_col=target_col, by_aoi=True,
                                   n_quantiles=4, max_samples=max_samples, seed=seed)
    sub = df.loc[idx].reset_index(drop=True)

    # target vector
    y = sub[target_col].to_numpy()
    if standardize:
        y = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-12)
    mask = np.isfinite(y)

    # ensure no NaNs in chosen features
    kept_cols = []
    Xs = []
    for f in feature_cols:
        x = sub[f].to_numpy()
        mask &= np.isfinite(x)
        Xs.append(x)
        kept_cols.append(f)

    # apply joint validity mask
    y = y[mask]
    Xs = [x[mask] for x in Xs]
    if y.size < 5:
        return [], pd.DataFrame(columns=["feature","weight"])

    # kernels
    Lc = _center_kernel(_gaussian_kernel_1d(y))
    Kcs = []
    for x in Xs:
        if standardize:
            x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
        Kcs.append(_center_kernel(_gaussian_kernel_1d(x)))

    d = len(Kcs)
    G = np.empty((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(i, d):
            val = _hsic_inner(Kcs[i], Kcs[j])
            G[i, j] = G[j, i] = val
    b = np.array([_hsic_inner(Kc, Lc) for Kc in Kcs], dtype=np.float64)

    # regularize & factorize
    eps = 1e-10 * np.trace(G) / max(d, 1)
    G_reg = G + eps * np.eye(d)
    try:
        R = np.linalg.cholesky(G_reg)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(G_reg)
        vals[vals < 0] = 0.0
        R = (np.sqrt(vals) * vecs).T

    y_t = np.linalg.solve(R, b)

    # non-negative lasso
    lasso = Lasso(alpha=alpha, fit_intercept=False, positive=True, max_iter=10000)
    lasso.fit(R, y_t)
    w = lasso.coef_.copy()

    order = np.argsort(-w)
    feats_sorted = [kept_cols[i] for i in order if w[i] > 0]
    weights_sorted = w[order][w[order] > 0]
    if top_k is not None:
        feats_sorted = feats_sorted[:top_k]
        weights_sorted = weights_sorted[:top_k]

    weights_df = pd.DataFrame({"feature": feats_sorted, "weight": weights_sorted})
    return feats_sorted, weights_df




