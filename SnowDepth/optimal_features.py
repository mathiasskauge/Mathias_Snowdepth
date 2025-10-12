import pandas as pd
import SnowDepth.data_loader as DL
from SnowDepth.feature_filtering import hsic_lasso_select, pcc_select, mi_select
from SnowDepth.config import FEATURE_NAMES


def optimal_feature_sets(df, top_k, n_per_aoi):

    # Sample 10000 pixels from each AOI in df
    samples = sample_per_aoi(df, n_per_aoi=n_per_aoi)

    # HSIC-Lasso
    hsic_feats, _ = hsic_lasso_select(
        df=samples,
        feature_cols=FEATURE_NAMES,
        top_k=top_k,
    )
    # PCC
    pcc_feats, _, _ = pcc_select(
        df=samples,
        feature_cols=FEATURE_NAMES,
        top_k=top_k,
        max_intercorr=0.90,
        min_abs_corr=0.0
    )
    # MI
    mi_feats, _, _ = mi_select(
        df=samples,
        feature_cols=FEATURE_NAMES,
        top_k=top_k,
        max_intercorr=0.90,
        n_neighbors=5,
        random_state=42,
    )

    # Create dict with algorithms and their selected features
    selected_features = {"HSIC": hsic_feats, "PCC": pcc_feats, "MI": mi_feats}

    print(f"HSIC (top {top_k}): {hsic_feats}")
    print(f"PCC (top {top_k}): {pcc_feats}")
    print(f"MI (top {top_k}): {mi_feats}")
    return selected_features


# Helper for sammpling
def sample_per_aoi(df, n_per_aoi, random_state=18):
    samples = []
    for aoi, group in df.groupby("aoi_name"):
        n = min(n_per_aoi, len(group))
        samples.append(group.sample(n=n, random_state=random_state))
    return pd.concat(samples, ignore_index=True)





