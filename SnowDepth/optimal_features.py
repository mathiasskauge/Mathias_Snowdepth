import json
from pathlib import Path
import SnowDepth.data_loader as DL
from SnowDepth.feature_filtering import hsic_lasso_select, pcc_select


def optimal_feature_sets_json(data_dir, holdout_aoi = "ID_BS", top_k = 12, upper_threshold=3.0, out_json = "optimal_features.json"):
    
    # Build df
    df = DL.build_df(str(data_dir), drop_invalid=True, upper_threshold=upper_threshold)

    # Exclude holdout AOI
    dev_df = df[df["aoi_name"] != holdout_aoi].copy()

    # HSIC-Lasso
    hsic_feats = get_hsic_features(
        df=dev_df,
        feature_cols=DL.FEATURE_NAMES,
        top_k=top_k,
    )

    # PCC
    pcc_feats = get_PCC_features(
        df=dev_df,
        feature_cols=DL.FEATURE_NAMES,
        top_k=top_k,
        max_intercorr=0.90,
        min_abs_corr=0.0
    )

    # Save with metadata
    payload = {
        "sets": {
            "HSIC": hsic_feats,
            "PCC": pcc_feats,
        },
        "meta": {
            "top_k": top_k,
            "data_dir": str(data_dir),
            "holdout_aoi": holdout_aoi,
            "upper_threshold": upper_threshold,
            "feature_pool": DL.FEATURE_NAMES,
        },
    }
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"HSIC (top {top_k}):  {hsic_feats}")
    print(f"PCC (top {top_k}): {pcc_feats}")
    print(f"Wrote feature sets to: {out_path.resolve()}")

    return out_path


def get_hsic_features(df, feature_cols, top_k):
    hsic_feats, hsic_scores = hsic_lasso_select(df, feature_cols, top_k)
    return hsic_feats


def get_PCC_features(df, feature_cols, top_k, max_intercorr, min_abs_corr):
    pcc_feats, pcc_rank, inter_corr = pcc_select(df, feature_cols, top_k, max_intercorr, min_abs_corr)
    return pcc_feats

