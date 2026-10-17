"""Diagnostics: XGBoost feature importance + per-fraud-type recall on current data."""

from __future__ import annotations

import os
import sys

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import numpy as np
import pandas as pd
import xgboost as xgb

from fraudgnn.config import load_baseline_config
from fraudgnn.data.features import compute_tabular_features, get_feature_column_names
from fraudgnn.training.evaluate import compute_metrics


def main() -> int:
    config = load_baseline_config("configs/baseline.yaml")

    print("Loading transactions from:", config.data_path)
    df_raw = pd.read_parquet(config.data_path)
    print(f"Total: {len(df_raw):,} tx, fraud={df_raw['is_fraud'].sum():,}")

    print("Computing tabular features...")
    df_feat = compute_tabular_features(df_raw, smoothing_alpha=config.smoothing_alpha)
    feature_cols = get_feature_column_names()

    # Temporal split
    timestamps = pd.to_datetime(df_feat["timestamp"])
    t_min, t_max = timestamps.min(), timestamps.max()
    duration = t_max - t_min
    train_cutoff = t_min + duration * config.split.train_ratio
    val_cutoff = t_min + duration * (config.split.train_ratio + config.split.val_ratio)

    train_mask = timestamps <= train_cutoff
    val_mask = (timestamps > train_cutoff) & (timestamps <= val_cutoff)
    test_mask = timestamps > val_cutoff

    train_df = df_feat[train_mask]
    val_df = df_feat[val_mask]
    test_df = df_feat[test_mask]

    X_train = train_df[feature_cols]
    y_train = train_df["is_fraud"]
    X_val = val_df[feature_cols]
    y_val = val_df["is_fraud"]
    X_test = test_df[feature_cols]
    y_test = test_df["is_fraud"]

    # Train
    y_tr = np.asarray(y_train, dtype=np.int64)
    spw = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))

    model = xgb.XGBClassifier(
        n_estimators=config.model.n_estimators,
        max_depth=config.model.max_depth,
        learning_rate=config.model.learning_rate,
        subsample=config.model.subsample,
        colsample_bytree=config.model.colsample_bytree,
        scale_pos_weight=spw,
        early_stopping_rounds=config.model.early_stopping_rounds,
        eval_metric="aucpr",
        random_state=config.model.random_state,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Feature importance (gain)
    print("\n" + "=" * 70)
    print("  TOP-10 FEATURE IMPORTANCE (GAIN)")
    print("=" * 70)
    importance = model.get_booster().get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for rank, (feat, gain) in enumerate(sorted_imp[:10], 1):
        print(f"  {rank:2d}. {feat:<30s} gain = {gain:,.2f}")

    # Val metrics & threshold selection
    y_val_prob = model.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(
        y_val.to_numpy(),
        y_val_prob,
        amounts=val_df["amount"].to_numpy(),
        review_cost=config.cost.review_cost,
        default_fraud_amount=config.cost.default_fraud_amount,
        fraud_types=val_df["fraud_type"].to_numpy(),
    )
    print(f"\nVal PR-AUC: {val_metrics.pr_auc:.4f}")
    print(f"Val best_threshold (F1-optimal): {val_metrics.best_threshold:.4f}")

    # Use val-selected threshold on test
    val_threshold = val_metrics.best_threshold

    y_test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(
        y_test.to_numpy(),
        y_test_prob,
        amounts=test_df["amount"].to_numpy(),
        review_cost=config.cost.review_cost,
        default_fraud_amount=config.cost.default_fraud_amount,
        threshold_override=val_threshold,
        fraud_types=test_df["fraud_type"].to_numpy(),
    )
    print(f"\nTest PR-AUC: {test_metrics.pr_auc:.4f}")
    print(f"Test ROC-AUC: {test_metrics.roc_auc:.4f}")
    print(f"Test recall@1%FPR: {test_metrics.recall_at_1pct_fpr:.4f}")

    # Per-fraud-type recall at val-selected threshold
    print("\n" + "=" * 70)
    print(f"  PER-FRAUD-TYPE RECALL @ Val-Selected Threshold = {val_threshold:.4f}")
    print("=" * 70)
    y_test_pred = (y_test_prob >= val_threshold).astype(int)

    fraud_types = ["fraud_ring", "mule", "card_testing", "account_takeover", "merchant_collusion"]
    for ft in fraud_types:
        ft_mask = np.asarray(test_df["fraud_type"].values == ft)
        n_total = int(np.sum(ft_mask))
        if n_total == 0:
            print(f"  {ft:<25s}: NO SAMPLES in test set")
            continue
        n_caught = int(np.sum(y_test_pred[ft_mask]))
        recall = n_caught / n_total
        print(f"  {ft:<25s}: recall = {recall:.4f} ({n_caught}/{n_total})")

    # Also show legitimate FPR
    legit_mask = np.asarray(test_df["fraud_type"].values == "legitimate")
    n_legit = int(np.sum(legit_mask))
    n_fp = int(np.sum(y_test_pred[legit_mask]))
    fpr = n_fp / n_legit if n_legit > 0 else 0.0
    print(f"  {'legitimate (FPR)':<25s}: FPR = {fpr:.4f} ({n_fp}/{n_legit})")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
