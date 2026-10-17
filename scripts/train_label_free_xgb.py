"""Train XGBoost on tabular features + strictly label-free causal graph features.

Fair baseline comparison for Graph Neural Networks (which are label-free by design).
Uses:
1. 27 causal tabular features without past fraud labels (log_merch_tx_count_past instead of merchant_risk_rate).
2. Label-free structural graph features:
   - device_distinct_users: number of distinct past users seen on device
   - ip_distinct_users: number of distinct past users seen on IP
   - merchant_degree: number of distinct past users who transacted at merchant
   - user_device_count: number of distinct devices used by user
   - user_ip_count: number of distinct IPs used by user
   - user_card_count: number of distinct cards used by user
   - two_hop_device_users: number of other users sharing the transaction device
   - two_hop_ip_users: number of other users sharing the transaction IP

Zero label-derived features are used (no fraud rates, fraud counts, or fraud_type).
Threshold is strictly chosen on validation data only and applied unchanged to test.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from fraudgnn.config import load_baseline_config
from fraudgnn.data.features import compute_tabular_features, get_feature_column_names
from fraudgnn.training.evaluate import compute_metrics


def compute_label_free_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly causal, label-free structural graph features per transaction.

    Args:
        df: Chronologically sorted transaction DataFrame.

    Returns:
        DataFrame of label-free graph feature columns.
    """
    n = len(df)
    users = df["user_id"].to_numpy()
    devices = df["device_id"].to_numpy()
    ips = df["ip_address"].to_numpy()
    merchants = df["merchant_id"].to_numpy()
    cards = df["card_id"].to_numpy()

    # Output arrays
    device_distinct_users = np.zeros(n, dtype=np.float32)
    ip_distinct_users = np.zeros(n, dtype=np.float32)
    merchant_degree = np.zeros(n, dtype=np.float32)
    user_device_count = np.zeros(n, dtype=np.float32)
    user_ip_count = np.zeros(n, dtype=np.float32)
    user_card_count = np.zeros(n, dtype=np.float32)
    two_hop_device_users = np.zeros(n, dtype=np.float32)
    two_hop_ip_users = np.zeros(n, dtype=np.float32)

    # State registries for causal streaming computation
    device_to_users: dict[str, set[str]] = {}
    ip_to_users: dict[str, set[str]] = {}
    merchant_to_users: dict[str, set[str]] = {}
    user_to_devices: dict[str, set[str]] = {}
    user_to_ips: dict[str, set[str]] = {}
    user_to_cards: dict[str, set[str]] = {}

    empty_set: set[str] = set()

    for i in range(n):
        u = str(users[i])
        d = str(devices[i])
        p = str(ips[i])
        m = str(merchants[i])
        c = str(cards[i])

        # 1. Causal Reads (Strictly past info, excluding current transaction)
        dev_users = device_to_users.get(d, empty_set)
        ip_users = ip_to_users.get(p, empty_set)
        merch_users = merchant_to_users.get(m, empty_set)
        u_devs = user_to_devices.get(u, empty_set)
        u_ips = user_to_ips.get(u, empty_set)
        u_cards = user_to_cards.get(u, empty_set)

        device_distinct_users[i] = len(dev_users)
        ip_distinct_users[i] = len(ip_users)
        merchant_degree[i] = len(merch_users)
        user_device_count[i] = len(u_devs)
        user_ip_count[i] = len(u_ips)
        user_card_count[i] = len(u_cards)
        two_hop_device_users[i] = len(dev_users - {u})
        two_hop_ip_users[i] = len(ip_users - {u})

        # 2. State Updates
        if d not in device_to_users:
            device_to_users[d] = set()
        device_to_users[d].add(u)

        if p not in ip_to_users:
            ip_to_users[p] = set()
        ip_to_users[p].add(u)

        if m not in merchant_to_users:
            merchant_to_users[m] = set()
        merchant_to_users[m].add(u)

        if u not in user_to_devices:
            user_to_devices[u] = set()
        user_to_devices[u].add(d)

        if u not in user_to_ips:
            user_to_ips[u] = set()
        user_to_ips[u].add(p)

        if u not in user_to_cards:
            user_to_cards[u] = set()
        user_to_cards[u].add(c)

    return pd.DataFrame(
        {
            "device_distinct_users": device_distinct_users,
            "ip_distinct_users": ip_distinct_users,
            "merchant_degree": merchant_degree,
            "user_device_count": user_device_count,
            "user_ip_count": user_ip_count,
            "user_card_count": user_card_count,
            "two_hop_device_users": two_hop_device_users,
            "two_hop_ip_users": two_hop_ip_users,
        }
    )


def train_and_eval_label_free_xgb(config_path: str = "configs/baseline.yaml") -> dict[str, Any]:
    """Train XGBoost on tabular + label-free graph features and return evaluation metrics."""
    config = load_baseline_config(config_path)

    print("\n" + "=" * 70)
    print("  TRAINING BASELINE: XGBOOST + LABEL-FREE GRAPH FEATURES")
    print("=" * 70)

    df_raw = pd.read_parquet(config.data_path)
    df_raw.sort_values(by="timestamp", ascending=True, inplace=True, ignore_index=True)

    print("[1/4] Computing 27 label-free causal tabular features...")
    df_tab = compute_tabular_features(df_raw, include_label_features=False)
    tab_cols = get_feature_column_names(include_label_features=False)

    print("[2/4] Computing label-free structural graph features...")
    df_graph = compute_label_free_graph_features(df_raw)
    graph_cols = list(df_graph.columns)

    all_cols = tab_cols + graph_cols
    for col in graph_cols:
        df_tab[col] = df_graph[col]

    timestamps = pd.to_datetime(df_tab["timestamp"])
    t_min, t_max = timestamps.min(), timestamps.max()
    dur = t_max - t_min

    train_cutoff = t_min + dur * config.split.train_ratio
    val_cutoff = t_min + dur * (config.split.train_ratio + config.split.val_ratio)

    train_mask = timestamps <= train_cutoff
    val_mask = (timestamps > train_cutoff) & (timestamps <= val_cutoff)
    test_mask = timestamps > val_cutoff

    train_df = df_tab[train_mask]
    val_df = df_tab[val_mask]
    test_df = df_tab[test_mask]

    X_train = train_df[all_cols]
    y_train = train_df["is_fraud"]
    X_val = val_df[all_cols]
    y_val = val_df["is_fraud"]
    X_test = test_df[all_cols]
    y_test = test_df["is_fraud"]

    test_amounts = test_df["amount"].to_numpy()
    test_fraud_types = test_df["fraud_type"].to_numpy()
    val_amounts = val_df["amount"].to_numpy()
    val_fraud_types = val_df["fraud_type"].to_numpy()

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = float(n_neg / max(1, n_pos))

    print(f"[3/4] Fitting XGBoost on {len(X_train):,} samples (features: {len(all_cols)})...")
    clf = xgb.XGBClassifier(
        n_estimators=config.model.n_estimators,
        max_depth=config.model.max_depth,
        learning_rate=config.model.learning_rate,
        subsample=config.model.subsample,
        colsample_bytree=config.model.colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=config.model.early_stopping_rounds,
        eval_metric="aucpr",
        random_state=config.model.random_state,
        tree_method="hist",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

    print("[4/4] Evaluating validation threshold and test metrics...")
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_eval = compute_metrics(
        y_true=y_val.to_numpy(),
        y_pred_proba=val_probs,
        amounts=val_amounts,
        review_cost=config.cost.review_cost,
        default_fraud_amount=config.cost.default_fraud_amount,
        fraud_types=val_fraud_types,
    )
    val_threshold = val_eval.best_threshold

    t0_score = time.perf_counter()
    test_probs = clf.predict_proba(X_test)[:, 1]
    score_elapsed = time.perf_counter() - t0_score
    latency_ms = (score_elapsed / max(1, len(test_probs))) * 1000.0

    test_eval = compute_metrics(
        y_true=y_test.to_numpy(),
        y_pred_proba=test_probs,
        amounts=test_amounts,
        review_cost=config.cost.review_cost,
        default_fraud_amount=config.cost.default_fraud_amount,
        threshold_override=val_threshold,
        fraud_types=test_fraud_types,
    )

    print("\n" + "=" * 70)
    print("  XGBOOST + LABEL-FREE GRAPH FEATURES RESULTS")
    print("=" * 70)
    print(f"Val PR-AUC             : {val_eval.pr_auc:.4f}")
    print(f"Val Selected Threshold : {val_threshold:.4f}")
    print(f"Test PR-AUC            : {test_eval.pr_auc:.4f}")
    print(f"Test ROC-AUC           : {test_eval.roc_auc:.4f}")
    print(f"Test Recall @ 1% FPR   : {test_eval.recall_at_1pct_fpr:.4f}")
    print(f"Test F1 @ Val Thresh   : {test_eval.f1:.4f}")
    print(f"Test Business Cost     : ${test_eval.total_cost:,.2f}")
    print(f"Inference Latency      : {latency_ms:.6f} ms/sample")
    print("-" * 70)
    print("Per-Fraud-Type Test Recall:")
    for ft, rec in sorted(test_eval.per_type_recall.items()):
        print(f"  - {ft:<24s}: {rec:.4f}")
    print("=" * 70 + "\n")

    return {
        "model_name": "XGBoost + Label-Free Graph Features",
        "pr_auc": test_eval.pr_auc,
        "roc_auc": test_eval.roc_auc,
        "recall_1pct_fpr": test_eval.recall_at_1pct_fpr,
        "f1": test_eval.f1,
        "business_cost": test_eval.total_cost,
        "latency_ms": latency_ms,
        "threshold": val_threshold,
        "ato_recall": test_eval.per_type_recall.get("account_takeover", 0.0),
        "card_test_recall": test_eval.per_type_recall.get("card_testing", 0.0),
        "ring_recall": test_eval.per_type_recall.get("fraud_ring", 0.0),
        "collusion_recall": test_eval.per_type_recall.get("merchant_collusion", 0.0),
        "mule_recall": test_eval.per_type_recall.get("mule_account", 0.0),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost with label-free graph features.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()
    train_and_eval_label_free_xgb(args.config)
