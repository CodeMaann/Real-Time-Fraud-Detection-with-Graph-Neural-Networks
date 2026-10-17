"""Graph-signal sanity check: XGBoost on tabular + causal hand-made graph features.

Proves:
(a) The synthetic dataset contains strong, learnable graph signal.
(b) Provides an 'XGBoost + Graph Features' baseline for comparison against GNNs.

All graph features are strictly causal (closed='left'), utilizing only information
available prior to each transaction timestamp. All operating thresholds are selected
on the validation partition only and applied unchanged to test.
"""

from __future__ import annotations

import os
import sys
import time

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import numpy as np
import pandas as pd
import xgboost as xgb

from fraudgnn.config import load_baseline_config
from fraudgnn.data.features import compute_tabular_features, get_feature_column_names
from fraudgnn.training.evaluate import compute_metrics


def compute_causal_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute hand-made graph features using strictly past transaction data.

    Features engineered per transaction:
    1. device_distinct_users: Number of distinct past users seen on the device.
    2. ip_distinct_users: Number of distinct past users seen on the IP.
    3. device_user_fraud_frac: Fraction of distinct past users on device with known fraud.
    4. ip_user_fraud_frac: Fraction of distinct past users on IP with known fraud.
    5. two_hop_neighbor_fraud_rate: Transaction fraud rate among 2-hop neighbor users
       (accounts sharing device or IP with the current user).
    6. two_hop_fraud_user_frac: Fraction of 2-hop neighbor users with known past fraud.
    7. merchant_degree: Number of distinct past users who transacted at the merchant.
    8. device_past_fraud_txs: Count of past fraud transactions on this device.
    9. ip_past_fraud_txs: Count of past fraud transactions on this IP.

    Args:
        df: Chronologically sorted transaction DataFrame.

    Returns:
        DataFrame containing only the newly engineered graph feature columns.
    """
    n = len(df)
    users = df["user_id"].to_numpy()
    devices = df["device_id"].to_numpy()
    ips = df["ip_address"].to_numpy()
    merchants = df["merchant_id"].to_numpy()
    labels = df["is_fraud"].to_numpy(dtype=np.int64)

    # Pre-allocate output arrays
    device_distinct_users = np.zeros(n, dtype=np.float32)
    ip_distinct_users = np.zeros(n, dtype=np.float32)
    device_user_fraud_frac = np.zeros(n, dtype=np.float32)
    ip_user_fraud_frac = np.zeros(n, dtype=np.float32)
    two_hop_neighbor_fraud_rate = np.zeros(n, dtype=np.float32)
    two_hop_fraud_user_frac = np.zeros(n, dtype=np.float32)
    merchant_degree = np.zeros(n, dtype=np.float32)
    device_past_fraud_txs = np.zeros(n, dtype=np.float32)
    ip_past_fraud_txs = np.zeros(n, dtype=np.float32)

    # State registries for causal streaming computation
    device_to_users: dict[str, set[str]] = {}
    ip_to_users: dict[str, set[str]] = {}
    merchant_to_users: dict[str, set[str]] = {}
    user_fraud_tx_count: dict[str, int] = {}
    user_total_tx_count: dict[str, int] = {}
    device_fraud_count: dict[str, int] = {}
    ip_fraud_count: dict[str, int] = {}

    empty_set: set[str] = set()

    for i in range(n):
        u = str(users[i])
        d = str(devices[i])
        p = str(ips[i])
        m = str(merchants[i])

        # 1. Causal Reads (Strictly past info, excluding current transaction)
        dev_users = device_to_users.get(d, empty_set)
        ip_users = ip_to_users.get(p, empty_set)
        merch_users = merchant_to_users.get(m, empty_set)

        n_dev_u = len(dev_users)
        n_ip_u = len(ip_users)
        device_distinct_users[i] = n_dev_u
        ip_distinct_users[i] = n_ip_u
        merchant_degree[i] = len(merch_users)
        device_past_fraud_txs[i] = device_fraud_count.get(d, 0)
        ip_past_fraud_txs[i] = ip_fraud_count.get(p, 0)

        # Fraction of users on this device/IP with past fraud
        if n_dev_u > 0:
            dev_fraud_u = sum(1 for usr in dev_users if user_fraud_tx_count.get(usr, 0) > 0)
            device_user_fraud_frac[i] = dev_fraud_u / n_dev_u

        if n_ip_u > 0:
            ip_fraud_u = sum(1 for usr in ip_users if user_fraud_tx_count.get(usr, 0) > 0)
            ip_user_fraud_frac[i] = ip_fraud_u / n_ip_u

        # 2-hop neighbor users: union of accounts connected via this device or IP
        two_hop_users = (dev_users | ip_users) - {u}
        if two_hop_users:
            tot_tx = sum(user_total_tx_count.get(usr, 0) for usr in two_hop_users)
            fr_tx = sum(user_fraud_tx_count.get(usr, 0) for usr in two_hop_users)
            if tot_tx > 0:
                two_hop_neighbor_fraud_rate[i] = fr_tx / tot_tx

            fraud_users_count = sum(
                1 for usr in two_hop_users if user_fraud_tx_count.get(usr, 0) > 0
            )
            two_hop_fraud_user_frac[i] = fraud_users_count / len(two_hop_users)

        # 2. State Updates (Occur after feature generation for this transaction)
        if d not in device_to_users:
            device_to_users[d] = set()
        device_to_users[d].add(u)

        if p not in ip_to_users:
            ip_to_users[p] = set()
        ip_to_users[p].add(u)

        if m not in merchant_to_users:
            merchant_to_users[m] = set()
        merchant_to_users[m].add(u)

        user_total_tx_count[u] = user_total_tx_count.get(u, 0) + 1
        if labels[i] == 1:
            user_fraud_tx_count[u] = user_fraud_tx_count.get(u, 0) + 1
            device_fraud_count[d] = device_fraud_count.get(d, 0) + 1
            ip_fraud_count[p] = ip_fraud_count.get(p, 0) + 1

    return pd.DataFrame(
        {
            "device_distinct_users": device_distinct_users,
            "ip_distinct_users": ip_distinct_users,
            "device_user_fraud_frac": device_user_fraud_frac,
            "ip_user_fraud_frac": ip_user_fraud_frac,
            "two_hop_neighbor_fraud_rate": two_hop_neighbor_fraud_rate,
            "two_hop_fraud_user_frac": two_hop_fraud_user_frac,
            "merchant_degree": merchant_degree,
            "device_past_fraud_txs": device_past_fraud_txs,
            "ip_past_fraud_txs": ip_past_fraud_txs,
        }
    )


def main() -> int:
    """Run graph-signal check and print performance comparison."""
    config = load_baseline_config("configs/baseline.yaml")

    print("\n" + "=" * 70)
    print("  GRAPH SIGNAL SANITY CHECK: TABULAR + CAUSAL GRAPH FEATURES")
    print("=" * 70)

    print(f"[1/5] Loading transactions from: {config.data_path}")
    df_raw = pd.read_parquet(config.data_path)
    df_raw.sort_values(by="timestamp", ascending=True, inplace=True, ignore_index=True)
    total_tx = len(df_raw)
    n_fraud = int(df_raw["is_fraud"].sum())
    print(
        f"      Total: {total_tx:,} transactions ({n_fraud:,} fraud, {n_fraud / total_tx * 100:.2f}%)"
    )

    print("[2/5] Computing standard tabular features...")
    t0_tab = time.perf_counter()
    df_feat = compute_tabular_features(df_raw, smoothing_alpha=config.smoothing_alpha)
    tab_cols = get_feature_column_names()
    print(
        f"      Engineered {len(tab_cols)} tabular features in {time.perf_counter() - t0_tab:.2f}s"
    )

    print("[3/5] Computing strictly causal hand-made graph features...")
    t0_graph = time.perf_counter()
    df_graph = compute_causal_graph_features(df_raw)
    graph_cols = list(df_graph.columns)
    print(
        f"      Engineered {len(graph_cols)} graph features in {time.perf_counter() - t0_graph:.2f}s"
    )

    # Combine tabular + graph features
    all_feature_cols = tab_cols + graph_cols
    for col in graph_cols:
        df_feat[col] = df_graph[col]

    # Strict temporal train / val / test split
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

    print(
        f"[4/5] Temporal Split:\n"
        f"      Train: {len(train_df):,} tx ({train_df['is_fraud'].sum():,} fraud) -> <= {train_cutoff}\n"
        f"      Val  : {len(val_df):,} tx ({val_df['is_fraud'].sum():,} fraud) -> {train_cutoff} to {val_cutoff}\n"
        f"      Test : {len(test_df):,} tx ({test_df['is_fraud'].sum():,} fraud) -> > {val_cutoff}"
    )

    X_train = train_df[all_feature_cols]
    y_train = train_df["is_fraud"]
    X_val = val_df[all_feature_cols]
    y_val = val_df["is_fraud"]
    X_test = test_df[all_feature_cols]
    y_test = test_df["is_fraud"]

    # Fit XGBoost with early stopping on validation PR-AUC
    y_tr_arr = np.asarray(y_train, dtype=np.int64)
    spw = float((y_tr_arr == 0).sum() / max(1, (y_tr_arr == 1).sum()))

    cfg_m = config.model
    model = xgb.XGBClassifier(
        n_estimators=cfg_m.n_estimators,
        max_depth=cfg_m.max_depth,
        learning_rate=cfg_m.learning_rate,
        subsample=cfg_m.subsample,
        colsample_bytree=cfg_m.colsample_bytree,
        scale_pos_weight=spw,
        early_stopping_rounds=cfg_m.early_stopping_rounds,
        eval_metric="aucpr",
        random_state=cfg_m.random_state,
        tree_method="hist",
        n_jobs=-1,
    )

    print(
        f"[5/5] Training XGBoost (Tabular + Graph, {len(all_feature_cols)} features, spw={spw:.2f})..."
    )
    t0_fit = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"      Fitted in {time.perf_counter() - t0_fit:.2f}s")

    # Evaluate validation set to determine operating threshold
    y_val_prob = model.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(
        y_true=y_val.to_numpy(),
        y_pred_proba=y_val_prob,
        amounts=val_df["amount"].to_numpy(),
        review_cost=config.cost.review_cost,
        default_fraud_amount=config.cost.default_fraud_amount,
        fraud_types=val_df["fraud_type"].to_numpy(),
    )

    # Evaluate test set strictly using the validation threshold
    val_thresh = val_metrics.best_threshold
    y_test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(
        y_true=y_test.to_numpy(),
        y_pred_proba=y_test_prob,
        amounts=test_df["amount"].to_numpy(),
        review_cost=config.cost.review_cost,
        default_fraud_amount=config.cost.default_fraud_amount,
        threshold_override=val_thresh,
        fraud_types=test_df["fraud_type"].to_numpy(),
    )

    # Top Feature Importance (Gain)
    raw_scores = model.get_booster().get_score(importance_type="gain")
    importance: dict[str, float] = {
        k: float(v if isinstance(v, (int, float)) else v[0]) for k, v in raw_scores.items()
    }
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 70)
    print("  TOP-15 FEATURE IMPORTANCE (GAIN) - TABULAR + GRAPH FEATURES")
    print("=" * 70)
    for rank, (feat, gain) in enumerate(sorted_imp[:15], 1):
        is_graph_feat = " [GRAPH]" if feat in graph_cols else " [TABULAR]"
        print(f"  {rank:2d}. {feat:<32s}{is_graph_feat:<10s} gain = {gain:,.2f}")

    print("\n" + "=" * 70)
    print("  PERFORMANCE COMPARISON: TABULAR BASELINE vs. TABULAR + GRAPH FEATURES")
    print("=" * 70)
    print(f"Validation PR-AUC            : {val_metrics.pr_auc:.4f}")
    print(f"Validation Recall @ 1% FPR   : {val_metrics.recall_at_1pct_fpr:.4f}")
    print(f"Validation Best Threshold    : {val_metrics.best_threshold:.4f}")
    print("-" * 70)
    print(f"TEST PR-AUC (Primary)        : {test_metrics.pr_auc:.4f}")
    print(f"TEST ROC-AUC                 : {test_metrics.roc_auc:.4f}")
    print(f"TEST Recall @ 1% FPR         : {test_metrics.recall_at_1pct_fpr:.4f}")
    print(f"TEST F1 @ Val Threshold      : {test_metrics.f1:.4f}")
    print(
        f"TEST Confusion Matrix        : TN={test_metrics.tn:,}, FP={test_metrics.fp:,}, FN={test_metrics.fn:,}, TP={test_metrics.tp:,}"
    )
    print(f"TEST Business Cost           : ${test_metrics.total_cost:,.2f}")

    print("\n" + "-" * 70)
    print(f"  PER-FRAUD-TYPE RECALL @ Val Threshold = {val_thresh:.4f}")
    print("-" * 70)
    print(f"  {'Type':<25s} {'Val Recall':>12s}  {'Test Recall':>12s}")
    all_types = sorted(
        set(list(val_metrics.per_type_recall.keys()) + list(test_metrics.per_type_recall.keys()))
    )
    for ft in all_types:
        v_r = val_metrics.per_type_recall.get(ft, float("nan"))
        t_r = test_metrics.per_type_recall.get(ft, float("nan"))
        print(f"  {ft:<25s} {v_r:>12.4f}  {t_r:>12.4f}")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
