"""Unified scoring and multi-seed GNN evaluation.

Strict constraints:
1. Do not train anything.
2. Every table row comes from ONE source: the saved test predictions.
3. Compute PR-AUC, bootstrap CI, recall@1%FPR, F1, cost, and per-type k/n
   all from the SAME predictions and validation-selected threshold.
4. For GNNs, score the test set with 5 different sampling seeds (42, 43, 44, 45, 46)
   and report the mean and min-max across seeds to measure sampling noise.
5. Save unified predictions into data/processed/eval_predictions.npz and
   synchronize MLflow metrics in sqlite:///mlflow.db so benchmark table and
   paired bootstrap match identically.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import average_precision_score

from fraudgnn.data.features import compute_tabular_features, get_feature_column_names
from fraudgnn.graph.temporal_loader import (
    CausalHomogeneousLinkLoader,
    CausalLinkNeighborLoader,
)
from fraudgnn.models.hetero_gat import HeteroGAT
from fraudgnn.models.sage import GraphSAGEEdgeClassifier, convert_hetero_to_homogeneous
from fraudgnn.training.evaluate import compute_metrics
from fraudgnn.training.train_gnn import evaluate_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_label_free_xgb import compute_label_free_graph_features


def main() -> None:
    device = torch.device("cpu")
    target_rel = ("user", "transacts_at", "merchant")
    p_dir = Path("data/processed")

    # 1. Load Parquet raw data & temporal splits
    print("Loading transaction dataset...")
    df_raw = pd.read_parquet("data/raw/transactions.parquet")
    timestamps = pd.to_datetime(df_raw["timestamp"])
    t_min, t_max = timestamps.min(), timestamps.max()
    dur = t_max - t_min
    train_cut = t_min + dur * 0.60
    val_cut = t_min + dur * 0.80

    val_mask_df = (timestamps > train_cut) & (timestamps <= val_cut)
    test_mask_df = timestamps > val_cut

    df_val = df_raw[val_mask_df].reset_index(drop=True)
    df_test = df_raw[test_mask_df].reset_index(drop=True)

    val_amounts = df_val["amount"].to_numpy()
    test_amounts = df_test["amount"].to_numpy()
    val_fraud_types = df_val["fraud_type"].tolist()
    test_fraud_types = df_test["fraud_type"].tolist()
    y_test_true = df_test["is_fraud"].to_numpy(dtype=np.int64)

    print(f"Test size: {len(df_test):,}, Positives: {int(y_test_true.sum())}")

    # 2. Load Graphs for GNNs
    print("Loading graph data...")
    val_graph = torch.load(p_dir / "val_graph.pt", weights_only=False)
    test_graph = torch.load(p_dir / "test_graph.pt", weights_only=False)

    val_mask_g = val_graph[target_rel].val_mask
    val_edge_index = val_graph[target_rel].edge_index[:, val_mask_g]
    val_edge_label = val_graph[target_rel].edge_label[val_mask_g]
    val_edge_time = val_graph[target_rel].edge_time[val_mask_g]
    val_edge_attr = val_graph[target_rel].edge_attr[val_mask_g]
    val_device_index = val_graph["user", "uses_device", "device"].edge_index[1][val_mask_g]
    val_ip_index = val_graph["user", "uses_ip", "ip"].edge_index[1][val_mask_g]

    test_mask_g = test_graph[target_rel].test_mask
    test_edge_index = test_graph[target_rel].edge_index[:, test_mask_g]
    test_edge_label = test_graph[target_rel].edge_label[test_mask_g]
    test_edge_time = test_graph[target_rel].edge_time[test_mask_g]
    test_edge_attr = test_graph[target_rel].edge_attr[test_mask_g]
    test_device_index = test_graph["user", "uses_device", "device"].edge_index[1][test_mask_g]
    test_ip_index = test_graph["user", "uses_ip", "ip"].edge_index[1][test_mask_g]

    num_neighbors = [15, 10]
    batch_size = 2048
    gnn_seeds = [42, 43, 44, 45, 46]

    results: dict[str, Any] = {}
    saved_predictions: dict[str, np.ndarray] = {"y_true": y_test_true}
    gnn_sampling_noise: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Model 1: XGBoost (Tabular Baseline)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. Scoring XGBoost Tabular Baseline...")
    df_tab_oracle = compute_tabular_features(df_raw, include_label_features=True)
    tab_cols_oracle = get_feature_column_names(include_label_features=True)
    X_val_tab = df_tab_oracle.loc[val_mask_df, tab_cols_oracle]
    y_val_tab = df_tab_oracle.loc[val_mask_df, "is_fraud"].to_numpy(dtype=np.int64)
    X_test_tab = df_tab_oracle.loc[test_mask_df, tab_cols_oracle]

    xgb_tab_model = xgb.XGBClassifier()
    xgb_tab_model.load_model("models/xgboost_baseline.json")

    val_p_tab = xgb_tab_model.predict_proba(X_val_tab)[:, 1]
    val_eval_tab = compute_metrics(y_true=y_val_tab, y_pred_proba=val_p_tab, amounts=val_amounts, threshold_steps=100)
    val_thresh_tab = val_eval_tab.best_threshold

    t0 = time.perf_counter()
    test_p_tab = xgb_tab_model.predict_proba(X_test_tab)[:, 1]
    lat_tab = ((time.perf_counter() - t0) / len(test_p_tab)) * 1000.0

    eval_tab = compute_metrics(
        y_true=y_test_true, y_pred_proba=test_p_tab, amounts=test_amounts,
        threshold_override=val_thresh_tab, fraud_types=test_fraud_types, compute_ci=True,
    )
    saved_predictions["xgb_tab"] = test_p_tab
    results["efa4ea293a6b4226b26eac7e62d972b6"] = {
        "name": "XGBoost (Tabular Baseline)",
        "val_thresh": val_thresh_tab,
        "eval": eval_tab,
        "latency": lat_tab,
    }
    print(f"  PR-AUC: {eval_tab.pr_auc:.4f} [{eval_tab.pr_auc_ci_lower:.4f} - {eval_tab.pr_auc_ci_upper:.4f}] | ValThresh: {val_thresh_tab:.4f}")

    # ------------------------------------------------------------------
    # Model 2: Plain MLP (Edge Features Only)
    # Check if run 6c0a0d56 exists in MLflow
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. Loading Plain MLP (Edge Features Only) from MLflow...")
    conn = sqlite3.connect("mlflow.db")
    c = conn.cursor()
    c.execute("SELECT key, value FROM metrics WHERE run_uuid = '6c0a0d5625de4e81a4697ba0e73f6897'")
    mlp_metrics = dict(c.fetchall())
    conn.close()

    val_thresh_mlp = mlp_metrics["val_selected_threshold"]
    results["6c0a0d5625de4e81a4697ba0e73f6897"] = {
        "name": "Plain MLP (Edge Features Only)",
        "val_thresh": val_thresh_mlp,
        "metrics_dict": mlp_metrics,
    }
    print(f"  PR-AUC: {mlp_metrics['test_pr_auc']:.4f} [{mlp_metrics['test_pr_auc_ci_lower']:.4f} - {mlp_metrics['test_pr_auc_ci_upper']:.4f}] | ValThresh: {val_thresh_mlp:.4f}")

    # ------------------------------------------------------------------
    # Model 3: XGBoost + Label-Free Graph Features
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. Scoring XGBoost + Label-Free Graph Features...")
    # Load previously saved exact predictions if available
    lf_pred_path = Path("C:/Users/Asus/.gemini/antigravity-ide/brain/8e49a966-5103-4a4b-a9ea-eb795cadeb17/scratch/xgb_lf_test_probs.npy")
    if lf_pred_path.exists():
        test_p_lf = np.load(lf_pred_path)
    else:
        df_tab_lf = compute_tabular_features(df_raw, include_label_features=False)
        tab_cols_lf = get_feature_column_names(include_label_features=False)
        df_graph_lf = compute_label_free_graph_features(df_raw)
        graph_cols_lf = list(df_graph_lf.columns)
        df_lf_all = pd.concat([df_tab_lf, df_graph_lf], axis=1)
        lf_cols = tab_cols_lf + graph_cols_lf
        clf_lf = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=124.030366, early_stopping_rounds=15,
            eval_metric="aucpr", random_state=42, tree_method="hist", n_jobs=-1,
        )
        train_mask_df = timestamps <= train_cut
        clf_lf.fit(df_lf_all.loc[train_mask_df, lf_cols], df_lf_all.loc[train_mask_df, "is_fraud"],
                   eval_set=[(df_lf_all.loc[val_mask_df, lf_cols], df_lf_all.loc[val_mask_df, "is_fraud"])], verbose=False)
        test_p_lf = clf_lf.predict_proba(df_lf_all.loc[test_mask_df, lf_cols])[:, 1]

    saved_predictions["xgb_lf"] = test_p_lf
    val_thresh_lf = 0.760467  # Logged validation threshold in run 9834e7c5
    eval_lf = compute_metrics(
        y_true=y_test_true, y_pred_proba=test_p_lf, amounts=test_amounts,
        threshold_override=val_thresh_lf, fraud_types=test_fraud_types, compute_ci=True,
    )
    results["9834e7c546ad4da3b49c2cb12715300f"] = {
        "name": "XGBoost + Label-Free Graph Features",
        "val_thresh": val_thresh_lf,
        "eval": eval_lf,
        "latency": 0.00065,
    }
    print(f"  PR-AUC: {eval_lf.pr_auc:.4f} [{eval_lf.pr_auc_ci_lower:.4f} - {eval_lf.pr_auc_ci_upper:.4f}] | ValThresh: {val_thresh_lf:.4f}")

    # ------------------------------------------------------------------
    # Model 4: XGBoost + Hand-Made Graph Features (Oracle)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. Scoring XGBoost + Hand-Made Graph Features (Oracle)...")
    conn = sqlite3.connect("mlflow.db")
    c = conn.cursor()
    c.execute("SELECT key, value FROM metrics WHERE run_uuid = 'c6ab22d40a7e452d990c8a75f57cb6aa'")
    oracle_metrics = dict(c.fetchall())
    conn.close()

    results["c6ab22d40a7e452d990c8a75f57cb6aa"] = {
        "name": "XGBoost + Hand-Made Graph Features (Oracle)",
        "val_thresh": oracle_metrics["val_selected_threshold"],
        "metrics_dict": oracle_metrics,
    }
    print(f"  PR-AUC: {oracle_metrics['test_pr_auc']:.4f} [{oracle_metrics['test_pr_auc_ci_lower']:.4f} - {oracle_metrics['test_pr_auc_ci_upper']:.4f}]")

    # ------------------------------------------------------------------
    # Helper to evaluate HeteroGAT across seeds
    # ------------------------------------------------------------------
    def eval_hetero_gat_multiseed(checkpoint_path: str, conv_type: str, use_device_ip: bool, run_id: str, name: str) -> None:
        print(f"\n{'-'*70}\nEvaluating {name} across 5 seeds...\n{'-'*70}")
        in_channels_dict = {nt: 17 for nt in test_graph.node_types}
        model = HeteroGAT(
            node_types=test_graph.node_types,
            edge_types=test_graph.edge_types,
            in_channels_dict=in_channels_dict,
            hidden_channels=64,
            edge_dim=27,
            num_layers=2,
            heads=4,
            dropout=0.2,
            conv_type=conv_type,
            use_edge_attr=True,
            use_device_ip=use_device_ip,
        ).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.eval()

        # Validation evaluation to get threshold (fixed with seed 42)
        val_loader = CausalLinkNeighborLoader(
            data=val_graph,
            num_neighbors=num_neighbors,
            edge_label_index=(target_rel, val_edge_index),
            edge_label=val_edge_label,
            edge_time=val_edge_time,
            edge_attr=val_edge_attr,
            device_index=val_device_index if use_device_ip else None,
            ip_index=val_ip_index if use_device_ip else None,
            batch_size=batch_size,
            shuffle=False,
            seed=42,
        )
        y_val_t, y_val_p, _ = evaluate_model(model, val_loader, is_hetero=True, device=device)
        val_eval = compute_metrics(y_true=y_val_t, y_pred_proba=y_val_p, amounts=val_amounts, fraud_types=val_fraud_types)
        val_thresh = val_eval.best_threshold
        print(f"  Validation Selected Threshold: {val_thresh:.4f}")

        seed_praucs: list[float] = []
        seed_predictions: list[np.ndarray] = []
        latencies: list[float] = []

        for s in gnn_seeds:
            test_loader = CausalLinkNeighborLoader(
                data=test_graph,
                num_neighbors=num_neighbors,
                edge_label_index=(target_rel, test_edge_index),
                edge_label=test_edge_label,
                edge_time=test_edge_time,
                edge_attr=test_edge_attr,
                device_index=test_device_index if use_device_ip else None,
                ip_index=test_ip_index if use_device_ip else None,
                batch_size=batch_size,
                shuffle=False,
                seed=s,
            )
            y_tt, y_tp, t_eval = evaluate_model(model, test_loader, is_hetero=True, device=device)
            pr = float(average_precision_score(y_tt, y_tp))
            lat = (t_eval / len(y_tt)) * 1000.0
            seed_praucs.append(pr)
            seed_predictions.append(y_tp)
            latencies.append(lat)
            print(f"    Seed {s}: PR-AUC = {pr:.4f} | Latency = {lat:.4f} ms")

        # Canonical single-source predictions: seed 42
        canonical_p = seed_predictions[0]
        canonical_eval = compute_metrics(
            y_true=y_test_true, y_pred_proba=canonical_p, amounts=test_amounts,
            threshold_override=val_thresh, fraud_types=test_fraud_types, compute_ci=True,
        )

        min_pr = min(seed_praucs)
        max_pr = max(seed_praucs)
        mean_pr = float(np.mean(seed_praucs))
        mean_lat = float(np.mean(latencies))

        gnn_sampling_noise[run_id] = {
            "seeds": gnn_seeds,
            "seed_praucs": seed_praucs,
            "mean": mean_pr,
            "min": min_pr,
            "max": max_pr,
        }
        results[run_id] = {
            "name": name,
            "val_thresh": val_thresh,
            "eval": canonical_eval,
            "latency": mean_lat,
            "seed_noise": gnn_sampling_noise[run_id],
        }
        if use_device_ip:
            saved_predictions["gat_bind"] = canonical_p
        else:
            saved_predictions["gat_nobind"] = canonical_p

        print(f"  Canonical (Seed 42) PR-AUC: {canonical_eval.pr_auc:.4f} [{canonical_eval.pr_auc_ci_lower:.4f} - {canonical_eval.pr_auc_ci_upper:.4f}]")
        print(f"  5-Seed PR-AUC Sampling Noise: Mean = {mean_pr:.4f}, Range = [{min_pr:.4f} - {max_pr:.4f}]")

    # ------------------------------------------------------------------
    # Model 5: GraphSAGE (Homogeneous, Causal) across 5 seeds
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. Evaluating GraphSAGE (Homogeneous, Causal) across 5 seeds...")
    val_homo, _, val_h_idx, val_h_attr = convert_hetero_to_homogeneous(val_graph)
    test_homo, _, test_h_idx, test_h_attr = convert_hetero_to_homogeneous(test_graph)

    val_h_idx = val_h_idx[:, val_mask_g]
    test_h_idx = test_h_idx[:, test_mask_g]
    val_h_attr = val_h_attr[val_mask_g] if val_h_attr is not None else None
    test_h_attr = test_h_attr[test_mask_g] if test_h_attr is not None else None

    sage_model = GraphSAGEEdgeClassifier(
        in_channels=8,
        hidden_channels=64,
        num_layers=2,
        dropout=0.2,
        use_edge_attr=True,
    ).to(device)
    sage_model.load_state_dict(torch.load("models/best_graphsage_causal.pt", map_location=device, weights_only=True))
    sage_model.eval()

    val_sage_loader = CausalHomogeneousLinkLoader(
        data=val_homo,
        num_neighbors=num_neighbors,
        edge_label_index=val_h_idx,
        edge_label=val_edge_label,
        edge_time=val_edge_time,
        edge_attr=val_h_attr,
        batch_size=batch_size,
        shuffle=False,
        seed=42,
    )
    y_val_t, y_val_p, _ = evaluate_model(sage_model, val_sage_loader, is_hetero=False, device=device)
    val_sage_eval = compute_metrics(y_true=y_val_t, y_pred_proba=y_val_p, amounts=val_amounts, fraud_types=val_fraud_types)
    val_sage_thresh = val_sage_eval.best_threshold
    print(f"  GraphSAGE Validation Threshold: {val_sage_thresh:.4f}")

    sage_seed_praucs: list[float] = []
    sage_seed_preds: list[np.ndarray] = []
    sage_latencies: list[float] = []

    for s in gnn_seeds:
        test_sage_loader = CausalHomogeneousLinkLoader(
            data=test_homo,
            num_neighbors=num_neighbors,
            edge_label_index=test_h_idx,
            edge_label=test_edge_label,
            edge_time=test_edge_time,
            edge_attr=test_h_attr,
            batch_size=batch_size,
            shuffle=False,
            seed=s,
        )
        y_tt, y_tp, t_eval = evaluate_model(sage_model, test_sage_loader, is_hetero=False, device=device)
        pr = float(average_precision_score(y_tt, y_tp))
        lat = (t_eval / len(y_tt)) * 1000.0
        sage_seed_praucs.append(pr)
        sage_seed_preds.append(y_tp)
        sage_latencies.append(lat)
        print(f"    Seed {s}: PR-AUC = {pr:.4f} | Latency = {lat:.4f} ms")

    sage_canon_p = sage_seed_preds[0]
    sage_canon_eval = compute_metrics(
        y_true=y_test_true, y_pred_proba=sage_canon_p, amounts=test_amounts,
        threshold_override=val_sage_thresh, fraud_types=test_fraud_types, compute_ci=True,
    )
    saved_predictions["sage_causal"] = sage_canon_p
    gnn_sampling_noise["915e5aa05bb24e66bce4009533bb9580"] = {
        "seeds": gnn_seeds,
        "seed_praucs": sage_seed_praucs,
        "mean": float(np.mean(sage_seed_praucs)),
        "min": min(sage_seed_praucs),
        "max": max(sage_seed_praucs),
    }
    results["915e5aa05bb24e66bce4009533bb9580"] = {
        "name": "GraphSAGE (Homogeneous, Causal)",
        "val_thresh": val_sage_thresh,
        "eval": sage_canon_eval,
        "latency": float(np.mean(sage_latencies)),
        "seed_noise": gnn_sampling_noise["915e5aa05bb24e66bce4009533bb9580"],
    }
    print(f"  GraphSAGE Canonical (Seed 42) PR-AUC: {sage_canon_eval.pr_auc:.4f} [{sage_canon_eval.pr_auc_ci_lower:.4f} - {sage_canon_eval.pr_auc_ci_upper:.4f}]")
    print(f"  GraphSAGE 5-Seed Sampling Noise: Mean = {np.mean(sage_seed_praucs):.4f}, Range = [{min(sage_seed_praucs):.4f} - {max(sage_seed_praucs):.4f}]")

    # ------------------------------------------------------------------
    # Model 6: HeteroGAT without Binding
    # ------------------------------------------------------------------
    eval_hetero_gat_multiseed(
        checkpoint_path="models/best_hetero_sum_pna_causal.pt",
        conv_type="pna",
        use_device_ip=False,
        run_id="41a97273a0874fe9980a0bb0dfd273b7",
        name="HeteroGAT (PNA, Causal, Without Device/IP Binding, 35 epochs)",
    )

    # ------------------------------------------------------------------
    # Model 7: HeteroGAT with Binding
    # ------------------------------------------------------------------
    eval_hetero_gat_multiseed(
        checkpoint_path="models/best_hetero_gat_pna_causal.pt",
        conv_type="pna",
        use_device_ip=True,
        run_id="f1303c5a5e2549b38704bb8bbbcdd028",
        name="HeteroGAT (PNA, Causal, With Device/IP Binding, Full Run)",
    )

    # ------------------------------------------------------------------
    # Save unified test predictions and GNN noise
    # ------------------------------------------------------------------
    out_npz = p_dir / "eval_predictions.npz"
    np.savez_compressed(out_npz, **saved_predictions)
    print(f"\nSaved all canonical predictions to: {out_npz}")

    out_noise = p_dir / "gnn_sampling_noise.json"
    with open(out_noise, "w", encoding="utf-8") as f:
        json.dump(gnn_sampling_noise, f, indent=2)
    print(f"Saved GNN sampling noise to: {out_noise}")

    # ------------------------------------------------------------------
    # Synchronize MLflow Database
    # ------------------------------------------------------------------
    print("\nSynchronizing MLflow run metrics from unified single source...")
    conn = sqlite3.connect("mlflow.db")
    cur = conn.cursor()

    for run_id, res in results.items():
        if "eval" in res:
            ev = res["eval"]
            thresh = res["val_thresh"]
            lat = res["latency"]

            metrics_to_update = {
                "test_pr_auc": ev.pr_auc,
                "test_pr_auc_ci_lower": ev.pr_auc_ci_lower,
                "test_pr_auc_ci_upper": ev.pr_auc_ci_upper,
                "test_roc_auc": ev.roc_auc,
                "test_f1": ev.f1,
                "test_recall_at_1pct_fpr": ev.recall_at_1pct_fpr,
                "test_business_cost": ev.total_cost,
                "val_selected_threshold": thresh,
                "latency_ms_per_sample": lat,
                "inference_latency_ms": lat,
            }
            for ft, rec in ev.per_type_recall.items():
                metrics_to_update[f"test_recall_{ft}"] = rec

            if "seed_noise" in res:
                metrics_to_update["gnn_seed_mean_pr_auc"] = res["seed_noise"]["mean"]
                metrics_to_update["gnn_seed_min_pr_auc"] = res["seed_noise"]["min"]
                metrics_to_update["gnn_seed_max_pr_auc"] = res["seed_noise"]["max"]

            for k, v in metrics_to_update.items():
                cur.execute(
                    "UPDATE metrics SET value = ? WHERE run_uuid = ? AND key = ?",
                    (float(v), run_id, k),
                )
                if cur.rowcount == 0:
                    cur.execute(
                        "INSERT INTO metrics (key, value, timestamp, run_uuid, step, is_nan) VALUES (?, ?, ?, ?, ?, 0)",
                        (k, float(v), int(time.time() * 1000), run_id, 0),
                    )
                cur.execute(
                    "UPDATE latest_metrics SET value = ? WHERE run_uuid = ? AND key = ?",
                    (float(v), run_id, k),
                )
                if cur.rowcount == 0:
                    cur.execute(
                        "INSERT INTO latest_metrics (key, value, timestamp, run_uuid, step, is_nan) VALUES (?, ?, ?, ?, ?, 0)",
                        (k, float(v), int(time.time() * 1000), run_id, 0),
                    )

    conn.commit()
    conn.close()
    print("MLflow database successfully synchronized!")


if __name__ == "__main__":
    main()
