"""Validation PR-AUC with bootstrap CIs for all models on the same validation rows.

Only the validation split is scored. Results are logged through the MLflow API (no SQL) to
runs that exist in the local store, under ``val_pr_auc_rescored*`` keys, and written to JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np
import pandas as pd
import torch
import yaml
from mlflow.entities import Metric, RunTag
from mlflow.tracking import MlflowClient

from fraudgnn.config import load_baseline_config, load_gnn_config
from fraudgnn.data.features import compute_tabular_features, get_feature_column_names
from fraudgnn.graph.temporal_loader import CausalLinkNeighborLoader
from fraudgnn.models.hetero_gat import HeteroGAT
from fraudgnn.training.evaluate import compute_metrics
from fraudgnn.training.train_gnn import evaluate_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_unified import TARGET_REL, sha256_file, split_targets, temporal_split
from scripts.train_label_free_xgb import compute_label_free_graph_features


def build_hetero_gat(entry: dict[str, Any], graph: Any) -> tuple[HeteroGAT, Any]:
    """Model from its training config with the checkpoint loaded, in eval mode."""
    gnn_cfg = load_gnn_config(entry["gnn_config"])
    model = HeteroGAT(
        node_types=graph.node_types,
        edge_types=graph.edge_types,
        in_channels_dict={nt: 17 for nt in graph.node_types},
        hidden_channels=gnn_cfg.model.hidden_dim,
        edge_dim=gnn_cfg.model.edge_dim,
        num_layers=gnn_cfg.model.num_layers,
        heads=gnn_cfg.model.heads,
        dropout=gnn_cfg.model.dropout,
        conv_type=entry["conv_type"],
        use_edge_attr=gnn_cfg.model.use_edge_attr,
        use_device_ip=bool(entry["use_device_ip"]),
    )
    model.load_state_dict(torch.load(entry["checkpoint"], map_location="cpu", weights_only=True))
    model.eval()
    return model, gnn_cfg


def score_val_gnn(entry: dict[str, Any], val_graph: Any) -> np.ndarray:
    """Validation probabilities of one HeteroGAT checkpoint (training loader settings)."""
    model, gnn_cfg = build_hetero_gat(entry, val_graph)
    tgt = split_targets(val_graph, "val_mask")
    use_dev = bool(entry["use_device_ip"])
    loader = CausalLinkNeighborLoader(
        data=val_graph,
        num_neighbors=gnn_cfg.training.num_neighbors[: gnn_cfg.model.num_layers],
        edge_label_index=(TARGET_REL, tgt["edge_index"]),
        edge_label=tgt["edge_label"],
        edge_time=tgt["edge_time"],
        edge_attr=tgt["edge_attr"],
        device_index=tgt["device_index"] if use_dev else None,
        ip_index=tgt["ip_index"] if use_dev else None,
        batch_size=gnn_cfg.training.batch_size,
        shuffle=False,
        isolate_targets=bool(entry["isolate_targets"]),
    )
    _, proba, _ = evaluate_model(model, loader, is_hetero=True)
    return proba


def main(argv: list[str] | None = None) -> int:
    """Score validation for every configured model and print one table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/validation_compare.yaml")
    parser.add_argument("--no-log", action="store_true", help="Do not write to MLflow")
    parser.add_argument("--only", nargs="*", default=None, help="Model keys to score")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    eval_cfg = yaml.safe_load(Path(cfg["eval_config"]).read_text(encoding="utf-8"))
    base_cfg = load_baseline_config(eval_cfg["baseline_config"])
    client = MlflowClient(tracking_uri=eval_cfg["tracking_uri"])

    df_raw = pd.read_parquet(base_cfg.data_path).sort_values("timestamp", ignore_index=True)
    split = temporal_split(df_raw, base_cfg)
    y_val = split.df_val["is_fraud"].to_numpy(dtype=np.int64)
    amounts = split.df_val["amount"].to_numpy()
    val_graph = torch.load(Path(eval_cfg["processed_dir"]) / "val_graph.pt", weights_only=False)
    graph_labels = split_targets(val_graph, "val_mask")["edge_label"].numpy().astype(np.int64)
    if not np.array_equal(graph_labels, y_val):
        raise ValueError("Validation graph targets are not aligned with the validation rows")
    print(f"Validation rows: {len(y_val):,}, positives {int(y_val.sum())}")

    rows: list[dict[str, Any]] = []
    for entry in cfg["models"]:
        if args.only is not None and entry["key"] not in args.only:
            continue
        row: dict[str, Any] = {
            "key": entry["key"],
            "name": entry["name"],
            "run_id": entry["run_id"],
        }
        kind = entry["kind"]
        if kind == "not_available":
            row["status"] = "not run (no saved model; would need retraining)"
            rows.append(row)
            continue
        t0 = time.perf_counter()
        if kind in ("xgb_tabular", "xgb_label_free"):
            import xgboost as xgb

            label_free = kind == "xgb_label_free"
            feats = compute_tabular_features(df_raw, include_label_features=not label_free)
            cols = get_feature_column_names(include_label_features=not label_free)
            if label_free:
                graph_feats = compute_label_free_graph_features(df_raw)
                feats = pd.concat([feats, graph_feats], axis=1)
                cols = cols + list(graph_feats.columns)
            clf = xgb.XGBClassifier()
            clf.load_model(entry["checkpoint"])
            proba = clf.predict_proba(feats.loc[split.val_mask, cols])[:, 1]
        else:
            proba = score_val_gnn(entry, val_graph)
        ev = compute_metrics(
            y_true=y_val,
            y_pred_proba=proba,
            amounts=amounts,
            review_cost=base_cfg.cost.review_cost,
            default_fraud_amount=base_cfg.cost.default_fraud_amount,
            compute_ci=True,
        )
        row.update(
            status="scored",
            val_pr_auc=ev.pr_auc,
            ci=[ev.pr_auc_ci_lower, ev.pr_auc_ci_upper],
            val_recall_at_1pct_fpr=ev.recall_at_1pct_fpr,
            checkpoint=entry["checkpoint"],
            checkpoint_sha256=sha256_file(Path(entry["checkpoint"])),
            seconds=time.perf_counter() - t0,
        )
        print(
            f"  {entry['key']:<18} val PR-AUC {ev.pr_auc:.4f} "
            f"[{ev.pr_auc_ci_lower:.4f}-{ev.pr_auc_ci_upper:.4f}] ({row['seconds']:.0f}s)",
            flush=True,
        )
        if entry["run_id"] and not args.no_log:
            prefix = entry.get("metric_prefix", "val_pr_auc_rescored")
            now = int(time.time() * 1000)
            client.log_batch(
                entry["run_id"],
                metrics=[
                    Metric(prefix, ev.pr_auc, now, 0),
                    Metric(f"{prefix}_ci_lower", ev.pr_auc_ci_lower, now, 0),
                    Metric(f"{prefix}_ci_upper", ev.pr_auc_ci_upper, now, 0),
                ],
                tags=[
                    RunTag(f"{prefix}_at", datetime.now(UTC).isoformat(timespec="seconds")),
                    RunTag(f"{prefix}_checkpoint_sha256", row["checkpoint_sha256"]),
                ],
            )
            row["logged_as"] = prefix
        rows.append(row)

    out_path = Path(cfg["results_out"])
    if args.only is not None and out_path.exists():
        kept = [
            r
            for r in json.loads(out_path.read_text(encoding="utf-8"))["models"]
            if r["key"] not in args.only
        ]
        rows = kept + rows
    out_path.write_text(
        json.dumps(
            {"validation_rows": len(y_val), "positives": int(y_val.sum()), "models": rows},
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n| Model | Val PR-AUC [95% CI] | Val recall@1%FPR | Logged as |")
    print("|---|---|---|---|")
    for r in rows:
        if r["status"] != "scored":
            print(f"| {r['name']} | {r['status']} | | |")
            continue
        print(
            f"| {r['name']} | {r['val_pr_auc']:.4f} [{r['ci'][0]:.4f}-{r['ci'][1]:.4f}] | "
            f"{r['val_recall_at_1pct_fpr']:.4f} | {r.get('logged_as', 'not logged (no local run)')} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
