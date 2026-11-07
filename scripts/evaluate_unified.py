"""Re-score saved models on the test split from one source of predictions.

Rules:
1. Nothing is trained. Saved checkpoints listed in ``configs/evaluate.yaml`` are scored.
2. The decision threshold is selected on VALIDATION predictions of the same checkpoint,
   then applied unchanged to test.
3. GNN loaders use the model's own training config (``num_neighbors``, ``batch_size``) and
   device/IP indices only for models trained with binding, so a re-score reproduces the
   training-time evaluation of the same checkpoint.
4. Results are logged to the original MLflow run through the MLflow API under ``rescored_*``
   keys. Metrics logged at training time are never modified.
5. Canonical test predictions go to ``eval_predictions.npz`` and the validation thresholds to
   ``eval_thresholds.json`` (used by the paired bootstrap and the explanation examples).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import yaml
from mlflow.entities import Metric, RunTag
from mlflow.tracking import MlflowClient
from torch_geometric.data import HeteroData

from fraudgnn.config import BaselineConfig, GNNConfig, load_baseline_config, load_gnn_config
from fraudgnn.data.features import compute_tabular_features, get_feature_column_names
from fraudgnn.graph.temporal_loader import CausalHomogeneousLinkLoader, CausalLinkNeighborLoader
from fraudgnn.models.hetero_gat import HeteroGAT
from fraudgnn.models.sage import GraphSAGEEdgeClassifier, convert_hetero_to_homogeneous
from fraudgnn.training.evaluate import EvaluationMetrics, compute_metrics
from fraudgnn.training.train_gnn import evaluate_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.graph_signal_check import compute_causal_graph_features
from scripts.train_label_free_xgb import compute_label_free_graph_features
from scripts.train_mlp_baseline import PlainMLP

TARGET_REL = ("user", "transacts_at", "merchant")
EDGE_FEATURE_DIM = 27  # per-transaction edge features the MLP was trained on
RESCORE_PREFIX = "rescored_"
SHARED_PREFIX = "rescored_shared_"  # legacy shared-subgraph GNN batches (leaky)


@dataclass
class Split:
    """Validation and test rows of the raw transaction table (temporal split)."""

    df_val: pd.DataFrame
    df_test: pd.DataFrame
    val_mask: pd.Series
    test_mask: pd.Series


@dataclass
class Scored:
    """Validation/test predictions of one checkpoint plus scoring metadata."""

    val_proba: np.ndarray
    test_proba: np.ndarray
    latency_ms: float
    tags: dict[str, str]


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_matches_run_artifact(client: MlflowClient, run_id: str, ckpt: Path) -> str:
    """Compare a local checkpoint with the same-named artifact logged in the run.

    Returns:
        "true" / "false" when an artifact with the same file name exists, else "no_artifact".
    """
    pending = [""]
    while pending:
        folder = pending.pop()
        for art in client.list_artifacts(run_id, folder or None):
            if art.is_dir:
                pending.append(art.path)
            elif Path(art.path).name == ckpt.name:
                with tempfile.TemporaryDirectory() as tmp:
                    local = Path(client.download_artifacts(run_id, art.path, dst_path=tmp))
                    return str(sha256_file(local) == sha256_file(ckpt)).lower()
    return "no_artifact"


def temporal_split(df_raw: pd.DataFrame, cfg: BaselineConfig) -> Split:
    """Split the raw table by time using the configured train/val ratios."""
    timestamps = pd.to_datetime(df_raw["timestamp"])
    t_min, t_max = timestamps.min(), timestamps.max()
    dur = t_max - t_min
    train_cut = t_min + dur * cfg.split.train_ratio
    val_cut = t_min + dur * (cfg.split.train_ratio + cfg.split.val_ratio)
    val_mask = (timestamps > train_cut) & (timestamps <= val_cut)
    test_mask = timestamps > val_cut
    return Split(
        df_val=df_raw[val_mask].reset_index(drop=True),
        df_test=df_raw[test_mask].reset_index(drop=True),
        val_mask=val_mask,
        test_mask=test_mask,
    )


def score_xgb(model_path: Path, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Scored:
    """Score a saved XGBoost model on validation and test feature matrices."""
    clf = xgb.XGBClassifier()
    clf.load_model(str(model_path))
    val_proba = clf.predict_proba(X_val)[:, 1]
    t0 = time.perf_counter()
    test_proba = clf.predict_proba(X_test)[:, 1]
    latency = (time.perf_counter() - t0) / max(1, len(test_proba)) * 1000.0
    return Scored(val_proba, test_proba, latency, {})


def split_targets(graph: HeteroData, mask_name: str) -> dict[str, torch.Tensor]:
    """Extract masked target edges and their per-transaction device/IP indices."""
    store = graph[TARGET_REL]
    mask = getattr(store, mask_name)
    return {
        "edge_index": store.edge_index[:, mask],
        "edge_label": store.edge_label[mask],
        "edge_time": store.edge_time[mask],
        "edge_attr": store.edge_attr[mask],
        # uses_device / uses_ip edges are built one per transaction, in transaction order.
        "device_index": graph["user", "uses_device", "device"].edge_index[1][mask],
        "ip_index": graph["user", "uses_ip", "ip"].edge_index[1][mask],
        "mask": mask,
    }


def score_hetero_gat(
    entry: dict[str, Any],
    gnn_cfg: GNNConfig,
    graphs: dict[str, HeteroData],
    isolate_targets: bool = True,
) -> Scored:
    """Score a HeteroGAT checkpoint with the loader settings of its training config.

    ``isolate_targets=True`` gives every target its own causal subgraph (the fixed loader);
    ``False`` reproduces the legacy shared-subgraph batches used at training time.
    """
    use_device_ip = bool(entry["use_device_ip"])
    num_neighbors = gnn_cfg.training.num_neighbors[: gnn_cfg.model.num_layers]
    test_graph = graphs["test"]
    model = HeteroGAT(
        node_types=test_graph.node_types,
        edge_types=test_graph.edge_types,
        in_channels_dict={nt: 17 for nt in test_graph.node_types},
        hidden_channels=gnn_cfg.model.hidden_dim,
        edge_dim=gnn_cfg.model.edge_dim,
        num_layers=gnn_cfg.model.num_layers,
        heads=gnn_cfg.model.heads,
        dropout=gnn_cfg.model.dropout,
        conv_type=entry.get("conv_type", gnn_cfg.model.conv_type),
        use_edge_attr=gnn_cfg.model.use_edge_attr,
        use_device_ip=use_device_ip,
    )
    model.load_state_dict(torch.load(entry["checkpoint"], map_location="cpu", weights_only=True))
    model.eval()

    probas: dict[str, np.ndarray] = {}
    latency = 0.0
    for split_name, mask_name in (("val", "val_mask"), ("test", "test_mask")):
        tgt = split_targets(graphs[split_name], mask_name)
        loader = CausalLinkNeighborLoader(
            data=graphs[split_name],
            num_neighbors=num_neighbors,
            edge_label_index=(TARGET_REL, tgt["edge_index"]),
            edge_label=tgt["edge_label"],
            edge_time=tgt["edge_time"],
            edge_attr=tgt["edge_attr"] if gnn_cfg.model.use_edge_attr else None,
            device_index=tgt["device_index"] if use_device_ip else None,
            ip_index=tgt["ip_index"] if use_device_ip else None,
            batch_size=gnn_cfg.training.batch_size,
            shuffle=False,
            seed=gnn_cfg.training.seed,
            isolate_targets=isolate_targets,
        )
        _, proba, elapsed = evaluate_model(model, loader, is_hetero=True)
        probas[split_name] = proba
        latency = elapsed / max(1, len(proba)) * 1000.0
    tags = {
        "rescore_num_neighbors": json.dumps(num_neighbors),
        "rescore_batch_size": str(gnn_cfg.training.batch_size),
        "rescore_use_device_ip": str(use_device_ip).lower(),
        "rescore_isolate_targets": str(isolate_targets).lower(),
    }
    return Scored(probas["val"], probas["test"], latency, tags)


def score_graphsage(
    entry: dict[str, Any],
    gnn_cfg: GNNConfig,
    graphs: dict[str, HeteroData],
    isolate_targets: bool = True,
) -> Scored:
    """Score a homogeneous GraphSAGE checkpoint with its training loader settings.

    ``isolate_targets=False`` reproduces the legacy shared-subgraph batches.
    """
    num_neighbors = gnn_cfg.training.num_neighbors[: gnn_cfg.model.num_layers]
    model: GraphSAGEEdgeClassifier | None = None
    probas: dict[str, np.ndarray] = {}
    latency = 0.0
    for split_name, mask_name in (("val", "val_mask"), ("test", "test_mask")):
        homo, _, h_idx, h_attr = convert_hetero_to_homogeneous(graphs[split_name])
        tgt = split_targets(graphs[split_name], mask_name)
        if model is None:
            model = GraphSAGEEdgeClassifier(
                in_channels=int(homo.x.size(1)),
                hidden_channels=gnn_cfg.model.hidden_dim,
                edge_dim=gnn_cfg.model.edge_dim,
                num_layers=gnn_cfg.model.num_layers,
                dropout=gnn_cfg.model.dropout,
                use_edge_attr=gnn_cfg.model.use_edge_attr,
            )
            model.load_state_dict(
                torch.load(entry["checkpoint"], map_location="cpu", weights_only=True)
            )
            model.eval()
        attr = h_attr[tgt["mask"]] if h_attr is not None else None
        loader = CausalHomogeneousLinkLoader(
            data=homo,
            edge_label_index=h_idx[:, tgt["mask"]],
            edge_label=tgt["edge_label"],
            edge_time=tgt["edge_time"],
            edge_attr=attr if gnn_cfg.model.use_edge_attr else None,
            batch_size=gnn_cfg.training.batch_size,
            num_neighbors=num_neighbors,
            shuffle=False,
            seed=gnn_cfg.training.seed,
            isolate_targets=isolate_targets,
        )
        _, proba, elapsed = evaluate_model(model, loader, is_hetero=False)
        probas[split_name] = proba
        latency = elapsed / max(1, len(proba)) * 1000.0
    tags = {
        "rescore_num_neighbors": json.dumps(num_neighbors),
        "rescore_batch_size": str(gnn_cfg.training.batch_size),
        "rescore_isolate_targets": str(isolate_targets).lower(),
    }
    return Scored(probas["val"], probas["test"], latency, tags)


def score_mlp(ckpt: Path, graphs: dict[str, HeteroData]) -> Scored:
    """Score the edge-feature MLP on the graph's 27 (train-scaled) target edge attributes."""
    model = PlainMLP(in_dim=EDGE_FEATURE_DIM)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    probas: dict[str, np.ndarray] = {}
    latency = 0.0
    with torch.no_grad():
        for split_name, mask_name in (("val", "val_mask"), ("test", "test_mask")):
            x = split_targets(graphs[split_name], mask_name)["edge_attr"].float()
            t0 = time.perf_counter()
            probas[split_name] = torch.sigmoid(model(x)).numpy()
            latency = (time.perf_counter() - t0) / max(1, len(x)) * 1000.0
    return Scored(probas["val"], probas["test"], latency, {})


def check_run_params(client: MlflowClient, entry: dict[str, Any], gnn_cfg: GNNConfig) -> None:
    """Fail loudly if the re-score architecture disagrees with the run's logged params."""
    params = client.get_run(entry["run_id"]).data.params
    expected = {
        "conv_type": str(entry.get("conv_type", gnn_cfg.model.conv_type)),
        "hidden_dim": str(gnn_cfg.model.hidden_dim),
        "num_layers": str(gnn_cfg.model.num_layers),
        "batch_size": str(gnn_cfg.training.batch_size),
    }
    if "use_device_ip" in entry:
        expected["use_device_ip"] = str(bool(entry["use_device_ip"]))
    for key, value in expected.items():
        if key in params and params[key] != value:
            raise ValueError(
                f"{entry['name']}: run param {key}={params[key]} but re-score uses {value}"
            )


def metrics_payload(
    ev: EvaluationMetrics, threshold: float, latency: float, prefix: str = RESCORE_PREFIX
) -> dict[str, float]:
    """Build the metric dictionary for one model, keys prefixed with ``prefix``."""
    payload = {
        "test_pr_auc": ev.pr_auc,
        "test_pr_auc_ci_lower": ev.pr_auc_ci_lower,
        "test_pr_auc_ci_upper": ev.pr_auc_ci_upper,
        "test_roc_auc": ev.roc_auc,
        "test_f1": ev.f1,
        "test_recall_at_1pct_fpr": ev.recall_at_1pct_fpr,
        "test_business_cost": ev.total_cost,
        "val_selected_threshold": threshold,
        "latency_ms_per_sample": latency,
    }
    for fraud_type, rec in ev.per_type_recall.items():
        payload[f"test_recall_{fraud_type}"] = rec
    return {f"{prefix}{k}": float(v) for k, v in payload.items()}


def main(argv: list[str] | None = None) -> int:
    """Re-score all configured checkpoints and log rescored_* metrics via the MLflow API."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/evaluate.yaml")
    parser.add_argument("--only", nargs="*", default=None, help="Model keys to re-score")
    parser.add_argument("--dry-run", action="store_true", help="Score but do not log to MLflow")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_cfg = load_baseline_config(cfg["baseline_config"])
    client = MlflowClient(tracking_uri=cfg["tracking_uri"])
    processed = Path(cfg["processed_dir"])

    df_raw = pd.read_parquet(base_cfg.data_path)
    df_raw = df_raw.sort_values("timestamp", ignore_index=True)
    split = temporal_split(df_raw, base_cfg)
    y_val = split.df_val["is_fraud"].to_numpy(dtype=np.int64)
    y_test = split.df_test["is_fraud"].to_numpy(dtype=np.int64)
    print(
        f"Validation rows: {len(y_val):,} | Test rows: {len(y_test):,}, positives {y_test.sum()}"
    )

    graphs: dict[str, HeteroData] = {}
    predictions: dict[str, np.ndarray] = {"y_true": y_test}
    thresholds: dict[str, dict[str, Any]] = {}

    pred_path = Path(cfg["predictions_out"])
    if pred_path.exists():
        with np.load(pred_path) as old:
            predictions.update({k: old[k] for k in old.files if k != "y_true"})

    for entry in cfg["models"]:
        key, kind = entry["key"], entry["kind"]
        if args.only is not None and key not in args.only:
            continue
        if kind == "logged_only":
            print(f"\n[{key}] {entry['name']}: no checkpoint saved; re-score not run.")
            continue

        print(f"\n[{key}] Re-scoring {entry['name']} ({entry['checkpoint']})...")
        ckpt = Path(entry["checkpoint"])
        if kind == "xgb_oracle":
            # Same features as scripts/graph_signal_check.py (uses past fraud labels).
            feats = compute_tabular_features(df_raw, smoothing_alpha=base_cfg.smoothing_alpha)
            graph_feats = compute_causal_graph_features(df_raw)
            feats = pd.concat([feats, graph_feats], axis=1)
            cols = get_feature_column_names() + list(graph_feats.columns)
            scored = score_xgb(
                ckpt, feats.loc[split.val_mask, cols], feats.loc[split.test_mask, cols]
            )
            variants = [(RESCORE_PREFIX, key, scored)]
        elif kind == "mlp":
            if not graphs:
                for name in ("val", "test"):
                    graphs[name] = torch.load(processed / f"{name}_graph.pt", weights_only=False)
            variants = [(RESCORE_PREFIX, key, score_mlp(ckpt, graphs))]
        elif kind in ("xgb_tabular", "xgb_label_free"):
            # The tabular baseline was trained with include_label_features=True (27 columns).
            label_free = kind == "xgb_label_free"
            feats = compute_tabular_features(df_raw, include_label_features=not label_free)
            cols = get_feature_column_names(include_label_features=not label_free)
            if label_free:
                graph_feats = compute_label_free_graph_features(df_raw)
                feats = pd.concat([feats, graph_feats], axis=1)
                cols = cols + list(graph_feats.columns)
            scored = score_xgb(
                ckpt, feats.loc[split.val_mask, cols], feats.loc[split.test_mask, cols]
            )
        else:
            if not graphs:
                for name in ("val", "test"):
                    graphs[name] = torch.load(processed / f"{name}_graph.pt", weights_only=False)
            gnn_cfg = load_gnn_config(entry["gnn_config"])
            check_run_params(client, entry, gnn_cfg)
            if kind == "hetero_gat":
                variants = [
                    (RESCORE_PREFIX, key, score_hetero_gat(entry, gnn_cfg, graphs, True)),
                    (
                        SHARED_PREFIX,
                        f"{key}_shared",
                        score_hetero_gat(entry, gnn_cfg, graphs, False),
                    ),
                ]
            else:
                variants = [
                    (RESCORE_PREFIX, key, score_graphsage(entry, gnn_cfg, graphs, True)),
                    (
                        SHARED_PREFIX,
                        f"{key}_shared",
                        score_graphsage(entry, gnn_cfg, graphs, False),
                    ),
                ]
        if kind in ("xgb_tabular", "xgb_label_free"):
            variants = [(RESCORE_PREFIX, key, scored)]

        for prefix, pred_key, scored in variants:
            val_ev = compute_metrics(
                y_true=y_val,
                y_pred_proba=scored.val_proba,
                amounts=split.df_val["amount"].to_numpy(),
                review_cost=base_cfg.cost.review_cost,
                default_fraud_amount=base_cfg.cost.default_fraud_amount,
            )
            threshold = val_ev.best_threshold
            test_ev = compute_metrics(
                y_true=y_test,
                y_pred_proba=scored.test_proba,
                amounts=split.df_test["amount"].to_numpy(),
                review_cost=base_cfg.cost.review_cost,
                default_fraud_amount=base_cfg.cost.default_fraud_amount,
                threshold_override=threshold,
                fraud_types=split.df_test["fraud_type"].to_numpy(),
                compute_ci=True,
            )
            print(
                f"  [{prefix}] Test PR-AUC {test_ev.pr_auc:.4f} [{test_ev.pr_auc_ci_lower:.4f}-"
                f"{test_ev.pr_auc_ci_upper:.4f}] | val threshold {threshold:.4f}"
            )
            predictions[pred_key] = scored.test_proba
            thresholds[pred_key] = {
                "run_id": entry["run_id"],
                "val_selected_threshold": threshold,
            }

            if args.dry_run:
                continue
            now_ms = int(time.time() * 1000)
            metrics = [
                Metric(k, v, now_ms, 0)
                for k, v in metrics_payload(test_ev, threshold, scored.latency_ms, prefix).items()
            ]
            tags = {
                "rescored_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "rescore_script": "scripts/evaluate_unified.py",
                "rescore_checkpoint": str(ckpt),
                "rescore_checkpoint_sha256": sha256_file(ckpt),
                "rescore_checkpoint_matches_run_artifact": checkpoint_matches_run_artifact(
                    client, entry["run_id"], ckpt
                ),
                **(scored.tags if prefix == RESCORE_PREFIX else {}),
            }
            client.log_batch(
                entry["run_id"],
                metrics=metrics,
                tags=[RunTag(k, v) for k, v in tags.items()],
            )
            print(f"  Logged {len(metrics)} {prefix}* metrics to run {entry['run_id'][:8]}.")

    np.savez_compressed(pred_path, **predictions)
    thr_path = Path(cfg["thresholds_out"])
    old_thr = json.loads(thr_path.read_text(encoding="utf-8")) if thr_path.exists() else {}
    thr_path.write_text(json.dumps({**old_thr, **thresholds}, indent=2), encoding="utf-8")
    print(f"\nSaved predictions to {pred_path} and thresholds to {thr_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
