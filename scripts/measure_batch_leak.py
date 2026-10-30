"""Measure how batch composition changes HeteroGAT scores (batch-sharing leak).

For each checkpoint and batch size, score the same fixed rows: all positives plus a seeded
random sample of negatives, in both validation and test. Batch size 1 uses isolated
per-target subgraphs (identical to scoring each target alone). Larger batch sizes use the
legacy shared-subgraph batches with the boundaries of a full ordered pass, so each row sees
the same batch neighbours as in the original evaluation.

Reported per (model, batch size), on the same test rows:
- PR-AUC on the subset (inflated: negatives are downsampled) and PR-AUC with negatives
  re-weighted by (all negatives / sampled negatives), an estimate of full-test PR-AUC,
  both with bootstrap CIs;
- recall at 1% FPR, with the threshold chosen on validation-subset negatives;
- the paired bootstrap difference of weighted PR-AUC against batch size 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score
from torch_geometric.data import HeteroData

from fraudgnn.config import load_gnn_config
from fraudgnn.graph.temporal_loader import CausalLinkNeighborLoader
from fraudgnn.models.hetero_gat import HeteroGAT
from fraudgnn.training.train_gnn import evaluate_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_unified import TARGET_REL, split_targets


def sample_rows(labels: np.ndarray, n_neg: int, rng: np.random.Generator) -> np.ndarray:
    """All positive positions plus ``n_neg`` random negative positions, sorted."""
    pos = np.flatnonzero(labels == 1)
    neg = rng.choice(np.flatnonzero(labels == 0), size=n_neg, replace=False)
    return np.sort(np.concatenate([pos, neg]))


def batches_for(
    loader: CausalLinkNeighborLoader, rows: np.ndarray, batch_size: int, isolated_chunk: int
) -> tuple[Iterator[HeteroData], np.ndarray]:
    """Batches covering ``rows`` and the target position of every scored output."""
    if loader.isolate_targets:
        chunks = [rows[i : i + isolated_chunk] for i in range(0, len(rows), isolated_chunk)]
    else:
        ids = np.unique(rows // batch_size)
        chunks = [
            np.arange(b * batch_size, min((b + 1) * batch_size, loader.num_edges)) for b in ids
        ]
    order = np.concatenate(chunks)
    return (loader.build_batch(c) for c in chunks), order


def score(
    model: HeteroGAT,
    graph: HeteroData,
    tgt: dict[str, torch.Tensor],
    rows: np.ndarray,
    batch_size: int,
    num_neighbors: list[int],
    use_device_ip: bool,
    isolated_chunk: int,
) -> np.ndarray:
    """Probabilities for ``rows`` (in row order) at the given batch size."""
    loader = CausalLinkNeighborLoader(
        data=graph,
        num_neighbors=num_neighbors,
        edge_label_index=(TARGET_REL, tgt["edge_index"]),
        edge_label=tgt["edge_label"],
        edge_time=tgt["edge_time"],
        edge_attr=tgt["edge_attr"],
        device_index=tgt["device_index"] if use_device_ip else None,
        ip_index=tgt["ip_index"] if use_device_ip else None,
        batch_size=batch_size,
        shuffle=False,
        isolate_targets=batch_size == 1,
    )
    batches, order = batches_for(loader, rows, batch_size, isolated_chunk)
    _, proba, _ = evaluate_model(model, batches, is_hetero=True)
    by_pos = dict(zip(order.tolist(), proba.tolist(), strict=True))
    return np.array([by_pos[r] for r in rows.tolist()])


def threshold_at_fpr(neg_scores: np.ndarray, fpr: float) -> float:
    """Smallest threshold whose false-positive rate on ``neg_scores`` is <= ``fpr``."""
    return float(np.quantile(neg_scores, 1.0 - fpr, method="higher"))


def bootstrap_ap(
    y: np.ndarray,
    scores: dict[int, np.ndarray],
    weights: np.ndarray,
    ref: int,
    n_iter: int,
    alpha: float,
    rng: np.random.Generator,
) -> dict[int, dict[str, float]]:
    """Weighted/unweighted AP with CIs and paired differences vs ``ref`` (same resamples)."""
    draws: dict[int, dict[str, list[float]]] = {b: {"w": [], "u": [], "d": []} for b in scores}
    for _ in range(n_iter):
        idx = rng.integers(0, len(y), size=len(y))
        if y[idx].sum() == 0:
            continue
        ref_w = average_precision_score(y[idx], scores[ref][idx], sample_weight=weights[idx])
        for b, s in scores.items():
            w_ap = average_precision_score(y[idx], s[idx], sample_weight=weights[idx])
            draws[b]["w"].append(w_ap)
            draws[b]["u"].append(average_precision_score(y[idx], s[idx]))
            draws[b]["d"].append(w_ap - ref_w)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out: dict[int, dict[str, float]] = {}
    for b, s in scores.items():
        out[b] = {
            "pr_auc_subset": float(average_precision_score(y, s)),
            "pr_auc_subset_ci": [float(np.percentile(draws[b]["u"], q)) for q in (lo, hi)],
            "pr_auc_weighted": float(average_precision_score(y, s, sample_weight=weights)),
            "pr_auc_weighted_ci": [float(np.percentile(draws[b]["w"], q)) for q in (lo, hi)],
            "diff_vs_ref": float(
                average_precision_score(y, s, sample_weight=weights)
                - average_precision_score(y, scores[ref], sample_weight=weights)
            ),
            "diff_vs_ref_ci": [float(np.percentile(draws[b]["d"], q)) for q in (lo, hi)],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the batch-size measurement for both HeteroGAT checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/batch_leak.yaml")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    eval_cfg = yaml.safe_load(Path(cfg["eval_config"]).read_text(encoding="utf-8"))
    entries = {e["key"]: e for e in eval_cfg["models"]}
    processed = Path(eval_cfg["processed_dir"])
    rng = np.random.default_rng(cfg["seed"])

    data: dict[str, dict[str, Any]] = {}
    for split, mask in (("val", "val_mask"), ("test", "test_mask")):
        graph = torch.load(processed / f"{split}_graph.pt", weights_only=False)
        tgt = split_targets(graph, mask)
        labels = tgt["edge_label"].numpy().astype(np.int64)
        rows = sample_rows(labels, cfg["negatives_per_split"], rng)
        n_neg_all = int((labels == 0).sum())
        data[split] = {
            "graph": graph,
            "tgt": tgt,
            "rows": rows,
            "y": labels[rows],
            "neg_weight": n_neg_all / cfg["negatives_per_split"],
        }
        print(
            f"{split}: {int(labels[rows].sum())} positives + {cfg['negatives_per_split']} "
            f"negatives (of {n_neg_all}); negative weight {data[split]['neg_weight']:.3f}"
        )

    y_test = data["test"]["y"]
    weights = np.where(y_test == 1, 1.0, data["test"]["neg_weight"])
    results: dict[str, Any] = {"config": cfg, "models": {}}

    for key in cfg["models"]:
        entry = entries[key]
        gnn_cfg = load_gnn_config(entry["gnn_config"])
        use_dev = bool(entry["use_device_ip"])
        num_neighbors = gnn_cfg.training.num_neighbors[: gnn_cfg.model.num_layers]
        graph0 = data["test"]["graph"]
        model = HeteroGAT(
            node_types=graph0.node_types,
            edge_types=graph0.edge_types,
            in_channels_dict={nt: 17 for nt in graph0.node_types},
            hidden_channels=gnn_cfg.model.hidden_dim,
            edge_dim=gnn_cfg.model.edge_dim,
            num_layers=gnn_cfg.model.num_layers,
            heads=gnn_cfg.model.heads,
            dropout=gnn_cfg.model.dropout,
            conv_type=entry.get("conv_type", gnn_cfg.model.conv_type),
            use_edge_attr=gnn_cfg.model.use_edge_attr,
            use_device_ip=use_dev,
        )
        model.load_state_dict(
            torch.load(entry["checkpoint"], map_location="cpu", weights_only=True)
        )
        model.eval()

        test_scores: dict[int, np.ndarray] = {}
        per_bs: dict[int, dict[str, float]] = {}
        for bs in cfg["batch_sizes"]:
            t0 = time.perf_counter()
            s = {
                split: score(
                    model,
                    d["graph"],
                    d["tgt"],
                    d["rows"],
                    bs,
                    num_neighbors,
                    use_dev,
                    cfg["isolated_chunk"],
                )
                for split, d in data.items()
            }
            thr = threshold_at_fpr(s["val"][data["val"]["y"] == 0], cfg["target_fpr"])
            test_scores[bs] = s["test"]
            per_bs[bs] = {
                "val_threshold_at_fpr": thr,
                "test_recall_at_val_thr": float((s["test"][y_test == 1] >= thr).mean()),
                "test_fpr_at_val_thr": float((s["test"][y_test == 0] >= thr).mean()),
                "seconds": time.perf_counter() - t0,
            }
            print(
                f"  [{key}] batch {bs}: recall {per_bs[bs]['test_recall_at_val_thr']:.4f} "
                f"(thr {thr:.4f}) in {per_bs[bs]['seconds']:.0f}s",
                flush=True,
            )

        boot = bootstrap_ap(
            y_test,
            test_scores,
            weights,
            ref=1,
            n_iter=cfg["bootstrap_iterations"],
            alpha=cfg["bootstrap_alpha"],
            rng=np.random.default_rng(cfg["seed"]),
        )
        for bs in cfg["batch_sizes"]:
            per_bs[bs].update(boot[bs])
            print(
                f"  [{key}] batch {bs}: weighted PR-AUC {boot[bs]['pr_auc_weighted']:.4f} "
                f"subset PR-AUC {boot[bs]['pr_auc_subset']:.4f} "
                f"diff vs 1 {boot[bs]['diff_vs_ref']:+.4f}"
            )
        results["models"][key] = {"name": entry["name"], "batch_sizes": per_bs}

    results["subset"] = {
        split: {
            "positives": int(d["y"].sum()),
            "negatives": int((d["y"] == 0).sum()),
            "negative_weight": d["neg_weight"],
        }
        for split, d in data.items()
    }
    Path(cfg["results_out"]).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {cfg['results_out']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
