"""Score the sealed holdout exactly once, after the freeze (docs/protocol.md §7).

Refuses to run unless docs/verification/freeze.json exists and matches every checkpoint, the
protocol and the holdout files, and refuses if a holdout result already exists. Thresholds
come from each model's scores on the ORIGINAL validation split; holdout targets are the last
20% of time in the holdout world (its test_mask), with all earlier holdout data as history.

``--rehearse`` runs the same scoring path on the ORIGINAL test split instead (bypassing only
the freeze and run-once guards), writes docs/verification/holdout_rehearsal.json and checks
that each model reproduces its known leak-free test PR-AUC from MLflow.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np
import pandas as pd
import torch
import yaml
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score

from fraudgnn.config import BaselineConfig, load_baseline_config, load_gnn_config
from fraudgnn.data.features import compute_tabular_features, get_feature_column_names
from fraudgnn.training.evaluate import compute_metrics

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_unified import (
    Split,
    score_hetero_gat,
    score_mlp,
    score_xgb,
    split_targets,
    temporal_split,
)
from scripts.graph_signal_check import compute_causal_graph_features
from scripts.holdout_guard import check_guard
from scripts.train_label_free_xgb import compute_label_free_graph_features


def read_raw_in_time_order(path: str | Path) -> pd.DataFrame:
    """Read a raw transaction file; it must already be in time order (never re-sorted)."""
    df = pd.read_parquet(path)
    if not pd.to_datetime(df["timestamp"]).is_monotonic_increasing:
        raise ValueError(f"{path} is not in time order; refusing to score")
    return df


def xgb_features(
    df: pd.DataFrame, kind: str, smoothing_alpha: float
) -> tuple[pd.DataFrame, list[str]]:
    """Feature matrix and column list for one XGBoost kind (computed per world)."""
    if kind == "xgb_tabular":
        return compute_tabular_features(
            df, include_label_features=True
        ), get_feature_column_names()
    if kind == "xgb_oracle":
        feats = compute_tabular_features(df, smoothing_alpha=smoothing_alpha)
        graph = compute_causal_graph_features(df)
        return pd.concat([feats, graph], axis=1), get_feature_column_names() + list(graph.columns)
    feats = compute_tabular_features(df, include_label_features=False)
    graph = compute_label_free_graph_features(df)
    cols = get_feature_column_names(include_label_features=False) + list(graph.columns)
    return pd.concat([feats, graph], axis=1), cols


def score_models(
    models: list[dict[str, Any]],
    base: BaselineConfig,
    df_orig: pd.DataFrame,
    split_orig: Split,
    df_target: pd.DataFrame,
    split_target: Split,
    graphs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    """Score every model: threshold on the original validation split, metrics on the target.

    Before any metric: the scored targets' graph labels must equal the dataframe labels
    exactly, and every prediction vector must have the right length.
    """
    y_val = split_orig.df_val["is_fraud"].to_numpy(dtype=np.int64)
    y_target = split_target.df_test["is_fraud"].to_numpy(dtype=np.int64)
    graph_val = split_targets(graphs["val"], "val_mask")["edge_label"].numpy().astype(np.int64)
    graph_target = (
        split_targets(graphs["test"], "test_mask")["edge_label"].numpy().astype(np.int64)
    )
    if not np.array_equal(graph_val, y_val):
        raise AssertionError("validation graph labels differ from the validation rows")
    if not np.array_equal(graph_target, y_target):
        raise AssertionError("scored graph labels differ from the scored rows")

    results: dict[str, Any] = {}
    probs: dict[str, np.ndarray] = {}
    for m in models:
        kind, ckpt = m["kind"], Path(m["checkpoint"])
        if kind == "hetero_gat":
            scored = score_hetero_gat(m, load_gnn_config(m["gnn_config"]), graphs, True)
        elif kind == "mlp":
            scored = score_mlp(ckpt, graphs)
        else:
            f_orig, cols = xgb_features(df_orig, kind, base.smoothing_alpha)
            f_target, _ = xgb_features(df_target, kind, base.smoothing_alpha)
            scored = score_xgb(
                ckpt,
                f_orig.loc[split_orig.val_mask, cols],
                f_target.loc[split_target.test_mask, cols],
            )
        if len(scored.val_proba) != len(y_val) or len(scored.test_proba) != len(y_target):
            raise AssertionError(
                f"{m['key']}: prediction lengths {len(scored.val_proba)}/{len(scored.test_proba)}"
                f" differ from labels {len(y_val)}/{len(y_target)}"
            )
        val_ev = compute_metrics(
            y_true=y_val,
            y_pred_proba=scored.val_proba,
            amounts=split_orig.df_val["amount"].to_numpy(),
            review_cost=base.cost.review_cost,
            default_fraud_amount=base.cost.default_fraud_amount,
        )
        ev = compute_metrics(
            y_true=y_target,
            y_pred_proba=scored.test_proba,
            amounts=split_target.df_test["amount"].to_numpy(),
            review_cost=base.cost.review_cost,
            default_fraud_amount=base.cost.default_fraud_amount,
            threshold_override=val_ev.best_threshold,
            fraud_types=split_target.df_test["fraud_type"].to_numpy(),
            compute_ci=True,
        )
        probs[m["key"]] = scored.test_proba
        results[m["key"]] = {
            "checkpoint": ckpt.as_posix(),
            "val_threshold_original": val_ev.best_threshold,
            "metrics": ev.to_dict(),
        }
        print(f"{m['key']}: PR-AUC {ev.pr_auc:.6f}", flush=True)
    return results, probs, y_target


def paired_bootstrap(
    pairs: list[list[str]], probs: dict[str, np.ndarray], y: np.ndarray, boot: dict[str, Any]
) -> list[dict[str, Any]]:
    """PR-AUC(a) - PR-AUC(b) on identical resamples for each pair."""
    rng = np.random.default_rng(boot["seed"])
    draws = [rng.integers(0, len(y), size=len(y)) for _ in range(boot["iterations"])]
    draws = [idx for idx in draws if y[idx].sum() > 0]
    out = []
    for a, b in pairs:
        diffs = np.asarray(
            [
                average_precision_score(y[i], probs[a][i])
                - average_precision_score(y[i], probs[b][i])
                for i in draws
            ]
        )
        out.append(
            {
                "a": a,
                "b": b,
                "point": float(
                    average_precision_score(y, probs[a]) - average_precision_score(y, probs[b])
                ),
                "ci": [
                    float(np.percentile(diffs, 100 * boot["alpha"] / 2)),
                    float(np.percentile(diffs, 100 * (1 - boot["alpha"] / 2))),
                ],
                "resamples": len(draws),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    """Score the holdout once (or rehearse on the original test split)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/holdout.yaml")
    parser.add_argument("--rehearse", action="store_true")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    freeze = None if args.rehearse else check_guard(cfg)  # real path: refuse before loading

    eval_cfg = yaml.safe_load(Path(cfg["eval_config"]).read_text(encoding="utf-8"))
    base = load_baseline_config(eval_cfg["baseline_config"])
    df_orig = read_raw_in_time_order(base.data_path)
    split_orig = temporal_split(df_orig, base)
    orig_processed = Path(eval_cfg["processed_dir"])
    graphs: dict[str, Any] = {
        "val": torch.load(orig_processed / "val_graph.pt", weights_only=False)
    }

    if args.rehearse:
        reh = cfg["rehearsal"]
        df_target, split_target = df_orig, split_orig
        graphs["test"] = torch.load(orig_processed / "test_graph.pt", weights_only=False)
        models = reh["models"]
    else:
        df_target = read_raw_in_time_order(cfg["holdout_raw"])
        split_target = temporal_split(df_target, base)
        graphs["test"] = torch.load(
            Path(cfg["holdout_processed"]) / "test_graph.pt", weights_only=False
        )
        models = cfg["models"]

    results, probs, y = score_models(
        models, base, df_orig, split_orig, df_target, split_target, graphs
    )
    doc: dict[str, Any] = {"scored_at": datetime.now(UTC).isoformat(timespec="seconds")}

    if args.rehearse:
        client = MlflowClient(tracking_uri=eval_cfg["tracking_uri"])
        checks = []
        for m in models:
            expected = client.get_run(m["run_id"]).data.metrics[reh["expected_metric"]]
            got = results[m["key"]]["metrics"]["pr_auc"]
            checks.append(
                {
                    "key": m["key"],
                    "run_id": m["run_id"],
                    "expected": expected,
                    "rehearsal": got,
                    "abs_diff": abs(got - expected),
                    "pass": abs(got - expected) <= reh["tolerance"],
                }
            )
        doc.update(
            mode="rehearsal on the ORIGINAL test split (freeze and run-once guards bypassed)",
            tolerance=reh["tolerance"],
            reproduction=checks,
            all_reproduced=all(c["pass"] for c in checks),
            models=results,
        )
        out = Path(reh["results_out"])
        out.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
        for c in checks:
            print(
                f"{c['key']:<24} expected {c['expected']:.10f} rehearsal {c['rehearsal']:.10f} "
                f"|diff| {c['abs_diff']:.1e} {'PASS' if c['pass'] else 'FAIL'}"
            )
        print(
            ("ALL REPRODUCED" if doc["all_reproduced"] else "REPRODUCTION FAILED") + f" -> {out}"
        )
        return 0 if doc["all_reproduced"] else 1

    doc.update(
        freeze=freeze,
        models=results,
        paired_bootstrap=paired_bootstrap(
            cfg["paired_bootstrap_pairs"], probs, y, cfg["bootstrap"]
        ),
    )
    out = Path(cfg["results_out"])
    with out.open("x", encoding="utf-8") as fh:  # exclusive create: scored exactly once
        fh.write(json.dumps(doc, indent=2, default=str) + "\n")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
