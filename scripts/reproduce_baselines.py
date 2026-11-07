"""Retrain the oracle XGBoost and the MLP, check test reproduction, then log validation PR-AUC.

Both original runs saved no model. Each is retrained with its original code, config and seed
(without creating a new MLflow run). The retrained model is accepted only if its test PR-AUC
is within ``test_pr_auc_abs_tolerance`` of the original run's logged value; only then is its validation PR-AUC (with CI)
logged to the original run under ``val_pr_auc_rescored*`` and the model saved.
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
import yaml
from mlflow.entities import Metric, RunTag
from mlflow.tracking import MlflowClient

from fraudgnn.config import load_baseline_config
from fraudgnn.training.evaluate import compute_metrics

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_unified import sha256_file
from scripts.graph_signal_check import fit_oracle
from scripts.train_mlp_baseline import train_and_eval_mlp


def check_and_log(
    client: MlflowClient,
    run_id: str,
    test_pr_auc: float,
    y_val: np.ndarray,
    val_probs: np.ndarray,
    amounts: np.ndarray,
    model_path: Path,
    log: bool,
    tolerance: float,
) -> dict[str, Any]:
    """Compare with the original run; log validation PR-AUC only if reproduced."""
    original = client.get_run(run_id).data.metrics
    lo, hi = original["test_pr_auc_ci_lower"], original["test_pr_auc_ci_upper"]
    signed = test_pr_auc - original["test_pr_auc"]
    result: dict[str, Any] = {
        "run_id": run_id,
        "original_test_pr_auc": original["test_pr_auc"],
        "original_ci": [lo, hi],
        "retrained_test_pr_auc": test_pr_auc,
        "signed_difference": signed,
        "tolerance": tolerance,
        "reproduced": abs(signed) <= tolerance,
    }
    print(
        f"  reproduction {run_id[:8]}: retrained {test_pr_auc:.10f} - original "
        f"{original['test_pr_auc']:.10f} = {signed:+.3e} (tolerance {tolerance:.0e}) -> "
        f"{'REPRODUCED' if result['reproduced'] else 'NOT REPRODUCED'}",
        flush=True,
    )
    if not result["reproduced"]:
        return result
    ev = compute_metrics(y_true=y_val, y_pred_proba=val_probs, amounts=amounts, compute_ci=True)
    result.update(
        val_pr_auc=ev.pr_auc,
        val_ci=[ev.pr_auc_ci_lower, ev.pr_auc_ci_upper],
        val_recall_at_1pct_fpr=ev.recall_at_1pct_fpr,
        model=str(model_path),
        model_sha256=sha256_file(model_path),
    )
    if log:
        now = int(time.time() * 1000)
        client.log_batch(
            run_id,
            metrics=[
                Metric("val_pr_auc_rescored", ev.pr_auc, now, 0),
                Metric("val_pr_auc_rescored_ci_lower", ev.pr_auc_ci_lower, now, 0),
                Metric("val_pr_auc_rescored_ci_upper", ev.pr_auc_ci_upper, now, 0),
            ],
            tags=[
                RunTag("val_pr_auc_rescored_at", datetime.now(UTC).isoformat(timespec="seconds")),
                RunTag(
                    "val_pr_auc_rescored_source", "retrained reproduction (same code/config/seed)"
                ),
                RunTag("val_pr_auc_rescored_checkpoint_sha256", result["model_sha256"]),
                RunTag("reproduced_test_pr_auc", f"{test_pr_auc:.6f}"),
            ],
        )
        result["logged_as"] = "val_pr_auc_rescored"
    return result


def main(argv: list[str] | None = None) -> int:
    """Run both reproductions."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/reproduce_baselines.yaml")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    eval_cfg = yaml.safe_load(Path(cfg["eval_config"]).read_text(encoding="utf-8"))
    client = MlflowClient(tracking_uri=eval_cfg["tracking_uri"])
    report: dict[str, Any] = {}

    t0 = time.perf_counter()
    res = fit_oracle(load_baseline_config(cfg["oracle"]["baseline_config"]))
    oracle_path = Path(cfg["oracle"]["save_model"])
    if oracle_path.exists():
        raise FileExistsError(f"{oracle_path} exists; refusing to overwrite")
    res["model"].save_model(str(oracle_path))
    report["oracle"] = check_and_log(
        client,
        cfg["oracle"]["run_id"],
        res["test_metrics"].pr_auc,
        res["val_df"]["is_fraud"].to_numpy(dtype=np.int64),
        res["y_val_prob"],
        res["val_df"]["amount"].to_numpy(),
        oracle_path,
        not args.no_log,
        cfg["test_pr_auc_abs_tolerance"],
    )
    report["oracle"]["seconds"] = time.perf_counter() - t0
    print("oracle:", report["oracle"], flush=True)

    t0 = time.perf_counter()
    mlp_path = Path(cfg["mlp"]["save_model"])
    if mlp_path.exists():
        raise FileExistsError(f"{mlp_path} exists; refusing to overwrite")
    mlp = train_and_eval_mlp(
        seed=cfg["mlp"]["seed"], log_mlflow=False, save_model_path=str(mlp_path)
    )
    report["mlp"] = check_and_log(
        client,
        cfg["mlp"]["run_id"],
        mlp["pr_auc"],
        mlp["y_val"].astype(np.int64),
        mlp["val_probs"],
        np.asarray(mlp["val_amounts"]),
        mlp_path,
        not args.no_log,
        cfg["test_pr_auc_abs_tolerance"],
    )
    report["mlp"]["best_epoch"] = mlp["best_epoch"]
    report["mlp"]["seconds"] = time.perf_counter() - t0
    print("mlp:", report["mlp"], flush=True)

    Path(cfg["results_out"]).write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
