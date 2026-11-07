"""Register the Colab-trained HeteroGAT as a new run in the local MLflow store.

The Colab run's MLflow database stayed on the Colab VM. This creates a NEW local run (never
touching existing runs) from the downloaded resume-state file: training params, the real
per-epoch history (train loss, validation PR-AUC), provenance tags and the best weights as an
artifact. Idempotent: if a run with the same state-file hash exists, it is reused.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import yaml
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.intake_colab_checkpoint import load_state_safely


def main(argv: list[str] | None = None) -> int:
    """Create (or find) the run and print its id."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", required=True, help="Downloaded resume-state .pt file")
    parser.add_argument("--config", default="configs/register_colab.yaml")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    state_path = Path(args.state)
    state_sha = hashlib.sha256(state_path.read_bytes()).hexdigest()
    client = MlflowClient(tracking_uri=cfg["tracking_uri"])
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    exp = mlflow.set_experiment(cfg["experiment"])
    existing = client.search_runs(
        [exp.experiment_id], filter_string=f"tags.colab_state_sha256 = '{state_sha}'"
    )
    if existing:
        print(existing[0].info.run_id)
        return 0

    state = load_state_safely(state_path)
    history = state["history"]
    best = max(history, key=lambda h: h["val_pr_auc"])
    with mlflow.start_run(run_name=cfg["run_name"]) as run:
        mlflow.log_params(state["params"])
        for h in history:
            mlflow.log_metrics(
                {
                    k: float(h[k])
                    for k in ("train_loss", "val_pr_auc", "val_roc_auc", "val_f1")
                    if k in h
                },
                step=int(h["epoch"]),
            )
        mlflow.log_metrics(
            {
                "best_val_pr_auc": float(state["best_val_pr_auc"]),
                "epochs_trained": float(state["epoch"]),
            }
        )
        mlflow.set_tags(
            {
                "sampling": "causal",
                "conv_type": state["params"]["conv_type"],
                "training_batches": "isolated",
                "trained_on": cfg["trained_on"],
                "colab_state_sha256": state_sha,
                "colab_mlflow_run_id": str(state.get("mlflow_run_id")),
                "best_epoch": str(best["epoch"]),
                "last_epoch": str(state["epoch"]),
                "stopped_note": cfg["stopped_note"],
            }
        )
        mlflow.log_artifact(cfg["best_model_file"], artifact_path="model")
        print(run.info.run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
