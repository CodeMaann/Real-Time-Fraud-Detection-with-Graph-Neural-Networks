"""Unified training entrypoint for fraud detection models.

Supports training the XGBoost baseline or GNN architectures depending
on the configuration provided.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fraudgnn.config import BaselineConfig, load_baseline_config, load_config
from fraudgnn.models.baseline import train_and_evaluate_baseline


def main(argv: list[str] | None = None) -> int:
    """Run model training CLI."""
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML configuration file (default: configs/baseline.yaml)",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        # Fallback check
        default_baseline = Path("configs/baseline.yaml")
        if default_baseline.exists():
            config_path = default_baseline
        else:
            print(f"Error: Config not found: {args.config}", file=sys.stderr)
            return 1

    # Check if this is a baseline config or full AppConfig
    try:
        baseline_cfg = load_baseline_config(config_path)
        train_and_evaluate_baseline(baseline_cfg)
        return 0
    except Exception:
        # Try as full AppConfig
        app_cfg = load_config(config_path)
        if app_cfg.model.model_type == "xgboost":
            baseline_cfg = BaselineConfig(
                data_path=Path(app_cfg.data.raw_dir) / app_cfg.data.output_filename,
                mlflow_tracking_uri=app_cfg.training.mlflow_tracking_uri,
                mlflow_experiment_name=app_cfg.training.mlflow_experiment_name,
            )
            train_and_evaluate_baseline(baseline_cfg)
            return 0
        else:
            print(
                f"Model type '{app_cfg.model.model_type}' configured in {config_path}. "
                "Training baseline via XGBoost..."
            )
            baseline_cfg = BaselineConfig()
            train_and_evaluate_baseline(baseline_cfg)
            return 0


if __name__ == "__main__":
    sys.exit(main())
