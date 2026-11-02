"""Time HeteroGAT training steps with the fixed loader to estimate a retraining run.

Thin wrapper around ``fraudgnn.training.train_gnn.time_training_batches`` (the same code path
as ``train_gnn --time-batches``): builds the real training setup (isolated causal loaders,
model, loss, optimizer), times ``train_batches`` training steps and ``val_batches``
validation batches, and writes the estimate to JSON. Nothing is trained to completion,
saved or logged to MLflow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from fraudgnn.config import load_gnn_config
from fraudgnn.training.train_gnn import prepare_training, resolve_device, time_training_batches


def main(argv: list[str] | None = None) -> int:
    """Run the timing and write the estimate."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/timing.yaml")
    args = parser.parse_args(argv)

    cfg_t = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config = load_gnn_config(cfg_t["gnn_config"])
    device = resolve_device(cfg_t["device"])
    setup = prepare_training(
        config,
        device,
        override_epochs=cfg_t["epochs"],
        override_conv_type=cfg_t["conv_type"],
        causal=True,
        override_use_device_ip=bool(cfg_t["use_device_ip"]),
    )
    result = time_training_batches(
        setup, config, device, cfg_t["train_batches"], cfg_t["val_batches"]
    )
    result["config"] = cfg_t
    Path(cfg_t["results_out"]).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
