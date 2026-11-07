"""Copy the best weights out of a Colab resume-state file into models/ and verify them.

The resume state (``train_gnn`` checkpoint) holds the last finished epoch plus ``best_model``,
the weights of the best validation epoch. This script loads it with ``weights_only=True``
(only the numpy types used by the RNG state are allow-listed, so no arbitrary pickle code
runs), writes ``best_model`` to a NEW file that must not exist yet, and checks that the
weights load into the current architecture in eval mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from numpy._core.multiarray import _reconstruct  # type: ignore[attr-defined]

from fraudgnn.config import load_gnn_config
from fraudgnn.models.hetero_gat import HeteroGAT

SAFE_GLOBALS = [np.ndarray, _reconstruct, np.dtype]


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_state_safely(path: Path) -> dict[str, Any]:
    """Load a resume state without executing arbitrary pickle code."""
    dtype_classes = [type(np.dtype(t)) for t in ("uint32", "int64", "float64", "uint8")]
    with torch.serialization.safe_globals(SAFE_GLOBALS + dtype_classes):
        state: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    return state


def main(argv: list[str] | None = None) -> int:
    """Run the intake."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", required=True, help="Downloaded resume-state .pt file")
    parser.add_argument("--config", default="configs/intake.yaml")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(cfg["output"])
    if out.exists():
        raise FileExistsError(f"{out} exists; refusing to overwrite")

    src = Path(args.state)
    state = load_state_safely(src)
    history = state["history"]
    best = max(history, key=lambda h: h["val_pr_auc"])
    if abs(best["val_pr_auc"] - state["best_val_pr_auc"]) > 0:
        raise ValueError("best_val_pr_auc does not match the history maximum")

    gnn_cfg = load_gnn_config(cfg["gnn_config"])
    graph = torch.load(Path(cfg["graph_for_types"]), weights_only=False)
    model = HeteroGAT(
        node_types=graph.node_types,
        edge_types=graph.edge_types,
        in_channels_dict={nt: 17 for nt in graph.node_types},
        hidden_channels=gnn_cfg.model.hidden_dim,
        edge_dim=gnn_cfg.model.edge_dim,
        num_layers=gnn_cfg.model.num_layers,
        heads=gnn_cfg.model.heads,
        dropout=gnn_cfg.model.dropout,
        conv_type=gnn_cfg.model.conv_type,
        use_edge_attr=gnn_cfg.model.use_edge_attr,
        use_device_ip=gnn_cfg.model.use_device_ip,
    )
    keys = model.load_state_dict(state["best_model"], strict=True)
    model.eval()

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state["best_model"], out)
    reloaded = torch.load(out, map_location="cpu", weights_only=True)
    same = all(torch.equal(reloaded[k], state["best_model"][k]) for k in reloaded)

    report = {
        "source_state_file": src.name,  # file name only: no local absolute paths
        "source_sha256": sha256_file(src),
        "output": str(out),
        "output_sha256": sha256_file(out),
        "state_last_epoch": state["epoch"],
        "best_epoch": best["epoch"],
        "best_val_pr_auc": best["val_pr_auc"],
        "patience_counter": state["patience_counter"],
        "params": state["params"],
        "history": [{k: h[k] for k in ("epoch", "train_loss", "val_pr_auc")} for h in history],
        "loads_strict_eval_mode": not keys.missing_keys and not keys.unexpected_keys,
        "model_in_eval_mode": not model.training,
        "written_equals_state": same,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
    Path(cfg["report_out"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
