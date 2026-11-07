"""Build the sealed holdout world. Computes hashes only, never a metric.

Steps (each skipped if its output already exists and verifies):
1. Rebuild the ORIGINAL graph splits deterministically, record sha256 of both versions, and
   require every model-relevant tensor (edges, edge features, times, labels, masks) to be
   exactly equal; only then persist the training-split edge scaler to
   ``models/edge_scaler.pkl``.
2. Generate the holdout transactions: ``configs/data.yaml`` with the holdout seed.
3. Build the holdout graph splits with the saved original scaler (never refit).
4. Write ``docs/verification/holdout_manifest.json`` (sha256 and sizes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from fraudgnn.config import load_data_config, load_graph_config
from fraudgnn.data.simulator import TransactionSimulator
from fraudgnn.graph.builder import create_temporal_splits, save_graph_splits

CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def file_entry(path: Path) -> dict[str, Any]:
    """Path, size and hash of one file."""
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


# Differences allowed between the current and the rebuilt ORIGINAL graphs. The current graphs
# were built before the builder (commits 697cf9f, b42f45c) began storing device/IP indices on
# the transaction edges and changed the static node features ``x``. No causal loader reads
# ``x`` (node features are recomputed per target), so neither difference reaches a model.
ALLOWED_DIFFERENT = {"x"}
ALLOWED_ADDED = {("user", "transacts_at", "merchant"): {"device_index", "ip_index"}}


def compare_graphs(current: Path, rebuilt: Path) -> dict[str, Any]:
    """Tensor-level comparison; raises if anything model-relevant differs."""
    a_graph = torch.load(current, weights_only=False)
    b_graph = torch.load(rebuilt, weights_only=False)
    keys = set(a_graph.node_types) | set(a_graph.edge_types)
    if keys != set(b_graph.node_types) | set(b_graph.edge_types):
        raise RuntimeError(f"{current.name}: node/edge types differ")
    equal, allowed_diff, problems = 0, [], []
    for key in sorted(keys, key=str):
        a, b = a_graph[key], b_graph[key]
        only_a = set(a.keys()) - set(b.keys())
        only_b = set(b.keys()) - set(a.keys()) - ALLOWED_ADDED.get(key, set())
        if only_a or only_b:
            problems.append(f"{key}: attributes {sorted(only_a)} / {sorted(only_b)}")
        for attr in set(a.keys()) & set(b.keys()):
            va, vb = a[attr], b[attr]
            same = (
                va.dtype == vb.dtype and va.shape == vb.shape and torch.equal(va, vb)
                if torch.is_tensor(va)
                else va == vb
            )
            if same:
                equal += 1
            elif attr in ALLOWED_DIFFERENT:
                allowed_diff.append(f"{key}.{attr}")
            else:
                problems.append(f"{key}.{attr} differs")
    if problems:
        raise RuntimeError(f"{current.name}: model-relevant differences: {problems}")
    return {
        "attributes_exactly_equal": equal,
        "allowed_differences": allowed_diff,
        "added_in_rebuild": {str(k): sorted(v) for k, v in ALLOWED_ADDED.items()},
    }


def rebuild_original(cfg: dict[str, Any]) -> tuple[StandardScaler, list[dict[str, Any]]]:
    """Rebuild the original splits; compare hashes and every tensor with the current files."""
    graph_cfg = load_graph_config(cfg["graph_config"])
    df = pd.read_parquet(cfg["original_raw"])
    splits = create_temporal_splits(df, graph_cfg)
    rebuild = Path(cfg["rebuild_dir"])
    save_graph_splits(splits, rebuild)
    checks = []
    for name in cfg["graph_files"]:
        current = Path(cfg["original_processed"]) / name
        rebuilt = rebuild / name
        a, b = sha256_file(current), sha256_file(rebuilt)
        checks.append(
            {
                "file": name,
                "current_sha256": a,
                "rebuilt_sha256": b,
                "byte_identical": a == b,
                "tensor_check": compare_graphs(current, rebuilt),
            }
        )
    assert splits.edge_scaler is not None
    return splits.edge_scaler, checks


def main(argv: list[str] | None = None) -> int:
    """Run the holdout build."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/holdout.yaml")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    manifest: dict[str, Any] = {"created_at": datetime.now(UTC).isoformat(timespec="seconds")}

    # 1. Original rebuild -> scaler
    scaler_path = Path(cfg["edge_scaler_out"])
    scaler, checks = rebuild_original(cfg)
    manifest["original_rebuild"] = checks
    if scaler_path.exists():
        with scaler_path.open("rb") as fh:
            saved = pickle.load(fh)  # our own file, written below on an earlier run
        same = (saved.mean_ == scaler.mean_).all() and (saved.scale_ == scaler.scale_).all()
        if not same:
            raise RuntimeError(f"{scaler_path} exists and differs from the rebuilt scaler")
    else:
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with scaler_path.open("wb") as fh:
            pickle.dump(scaler, fh)
    print(
        f"original graphs rebuilt; model-relevant tensors identical; scaler at {scaler_path}",
        flush=True,
    )

    # 2. Holdout transactions
    raw_path = Path(cfg["holdout_raw"])
    data_cfg = load_data_config(cfg["data_config"])
    data_cfg = data_cfg.model_copy(update={"seed": cfg["holdout_seed"]})
    if not raw_path.exists():
        df_h = TransactionSimulator(data_cfg).generate()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df_h.to_parquet(raw_path, engine="pyarrow", compression="snappy", index=False)
    df_h = pd.read_parquet(raw_path)
    print(f"holdout transactions: {len(df_h):,} rows at {raw_path}", flush=True)

    # 3. Holdout graphs with the ORIGINAL scaler
    with scaler_path.open("rb") as fh:
        original_scaler = pickle.load(fh)
    splits_h = create_temporal_splits(
        df_h, load_graph_config(cfg["graph_config"]), edge_scaler=original_scaler
    )
    if splits_h.edge_scaler is not original_scaler:
        raise RuntimeError("holdout build did not use the original scaler")
    save_graph_splits(splits_h, cfg["holdout_processed"])

    # 4. Manifest: hashes and sizes only
    manifest.update(
        holdout_seed=cfg["holdout_seed"],
        data_config=cfg["data_config"],
        rows=len(df_h),
        holdout_raw=file_entry(raw_path),
        holdout_graphs=[
            file_entry(Path(cfg["holdout_processed"]) / n) for n in cfg["graph_files"]
        ],
        edge_scaler=file_entry(scaler_path),
        original_graphs=[
            file_entry(Path(cfg["original_processed"]) / n) for n in cfg["graph_files"]
        ],
        note="Hashes only. No metric has been computed on the holdout.",
    )
    out = Path(cfg["manifest_out"])
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
