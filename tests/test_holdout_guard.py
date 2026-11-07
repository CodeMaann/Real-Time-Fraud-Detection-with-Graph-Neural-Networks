"""Tests for the holdout freeze and guard (docs/protocol.md §7). Temporary files only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from scripts.holdout_guard import GuardError, check_guard, write_freeze


def _setup(tmp_path: Path) -> dict[str, Any]:
    """A minimal holdout config with one GNN and one XGB 'checkpoint' and fake holdout files."""
    (tmp_path / "gnn.pt").write_bytes(b"gnn-weights")
    (tmp_path / "xgb.json").write_bytes(b"xgb-model")
    (tmp_path / "protocol.md").write_text("protocol", encoding="utf-8")
    raw = tmp_path / "holdout.parquet"
    graph = tmp_path / "test_graph.pt"
    raw.write_bytes(b"raw")
    graph.write_bytes(b"graph")
    from scripts.holdout_guard import sha256_file

    manifest = {
        "holdout_raw": {"path": raw.as_posix(), "sha256": sha256_file(raw)},
        "holdout_graphs": [{"path": graph.as_posix(), "sha256": sha256_file(graph)}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cfg = {
        "freeze_file": str(tmp_path / "freeze.json"),
        "results_out": str(tmp_path / "results.json"),
        "manifest_out": str(tmp_path / "manifest.json"),
        "protocol_file": str(tmp_path / "protocol.md"),
        "battery_splits": ["val", "test", "holdout"],
        "battery_file_pattern": str(tmp_path / "battery_{key}_{split}.json"),
        "models": [
            {"key": "gnn", "kind": "hetero_gat", "checkpoint": str(tmp_path / "gnn.pt")},
            {"key": "xgb", "kind": "xgb_label_free", "checkpoint": str(tmp_path / "xgb.json")},
        ],
    }
    return cfg


def _pass_batteries(tmp_path: Path) -> None:
    for split in ("val", "test", "holdout"):
        (tmp_path / f"battery_gnn_{split}.json").write_text(
            '{"gate_pass": true}', encoding="utf-8"
        )


def test_refuses_without_freeze(tmp_path: Path) -> None:
    cfg = _setup(tmp_path)
    with pytest.raises(GuardError, match="No freeze"):
        check_guard(cfg)


def test_freeze_requires_passing_batteries(tmp_path: Path) -> None:
    cfg = _setup(tmp_path)
    with pytest.raises(GuardError, match="no passing leakage battery"):
        write_freeze(cfg)
    (tmp_path / "battery_gnn_val.json").write_text('{"gate_pass": false}', encoding="utf-8")
    with pytest.raises(GuardError, match="no passing leakage battery"):
        write_freeze(cfg)
    assert not Path(cfg["freeze_file"]).exists()


def test_freeze_then_guard_passes_and_freeze_is_final(tmp_path: Path) -> None:
    cfg = _setup(tmp_path)
    _pass_batteries(tmp_path)
    freeze = write_freeze(cfg)
    assert {e["key"] for e in freeze["models"]} == {"gnn", "xgb"}
    assert check_guard(cfg)["models"] == freeze["models"]
    with pytest.raises(GuardError, match="already exists"):
        write_freeze(cfg)


def test_refuses_on_changed_checkpoint(tmp_path: Path) -> None:
    cfg = _setup(tmp_path)
    _pass_batteries(tmp_path)
    write_freeze(cfg)
    (tmp_path / "xgb.json").write_bytes(b"retrained-after-freeze")
    with pytest.raises(GuardError, match="xgb: checkpoint missing or sha256 differs"):
        check_guard(cfg)


def test_refuses_on_changed_holdout_or_protocol(tmp_path: Path) -> None:
    cfg = _setup(tmp_path)
    _pass_batteries(tmp_path)
    write_freeze(cfg)
    (tmp_path / "test_graph.pt").write_bytes(b"tampered")
    (tmp_path / "protocol.md").write_text("edited after freeze", encoding="utf-8")
    with pytest.raises(GuardError) as err:
        check_guard(cfg)
    assert "differs from the manifest" in str(err.value)
    assert "protocol.md changed" in str(err.value)


def test_refuses_to_overwrite_existing_result(tmp_path: Path) -> None:
    cfg = _setup(tmp_path)
    _pass_batteries(tmp_path)
    write_freeze(cfg)
    Path(cfg["results_out"]).write_text("{}", encoding="utf-8")
    with pytest.raises(GuardError, match="already been scored"):
        check_guard(cfg)


def test_refuses_model_missing_from_freeze(tmp_path: Path) -> None:
    cfg = _setup(tmp_path)
    _pass_batteries(tmp_path)
    write_freeze(cfg)
    (tmp_path / "late.pt").write_bytes(b"late")
    cfg["models"].append({"key": "late", "kind": "mlp", "checkpoint": str(tmp_path / "late.pt")})
    with pytest.raises(GuardError, match="late: not listed in the freeze"):
        check_guard(cfg)
