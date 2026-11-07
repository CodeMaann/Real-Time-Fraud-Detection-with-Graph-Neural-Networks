"""Leakage checks on the REAL validation graph (skipped when local data is absent).

The synthetic-fixture tests cannot catch a leak that only appears on real data, so these
tests run small versions of ``scripts/leakage_battery.py`` against data/processed. The model
is randomly initialised (seeded): the checks concern the data pipeline, not the weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from scripts.leakage_battery import (
    build_model,
    check_alignment,
    check_brute_force,
    check_injection,
    check_ties_and_own_edges,
    load_context,
)

from fraudgnn.graph.builder import extract_entity_mappings

DATA_FILES = [
    Path("data/raw/transactions.parquet"),
    Path("data/processed/val_graph.pt"),
    Path("data/processed/test_graph.pt"),
]
MODEL_CFG = {
    "gnn_config": "configs/hetero_gat_colab.yaml",
    "conv_type": "pna",
    "use_device_ip": True,
}
NUM_NEIGHBORS = [10, 10]
SEED = 42

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in DATA_FILES), reason="local real data not available"
)


@pytest.fixture(scope="module")
def ctx() -> dict[str, Any]:
    """Raw rows, graphs and targets for val and test."""
    return load_context("configs/evaluate.yaml")


@pytest.fixture(scope="module")
def model(ctx: dict[str, Any]) -> torch.nn.Module:
    """Randomly initialised HeteroGAT with the audited architecture."""
    torch.manual_seed(SEED)
    return build_model(MODEL_CFG, ctx["val"]["graph"], weights=False)


def test_brute_force_reference_matches_loader(ctx: dict[str, Any], model: torch.nn.Module) -> None:
    """Deleting all edges at/after each target's time reproduces the loader exactly."""
    val = ctx["val"]
    rng = np.random.default_rng(SEED)
    positions = np.sort(rng.choice(val["tgt"]["edge_index"].size(1), 10, replace=False))
    result = check_brute_force(model, val["graph"], val["tgt"], NUM_NEIGHBORS, positions, 2048)
    assert result["structure_mismatches"] == 0, result
    assert result["max_score_abs_diff"] < 1e-5, result


def test_future_injection_real_data(ctx: dict[str, Any], model: torch.nn.Module) -> None:
    """Earlier targets are bit-identical after injecting later transactions; later ones move."""
    val = ctx["val"]
    result = check_injection(
        model, val["graph"], val["tgt"], NUM_NEIGHBORS, 256, [1, 2, 64, 256], 1000.0
    )
    for bs in (1, 2, 64, 256):
        assert result[f"bs{bs}"]["early_bit_identical"], result
        assert result[f"bs{bs}"]["late_changed"] > 0, result


def test_no_own_or_later_edges_in_subgraphs(ctx: dict[str, Any]) -> None:
    """No sampled edge has time >= its target's time; the target's own edges are absent."""
    val = ctx["val"]
    result = check_ties_and_own_edges(
        val["graph"], val["tgt"], val["rows"], val["history"], NUM_NEIGHBORS, 2048, limit=4096
    )
    assert result["edges_at_or_after_target_time"] == 0, result
    assert result["own_edges_present"] == 0, result


def test_device_ip_alignment_val_and_test(ctx: dict[str, Any]) -> None:
    """Each target's user, merchant, card, device, IP, time and label match its raw row."""
    mappings = extract_entity_mappings(ctx["df"])
    for name, mask in (("val", "val_mask"), ("test", "test_mask")):
        result = check_alignment(ctx[name]["graph"], mask, ctx[name]["rows"], mappings)
        assert all(v == 0 for k, v in result.items() if k != "rows"), (name, result)


@pytest.mark.skipif(
    not Path("data/holdout/processed/test_graph.pt").exists(), reason="holdout not built"
)
def test_holdout_targets_align_with_holdout_rows() -> None:
    """The sealed holdout graph matches its own raw rows (integrity only; no model, no metric)."""
    import yaml
    from scripts.leakage_battery import check_alignment, load_split

    cfg = yaml.safe_load(Path("configs/leakage_battery.yaml").read_text(encoding="utf-8"))
    data = load_split(cfg, "holdout")
    result = check_alignment(
        data["graph"], data["mask_name"], data["rows"], extract_entity_mappings(data["df"])
    )
    assert result["rows"] > 0
    assert all(v == 0 for k, v in result.items() if k != "rows"), result
