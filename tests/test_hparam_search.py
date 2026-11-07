"""Tests for the hyperparameter search and final-config writer (tiny simulated data)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from scripts import hparam_search
from scripts.hparam_search import mean_epoch_seconds, run_search, sample_configs, select_winner
from scripts.write_final_config import main as write_final_main

from fraudgnn.config import DataConfig, GNNConfig, GraphConfig, ModelConfig, TrainingConfig
from fraudgnn.data.simulator import TransactionSimulator
from fraudgnn.graph.builder import create_temporal_splits, save_graph_splits

T4 = [{"epoch": 1, "epoch_seconds": 60.0, "device_name": "Tesla T4"}]
SPACE = {
    "learning_rate": [0.001, 0.002],
    "hidden_dim": [8, 16],
    "dropout": [0.1],
    "weight_decay": [0.00001],
    "num_neighbors": [[5, 5]],
    "loss_type": ["weighted_bce"],
}


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Small simulated dataset written like the real pipeline."""
    root = tmp_path_factory.mktemp("search_data")
    df = (
        TransactionSimulator(
            DataConfig(
                num_transactions=2_000,
                num_days=10,
                num_users=100,
                num_merchants=20,
                num_devices=150,
                num_ips=200,
                fraud_ratio=0.03,
                seed=7,
            )
        )
        .generate()
        .sort_values("timestamp", ignore_index=True)
    )
    df.to_parquet(root / "transactions.parquet")
    save_graph_splits(create_temporal_splits(df, GraphConfig()), root / "processed")
    return root


def _setup(tiny_data: Path, tmp_path: Path) -> tuple[dict[str, Any], GNNConfig, dict[str, Any]]:
    search_cfg = {
        "seed": 42,
        "n_configs": 2,
        "space": SPACE,
        "budget": {
            "train_target_fraction": 0.5,
            "train_subset_seed": 42,
            "epochs": 1,
            "model_seed": 42,
        },
        "mlflow_experiment": "search-test",
        "selection": {"margin": 0.01},
    }
    base = GNNConfig(
        model=ModelConfig(
            model_type="hetero_gat",
            conv_type="pna",
            hidden_dim=8,
            num_layers=2,
            heads=2,
            edge_dim=27,
            use_device_ip=True,
        ),
        training=TrainingConfig(batch_size=256, loss_type="weighted_bce"),
    )
    paths = {
        "data_dir": str(tiny_data / "processed"),
        "raw_data_path": str(tiny_data / "transactions.parquet"),
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "best_model_dir": str(tmp_path / "models"),
        "tracking_uri": (tmp_path / "mlruns").as_uri(),
        "results_out": str(tmp_path / "hparam_search.json"),
    }
    return search_cfg, base, paths


def test_sampling_is_fixed_and_distinct() -> None:
    a = sample_configs(SPACE, 3, 42)
    assert a == sample_configs(SPACE, 3, 42)
    assert len({json.dumps(p, sort_keys=True) for p in a}) == 3


def test_search_runs_resumes_and_skips(
    tiny_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every configuration finishes on a training subset; a rerun trains nothing."""
    search_cfg, base, paths = _setup(tiny_data, tmp_path)
    results = run_search(search_cfg, base, paths, torch.device("cpu"))
    assert [e["status"] for e in results["configs"]] == ["finished", "finished"]
    assert all(0.0 <= e["best_val_pr_auc"] <= 1.0 for e in results["configs"])
    assert all(e["mean_epoch_seconds"] > 0 for e in results["configs"])
    winner, report = select_winner(results["configs"], 0.01)
    assert results["winner_index"] == winner["index"] == report["winner_index"]

    def must_not_train(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("finished configurations must be skipped")

    monkeypatch.setattr(hparam_search, "train_and_evaluate_gnn", must_not_train)
    again = run_search(search_cfg, base, paths, torch.device("cpu"))
    assert again["configs"] == results["configs"]


def test_search_refuses_a_different_sample(tiny_data: Path, tmp_path: Path) -> None:
    search_cfg, base, paths = _setup(tiny_data, tmp_path)
    Path(paths["results_out"]).write_text(
        json.dumps({"configs": [{"params": {"other": 1}}]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="different sample"):
        run_search(search_cfg, base, paths, torch.device("cpu"))


def test_final_config_requires_finished_search(tmp_path: Path) -> None:
    results = {
        "configs": [
            {
                "index": 0,
                "status": "finished",
                "best_val_pr_auc": 0.5,
                "mean_epoch_seconds": 60.0,
                "params": SPACE,
            },
            {"index": 1, "status": "pending", "params": SPACE},
        ]
    }
    res_file = tmp_path / "res.json"
    res_file.write_text(json.dumps(results), encoding="utf-8")
    with pytest.raises(SystemExit, match="not finished"):
        write_final_main(["--results", str(res_file), "--out", str(tmp_path / "final.yaml")])


def test_final_config_takes_winner_and_final_budget(tmp_path: Path) -> None:
    winner_params = {
        "learning_rate": 0.002,
        "hidden_dim": 128,
        "dropout": 0.1,
        "weight_decay": 0.0001,
        "num_neighbors": [15, 10],
        "loss_type": "focal",
    }
    results = {
        "configs": [
            {
                "index": 0,
                "status": "finished",
                "best_val_pr_auc": 0.40,
                "mean_epoch_seconds": 60.0,
                "params": SPACE,
                "history": T4,
            },
            {
                "index": 1,
                "status": "finished",
                "best_val_pr_auc": 0.45,
                "mean_epoch_seconds": 90.0,
                "params": winner_params,
                "history": T4,
            },
        ]
    }
    res_file = tmp_path / "res.json"
    res_file.write_text(json.dumps(results), encoding="utf-8")
    out = tmp_path / "final.yaml"
    write_final_main(["--results", str(res_file), "--out", str(out)])
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert doc["model"]["hidden_dim"] == 128 and doc["model"]["dropout"] == 0.1
    t = doc["training"]
    assert (t["learning_rate"], t["loss_type"], t["num_neighbors"]) == (0.002, "focal", [15, 10])
    assert (t["epochs"], t["patience"], t["train_target_fraction"]) == (30, 6, None)


def test_anchor_is_appended_after_sampled_configs() -> None:
    """The known-good anchor runs as the last configuration, after the sampled ones."""
    from scripts.hparam_search import search_points

    anchor = {
        "learning_rate": 0.001,
        "hidden_dim": 64,
        "dropout": 0.2,
        "weight_decay": 0.00001,
        "num_neighbors": [10, 10],
        "loss_type": "weighted_bce",
    }
    cfg = {"space": SPACE, "n_configs": 2, "seed": 42, "anchor": anchor}
    points = search_points(cfg)
    assert points[:2] == sample_configs(SPACE, 2, 42)
    assert points[2] == anchor and len(points) == 3


def _entry(
    index: int,
    val: float,
    seconds: float,
    device: str = "Tesla T4",
    hidden: int = 64,
    neighbors: list[int] | None = None,
) -> dict[str, Any]:
    params = {
        "learning_rate": 0.001,
        "hidden_dim": hidden,
        "dropout": 0.2,
        "weight_decay": 0.00001,
        "num_neighbors": neighbors or [10, 10],
        "loss_type": "weighted_bce",
    }
    history = [{"epoch": i, "epoch_seconds": seconds, "device_name": device} for i in (1, 2)]
    return {
        "index": index,
        "status": "finished",
        "best_val_pr_auc": val,
        "mean_epoch_seconds": seconds,
        "params": params,
        "history": history,
    }


COSTS = {"hidden64_nn10x10": 1.0, "hidden128_nn15x10": 3.07}


def test_rule_slower_config_inside_margin_loses() -> None:
    """Best score but slower, and the faster one is within 0.01: the faster one wins."""
    slow_best, fast_close = _entry(0, 0.800, 600.0), _entry(1, 0.795, 300.0)
    winner, report = select_winner([slow_best, fast_close], 0.01)
    assert winner is fast_close
    assert [e["index"] for e in report["within_margin"]] == [0, 1]
    assert "config 1" in report["reason"]


def test_rule_slower_config_outside_margin_wins() -> None:
    """The faster config is more than 0.01 below the best: the slower best one wins."""
    slow_best, fast_far = _entry(0, 0.800, 600.0), _entry(1, 0.780, 300.0)
    winner, report = select_winner([slow_best, fast_far], 0.01)
    assert winner is slow_best
    assert [e["index"] for e in report["within_margin"]] == [0]


def test_rule_equal_time_ties_go_to_lower_index() -> None:
    winner, _ = select_winner([_entry(3, 0.799, 300.0), _entry(1, 0.800, 300.0)], 0.01)
    assert winner["index"] == 1


def test_rule_needs_measured_epoch_times() -> None:
    assert mean_epoch_seconds([{"epoch_seconds": 100.0}, {"epoch_seconds": 300.0}]) == 200.0
    with pytest.raises(ValueError, match="epoch_seconds"):
        mean_epoch_seconds([{"epoch_seconds": 100.0}, {"val_pr_auc": 0.5}])


def test_fallback_mixed_devices_uses_cpu_ratio() -> None:
    """Mixed GPUs inside the margin: measured times are ignored, CPU relative cost decides.

    Config 0 looks faster on its measured time (300 s on an A100) but is the expensive
    hidden 128 / [15, 10] combination; config 1 ran on a T4 (400 s) and is the cheap one.
    """
    a = _entry(0, 0.800, 300.0, device="NVIDIA A100", hidden=128, neighbors=[15, 10])
    b = _entry(1, 0.795, 400.0, device="Tesla T4")
    winner, report = select_winner([a, b], 0.01, COSTS)
    assert winner is b
    assert "fallback" in report["ranking_basis"]
    assert report["devices_within_margin"] == ["NVIDIA A100", "Tesla T4"]


def test_fallback_when_one_config_switched_gpu_mid_run() -> None:
    """A single configuration with epochs on two GPUs also triggers the fallback."""
    a = _entry(0, 0.800, 300.0, hidden=128, neighbors=[15, 10])
    a["history"][1]["device_name"] = "NVIDIA L4"
    b = _entry(1, 0.795, 400.0)
    winner, report = select_winner([a, b], 0.01, COSTS)
    assert winner is b and "fallback" in report["ranking_basis"]


def test_same_device_keeps_measured_time() -> None:
    """All epochs inside the margin on one GPU: the measured time decides."""
    a = _entry(0, 0.800, 300.0, hidden=128, neighbors=[15, 10])
    b = _entry(1, 0.795, 400.0)
    winner, report = select_winner([a, b], 0.01, COSTS)
    assert winner is a
    assert report["ranking_basis"] == "measured mean epoch time on Tesla T4"
