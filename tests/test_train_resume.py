"""Tests for device selection, per-epoch checkpointing, resume and the GPU timing mode.

Uses a tiny simulated dataset written to a temporary data folder, so no local data is needed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from fraudgnn.config import DataConfig, GNNConfig, GraphConfig, ModelConfig, TrainingConfig
from fraudgnn.data.simulator import TransactionSimulator
from fraudgnn.graph.builder import create_temporal_splits, save_graph_splits
from fraudgnn.training.train_gnn import (
    load_training_state,
    prepare_training,
    resolve_device,
    time_training_batches,
    train_and_evaluate_gnn,
    train_epoch,
)

EPOCHS = 3
STOP_AFTER = 2


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Simulate a small dataset and write raw parquet + graph splits like the real pipeline."""
    root = tmp_path_factory.mktemp("tiny_data")
    df = TransactionSimulator(
        DataConfig(
            num_transactions=2_000,
            num_days=10,
            num_users=100,
            num_merchants=20,
            num_devices=150,
            num_ips=200,
            fraud_ratio=0.03,
            seed=42,
        )
    ).generate()
    df = df.sort_values("timestamp", ignore_index=True)
    df.to_parquet(root / "transactions.parquet")
    save_graph_splits(create_temporal_splits(df, GraphConfig()), root / "processed")
    return root


def _config(data_root: Path, run_dir: Path, checkpoint: str) -> GNNConfig:
    return GNNConfig(
        data_dir=data_root / "processed",
        raw_data_path=data_root / "transactions.parquet",
        model=ModelConfig(
            model_type="hetero_gat",
            conv_type="pna",
            hidden_dim=16,
            num_layers=2,
            heads=2,
            dropout=0.2,
            edge_dim=27,
            use_device_ip=True,
        ),
        training=TrainingConfig(
            epochs=EPOCHS,
            batch_size=256,
            patience=10,
            loss_type="weighted_bce",
            num_neighbors=[5, 5],
            model_save_path=Path("best_tiny.pt"),
            best_model_dir=run_dir / "models",
            checkpoint_path=run_dir / checkpoint,
            mlflow_tracking_uri=(run_dir / "mlruns").as_uri(),
            mlflow_experiment_name="resume-test",
        ),
    )


def test_resolve_device() -> None:
    """cpu always works, auto never fails, unknown names are rejected."""
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in ("cpu", "cuda")
    with pytest.raises(ValueError):
        resolve_device("tpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            resolve_device("cuda")


def test_stop_and_resume_matches_uninterrupted_run(tiny_data: Path, tmp_path: Path) -> None:
    """Stopping after epoch 2 and resuming gives exactly the uninterrupted 3-epoch result."""
    straight_cfg = _config(tiny_data, tmp_path / "straight", "state.pt")
    straight = train_and_evaluate_gnn(straight_cfg, causal=True)
    assert straight["status"] == "finished"

    split_cfg = _config(tiny_data, tmp_path / "split", "state.pt")
    first = train_and_evaluate_gnn(split_cfg, causal=True, stop_after_epochs=STOP_AFTER)
    assert first["status"] == "interrupted"
    state = load_training_state(Path(str(split_cfg.training.checkpoint_path)))
    assert state["epoch"] == STOP_AFTER
    for key in ("model", "optimizer", "scheduler", "best_val_pr_auc", "patience_counter", "rng"):
        assert key in state

    # Simulate a fresh VM: the local best-model file is gone, only the state file remains.
    best_file = Path(split_cfg.training.best_model_dir) / "best_tiny_pna_causal.pt"
    best_file.unlink()
    resumed = train_and_evaluate_gnn(split_cfg, causal=True, resume=True)
    assert resumed["status"] == "finished"

    assert [h["epoch"] for h in resumed["history"]] == list(range(1, EPOCHS + 1))
    for a, b in zip(straight["history"], resumed["history"], strict=True):
        assert a["train_loss"] == b["train_loss"]
        assert a["val_pr_auc"] == b["val_pr_auc"]
    assert straight["test_metrics"].pr_auc == resumed["test_metrics"].pr_auc
    assert straight["threshold"] == resumed["threshold"]

    best_straight = torch.load(
        Path(straight_cfg.training.best_model_dir) / "best_tiny_pna_causal.pt", weights_only=True
    )
    best_resumed = torch.load(best_file, weights_only=True)
    assert all(torch.equal(best_straight[k], best_resumed[k]) for k in best_straight)
    # The resumed run continued the same MLflow run in the same tracking store.
    assert resumed["mlflow_run_id"] == state["mlflow_run_id"]


def test_resume_refuses_changed_settings(tiny_data: Path, tmp_path: Path) -> None:
    """A checkpoint written with other settings must not be resumed silently."""
    cfg = _config(tiny_data, tmp_path, "state.pt")
    train_and_evaluate_gnn(cfg, causal=True, stop_after_epochs=1)
    changed = cfg.model_copy(deep=True)
    changed.training.learning_rate = cfg.training.learning_rate * 2
    with pytest.raises(ValueError, match="different settings"):
        train_and_evaluate_gnn(changed, causal=True, resume=True)


def test_resume_without_checkpoint_starts_fresh(tiny_data: Path, tmp_path: Path) -> None:
    """--resume with no state file starts at epoch 1 instead of failing."""
    cfg = _config(tiny_data, tmp_path, "missing.pt")
    result = train_and_evaluate_gnn(cfg, causal=True, resume=True, stop_after_epochs=1)
    assert result["status"] == "interrupted"
    assert result["history"][0]["epoch"] == 1


def test_time_training_batches_reports_epoch_estimate(tiny_data: Path, tmp_path: Path) -> None:
    """Timing mode times real optimizer steps and extrapolates to one epoch."""
    cfg = _config(tiny_data, tmp_path, "unused.pt")
    device = torch.device("cpu")
    setup = prepare_training(cfg, device, causal=True)
    timing = time_training_batches(setup, cfg, device, n_batches=2, n_val_batches=1)
    assert timing["timed_train_steps"] == 2
    assert timing["epoch_seconds"] > 0
    assert timing["train_steps_per_epoch"] == len(setup.train_loader)
    assert not Path(str(cfg.training.checkpoint_path)).exists()
    assert pd.notna(timing["estimate_hours_for_max_epochs"])


def test_micro_batch_accumulation_matches_full_batch_step(tiny_data: Path, tmp_path: Path) -> None:
    """Four isolated micro-batches of 64 give the same optimizer step as one batch of 256."""
    device = torch.device("cpu")
    params_after: list[dict[str, torch.Tensor]] = []
    for micro in (None, 64):
        cfg = _config(tiny_data, tmp_path, "unused.pt")
        cfg.model.dropout = 0.0
        cfg.training.micro_batch_size = micro
        setup = prepare_training(cfg, device, causal=True)
        k = setup.accumulation_steps
        assert k == (1 if micro is None else 4)
        it = iter(setup.train_loader)
        train_epoch(
            setup.model,
            [next(it) for _ in range(k)],
            setup.optimizer,
            setup.loss_fn,
            cfg.training.gradient_clip_val,
            is_hetero=True,
            device=device,
            accumulation_steps=k,
        )
        params_after.append({n: p.detach().clone() for n, p in setup.model.named_parameters()})
    full, micro_run = params_after
    assert full.keys() == micro_run.keys()
    for name in full:
        assert torch.allclose(full[name], micro_run[name], atol=1e-6), name


def test_micro_batch_size_must_divide_batch_size(tiny_data: Path, tmp_path: Path) -> None:
    """A micro-batch size that does not divide the batch size is rejected."""
    cfg = _config(tiny_data, tmp_path, "unused.pt")
    cfg.training.micro_batch_size = 100
    with pytest.raises(ValueError, match="must divide"):
        prepare_training(cfg, torch.device("cpu"), causal=True)
