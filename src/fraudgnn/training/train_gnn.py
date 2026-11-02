"""End-to-end training and evaluation script for Graph Neural Network fraud detectors.

Supports:
1. HeteroGAT (relational heterogeneous message passing with GATConv/SAGEConv)
2. GraphSAGE (homogeneous edge classifier)
3. Strict inductive temporal splitting with zero future edge leakage
4. Early stopping on validation PR-AUC with learning rate scheduler
5. Decision threshold selection exclusively on validation, applied to test
6. Full MLflow tracking: parameters, per-epoch metrics, per-fraud-type recall, model artifacts
7. --device cpu|cuda|auto (CPU default), per-epoch checkpoint and --resume, --time-batches
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Configure MLflow environment before imports
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from torch_geometric.data import Data, HeteroData

from fraudgnn.config import GNNConfig, load_gnn_config
from fraudgnn.graph.loader import HomogeneousLinkLoader, LinkNeighborLoader
from fraudgnn.graph.temporal_loader import (
    CausalHomogeneousLinkLoader,
    CausalLinkNeighborLoader,
)
from fraudgnn.models.hetero_gat import HeteroGAT
from fraudgnn.models.losses import get_loss_fn
from fraudgnn.models.sage import GraphSAGEEdgeClassifier, convert_hetero_to_homogeneous
from fraudgnn.training.evaluate import compute_metrics


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Set random seeds across Python, NumPy, and PyTorch (CPU and all GPUs).

    Args:
        seed: Seed value.
        deterministic: Also force deterministic cuDNN kernels. GPU scatter/aggregation in
            message passing can still use non-deterministic atomics, so CUDA runs are
            reproducible only up to floating-point summation order.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    # Batch shapes change every step in GNN training, so cuDNN autotuning would not pay off.
    torch.backends.cudnn.benchmark = False


TARGET_REL = ("user", "transacts_at", "merchant")


def _forward_batch(
    model: nn.Module, batch: Any, is_hetero: bool, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move one batch to ``device`` and run the model.

    Returns:
        (logits on ``device``, labels on CPU).
    """
    if is_hetero:
        assert isinstance(batch, HeteroData)
        store = batch[TARGET_REL]
        raw_attr = getattr(store, "edge_attr", None)
        raw_dev = getattr(store, "device_index", None)
        raw_ip = getattr(store, "ip_index", None)
        logits = model(
            x_dict={nt: batch[nt].x.to(device) for nt in batch.node_types},
            edge_index_dict={et: batch[et].edge_index.to(device) for et in batch.edge_types},
            edge_label_index=store.edge_label_index.to(device),
            edge_attr=raw_attr.to(device) if raw_attr is not None else None,
            device_index=raw_dev.to(device) if raw_dev is not None else None,
            ip_index=raw_ip.to(device) if raw_ip is not None else None,
        )
        return logits, store.edge_label

    assert isinstance(batch, Data)
    raw_attr_h = getattr(batch, "edge_attr", None)
    logits = model(
        x=batch.x.to(device),
        edge_index=batch.edge_index.to(device),
        edge_label_index=batch.edge_label_index.to(device),
        edge_attr=raw_attr_h.to(device) if raw_attr_h is not None else None,
    )
    return logits, batch.edge_label


def _optimizer_step(
    model: nn.Module,
    micro_batches: list[Any],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    gradient_clip_val: float,
    is_hetero: bool,
    device: torch.device,
) -> float:
    """One optimizer step over one or more micro-batches (gradient accumulation).

    Each micro-batch loss is weighted by its share of the step's targets, so the
    accumulated gradient equals the gradient of the mean loss over all targets of the step.
    Activations of one micro-batch are freed before the next one runs.

    Returns:
        Mean loss over the step's targets.
    """
    optimizer.zero_grad()
    sizes = [int(_num_targets(b, is_hetero)) for b in micro_batches]
    total = sum(sizes)
    step_loss = 0.0
    for batch, size in zip(micro_batches, sizes, strict=True):
        logits, labels = _forward_batch(model, batch, is_hetero, device)
        loss = loss_fn(logits, labels.to(device))
        share = size / total
        (loss * share).backward()
        step_loss += float(loss.item()) * share

    if gradient_clip_val > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_val)

    optimizer.step()
    return step_loss


def _num_targets(batch: Any, is_hetero: bool) -> int:
    """Number of target edges (labels) in a batch."""
    labels = batch[TARGET_REL].edge_label if is_hetero else batch.edge_label
    return int(labels.numel())


def train_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    gradient_clip_val: float = 1.0,
    is_hetero: bool = True,
    device: torch.device = torch.device("cpu"),
    accumulation_steps: int = 1,
) -> float:
    """Run one training epoch.

    Args:
        model: GNN model module.
        loader: Batch loader (LinkNeighborLoader or HomogeneousLinkLoader).
        optimizer: PyTorch optimizer.
        loss_fn: Binary classification loss function.
        gradient_clip_val: Max grad norm for gradient clipping.
        is_hetero: True for HeteroGAT, False for GraphSAGE.
        device: Target torch device.
        accumulation_steps: Consecutive loader batches per optimizer step. With
            ``accumulation_steps=k`` and a loader of micro-batches of size ``m`` each step
            covers ``k * m`` targets, with the same gradient as one batch of that size when
            targets are isolated (only dropout masks differ).

    Returns:
        Average training loss over all optimizer steps.
    """
    model.train()
    total_loss = 0.0
    num_steps = 0
    group: list[Any] = []

    for batch in loader:
        group.append(batch)
        if len(group) == accumulation_steps:
            total_loss += _optimizer_step(
                model, group, optimizer, loss_fn, gradient_clip_val, is_hetero, device
            )
            num_steps += 1
            group = []
    if group:
        total_loss += _optimizer_step(
            model, group, optimizer, loss_fn, gradient_clip_val, is_hetero, device
        )
        num_steps += 1

    return total_loss / max(1, num_steps)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: Any,
    is_hetero: bool = True,
    device: torch.device = torch.device("cpu"),
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run model inference to compute fraud probabilities and total latency.

    Args:
        model: Evaluated GNN model.
        loader: Batch loader for evaluation partition.
        is_hetero: True for HeteroGAT, False for GraphSAGE.
        device: Target torch device.

    Returns:
        Tuple of (y_true_array, y_prob_array, total_elapsed_seconds).
    """
    model.eval()
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    t0 = time.perf_counter()

    for batch in loader:
        logits, edge_label = _forward_batch(model, batch, is_hetero, device)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(edge_label.cpu().numpy())

    elapsed = time.perf_counter() - t0

    y_prob = np.concatenate(all_probs) if all_probs else np.empty(0, dtype=np.float32)
    y_true = np.concatenate(all_targets) if all_targets else np.empty(0, dtype=np.float32)

    return y_true, y_prob, elapsed


CHECKPOINT_FORMAT_VERSION = 1


def resolve_device(name: str) -> torch.device:
    """Map a ``--device`` choice (cpu, cuda, auto) to a torch device.

    Raises:
        RuntimeError: If ``cuda`` is requested but no CUDA device is available.
        ValueError: For an unknown choice.
    """
    choice = name.lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown device '{name}'. Use cpu, cuda or auto.")


@dataclass
class TrainingSetup:
    """Everything the epoch loop needs: data loaders, model, optimizer and run settings."""

    model: nn.Module
    train_loader: Any
    val_loader: Any
    test_loader: Any
    is_hetero: bool
    loss_fn: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau
    val_amounts: np.ndarray
    val_fraud_types: np.ndarray
    test_amounts: np.ndarray
    test_fraud_types: np.ndarray
    epochs: int
    patience: int
    conv_type: str
    causal: bool
    best_checkpoint_path: Path
    accumulation_steps: int = 1
    params: dict[str, Any] = field(default_factory=dict)


def prepare_training(
    config: GNNConfig,
    device: torch.device,
    subset_fraction: float | None = None,
    override_epochs: int | None = None,
    override_use_edge_attr: bool | None = None,
    override_num_hops: int | None = None,
    override_conv_type: str | None = None,
    override_patience: int | None = None,
    causal: bool = False,
    override_use_device_ip: bool | None = None,
) -> TrainingSetup:
    """Load graph splits, build loaders, model, loss, optimizer and scheduler.

    Args:
        config: Loaded GNNConfig instance.
        device: Device for the model and loss (loaders always produce CPU batches).
        subset_fraction: Optional fraction of transactions to use (for rapid verification).
        override_epochs: Optional epoch count override.
        override_use_edge_attr: Optional flag to toggle edge features (ablation).
        override_num_hops: Optional flag to toggle 1 vs 2 hops (ablation).
        override_conv_type: Optional convolution type override ('gat', 'sum', 'pna', 'sage').
        override_patience: Optional early-stopping patience override.
        causal: Whether to enforce strict causal temporal sampling (no future leakage).
        override_use_device_ip: Optional flag to toggle device/ip embedding decoder binding.

    Returns:
        A ready TrainingSetup.
    """
    seed_everything(config.training.seed, deterministic=config.training.deterministic)

    use_edge_attr = (
        override_use_edge_attr
        if override_use_edge_attr is not None
        else config.model.use_edge_attr
    )
    use_device_ip = (
        override_use_device_ip
        if override_use_device_ip is not None
        else config.model.use_device_ip
    )
    num_hops = override_num_hops if override_num_hops is not None else config.model.num_layers
    epochs = override_epochs if override_epochs is not None else config.training.epochs
    conv_type = (
        override_conv_type.lower()
        if override_conv_type is not None
        else config.model.conv_type.lower()
    )
    patience = override_patience if override_patience is not None else config.training.patience

    num_neighbors = (
        [10] * num_hops
        if override_num_hops is not None
        else config.training.num_neighbors[:num_hops]
    )

    print("=" * 70, flush=True)
    print(f"  TRAINING GNN MODEL: {config.model.model_type.upper()}", flush=True)
    print(
        f"  Configuration : hops={num_hops}, conv={conv_type}, edge_attr={use_edge_attr}, epochs={epochs}, patience={patience}",
        flush=True,
    )
    print(f"  Device        : {device}", flush=True)
    print("=" * 70, flush=True)

    # 1. Load Temporal Graph Splits
    data_dir = Path(config.data_dir)
    print(f"Loading graph splits from: {data_dir}...", flush=True)
    train_graph: HeteroData = torch.load(data_dir / "train_graph.pt", weights_only=False)
    val_graph: HeteroData = torch.load(data_dir / "val_graph.pt", weights_only=False)
    test_graph: HeteroData = torch.load(data_dir / "test_graph.pt", weights_only=False)

    # Load raw transactions to extract amounts and hidden fraud_types for test slicing
    df_raw = pd.read_parquet(config.raw_data_path)

    target_rel = ("user", "transacts_at", "merchant")

    # Extract target evaluation subsets
    # Train: all edges in train_graph are strictly <= train_cutoff
    tr_edge_index = train_graph[target_rel].edge_index
    tr_edge_label = train_graph[target_rel].edge_label
    tr_edge_attr = train_graph[target_rel].edge_attr
    tr_raw_time = getattr(train_graph[target_rel], "edge_time", None)
    tr_edge_time = (
        tr_raw_time
        if tr_raw_time is not None
        else torch.zeros(tr_edge_index.size(1), dtype=torch.long)
    )

    # Val: edges strictly where val_mask is True
    val_mask = val_graph[target_rel].val_mask
    val_edge_index = val_graph[target_rel].edge_index[:, val_mask]
    val_edge_label = val_graph[target_rel].edge_label[val_mask]
    val_edge_attr = val_graph[target_rel].edge_attr[val_mask]
    val_raw_time = getattr(val_graph[target_rel], "edge_time", None)
    val_edge_time = (
        val_raw_time[val_mask]
        if val_raw_time is not None
        else torch.zeros(val_edge_index.size(1), dtype=torch.long)
    )

    # Test: edges strictly where test_mask is True
    test_mask = test_graph[target_rel].test_mask
    test_edge_index = test_graph[target_rel].edge_index[:, test_mask]
    test_edge_label = test_graph[target_rel].edge_label[test_mask]
    test_edge_attr = test_graph[target_rel].edge_attr[test_mask]
    test_raw_time = getattr(test_graph[target_rel], "edge_time", None)
    test_edge_time = (
        test_raw_time[test_mask]
        if test_raw_time is not None
        else torch.zeros(test_edge_index.size(1), dtype=torch.long)
    )

    # Align amounts and fraud types for test evaluation
    val_timestamps = pd.to_datetime(df_raw["timestamp"])
    t_min = val_timestamps.min()
    t_max = val_timestamps.max()
    dur = t_max - t_min
    train_cut = t_min + dur * 0.60
    val_cut = t_min + dur * 0.80

    df_val = df_raw[(val_timestamps > train_cut) & (val_timestamps <= val_cut)].reset_index(
        drop=True
    )
    df_test = df_raw[val_timestamps > val_cut].reset_index(drop=True)

    val_amounts = df_val["amount"].to_numpy()
    val_fraud_types = df_val["fraud_type"].to_numpy()

    test_amounts = df_test["amount"].to_numpy()
    test_fraud_types = df_test["fraud_type"].to_numpy()

    # Device and IP indices extraction
    tr_device_index = (
        train_graph[target_rel].device_index
        if hasattr(train_graph[target_rel], "device_index")
        and train_graph[target_rel].device_index is not None
        else train_graph["user", "uses_device", "device"].edge_index[1]
    )
    tr_ip_index = (
        train_graph[target_rel].ip_index
        if hasattr(train_graph[target_rel], "ip_index")
        and train_graph[target_rel].ip_index is not None
        else train_graph["user", "uses_ip", "ip"].edge_index[1]
    )

    val_device_index = (
        val_graph[target_rel].device_index[val_mask]
        if hasattr(val_graph[target_rel], "device_index")
        and val_graph[target_rel].device_index is not None
        else val_graph["user", "uses_device", "device"].edge_index[1][val_mask]
    )
    val_ip_index = (
        val_graph[target_rel].ip_index[val_mask]
        if hasattr(val_graph[target_rel], "ip_index")
        and val_graph[target_rel].ip_index is not None
        else val_graph["user", "uses_ip", "ip"].edge_index[1][val_mask]
    )

    test_device_index = (
        test_graph[target_rel].device_index[test_mask]
        if hasattr(test_graph[target_rel], "device_index")
        and test_graph[target_rel].device_index is not None
        else test_graph["user", "uses_device", "device"].edge_index[1][test_mask]
    )
    test_ip_index = (
        test_graph[target_rel].ip_index[test_mask]
        if hasattr(test_graph[target_rel], "ip_index")
        and test_graph[target_rel].ip_index is not None
        else test_graph["user", "uses_ip", "ip"].edge_index[1][test_mask]
    )

    if subset_fraction is not None and 0.0 < subset_fraction < 1.0:
        n_tr_sub = max(100, int(tr_edge_index.size(1) * subset_fraction))
        tr_edge_index = tr_edge_index[:, :n_tr_sub]
        tr_edge_label = tr_edge_label[:n_tr_sub]
        tr_edge_attr = tr_edge_attr[:n_tr_sub]
        tr_edge_time = tr_edge_time[:n_tr_sub]
        tr_device_index = tr_device_index[:n_tr_sub]
        tr_ip_index = tr_ip_index[:n_tr_sub]

        n_val_sub = max(50, int(val_edge_index.size(1) * subset_fraction))
        val_edge_index = val_edge_index[:, :n_val_sub]
        val_edge_label = val_edge_label[:n_val_sub]
        val_edge_attr = val_edge_attr[:n_val_sub]
        val_edge_time = val_edge_time[:n_val_sub]
        val_device_index = val_device_index[:n_val_sub]
        val_ip_index = val_ip_index[:n_val_sub]
        val_amounts = val_amounts[:n_val_sub]
        val_fraud_types = val_fraud_types[:n_val_sub]

        n_te_sub = max(50, int(test_edge_index.size(1) * subset_fraction))
        test_edge_index = test_edge_index[:, :n_te_sub]
        test_edge_label = test_edge_label[:n_te_sub]
        test_edge_attr = test_edge_attr[:n_te_sub]
        test_edge_time = test_edge_time[:n_te_sub]
        test_device_index = test_device_index[:n_te_sub]
        test_ip_index = test_ip_index[:n_te_sub]
        test_amounts = test_amounts[:n_te_sub]
        test_fraud_types = test_fraud_types[:n_te_sub]
        print(
            f"Subset active: {subset_fraction * 100:.1f}% -> train {n_tr_sub:,}, val {n_val_sub:,}, test {n_te_sub:,}"
        )

    # Build Loaders and Models based on architecture
    is_hetero = config.model.model_type == "hetero_gat"
    batch_size = config.training.batch_size
    # Training loader yields micro-batches; accumulation_steps of them form one optimizer
    # step of batch_size targets (bounded memory, same targets per step).
    micro_batch_size = config.training.micro_batch_size or batch_size
    if batch_size % micro_batch_size != 0:
        raise ValueError(
            f"micro_batch_size ({micro_batch_size}) must divide batch_size ({batch_size})"
        )
    accumulation_steps = batch_size // micro_batch_size
    train_loader: Any = None
    val_loader: Any = None
    test_loader: Any = None

    if is_hetero:
        if causal:
            print("Instantiating CausalLinkNeighborLoader (strict causal sampling)...")
            train_loader = CausalLinkNeighborLoader(
                data=train_graph,
                num_neighbors=num_neighbors,
                edge_label_index=(target_rel, tr_edge_index),
                edge_label=tr_edge_label,
                edge_time=tr_edge_time,
                edge_attr=tr_edge_attr if use_edge_attr else None,
                device_index=tr_device_index if use_device_ip else None,
                ip_index=tr_ip_index if use_device_ip else None,
                batch_size=micro_batch_size,
                shuffle=True,
                seed=config.training.seed,
            )
            val_loader = CausalLinkNeighborLoader(
                data=val_graph,
                num_neighbors=num_neighbors,
                edge_label_index=(target_rel, val_edge_index),
                edge_label=val_edge_label,
                edge_time=val_edge_time,
                edge_attr=val_edge_attr if use_edge_attr else None,
                device_index=val_device_index if use_device_ip else None,
                ip_index=val_ip_index if use_device_ip else None,
                batch_size=batch_size,
                shuffle=False,
                seed=config.training.seed,
            )
            test_loader = CausalLinkNeighborLoader(
                data=test_graph,
                num_neighbors=num_neighbors,
                edge_label_index=(target_rel, test_edge_index),
                edge_label=test_edge_label,
                edge_time=test_edge_time,
                edge_attr=test_edge_attr if use_edge_attr else None,
                device_index=test_device_index if use_device_ip else None,
                ip_index=test_ip_index if use_device_ip else None,
                batch_size=batch_size,
                shuffle=False,
                seed=config.training.seed,
            )
        else:
            train_loader = LinkNeighborLoader(
                data=train_graph,
                num_neighbors=num_neighbors,
                edge_label_index=(target_rel, tr_edge_index),
                edge_label=tr_edge_label,
                edge_attr=tr_edge_attr if use_edge_attr else None,
                batch_size=micro_batch_size,
                shuffle=True,
                seed=config.training.seed,
            )
            val_loader = LinkNeighborLoader(
                data=val_graph,
                num_neighbors=num_neighbors,
                edge_label_index=(target_rel, val_edge_index),
                edge_label=val_edge_label,
                edge_attr=val_edge_attr if use_edge_attr else None,
                batch_size=batch_size,
                shuffle=False,
                seed=config.training.seed,
            )
            test_loader = LinkNeighborLoader(
                data=test_graph,
                num_neighbors=num_neighbors,
                edge_label_index=(target_rel, test_edge_index),
                edge_label=test_edge_label,
                edge_attr=test_edge_attr if use_edge_attr else None,
                batch_size=batch_size,
                shuffle=False,
                seed=config.training.seed,
            )

        if causal:
            in_channels_dict = {nt: 17 for nt in train_graph.node_types}
        else:
            in_channels_dict = {
                nt: int(train_graph[nt].x.size(1)) for nt in train_graph.node_types
            }
        model: nn.Module = HeteroGAT(
            node_types=train_graph.node_types,
            edge_types=train_graph.edge_types,
            in_channels_dict=in_channels_dict,
            hidden_channels=config.model.hidden_dim,
            edge_dim=config.model.edge_dim,
            num_layers=num_hops,
            heads=config.model.heads,
            dropout=config.model.dropout,
            conv_type=conv_type,
            use_edge_attr=use_edge_attr,
            use_device_ip=use_device_ip,
        ).to(device)

    else:
        # Homogeneous GraphSAGE
        print("Converting HeteroData splits to homogeneous graphs...")
        train_homo, _, tr_h_idx, tr_h_attr = convert_hetero_to_homogeneous(train_graph)
        val_homo, _, val_h_idx, val_h_attr = convert_hetero_to_homogeneous(val_graph)
        test_homo, _, test_h_idx, test_h_attr = convert_hetero_to_homogeneous(test_graph)

        if subset_fraction is not None and 0.0 < subset_fraction < 1.0:
            tr_h_idx = tr_h_idx[:, : len(tr_edge_label)]
            val_h_idx = val_h_idx[:, val_mask][:, : len(val_edge_label)]
            test_h_idx = test_h_idx[:, test_mask][:, : len(test_edge_label)]
            tr_h_attr = tr_h_attr[: len(tr_edge_label)] if tr_h_attr is not None else None
            val_h_attr = (
                val_h_attr[val_mask][: len(val_edge_label)] if val_h_attr is not None else None
            )
            test_h_attr = (
                test_h_attr[test_mask][: len(test_edge_label)] if test_h_attr is not None else None
            )
        else:
            val_h_idx = val_h_idx[:, val_mask]
            test_h_idx = test_h_idx[:, test_mask]
            val_h_attr = val_h_attr[val_mask] if val_h_attr is not None else None
            test_h_attr = test_h_attr[test_mask] if test_h_attr is not None else None

        if causal:
            print("Instantiating CausalHomogeneousLinkLoader (strict causal sampling)...")
            train_loader = CausalHomogeneousLinkLoader(
                data=train_homo,
                edge_label_index=tr_h_idx,
                edge_label=tr_edge_label,
                edge_time=tr_edge_time,
                edge_attr=tr_h_attr if use_edge_attr else None,
                batch_size=micro_batch_size,
                num_neighbors=num_neighbors,
                shuffle=True,
                seed=config.training.seed,
            )
            val_loader = CausalHomogeneousLinkLoader(
                data=val_homo,
                edge_label_index=val_h_idx,
                edge_label=val_edge_label,
                edge_time=val_edge_time,
                edge_attr=val_h_attr if use_edge_attr else None,
                batch_size=batch_size,
                num_neighbors=num_neighbors,
                shuffle=False,
                seed=config.training.seed,
            )
            test_loader = CausalHomogeneousLinkLoader(
                data=test_homo,
                edge_label_index=test_h_idx,
                edge_label=test_edge_label,
                edge_time=test_edge_time,
                edge_attr=test_h_attr if use_edge_attr else None,
                batch_size=batch_size,
                num_neighbors=num_neighbors,
                shuffle=False,
                seed=config.training.seed,
            )
        else:
            train_loader = HomogeneousLinkLoader(
                data=train_homo,
                edge_label_index=tr_h_idx,
                edge_label=tr_edge_label,
                edge_attr=tr_h_attr if use_edge_attr else None,
                batch_size=micro_batch_size,
                num_neighbors=num_neighbors,
                shuffle=True,
                seed=config.training.seed,
            )
            val_loader = HomogeneousLinkLoader(
                data=val_homo,
                edge_label_index=val_h_idx,
                edge_label=val_edge_label,
                edge_attr=val_h_attr if use_edge_attr else None,
                batch_size=batch_size,
                num_neighbors=num_neighbors,
                shuffle=False,
                seed=config.training.seed,
            )
            test_loader = HomogeneousLinkLoader(
                data=test_homo,
                edge_label_index=test_h_idx,
                edge_label=test_edge_label,
                edge_attr=test_h_attr if use_edge_attr else None,
                batch_size=batch_size,
                num_neighbors=num_neighbors,
                shuffle=False,
                seed=config.training.seed,
            )

        model = GraphSAGEEdgeClassifier(
            in_channels=int(train_homo.x.size(1)),
            hidden_channels=config.model.hidden_dim,
            edge_dim=config.model.edge_dim,
            num_layers=num_hops,
            dropout=config.model.dropout,
            use_edge_attr=use_edge_attr,
        ).to(device)

    # Class balance and Loss setup
    n_pos = int((tr_edge_label == 1).sum().item())
    n_neg = int((tr_edge_label == 0).sum().item())
    computed_pos_weight = (
        float(n_neg / max(1, n_pos))
        if config.training.pos_weight is None
        else config.training.pos_weight
    )

    # The loss holds pos_weight as a buffer, so it must live on the model's device.
    loss_fn = get_loss_fn(
        loss_type=config.training.loss_type,
        pos_weight=computed_pos_weight if config.training.loss_type != "focal" else None,
        gamma=config.training.focal_gamma,
        alpha=config.training.focal_alpha,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.training.lr_scheduler_factor,
        patience=config.training.lr_scheduler_patience,
    )

    save_stem = Path(config.training.model_save_path).stem
    if causal:
        conv_tag = f"_{conv_type}" if conv_type not in save_stem else ""
        best_checkpoint_path = Path(config.training.best_model_dir) / (
            f"{save_stem}{conv_tag}_causal.pt"
        )
    else:
        best_checkpoint_path = Path(config.training.model_save_path)

    params = {
        "model_type": config.model.model_type,
        "conv_type": conv_type,
        "causal": causal,
        # Hetero causal loaders isolate each target subgraph (no batch-sharing leak).
        "isolate_targets": causal and is_hetero,
        "hidden_dim": config.model.hidden_dim,
        "num_layers": num_hops,
        "use_edge_attr": use_edge_attr,
        "use_device_ip": use_device_ip,
        "epochs": epochs,
        "batch_size": batch_size,
        "micro_batch_size": micro_batch_size,
        "learning_rate": config.training.learning_rate,
        "loss_type": config.training.loss_type,
        "patience": patience,
        "seed": config.training.seed,
        "pos_weight": computed_pos_weight,
        "subset_fraction": subset_fraction,
    }

    return TrainingSetup(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        is_hetero=is_hetero,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        val_amounts=val_amounts,
        val_fraud_types=val_fraud_types,
        test_amounts=test_amounts,
        test_fraud_types=test_fraud_types,
        epochs=epochs,
        patience=patience,
        conv_type=conv_type,
        causal=causal,
        best_checkpoint_path=best_checkpoint_path,
        accumulation_steps=accumulation_steps,
        params=params,
    )


def _sync(device: torch.device) -> None:
    """Wait for queued GPU work so wall-clock timings are real."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_training_batches(
    setup: TrainingSetup,
    config: GNNConfig,
    device: torch.device,
    n_batches: int,
    n_val_batches: int | None = None,
) -> dict[str, Any]:
    """Time ``n_batches`` optimizer steps and ``n_val_batches`` (default: same) val batches.

    One optimizer step covers ``batch_size`` targets (``accumulation_steps`` micro-batches).
    The first step is reported separately (CUDA context, kernel selection and allocator
    warm-up) and excluded from the per-step estimate when more than one step is timed.
    Nothing is saved and nothing is logged to MLflow.

    Returns:
        Timing breakdown including the estimated seconds per epoch.
    """
    k = setup.accumulation_steps
    steps_per_epoch = math.ceil(len(setup.train_loader) / k)
    step_seconds: list[float] = []
    loader_iter = iter(setup.train_loader)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(min(n_batches, steps_per_epoch)):
        t0 = time.perf_counter()
        micro = [b for _, b in zip(range(k), loader_iter, strict=False)]
        train_epoch(
            setup.model,
            micro,
            setup.optimizer,
            setup.loss_fn,
            config.training.gradient_clip_val,
            is_hetero=setup.is_hetero,
            device=device,
            accumulation_steps=k,
        )
        _sync(device)
        step_seconds.append(time.perf_counter() - t0)

    val_iter = iter(setup.val_loader)
    n_val = min(n_val_batches or n_batches, len(setup.val_loader))
    t0 = time.perf_counter()
    evaluate_model(
        setup.model,
        (next(val_iter) for _ in range(n_val)),
        is_hetero=setup.is_hetero,
        device=device,
    )
    _sync(device)
    val_seconds = time.perf_counter() - t0

    steady = step_seconds[1:] if len(step_seconds) > 1 else step_seconds
    sec_step = float(np.mean(steady))
    sec_val = val_seconds / max(1, n_val)
    epoch_train = sec_step * steps_per_epoch
    epoch_val = sec_val * len(setup.val_loader)
    result: dict[str, Any] = {
        "device": str(device),
        "timed_train_steps": len(step_seconds),
        "micro_batches_per_step": k,
        "first_step_seconds": step_seconds[0],
        "sec_per_train_step": sec_step,
        "sec_per_val_batch": sec_val,
        "train_steps_per_epoch": steps_per_epoch,
        "val_batches_per_epoch": len(setup.val_loader),
        "epoch_train_seconds": epoch_train,
        "epoch_val_seconds": epoch_val,
        "epoch_seconds": epoch_train + epoch_val,
        "max_epochs": setup.epochs,
        "estimate_hours_for_max_epochs": (epoch_train + epoch_val) * setup.epochs / 3600.0,
    }
    if device.type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(device)
        result["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
    return result


def _rng_state(train_loader: Any) -> dict[str, Any]:
    """Capture every random state that influences the next epoch."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    loader_rng = getattr(train_loader, "rng", None)
    if isinstance(loader_rng, np.random.Generator):
        state["train_loader"] = loader_rng.bit_generator.state
    return state


def _restore_rng_state(state: dict[str, Any], train_loader: Any) -> None:
    """Restore random states captured by ``_rng_state``."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    loader_rng = getattr(train_loader, "rng", None)
    if "train_loader" in state and isinstance(loader_rng, np.random.Generator):
        loader_rng.bit_generator.state = state["train_loader"]


def save_training_state(path: Path, state: dict[str, Any]) -> None:
    """Write the resume state atomically (temp file, then rename) so a crash cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_training_state(path: Path) -> dict[str, Any]:
    """Load a resume state written by ``save_training_state`` (our own file, full pickle)."""
    state: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format in {path}")
    return state


def _start_mlflow_run(run_name: str, previous_run_id: str | None) -> tuple[Any, bool]:
    """Continue the previous MLflow run if this tracking store has it, else start a new one.

    Returns:
        (active run, whether the previous run was continued).
    """
    if previous_run_id is not None:
        try:
            MlflowClient().get_run(previous_run_id)
            return mlflow.start_run(run_id=previous_run_id), True
        except MlflowException:
            pass
    return mlflow.start_run(run_name=run_name), False


def train_and_evaluate_gnn(
    config: GNNConfig,
    subset_fraction: float | None = None,
    override_epochs: int | None = None,
    override_use_edge_attr: bool | None = None,
    override_num_hops: int | None = None,
    override_conv_type: str | None = None,
    override_patience: int | None = None,
    causal: bool = False,
    override_use_device_ip: bool | None = None,
    device: torch.device | None = None,
    resume: bool = False,
    stop_after_epochs: int | None = None,
) -> dict[str, Any]:
    """Train GNN model, perform early stopping on validation PR-AUC, and evaluate on test set.

    After every epoch the full training state (model, optimizer, scheduler, epoch, best
    validation PR-AUC, patience counter, history, best weights and random states) is written
    to ``config.training.checkpoint_path`` when it is set. With ``resume=True`` training
    continues from that state at the next epoch.

    Args:
        config: Loaded GNNConfig instance.
        subset_fraction: Optional fraction of transactions to use (for rapid verification).
        override_epochs: Optional epoch count override.
        override_use_edge_attr: Optional flag to toggle edge features (ablation).
        override_num_hops: Optional flag to toggle 1 vs 2 hops (ablation).
        override_conv_type: Optional convolution type override ('gat', 'sum', 'pna', 'sage').
        override_patience: Optional early-stopping patience override.
        causal: Whether to enforce strict causal temporal sampling (no future leakage).
        override_use_device_ip: Optional flag to toggle device/ip embedding decoder binding.
        device: Torch device (default CPU).
        resume: Continue from ``config.training.checkpoint_path`` if it exists.
        stop_after_epochs: Stop (as if interrupted) after this epoch number, after saving the
            resume state. Used to split long runs into sessions and in tests.

    Returns:
        Dictionary of final test and validation metrics, or ``status="interrupted"``.
    """
    device = device or torch.device("cpu")
    setup = prepare_training(
        config,
        device,
        subset_fraction=subset_fraction,
        override_epochs=override_epochs,
        override_use_edge_attr=override_use_edge_attr,
        override_num_hops=override_num_hops,
        override_conv_type=override_conv_type,
        override_patience=override_patience,
        causal=causal,
        override_use_device_ip=override_use_device_ip,
    )
    model, epochs, patience = setup.model, setup.epochs, setup.patience
    ckpt_path = Path(config.training.checkpoint_path) if config.training.checkpoint_path else None
    best_checkpoint_path = setup.best_checkpoint_path
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_pr_auc = -1.0
    patience_counter = 0
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    start_epoch = 1
    previous_run_id: str | None = None

    if resume and ckpt_path is not None and ckpt_path.exists():
        state = load_training_state(ckpt_path)
        if state["params"] != setup.params:
            raise ValueError(
                "Checkpoint was written with different settings; refusing to resume.\n"
                f"checkpoint: {state['params']}\ncurrent:    {setup.params}"
            )
        model.load_state_dict(state["model"])
        setup.optimizer.load_state_dict(state["optimizer"])
        setup.scheduler.load_state_dict(state["scheduler"])
        best_val_pr_auc = state["best_val_pr_auc"]
        patience_counter = state["patience_counter"]
        history = state["history"]
        best_state = state["best_model"]
        start_epoch = state["epoch"] + 1
        previous_run_id = state["mlflow_run_id"]
        _restore_rng_state(state["rng"], setup.train_loader)
        if best_state is not None and not best_checkpoint_path.exists():
            torch.save(best_state, best_checkpoint_path)
        print(
            f"Resumed from {ckpt_path}: epoch {state['epoch']} done, best val PR-AUC "
            f"{best_val_pr_auc:.4f}, patience {patience_counter}/{patience}",
            flush=True,
        )
    elif resume:
        print(f"--resume: no checkpoint at {ckpt_path}; starting from epoch 1.", flush=True)

    mlflow.set_tracking_uri(config.training.mlflow_tracking_uri)
    mlflow.set_experiment(config.training.mlflow_experiment_name)
    run_name = (
        f"{config.model.model_type}_{setup.conv_type}_{'causal' if causal else 'non_causal'}"
    )
    active_run, continued = _start_mlflow_run(run_name, previous_run_id)
    run_id: str = active_run.info.run_id
    if not continued:
        mlflow.set_tag("sampling", "causal" if causal else "non_causal_preliminary")
        mlflow.set_tag("conv_type", setup.conv_type)
        mlflow.set_tag("device", str(device))
        mlflow.log_params(setup.params)
        if previous_run_id is not None:
            # The original run lives in another tracking store (e.g. a lost Colab VM):
            # record the link and replay the per-epoch history so the curves are complete.
            mlflow.set_tag("resumed_from_run", previous_run_id)
            for h in history:
                mlflow.log_metrics(
                    {k: float(v) for k, v in h.items() if k != "epoch"}, step=int(h["epoch"])
                )

    early_stopped = patience_counter >= patience
    actual_epochs_run = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        if early_stopped:
            break
        actual_epochs_run = epoch
        t_epoch_start = time.perf_counter()
        tr_loss = train_epoch(
            model=model,
            loader=setup.train_loader,
            optimizer=setup.optimizer,
            loss_fn=setup.loss_fn,
            gradient_clip_val=config.training.gradient_clip_val,
            is_hetero=setup.is_hetero,
            device=device,
            accumulation_steps=setup.accumulation_steps,
        )

        # Evaluate on validation
        y_val_true, y_val_prob, _ = evaluate_model(
            model=model, loader=setup.val_loader, is_hetero=setup.is_hetero, device=device
        )

        val_metrics = compute_metrics(
            y_true=y_val_true,
            y_pred_proba=y_val_prob,
            amounts=setup.val_amounts,
            review_cost=config.cost.review_cost,
            default_fraud_amount=config.cost.default_fraud_amount,
            fraud_types=setup.val_fraud_types,
        )

        setup.scheduler.step(val_metrics.pr_auc)
        epoch_time = time.perf_counter() - t_epoch_start

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] - "
            f"Train Loss: {tr_loss:.4f} | "
            f"Val PR-AUC: {val_metrics.pr_auc:.4f} | "
            f"Val F1: {val_metrics.f1:.4f} (thresh={val_metrics.best_threshold:.4f}) | "
            f"Val Recall@1%FPR: {val_metrics.recall_at_1pct_fpr:.4f}",
            flush=True,
        )

        epoch_metrics = {
            "train_loss": tr_loss,
            "val_pr_auc": val_metrics.pr_auc,
            "val_roc_auc": val_metrics.roc_auc,
            "val_f1": val_metrics.f1,
            "val_recall_at_1pct_fpr": val_metrics.recall_at_1pct_fpr,
            "epoch_seconds": epoch_time,
        }
        mlflow.log_metrics(epoch_metrics, step=epoch)
        history.append({"epoch": epoch, **epoch_metrics})

        # Early stopping check on validation PR-AUC
        if val_metrics.pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_metrics.pr_auc
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, best_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                early_stopped = True
                print(
                    f"Early stopping triggered at epoch {epoch} (patience: {patience})",
                    flush=True,
                )

        if ckpt_path is not None:
            save_training_state(
                ckpt_path,
                {
                    "format_version": CHECKPOINT_FORMAT_VERSION,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": setup.optimizer.state_dict(),
                    "scheduler": setup.scheduler.state_dict(),
                    "best_val_pr_auc": best_val_pr_auc,
                    "patience_counter": patience_counter,
                    "history": history,
                    "best_model": best_state,
                    "rng": _rng_state(setup.train_loader),
                    "params": setup.params,
                    "mlflow_run_id": run_id,
                },
            )

        if stop_after_epochs is not None and epoch >= stop_after_epochs and not early_stopped:
            print(f"Stopping after epoch {epoch} (--stop-after-epochs); resume with --resume.")
            mlflow.end_run(status="KILLED")
            return {"status": "interrupted", "epochs_run": epoch, "history": history}

    if best_state is None:
        raise RuntimeError("No epoch finished; nothing to evaluate.")

    # Final Evaluation: Load best weights
    print(f"\nLoading best weights (saved at {best_checkpoint_path})...")
    model.load_state_dict(best_state)

    # Re-evaluate validation to confirm optimal threshold
    y_val_true, y_val_prob, _ = evaluate_model(
        model=model, loader=setup.val_loader, is_hetero=setup.is_hetero, device=device
    )
    val_eval = compute_metrics(
        y_true=y_val_true,
        y_pred_proba=y_val_prob,
        amounts=setup.val_amounts,
        review_cost=config.cost.review_cost,
        default_fraud_amount=config.cost.default_fraud_amount,
        fraud_types=setup.val_fraud_types,
    )
    val_selected_threshold = val_eval.best_threshold

    # Evaluate Test Set STRICTLY using validation threshold
    y_test_true, y_test_prob, total_eval_time = evaluate_model(
        model=model, loader=setup.test_loader, is_hetero=setup.is_hetero, device=device
    )
    latency_ms_per_sample = (total_eval_time / max(1, len(y_test_true))) * 1000.0

    test_eval = compute_metrics(
        y_true=y_test_true,
        y_pred_proba=y_test_prob,
        amounts=setup.test_amounts,
        review_cost=config.cost.review_cost,
        default_fraud_amount=config.cost.default_fraud_amount,
        threshold_override=val_selected_threshold,
        fraud_types=setup.test_fraud_types,
        compute_ci=True,
    )

    # Log final metrics to MLflow
    mlflow.log_metrics(
        {
            "test_pr_auc": test_eval.pr_auc,
            "test_pr_auc_ci_lower": test_eval.pr_auc_ci_lower,
            "test_pr_auc_ci_upper": test_eval.pr_auc_ci_upper,
            "test_roc_auc": test_eval.roc_auc,
            "test_f1": test_eval.f1,
            "test_recall_at_1pct_fpr": test_eval.recall_at_1pct_fpr,
            "test_business_cost": test_eval.total_cost,
            "val_selected_threshold": val_selected_threshold,
            "latency_ms_per_sample": latency_ms_per_sample,
            "epochs_trained": float(actual_epochs_run),
        }
    )
    for ft, rec in test_eval.per_type_recall.items():
        mlflow.log_metric(f"test_recall_{ft}", rec)

    mlflow.log_artifact(str(best_checkpoint_path), artifact_path="model")
    mlflow.end_run()

    # Report results
    print("\n" + "=" * 70, flush=True)
    print(f"  {config.model.model_type.upper()} FINAL TEST RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(f"Validation PR-AUC (Primary) : {val_eval.pr_auc:.4f}", flush=True)
    print(f"Validation Selected Thresh  : {val_selected_threshold:.4f}", flush=True)
    print("-" * 70, flush=True)
    print(
        f"TEST PR-AUC (Primary)       : {test_eval.pr_auc:.4f} [95% CI: {test_eval.pr_auc_ci_lower:.4f} - {test_eval.pr_auc_ci_upper:.4f}]",
        flush=True,
    )
    print(f"TEST ROC-AUC                : {test_eval.roc_auc:.4f}", flush=True)
    print(f"TEST Recall @ 1% FPR        : {test_eval.recall_at_1pct_fpr:.4f}", flush=True)
    print(f"TEST F1 @ Val Threshold     : {test_eval.f1:.4f}", flush=True)
    print(f"TEST Business Cost          : ${test_eval.total_cost:,.2f}", flush=True)
    print(f"Inference Latency           : {latency_ms_per_sample:.4f} ms/sample", flush=True)
    print("-" * 70, flush=True)
    print("  PER-FRAUD-TYPE TEST RECALL:", flush=True)
    for ft, rec in sorted(test_eval.per_type_recall.items()):
        print(f"    - {ft:<24s}: {rec:.4f}", flush=True)
    print("=" * 70 + "\n", flush=True)

    return {
        "status": "finished",
        "val_metrics": val_eval,
        "test_metrics": test_eval,
        "latency_ms": latency_ms_per_sample,
        "threshold": val_selected_threshold,
        "epochs_run": actual_epochs_run,
        "history": history,
        "mlflow_run_id": run_id,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI Entry point for GNN training."""
    parser = argparse.ArgumentParser(description="Train Graph Neural Network for fraud detection.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hetero_gat.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--subset-fraction",
        type=float,
        default=None,
        help="Fraction of data for quick sanity test (e.g. 0.1 for fast run)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--no-edge-attr",
        action="store_true",
        help="Ablation: Disable edge attributes in decoder",
    )
    parser.add_argument(
        "--num-hops",
        type=int,
        default=None,
        choices=[1, 2],
        help="Ablation: Number of message passing hops (1 or 2)",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Enable strict causal temporal sampling (zero within-window future leakage)",
    )
    parser.add_argument(
        "--conv-type",
        type=str,
        default=None,
        choices=["gat", "sum", "pna", "sage"],
        help="Override convolution type: gat, sum, pna, or sage",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override early stopping patience",
    )
    parser.add_argument(
        "--no-device-ip",
        action="store_true",
        help="Ablation: Disable device and IP node binding in decoder",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Compute device (default cpu; auto = cuda if available)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from training.checkpoint_path if it exists",
    )
    parser.add_argument(
        "--stop-after-epochs",
        type=int,
        default=None,
        help="Save the resume state and stop after this epoch number",
    )
    parser.add_argument(
        "--time-batches",
        type=int,
        default=None,
        help="Timing mode: time N train and N val batches, print seconds per epoch, no training run",
    )
    args = parser.parse_args(argv)

    config = load_gnn_config(args.config)
    device = resolve_device(args.device)
    overrides: dict[str, Any] = {
        "subset_fraction": args.subset_fraction,
        "override_epochs": args.epochs,
        "override_use_edge_attr": False if args.no_edge_attr else None,
        "override_num_hops": args.num_hops,
        "override_conv_type": args.conv_type,
        "override_patience": args.patience,
        "causal": args.causal,
        "override_use_device_ip": False if args.no_device_ip else None,
    }
    if args.time_batches is not None:
        setup = prepare_training(config, device, **overrides)
        timing = time_training_batches(setup, config, device, args.time_batches)
        print(json.dumps(timing, indent=2), flush=True)
        return 0

    train_and_evaluate_gnn(
        config=config,
        device=device,
        resume=args.resume,
        stop_after_epochs=args.stop_after_epochs,
        **overrides,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
