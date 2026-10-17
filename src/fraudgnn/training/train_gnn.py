"""End-to-end training and evaluation script for Graph Neural Network fraud detectors.

Supports:
1. HeteroGAT (relational heterogeneous message passing with GATConv/SAGEConv)
2. GraphSAGE (homogeneous edge classifier)
3. Strict inductive temporal splitting with zero future edge leakage
4. Early stopping on validation PR-AUC with learning rate scheduler
5. Decision threshold selection exclusively on validation, applied to test
6. Full MLflow tracking: parameters, per-epoch metrics, per-fraud-type recall, model artifacts
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
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


def seed_everything(seed: int) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    gradient_clip_val: float = 1.0,
    is_hetero: bool = True,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Run one training epoch over mini-batches.

    Args:
        model: GNN model module.
        loader: Batch loader (LinkNeighborLoader or HomogeneousLinkLoader).
        optimizer: PyTorch optimizer.
        loss_fn: Binary classification loss function.
        gradient_clip_val: Max grad norm for gradient clipping.
        is_hetero: True for HeteroGAT, False for GraphSAGE.
        device: Target torch device.

    Returns:
        Average training loss over all batches.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        optimizer.zero_grad()
        if is_hetero:
            assert isinstance(batch, HeteroData)
            target_rel = ("user", "transacts_at", "merchant")
            edge_label_index = batch[target_rel].edge_label_index.to(device)
            edge_label = batch[target_rel].edge_label.to(device)
            raw_attr = getattr(batch[target_rel], "edge_attr", None)
            edge_attr = raw_attr.to(device) if raw_attr is not None else None

            x_dict = {nt: batch[nt].x.to(device) for nt in batch.node_types}
            edge_index_dict = {et: batch[et].edge_index.to(device) for et in batch.edge_types}

            logits = model(
                x_dict=x_dict,
                edge_index_dict=edge_index_dict,
                edge_label_index=edge_label_index,
                edge_attr=edge_attr,
            )
        else:
            assert isinstance(batch, Data)
            edge_label_index = batch.edge_label_index.to(device)
            edge_label = batch.edge_label.to(device)
            raw_attr_h = getattr(batch, "edge_attr", None)
            edge_attr = raw_attr_h.to(device) if raw_attr_h is not None else None

            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)

            logits = model(
                x=x,
                edge_index=edge_index,
                edge_label_index=edge_label_index,
                edge_attr=edge_attr,
            )

        loss = loss_fn(logits, edge_label)
        loss.backward()

        if gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_val)

        optimizer.step()
        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(1, num_batches)


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
        if is_hetero:
            assert isinstance(batch, HeteroData)
            target_rel = ("user", "transacts_at", "merchant")
            edge_label_index = batch[target_rel].edge_label_index.to(device)
            edge_label = batch[target_rel].edge_label
            raw_attr = getattr(batch[target_rel], "edge_attr", None)
            edge_attr = raw_attr.to(device) if raw_attr is not None else None

            x_dict = {nt: batch[nt].x.to(device) for nt in batch.node_types}
            edge_index_dict = {et: batch[et].edge_index.to(device) for et in batch.edge_types}

            logits = model(
                x_dict=x_dict,
                edge_index_dict=edge_index_dict,
                edge_label_index=edge_label_index,
                edge_attr=edge_attr,
            )
        else:
            assert isinstance(batch, Data)
            edge_label_index = batch.edge_label_index.to(device)
            edge_label = batch.edge_label
            raw_attr_h = getattr(batch, "edge_attr", None)
            edge_attr = raw_attr_h.to(device) if raw_attr_h is not None else None

            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)

            logits = model(
                x=x,
                edge_index=edge_index,
                edge_label_index=edge_label_index,
                edge_attr=edge_attr,
            )

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(edge_label.cpu().numpy())

    elapsed = time.perf_counter() - t0

    y_prob = np.concatenate(all_probs) if all_probs else np.empty(0, dtype=np.float32)
    y_true = np.concatenate(all_targets) if all_targets else np.empty(0, dtype=np.float32)

    return y_true, y_prob, elapsed


def train_and_evaluate_gnn(
    config: GNNConfig,
    subset_fraction: float | None = None,
    override_epochs: int | None = None,
    override_use_edge_attr: bool | None = None,
    override_num_hops: int | None = None,
    override_conv_type: str | None = None,
    override_patience: int | None = None,
    causal: bool = False,
) -> dict[str, Any]:
    """Train GNN model, perform early stopping on validation PR-AUC, and evaluate on test set.

    Args:
        config: Loaded GNNConfig instance.
        subset_fraction: Optional fraction of transactions to use (for rapid verification).
        override_epochs: Optional epoch count override.
        override_use_edge_attr: Optional flag to toggle edge features (ablation).
        override_num_hops: Optional flag to toggle 1 vs 2 hops (ablation).
        override_conv_type: Optional convolution type override ('gat', 'sum', 'pna', 'sage').
        override_patience: Optional early-stopping patience override.
        causal: Whether to enforce strict causal temporal sampling (no future leakage).

    Returns:
        Dictionary of final test and validation metrics.
    """
    seed_everything(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_edge_attr = (
        override_use_edge_attr
        if override_use_edge_attr is not None
        else config.model.use_edge_attr
    )
    num_hops = override_num_hops if override_num_hops is not None else config.model.num_layers
    epochs = override_epochs if override_epochs is not None else config.training.epochs
    conv_type = (
        override_conv_type.lower()
        if override_conv_type is not None
        else config.model.conv_type.lower()
    )
    patience = (
        override_patience
        if override_patience is not None
        else config.training.patience
    )

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

    if subset_fraction is not None and 0.0 < subset_fraction < 1.0:
        n_tr_sub = max(100, int(tr_edge_index.size(1) * subset_fraction))
        tr_edge_index = tr_edge_index[:, :n_tr_sub]
        tr_edge_label = tr_edge_label[:n_tr_sub]
        tr_edge_attr = tr_edge_attr[:n_tr_sub]
        tr_edge_time = tr_edge_time[:n_tr_sub]

        n_val_sub = max(50, int(val_edge_index.size(1) * subset_fraction))
        val_edge_index = val_edge_index[:, :n_val_sub]
        val_edge_label = val_edge_label[:n_val_sub]
        val_edge_attr = val_edge_attr[:n_val_sub]
        val_edge_time = val_edge_time[:n_val_sub]
        val_amounts = val_amounts[:n_val_sub]
        val_fraud_types = val_fraud_types[:n_val_sub]

        n_te_sub = max(50, int(test_edge_index.size(1) * subset_fraction))
        test_edge_index = test_edge_index[:, :n_te_sub]
        test_edge_label = test_edge_label[:n_te_sub]
        test_edge_attr = test_edge_attr[:n_te_sub]
        test_edge_time = test_edge_time[:n_te_sub]
        test_amounts = test_amounts[:n_te_sub]
        test_fraud_types = test_fraud_types[:n_te_sub]
        print(
            f"Subset active: {subset_fraction * 100:.1f}% -> train {n_tr_sub:,}, val {n_val_sub:,}, test {n_te_sub:,}"
        )

    # Build Loaders and Models based on architecture
    is_hetero = config.model.model_type == "hetero_gat"
    batch_size = config.training.batch_size
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
                batch_size=batch_size,
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
                batch_size=batch_size,
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
            in_channels_dict = {nt: int(train_graph[nt].x.size(1)) for nt in train_graph.node_types}
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
                batch_size=batch_size,
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
                batch_size=batch_size,
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

    loss_fn = get_loss_fn(
        loss_type=config.training.loss_type,
        pos_weight=computed_pos_weight if config.training.loss_type != "focal" else None,
        gamma=config.training.focal_gamma,
        alpha=config.training.focal_alpha,
    )

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

    # MLflow tracking
    mlflow.set_tracking_uri(config.training.mlflow_tracking_uri)
    mlflow.set_experiment(config.training.mlflow_experiment_name)

    best_val_pr_auc = -1.0
    patience_counter = 0
    save_stem = Path(config.training.model_save_path).stem
    if causal:
        conv_tag = f"_{conv_type}" if conv_type not in save_stem else ""
        best_checkpoint_path = Path("models") / f"{save_stem}{conv_tag}_causal.pt"
    else:
        best_checkpoint_path = Path(config.training.model_save_path)
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    actual_epochs_run = 0

    run_name = f"{config.model.model_type}_{conv_type}_{'causal' if causal else 'non_causal'}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("sampling", "causal" if causal else "non_causal_preliminary")
        mlflow.set_tag("conv_type", conv_type)
        mlflow.log_params(
            {
                "model_type": config.model.model_type,
                "conv_type": conv_type,
                "causal": causal,
                "hidden_dim": config.model.hidden_dim,
                "num_layers": num_hops,
                "use_edge_attr": use_edge_attr,
                "batch_size": batch_size,
                "learning_rate": config.training.learning_rate,
                "loss_type": config.training.loss_type,
                "patience": patience,
                "seed": config.training.seed,
                "pos_weight": computed_pos_weight,
            }
        )

        for epoch in range(1, epochs + 1):
            actual_epochs_run = epoch
            t_epoch_start = time.perf_counter()
            tr_loss = train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                gradient_clip_val=config.training.gradient_clip_val,
                is_hetero=is_hetero,
                device=device,
            )

            # Evaluate on validation
            y_val_true, y_val_prob, _ = evaluate_model(
                model=model, loader=val_loader, is_hetero=is_hetero, device=device
            )

            val_metrics = compute_metrics(
                y_true=y_val_true,
                y_pred_proba=y_val_prob,
                amounts=val_amounts,
                review_cost=config.cost.review_cost,
                default_fraud_amount=config.cost.default_fraud_amount,
                fraud_types=val_fraud_types,
            )

            scheduler.step(val_metrics.pr_auc)
            epoch_time = time.perf_counter() - t_epoch_start

            print(
                f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] - "
                f"Train Loss: {tr_loss:.4f} | "
                f"Val PR-AUC: {val_metrics.pr_auc:.4f} | "
                f"Val F1: {val_metrics.f1:.4f} (thresh={val_metrics.best_threshold:.4f}) | "
                f"Val Recall@1%FPR: {val_metrics.recall_at_1pct_fpr:.4f}",
                flush=True,
            )

            mlflow.log_metrics(
                {
                    "train_loss": tr_loss,
                    "val_pr_auc": val_metrics.pr_auc,
                    "val_roc_auc": val_metrics.roc_auc,
                    "val_f1": val_metrics.f1,
                    "val_recall_at_1pct_fpr": val_metrics.recall_at_1pct_fpr,
                },
                step=epoch,
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "val_pr_auc": val_metrics.pr_auc,
                    "val_roc_auc": val_metrics.roc_auc,
                    "val_f1": val_metrics.f1,
                    "val_recall_1pct_fpr": val_metrics.recall_at_1pct_fpr,
                }
            )

            # Early stopping check on validation PR-AUC
            if val_metrics.pr_auc > best_val_pr_auc:
                best_val_pr_auc = val_metrics.pr_auc
                patience_counter = 0
                torch.save(model.state_dict(), best_checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch} (patience: {patience})",
                        flush=True,
                    )
                    break

        # Final Evaluation: Load best weights
        print(f"\nLoading best checkpoint from {best_checkpoint_path}...")
        model.load_state_dict(torch.load(best_checkpoint_path, weights_only=True))

        # Re-evaluate validation to confirm optimal threshold
        y_val_true, y_val_prob, _ = evaluate_model(
            model=model, loader=val_loader, is_hetero=is_hetero, device=device
        )
        val_eval = compute_metrics(
            y_true=y_val_true,
            y_pred_proba=y_val_prob,
            amounts=val_amounts,
            review_cost=config.cost.review_cost,
            default_fraud_amount=config.cost.default_fraud_amount,
            fraud_types=val_fraud_types,
        )
        val_selected_threshold = val_eval.best_threshold

        # Evaluate Test Set STRICTLY using validation threshold
        y_test_true, y_test_prob, total_eval_time = evaluate_model(
            model=model, loader=test_loader, is_hetero=is_hetero, device=device
        )
        latency_ms_per_sample = (total_eval_time / max(1, len(y_test_true))) * 1000.0

        test_eval = compute_metrics(
            y_true=y_test_true,
            y_pred_proba=y_test_prob,
            amounts=test_amounts,
            review_cost=config.cost.review_cost,
            default_fraud_amount=config.cost.default_fraud_amount,
            threshold_override=val_selected_threshold,
            fraud_types=test_fraud_types,
        )

        # Log final metrics to MLflow
        mlflow.log_metrics(
            {
                "test_pr_auc": test_eval.pr_auc,
                "test_roc_auc": test_eval.roc_auc,
                "test_f1": test_eval.f1,
                "test_recall_at_1pct_fpr": test_eval.recall_at_1pct_fpr,
                "test_business_cost": test_eval.total_cost,
                "val_selected_threshold": val_selected_threshold,
                "latency_ms_per_sample": latency_ms_per_sample,
            }
        )
        for ft, rec in test_eval.per_type_recall.items():
            mlflow.log_metric(f"test_recall_{ft}", rec)

        mlflow.log_artifact(str(best_checkpoint_path), artifact_path="model")

        # Report results
        print("\n" + "=" * 70, flush=True)
        print(f"  {config.model.model_type.upper()} FINAL TEST RESULTS", flush=True)
        print("=" * 70, flush=True)
        print(f"Validation PR-AUC (Primary) : {val_eval.pr_auc:.4f}", flush=True)
        print(f"Validation Selected Thresh  : {val_selected_threshold:.4f}", flush=True)
        print("-" * 70, flush=True)
        print(f"TEST PR-AUC (Primary)       : {test_eval.pr_auc:.4f}", flush=True)
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
            "val_metrics": val_eval,
            "test_metrics": test_eval,
            "latency_ms": latency_ms_per_sample,
            "threshold": val_selected_threshold,
            "epochs_run": actual_epochs_run,
            "history": history,
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
    args = parser.parse_args(argv)

    config = load_gnn_config(args.config)
    train_and_evaluate_gnn(
        config=config,
        subset_fraction=args.subset_fraction,
        override_epochs=args.epochs,
        override_use_edge_attr=False if args.no_edge_attr else None,
        override_num_hops=args.num_hops,
        override_conv_type=args.conv_type,
        override_patience=args.patience,
        causal=args.causal,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
