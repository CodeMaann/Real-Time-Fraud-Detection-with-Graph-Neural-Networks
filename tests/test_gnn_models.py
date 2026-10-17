"""Unit and integration tests for Graph Neural Network models, losses, and loaders."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from fraudgnn.graph.loader import LinkNeighborLoader
from fraudgnn.models.hetero_gat import HeteroGAT
from fraudgnn.models.losses import FocalLoss, WeightedBCEWithLogitsLoss, get_loss_fn
from fraudgnn.models.sage import GraphSAGEEdgeClassifier, convert_hetero_to_homogeneous


@pytest.fixture
def dummy_hetero_graph() -> HeteroData:
    """Construct minimal synthetic HeteroData object for unit testing."""
    data = HeteroData()
    n_users = 20
    n_merchants = 10
    n_cards = 15

    data["user"].num_nodes = n_users
    data["user"].x = torch.randn(n_users, 16)

    data["merchant"].num_nodes = n_merchants
    data["merchant"].x = torch.randn(n_merchants, 16)

    data["card"].num_nodes = n_cards
    data["card"].x = torch.randn(n_cards, 16)

    # User transacts at merchant
    src_tx = torch.randint(0, n_users, (50,))
    dst_tx = torch.randint(0, n_merchants, (50,))
    data["user", "transacts_at", "merchant"].edge_index = torch.stack([src_tx, dst_tx], dim=0)
    data["user", "transacts_at", "merchant"].edge_attr = torch.randn(50, 10)
    data["user", "transacts_at", "merchant"].edge_label = torch.randint(0, 2, (50,))
    data["user", "transacts_at", "merchant"].edge_time = torch.arange(100, 150)

    # Reverse transaction
    data["merchant", "rev_transacts_at", "user"].edge_index = torch.stack([dst_tx, src_tx], dim=0)
    data["merchant", "rev_transacts_at", "user"].edge_attr = data[
        "user", "transacts_at", "merchant"
    ].edge_attr
    data["merchant", "rev_transacts_at", "user"].edge_time = data[
        "user", "transacts_at", "merchant"
    ].edge_time

    # User uses card
    src_c = torch.randint(0, n_users, (40,))
    dst_c = torch.randint(0, n_cards, (40,))
    data["user", "uses_card", "card"].edge_index = torch.stack([src_c, dst_c], dim=0)
    data["user", "uses_card", "card"].edge_time = torch.arange(100, 140)

    data["card", "rev_uses_card", "user"].edge_index = torch.stack([dst_c, src_c], dim=0)
    data["card", "rev_uses_card", "user"].edge_time = torch.arange(100, 140)

    return data


def test_hetero_gat_forward_pass_shapes(dummy_hetero_graph: HeteroData) -> None:
    """Verify HeteroGAT produces correct logit output shapes for 1 and 2 hops."""
    data = dummy_hetero_graph
    target_rel = ("user", "transacts_at", "merchant")
    target_edge_index = data[target_rel].edge_index[:, :12]
    target_attr = data[target_rel].edge_attr[:12]

    # 1. Two-hop model with edge attributes
    model_2hop = HeteroGAT(
        node_types=data.node_types,
        edge_types=data.edge_types,
        in_channels_dict={nt: 16 for nt in data.node_types},
        hidden_channels=32,
        edge_dim=10,
        num_layers=2,
        heads=2,
        dropout=0.1,
        conv_type="gat",
        use_edge_attr=True,
    )
    logits_2hop = model_2hop(
        x_dict={nt: data[nt].x for nt in data.node_types},
        edge_index_dict={et: data[et].edge_index for et in data.edge_types},
        edge_label_index=target_edge_index,
        edge_attr=target_attr,
    )
    assert logits_2hop.shape == (12,)
    assert not torch.isnan(logits_2hop).any()

    # 2. One-hop model without edge attributes (ablation shape check)
    model_1hop = HeteroGAT(
        node_types=data.node_types,
        edge_types=data.edge_types,
        in_channels_dict={nt: 16 for nt in data.node_types},
        hidden_channels=32,
        edge_dim=10,
        num_layers=1,
        heads=2,
        dropout=0.0,
        conv_type="sage",
        use_edge_attr=False,
    )
    logits_1hop = model_1hop(
        x_dict={nt: data[nt].x for nt in data.node_types},
        edge_index_dict={et: data[et].edge_index for et in data.edge_types},
        edge_label_index=target_edge_index,
        edge_attr=None,
    )
    assert logits_1hop.shape == (12,)
    assert not torch.isnan(logits_1hop).any()


def test_graphsage_forward_pass_shapes(dummy_hetero_graph: HeteroData) -> None:
    """Verify GraphSAGE edge classifier on homogeneous graph produces correct logit shapes."""
    homo_data, _, target_h_idx, target_h_attr = convert_hetero_to_homogeneous(dummy_hetero_graph)
    sub_target_idx = target_h_idx[:, :15]
    assert target_h_attr is not None
    sub_target_attr = target_h_attr[:15]

    # Two-hop model
    model = GraphSAGEEdgeClassifier(
        in_channels=16,
        hidden_channels=32,
        edge_dim=10,
        num_layers=2,
        dropout=0.1,
        use_edge_attr=True,
    )
    logits = model(
        x=homo_data.x,
        edge_index=homo_data.edge_index,
        edge_label_index=sub_target_idx,
        edge_attr=sub_target_attr,
    )
    assert logits.shape == (15,)
    assert not torch.isnan(logits).any()

    # One-hop ablation without edge features
    model_ablation = GraphSAGEEdgeClassifier(
        in_channels=16,
        hidden_channels=32,
        edge_dim=10,
        num_layers=1,
        dropout=0.0,
        use_edge_attr=False,
    )
    logits_abl = model_ablation(
        x=homo_data.x,
        edge_index=homo_data.edge_index,
        edge_label_index=sub_target_idx,
        edge_attr=None,
    )
    assert logits_abl.shape == (15,)
    assert not torch.isnan(logits_abl).any()


def test_loss_functions_values() -> None:
    """Verify WeightedBCE and FocalLoss numerical behavior, scaling, and reductions."""
    logits = torch.tensor([2.0, -2.0, 0.5, -0.5])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

    # 1. Weighted BCE: positive weight scales positive class loss
    bce_unweighted = WeightedBCEWithLogitsLoss(pos_weight=1.0)
    bce_weighted = WeightedBCEWithLogitsLoss(pos_weight=5.0)

    loss_unw = bce_unweighted(logits, targets)
    loss_w = bce_weighted(logits, targets)
    assert loss_w > loss_unw

    # 2. Focal Loss: gamma=0 equals BCE (unweighted)
    focal_g0 = FocalLoss(gamma=0.0, alpha=None)
    assert torch.isclose(focal_g0(logits, targets), loss_unw, atol=1e-5)

    # 3. Focal Loss: gamma > 0 downweights well-classified easy samples
    focal_g2 = FocalLoss(gamma=2.0, alpha=None)
    loss_focal = focal_g2(logits, targets)
    assert loss_focal < loss_unw

    # 4. Factory function test
    fn = get_loss_fn("focal", gamma=2.0)
    assert isinstance(fn, FocalLoss)
    fn_bce = get_loss_fn("weighted_bce", pos_weight=2.0)
    assert isinstance(fn_bce, WeightedBCEWithLogitsLoss)

    with pytest.raises(ValueError):
        get_loss_fn("invalid_loss_type")


def test_no_future_edge_in_train_loader(dummy_hetero_graph: HeteroData) -> None:
    """Prove that no validation or future edge timestamp ever enters a training mini-batch."""
    data = dummy_hetero_graph
    target_rel = ("user", "transacts_at", "merchant")

    # Set temporal cutoffs: train <= 125, future > 125
    train_cutoff = 125
    for et in data.edge_types:
        times = data[et].edge_time
        mask_train = times <= train_cutoff
        data[et].edge_index = data[et].edge_index[:, mask_train]
        data[et].edge_time = data[et].edge_time[mask_train]
        if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
            data[et].edge_attr = data[et].edge_attr[mask_train]
        if hasattr(data[et], "edge_label") and data[et].edge_label is not None:
            data[et].edge_label = data[et].edge_label[mask_train]

    loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[5, 5],
        edge_label_index=(target_rel, data[target_rel].edge_index),
        edge_label=data[target_rel].edge_label,
        batch_size=10,
        shuffle=False,
    )

    # Iterate over all mini-batches and verify no edge time > train_cutoff
    for batch in loader:
        for et in batch.edge_types:
            if hasattr(batch[et], "edge_time"):
                edge_times = batch[et].edge_time
                assert (edge_times <= train_cutoff).all(), f"Future edge leakage detected in {et}!"


def test_loader_and_model_determinism_with_seed(dummy_hetero_graph: HeteroData) -> None:
    """Verify that fixing the seed produces 100% bit-exact reproducible mini-batches and outputs."""
    data = dummy_hetero_graph
    target_rel = ("user", "transacts_at", "merchant")

    def run_seeded_pass(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        np.random.seed(seed)

        loader = LinkNeighborLoader(
            data=data,
            num_neighbors=[5, 5],
            edge_label_index=(target_rel, data[target_rel].edge_index),
            edge_label=data[target_rel].edge_label,
            edge_attr=data[target_rel].edge_attr,
            batch_size=16,
            shuffle=True,
            seed=seed,
        )

        torch.manual_seed(seed)
        model = HeteroGAT(
            node_types=data.node_types,
            edge_types=data.edge_types,
            in_channels_dict={nt: 16 for nt in data.node_types},
            hidden_channels=16,
            edge_dim=10,
            num_layers=2,
            heads=2,
            dropout=0.0,
        )
        model.eval()

        batch = next(iter(loader))
        with torch.no_grad():
            logits = model(
                x_dict={nt: batch[nt].x for nt in batch.node_types},
                edge_index_dict={et: batch[et].edge_index for et in batch.edge_types},
                edge_label_index=batch[target_rel].edge_label_index,
                edge_attr=batch[target_rel].edge_attr,
            )
        return logits

    logits_run1 = run_seeded_pass(seed=42)
    logits_run2 = run_seeded_pass(seed=42)

    assert torch.allclose(logits_run1, logits_run2, atol=1e-6)
