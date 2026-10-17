"""Unit tests for strict causal temporal sampling and zero within-window lookahead leakage.

Validates:
1. When sampling neighborhoods for a target edge at time t, only edges with
   edge_time < t (strictly before) are sampled across all relations.
2. Injecting a huge fake transaction AFTER a target edge in the same window leaves
   the target edge's sampled neighborhood, node features, and model output 100% bit-identical.
3. Contrast proof: Standard non-causal loader suffers from future edge leakage.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from fraudgnn.graph.loader import LinkNeighborLoader
from fraudgnn.graph.temporal_loader import (
    CausalHomogeneousLinkLoader,
    CausalLinkNeighborLoader,
    FastTemporalAdjacency,
)
from fraudgnn.models.hetero_gat import HeteroGAT
from fraudgnn.models.sage import GraphSAGEEdgeClassifier, convert_hetero_to_homogeneous


@pytest.fixture
def synthetic_temporal_graph() -> HeteroData:
    """Create a multi-relational temporal graph with realistic sequence of transactions."""
    torch.manual_seed(123)
    np.random.seed(123)

    data = HeteroData()
    n_users = 10
    n_merchants = 6
    n_cards = 10
    n_devices = 8
    n_ips = 8

    for nt, count in [
        ("user", n_users),
        ("merchant", n_merchants),
        ("card", n_cards),
        ("device", n_devices),
        ("ip", n_ips),
    ]:
        data[nt].num_nodes = count
        data[nt].x = torch.zeros((count, 8), dtype=torch.float32)

    # Historical transactions: times 100, 200, 300, 400, 500
    src_u = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    dst_m = torch.tensor([0, 1, 0, 1, 2], dtype=torch.long)
    times = torch.tensor([1000, 2000, 3000, 4000, 5000], dtype=torch.long)
    amounts = torch.tensor([50.0, 120.0, 30.0, 200.0, 85.0], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 0, 0], dtype=torch.long)

    # 27-dim edge attributes
    attrs = torch.zeros((5, 27), dtype=torch.float32)
    attrs[:, 0] = amounts
    attrs[:, 1] = torch.log1p(amounts)

    data["user", "transacts_at", "merchant"].edge_index = torch.stack([src_u, dst_m], dim=0)
    data["user", "transacts_at", "merchant"].edge_time = times
    data["user", "transacts_at", "merchant"].edge_attr = attrs
    data["user", "transacts_at", "merchant"].edge_label = labels

    data["merchant", "rev_transacts_at", "user"].edge_index = torch.stack([dst_m, src_u], dim=0)
    data["merchant", "rev_transacts_at", "user"].edge_time = times
    data["merchant", "rev_transacts_at", "user"].edge_attr = attrs

    # Auxiliary relations (cards, devices, ips)
    cards = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    data["user", "uses_card", "card"].edge_index = torch.stack([src_u, cards], dim=0)
    data["user", "uses_card", "card"].edge_time = times
    data["card", "rev_uses_card", "user"].edge_index = torch.stack([cards, src_u], dim=0)
    data["card", "rev_uses_card", "user"].edge_time = times

    devices = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    data["user", "uses_device", "device"].edge_index = torch.stack([src_u, devices], dim=0)
    data["user", "uses_device", "device"].edge_time = times
    data["device", "rev_uses_device", "user"].edge_index = torch.stack([devices, src_u], dim=0)
    data["device", "rev_uses_device", "user"].edge_time = times

    ips = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    data["user", "uses_ip", "ip"].edge_index = torch.stack([src_u, ips], dim=0)
    data["user", "uses_ip", "ip"].edge_time = times
    data["ip", "rev_uses_ip", "user"].edge_index = torch.stack([ips, src_u], dim=0)
    data["ip", "rev_uses_ip", "user"].edge_time = times

    data["card", "pays_merchant", "merchant"].edge_index = torch.stack([cards, dst_m], dim=0)
    data["card", "pays_merchant", "merchant"].edge_time = times
    data["merchant", "rev_pays_merchant", "card"].edge_index = torch.stack([dst_m, cards], dim=0)
    data["merchant", "rev_pays_merchant", "card"].edge_time = times

    return data


def test_fast_temporal_adjacency_strictly_before() -> None:
    """Verify FastTemporalAdjacency enforces edge_time < cutoff strictly."""
    edge_index = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]], dtype=torch.long)
    edge_time = torch.tensor([100, 200, 200, 300], dtype=torch.long)
    edge_amount = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)

    adj = FastTemporalAdjacency(edge_index, edge_time, num_src_nodes=1, edge_amount=edge_amount)

    # Cutoff at 200: must only see edge at time 100 (strictly before 200)
    dsts, eids, _cuts = adj.sample_causal_neighbors(
        [0], [200], max_neighbors=10, rng=np.random.default_rng(42)
    )
    assert len(dsts) == 1
    assert dsts[0] == 1
    assert eids[0] == 0

    deg, dist, amt, ft = adj.get_causal_node_stats([0], [200])
    assert deg[0] == 1.0
    assert dist[0] == 1.0
    assert amt[0] == 10.0
    assert ft[0] == 100.0

    # Cutoff at 250: must see edges at 100, 200, 200 (not 300)
    dsts, eids, _cuts = adj.sample_causal_neighbors(
        [0], [250], max_neighbors=10, rng=np.random.default_rng(42)
    )
    assert len(dsts) == 3
    assert set(dsts.tolist()) == {1, 2, 3}


def test_future_injection_bit_identical_invariance(synthetic_temporal_graph: HeteroData) -> None:
    """Inject a huge fake transaction AFTER a target edge and prove bit-identical invariance."""
    data = synthetic_temporal_graph
    target_rel = ("user", "transacts_at", "merchant")

    # Target edge index 2: occurring at timestamp 3000 (user 2 -> merchant 0)
    target_idx = 2
    t0 = int(data[target_rel].edge_time[target_idx].item())
    target_edge_index = data[target_rel].edge_index[:, target_idx : target_idx + 1]
    target_label = data[target_rel].edge_label[target_idx : target_idx + 1]
    target_time = data[target_rel].edge_time[target_idx : target_idx + 1]
    target_attr = data[target_rel].edge_attr[target_idx : target_idx + 1]

    # Initialize model with deterministic seed
    torch.manual_seed(999)
    model = HeteroGAT(
        node_types=data.node_types,
        edge_types=data.edge_types,
        in_channels_dict={nt: 17 for nt in data.node_types},
        hidden_channels=32,
        edge_dim=27,
        num_layers=2,
        heads=2,
        dropout=0.0,
        conv_type="gat",
        use_edge_attr=True,
    )
    model.eval()

    # 1. Sample and evaluate on clean graph
    loader_clean = CausalLinkNeighborLoader(
        data=data,
        num_neighbors=[10, 10],
        edge_label_index=(target_rel, target_edge_index),
        edge_label=target_label,
        edge_time=target_time,
        edge_attr=target_attr,
        batch_size=1,
        shuffle=False,
        seed=42,
    )

    batch_clean = next(iter(loader_clean))
    with torch.no_grad():
        out_clean = model(
            x_dict={nt: batch_clean[nt].x for nt in batch_clean.node_types},
            edge_index_dict={et: batch_clean[et].edge_index for et in batch_clean.edge_types},
            edge_label_index=batch_clean[target_rel].edge_label_index,
            edge_attr=batch_clean[target_rel].edge_attr,
        )

    # 2. Inject huge fake transaction strictly AFTER target edge in the same window (e.g. t = t0 + 500 = 3500)
    t_fake = t0 + 500
    fake_amount = 1_000_000.0
    data_injected = copy.deepcopy(data)

    # Append to user-transacts-at-merchant
    fake_edge = torch.tensor([[2], [0]], dtype=torch.long)  # Same endpoints
    fake_t = torch.tensor([t_fake], dtype=torch.long)
    fake_attr = torch.zeros((1, 27), dtype=torch.float32)
    fake_attr[0, 0] = fake_amount
    fake_attr[0, 1] = float(np.log1p(fake_amount))
    fake_lbl = torch.tensor([1], dtype=torch.long)

    data_injected[target_rel].edge_index = torch.cat(
        [data_injected[target_rel].edge_index, fake_edge], dim=1
    )
    data_injected[target_rel].edge_time = torch.cat(
        [data_injected[target_rel].edge_time, fake_t], dim=0
    )
    data_injected[target_rel].edge_attr = torch.cat(
        [data_injected[target_rel].edge_attr, fake_attr], dim=0
    )
    data_injected[target_rel].edge_label = torch.cat(
        [data_injected[target_rel].edge_label, fake_lbl], dim=0
    )

    # Append reverse
    rev_rel = ("merchant", "rev_transacts_at", "user")
    fake_rev = torch.tensor([[0], [2]], dtype=torch.long)
    data_injected[rev_rel].edge_index = torch.cat(
        [data_injected[rev_rel].edge_index, fake_rev], dim=1
    )
    data_injected[rev_rel].edge_time = torch.cat([data_injected[rev_rel].edge_time, fake_t], dim=0)
    data_injected[rev_rel].edge_attr = torch.cat(
        [data_injected[rev_rel].edge_attr, fake_attr], dim=0
    )

    # Append to auxiliary relations
    for rel_k in [
        ("user", "uses_card", "card"),
        ("user", "uses_device", "device"),
        ("user", "uses_ip", "ip"),
    ]:
        data_injected[rel_k].edge_index = torch.cat(
            [data_injected[rel_k].edge_index, torch.tensor([[2], [0]], dtype=torch.long)], dim=1
        )
        data_injected[rel_k].edge_time = torch.cat([data_injected[rel_k].edge_time, fake_t], dim=0)

    # 3. Sample and evaluate on injected graph
    loader_injected = CausalLinkNeighborLoader(
        data=data_injected,
        num_neighbors=[10, 10],
        edge_label_index=(target_rel, target_edge_index),
        edge_label=target_label,
        edge_time=target_time,
        edge_attr=target_attr,
        batch_size=1,
        shuffle=False,
        seed=42,
    )

    batch_injected = next(iter(loader_injected))
    with torch.no_grad():
        out_injected = model(
            x_dict={nt: batch_injected[nt].x for nt in batch_injected.node_types},
            edge_index_dict={
                et: batch_injected[et].edge_index for et in batch_injected.edge_types
            },
            edge_label_index=batch_injected[target_rel].edge_label_index,
            edge_attr=batch_injected[target_rel].edge_attr,
        )

    # 4. Rigorous assertions: Bit-identical neighborhood, node features, and model output
    # Check sampled nodes
    for nt in data.node_types:
        assert torch.equal(batch_clean[nt].n_id, batch_injected[nt].n_id), (
            f"Node IDs differ for {nt}!"
        )
        assert torch.equal(batch_clean[nt].x, batch_injected[nt].x), (
            f"Node features differ for {nt}!"
        )

    # Check sampled edges
    for et in data.edge_types:
        assert torch.equal(batch_clean[et].edge_index, batch_injected[et].edge_index), (
            f"Edge index differs for {et}!"
        )

    # Check model output: 100% bit-identical
    assert torch.equal(out_clean, out_injected), (
        f"Model output changed after future transaction injection! "
        f"Clean: {out_clean.item():.8f}, Injected: {out_injected.item():.8f}"
    )


def test_contrast_non_causal_loader_leaks_future_transaction(
    synthetic_temporal_graph: HeteroData,
) -> None:
    """Demonstrate that standard non-causal loader fails the future injection invariance test."""
    data = synthetic_temporal_graph
    target_rel = ("user", "transacts_at", "merchant")

    target_idx = 2
    t0 = int(data[target_rel].edge_time[target_idx].item())
    target_edge_index = data[target_rel].edge_index[:, target_idx : target_idx + 1]
    target_label = data[target_rel].edge_label[target_idx : target_idx + 1]

    # Clean non-causal loader
    loader_nc_clean = LinkNeighborLoader(
        data=data,
        num_neighbors=[10, 10],
        edge_label_index=(target_rel, target_edge_index),
        edge_label=target_label,
        batch_size=1,
        shuffle=False,
    )
    batch_nc_clean = next(iter(loader_nc_clean))

    # Injected non-causal graph
    data_injected = copy.deepcopy(data)
    fake_edge = torch.tensor([[2], [0]], dtype=torch.long)
    fake_t = torch.tensor([t0 + 500], dtype=torch.long)
    data_injected[target_rel].edge_index = torch.cat(
        [data_injected[target_rel].edge_index, fake_edge], dim=1
    )
    data_injected[target_rel].edge_time = torch.cat(
        [data_injected[target_rel].edge_time, fake_t], dim=0
    )
    data_injected[target_rel].edge_label = torch.cat(
        [data_injected[target_rel].edge_label, torch.tensor([1])], dim=0
    )

    loader_nc_injected = LinkNeighborLoader(
        data=data_injected,
        num_neighbors=[10, 10],
        edge_label_index=(target_rel, target_edge_index),
        edge_label=target_label,
        batch_size=1,
        shuffle=False,
    )
    batch_nc_injected = next(iter(loader_nc_injected))

    # The non-causal loader sees more edges because it samples the future edge!
    n_edges_clean = batch_nc_clean[target_rel].edge_index.size(1)
    n_edges_inj = batch_nc_injected[target_rel].edge_index.size(1)
    assert n_edges_inj > n_edges_clean, "Non-causal loader should have sampled the future edge!"


def test_graphsage_causal_loader_future_invariance(synthetic_temporal_graph: HeteroData) -> None:
    """Verify GraphSAGE homogeneous causal loader satisfies 100% bit-identical future invariance."""
    data = synthetic_temporal_graph
    homo_data, _, target_h_idx, target_h_attr = convert_hetero_to_homogeneous(data)

    target_idx = 2
    t0 = int(data["user", "transacts_at", "merchant"].edge_time[target_idx].item())
    sub_edge_index = target_h_idx[:, target_idx : target_idx + 1]
    sub_label = (
        data["user", "transacts_at", "merchant"]
        .edge_label[target_idx : target_idx + 1]
        .to(dtype=torch.float32)
    )
    sub_time = data["user", "transacts_at", "merchant"].edge_time[target_idx : target_idx + 1]
    sub_attr = target_h_attr[target_idx : target_idx + 1] if target_h_attr is not None else None

    torch.manual_seed(888)
    model = GraphSAGEEdgeClassifier(
        in_channels=8,
        hidden_channels=32,
        edge_dim=27,
        num_layers=2,
        dropout=0.0,
        use_edge_attr=True,
    )
    model.eval()

    loader_clean = CausalHomogeneousLinkLoader(
        data=homo_data,
        edge_label_index=sub_edge_index,
        edge_label=sub_label,
        edge_time=sub_time,
        edge_attr=sub_attr,
        batch_size=1,
        num_neighbors=[10, 10],
        shuffle=False,
        seed=42,
    )
    batch_clean = next(iter(loader_clean))

    with torch.no_grad():
        out_clean = model(
            x=batch_clean.x,
            edge_index=batch_clean.edge_index,
            edge_label_index=batch_clean.edge_label_index,
            edge_attr=batch_clean.edge_attr,
        )

    # Injected homogeneous graph with a future edge
    homo_injected = copy.deepcopy(homo_data)
    fake_edge = torch.tensor(
        [[sub_edge_index[0, 0].item()], [sub_edge_index[1, 0].item()]], dtype=torch.long
    )
    fake_time = torch.tensor([t0 + 500], dtype=torch.long)

    homo_injected.edge_index = torch.cat([homo_injected.edge_index, fake_edge], dim=1)
    homo_injected.edge_time = torch.cat([homo_injected.edge_time, fake_time], dim=0)
    if hasattr(homo_injected, "edge_attr") and homo_injected.edge_attr is not None:
        fake_attr = torch.zeros((1, 27), dtype=torch.float32)
        fake_attr[0, 0] = 50000.0
        homo_injected.edge_attr = torch.cat([homo_injected.edge_attr, fake_attr], dim=0)

    loader_inj = CausalHomogeneousLinkLoader(
        data=homo_injected,
        edge_label_index=sub_edge_index,
        edge_label=sub_label,
        edge_time=sub_time,
        edge_attr=sub_attr,
        batch_size=1,
        num_neighbors=[10, 10],
        shuffle=False,
        seed=42,
    )
    batch_inj = next(iter(loader_inj))

    with torch.no_grad():
        out_inj = model(
            x=batch_inj.x,
            edge_index=batch_inj.edge_index,
            edge_label_index=batch_inj.edge_label_index,
            edge_attr=batch_inj.edge_attr,
        )

    assert torch.equal(batch_clean.n_id, batch_inj.n_id)
    assert torch.equal(batch_clean.x, batch_inj.x)
    assert torch.equal(batch_clean.edge_index, batch_inj.edge_index)
    assert torch.equal(out_clean, out_inj), (
        f"GraphSAGE output changed after future injection: {out_clean} vs {out_inj}"
    )
