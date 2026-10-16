"""Tests for heterogeneous graph construction, edge features, and temporal split leakage prevention."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from fraudgnn.config import DataConfig, GraphConfig
from fraudgnn.data.simulator import TransactionSimulator
from fraudgnn.graph.builder import (
    build_hetero_data,
    compute_edge_features,
    create_temporal_splits,
    extract_entity_mappings,
)


@pytest.fixture
def synthetic_tx_df() -> pd.DataFrame:
    """Generate a small deterministic transaction DataFrame for graph testing."""
    config = DataConfig(
        num_transactions=2_000,
        num_days=10,
        num_users=100,
        num_merchants=20,
        num_devices=150,
        num_ips=200,
        fraud_ratio=0.03,
        seed=42,
    )
    return TransactionSimulator(config).generate()


def test_entity_mappings_completeness(synthetic_tx_df: pd.DataFrame) -> None:
    """Mappings must cover every unique entity ID and have exact 1-to-1 bijection."""
    mappings = extract_entity_mappings(synthetic_tx_df)

    assert len(mappings.user_to_idx) == synthetic_tx_df["user_id"].nunique()
    assert len(mappings.card_to_idx) == synthetic_tx_df["card_id"].nunique()
    assert len(mappings.merchant_to_idx) == synthetic_tx_df["merchant_id"].nunique()
    assert len(mappings.device_to_idx) == synthetic_tx_df["device_id"].nunique()
    assert len(mappings.ip_to_idx) == synthetic_tx_df["ip_address"].nunique()

    # Bijection check
    for u, idx in mappings.user_to_idx.items():
        assert mappings.idx_to_user[idx] == u


def test_edge_features_shape_and_values(synthetic_tx_df: pd.DataFrame) -> None:
    """Edge features must have correct dimensions, no NaNs, and valid cyclical bounds."""
    edge_attrs = compute_edge_features(synthetic_tx_df)

    assert isinstance(edge_attrs, torch.Tensor)
    assert edge_attrs.shape[0] == len(synthetic_tx_df)
    assert edge_attrs.shape[1] == 10  # 10 engineered edge features
    assert not torch.isnan(edge_attrs).any(), "Edge attributes contain NaN values!"

    # Sin/Cos components must lie in [-1.0, 1.0]
    hour_sin = edge_attrs[:, 1]
    hour_cos = edge_attrs[:, 2]
    assert (hour_sin >= -1.0001).all() and (hour_sin <= 1.0001).all()
    assert (hour_cos >= -1.0001).all() and (hour_cos <= 1.0001).all()


def test_hetero_data_structure(synthetic_tx_df: pd.DataFrame) -> None:
    """HeteroData object must contain all 5 node types and bidirectional relational edges."""
    mappings = extract_entity_mappings(synthetic_tx_df)
    data = build_hetero_data(synthetic_tx_df, mappings)

    expected_node_types = {"user", "card", "merchant", "device", "ip"}
    assert set(data.node_types) == expected_node_types

    # Check node counts
    assert data["user"].num_nodes == len(mappings.user_to_idx)
    assert data["merchant"].num_nodes == len(mappings.merchant_to_idx)

    # Core transaction relation
    assert ("user", "transacts_at", "merchant") in data.edge_types
    assert ("merchant", "rev_transacts_at", "user") in data.edge_types

    tx_edges = data["user", "transacts_at", "merchant"]
    assert tx_edges.edge_index.shape[0] == 2
    assert tx_edges.edge_index.shape[1] == len(synthetic_tx_df)
    assert tx_edges.edge_attr.shape[0] == len(synthetic_tx_df)
    assert tx_edges.edge_label.shape[0] == len(synthetic_tx_df)
    assert tx_edges.edge_time.shape[0] == len(synthetic_tx_df)


def test_strict_temporal_split_no_future_leakage(synthetic_tx_df: pd.DataFrame) -> None:
    """CRITICAL TEST: Proves that zero val- or test-period edges appear in the train graph.

    Train graph must strictly contain only transactions occurring on or before train_cutoff.
    Message passing on train_data cannot see future edges.
    """
    graph_cfg = GraphConfig(split={"train_ratio": 0.60, "val_ratio": 0.20, "test_ratio": 0.20})
    splits = create_temporal_splits(synthetic_tx_df, graph_cfg)

    train_data = splits.train_data
    val_data = splits.val_data
    test_data = splits.test_data

    train_cutoff_sec = int(pd.to_datetime(splits.train_cutoff).timestamp())
    val_cutoff_sec = int(pd.to_datetime(splits.val_cutoff).timestamp())

    # 1. Inspect EVERY edge relation in train_data: ALL edge_time must be <= train_cutoff_sec
    for edge_type in train_data.edge_types:
        edge_store = train_data[edge_type]
        if hasattr(edge_store, "edge_time"):
            max_train_edge_time = int(edge_store.edge_time.max().item())
            assert max_train_edge_time <= train_cutoff_sec, (
                f"LEAKAGE DETECTED: Edge type {edge_type} has edge time {max_train_edge_time} > train cutoff {train_cutoff_sec}!"
            )

    # 2. Verify transaction edge count in train graph exactly equals train period transactions
    train_df_count = (pd.to_datetime(synthetic_tx_df["timestamp"]) <= splits.train_cutoff).sum()
    train_tx_edge_count = train_data["user", "transacts_at", "merchant"].edge_index.shape[1]
    assert train_tx_edge_count == train_df_count

    # 3. Verify val_data has zero test-period edges (all edge_time <= val_cutoff_sec)
    for edge_type in val_data.edge_types:
        edge_store = val_data[edge_type]
        if hasattr(edge_store, "edge_time"):
            max_val_edge_time = int(edge_store.edge_time.max().item())
            assert max_val_edge_time <= val_cutoff_sec, (
                f"LEAKAGE DETECTED: Val graph has edge time {max_val_edge_time} > val cutoff {val_cutoff_sec}!"
            )

    # 4. Verify test_data has full transaction edge count
    assert test_data["user", "transacts_at", "merchant"].edge_index.shape[1] == len(
        synthetic_tx_df
    )
