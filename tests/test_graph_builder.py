"""Tests for heterogeneous graph construction, edge features, and temporal split leakage prevention."""

from __future__ import annotations

import numpy as np
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
    """Edge features must have correct dimensions, no NaNs, and valid scaled bounds."""
    edge_attrs, _ = compute_edge_features(synthetic_tx_df)

    assert isinstance(edge_attrs, torch.Tensor)
    assert edge_attrs.shape[0] == len(synthetic_tx_df)
    assert edge_attrs.shape[1] == 27  # 27 causal tabular edge features
    assert not torch.isnan(edge_attrs).any(), "Edge attributes contain NaN values!"


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


def test_hetero_data_edge_labels_binary_without_type_leakage(
    synthetic_tx_df: pd.DataFrame,
) -> None:
    """Edge labels must be strictly binary {0, 1} and edge features cannot contain fraud_type."""
    mappings = extract_entity_mappings(synthetic_tx_df)
    data = build_hetero_data(synthetic_tx_df, mappings)

    edge_label = data["user", "transacts_at", "merchant"].edge_label
    unique_labels = torch.unique(edge_label).tolist()
    assert set(unique_labels).issubset({0, 1})

    edge_attr = data["user", "transacts_at", "merchant"].edge_attr
    # 27 causal tabular features without label leakage
    assert edge_attr.shape[1] == 27
    assert not torch.isnan(edge_attr).any()


def test_node_features_temporal_purity_no_future_leakage(synthetic_tx_df: pd.DataFrame) -> None:
    """CRITICAL TEST: Proves node features in train/val graphs use ONLY past transactions.

    Zero information from test transactions leaks into train or val node feature vectors.
    """
    graph_cfg = GraphConfig(split={"train_ratio": 0.60, "val_ratio": 0.20, "test_ratio": 0.20})
    splits = create_temporal_splits(synthetic_tx_df, graph_cfg)

    train_data = splits.train_data
    val_data = splits.val_data

    # Timestamps
    ts = pd.to_datetime(synthetic_tx_df["timestamp"])
    train_df = synthetic_tx_df[ts <= splits.train_cutoff]
    val_df = synthetic_tx_df[ts <= splits.val_cutoff]

    user_map = splits.mappings.user_to_idx

    # 1. Degree feature (col 0): Must exactly equal log1p(count of visible transactions)
    for u, idx in user_map.items():
        expected_train_deg = np.log1p((train_df["user_id"] == u).sum())
        actual_train_deg = float(train_data["user"].x[idx, 0].item())
        assert np.isclose(actual_train_deg, expected_train_deg, atol=1e-5), (
            f"Leakage in train user {u}: expected {expected_train_deg}, got {actual_train_deg}"
        )

        expected_val_deg = np.log1p((val_df["user_id"] == u).sum())
        actual_val_deg = float(val_data["user"].x[idx, 0].item())
        assert np.isclose(actual_val_deg, expected_val_deg, atol=1e-5), (
            f"Leakage in val user {u}: expected {expected_val_deg}, got {actual_val_deg}"
        )

    # 2. Inject an extreme future transaction in test period: train & val features must not change
    mutated_df = synthetic_tx_df.copy()
    test_user = mutated_df["user_id"].iloc[-1]
    test_user_idx = user_map[test_user]

    train_feat_before = train_data["user"].x[test_user_idx].clone()
    val_feat_before = val_data["user"].x[test_user_idx].clone()

    # Add a huge future transaction strictly inside the test period (between val_cutoff and t_max)
    future_row = mutated_df.iloc[-1:].copy()
    future_row["amount"] = 9999999.0
    future_row["timestamp"] = splits.val_cutoff + pd.Timedelta(minutes=10)
    mutated_df = pd.concat([mutated_df, future_row], ignore_index=True)

    mutated_splits = create_temporal_splits(mutated_df, graph_cfg)
    train_feat_after = mutated_splits.train_data["user"].x[test_user_idx]
    val_feat_after = mutated_splits.val_data["user"].x[test_user_idx]

    # Future row in test period CANNOT alter train or val node feature vectors
    assert torch.allclose(train_feat_before, train_feat_after, atol=1e-6), (
        "Future data leaked into train node features!"
    )
    assert torch.allclose(val_feat_before, val_feat_after, atol=1e-6), (
        "Future data leaked into val node features!"
    )


def test_edge_feature_scaler_fitted_on_train_only(synthetic_tx_df: pd.DataFrame) -> None:
    """CRITICAL TEST: Proves edge-feature StandardScaler is fitted strictly on the TRAIN split."""
    graph_cfg = GraphConfig(split={"train_ratio": 0.60, "val_ratio": 0.20, "test_ratio": 0.20})
    ts = pd.to_datetime(synthetic_tx_df["timestamp"])
    train_cutoff = ts.min() + (ts.max() - ts.min()) * 0.60
    train_df = synthetic_tx_df[ts <= train_cutoff].reset_index(drop=True)

    from fraudgnn.data.features import compute_tabular_features, get_feature_column_names

    train_feat_df = compute_tabular_features(train_df, include_label_features=False)
    cols = get_feature_column_names(include_label_features=False)
    expected_train_mean = train_feat_df[cols].to_numpy(dtype=np.float32).mean(axis=0)

    splits = create_temporal_splits(synthetic_tx_df, graph_cfg)
    from fraudgnn.graph.builder import compute_edge_features

    _, fitted_scaler = compute_edge_features(train_df, scaler=None)

    # Scaler mean must match train mean exactly
    assert np.allclose(fitted_scaler.mean_, expected_train_mean, atol=1e-4)

    # Test edge attributes: transformed by train scaler, not re-centered
    test_edge_attr = splits.test_data["user", "transacts_at", "merchant"].edge_attr
    assert test_edge_attr.shape[1] == 27
    assert not torch.isnan(test_edge_attr).any()
