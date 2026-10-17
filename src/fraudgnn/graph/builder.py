"""Heterogeneous Graph Builder for FraudGNN.

Converts transaction streams into PyTorch Geometric HeteroData objects
with strict temporal splitting to guarantee zero future leakage during
graph message passing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData

from fraudgnn.config import GraphConfig
from fraudgnn.data.features import compute_tabular_features, get_feature_column_names


@dataclass
class EntityMappings:
    """Bi-directional mappings from entity string IDs to contiguous tensor indices."""

    user_to_idx: dict[str, int]
    card_to_idx: dict[str, int]
    merchant_to_idx: dict[str, int]
    device_to_idx: dict[str, int]
    ip_to_idx: dict[str, int]

    idx_to_user: dict[int, str]
    idx_to_card: dict[int, str]
    idx_to_merchant: dict[int, str]
    idx_to_device: dict[int, str]
    idx_to_ip: dict[int, str]


@dataclass
class TemporalGraphSplits:
    """Container for temporal inductive graph splits."""

    train_data: HeteroData
    val_data: HeteroData
    test_data: HeteroData
    train_cutoff: pd.Timestamp
    val_cutoff: pd.Timestamp
    mappings: EntityMappings


def extract_entity_mappings(df: pd.DataFrame) -> EntityMappings:
    """Create deterministic string-to-index and index-to-string mappings for all nodes.

    Args:
        df: Transaction DataFrame containing entity columns.

    Returns:
        Populated EntityMappings object.
    """
    unique_users = sorted(df["user_id"].unique().tolist())
    unique_cards = sorted(df["card_id"].unique().tolist())
    unique_merchants = sorted(df["merchant_id"].unique().tolist())
    unique_devices = sorted(df["device_id"].unique().tolist())
    unique_ips = sorted(df["ip_address"].unique().tolist())

    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    card_to_idx = {c: i for i, c in enumerate(unique_cards)}
    merchant_to_idx = {m: i for i, m in enumerate(unique_merchants)}
    device_to_idx = {d: i for i, d in enumerate(unique_devices)}
    ip_to_idx = {ip: i for i, ip in enumerate(unique_ips)}

    return EntityMappings(
        user_to_idx=user_to_idx,
        card_to_idx=card_to_idx,
        merchant_to_idx=merchant_to_idx,
        device_to_idx=device_to_idx,
        ip_to_idx=ip_to_idx,
        idx_to_user={i: u for u, i in user_to_idx.items()},
        idx_to_card={i: c for c, i in card_to_idx.items()},
        idx_to_merchant={i: m for m, i in merchant_to_idx.items()},
        idx_to_device={i: d for d, i in device_to_idx.items()},
        idx_to_ip={i: ip for ip, i in ip_to_idx.items()},
    )


def compute_label_free_node_features(
    df_split: pd.DataFrame,
    mappings: EntityMappings,
    cutoff: pd.Timestamp | None = None,
) -> dict[str, torch.Tensor]:
    """Compute 8-dimensional label-free structural node features from visible history.

    Features per node type:
    1. log1p(degree): Transaction count seen so far
    2. log1p(distinct_users): Number of distinct users connected
    3. log1p(distinct_cards): Number of distinct cards connected
    4. log1p(distinct_devices): Number of distinct devices connected
    5. log1p(distinct_ips): Number of distinct IPs connected
    6. log1p(distinct_merchants): Number of distinct merchants connected
    7. log1p(total_amount): Cumulative transaction volume
    8. log1p(age_hours): Hours since entity first appeared in the split graph

    Args:
        df_split: Transactions DataFrame visible up to the split cutoff.
        mappings: Master EntityMappings.
        cutoff: Optional split cutoff timestamp. If None, uses max timestamp of df_split.

    Returns:
        Dict mapping node_type to FloatTensor of shape [num_nodes, 8].
    """
    node_features: dict[str, torch.Tensor] = {}
    entities = [
        ("user", "user_id", mappings.user_to_idx),
        ("card", "card_id", mappings.card_to_idx),
        ("merchant", "merchant_id", mappings.merchant_to_idx),
        ("device", "device_id", mappings.device_to_idx),
        ("ip", "ip_address", mappings.ip_to_idx),
    ]

    if len(df_split) == 0:
        for node_type, _, mapping in entities:
            node_features[node_type] = torch.zeros((len(mapping), 8), dtype=torch.float32)
        return node_features

    timestamps = pd.to_datetime(df_split["timestamp"])
    max_ts = pd.to_datetime(cutoff) if cutoff is not None else timestamps.max()
    df_w = pd.DataFrame(
        {
            "user_id": df_split["user_id"],
            "card_id": df_split["card_id"],
            "merchant_id": df_split["merchant_id"],
            "device_id": df_split["device_id"],
            "ip_address": df_split["ip_address"],
            "amount": df_split["amount"],
            "ts": timestamps,
        }
    )

    for node_type, id_col, mapping in entities:
        feats = np.zeros((len(mapping), 8), dtype=np.float32)
        grp = df_w.groupby(id_col)
        deg = grp.size()
        n_u = grp["user_id"].nunique() if id_col != "user_id" else None
        n_c = grp["card_id"].nunique() if id_col != "card_id" else None
        n_d = grp["device_id"].nunique() if id_col != "device_id" else None
        n_ip = grp["ip_address"].nunique() if id_col != "ip_address" else None
        n_m = grp["merchant_id"].nunique() if id_col != "merchant_id" else None
        tot_amt = grp["amount"].sum()
        min_ts = grp["ts"].min()
        age_hours = (max_ts - min_ts).dt.total_seconds() / 3600.0

        for ent_id, d in deg.items():
            idx = mapping.get(str(ent_id))
            if idx is None:
                continue
            feats[idx, 0] = np.log1p(d)
            feats[idx, 1] = np.log1p(1.0 if n_u is None else n_u.get(ent_id, 0))
            feats[idx, 2] = np.log1p(1.0 if n_c is None else n_c.get(ent_id, 0))
            feats[idx, 3] = np.log1p(1.0 if n_d is None else n_d.get(ent_id, 0))
            feats[idx, 4] = np.log1p(1.0 if n_ip is None else n_ip.get(ent_id, 0))
            feats[idx, 5] = np.log1p(1.0 if n_m is None else n_m.get(ent_id, 0))
            feats[idx, 6] = np.log1p(max(0.0, float(tot_amt.get(ent_id, 0.0))))
            feats[idx, 7] = np.log1p(max(0.0, float(age_hours.get(ent_id, 0.0))))

        node_features[node_type] = torch.from_numpy(feats)

    return node_features


def compute_edge_features(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
) -> tuple[torch.Tensor, StandardScaler]:
    """Extract strictly causal, label-free 27 tabular features for transactions.

    Args:
        df: Chronologically sorted transaction DataFrame.
        scaler: Optional fitted StandardScaler. If None, fits a new scaler.

    Returns:
        Tuple of (feature_matrix of shape [N, 27], fitted_scaler).
    """
    if len(df) == 0:
        empty_scaler = scaler if scaler is not None else StandardScaler()
        return torch.empty((0, 27), dtype=torch.float32), empty_scaler

    feat_df = compute_tabular_features(df, include_label_features=False)
    cols = get_feature_column_names(include_label_features=False)
    mat = feat_df[cols].to_numpy(dtype=np.float32)

    if scaler is None:
        scaler = StandardScaler()
        mat_scaled = scaler.fit_transform(mat)
    else:
        mat_scaled = scaler.transform(mat)

    return torch.from_numpy(mat_scaled.astype(np.float32)), scaler


def build_hetero_data(
    df: pd.DataFrame,
    mappings: EntityMappings,
    initial_node_dim: int = 8,
    node_features: dict[str, torch.Tensor] | None = None,
    edge_attrs: torch.Tensor | None = None,
) -> HeteroData:
    """Construct a PyG HeteroData object from transactions.

    Args:
        df: Transactions DataFrame.
        mappings: EntityMappings holding string-to-index dicts.
        initial_node_dim: Dimension for fallback initial node features (default: 8).
        node_features: Optional precomputed label-free node feature dictionary.
        edge_attrs: Optional precomputed 27-dim edge feature tensor.

    Returns:
        PyG HeteroData object.
    """
    data = HeteroData()

    # 1. Register Nodes and initial features
    for node_type, mapping in [
        ("user", mappings.user_to_idx),
        ("card", mappings.card_to_idx),
        ("merchant", mappings.merchant_to_idx),
        ("device", mappings.device_to_idx),
        ("ip", mappings.ip_to_idx),
    ]:
        n_nodes = len(mapping)
        data[node_type].num_nodes = n_nodes
        if node_features is not None and node_type in node_features:
            data[node_type].x = node_features[node_type]
        else:
            data[node_type].x = torch.zeros((n_nodes, initial_node_dim), dtype=torch.float32)

    if len(df) == 0:
        return data

    # 2. Extract edge indices using mapped contiguous integer IDs
    user_indices = torch.tensor(
        [mappings.user_to_idx[u] for u in df["user_id"]],
        dtype=torch.long,
    )
    card_indices = torch.tensor(
        [mappings.card_to_idx[c] for c in df["card_id"]],
        dtype=torch.long,
    )
    merchant_indices = torch.tensor(
        [mappings.merchant_to_idx[m] for m in df["merchant_id"]],
        dtype=torch.long,
    )
    device_indices = torch.tensor(
        [mappings.device_to_idx[d] for d in df["device_id"]],
        dtype=torch.long,
    )
    ip_indices = torch.tensor(
        [mappings.ip_to_idx[ip] for ip in df["ip_address"]],
        dtype=torch.long,
    )

    timestamps = pd.to_datetime(df["timestamp"])
    edge_times = torch.tensor(
        timestamps.astype("int64") // 10**9,
        dtype=torch.long,
    )
    edge_labels = torch.tensor(df["is_fraud"].to_numpy(), dtype=torch.long)
    if edge_attrs is None:
        computed_attrs, _ = compute_edge_features(df)
    else:
        computed_attrs = edge_attrs

    # 3. Core transaction relation: (user, transacts_at, merchant)
    tx_edge_index = torch.stack([user_indices, merchant_indices], dim=0)
    data["user", "transacts_at", "merchant"].edge_index = tx_edge_index
    data["user", "transacts_at", "merchant"].edge_attr = computed_attrs
    data["user", "transacts_at", "merchant"].edge_label = edge_labels
    data["user", "transacts_at", "merchant"].edge_time = edge_times

    # Reverse transaction edge for bidirectional message passing
    rev_tx_edge_index = torch.stack([merchant_indices, user_indices], dim=0)
    data["merchant", "rev_transacts_at", "user"].edge_index = rev_tx_edge_index
    data["merchant", "rev_transacts_at", "user"].edge_attr = computed_attrs
    data["merchant", "rev_transacts_at", "user"].edge_time = edge_times

    # 4. Entity connection edges
    u_c_index = torch.stack([user_indices, card_indices], dim=0)
    data["user", "uses_card", "card"].edge_index = u_c_index
    data["user", "uses_card", "card"].edge_time = edge_times
    data["card", "rev_uses_card", "user"].edge_index = torch.stack(
        [card_indices, user_indices], dim=0
    )
    data["card", "rev_uses_card", "user"].edge_time = edge_times

    u_d_index = torch.stack([user_indices, device_indices], dim=0)
    data["user", "uses_device", "device"].edge_index = u_d_index
    data["user", "uses_device", "device"].edge_time = edge_times
    data["device", "rev_uses_device", "user"].edge_index = torch.stack(
        [device_indices, user_indices], dim=0
    )
    data["device", "rev_uses_device", "user"].edge_time = edge_times

    u_ip_index = torch.stack([user_indices, ip_indices], dim=0)
    data["user", "uses_ip", "ip"].edge_index = u_ip_index
    data["user", "uses_ip", "ip"].edge_time = edge_times
    data["ip", "rev_uses_ip", "user"].edge_index = torch.stack([ip_indices, user_indices], dim=0)
    data["ip", "rev_uses_ip", "user"].edge_time = edge_times

    c_m_index = torch.stack([card_indices, merchant_indices], dim=0)
    data["card", "pays_merchant", "merchant"].edge_index = c_m_index
    data["card", "pays_merchant", "merchant"].edge_time = edge_times
    data["merchant", "rev_pays_merchant", "card"].edge_index = torch.stack(
        [merchant_indices, card_indices], dim=0
    )
    data["merchant", "rev_pays_merchant", "card"].edge_time = edge_times

    return data


def create_temporal_splits(
    df: pd.DataFrame,
    config: GraphConfig,
) -> TemporalGraphSplits:
    """Create temporal inductive graph splits with zero future leakage.

    Splitting rule:
    - Train split: first 60% of timeline (Edges strictly <= train_cutoff).
      Zero validation or test edges are present in train_data.
    - Val split: first 80% of timeline (Edges strictly <= val_cutoff).
      Validation target edges flagged with val_mask.
    - Test split: 100% of timeline. Test target edges flagged with test_mask.

    All edge features are 27-dimensional causal tabular features.
    All node features are 8-dimensional label-free structural features.

    Args:
        df: Chronologically sorted transactions DataFrame.
        config: GraphConfig holding split ratios and paths.

    Returns:
        TemporalGraphSplits container.
    """
    timestamps = pd.to_datetime(df["timestamp"])
    t_min = timestamps.min()
    t_max = timestamps.max()
    total_duration = t_max - t_min

    train_cutoff = t_min + total_duration * config.split.train_ratio
    val_cutoff = t_min + total_duration * (config.split.train_ratio + config.split.val_ratio)

    mappings = extract_entity_mappings(df)

    train_df = df[timestamps <= train_cutoff].reset_index(drop=True)
    val_cum_df = df[timestamps <= val_cutoff].reset_index(drop=True)
    test_cum_df = df.copy().reset_index(drop=True)

    # 1. Train Graph: Strictly contains ONLY train-period transactions
    train_node_feats = compute_label_free_node_features(train_df, mappings, cutoff=train_cutoff)
    train_edge_attrs, train_scaler = compute_edge_features(train_df, scaler=None)
    train_data = build_hetero_data(
        train_df, mappings, node_features=train_node_feats, edge_attrs=train_edge_attrs
    )
    n_train_tx = len(train_df)
    train_data["user", "transacts_at", "merchant"].train_mask = torch.ones(
        n_train_tx, dtype=torch.bool
    )
    train_data["user", "transacts_at", "merchant"].val_mask = torch.zeros(
        n_train_tx, dtype=torch.bool
    )
    train_data["user", "transacts_at", "merchant"].test_mask = torch.zeros(
        n_train_tx, dtype=torch.bool
    )

    # 2. Val Graph: Contains edges up to val_cutoff. Only val-period edges are evaluated
    val_node_feats = compute_label_free_node_features(val_cum_df, mappings, cutoff=val_cutoff)
    val_edge_attrs, _ = compute_edge_features(val_cum_df, scaler=train_scaler)
    val_data = build_hetero_data(
        val_cum_df, mappings, node_features=val_node_feats, edge_attrs=val_edge_attrs
    )
    val_timestamps = pd.to_datetime(val_cum_df["timestamp"])
    val_data["user", "transacts_at", "merchant"].train_mask = torch.tensor(
        (val_timestamps <= train_cutoff).to_numpy(), dtype=torch.bool
    )
    val_data["user", "transacts_at", "merchant"].val_mask = torch.tensor(
        ((val_timestamps > train_cutoff) & (val_timestamps <= val_cutoff)).to_numpy(),
        dtype=torch.bool,
    )
    val_data["user", "transacts_at", "merchant"].test_mask = torch.zeros(
        len(val_cum_df), dtype=torch.bool
    )

    # 3. Test Graph: Contains full history. Only test-period edges are evaluated
    test_node_feats = compute_label_free_node_features(test_cum_df, mappings, cutoff=t_max)
    test_edge_attrs, _ = compute_edge_features(test_cum_df, scaler=train_scaler)
    test_data = build_hetero_data(
        test_cum_df, mappings, node_features=test_node_feats, edge_attrs=test_edge_attrs
    )
    test_timestamps = pd.to_datetime(test_cum_df["timestamp"])
    test_data["user", "transacts_at", "merchant"].train_mask = torch.tensor(
        (test_timestamps <= train_cutoff).to_numpy(), dtype=torch.bool
    )
    test_data["user", "transacts_at", "merchant"].val_mask = torch.tensor(
        ((test_timestamps > train_cutoff) & (test_timestamps <= val_cutoff)).to_numpy(),
        dtype=torch.bool,
    )
    test_data["user", "transacts_at", "merchant"].test_mask = torch.tensor(
        (test_timestamps > val_cutoff).to_numpy(), dtype=torch.bool
    )

    return TemporalGraphSplits(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        train_cutoff=train_cutoff,
        val_cutoff=val_cutoff,
        mappings=mappings,
    )


def save_graph_splits(splits: TemporalGraphSplits, output_dir: Path | str) -> dict[str, Path]:
    """Persist PyG graph objects and entity mappings to disk.

    Args:
        splits: TemporalGraphSplits instance.
        output_dir: Directory where artifacts are saved.

    Returns:
        Dictionary of saved file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train_graph": out_dir / "train_graph.pt",
        "val_graph": out_dir / "val_graph.pt",
        "test_graph": out_dir / "test_graph.pt",
        "mappings": out_dir / "entity_mappings.pt",
    }

    torch.save(splits.train_data, paths["train_graph"])
    torch.save(splits.val_data, paths["val_graph"])
    torch.save(splits.test_data, paths["test_graph"])
    torch.save(splits.mappings, paths["mappings"])

    return paths


def main(argv: list[str] | None = None) -> int:
    """Run heterogeneous graph construction CLI."""
    import argparse

    from fraudgnn.config import load_graph_config

    parser = argparse.ArgumentParser(
        description="Build PyG HeteroData graph from transaction parquet."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/graph.yaml",
        help="Path to YAML configuration file (default: configs/graph.yaml)",
    )
    args = parser.parse_args(argv)

    config = load_graph_config(args.config)
    print(f"Loading transactions from: {config.data_path}")
    df = pd.read_parquet(config.data_path)

    print(f"Building temporal HeteroData graph splits ({len(df):,} transactions)...")
    splits = create_temporal_splits(df, config)

    saved_paths = save_graph_splits(splits, config.processed_dir)
    print("Graph construction complete! Artifacts saved:")
    for k, p in saved_paths.items():
        print(f"  - {k:<15}: {p}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
