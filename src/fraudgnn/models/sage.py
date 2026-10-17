"""GraphSAGE edge classifier on homogeneous representations of the graph.

Converts the multi-relational transaction graph into a homogeneous topology
and performs inductive neighborhood aggregation via SAGEConv layers, followed
by an edge classification MLP decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import SAGEConv

from fraudgnn.models.hetero_gat import EdgeDecoder


def convert_hetero_to_homogeneous(
    data: HeteroData,
    target_edge_type: tuple[str, str, str] = ("user", "transacts_at", "merchant"),
) -> tuple[Data, dict[str, int], torch.Tensor, torch.Tensor | None]:
    """Convert HeteroData graph into a homogeneous PyG Data object with tracked transaction edges.

    Computes node offsets per node type and transforms all multi-relational edge
    indices into a unified index space. Preserves target transaction edges and features.

    Args:
        data: HeteroData object with node types and edge relations.
        target_edge_type: Relation triplet for the primary transaction edges.

    Returns:
        Tuple containing:
            1. Homogeneous PyG Data object.
            2. Dictionary mapping node_type string -> global index offset.
            3. Homogeneous target edge index tensor of shape [2, E_target].
            4. Target edge attributes tensor (or None).
    """
    node_types = sorted(data.node_types)
    node_offsets: dict[str, int] = {}
    total_nodes = 0
    x_list: list[torch.Tensor] = []
    type_list: list[torch.Tensor] = []

    for idx, nt in enumerate(node_types):
        node_offsets[nt] = total_nodes
        n_count = data[nt].num_nodes
        x_nt = getattr(data[nt], "x", None)
        if x_nt is None or x_nt.size(0) == 0:
            x_nt = torch.zeros((n_count, 8), dtype=torch.float32)
        x_list.append(x_nt)
        type_list.append(torch.full((n_count,), idx, dtype=torch.long))
        total_nodes += n_count

    all_x = torch.cat(x_list, dim=0)
    all_node_types = torch.cat(type_list, dim=0)

    # Accumulate all edges with proper offsets
    edge_index_list: list[torch.Tensor] = []
    edge_time_list: list[torch.Tensor] = []
    edge_attr_list: list[torch.Tensor] = []
    for edge_type in data.edge_types:
        src_t, _, dst_t = edge_type
        e_idx = data[edge_type].edge_index
        n_e = e_idx.size(1)
        if n_e > 0:
            shifted = torch.stack(
                [
                    e_idx[0] + node_offsets[src_t],
                    e_idx[1] + node_offsets[dst_t],
                ],
                dim=0,
            )
            edge_index_list.append(shifted)
            if hasattr(data[edge_type], "edge_time"):
                edge_time_list.append(data[edge_type].edge_time)
            else:
                edge_time_list.append(torch.zeros(n_e, dtype=torch.long))

            if hasattr(data[edge_type], "edge_attr") and data[edge_type].edge_attr is not None:
                edge_attr_list.append(data[edge_type].edge_attr)
            else:
                target_attr = getattr(data[target_edge_type], "edge_attr", None)
                dim = target_attr.size(1) if target_attr is not None else 27
                edge_attr_list.append(torch.zeros((n_e, dim), dtype=torch.float32))

    all_edge_index = (
        torch.cat(edge_index_list, dim=1)
        if edge_index_list
        else torch.empty((2, 0), dtype=torch.long)
    )
    all_edge_time = (
        torch.cat(edge_time_list, dim=0) if edge_time_list else torch.empty(0, dtype=torch.long)
    )
    all_edge_attr = (
        torch.cat(edge_attr_list, dim=0)
        if edge_attr_list
        else torch.empty((0, 27), dtype=torch.float32)
    )

    homo_data = Data(x=all_x, edge_index=all_edge_index)
    homo_data.node_type = all_node_types
    homo_data.num_nodes = total_nodes
    homo_data.edge_time = all_edge_time
    homo_data.edge_attr = all_edge_attr

    # Target transaction edges mapping
    target_rel = data[target_edge_type]
    t_idx = target_rel.edge_index
    src_t, _, dst_t = target_edge_type
    target_homo_idx = torch.stack(
        [
            t_idx[0] + node_offsets[src_t],
            t_idx[1] + node_offsets[dst_t],
        ],
        dim=0,
    )
    target_attr = getattr(target_rel, "edge_attr", None)

    return homo_data, node_offsets, target_homo_idx, target_attr


class GraphSAGEEdgeClassifier(nn.Module):
    """Homogeneous GraphSAGE model for edge-level fraud classification.

    Applies sequential SAGEConv layers across the unified graph, then evaluates
    the edge decoder MLP on the concatenated endpoint embeddings and edge features.

    Args:
        in_channels: Input node feature dimensionality.
        hidden_channels: Hidden representation dimension.
        edge_dim: Transaction edge attribute dimension.
        num_layers: Number of SAGE message passing layers (1 or 2).
        dropout: Dropout rate.
        use_edge_attr: Whether to pass edge attributes to the edge decoder.
        aggr: Aggregation scheme ('mean', 'max', or 'lstm').
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 64,
        edge_dim: int = 27,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_edge_attr: bool = True,
        aggr: str = "mean",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.use_edge_attr = use_edge_attr

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
        )

        # Message passing layers
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.layer_norms.append(nn.LayerNorm(hidden_channels))

        self.dropout = nn.Dropout(dropout)

        # Edge classification decoder
        decoder_in_dim = 2 * hidden_channels + (edge_dim if use_edge_attr else 0)
        self.decoder = EdgeDecoder(
            in_dim=decoder_in_dim,
            hidden_dim=hidden_channels,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for homogeneous edge classification.

        Args:
            x: Node feature matrix of shape [N, in_channels].
            edge_index: Homogeneous connectivity graph of shape [2, E_all].
            edge_label_index: Target edges to classify of shape [2, E_target].
            edge_attr: Optional edge attributes of shape [E_target, edge_dim].

        Returns:
            Fraud prediction logits of shape [E_target].
        """
        # 1. Project input features
        h = self.input_proj(x)

        # 2. Sequential SAGE convolution with residual connections
        for layer_idx in range(self.num_layers):
            conv_out = self.convs[layer_idx](h, edge_index)
            res = self.layer_norms[layer_idx](h + conv_out)
            h = self.dropout(torch.relu(res))

        # 3. Decode target edges
        src_nodes = edge_label_index[0]
        dst_nodes = edge_label_index[1]
        src_emb = h[src_nodes]
        dst_emb = h[dst_nodes]

        attr = edge_attr if self.use_edge_attr else None
        logits: torch.Tensor = self.decoder(src_emb, dst_emb, edge_attr=attr)
        return logits
