"""Heterogeneous Graph Attention Network (HeteroGAT) for Transaction Fraud Detection.

Implements:
1. Per-node-type input projection layers: Projects heterogeneous node feature dimensions
   to a unified latent embedding space.
2. 2 message-passing layers: HeteroConv wrapping GATConv or SAGEConv with residual connections,
   activation, LayerNorm, and dropout.
3. Edge decoder: Concat(src_emb, dst_emb, edge_attr) -> Multi-Layer Perceptron (MLP) -> fraud logit.
   Supports ablations with or without edge features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv


class EdgeDecoder(nn.Module):
    """Decodes edge fraud logit from source node, destination node, and optional edge features.

    Architecture:
        Linear(in_dim, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, 32) -> ReLU -> Linear(32, 1)

    Args:
        in_dim: Total concatenated input dimension (src_dim + dst_dim [+ edge_dim]).
        hidden_dim: Hidden dimension for intermediate MLP projection.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through edge classification MLP.

        Args:
            src_emb: Source node representations of shape [E, D].
            dst_emb: Destination node representations of shape [E, D].
            edge_attr: Optional edge attributes of shape [E, edge_dim].

        Returns:
            Raw fraud prediction logits of shape [E].
        """
        features = [src_emb, dst_emb, edge_attr] if edge_attr is not None else [src_emb, dst_emb]

        cat_emb = torch.cat(features, dim=-1)
        logits: torch.Tensor = self.mlp(cat_emb).squeeze(-1)
        return logits


class HeteroGAT(nn.Module):
    """Heterogeneous Graph Neural Network for Edge-Level Fraud Classification.

    Combines per-node-type input linear projections, two relational message passing
    layers using HeteroConv, and an edge classification decoder.

    Args:
        node_types: List of node types present in the heterogeneous graph.
        edge_types: List of relation triplets (src_type, relation, dst_type).
        in_channels_dict: Mapping of node_type -> input feature dimension.
        hidden_channels: Dimension of node latent representations.
        edge_dim: Dimension of transaction edge attributes.
        num_layers: Number of message passing layers (1 or 2).
        heads: Number of attention heads for GATConv (hidden_channels must be divisible by heads).
        dropout: Dropout rate across message passing and decoder layers.
        conv_type: Convolution type ('gat' or 'sage').
        use_edge_attr: Whether the decoder should consume edge attributes.
        target_edge_type: The transaction relation being classified.
    """

    def __init__(
        self,
        node_types: list[str],
        edge_types: list[tuple[str, str, str]],
        in_channels_dict: dict[str, int],
        hidden_channels: int = 64,
        edge_dim: int = 27,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        conv_type: str = "gat",
        use_edge_attr: bool = True,
        target_edge_type: tuple[str, str, str] = ("user", "transacts_at", "merchant"),
    ) -> None:
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_channels = hidden_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.conv_type = conv_type.lower()
        self.use_edge_attr = use_edge_attr
        self.target_edge_type = target_edge_type

        # 1. Per-node-type input projections
        self.input_projections = nn.ModuleDict()
        for nt in node_types:
            in_dim = in_channels_dict.get(nt, 16)
            self.input_projections[nt] = nn.Sequential(
                nn.Linear(in_dim, hidden_channels),
                nn.ReLU(),
            )

        # 2. Relational Message Passing Layers
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        head_out_dim = hidden_channels // heads if self.conv_type == "gat" else hidden_channels

        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                if self.conv_type == "gat":
                    conv_dict[edge_type] = GATConv(
                        in_channels=(-1, -1),
                        out_channels=head_out_dim,
                        heads=heads,
                        add_self_loops=False,
                        dropout=dropout,
                    )
                elif self.conv_type in ("sum", "sage_sum"):
                    conv_dict[edge_type] = SAGEConv(
                        in_channels=(-1, -1),
                        out_channels=hidden_channels,
                        aggr="sum",
                    )
                elif self.conv_type in ("pna", "multi_aggr"):
                    conv_dict[edge_type] = SAGEConv(
                        in_channels=(-1, -1),
                        out_channels=hidden_channels,
                        aggr=["mean", "sum", "max"],
                    )
                elif self.conv_type == "sage":
                    conv_dict[edge_type] = SAGEConv(
                        in_channels=(-1, -1),
                        out_channels=hidden_channels,
                        aggr="mean",
                    )
                else:
                    raise ValueError(
                        f"Unsupported conv_type: {self.conv_type}. Use 'gat', 'sum', 'pna', or 'sage'"
                    )

            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

            norm_dict = nn.ModuleDict({nt: nn.LayerNorm(hidden_channels) for nt in node_types})
            self.layer_norms.append(norm_dict)

        # 3. Edge classification decoder
        decoder_in_dim = 2 * hidden_channels + (edge_dim if use_edge_attr else 0)
        self.decoder = EdgeDecoder(
            in_dim=decoder_in_dim,
            hidden_dim=hidden_channels,
            dropout=dropout,
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        edge_label_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for heterogeneous edge classification.

        Args:
            x_dict: Dictionary of node features per node type.
            edge_index_dict: Dictionary of edge index tensors per relation type.
            edge_label_index: Tensor of shape [2, E] containing source and target node
                indices for the target edges to classify.
            edge_attr: Optional tensor of shape [E, edge_dim] with transaction attributes.

        Returns:
            Logits of shape [E] predicting fraud for each edge in edge_label_index.
        """
        # 1. Project input representations to uniform hidden_channels
        h_dict: dict[str, torch.Tensor] = {}
        for nt, x in x_dict.items():
            if nt in self.input_projections:
                h_dict[nt] = self.input_projections[nt](x)
            else:
                h_dict[nt] = x

        # 2. Sequential message passing
        for layer_idx in range(self.num_layers):
            out_conv = self.convs[layer_idx](h_dict, edge_index_dict)
            new_h_dict: dict[str, torch.Tensor] = {}
            for nt in self.node_types:
                if nt in out_conv:
                    # Residual skip + LayerNorm + Activation + Dropout
                    res = h_dict[nt] + out_conv[nt]
                    norm_dict = self.layer_norms[layer_idx]
                    assert isinstance(norm_dict, nn.ModuleDict)
                    res = norm_dict[nt](res)
                    new_h_dict[nt] = self.dropout(torch.relu(res))
                else:
                    new_h_dict[nt] = h_dict[nt]
            h_dict = new_h_dict

        # 3. Extract source and destination embeddings for target transaction edges
        src_type, _, dst_type = self.target_edge_type
        src_nodes = edge_label_index[0]
        dst_nodes = edge_label_index[1]

        src_emb = h_dict[src_type][src_nodes]
        dst_emb = h_dict[dst_type][dst_nodes]

        attr = edge_attr if self.use_edge_attr else None
        logits: torch.Tensor = self.decoder(src_emb, dst_emb, edge_attr=attr)
        return logits
