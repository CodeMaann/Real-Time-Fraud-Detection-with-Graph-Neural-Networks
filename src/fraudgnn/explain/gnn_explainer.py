"""Perturbation-based explanation for HeteroGAT with device/IP binding.

Measures how much the predicted fraud probability of one transaction drops when a single
component is removed: the device or IP embedding bound to the decoder, one group of edge
features, or one sampled neighbor edge. Deltas are one-at-a-time and not additive.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch_geometric.data import HeteroData

from fraudgnn.config import GNNConfig
from fraudgnn.models.hetero_gat import HeteroGAT

DEVICE_COMPONENT = "device_binding"
IP_COMPONENT = "ip_binding"


@dataclass
class GNNExplanation:
    """Perturbation explanation for a single transaction in HeteroGAT."""

    transaction_id: str | None
    base_probability: float
    device_impact: float
    ip_impact: float
    edge_group_impacts: dict[str, float]
    top_neighbor_edges: list[dict[str, Any]]
    explanation_method: str
    method_rationale: str
    elapsed_time_ms: float

    @property
    def component_impacts(self) -> dict[str, float]:
        """All removable components (device, IP, edge-feature groups) and their deltas."""
        return {
            DEVICE_COMPONENT: self.device_impact,
            IP_COMPONENT: self.ip_impact,
            **self.edge_group_impacts,
        }


class HeteroGATExplainer:
    """Explainer for HeteroGAT using one-at-a-time perturbation ablation."""

    EDGE_FEATURE_GROUPS: ClassVar[dict[str, list[int]]] = {
        "amount": [0, 1],  # amount, log_amount
        "cyclical_time": [2, 3, 4, 5],  # hour_sin, hour_cos, day_sin, day_cos
        "time_delta": [6, 7, 8, 9],  # time_delta_card, log, time_delta_user, log
        "user_velocity": [10, 11, 12, 13, 14, 15],  # 1h, 24h, 7d counts & sums
        "card_velocity": [16, 17, 18, 19, 20, 21],  # 1h, 24h, 7d counts & sums
        "historical_merchant": [22, 23],  # device_user_count_past, log_merch_tx_count_past
        "channel": [24, 25, 26],  # channel_mobile, channel_web, channel_pos
    }

    RATIONALE: ClassVar[str] = (
        "PyG's Explainer with GNNExplainer raised a shape error on this model "
        "(mat1 and mat2 shapes cannot be multiplied: 1x256 and 283x64); it was not adapted to "
        "the device/IP-bound decoder. Perturbation ablation is used instead: each delta is the "
        "drop in predicted probability when one component is removed."
    )

    def __init__(self, model: HeteroGAT, device: torch.device | None = None) -> None:
        """Wrap a loaded HeteroGAT model.

        Args:
            model: HeteroGAT model with loaded weights.
            device: Torch execution device (defaults to CPU).
        """
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.target_rel = ("user", "transacts_at", "merchant")

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str | Path,
        gnn_config: GNNConfig,
        node_types: list[str],
        edge_types: list[tuple[str, str, str]],
        conv_type: str | None = None,
        use_device_ip: bool = True,
        device: torch.device | None = None,
    ) -> HeteroGATExplainer:
        """Build the model from its training config and load checkpoint weights.

        Args:
            model_path: Path to the saved state dict.
            gnn_config: Training config of the checkpoint (architecture source).
            node_types: Graph node types.
            edge_types: Graph edge types.
            conv_type: Override for the config's conv type (as used in training).
            use_device_ip: Whether the checkpoint was trained with device/IP binding.
            device: Torch execution device.

        Returns:
            Ready-to-use explainer.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"HeteroGAT checkpoint not found at: {model_path}")
        model = HeteroGAT(
            node_types=node_types,
            edge_types=edge_types,
            in_channels_dict={nt: 17 for nt in node_types},
            hidden_channels=gnn_config.model.hidden_dim,
            edge_dim=gnn_config.model.edge_dim,
            num_layers=gnn_config.model.num_layers,
            heads=gnn_config.model.heads,
            dropout=gnn_config.model.dropout,
            conv_type=conv_type or gnn_config.model.conv_type,
            use_edge_attr=gnn_config.model.use_edge_attr,
            use_device_ip=use_device_ip,
        )
        model.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=True))
        return cls(model, device=device)

    @torch.no_grad()
    def predict(
        self,
        batch: HeteroData,
        drop_device: bool = False,
        drop_ip: bool = False,
        zero_groups: Sequence[str] = (),
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] | None = None,
    ) -> float:
        """Probability for the first target edge, optionally with components removed.

        Args:
            batch: Single-target HeteroData batch from ``CausalLinkNeighborLoader``.
            drop_device: Remove the device embedding from the decoder input.
            drop_ip: Remove the IP embedding from the decoder input.
            zero_groups: Edge-feature groups (keys of ``EDGE_FEATURE_GROUPS``) set to zero.
            edge_index_dict: Optional replacement message-passing edges.

        Returns:
            Predicted fraud probability.
        """
        store = batch[self.target_rel]
        edge_attr = getattr(store, "edge_attr", None)
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.device)
            if zero_groups:
                edge_attr = edge_attr.clone()
                for group in zero_groups:
                    edge_attr[:, self.EDGE_FEATURE_GROUPS[group]] = 0.0
        device_index = None if drop_device else getattr(store, "device_index", None)
        ip_index = None if drop_ip else getattr(store, "ip_index", None)
        e_dict = edge_index_dict or {et: batch[et].edge_index for et in batch.edge_types}
        logits = self.model(
            x_dict={nt: batch[nt].x.to(self.device) for nt in batch.node_types},
            edge_index_dict={et: ei.to(self.device) for et, ei in e_dict.items()},
            edge_label_index=store.edge_label_index.to(self.device),
            edge_attr=edge_attr,
            device_index=device_index.to(self.device) if device_index is not None else None,
            ip_index=ip_index.to(self.device) if ip_index is not None else None,
        )
        return float(torch.sigmoid(logits[0]).item())

    def predict_without(self, batch: HeteroData, components: Sequence[str]) -> float:
        """Probability with several components (device/IP/edge groups) removed together."""
        groups = [c for c in components if c in self.EDGE_FEATURE_GROUPS]
        return self.predict(
            batch,
            drop_device=DEVICE_COMPONENT in components,
            drop_ip=IP_COMPONENT in components,
            zero_groups=groups,
        )

    def explain_subgraph(
        self,
        batch: HeteroData,
        transaction_id: str | None = None,
        max_edge_ablations: int = 50,
    ) -> GNNExplanation:
        """Compute one-at-a-time perturbation deltas for a single transaction's subgraph.

        Args:
            batch: Sampled single-target HeteroData batch.
            transaction_id: Optional ID for identification.
            max_edge_ablations: Maximum neighbor edges to drop one at a time.

        Returns:
            GNNExplanation with probability deltas (positive = removal lowers the score).
        """
        t0 = time.perf_counter()
        base_prob = self.predict(batch)
        device_impact = base_prob - self.predict(batch, drop_device=True)
        ip_impact = base_prob - self.predict(batch, drop_ip=True)

        group_impacts: dict[str, float] = {}
        edge_attr = getattr(batch[self.target_rel], "edge_attr", None)
        if edge_attr is not None:
            for group in self.EDGE_FEATURE_GROUPS:
                group_impacts[group] = base_prob - self.predict(batch, zero_groups=[group])

        e_dict = {et: batch[et].edge_index for et in batch.edge_types}
        edge_ablations: list[dict[str, Any]] = []
        for etype, e_idx in e_dict.items():
            if etype == self.target_rel:
                continue
            for edge_i in range(e_idx.size(1)):
                if len(edge_ablations) >= max_edge_ablations:
                    break
                keep = torch.ones(e_idx.size(1), dtype=torch.bool)
                keep[edge_i] = False
                dropped = self.predict(batch, edge_index_dict={**e_dict, etype: e_idx[:, keep]})
                edge_ablations.append(
                    {
                        "relation": etype,
                        "edge_idx": edge_i,
                        "source_node": int(e_idx[0, edge_i]),
                        "dest_node": int(e_idx[1, edge_i]),
                        "score_impact": base_prob - dropped,
                    }
                )
        edge_ablations.sort(key=lambda x: abs(x["score_impact"]), reverse=True)

        return GNNExplanation(
            transaction_id=transaction_id,
            base_probability=base_prob,
            device_impact=device_impact,
            ip_impact=ip_impact,
            edge_group_impacts=group_impacts,
            top_neighbor_edges=edge_ablations[:5],
            explanation_method="perturbation_ablation",
            method_rationale=self.RATIONALE,
            elapsed_time_ms=(time.perf_counter() - t0) * 1000.0,
        )
