"""Heterogeneous graph construction and neighborhood sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fraudgnn.graph.builder import (
        EntityMappings,
        TemporalGraphSplits,
        build_hetero_data,
        compute_edge_features,
        create_temporal_splits,
        extract_entity_mappings,
        save_graph_splits,
    )
    from fraudgnn.graph.loader import (
        FastAdjacency,
        HomogeneousLinkLoader,
        LinkNeighborLoader,
    )


def __getattr__(name: str) -> Any:
    if name in (
        "EntityMappings",
        "TemporalGraphSplits",
        "build_hetero_data",
        "compute_edge_features",
        "create_temporal_splits",
        "extract_entity_mappings",
        "save_graph_splits",
    ):
        from fraudgnn.graph import builder

        return getattr(builder, name)
    elif name in (
        "FastAdjacency",
        "HomogeneousLinkLoader",
        "LinkNeighborLoader",
    ):
        from fraudgnn.graph import loader

        return getattr(loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EntityMappings",
    "FastAdjacency",
    "HomogeneousLinkLoader",
    "LinkNeighborLoader",
    "TemporalGraphSplits",
    "build_hetero_data",
    "compute_edge_features",
    "create_temporal_splits",
    "extract_entity_mappings",
    "save_graph_splits",
]
