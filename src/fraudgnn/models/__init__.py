"""Model definitions: GraphSAGE, Heterogeneous GAT, XGBoost baseline, losses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fraudgnn.models.baseline import XGBoostBaseline, train_and_evaluate_baseline
    from fraudgnn.models.hetero_gat import EdgeDecoder, HeteroGAT
    from fraudgnn.models.losses import FocalLoss, WeightedBCEWithLogitsLoss, get_loss_fn
    from fraudgnn.models.sage import GraphSAGEEdgeClassifier, convert_hetero_to_homogeneous


def __getattr__(name: str) -> Any:
    if name in ("XGBoostBaseline", "train_and_evaluate_baseline"):
        from fraudgnn.models import baseline

        return getattr(baseline, name)
    elif name in ("HeteroGAT", "EdgeDecoder"):
        from fraudgnn.models import hetero_gat

        return getattr(hetero_gat, name)
    elif name in ("GraphSAGEEdgeClassifier", "convert_hetero_to_homogeneous"):
        from fraudgnn.models import sage

        return getattr(sage, name)
    elif name in ("WeightedBCEWithLogitsLoss", "FocalLoss", "get_loss_fn"):
        from fraudgnn.models import losses

        return getattr(losses, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EdgeDecoder",
    "FocalLoss",
    "GraphSAGEEdgeClassifier",
    "HeteroGAT",
    "WeightedBCEWithLogitsLoss",
    "XGBoostBaseline",
    "convert_hetero_to_homogeneous",
    "get_loss_fn",
    "train_and_evaluate_baseline",
]
