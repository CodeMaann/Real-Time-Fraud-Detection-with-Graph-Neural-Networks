"""Model definitions: GraphSAGE, Heterogeneous GAT, XGBoost baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fraudgnn.models.baseline import XGBoostBaseline, train_and_evaluate_baseline


def __getattr__(name: str) -> Any:
    if name in ("XGBoostBaseline", "train_and_evaluate_baseline"):
        from fraudgnn.models import baseline

        return getattr(baseline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["XGBoostBaseline", "train_and_evaluate_baseline"]
