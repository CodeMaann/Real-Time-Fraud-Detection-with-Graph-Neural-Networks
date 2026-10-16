"""Training loops, evaluation, and MLflow experiment logging."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fraudgnn.training.evaluate import (
        EvaluationMetrics,
        compute_metrics,
        plot_confusion_matrix,
        plot_cost_curve,
        plot_pr_curve,
    )


def __getattr__(name: str) -> Any:
    if name in (
        "EvaluationMetrics",
        "compute_metrics",
        "plot_confusion_matrix",
        "plot_cost_curve",
        "plot_pr_curve",
    ):
        from fraudgnn.training import evaluate

        return getattr(evaluate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EvaluationMetrics",
    "compute_metrics",
    "plot_confusion_matrix",
    "plot_cost_curve",
    "plot_pr_curve",
]
