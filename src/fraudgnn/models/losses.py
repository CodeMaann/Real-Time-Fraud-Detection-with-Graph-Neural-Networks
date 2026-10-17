"""Configurable loss functions for fraud detection with Graph Neural Networks.

Provides implementations for:
1. WeightedBCEWithLogitsLoss: Binary cross-entropy with positive class weighting
   to handle extreme class imbalance (~0.8% - 2% fraud prevalence).
2. FocalLoss: Lin et al. (2017) focal loss for addressing severe class imbalance
   by down-weighting well-classified easy negatives and focusing on hard fraud examples.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class WeightedBCEWithLogitsLoss(nn.Module):
    """Numerically stable binary cross-entropy with configurable positive weighting.

    Args:
        pos_weight: Multiplier for positive class loss (e.g. n_neg / n_pos).
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        pos_weight: float | torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.reduction = reduction
        if pos_weight is not None:
            if isinstance(pos_weight, (int, float)):
                weight_tensor = torch.tensor([float(pos_weight)], dtype=torch.float32)
            else:
                weight_tensor = pos_weight.clone().detach().to(dtype=torch.float32)
            self.register_buffer("pos_weight", weight_tensor)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted binary cross entropy with logits.

        Args:
            logits: Predicted logits of shape [N] or [N, 1].
            targets: Binary ground truth labels of same shape, values in {0, 1}.

        Returns:
            Computed scalar loss (if reduction in ('mean', 'sum')) or per-element tensor.
        """
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1).to(dtype=logits_flat.dtype)

        pos_w = self.pos_weight
        if pos_w is not None and pos_w.device != logits.device:
            pos_w = pos_w.to(logits.device)

        return F.binary_cross_entropy_with_logits(
            logits_flat,
            targets_flat,
            pos_weight=pos_w,
            reduction=self.reduction,
        )


class FocalLoss(nn.Module):
    """Binary Focal Loss with support for class weighting and focusing parameter.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where p_t is the model's estimated probability for the ground-truth class,
    gamma is the focusing parameter (gamma >= 0), and alpha is the class weight
    balancing positive vs. negative examples.

    Args:
        gamma: Focusing parameter. When gamma=0, equivalent to standard BCE.
        alpha: Weight for positive class in [0, 1], (1 - alpha) for negative class.
            If None, no alpha weighting is applied.
        pos_weight: Optional additional scalar multiplier for the positive class.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = 0.25,
        pos_weight: float | torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"Focal loss gamma must be non-negative, got {gamma}")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

        if pos_weight is not None:
            if isinstance(pos_weight, (int, float)):
                weight_tensor = torch.tensor([float(pos_weight)], dtype=torch.float32)
            else:
                weight_tensor = pos_weight.clone().detach().to(dtype=torch.float32)
            self.register_buffer("pos_weight", weight_tensor)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute binary focal loss from logits.

        Args:
            logits: Predicted logits of shape [N] or [N, 1].
            targets: Binary ground truth labels of same shape, values in {0, 1}.

        Returns:
            Computed scalar loss (if reduction in ('mean', 'sum')) or per-element tensor.
        """
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1).to(dtype=logits_flat.dtype)

        pos_w = self.pos_weight
        if pos_w is not None and pos_w.device != logits.device:
            pos_w = pos_w.to(logits.device)

        # Base binary cross entropy per element without reduction
        bce = F.binary_cross_entropy_with_logits(
            logits_flat,
            targets_flat,
            pos_weight=pos_w,
            reduction="none",
        )

        # Probability p_t = exp(-bce) is numerically stable
        p_t = torch.exp(-bce)
        modulating_factor = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = targets_flat * self.alpha + (1.0 - targets_flat) * (1.0 - self.alpha)
            loss = alpha_t * modulating_factor * bce
        else:
            loss = modulating_factor * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


def get_loss_fn(
    loss_type: str,
    pos_weight: float | torch.Tensor | None = None,
    gamma: float = 2.0,
    alpha: float | None = 0.25,
    reduction: str = "mean",
) -> nn.Module:
    """Factory function to build a configurable loss module.

    Args:
        loss_type: 'bce', 'weighted_bce', or 'focal'.
        pos_weight: Optional positive class loss multiplier.
        gamma: Focusing parameter for focal loss.
        alpha: Positive class weight factor for focal loss.
        reduction: Reduction method ('mean', 'sum', 'none').

    Returns:
        Configured PyTorch loss module.
    """
    normalized = loss_type.lower().strip()
    if normalized in ("weighted_bce", "bce"):
        return WeightedBCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)
    elif normalized == "focal":
        return FocalLoss(
            gamma=gamma,
            alpha=alpha,
            pos_weight=pos_weight,
            reduction=reduction,
        )
    else:
        raise ValueError(
            f"Unsupported loss_type: '{loss_type}'. Expected one of: 'bce', 'weighted_bce', 'focal'"
        )
