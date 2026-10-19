"""Comprehensive model evaluation module for fraud detection.

Implements primary metric (PR-AUC), secondary metrics (ROC-AUC, F1, recall@1%FPR),
per-fraud-type recall, confusion matrix extraction, visual diagnostic curves
(PR curve, cost-vs-threshold curve), and business-cost loss evaluation:
    Cost = FN * avg_fraud_amount + FP * review_cost

Design: ALL thresholds are selected on validation data only; test evaluation
uses the validation-selected thresholds without re-optimizing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvaluationMetrics:
    """Container for comprehensive fraud evaluation metrics."""

    pr_auc: float
    roc_auc: float
    f1: float
    best_threshold: float
    recall_at_1pct_fpr: float
    threshold_at_1pct_fpr: float
    tn: int
    fp: int
    fn: int
    tp: int
    total_cost: float
    min_cost: float
    min_cost_threshold: float
    review_cost: float
    avg_fraud_amount: float
    per_type_recall: dict[str, float] = field(default_factory=dict)
    pr_auc_ci_lower: float = 0.0
    pr_auc_ci_upper: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for logging and serialization."""
        return asdict(self)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    amounts: np.ndarray | None = None,
    review_cost: float = 10.0,
    default_fraud_amount: float = 500.0,
    threshold_steps: int = 100,
    threshold_override: float | None = None,
    fraud_types: np.ndarray | None = None,
    compute_ci: bool = False,
) -> EvaluationMetrics:
    """Compute primary and secondary fraud detection metrics including business cost.

    When ``threshold_override`` is provided, all threshold-dependent metrics
    (F1, confusion matrix, business cost) use that threshold instead of
    optimizing on the current data. This ensures test metrics use
    validation-selected thresholds (no test-set threshold selection).

    Args:
        y_true: Binary ground truth labels (0 = legit, 1 = fraud).
        y_pred_proba: Predicted probability of fraud [0, 1].
        amounts: Optional array of transaction amounts. If None, default_fraud_amount is used.
        review_cost: Cost in USD to manually investigate an alert (FP).
        default_fraud_amount: Fallback average dollar amount lost when fraud is missed (FN).
        threshold_steps: Number of threshold evaluation points.
        threshold_override: If set, use this threshold for confusion matrix and cost
            instead of optimizing on the current data. Designed for test evaluation
            using a validation-selected threshold.
        fraud_types: Optional array of fraud type strings for per-type recall.

    Returns:
        Populated EvaluationMetrics instance.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64)

    # 1. PR-AUC (Primary Metric) and ROC-AUC
    pr_auc = float(average_precision_score(y_true, y_pred_proba))
    roc_auc = float(roc_auc_score(y_true, y_pred_proba))

    # 2. Optimal F1 Score search (always computed for reference)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    # Avoid division by zero
    denom = precisions + recalls
    f1_scores = np.divide(
        2 * (precisions * recalls),
        denom,
        out=np.zeros_like(precisions),
        where=denom > 0,
    )
    best_idx = int(np.argmax(f1_scores))
    best_f1_data = float(f1_scores[best_idx])
    best_threshold_data = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.50

    # 3. Recall at 1% FPR
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
    # Find operating point where FPR <= 0.01 (1%)
    valid_fpr_idx = np.where(fpr <= 0.01)[0]
    if len(valid_fpr_idx) > 0:
        target_idx = valid_fpr_idx[-1]
        recall_at_1pct_fpr = float(tpr[target_idx])
        threshold_at_1pct_fpr = float(roc_thresholds[target_idx])
    else:
        recall_at_1pct_fpr = 0.0
        threshold_at_1pct_fpr = 1.0

    # 4. Decide which threshold to use for confusion matrix and cost
    if threshold_override is not None:
        operating_threshold = threshold_override
        # Recompute F1 at the override threshold
        y_pred_override = (y_pred_proba >= operating_threshold).astype(int)
        tp_o = int(((y_pred_override == 1) & (y_true == 1)).sum())
        fp_o = int(((y_pred_override == 1) & (y_true == 0)).sum())
        fn_o = int(((y_pred_override == 0) & (y_true == 1)).sum())
        prec_o = tp_o / max(1, tp_o + fp_o)
        rec_o = tp_o / max(1, tp_o + fn_o)
        best_f1 = float(2 * prec_o * rec_o / max(1e-9, prec_o + rec_o))
        best_threshold = operating_threshold
    else:
        best_f1 = best_f1_data
        best_threshold = best_threshold_data
        operating_threshold = best_threshold

    # 5. Average fraud amount calculation for business cost
    if amounts is not None and len(amounts) == len(y_true):
        fraud_mask = y_true == 1
        avg_fraud_amt = (
            float(amounts[fraud_mask].mean()) if fraud_mask.sum() > 0 else default_fraud_amount
        )
    else:
        avg_fraud_amt = default_fraud_amount

    # 6. Confusion matrix at operating threshold
    y_pred_best = (y_pred_proba >= operating_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_best, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())

    # Business cost at operating threshold:
    # Cost = FN * avg_fraud_amount + FP * review_cost
    total_cost = float(fn * avg_fraud_amt + fp * review_cost)

    # 7. Global Minimum Cost Threshold Search
    eval_thresholds = np.linspace(0.01, 0.99, threshold_steps)
    min_cost = float("inf")
    min_cost_thresh = 0.5

    for thresh in eval_thresholds:
        y_p = (y_pred_proba >= thresh).astype(int)
        curr_cm = confusion_matrix(y_true, y_p, labels=[0, 1])
        _, c_fp, c_fn, _ = curr_cm.ravel()
        cost_val = float(c_fn * avg_fraud_amt + c_fp * review_cost)
        if cost_val < min_cost:
            min_cost = cost_val
            min_cost_thresh = float(thresh)

    # 8. Per-fraud-type recall at operating threshold (strictly on is_fraud == 1 positives)
    per_type_recall: dict[str, float] = {}
    if fraud_types is not None and len(fraud_types) == len(y_true):
        fraud_type_arr = np.asarray(fraud_types)
        unique_types = sorted(set(fraud_type_arr[y_true == 1].tolist()) - {"legitimate"})
        for ft in unique_types:
            ft_mask = (fraud_type_arr == ft) & (y_true == 1)
            n_total = int(ft_mask.sum())
            if n_total > 0:
                n_caught = int(((y_pred_best == 1) & ft_mask).sum())
                per_type_recall[ft] = float(n_caught / n_total)
            else:
                per_type_recall[ft] = 0.0

    # 9. Optional 95% Bootstrap Confidence Interval for PR-AUC
    if compute_ci:
        ci_lower, ci_upper = compute_pr_auc_bootstrap_ci(y_true, y_pred_proba)
    else:
        ci_lower, ci_upper = pr_auc, pr_auc

    return EvaluationMetrics(
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        f1=best_f1,
        best_threshold=best_threshold,
        recall_at_1pct_fpr=recall_at_1pct_fpr,
        threshold_at_1pct_fpr=threshold_at_1pct_fpr,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        total_cost=total_cost,
        min_cost=min_cost,
        min_cost_threshold=min_cost_thresh,
        review_cost=review_cost,
        avg_fraud_amount=avg_fraud_amt,
        per_type_recall=per_type_recall,
        pr_auc_ci_lower=ci_lower,
        pr_auc_ci_upper=ci_upper,
    )


def compute_pr_auc_bootstrap_ci(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bootstraps: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute (1 - alpha) bootstrap confidence interval for PR-AUC.

    Args:
        y_true: Binary ground truth labels.
        y_pred_proba: Predicted fraud probabilities.
        n_bootstraps: Number of bootstrap iterations.
        alpha: Significance level (default 0.05 for 95% CI).
        seed: Random seed.

    Returns:
        Tuple of (ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    pr_aucs: list[float] = []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        sub_y = y_true[idx]
        if sub_y.sum() == 0:
            continue
        pr_aucs.append(float(average_precision_score(sub_y, y_pred_proba[idx])))

    if not pr_aucs:
        base = float(average_precision_score(y_true, y_pred_proba))
        return base, base

    arr = np.array(pr_aucs)
    lower_p = 100.0 * (alpha / 2.0)
    upper_p = 100.0 * (1.0 - alpha / 2.0)
    return float(np.percentile(arr, lower_p)), float(np.percentile(arr, upper_p))


def plot_pr_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Path | str,
    title: str = "Precision-Recall Curve",
) -> Path:
    """Plot and save Precision-Recall curve with no-skill baseline and PR-AUC score.

    Args:
        y_true: Ground truth labels.
        y_pred_proba: Predicted probabilities.
        output_path: Path to save PNG figure.
        title: Plot title.

    Returns:
        Path to saved figure.
    """
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred_proba)
    score = average_precision_score(y_true, y_pred_proba)
    baseline = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, precisions, color="#2b5c8f", lw=2.5, label=f"Model (PR-AUC = {score:.4f})")
    ax.axhline(
        baseline,
        color="#e74c3c",
        linestyle="--",
        lw=1.5,
        label=f"No-Skill Baseline ({baseline:.4f})",
    )
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_cost_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Path | str,
    review_cost: float = 10.0,
    avg_fraud_amount: float = 500.0,
) -> Path:
    """Plot total financial business cost as a function of decision threshold.

    Cost = FN * avg_fraud_amount + FP * review_cost.

    Args:
        y_true: Binary ground truth labels.
        y_pred_proba: Predicted fraud probabilities.
        output_path: Path to save PNG figure.
        review_cost: Cost per false positive review ($ USD).
        avg_fraud_amount: Average loss per missed fraud ($ USD).

    Returns:
        Path to saved figure.
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    total_costs: list[float] = []
    fn_costs: list[float] = []
    fp_costs: list[float] = []

    for thresh in thresholds:
        y_p = (y_pred_proba >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_p, labels=[0, 1])
        _, fp_val, fn_val, _ = cm.ravel()
        cost_fp = float(fp_val * review_cost)
        cost_fn = float(fn_val * avg_fraud_amount)
        fp_costs.append(cost_fp)
        fn_costs.append(cost_fn)
        total_costs.append(cost_fp + cost_fn)

    min_idx = int(np.argmin(total_costs))
    best_thresh = thresholds[min_idx]
    best_cost = total_costs[min_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, total_costs, color="#2c3e50", lw=2.5, label="Total Financial Cost")
    ax.plot(
        thresholds,
        fn_costs,
        color="#e74c3c",
        lw=1.5,
        linestyle="--",
        label=f"FN Cost (${avg_fraud_amount:.0f}/miss)",
    )
    ax.plot(
        thresholds,
        fp_costs,
        color="#f39c12",
        lw=1.5,
        linestyle="--",
        label=f"FP Review Cost (${review_cost:.0f}/alert)",
    )
    ax.axvline(
        best_thresh,
        color="#27ae60",
        linestyle=":",
        lw=2,
        label=f"Min-Cost Threshold ({best_thresh:.2f}: ${best_cost:,.0f})",
    )

    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Total Business Cost ($ USD)", fontsize=12)
    ax.set_title("Operational Business Cost vs. Alert Threshold", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_confusion_matrix(
    metrics: EvaluationMetrics,
    output_path: Path | str,
    title: str = "Confusion Matrix (Optimal F1 Threshold)",
) -> Path:
    """Plot and save confusion matrix heatmap with cell counts and percentages.

    Args:
        metrics: EvaluationMetrics containing tn, fp, fn, tp.
        output_path: Path to save PNG figure.
        title: Title of figure.

    Returns:
        Path to saved figure.
    """
    matrix = np.array([[metrics.tn, metrics.fp], [metrics.fn, metrics.tp]])
    total = matrix.sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(matrix, cmap="Blues", alpha=0.8)
    fig.colorbar(cax)

    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            pct = val / total * 100
            ax.text(
                j,
                i,
                f"{val:,}\n({pct:.2f}%)",
                ha="center",
                va="center",
                color="black" if val < matrix.max() / 2 else "white",
                fontweight="bold",
                fontsize=11,
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Legit (0)", "Pred Fraud (1)"])
    ax.set_yticklabels(["True Legit (0)", "True Fraud (1)"])
    ax.set_title(title, fontsize=12, pad=20, fontweight="bold")
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
