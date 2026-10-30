"""Reason codes for one model's explanation of one flagged transaction.

Every explanation describes exactly one model: the sentences come only from that model's
attributions (SHAP for XGBoost, perturbation deltas for HeteroGAT) and quote that model's
score and validation-selected threshold. Mixing models in one explanation would attribute a
score to evidence the scoring model never used.
"""

from __future__ import annotations

from fraudgnn.explain.gnn_explainer import DEVICE_COMPONENT, IP_COMPONENT, GNNExplanation
from fraudgnn.explain.xgb_explainer import TransactionExplanation

MIN_SENTENCES = 3
MAX_SENTENCES = 5

COMPONENT_LABELS = {
    DEVICE_COMPONENT: "the device embedding bound to the decoder",
    IP_COMPONENT: "the IP embedding bound to the decoder",
    "amount": "the amount features",
    "cyclical_time": "the time-of-day and day-of-week features",
    "time_delta": "the time-since-previous-transaction features",
    "user_velocity": "the account velocity features",
    "card_velocity": "the card velocity features",
    "historical_merchant": "the past device-sharing and merchant-volume features",
    "channel": "the channel features",
}


def build_reason_codes(
    model_name: str,
    threshold: float,
    xgb_explanation: TransactionExplanation | None = None,
    gnn_explanation: GNNExplanation | None = None,
    transaction_id: str | None = None,
) -> list[str]:
    """Build 3 to 5 sentences explaining one model's score for one transaction.

    Args:
        model_name: Human-readable name of the scoring model.
        threshold: That model's validation-selected decision threshold.
        xgb_explanation: SHAP explanation (pass this or ``gnn_explanation``, not both).
        gnn_explanation: Perturbation explanation (pass this or ``xgb_explanation``).
        transaction_id: Optional transaction ID quoted in the first sentence.

    Returns:
        List of 3 to 5 sentences.

    Raises:
        ValueError: If not exactly one explanation is given.
    """
    if (xgb_explanation is None) == (gnn_explanation is None):
        raise ValueError("Pass exactly one model explanation (XGBoost or HeteroGAT).")

    tx = f"Transaction {transaction_id}" if transaction_id else "This transaction"
    sentences: list[str] = []

    if xgb_explanation is not None:
        prob = xgb_explanation.predicted_probability
        sentences.append(
            f"{tx} scored {prob:.3f} with {model_name}; its validation-selected threshold "
            f"is {threshold:.3f}."
        )
        raising = [f for f in xgb_explanation.top_features if f.shap_value > 0]
        lowering = [f for f in xgb_explanation.top_features if f.shap_value <= 0]
        for feat in raising[: MAX_SENTENCES - 2]:
            sentences.append(f"{feat.reason_code}.")
        if lowering and len(sentences) < MAX_SENTENCES - 1:
            sentences.append(f"Pulling the other way: {lowering[0].reason_code}.")
        sentences.append(
            f"SHAP values are relative to the model's average log-odds "
            f"({xgb_explanation.base_value:+.3f}) and add up to this transaction's score."
        )
    else:
        assert gnn_explanation is not None
        sentences.append(
            f"{tx} scored {gnn_explanation.base_probability:.3f} with {model_name} on its "
            f"own causal subgraph; its validation-selected threshold is {threshold:.3f}."
        )
        ranked = sorted(
            gnn_explanation.component_impacts.items(), key=lambda kv: kv[1], reverse=True
        )
        for component, delta in [kv for kv in ranked if kv[1] > 0][: MAX_SENTENCES - 3]:
            sentences.append(
                f"Removing {COMPONENT_LABELS.get(component, component)} lowers the score "
                f"by {delta:.3f}."
            )
        top_edges = [e for e in gnn_explanation.top_neighbor_edges if e["score_impact"] > 0]
        if top_edges:
            edge = top_edges[0]
            relation = edge["relation"][1].replace("_", " ")
            sentences.append(
                f"Dropping the single most influential sampled '{relation}' edge lowers the "
                f"score by {edge['score_impact']:.3f}."
            )
        sentences.append(
            "Each change is measured with one component removed at a time; changes are "
            "not additive."
        )

    if len(sentences) < MIN_SENTENCES:
        sentences.insert(1, "No single attribution raised the score for this transaction.")
    return sentences[:MAX_SENTENCES]
