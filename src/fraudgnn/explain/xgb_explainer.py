"""SHAP TreeExplainer for the label-free XGBoost + graph features model.

Computes exact Tree SHAP attributions and formats the top contributing
features into reason codes that state the value, an optional data-derived reference
(e.g. training-split median) and the sign of the SHAP contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import shap
import xgboost as xgb


@dataclass
class FeatureExplanation:
    """Detailed attribution for a single feature."""

    feature_name: str
    feature_value: float
    shap_value: float
    direction: str  # "increases_risk" or "decreases_risk"
    reason_code: str


@dataclass
class TransactionExplanation:
    """Complete SHAP explanation for a single transaction."""

    transaction_id: str | None
    predicted_probability: float
    base_value: float
    top_features: list[FeatureExplanation]
    all_shap_values: dict[str, float]


class XGBoostExplainer:
    """TreeExplainer wrapper for label-free XGBoost with graph features."""

    FEATURE_LABELS: ClassVar[dict[str, str]] = {
        "device_distinct_users": "Distinct accounts seen on this device before the transaction",
        "ip_distinct_users": "Distinct accounts seen on this IP before the transaction",
        "two_hop_ip_users": "Other accounts reachable through this IP (2 hops)",
        "two_hop_device_users": "Other accounts reachable through this device (2 hops)",
        "user_device_count": "Distinct devices used by this account",
        "user_ip_count": "Distinct IPs used by this account",
        "user_card_count": "Distinct cards used by this account",
        "merchant_degree": "Distinct accounts seen at this merchant",
        "card_tx_count_1h": "Card transactions in the previous hour",
        "card_amt_sum_1h": "Card spend in the previous hour",
        "card_tx_count_24h": "Card transactions in the previous 24 hours",
        "card_amt_sum_24h": "Card spend in the previous 24 hours",
        "user_tx_count_1h": "Account transactions in the previous hour",
        "user_amt_sum_1h": "Account spend in the previous hour",
        "user_tx_count_24h": "Account transactions in the previous 24 hours",
        "user_amt_sum_24h": "Account spend in the previous 24 hours",
        "amount": "Transaction amount",
        "time_delta_card": "Seconds since the card's previous transaction",
        "time_delta_user": "Seconds since the account's previous transaction",
    }

    def __init__(
        self,
        model_path: str = "models/xgboost_label_free.json",
        feature_names: list[str] | None = None,
        reference_values: dict[str, float] | None = None,
        reference_name: str = "training median",
    ) -> None:
        """Initialize SHAP explainer with trained XGBoost model.

        Args:
            model_path: Path to serialized XGBoost model json.
            feature_names: Optional explicit list of feature names.
            reference_values: Optional per-feature reference (e.g. training-split median)
                quoted next to the value in reason codes and used by the deletion test.
            reference_name: How the reference is described in reason codes.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Trained model not found at: {model_path}")

        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))
        self.booster = self.model.get_booster()

        if feature_names is not None:
            self.feature_names = list(feature_names)
        elif self.booster.feature_names is not None:
            self.feature_names = list(self.booster.feature_names)
        else:
            raise ValueError("Feature names could not be determined from model booster.")

        self.reference_values = dict(reference_values or {})
        self.reference_name = reference_name
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        """Return fraud probabilities for feature rows (columns in model order)."""
        proba: np.ndarray = self.model.predict_proba(rows[self.feature_names])[:, 1]
        return proba

    def _format_reason_code(self, feature: str, val: float, shap_val: float) -> str:
        """Describe one feature value and the direction of its SHAP contribution."""
        label = self.FEATURE_LABELS.get(feature, f"Feature '{feature}'")
        effect = "raises" if shap_val > 0 else "lowers"
        text = f"{label}: {val:,.2f}"
        if feature in self.reference_values:
            text += f" ({self.reference_name} {self.reference_values[feature]:,.2f})"
        return f"{text}; this {effect} the model score (SHAP {shap_val:+.3f} log-odds)"

    def explain_transaction(
        self,
        features: pd.Series | pd.DataFrame | np.ndarray | dict[str, float],
        transaction_id: str | None = None,
        top_k: int = 5,
    ) -> TransactionExplanation:
        """Compute top-K SHAP feature attributions and reason codes for a single transaction.

        Args:
            features: Feature values as Series, DataFrame, 1D array, or dict.
            transaction_id: Optional transaction identifier for tracking.
            top_k: Number of top features to report.

        Returns:
            TransactionExplanation with reason codes, attributions, and predicted probability.
        """
        if isinstance(features, dict):
            row_df = pd.DataFrame([{col: features.get(col, 0.0) for col in self.feature_names}])
        elif isinstance(features, pd.Series):
            row_df = pd.DataFrame([features[self.feature_names].to_dict()])
        elif isinstance(features, pd.DataFrame):
            row_df = features[self.feature_names].iloc[[0]]
        else:
            arr = np.asarray(features)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            row_df = pd.DataFrame(arr, columns=self.feature_names)

        # Predict probability
        prob = float(self.model.predict_proba(row_df)[0, 1])

        # Compute SHAP values
        shap_vals = self.explainer.shap_values(row_df)
        if isinstance(shap_vals, list):
            # Binary classification list: [class_0, class_1]
            s_vec = np.asarray(shap_vals[1][0])
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
            s_vec = shap_vals[0]
        else:
            s_vec = np.asarray(shap_vals)

        base_val = (
            float(self.explainer.expected_value)
            if not isinstance(self.explainer.expected_value, list)
            else float(self.explainer.expected_value[1])
        )

        all_dict: dict[str, float] = {}
        contributions: list[FeatureExplanation] = []

        for name, s_val in zip(self.feature_names, s_vec, strict=False):
            f_val = float(row_df[name].iloc[0])
            s_float = float(s_val)
            all_dict[name] = s_float

            direction = "increases_risk" if s_float > 0.0 else "decreases_risk"
            reason = self._format_reason_code(name, f_val, s_float)
            contributions.append(
                FeatureExplanation(
                    feature_name=name,
                    feature_value=f_val,
                    shap_value=s_float,
                    direction=direction,
                    reason_code=reason,
                )
            )

        # Sort by absolute SHAP impact (or highest positive push towards fraud)
        contributions.sort(key=lambda x: abs(x.shap_value), reverse=True)
        top_contributions = contributions[:top_k]

        return TransactionExplanation(
            transaction_id=transaction_id,
            predicted_probability=prob,
            base_value=base_val,
            top_features=top_contributions,
            all_shap_values=all_dict,
        )
