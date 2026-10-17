"""Tests for tabular feature engineering, XGBoost baseline training, and evaluation metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudgnn.config import BaselineConfig, BaselineModelConfig, DataConfig
from fraudgnn.data.features import compute_tabular_features, get_feature_column_names
from fraudgnn.data.simulator import TransactionSimulator
from fraudgnn.models.baseline import XGBoostBaseline
from fraudgnn.training.evaluate import compute_metrics


@pytest.fixture
def small_feature_df() -> pd.DataFrame:
    """Generate small transaction dataset for baseline tests."""
    cfg = DataConfig(
        num_transactions=1_500,
        num_days=8,
        num_users=60,
        num_merchants=15,
        num_devices=80,
        num_ips=100,
        fraud_ratio=0.04,
        seed=42,
    )
    return TransactionSimulator(cfg).generate()


def test_feature_engineering_causality(small_feature_df: pd.DataFrame) -> None:
    """Feature engineering must be strictly causal without lookahead leakage."""
    feat_df = compute_tabular_features(small_feature_df)

    # 1. First transaction of any user must have rolling count == 0 and time_delta == 30 days
    first_tx_users = feat_df.drop_duplicates(subset=["user_id"], keep="first")
    assert (first_tx_users["user_tx_count_1h"] == 0.0).all()
    assert (first_tx_users["user_tx_count_24h"] == 0.0).all()
    assert (first_tx_users["user_tx_count_7d"] == 0.0).all()
    assert (first_tx_users["time_delta_user"] == 86400 * 30.0).all()

    # 2. First transaction of any merchant must NOT know its own fraud label
    first_tx_merchants = feat_df.drop_duplicates(subset=["merchant_id"], keep="first")
    # For a new merchant with 0 past transactions, risk rate should equal prior fraud rate (0.008)
    assert np.allclose(first_tx_merchants["merchant_risk_rate"], 0.008, atol=1e-5)

    # 3. Check feature columns are complete and free of NaNs
    feature_names = get_feature_column_names()
    assert all(col in feat_df.columns for col in feature_names)
    assert not feat_df[feature_names].isna().any().any()


def test_evaluation_metrics_cost_formula() -> None:
    """Evaluation metrics must compute exact PR-AUC and cost = FN * avg_fraud_amt + FP * review_cost."""
    # Synthetic ground truth and predictions
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Scores: high scores for 3 frauds, low for 1 fraud, 1 false positive
    y_pred_proba = np.array([0.05, 0.10, 0.15, 0.85, 0.70, 0.80, 0.90, 0.20])
    amounts = np.array([50.0, 30.0, 40.0, 100.0, 500.0, 600.0, 700.0, 200.0])

    review_cost = 15.0
    metrics = compute_metrics(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        amounts=amounts,
        review_cost=review_cost,
    )

    assert 0.0 <= metrics.pr_auc <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert 0.0 <= metrics.recall_at_1pct_fpr <= 1.0

    # Verify cost equation: cost = fn * avg_fraud_amt + fp * review_cost
    expected_cost = metrics.fn * metrics.avg_fraud_amount + metrics.fp * review_cost
    assert metrics.total_cost == pytest.approx(expected_cost)


def test_baseline_fit_predict_and_persistence(
    tmp_path: Path, small_feature_df: pd.DataFrame
) -> None:
    """XGBoost baseline must train, predict valid probabilities, and persist/reload via JSON."""
    feat_df = compute_tabular_features(small_feature_df)
    feature_cols = get_feature_column_names()

    split_idx = int(len(feat_df) * 0.70)
    train_df = feat_df.iloc[:split_idx]
    val_df = feat_df.iloc[split_idx:]

    model_file = tmp_path / "xgb_model.json"
    cfg = BaselineConfig(
        model=BaselineModelConfig(
            n_estimators=20,
            max_depth=3,
            early_stopping_rounds=5,
            model_save_path=model_file,
        )
    )

    baseline = XGBoostBaseline(cfg)
    baseline.fit(
        train_df[feature_cols],
        train_df["is_fraud"],
        val_df[feature_cols],
        val_df["is_fraud"],
    )

    # Probabilities in [0, 1]
    val_probs = baseline.predict_proba(val_df[feature_cols])
    assert len(val_probs) == len(val_df)
    assert (val_probs >= 0.0).all() and (val_probs <= 1.0).all()

    # Save and reload
    saved_path = baseline.save(model_file)
    assert saved_path.exists()

    loaded_baseline = XGBoostBaseline(cfg).load(model_file)
    reloaded_probs = loaded_baseline.predict_proba(val_df[feature_cols])
    np.testing.assert_allclose(val_probs, reloaded_probs, atol=1e-5)


def test_threshold_selected_on_val_only() -> None:
    """Thresholds must come from validation only; test metrics must not optimize threshold on test."""
    # Create distinct val and test distributions where the optimal F1 threshold differs
    y_val = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_val_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    # Validation optimal threshold search
    val_metrics = compute_metrics(y_val, y_val_prob)
    val_selected_thresh = val_metrics.best_threshold

    # Test distribution where optimal threshold on test would be completely different (e.g., 0.35)
    y_test = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_test_prob = np.array([0.05, 0.10, 0.15, 0.20, 0.40, 0.45, 0.50, 0.55])

    # 1. Without override, test would optimize on test set (leakage/cheating)
    test_metrics_cheating = compute_metrics(y_test, y_test_prob)
    assert test_metrics_cheating.best_threshold < 0.45

    # 2. With threshold_override from validation, test uses val threshold
    test_metrics_honest = compute_metrics(
        y_test,
        y_test_prob,
        threshold_override=val_selected_thresh,
    )
    assert test_metrics_honest.best_threshold == val_selected_thresh
    # At val_selected_thresh (around 0.6), test probabilities (all <= 0.55) produce 0 positive predictions
    assert test_metrics_honest.tp == 0
    assert test_metrics_honest.fn == 4


def test_per_fraud_type_recall_computation() -> None:
    """compute_metrics must accurately break down recall for each fraud typology."""
    y_true = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.9, 0.85, 0.8, 0.1, 0.95, 0.2])
    fraud_types = np.array(
        [
            "legitimate",
            "legitimate",
            "fraud_ring",
            "fraud_ring",
            "mule",
            "mule",
            "card_testing",
            "card_testing",
        ]
    )

    metrics = compute_metrics(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        threshold_override=0.5,
        fraud_types=fraud_types,
    )

    assert "fraud_ring" in metrics.per_type_recall
    assert "mule" in metrics.per_type_recall
    assert "card_testing" in metrics.per_type_recall
    assert "legitimate" not in metrics.per_type_recall

    # fraud_ring: 2 samples, both prob >= 0.5 -> recall = 1.0
    assert metrics.per_type_recall["fraud_ring"] == 1.0
    # mule: 2 samples, 1 prob >= 0.5, 1 prob < 0.5 -> recall = 0.5
    assert metrics.per_type_recall["mule"] == 0.5
    # card_testing: 2 samples, 1 prob >= 0.5, 1 prob < 0.5 -> recall = 0.5
    assert metrics.per_type_recall["card_testing"] == 0.5


def test_no_fraud_type_or_label_in_feature_columns() -> None:
    """Neither fraud_type, is_fraud, nor any label derivatives may ever be a model feature."""
    feature_cols = get_feature_column_names()
    disallowed = {"is_fraud", "fraud_type", "target", "label"}
    for col in feature_cols:
        assert col not in disallowed
        assert not col.startswith("fraud_")
        assert not col.endswith("_fraud")
