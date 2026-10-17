"""XGBoost tabular baseline for fraud detection.

Implements gradient-boosted decision trees with:
1. Class imbalance handling via exact scale_pos_weight.
2. Early stopping based on validation PR-AUC (eval_metric="aucpr").
3. End-to-end MLflow experiment tracking: parameters, metrics, plots, and model artifact.
4. Per-fraud-type recall breakdown.
5. All thresholds selected on VALIDATION, applied to test (no test-set optimization).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

# Configure MLflow environment before imports
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb

from fraudgnn.config import BaselineConfig, load_baseline_config
from fraudgnn.data.features import compute_tabular_features, get_feature_column_names
from fraudgnn.training.evaluate import (
    EvaluationMetrics,
    compute_metrics,
    plot_confusion_matrix,
    plot_cost_curve,
    plot_pr_curve,
)


class XGBoostBaseline:
    """XGBoost Classifier baseline optimized for imbalanced fraud detection.

    Parameters:
        config: Validated BaselineConfig containing model and split hyperparameters.
    """

    def __init__(self, config: BaselineConfig) -> None:
        self.config = config
        self.model: xgb.XGBClassifier | None = None
        self.feature_names: list[str] = []
        self.scale_pos_weight: float = 1.0

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame,
        y_val: pd.Series | np.ndarray,
    ) -> XGBoostBaseline:
        """Fit the XGBoost classifier with early stopping on validation PR-AUC.

        Args:
            X_train: Training feature DataFrame.
            y_train: Training binary labels.
            X_val: Validation feature DataFrame.
            y_val: Validation binary labels.

        Returns:
            Fitted XGBoostBaseline instance.
        """
        self.feature_names = list(X_train.columns)

        y_tr_arr = np.asarray(y_train, dtype=np.int64)
        n_pos = int((y_tr_arr == 1).sum())
        n_neg = int((y_tr_arr == 0).sum())
        self.scale_pos_weight = float(n_neg / max(1, n_pos))

        cfg_m = self.config.model

        self.model = xgb.XGBClassifier(
            n_estimators=cfg_m.n_estimators,
            max_depth=cfg_m.max_depth,
            learning_rate=cfg_m.learning_rate,
            subsample=cfg_m.subsample,
            colsample_bytree=cfg_m.colsample_bytree,
            scale_pos_weight=self.scale_pos_weight,
            early_stopping_rounds=cfg_m.early_stopping_rounds,
            eval_metric="aucpr",  # Primary metric: Area under PR Curve
            random_state=cfg_m.random_state,
            tree_method="hist",  # Fast CPU histogram method
            n_jobs=-1,
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probability scores for samples.

        Args:
            X: Feature DataFrame.

        Returns:
            1D array of fraud probabilities [0, 1].
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        # Return probability of positive class (fraud)
        probs = self.model.predict_proba(X)
        return np.asarray(probs[:, 1], dtype=np.float64)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary fraud labels using specified decision threshold.

        Args:
            X: Feature DataFrame.
            threshold: Probability threshold for positive fraud classification.

        Returns:
            Binary predictions (0 or 1).
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, output_path: Path | str) -> Path:
        """Save trained model in portable XGBoost JSON format.

        Args:
            output_path: Path to write model JSON.

        Returns:
            Path of saved model.
        """
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted model.")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(out))
        return out

    def load(self, model_path: Path | str) -> XGBoostBaseline:
        """Load trained XGBoost model from JSON file.

        Args:
            model_path: Path to model JSON.

        Returns:
            Self with loaded model.
        """
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))
        return self


def train_and_evaluate_baseline(
    config: BaselineConfig,
    run_name: str = "xgboost_temporal_baseline",
) -> tuple[XGBoostBaseline, EvaluationMetrics, EvaluationMetrics]:
    """Execute end-to-end training and evaluation of the XGBoost baseline.

    Performs:
    1. Chronological feature engineering.
    2. Strict temporal splitting (train / val / test).
    3. Model training with early stopping on PR-AUC.
    4. Metric evaluation: PR-AUC, ROC-AUC, F1, recall@1%FPR, per-type recall, cost.
    5. ALL thresholds selected on validation; applied to test without re-optimization.
    6. MLflow experiment logging with visual artifacts.

    Args:
        config: Validated BaselineConfig.
        run_name: MLflow run name.

    Returns:
        Tuple of (fitted model, val_metrics, test_metrics).
    """
    print(f"\n[1/5] Loading transactions from: {config.data_path}")
    df_raw = pd.read_parquet(config.data_path)
    total_tx = len(df_raw)

    print(f"[2/5] Computing causal temporal features ({total_tx:,} transactions)...")
    t0_feat = time.perf_counter()
    df_feat = compute_tabular_features(
        df_raw,
        smoothing_alpha=config.smoothing_alpha,
    )
    feature_cols = get_feature_column_names()
    print(f"      Engineered {len(feature_cols)} features in {time.perf_counter() - t0_feat:.2f}s")

    # Temporal split
    timestamps = pd.to_datetime(df_feat["timestamp"])
    t_min = timestamps.min()
    t_max = timestamps.max()
    duration = t_max - t_min

    train_cutoff = t_min + duration * config.split.train_ratio
    val_cutoff = t_min + duration * (config.split.train_ratio + config.split.val_ratio)

    train_mask = timestamps <= train_cutoff
    val_mask = (timestamps > train_cutoff) & (timestamps <= val_cutoff)
    test_mask = timestamps > val_cutoff

    train_df = df_feat[train_mask]
    val_df = df_feat[val_mask]
    test_df = df_feat[test_mask]

    print(
        f"[3/5] Temporal Split:\n"
        f"      Train : {len(train_df):,} tx ({train_df['is_fraud'].sum():,} fraud) -> <= {train_cutoff}\n"
        f"      Val   : {len(val_df):,} tx ({val_df['is_fraud'].sum():,} fraud) -> {train_cutoff} to {val_cutoff}\n"
        f"      Test  : {len(test_df):,} tx ({test_df['is_fraud'].sum():,} fraud) -> > {val_cutoff}"
    )

    X_train = train_df[feature_cols]
    y_train = train_df["is_fraud"]

    X_val = val_df[feature_cols]
    y_val = val_df["is_fraud"]

    X_test = test_df[feature_cols]
    y_test = test_df["is_fraud"]

    # Configure MLflow
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    print(f"[4/5] Training XGBoost baseline with MLflow logging (run: {run_name})...")
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(
            {
                "n_estimators": config.model.n_estimators,
                "max_depth": config.model.max_depth,
                "learning_rate": config.model.learning_rate,
                "subsample": config.model.subsample,
                "colsample_bytree": config.model.colsample_bytree,
                "early_stopping_rounds": config.model.early_stopping_rounds,
                "train_ratio": config.split.train_ratio,
                "val_ratio": config.split.val_ratio,
                "test_ratio": config.split.test_ratio,
                "num_features": len(feature_cols),
                "review_cost": config.cost.review_cost,
                "default_fraud_amount": config.cost.default_fraud_amount,
            }
        )

        baseline = XGBoostBaseline(config)
        t0_train = time.perf_counter()
        baseline.fit(X_train, y_train, X_val, y_val)
        train_elapsed = time.perf_counter() - t0_train

        mlflow.log_metric("train_time_sec", train_elapsed)
        mlflow.log_metric("scale_pos_weight", baseline.scale_pos_weight)
        if baseline.model and hasattr(baseline.model, "best_iteration"):
            mlflow.log_metric("best_iteration", baseline.model.best_iteration)

        print(
            f"      Trained in {train_elapsed:.2f}s (scale_pos_weight={baseline.scale_pos_weight:.2f})"
        )

        # Predictions
        print("[5/5] Evaluating on validation and test sets...")
        y_val_prob = baseline.predict_proba(X_val)
        y_test_prob = baseline.predict_proba(X_test)

        # Measure inference latency
        t0_lat = time.perf_counter()
        _ = baseline.predict_proba(X_test)
        inference_latency_ms = (time.perf_counter() - t0_lat) / len(X_test) * 1000

        # Val metrics: threshold is optimized HERE on validation data
        val_metrics = compute_metrics(
            y_val.to_numpy(),
            y_val_prob,
            amounts=val_df["amount"].to_numpy(),
            review_cost=config.cost.review_cost,
            default_fraud_amount=config.cost.default_fraud_amount,
            fraud_types=val_df["fraud_type"].to_numpy(),
        )

        # Test metrics: use validation-selected threshold (NO test-set optimization)
        test_metrics = compute_metrics(
            y_test.to_numpy(),
            y_test_prob,
            amounts=test_df["amount"].to_numpy(),
            review_cost=config.cost.review_cost,
            default_fraud_amount=config.cost.default_fraud_amount,
            threshold_override=val_metrics.best_threshold,
            fraud_types=test_df["fraud_type"].to_numpy(),
        )

        # Feature importance
        importance: dict[str, float] = {}
        if baseline.model is not None:
            raw_scores = baseline.model.get_booster().get_score(importance_type="gain")
            importance = {
                k: float(v if isinstance(v, (int, float)) else v[0]) for k, v in raw_scores.items()
            }

        # Log MLflow metrics
        mlflow.log_metrics(
            {
                "val_pr_auc": val_metrics.pr_auc,
                "val_roc_auc": val_metrics.roc_auc,
                "val_f1": val_metrics.f1,
                "val_recall_1pct_fpr": val_metrics.recall_at_1pct_fpr,
                "val_total_cost": val_metrics.total_cost,
                "val_threshold": val_metrics.best_threshold,
                "test_pr_auc": test_metrics.pr_auc,
                "test_roc_auc": test_metrics.roc_auc,
                "test_f1": test_metrics.f1,
                "test_recall_1pct_fpr": test_metrics.recall_at_1pct_fpr,
                "test_total_cost": test_metrics.total_cost,
                "test_min_cost": test_metrics.min_cost,
                "test_min_cost_threshold": test_metrics.min_cost_threshold,
                "inference_latency_ms": inference_latency_ms,
            }
        )

        # Log per-fraud-type recall to MLflow
        for ft, recall in val_metrics.per_type_recall.items():
            mlflow.log_metric(f"val_recall_{ft}", recall)
        for ft, recall in test_metrics.per_type_recall.items():
            mlflow.log_metric(f"test_recall_{ft}", recall)

        # Generate & log artifact plots
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)

            pr_plot = plot_pr_curve(
                y_test.to_numpy(),
                y_test_prob,
                tmp_p / "test_pr_curve.png",
                title="XGBoost Baseline: Test Precision-Recall Curve",
            )
            cost_plot = plot_cost_curve(
                y_test.to_numpy(),
                y_test_prob,
                tmp_p / "test_cost_curve.png",
                review_cost=config.cost.review_cost,
                avg_fraud_amount=test_metrics.avg_fraud_amount,
            )
            cm_plot = plot_confusion_matrix(
                test_metrics,
                tmp_p / "test_confusion_matrix.png",
                title=f"XGBoost Baseline CM (Val Thresh={val_metrics.best_threshold:.2f})",
            )

            mlflow.log_artifact(str(pr_plot))
            mlflow.log_artifact(str(cost_plot))
            mlflow.log_artifact(str(cm_plot))

            # Save model to disk and MLflow
            model_path = baseline.save(config.model.model_save_path)
            mlflow.log_artifact(str(model_path))

    # Print baseline summary
    print("\n" + "=" * 70)
    print("  XGBOOST TABULAR BASELINE EVALUATION REPORT")
    print("=" * 70)
    print(f"Validation PR-AUC (Primary) : {val_metrics.pr_auc:.4f}")
    print(f"Validation ROC-AUC          : {val_metrics.roc_auc:.4f}")
    print(f"Validation Recall @ 1% FPR  : {val_metrics.recall_at_1pct_fpr:.4f}")
    print(
        f"Validation Best F1          : {val_metrics.f1:.4f} (Thresh: {val_metrics.best_threshold:.4f})"
    )
    print("-" * 70)
    print(f"TEST PR-AUC (Primary)       : {test_metrics.pr_auc:.4f}")
    print(f"TEST ROC-AUC                : {test_metrics.roc_auc:.4f}")
    print(f"TEST Recall @ 1% FPR        : {test_metrics.recall_at_1pct_fpr:.4f}")
    print(
        f"TEST F1 @ Val Threshold     : {test_metrics.f1:.4f} (Using Val Thresh: {val_metrics.best_threshold:.4f})"
    )
    print(
        f"TEST Confusion Matrix       : TN={test_metrics.tn:,}, FP={test_metrics.fp:,}, FN={test_metrics.fn:,}, TP={test_metrics.tp:,}"
    )
    print(f"TEST Business Cost          : ${test_metrics.total_cost:,.2f}")
    print(
        f"TEST Minimum Business Cost  : ${test_metrics.min_cost:,.2f} (at Thresh: {test_metrics.min_cost_threshold:.2f})"
    )
    print(f"Inference Latency           : {inference_latency_ms:.4f} ms/sample")

    # Feature importance
    if importance:
        print("\n" + "-" * 70)
        print("  TOP-10 FEATURE IMPORTANCE (GAIN)")
        print("-" * 70)
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for rank, (feat, gain) in enumerate(sorted_imp[:10], 1):
            print(f"  {rank:2d}. {feat:<30s} gain = {gain:,.2f}")

    # Per-fraud-type recall
    print("\n" + "-" * 70)
    print(f"  PER-FRAUD-TYPE RECALL @ Val-Selected Threshold = {val_metrics.best_threshold:.4f}")
    print("-" * 70)
    print(f"  {'Type':<25s} {'Val Recall':>12s}  {'Test Recall':>12s}")
    all_types = sorted(
        set(list(val_metrics.per_type_recall.keys()) + list(test_metrics.per_type_recall.keys()))
    )
    for ft in all_types:
        v_r = val_metrics.per_type_recall.get(ft, float("nan"))
        t_r = test_metrics.per_type_recall.get(ft, float("nan"))
        print(f"  {ft:<25s} {v_r:>12.4f}  {t_r:>12.4f}")

    print(f"\nModel saved to              : {config.model.model_save_path}")
    print(
        f"MLflow Run Logged to        : {config.mlflow_tracking_uri} ({config.mlflow_experiment_name})"
    )
    print("=" * 70 + "\n")

    return baseline, val_metrics, test_metrics


def main(argv: list[str] | None = None) -> int:
    """Run baseline training CLI."""
    parser = argparse.ArgumentParser(description="Train and evaluate XGBoost fraud baseline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML configuration file (default: configs/baseline.yaml)",
    )
    args = parser.parse_args(argv)

    config = load_baseline_config(args.config)
    train_and_evaluate_baseline(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
