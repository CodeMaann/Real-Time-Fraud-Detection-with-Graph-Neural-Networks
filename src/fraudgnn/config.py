"""Configuration loader: YAML files → validated Pydantic models.

Design decision: every hyperparameter, path, and threshold lives in YAML
configs under ``/configs``.  This module provides a single ``load_config``
entry-point that reads a YAML file and returns a fully-validated, typed
``AppConfig`` object.  No magic numbers in code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# ── Section configs ───────────────────────────────────────────────────


class DataConfig(BaseModel):
    """Configuration for synthetic data generation and loading."""

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    output_filename: str = "transactions.parquet"
    num_transactions: int = Field(default=500_000, gt=0)
    num_days: int = Field(default=90, ge=1)
    start_date: str = "2024-01-01"
    num_users: int = Field(default=10_000, gt=0)
    num_merchants: int = Field(default=1_000, gt=0)
    num_devices: int = Field(default=15_000, gt=0)
    num_ips: int = Field(default=25_000, gt=0)
    fraud_ratio: float = Field(default=0.008, gt=0.0, lt=1.0)
    seed: int = 42

    # Fraud pattern distribution weights (relative mix of fraud types)
    fraud_ring_weight: float = Field(default=0.25, ge=0.0)
    mule_weight: float = Field(default=0.20, ge=0.0)
    card_testing_weight: float = Field(default=0.20, ge=0.0)
    account_takeover_weight: float = Field(default=0.20, ge=0.0)
    merchant_collusion_weight: float = Field(default=0.15, ge=0.0)

    # Pattern specific hyper-parameters (no magic numbers in code)
    ring_size_min: int = Field(default=5, ge=2)
    ring_size_max: int = Field(default=30, ge=5)
    mule_burst_min: int = Field(default=2, ge=1)
    mule_burst_max: int = Field(default=5, ge=2)
    card_testing_burst_min: int = Field(default=3, ge=2)
    card_testing_burst_max: int = Field(default=6, ge=3)
    card_testing_small_min: float = Field(default=0.50, gt=0.0)
    card_testing_small_max: float = Field(default=3.00, gt=0.0)
    card_testing_large_min: float = Field(default=800.0, gt=0.0)
    card_testing_large_max: float = Field(default=3000.0, gt=0.0)
    ato_amount_multiplier: float = Field(default=4.5, gt=1.0)
    collusion_fraud_rate: float = Field(default=0.65, gt=0.0, le=1.0)

    # Legitimate noise parameters
    legit_shared_device_prob: float = Field(default=0.08, ge=0.0, le=1.0)
    legit_shared_ip_prob: float = Field(default=0.15, ge=0.0, le=1.0)

    # Spending distribution parameters
    base_amount_loc: float = Field(default=3.5)
    base_amount_scale: float = Field(default=0.9, gt=0.0)

    # Circadian diurnal distribution (24 hourly weights, normalized internally)
    hourly_weights: list[float] = Field(
        default=[
            0.015,
            0.010,
            0.007,
            0.005,
            0.006,
            0.012,
            0.025,
            0.040,
            0.055,
            0.060,
            0.065,
            0.065,
            0.065,
            0.060,
            0.060,
            0.060,
            0.065,
            0.070,
            0.075,
            0.075,
            0.065,
            0.055,
            0.045,
            0.028,
        ]
    )


class TemporalSplitConfig(BaseModel):
    """Configuration for temporal train/val/test splits."""

    train_ratio: float = Field(default=0.60, gt=0.0, lt=1.0)
    val_ratio: float = Field(default=0.20, gt=0.0, lt=1.0)
    test_ratio: float = Field(default=0.20, gt=0.0, lt=1.0)


class GraphConfig(BaseModel):
    """Configuration for heterogeneous graph construction."""

    data_path: Path = Path("data/raw/transactions.parquet")
    processed_dir: Path = Path("data/processed")
    num_hops: int = Field(default=2, ge=1, le=5)
    max_neighbors_per_hop: int = Field(default=10, ge=1)
    split: TemporalSplitConfig = TemporalSplitConfig()
    node_types: list[str] = Field(
        default=["user", "card", "merchant", "device", "ip"],
    )


class BaselineModelConfig(BaseModel):
    """Configuration for XGBoost tabular baseline."""

    n_estimators: int = Field(default=300, gt=0)
    max_depth: int = Field(default=6, ge=1)
    learning_rate: float = Field(default=0.05, gt=0.0)
    subsample: float = Field(default=0.8, gt=0.0, le=1.0)
    colsample_bytree: float = Field(default=0.8, gt=0.0, le=1.0)
    early_stopping_rounds: int = Field(default=30, ge=1)
    random_state: int = 42
    model_save_path: Path = Path("models/xgboost_baseline.json")


class CostConfig(BaseModel):
    """Business cost matrix configuration."""

    review_cost: float = Field(
        default=10.0, ge=0.0, description="Cost in USD to manually review an alert (FP)"
    )
    default_fraud_amount: float = Field(
        default=500.0, gt=0.0, description="Default loss in USD if fraud is missed (FN)"
    )


class BaselineConfig(BaseModel):
    """Full configuration for tabular feature engineering and XGBoost baseline."""

    data_path: Path = Path("data/raw/transactions.parquet")
    split: TemporalSplitConfig = TemporalSplitConfig()
    model: BaselineModelConfig = BaselineModelConfig()
    cost: CostConfig = CostConfig()
    smoothing_alpha: float = Field(default=10.0, ge=0.0)
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "fraud-gnn-baseline"


class ModelConfig(BaseModel):
    """Configuration for model architecture."""

    model_type: str = Field(
        default="graphsage",
        pattern="^(graphsage|hetero_gat|xgboost)$",
    )
    hidden_dim: int = Field(default=128, gt=0)
    num_layers: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.3, ge=0.0, le=1.0)
    heads: int = Field(default=4, ge=1, description="Attention heads (GAT only)")


class TrainingConfig(BaseModel):
    """Configuration for the training loop."""

    epochs: int = Field(default=100, gt=0)
    batch_size: int = Field(default=512, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-5, ge=0.0)
    patience: int = Field(default=10, ge=1, description="Early-stopping patience")
    seed: int = 42
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "fraud-gnn"


class ServingConfig(BaseModel):
    """Configuration for the FastAPI inference service."""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    model_path: Path = Path("models/best_model.pt")
    fallback_model_path: Path = Path("models/xgboost_baseline.json")
    redis_url: str = "redis://localhost:6379"


class StreamingConfig(BaseModel):
    """Configuration for Kafka / Redpanda streaming."""

    bootstrap_servers: str = "localhost:9092"
    input_topic: str = "transactions"
    output_topic: str = "scored_transactions"
    consumer_group: str = "fraud-detector"


class MonitoringConfig(BaseModel):
    """Configuration for Prometheus metrics and drift detection."""

    prometheus_port: int = Field(default=9090, ge=1, le=65535)
    drift_window_size: int = Field(default=1000, gt=0)
    drift_threshold: float = Field(default=0.05, gt=0.0, lt=1.0)


# ── Root config ───────────────────────────────────────────────────────


class AppConfig(BaseModel):
    """Root application configuration aggregating all sections."""

    project_name: str = "fraud-gnn"
    data: DataConfig = DataConfig()
    graph: GraphConfig = GraphConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    serving: ServingConfig = ServingConfig()
    streaming: StreamingConfig = StreamingConfig()
    monitoring: MonitoringConfig = MonitoringConfig()


# ── Loader ────────────────────────────────────────────────────────────


def load_config(path: str | Path) -> AppConfig:
    """Load and validate application configuration from a YAML file.

    Args:
        path: Filesystem path to a YAML configuration file.

    Returns:
        A fully validated ``AppConfig`` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        pydantic.ValidationError: If the YAML content fails validation.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    return AppConfig(**raw)


def load_data_config(path: str | Path) -> DataConfig:
    """Load and validate DataConfig from a YAML file.

    Accepts YAML containing either a standalone DataConfig or a nested 'data:' section.

    Args:
        path: Filesystem path to a YAML configuration file.

    Returns:
        A validated ``DataConfig`` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        pydantic.ValidationError: If the YAML content fails validation.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    if "data" in raw and isinstance(raw["data"], dict):
        return DataConfig(**raw["data"])
    return DataConfig(**raw)


def load_graph_config(path: str | Path) -> GraphConfig:
    """Load and validate GraphConfig from a YAML file.

    Accepts YAML containing either a standalone GraphConfig or a nested 'graph:' section.

    Args:
        path: Filesystem path to a YAML configuration file.

    Returns:
        A validated ``GraphConfig`` instance.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    if "graph" in raw and isinstance(raw["graph"], dict):
        return GraphConfig(**raw["graph"])
    return GraphConfig(**raw)


def load_baseline_config(path: str | Path) -> BaselineConfig:
    """Load and validate BaselineConfig from a YAML file.

    Accepts YAML containing either a standalone BaselineConfig or a nested 'baseline:' section.

    Args:
        path: Filesystem path to a YAML configuration file.

    Returns:
        A validated ``BaselineConfig`` instance.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    if "baseline" in raw and isinstance(raw["baseline"], dict):
        return BaselineConfig(**raw["baseline"])
    return BaselineConfig(**raw)
