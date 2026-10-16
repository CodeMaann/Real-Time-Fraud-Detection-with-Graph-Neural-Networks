"""Tests for the synthetic transaction simulator, data integrity, and leakage prevention."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fraudgnn.config import DataConfig, load_data_config
from fraudgnn.data.simulator import TransactionSimulator

# Reference schema of permitted columns
EXPECTED_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "timestamp",
    "user_id",
    "card_id",
    "merchant_id",
    "device_id",
    "ip_address",
    "amount",
    "currency",
    "country",
    "merchant_category",
    "channel",
    "is_fraud",
    "fraud_type",
)

EXPECTED_FRAUD_TYPES: set[str] = {
    "legitimate",
    "fraud_ring",
    "mule",
    "card_testing",
    "account_takeover",
    "merchant_collusion",
}


@pytest.fixture
def small_config() -> DataConfig:
    """Fixture providing a fast-running DataConfig for testing."""
    return DataConfig(
        num_transactions=3_000,
        num_days=10,
        num_users=200,
        num_merchants=50,
        num_devices=300,
        num_ips=400,
        fraud_ratio=0.02,
        seed=42,
    )


def test_simulation_determinism(small_config: DataConfig) -> None:
    """Identical seed must produce bit-for-bit identical DataFrames."""
    sim1 = TransactionSimulator(small_config)
    df1 = sim1.generate()

    sim2 = TransactionSimulator(small_config)
    df2 = sim2.generate()

    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_produce_different_data(small_config: DataConfig) -> None:
    """Different seeds must produce different transaction amounts and users."""
    cfg_other = small_config.model_copy(update={"seed": 999})
    df1 = TransactionSimulator(small_config).generate()
    df2 = TransactionSimulator(cfg_other).generate()

    assert not df1["amount"].equals(df2["amount"])


def test_fraud_rate_within_tolerance(small_config: DataConfig) -> None:
    """Injected fraud rate must closely match the configured target ratio."""
    df = TransactionSimulator(small_config).generate()
    actual_rate = df["is_fraud"].mean()

    # For 3,000 tx at 0.02 ratio (60 fraud tx), allow small rounding tolerance
    assert 0.015 <= actual_rate <= 0.025


def test_no_label_leakage_columns(small_config: DataConfig) -> None:
    """Only expected feature columns, is_fraud, and hidden fraud_type exist.

    No surrogate indicators (e.g. is_compromised, ring_id, attack_vector) are permitted.
    """
    df = TransactionSimulator(small_config).generate()

    assert list(df.columns) == list(EXPECTED_COLUMNS)

    # Disallowed leakage substrings
    leakage_terms = ["leak", "target", "label", "score", "weight", "ring_id", "mule_id"]
    for col in df.columns:
        if col in ("is_fraud", "fraud_type"):
            continue
        for term in leakage_terms:
            assert term not in col.lower(), f"Potential leakage column detected: {col}"


def test_all_fraud_types_present(small_config: DataConfig) -> None:
    """All 5 injected financial fraud typologies must be generated."""
    df = TransactionSimulator(small_config).generate()
    present_types = set(df["fraud_type"].unique())

    assert present_types == EXPECTED_FRAUD_TYPES


def test_temporal_ordering(small_config: DataConfig) -> None:
    """Transactions must be strictly sorted by timestamp (non-decreasing)."""
    df = TransactionSimulator(small_config).generate()

    assert df["timestamp"].is_monotonic_increasing, "Timestamps are not chronologically sorted!"
    # Verify sequential transaction IDs
    expected_ids = [f"tx_{i:08d}" for i in range(len(df))]
    assert df["transaction_id"].tolist() == expected_ids


def test_benign_noise_shared_devices(small_config: DataConfig) -> None:
    """Legitimate transactions must include shared devices (benign graph noise)."""
    df = TransactionSimulator(small_config).generate()
    legit_df = df[df["is_fraud"] == 0]

    # Count legitimate users per device
    users_per_dev = legit_df.groupby("device_id")["user_id"].nunique()
    shared_devices = (users_per_dev > 1).sum()

    assert shared_devices > 0, "Expected at least one benign shared device in the network."


def test_save_and_reload_parquet(tmp_path: Path, small_config: DataConfig) -> None:
    """Parquet output must persist cleanly and reload with identical data."""
    cfg = small_config.model_copy(update={"raw_dir": tmp_path})
    sim = TransactionSimulator(cfg)
    df = sim.generate()

    out_file = sim.save_to_parquet(df)
    assert out_file.exists()
    assert out_file.suffix == ".parquet"

    reloaded = pd.read_parquet(out_file)
    pd.testing.assert_frame_equal(df, reloaded)


def test_load_data_config() -> None:
    """load_data_config must correctly parse configs/data.yaml and configs/default.yaml."""
    cfg_data = load_data_config("configs/data.yaml")
    assert cfg_data.num_transactions == 500_000
    assert cfg_data.num_days == 90
    assert cfg_data.fraud_ratio == pytest.approx(0.008)

    cfg_default = load_data_config("configs/default.yaml")
    assert cfg_default.num_transactions == 500_000
    assert cfg_default.fraud_ratio == pytest.approx(0.008)
