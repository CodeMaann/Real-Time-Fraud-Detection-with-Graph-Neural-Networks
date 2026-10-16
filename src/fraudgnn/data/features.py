"""Feature engineering pipeline for tabular fraud detection baselines.

Implements causal, leakage-free feature computation:
1. Rolling transaction counts and amount sums per card and user (1h, 24h, 7d windows).
2. Prior inter-arrival time deltas for cards and users.
3. Cumulative past device-sharing counts (distinct users per device seen so far).
4. Expanding past merchant risk rate smoothed with Empirical Bayes prior.
5. Circadian and weekly cyclical temporal encodings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_tabular_features(
    df: pd.DataFrame,
    smoothing_alpha: float = 10.0,
    prior_fraud_rate: float = 0.008,
) -> pd.DataFrame:
    """Engineer strictly causal tabular features for fraud detection.

    All rolling windows use ``closed='left'`` to exclude the current transaction.
    Device sharing and merchant risk rates are computed expanding-only over historical data.

    Args:
        df: Chronologically sorted transaction DataFrame.
        smoothing_alpha: Empirical Bayes smoothing parameter for merchant risk rate.
        prior_fraud_rate: Prior expected fraud rate for Bayesian smoothing.

    Returns:
        DataFrame enriched with engineered feature columns.
    """
    # Create working copy sorted strictly by timestamp
    feat_df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
    n_rows = len(feat_df)
    feat_df["_orig_idx"] = np.arange(n_rows)

    timestamps = pd.to_datetime(feat_df["timestamp"])
    ts_seconds = (timestamps.astype("int64") // 10**9).to_numpy()
    feat_df["_ts_sec"] = ts_seconds

    # ── 1. Base & Cyclical Temporal Features ──────────────────────────────
    feat_df["log_amount"] = np.log1p(feat_df["amount"].to_numpy(dtype=np.float64))

    hours = timestamps.dt.hour.to_numpy(dtype=np.float64)
    days_of_week = timestamps.dt.dayofweek.to_numpy(dtype=np.float64)

    feat_df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    feat_df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    feat_df["day_sin"] = np.sin(2 * np.pi * days_of_week / 7.0)
    feat_df["day_cos"] = np.cos(2 * np.pi * days_of_week / 7.0)

    # ── 2. Prior Inter-Arrival Time Deltas (seconds) ─────────────────────
    # Seconds since previous card transaction (capped at 30 days)
    delta_card = (
        feat_df.groupby("card_id")["_ts_sec"]
        .diff()
        .fillna(86400 * 30.0)
        .clip(lower=0.0, upper=86400 * 30.0)
        .to_numpy(dtype=np.float64)
    )
    feat_df["time_delta_card"] = delta_card
    feat_df["log_time_delta_card"] = np.log1p(delta_card)

    # Seconds since previous user transaction
    delta_user = (
        feat_df.groupby("user_id")["_ts_sec"]
        .diff()
        .fillna(86400 * 30.0)
        .clip(lower=0.0, upper=86400 * 30.0)
        .to_numpy(dtype=np.float64)
    )
    feat_df["time_delta_user"] = delta_user
    feat_df["log_time_delta_user"] = np.log1p(delta_user)

    # ── 3. Rolling Windows per User (1h, 24h, 7d) ─────────────────────────
    # Sort by user_id and timestamp to apply contiguous rolling
    df_u = feat_df.sort_values(["user_id", "timestamp"])
    u_grouped = df_u.set_index("timestamp").groupby("user_id")["amount"]

    for window in ["1h", "24h", "7d"]:
        # closed='left' strictly excludes current transaction timestamp
        c_vals = u_grouped.rolling(window, closed="left").count().fillna(0.0).to_numpy()
        s_vals = u_grouped.rolling(window, closed="left").sum().fillna(0.0).to_numpy()
        df_u[f"user_tx_count_{window}"] = c_vals
        df_u[f"user_amt_sum_{window}"] = s_vals

    # Restore temporal ordering
    feat_df = df_u.sort_values("_orig_idx").reset_index(drop=True)

    # ── 4. Rolling Windows per Card (1h, 24h, 7d) ─────────────────────────
    df_c = feat_df.sort_values(["card_id", "timestamp"])
    c_grouped = df_c.set_index("timestamp").groupby("card_id")["amount"]

    for window in ["1h", "24h", "7d"]:
        c_vals = c_grouped.rolling(window, closed="left").count().fillna(0.0).to_numpy()
        s_vals = c_grouped.rolling(window, closed="left").sum().fillna(0.0).to_numpy()
        df_c[f"card_tx_count_{window}"] = c_vals
        df_c[f"card_amt_sum_{window}"] = s_vals

    feat_df = df_c.sort_values("_orig_idx").reset_index(drop=True)

    # ── 5. Causal Device-Sharing Count ───────────────────────────────────
    # Count how many distinct users have operated this device up to current transaction
    # First time a user is seen on this device marks a new distinct user (+1)
    first_seen_mask = ~feat_df.duplicated(subset=["device_id", "user_id"])
    feat_df["device_user_count_past"] = (
        first_seen_mask.astype(int)
        .groupby(feat_df["device_id"])
        .cumsum()
        .to_numpy(dtype=np.float64)
    )

    # ── 6. Causal Merchant Risk Rate (Empirical Bayes Smoothed) ───────────
    # Computed using ONLY prior transactions of that merchant
    merch_past_tx = feat_df.groupby("merchant_id").cumcount().to_numpy(dtype=np.float64)
    merch_past_fraud = (
        feat_df.groupby("merchant_id")["is_fraud"].cumsum() - feat_df["is_fraud"]
    ).to_numpy(dtype=np.float64)
    feat_df["merchant_risk_rate"] = (merch_past_fraud + smoothing_alpha * prior_fraud_rate) / (
        merch_past_tx + smoothing_alpha
    )

    # ── 7. Categorical Encodings ──────────────────────────────────────────
    # One-hot encode channel
    for ch in ["mobile", "web", "pos"]:
        feat_df[f"channel_{ch}"] = (feat_df["channel"] == ch).astype(np.float32)

    # Clean up temporary index columns
    feat_df.drop(columns=["_orig_idx", "_ts_sec"], inplace=True)

    return feat_df


def get_feature_column_names() -> list[str]:
    """Return list of model feature column names (excluding identifiers and targets).

    Returns:
        List of feature column names.
    """
    return [
        "amount",
        "log_amount",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "time_delta_card",
        "log_time_delta_card",
        "time_delta_user",
        "log_time_delta_user",
        "user_tx_count_1h",
        "user_amt_sum_1h",
        "user_tx_count_24h",
        "user_amt_sum_24h",
        "user_tx_count_7d",
        "user_amt_sum_7d",
        "card_tx_count_1h",
        "card_amt_sum_1h",
        "card_tx_count_24h",
        "card_amt_sum_24h",
        "card_tx_count_7d",
        "card_amt_sum_7d",
        "device_user_count_past",
        "merchant_risk_rate",
        "channel_mobile",
        "channel_web",
        "channel_pos",
    ]
