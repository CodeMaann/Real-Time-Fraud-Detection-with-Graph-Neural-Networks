"""CLI entrypoint for generating synthetic fraud detection datasets.

Usage:
    python -m fraudgnn.data.generate --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from fraudgnn.config import load_data_config
from fraudgnn.data.simulator import TransactionSimulator


def print_summary_statistics(df: pd.DataFrame, output_path: Path, elapsed_sec: float) -> None:
    """Print comprehensive summary statistics of generated dataset.

    Args:
        df: Generated transaction DataFrame.
        output_path: Path where Parquet file was saved.
        elapsed_sec: Time taken in seconds.
    """
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    total_tx = len(df)
    fraud_tx = int(df["is_fraud"].sum())
    legit_tx = total_tx - fraud_tx
    fraud_rate = (fraud_tx / total_tx) * 100

    min_date = df["timestamp"].min()
    max_date = df["timestamp"].max()

    print("\n" + "=" * 70)
    print("  FRAUD-GNN SYNTHETIC TRANSACTION STREAM GENERATOR")
    print("=" * 70)
    print(f"Elapsed Time       : {elapsed_sec:.2f} seconds")
    print(f"Output File        : {output_path}")
    print(f"File Size          : {file_size_mb:.2f} MB")
    print(f"Total Transactions : {total_tx:,}")
    print(f"Simulated Range    : {min_date} --> {max_date}")
    print(f"Time Span (Days)   : {(max_date - min_date).total_seconds() / 86400:.1f} days")

    print("\n[ Class Balance ]")
    print(f"  Legitimate (0)   : {legit_tx:,} ({100 - fraud_rate:.2f}%)")
    print(f"  Fraud (1)        : {fraud_tx:,} ({fraud_rate:.2f}%)")

    print("\n[ Injected Fraud Typologies ]")
    type_counts = df["fraud_type"].value_counts()
    for ftype, count in type_counts.items():
        pct = (count / total_tx) * 100
        print(f"  - {ftype:<20}: {count:,} ({pct:.3f}%)")

    print("\n[ Unique Graph Entities ]")
    print(f"  Users            : {df['user_id'].nunique():,}")
    print(f"  Cards            : {df['card_id'].nunique():,}")
    print(f"  Merchants        : {df['merchant_id'].nunique():,}")
    print(f"  Devices          : {df['device_id'].nunique():,}")
    print(f"  IP Addresses     : {df['ip_address'].nunique():,}")

    print("\n[ Transaction Amounts ($ USD) ]")
    legit_amounts = df[df["is_fraud"] == 0]["amount"]
    fraud_amounts = df[df["is_fraud"] == 1]["amount"]

    print(
        f"  Legit : min=${legit_amounts.min():.2f}, "
        f"mean=${legit_amounts.mean():.2f}, "
        f"median=${legit_amounts.median():.2f}, "
        f"p95=${legit_amounts.quantile(0.95):.2f}, "
        f"max=${legit_amounts.max():.2f}"
    )
    print(
        f"  Fraud : min=${fraud_amounts.min():.2f}, "
        f"mean=${fraud_amounts.mean():.2f}, "
        f"median=${fraud_amounts.median():.2f}, "
        f"p95=${fraud_amounts.quantile(0.95):.2f}, "
        f"max=${fraud_amounts.max():.2f}"
    )

    # Graph noise check: legitimate users sharing devices
    legit_df = df[df["is_fraud"] == 0]
    dev_users = legit_df.groupby("device_id")["user_id"].nunique()
    shared_devs = (dev_users > 1).sum()
    print("\n[ Graph Structure & Noise ]")
    print(f"  Legitimate Devices Shared by Multiple Users: {shared_devs:,}")
    print("=" * 70 + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run transaction generation CLI.

    Args:
        argv: Command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic transaction stream with graph structure and fraud patterns."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data.yaml",
        help="Path to YAML configuration file (default: configs/data.yaml)",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        # Fallback to configs/default.yaml if specified doesn't exist
        default_path = Path("configs/default.yaml")
        if default_path.exists():
            config_path = default_path
        else:
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            return 1

    print(f"Loading configuration from: {config_path}")
    config = load_data_config(config_path)

    print(
        f"Simulating {config.num_transactions:,} transactions over {config.num_days} days "
        f"(target fraud ratio: {config.fraud_ratio * 100:.2f}%)..."
    )

    t0 = time.perf_counter()
    simulator = TransactionSimulator(config)
    df = simulator.generate()
    output_path = simulator.save_to_parquet(df)
    elapsed = time.perf_counter() - t0

    print_summary_statistics(df, output_path, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
