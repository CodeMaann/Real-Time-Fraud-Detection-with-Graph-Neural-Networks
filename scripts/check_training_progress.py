"""Check real-time GNN training progress and epoch metrics from MLflow database."""

import sqlite3

import pandas as pd

conn = sqlite3.connect("mlflow.db")
query = """
SELECT step, value as val_pr_auc, timestamp
FROM metrics
WHERE key = 'val_pr_auc'
ORDER BY timestamp DESC
"""
df = pd.read_sql(query, conn)
# Get the most recent run
if len(df) > 0:
    run_query = (
        "SELECT run_uuid, status, start_time, end_time FROM runs ORDER BY start_time DESC LIMIT 1"
    )
    run_df = pd.read_sql(run_query, conn)
    run_id = run_df.iloc[0]["run_uuid"]
    status = run_df.iloc[0]["status"]
    start_time = run_df.iloc[0]["start_time"]

    metrics_query = f"""
    SELECT step, key, value, timestamp
    FROM metrics
    WHERE run_uuid = '{run_id}'
    ORDER BY step ASC, key ASC
    """
    m_df = pd.read_sql(metrics_query, conn)
    pivot = m_df.pivot(index="step", columns="key", values="value")
    print(f"Latest Run: {run_id} | Status: {status}")
    print("\nEpoch Metrics History:")
    print(pivot)

    # Calculate duration per epoch
    step_times = m_df.groupby("step")["timestamp"].max().sort_index()
    if len(step_times) > 1:
        diffs = step_times.diff().dropna() / 1000.0
        print(f"\nAverage seconds per epoch: {diffs.mean():.1f}s")
        for st, sec in diffs.items():
            print(f"  Epoch {st}: {sec:.1f}s")
    elif len(step_times) == 1:
        elapsed_since_start = (step_times.iloc[0] - start_time) / 1000.0
        print(f"\nEpoch 1 elapsed: {elapsed_since_start:.1f}s")
else:
    print("No metrics found.")
