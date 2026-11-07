"""Generate the benchmark report (docs/benchmarks.md) and README results table from MLflow.

Rules:
1. Every number is read from MLflow (via the API), a saved prediction file, or a saved
   training console log. Nothing is typed by hand and there are no fallback numbers.
2. Two PR-AUC columns per model: the value logged at training time and the re-score of the
   same checkpoint (``rescored_*`` keys written by ``scripts/evaluate_unified.py``).
3. Runs whose training-time metrics were overwritten by the removed raw-SQL sync are listed
   in ``configs/evaluate.yaml``; for them the logged value comes from the run's training log
   when one exists, otherwise it is reported as not recoverable.
4. Per-type recall is shown as k/n over test rows with ``is_fraud == 1``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score

from fraudgnn.config import BaselineConfig, load_baseline_config

FRAUD_TYPES = ["fraud_ring", "account_takeover", "merchant_collusion", "card_testing", "mule"]
README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"
NOT_RUN = "not run"


@dataclass
class ModelRow:
    """All reported values for one model (None means not available)."""

    key: str
    name: str
    run_id: str
    logged_pr_auc: float | None = None
    logged_ci: tuple[float, float] | None = None
    logged_source: str = "MLflow"
    training_batches: str = "n/a"
    overwritten_value: float | None = None
    rescored: dict[str, float] = field(default_factory=dict)
    logged: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)

    def metric(self, key: str) -> tuple[float | None, str]:
        """Return a secondary metric from the re-score if present, else the logged value."""
        if f"rescored_{key}" in self.rescored:
            return self.rescored[f"rescored_{key}"], "rescored"
        if key in self.logged:
            return self.logged[key], "logged"
        return None, ""

    @property
    def best_pr_auc(self) -> float | None:
        """Re-scored PR-AUC when available, else the training-time value."""
        return self.rescored.get("rescored_test_pr_auc", self.logged_pr_auc)

    @property
    def best_ci(self) -> tuple[float, float] | None:
        """CI belonging to ``best_pr_auc``."""
        if "rescored_test_pr_auc" in self.rescored:
            return (
                self.rescored["rescored_test_pr_auc_ci_lower"],
                self.rescored["rescored_test_pr_auc_ci_upper"],
            )
        return self.logged_ci


def get_test_fraud_counts(cfg: BaselineConfig) -> dict[str, int]:
    """Count test rows with ``is_fraud == 1`` per fraud type (temporal split from config)."""
    df = pd.read_parquet(cfg.data_path, columns=["timestamp", "is_fraud", "fraud_type"])
    ts = pd.to_datetime(df["timestamp"])
    val_cut = ts.min() + (ts.max() - ts.min()) * (cfg.split.train_ratio + cfg.split.val_ratio)
    fraud = df[(ts > val_cut) & (df["is_fraud"] == 1)]
    return {str(k): int(v) for k, v in fraud["fraud_type"].value_counts().items()}


def parse_training_log(path: Path) -> tuple[float, tuple[float, float] | None] | None:
    """Read the training-time test PR-AUC (and CI if printed) from a train_gnn console log."""
    if not path.exists():
        return None
    raw = path.read_bytes()
    text = raw.decode("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8")
    match = re.search(
        r"TEST PR-AUC \(Primary\)\s*:\s*([0-9.]+)(?:\s*\[95% CI:\s*([0-9.]+)\s*-\s*([0-9.]+)\])?",
        text,
    )
    if match is None:
        return None
    ci = (float(match.group(2)), float(match.group(3))) if match.group(2) else None
    return float(match.group(1)), ci


def load_row(client: MlflowClient, entry: dict[str, Any], overwritten: set[str]) -> ModelRow:
    """Collect logged and re-scored values for one configured model."""
    row = ModelRow(
        key=entry["key"],
        name=entry["name"],
        run_id=entry["run_id"],
        training_batches=entry.get("training_batches", "n/a (not a GNN)"),
    )
    run = client.get_run(entry["run_id"])
    metrics = run.data.metrics
    row.tags = dict(run.data.tags)
    row.rescored = {k: v for k, v in metrics.items() if k.startswith("rescored_")}
    row.logged = {k: v for k, v in metrics.items() if not k.startswith("rescored_")}

    if entry["run_id"] in overwritten:
        row.overwritten_value = metrics.get("test_pr_auc")
        row.logged = {}  # every test_* value of this run is a re-score, not training-time
        parsed = (
            parse_training_log(Path(entry["training_log"])) if "training_log" in entry else None
        )
        if parsed is not None:
            row.logged_pr_auc, row.logged_ci = parsed
            row.logged_source = f"training log `{entry['training_log']}`"
        else:
            row.logged_source = "overwritten"
    elif "test_pr_auc" not in metrics and "stopped_note" in row.tags:
        row.logged_source = "stopped"
    elif "test_pr_auc" in metrics:
        row.logged_pr_auc = metrics["test_pr_auc"]
        if "test_pr_auc_ci_lower" in metrics and "test_pr_auc_ci_upper" in metrics:
            row.logged_ci = (metrics["test_pr_auc_ci_lower"], metrics["test_pr_auc_ci_upper"])
    return row


def fmt_pr(value: float | None, ci: tuple[float, float] | None) -> str:
    """Format PR-AUC with optional CI."""
    if value is None:
        return NOT_RUN
    return f"{value:.4f} [{ci[0]:.4f}-{ci[1]:.4f}]" if ci else f"{value:.4f}"


def fmt_logged(row: ModelRow) -> str:
    """Format the training-time PR-AUC cell."""
    if row.logged_pr_auc is None:
        if row.logged_source == "stopped":
            return "n/a (run stopped before its test step)"
        return "overwritten, not recoverable" if row.logged_source == "overwritten" else NOT_RUN
    cell = fmt_pr(row.logged_pr_auc, row.logged_ci)
    return f"{cell} (log)" if row.logged_source.startswith("training log") else cell


def fmt_recall(row: ModelRow, fraud_type: str, n: int) -> str:
    """Format per-type recall as k/n, asserting that k is an integer count."""
    value, _ = row.metric(f"test_recall_{fraud_type}")
    if value is None:
        return NOT_RUN
    k = round(value * n)
    if abs(value - k / n) >= 1e-4:
        raise AssertionError(f"{row.name} {fraud_type}: recall {value} is not k/{n}")
    return f"{k}/{n}"


def fmt_metric(row: ModelRow, key: str, fmt: str) -> str:
    """Format a secondary metric (re-scored preferred)."""
    value, _ = row.metric(key)
    return NOT_RUN if value is None else fmt.format(value)


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render a markdown table."""
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def ci_overlap(a: tuple[float, float] | None, b: tuple[float, float] | None) -> bool | None:
    """Whether two intervals overlap (None if either is missing)."""
    if a is None or b is None:
        return None
    return max(a[0], b[0]) <= min(a[1], b[1])


def paired_bootstrap(
    predictions_path: Path,
    pairs: list[list[str]],
    n_iter: int,
    seed: int,
    alpha: float,
) -> list[dict[str, Any]]:
    """Paired bootstrap of PR-AUC(a) - PR-AUC(b) for several pairs on the SAME test resamples."""
    if not predictions_path.exists():
        return []
    with np.load(predictions_path) as data:
        y = data["y_true"]
        preds = {k: data[k] for k in data.files}
    usable = [pair for pair in pairs if pair[0] in preds and pair[1] in preds]
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(y), size=len(y)) for _ in range(n_iter)]
    draws = [idx for idx in draws if y[idx].sum() > 0]
    results = []
    for a_key, b_key in usable:
        pa, pb = preds[a_key], preds[b_key]
        diffs = np.asarray(
            [
                average_precision_score(y[idx], pa[idx]) - average_precision_score(y[idx], pb[idx])
                for idx in draws
            ]
        )
        results.append(
            {
                "a": a_key,
                "b": b_key,
                "point": float(average_precision_score(y, pa) - average_precision_score(y, pb)),
                "lower": float(np.percentile(diffs, 100 * alpha / 2)),
                "upper": float(np.percentile(diffs, 100 * (1 - alpha / 2))),
                "resamples": len(draws),
            }
        )
    return results


def battery_section(path: Path) -> list[str]:
    """Verification section from ``scripts/leakage_battery.py`` results (if present)."""
    if not path.exists():
        return ["## Leakage battery", "", "Not run.", ""]
    r = json.loads(path.read_text(encoding="utf-8"))
    a, b, c, d, e = (
        r[k] for k in ("a_brute_force", "b_injection", "c_ties", "d_alignment", "e_ablation")
    )
    sizes = [k for k in b if k.startswith("bs")]
    status = "PASS" if r["gate_pass"] else "FAIL"
    return [
        "## Leakage battery (real validation data)",
        "",
        f"Generated from `{path.as_posix()}` (`scripts/leakage_battery.py`). Gate: **{status}**.",
        "",
        *markdown_table(
            ["Check", "What it proves", "Result"],
            [
                [
                    "a) Brute-force reference",
                    "Loader subgraphs equal an independent rebuild after physically deleting every "
                    "edge at/after each target's time",
                    f"{a['targets']} targets, {a['structure_mismatches']} structure mismatches, "
                    f"max score diff {a['max_score_abs_diff']:.2e}",
                ],
                [
                    "b) Future injection",
                    "Transactions added after a target cannot change its score, at any batch size",
                    f"earlier targets bit-identical at batch sizes "
                    f"{', '.join(k[2:] for k in sizes if b[k]['early_bit_identical'])}; "
                    f"control: {b[sizes[0]]['late_changed']}/{b[sizes[0]]['late_targets']} later "
                    "targets changed",
                ],
                [
                    "c) Ties and own edges",
                    "No sampled edge at/after its target's time; the target's own edges are absent",
                    f"{c['sampled_edges_checked']:,} sampled edges, "
                    f"{c['edges_at_or_after_target_time']} violations, "
                    f"{c['own_edges_present']} own edges; {c['targets_with_exact_time_tie']} "
                    "targets with exact-second ties (excluded by strict <)",
                ],
                [
                    "d) Alignment",
                    "Each target's user, merchant, card, device, IP, time and label match its raw row",
                    f"val {sum(v for k, v in d['val'].items() if k != 'rows')} and test "
                    f"{sum(v for k, v in d['test'].items() if k != 'rows')} mismatches",
                ],
                [
                    "e) Message-passing ablation",
                    "How much of the validation score depends on neighbor messages",
                    f"full {e['full_model']:.4f}; no message passing {e['no_message_passing']:.4f} "
                    f"({e['no_message_passing'] / e['full_model']:.1%} of full)",
                ],
            ],
        ),
        "",
    ]


def batch_leak_section(path: Path) -> list[str]:
    """Markdown section from ``scripts/measure_batch_leak.py`` results (if present)."""
    if not path.exists():
        return ["## Batch-sharing leak", "", "Not run.", ""]
    res = json.loads(path.read_text(encoding="utf-8"))
    sub_test = res["subset"]["test"]
    sub_val = res["subset"]["val"]
    fpr = res["config"]["target_fpr"]
    md = [
        "## Batch-sharing leak",
        "",
        f"Generated from `{path.as_posix()}` (`scripts/measure_batch_leak.py`). Same rows at every batch "
        f"size: test = all {sub_test['positives']} positives + {sub_test['negatives']} random "
        f"negatives; validation = all {sub_val['positives']} positives + {sub_val['negatives']} "
        "random negatives (seed "
        f"{res['config']['seed']}). Batch size 1 = isolated per-target subgraphs; larger sizes = "
        "legacy shared batches with the same boundaries as a full ordered pass.",
        "",
        "Downsampling negatives raises PR-AUC (precision is computed against fewer negatives), "
        "so 'subset PR-AUC' is only comparable across batch sizes. 'Weighted PR-AUC' gives each "
        f"sampled negative weight {sub_test['negative_weight']:.2f} (all test negatives / sampled), "
        "an estimate of full-test PR-AUC. Recall uses a threshold set at "
        f"{fpr:.0%} FPR on the validation-subset negatives of the same batch size.",
        "",
    ]
    for model in res["models"].values():
        rows = []
        for bs, m in model["batch_sizes"].items():
            rows.append(
                [
                    bs,
                    fmt_pr(m["pr_auc_subset"], tuple(m["pr_auc_subset_ci"])),
                    fmt_pr(m["pr_auc_weighted"], tuple(m["pr_auc_weighted_ci"])),
                    f"{m['diff_vs_ref']:+.4f} [{m['diff_vs_ref_ci'][0]:+.4f}, "
                    f"{m['diff_vs_ref_ci'][1]:+.4f}]",
                    f"{m['test_recall_at_val_thr']:.4f}",
                    f"{m['test_fpr_at_val_thr']:.4f}",
                ]
            )
        md += [
            f"### {model['name']}",
            "",
            *markdown_table(
                [
                    "Batch size",
                    "Subset PR-AUC [95% CI]",
                    "Weighted PR-AUC [95% CI]",
                    "Weighted PR-AUC minus batch 1 (paired) [95% CI]",
                    f"Test recall @ val {fpr:.0%} FPR",
                    "Test FPR at that threshold",
                ],
                rows,
            ),
            "",
        ]
    return md


def build_report(cfg: dict[str, Any]) -> tuple[str, str]:
    """Build (benchmarks markdown, README table block)."""
    base_cfg = load_baseline_config(cfg["baseline_config"])
    client = MlflowClient(tracking_uri=cfg["tracking_uri"])
    overwritten = set(cfg.get("sql_sync_overwritten_runs", []))
    counts = get_test_fraud_counts(base_cfg)
    missing = [ft for ft in FRAUD_TYPES if ft not in counts]
    if missing:
        raise KeyError(f"No test positives found for fraud types: {missing}")
    rows = [load_row(client, entry, overwritten) for entry in cfg["models"]]
    rep = cfg["report"]

    headers = [
        "Model",
        "Run",
        "Training batches",
        "PR-AUC logged at training",
        "PR-AUC re-scored, shared batches (legacy, leaky)",
        "PR-AUC re-scored [95% CI]",
        "ROC-AUC",
        "Recall@1%FPR",
        "F1@val thr",
        "Business cost",
        "Scoring ms/sample",
        *[f"{ft} k/n" for ft in FRAUD_TYPES],
    ]
    table_rows = []
    for r in rows:
        rescored = r.rescored.get("rescored_test_pr_auc")
        rescored_cell = (
            fmt_pr(
                rescored,
                (
                    r.rescored["rescored_test_pr_auc_ci_lower"],
                    r.rescored["rescored_test_pr_auc_ci_upper"],
                ),
            )
            if rescored is not None
            else NOT_RUN
        )
        if r.tags.get("rescore_checkpoint_matches_run_artifact") == "no_artifact":
            rescored_cell += " (model file not a run artifact)"
        shared = r.rescored.get("rescored_shared_test_pr_auc")
        shared_cell = (
            fmt_pr(
                shared,
                (
                    r.rescored["rescored_shared_test_pr_auc_ci_lower"],
                    r.rescored["rescored_shared_test_pr_auc_ci_upper"],
                ),
            )
            if shared is not None
            else "n/a"
        )
        table_rows.append(
            [
                r.name,
                f"`{r.run_id[:8]}`",
                r.training_batches,
                fmt_logged(r),
                shared_cell,
                rescored_cell,
                fmt_metric(r, "test_roc_auc", "{:.4f}"),
                fmt_metric(r, "test_recall_at_1pct_fpr", "{:.4f}"),
                fmt_metric(r, "test_f1", "{:.4f}"),
                fmt_metric(r, "test_business_cost", "${:,.0f}"),
                fmt_metric(r, "latency_ms_per_sample", "{:.4f}"),
                *[fmt_recall(r, ft, counts[ft]) for ft in FRAUD_TYPES],
            ]
        )

    notes = [
        "- Secondary metrics come from the re-score when one exists, otherwise from the value "
        "logged at training time. Thresholds are always selected on validation.",
        "- Scoring time is wall-clock per sample on CPU over the whole test pass. For GNNs it "
        "includes causal neighbor sampling and node-feature building, so isolated (leak-free) "
        "scoring is slower than shared batches. Not end-to-end serving latency.",
        "- GNN neighbor selection is deterministic (most recent k edges before the cutoff), so "
        "a sampling seed has no effect and there is no seed-to-seed range to report.",
        "- GNN 're-scored' values use isolated per-target subgraphs (`isolate_targets=True`) "
        "and exact per-node float64 amount sums: each transaction only sees edges before its "
        "own timestamp. The 'shared batches' column scores the same checkpoint with the legacy "
        "loader, where all targets in a batch share one subgraph and a transaction can see "
        "edges sampled for later transactions in the same batch.",
        "- 'Training batches' says how the checkpoint was trained: 'shared' = leaky "
        "shared-subgraph batches, 'isolated' = per-target subgraphs.",
        "- The isolated-batch HeteroGAT is the epoch-18 checkpoint of a run stopped by hand at its "
        "best epoch (max 30, patience 5, 12 epochs unused); its score is a lower bound for "
        "that training setup.",
    ]
    for r in rows:
        if r.overwritten_value is not None:
            notes.append(
                f"- `{r.run_id[:8]}` ({r.name}): training-time metrics in MLflow were "
                f"overwritten on 2026-10-19 by a raw-SQL sync (since removed); MLflow now holds "
                f"test_pr_auc {r.overwritten_value:.4f}. Logged column source: {r.logged_source}."
            )

    findings: list[str] = []
    ranked = [r for r in rows if r.best_pr_auc is not None]
    lf = next((r for r in rows if r.key == "xgb_lf"), None)
    if ranked:
        top = max(ranked, key=lambda r: float(r.best_pr_auc or 0.0))
        findings.append(f"- Highest test PR-AUC: {top.name} ({top.best_pr_auc:.4f}).")
    if lf is not None and lf.best_pr_auc is not None:
        for r in rows:
            if r.key.startswith(("gat", "sage")) and r.best_pr_auc is not None:
                overlap = ci_overlap(lf.best_ci, r.best_ci)
                verdict = {True: "CIs overlap", False: "CIs do not overlap", None: "no CI"}[
                    overlap
                ]
                findings.append(
                    f"- {r.name} {fmt_pr(r.best_pr_auc, r.best_ci)} vs label-free XGBoost "
                    f"{fmt_pr(lf.best_pr_auc, lf.best_ci)}: {verdict}."
                )
    names = {r.key: r.name for r in rows}
    for boot in paired_bootstrap(
        Path(cfg["predictions_out"]),
        rep["paired_bootstrap_pairs"],
        rep["bootstrap_iterations"],
        rep["bootstrap_seed"],
        rep["bootstrap_alpha"],
    ):
        findings.append(
            f"- Paired bootstrap ({boot['resamples']} shared test resamples), "
            f"{names.get(boot['a'], boot['a'])} minus {names.get(boot['b'], boot['b'])}: "
            f"{boot['point']:+.4f} [{boot['lower']:+.4f}, {boot['upper']:+.4f}]."
        )

    md = [
        "# Benchmark Report",
        "",
        "Generated by `scripts/make_results_table.py` from MLflow and "
        "`data/processed/eval_predictions.npz`. Do not edit by hand.",
        "",
        "Temporal split; thresholds selected on validation only; per-type recall is k/n over "
        "test rows with `is_fraud == 1`.",
        "",
        *markdown_table(headers, table_rows),
        "",
        "## Notes",
        "",
        *notes,
        "",
        "## Findings computed from the table",
        "",
        *(findings or ["Not run."]),
        "",
        *battery_section(Path(rep["battery_results"])),
        *batch_leak_section(Path(rep["batch_leak_results"])),
        "## Ablations",
        "",
        "Not run under the full-data protocol.",
        "",
    ]

    readme_rows = [
        [r.name, r.training_batches, fmt_logged(r), row[4], row[5], row[7]]
        for r, row in zip(rows, table_rows, strict=True)
    ]
    readme = [
        README_START,
        *markdown_table(
            [
                "Model",
                "Training batches",
                "Test PR-AUC (logged)",
                "Test PR-AUC, shared batches (legacy, leaky)",
                "Test PR-AUC (re-scored) [95% CI]",
                "Recall @ 1% FPR",
            ],
            readme_rows,
        ),
        "",
        "Generated by `scripts/make_results_table.py`; full table and notes in "
        "[docs/benchmarks.md](docs/benchmarks.md).",
        README_END,
    ]
    return "\n".join(md), "\n".join(readme)


def write_readme_block(readme_path: Path, block: str) -> None:
    """Replace the text between the README result markers."""
    text = readme_path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise ValueError(f"{readme_path} has no {README_START} ... {README_END} block")
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    readme_path.write_text(head + block + tail, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate benchmark report from MLflow.")
    parser.add_argument("--config", default="configs/evaluate.yaml")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    report, readme_block = build_report(cfg)
    out = Path(cfg["report"]["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    write_readme_block(Path(cfg["report"]["readme"]), readme_block)
    print(report)
    print(f"\nWrote {out} and the results block in {cfg['report']['readme']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
