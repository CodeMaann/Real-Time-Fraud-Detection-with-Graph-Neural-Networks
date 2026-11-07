"""Random hyperparameter search for the isolated-batch HeteroGAT (docs/protocol.md §4).

Samples ``n_configs`` distinct points from the full grid (fixed enumeration order, sampling
seed from the config). Each configuration trains on a fixed random subset of training targets
for a fixed number of epochs (no early stopping), is scored on the FULL validation split after
every epoch, and never touches the test split. Each configuration resumes from its own
checkpoint; results are written after every configuration and finished ones are skipped.

An optional ``anchor`` configuration (the known-good settings) is appended after the sampled
ones and run at the same budget, so tuning cannot silently select a worse model.

Modes:
  default             run (or resume) the search
  --dry-run-batches N time N training steps and N validation batches of configuration 0;
                      nothing is written or logged
  --measure-timing N  time N steps / N validation batches for one configuration per
                      (hidden_dim, num_neighbors) combination; writes the timing file
  --estimate          print the per-configuration Colab time estimate from that file
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from fraudgnn.config import GNNConfig, load_gnn_config
from fraudgnn.training.train_gnn import (
    prepare_training,
    resolve_device,
    time_training_batches,
    train_and_evaluate_gnn,
)


def sample_configs(space: dict[str, list[Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """n distinct grid points, sampled with a fixed seed from a fixed enumeration order."""
    keys = list(space)
    grid = [dict(zip(keys, values, strict=True)) for values in itertools.product(*space.values())]
    picks = np.random.default_rng(seed).choice(len(grid), size=n, replace=False)
    return [grid[int(i)] for i in picks]


def search_points(search_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Sampled configurations, then the anchor (if configured) as the last one."""
    points = sample_configs(search_cfg["space"], search_cfg["n_configs"], search_cfg["seed"])
    anchor = search_cfg.get("anchor")
    if anchor is not None:
        points.append(dict(anchor))
    return points


def combo_key(point: dict[str, Any]) -> str:
    """Cost-relevant part of a configuration: hidden size and neighbor fan-out."""
    return f"hidden{point['hidden_dim']}_nn{'x'.join(str(k) for k in point['num_neighbors'])}"


def config_for(
    base: GNNConfig,
    point: dict[str, Any],
    budget: dict[str, Any],
    index: int,
    paths: dict[str, Any],
) -> GNNConfig:
    """Training config of one search point."""
    cfg = base.model_copy(deep=True)
    cfg.data_dir = Path(paths["data_dir"])
    cfg.raw_data_path = Path(paths["raw_data_path"])
    cfg.model.hidden_dim = int(point["hidden_dim"])
    cfg.model.dropout = float(point["dropout"])
    t = cfg.training
    t.learning_rate = float(point["learning_rate"])
    t.weight_decay = float(point["weight_decay"])
    t.num_neighbors = [int(k) for k in point["num_neighbors"]]
    t.loss_type = str(point["loss_type"])
    t.epochs = int(budget["epochs"])
    t.patience = int(budget["epochs"])  # counter can reach epochs-1 at most: no early stop
    t.seed = int(budget["model_seed"])
    t.train_target_fraction = float(budget["train_target_fraction"])
    t.train_subset_seed = int(budget["train_subset_seed"])
    t.model_save_path = Path(f"search_config_{index}.pt")
    t.best_model_dir = Path(paths["best_model_dir"])
    t.checkpoint_path = Path(paths["checkpoint_dir"]) / f"search_config_{index}_state.pt"
    t.mlflow_tracking_uri = str(paths["tracking_uri"])
    return cfg


def write_results(path: Path, results: dict[str, Any]) -> None:
    """Atomic JSON write (temp file, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def mean_epoch_seconds(history: list[dict[str, Any]]) -> float:
    """Mean measured epoch wall-clock (train + validation) of one search run."""
    seconds = [float(h["epoch_seconds"]) for h in history if "epoch_seconds" in h]
    if len(seconds) != len(history) or not seconds:
        raise ValueError("every epoch needs a measured epoch_seconds")
    return sum(seconds) / len(seconds)


def relative_costs(search_cfg: dict[str, Any]) -> dict[str, float]:
    """Hardware-independent cost per (hidden_dim, num_neighbors) combination.

    Full-epoch CPU seconds from the timing file, relative to the reference combination.
    """
    est = search_cfg["estimate"]
    combos = json.loads(Path(est["timing_file"]).read_text(encoding="utf-8"))["combos"]

    def full_epoch(t: dict[str, Any]) -> float:
        return float(
            t["sec_per_train_step"] * t["full_steps_per_epoch"]
            + t["sec_per_val_batch"] * t["val_batches_per_epoch"]
        )

    ref = full_epoch(combos[est["reference_combo"]])
    return {key: full_epoch(t) / ref for key, t in combos.items()}


def epoch_devices(entry: dict[str, Any]) -> list[str]:
    """Device name of every epoch of one configuration ("unknown" if not recorded)."""
    return [str(h.get("device_name", "unknown")) for h in entry.get("history", [])]


def select_winner(
    entries: list[dict[str, Any]],
    margin: float,
    relative_cost: dict[str, float] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Pre-registered rule (protocol section 4) over the finished configurations.

    Among configurations whose best validation PR-AUC is within ``margin`` of the highest,
    pick the lowest measured mean epoch time from the search itself; ties go to the lower
    index. Fallback: if those configurations did not all run every epoch on the same device,
    measured times are not comparable, so they are ranked by ``relative_cost`` (CPU-measured
    relative cost of their hidden_dim / num_neighbors combination) instead.
    Returns (winner, report explaining the choice).
    """
    done = [e for e in entries if e.get("status") == "finished"]
    if not done:
        return None, {"reason": "no finished configuration"}
    best = max(e["best_val_pr_auc"] for e in done)
    within = [e for e in done if e["best_val_pr_auc"] >= best - margin]
    devices = sorted({d for e in within for d in epoch_devices(e)})
    same_device = len(devices) == 1 and devices[0] != "unknown"
    if same_device:
        basis = f"measured mean epoch time on {devices[0]}"
        winner = min(within, key=lambda e: (e["mean_epoch_seconds"], e["index"]))
        cost_text = f"{winner['mean_epoch_seconds']:.1f} s"
    else:
        if relative_cost is None:
            raise ValueError("mixed devices inside the margin and no relative cost given")
        basis = (
            f"CPU-measured relative cost (fallback: devices inside the margin differ: {devices})"
        )
        winner = min(within, key=lambda e: (relative_cost[combo_key(e["params"])], e["index"]))
        cost_text = f"relative cost {relative_cost[combo_key(winner['params'])]:.2f}"
    report = {
        "highest_val_pr_auc": best,
        "margin": margin,
        "ranking_basis": basis,
        "devices_within_margin": devices,
        "within_margin": [
            {
                "index": e["index"],
                "best_val_pr_auc": e["best_val_pr_auc"],
                "mean_epoch_seconds": e["mean_epoch_seconds"],
                "devices": sorted(set(epoch_devices(e))),
                "relative_cost": (
                    None if relative_cost is None else relative_cost.get(combo_key(e["params"]))
                ),
            }
            for e in sorted(within, key=lambda e: e["index"])
        ],
        "winner_index": winner["index"],
        "reason": (
            f"config {winner['index']} is cheapest ({cost_text}) by {basis} among the "
            f"{len(within)} configuration(s) within {margin} of the highest validation PR-AUC "
            f"({best:.4f})"
        ),
    }
    return winner, report


def print_selection(report: dict[str, Any]) -> None:
    """Human-readable account of the selection."""
    if "within_margin" not in report:
        print(report["reason"])
        return
    print(
        f"Highest validation PR-AUC {report['highest_val_pr_auc']:.4f}; configurations within "
        f"{report['margin']}:"
    )
    for e in report["within_margin"]:
        print(
            f"  config {e['index']}: val PR-AUC {e['best_val_pr_auc']:.4f}, "
            f"mean epoch {e['mean_epoch_seconds']:.1f} s on {e['devices']}, "
            f"relative cost {e['relative_cost']}"
        )
    print("Ranking basis:", report["ranking_basis"])
    print("Winner:", report["reason"])


def run_search(
    search_cfg: dict[str, Any],
    base: GNNConfig,
    paths: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Run or resume all configurations; returns the results document."""
    results_path = Path(paths["results_out"])
    points = search_points(search_cfg)
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        if [e["params"] for e in results["configs"]] != points:
            raise ValueError(f"{results_path} was written for a different sample; refusing")
    else:
        results = {
            "protocol": "docs/protocol.md section 4",
            "sampling_seed": search_cfg["seed"],
            "budget": search_cfg["budget"],
            "configs": [
                {"index": i, "params": p, "status": "pending"} for i, p in enumerate(points)
            ],
        }
        write_results(results_path, results)

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    for entry in results["configs"]:
        if entry["status"] == "finished":
            print(f"config {entry['index']}: done ({entry['best_val_pr_auc']:.4f}), skipping")
            continue
        cfg = config_for(base, entry["params"], search_cfg["budget"], entry["index"], paths)
        cfg.training.mlflow_experiment_name = search_cfg["mlflow_experiment"]
        print(f"\n=== config {entry['index']}: {entry['params']} ===", flush=True)
        t0 = time.perf_counter()
        out = train_and_evaluate_gnn(
            cfg, causal=True, device=device, resume=True, evaluate_test=False
        )
        entry.update(
            status=out["status"],
            best_val_pr_auc=out.get("best_val_pr_auc"),
            best_epoch=out.get("best_epoch"),
            history=[
                {
                    k: h[k]
                    for k in ("epoch", "train_loss", "val_pr_auc", "epoch_seconds", "device_name")
                    if k in h
                }
                for h in out["history"]
            ],
            mean_epoch_seconds=mean_epoch_seconds(out["history"]),
            mlflow_run_id=out.get("mlflow_run_id"),
            seconds_this_session=time.perf_counter() - t0,
        )
        costs = relative_costs(search_cfg) if "estimate" in search_cfg else None
        winner, report = select_winner(
            results["configs"], search_cfg["selection"]["margin"], costs
        )
        results["winner_index"] = None if winner is None else winner["index"]
        results["selection"] = report
        write_results(results_path, results)
    return results


def estimate(search_cfg: dict[str, Any]) -> dict[str, Any]:
    """Per-configuration Colab minutes from measured CPU timings (see protocol section 4).

    The reference Colab epoch (``colab_minutes_per_full_epoch``) was measured for the
    ``reference_combo`` at full data. Each configuration's minutes scale that reference by the
    ratio of its measured CPU time to the reference combination's CPU time. Assumption (not
    measured on GPU): T4 cost scales across combinations like CPU cost.
    """
    est = search_cfg["estimate"]
    timing = json.loads(Path(est["timing_file"]).read_text(encoding="utf-8"))["combos"]
    ref = timing[est["reference_combo"]]
    ref_full_epoch = ref["sec_per_train_step"] * ref["full_steps_per_epoch"] + (
        ref["sec_per_val_batch"] * ref["val_batches_per_epoch"]
    )
    per_min = est["colab_minutes_per_full_epoch"] / ref_full_epoch  # Colab min per CPU second
    budget_epochs = search_cfg["budget"]["epochs"]
    rows = []
    for i, point in enumerate(search_points(search_cfg)):
        t = timing[combo_key(point)]
        search_epoch = (
            t["sec_per_train_step"] * t["search_steps_per_epoch"]
            + t["sec_per_val_batch"] * t["val_batches_per_epoch"]
        )
        full_epoch = (
            t["sec_per_train_step"] * t["full_steps_per_epoch"]
            + t["sec_per_val_batch"] * t["val_batches_per_epoch"]
        )
        rows.append(
            {
                "index": i,
                "combo": combo_key(point),
                "search_minutes": search_epoch * per_min * budget_epochs,
                "final_minutes_per_epoch_if_winner": full_epoch * per_min,
            }
        )
    final = search_cfg["final"]
    final_runs = [
        r["final_minutes_per_epoch_if_winner"] * final["max_epochs"] * len(final["seeds"])
        for r in rows
    ]
    return {
        "assumption": "T4 cost scales across (hidden_dim, num_neighbors) like measured CPU cost",
        "configs": rows,
        "search_total_minutes": sum(r["search_minutes"] for r in rows),
        "final_runs_minutes_upper_bound_range": [min(final_runs), max(final_runs)],
    }


def measure_timing(
    search_cfg: dict[str, Any],
    base: GNNConfig,
    paths: dict[str, Any],
    device: torch.device,
    n: int,
) -> dict[str, Any]:
    """Time one configuration per (hidden_dim, num_neighbors) combination, same settings."""
    combos: dict[str, Any] = {}
    for point in search_points(search_cfg):
        key = combo_key(point)
        if key in combos:
            continue
        cfg = config_for(base, point, search_cfg["budget"], 0, paths)
        setup = prepare_training(cfg, device, causal=True)
        timing = time_training_batches(setup, cfg, device, n)
        full_steps = math.ceil(
            len(setup.train_loader)
            / setup.accumulation_steps
            / search_cfg["budget"]["train_target_fraction"]
        )
        combos[key] = {
            "measured_with": point,
            "sec_per_train_step": timing["sec_per_train_step"],
            "sec_per_val_batch": timing["sec_per_val_batch"],
            "search_steps_per_epoch": timing["train_steps_per_epoch"],
            "full_steps_per_epoch": full_steps,
            "val_batches_per_epoch": timing["val_batches_per_epoch"],
            "micro_batch_size": cfg.training.micro_batch_size,
            "device": str(device),
        }
        print(f"{key}: {combos[key]}", flush=True)
    return {"steps_timed": n, "combos": combos}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/hparam_search.yaml")
    parser.add_argument("--paths", default="colab", choices=["colab", "local"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--dry-run-batches", type=int, default=None)
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--measure-timing", type=int, default=None)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Override the forward/backward chunk (same gradient; lower memory)",
    )
    args = parser.parse_args(argv)

    search_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    points = search_points(search_cfg)
    print("Configurations (sampled, then the anchor):")
    for i, p in enumerate(points):
        print(f"  {i}: {p}")
    if args.estimate:
        print(json.dumps(estimate(search_cfg), indent=2))
        return 0

    base = load_gnn_config(search_cfg["base_config"])
    if args.micro_batch_size is not None:
        base.training.micro_batch_size = args.micro_batch_size
    paths = search_cfg["paths"][args.paths]
    device = resolve_device(args.device)
    if args.measure_timing is not None:
        out = Path(search_cfg["estimate"]["timing_file"])
        out.write_text(
            json.dumps(
                measure_timing(search_cfg, base, paths, device, args.measure_timing), indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {out}")
        return 0
    if args.dry_run_batches is not None:
        cfg = config_for(base, points[0], search_cfg["budget"], 0, paths)
        setup = prepare_training(cfg, device, causal=True)
        timing = time_training_batches(setup, cfg, device, args.dry_run_batches)
        timing["train_steps_per_epoch_at_search_budget"] = math.ceil(
            len(setup.train_loader) / setup.accumulation_steps
        )
        print(json.dumps({"config_0": points[0], "timing": timing}, indent=2))
        return 0

    results = run_search(search_cfg, base, paths, device)
    winner, report = select_winner(
        results["configs"], search_cfg["selection"]["margin"], relative_costs(search_cfg)
    )
    print_selection(report)
    print("winner params:", None if winner is None else winner["params"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
