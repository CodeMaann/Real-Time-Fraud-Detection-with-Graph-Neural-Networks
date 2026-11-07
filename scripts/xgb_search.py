"""Equal-effort random search for the label-free XGBoost (docs/protocol.md §6).

Default mode: sample ``n_configs`` grid points (fixed enumeration order and seed), fit each on
the training split with early stopping on validation, score validation PR-AUC only, and write
results after every configuration (finished ones are skipped on rerun). The test split and
the holdout are never touched.

``--final``: retrain the winner with each final seed and save the models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import average_precision_score

from fraudgnn.config import load_baseline_config
from fraudgnn.data.features import compute_tabular_features, get_feature_column_names

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.hparam_search import sample_configs
from scripts.train_label_free_xgb import compute_label_free_graph_features


def load_features(baseline_config: str) -> dict[str, Any]:
    """Label-free tabular + graph features and the temporal train/val masks."""
    cfg = load_baseline_config(baseline_config)
    df = pd.read_parquet(cfg.data_path).sort_values("timestamp", ignore_index=True)
    feats = compute_tabular_features(df, include_label_features=False)
    graph = compute_label_free_graph_features(df)
    feats = pd.concat([feats, graph], axis=1)
    cols = get_feature_column_names(include_label_features=False) + list(graph.columns)
    ts = pd.to_datetime(df["timestamp"])
    t0, dur = ts.min(), ts.max() - ts.min()
    train = ts <= t0 + dur * cfg.split.train_ratio
    val = (ts > t0 + dur * cfg.split.train_ratio) & (
        ts <= t0 + dur * (cfg.split.train_ratio + cfg.split.val_ratio)
    )
    return {
        "X_train": feats.loc[train, cols],
        "y_train": df.loc[train, "is_fraud"].to_numpy(dtype=np.int64),
        "X_val": feats.loc[val, cols],
        "y_val": df.loc[val, "is_fraud"].to_numpy(dtype=np.int64),
    }


def fit(
    data: dict[str, Any], params: dict[str, Any], fixed: dict[str, Any], seed: int
) -> xgb.XGBClassifier:
    """Fit one model with early stopping on validation PR-AUC."""
    y = data["y_train"]
    clf = xgb.XGBClassifier(
        **params,
        scale_pos_weight=float((y == 0).sum() / max(1, (y == 1).sum())),
        early_stopping_rounds=fixed["early_stopping_rounds"],
        eval_metric="aucpr",
        random_state=seed,
        tree_method=fixed["tree_method"],
        n_jobs=-1,
    )
    clf.fit(data["X_train"], y, eval_set=[(data["X_val"], data["y_val"])], verbose=False)
    return clf


def write_json(path: Path, doc: dict[str, Any]) -> None:
    """Atomic JSON write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_search(cfg: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Run or resume the search; returns the results document."""
    out = Path(cfg["results_out"])
    points = sample_configs(cfg["space"], cfg["n_configs"], cfg["seed"])
    if out.exists():
        doc = json.loads(out.read_text(encoding="utf-8"))
        if [e["params"] for e in doc["configs"]] != points:
            raise ValueError(f"{out} was written for a different sample; refusing")
    else:
        doc = {
            "protocol": "docs/protocol.md section 6",
            "sampling_seed": cfg["seed"],
            "fixed": cfg["fixed"],
            "configs": [
                {"index": i, "params": p, "status": "pending"} for i, p in enumerate(points)
            ],
        }
    for entry in doc["configs"]:
        if entry["status"] == "finished":
            continue
        t0 = time.perf_counter()
        clf = fit(data, entry["params"], cfg["fixed"], cfg["fixed"]["random_state"])
        proba = clf.predict_proba(data["X_val"])[:, 1]
        entry.update(
            status="finished",
            val_pr_auc=float(average_precision_score(data["y_val"], proba)),
            best_iteration=int(clf.best_iteration),
            seconds=time.perf_counter() - t0,
        )
        done = [e for e in doc["configs"] if e["status"] == "finished"]
        doc["winner_index"] = max(done, key=lambda e: (e["val_pr_auc"], -e["index"]))["index"]
        write_json(out, doc)
        print(f"config {entry['index']}: val PR-AUC {entry['val_pr_auc']:.4f} {entry['params']}")
    return doc


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/xgb_search.yaml")
    parser.add_argument("--final", action="store_true", help="Retrain the winner per final seed")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data = load_features(cfg["baseline_config"])
    if not args.final:
        run_search(cfg, data)
        return 0

    doc = json.loads(Path(cfg["results_out"]).read_text(encoding="utf-8"))
    if any(e["status"] != "finished" for e in doc["configs"]):
        raise SystemExit("Search not finished; refusing to train final models")
    winner = doc["configs"][doc["winner_index"]]
    finals = []
    for seed in cfg["final"]["seeds"]:
        path = Path(cfg["final"]["model_dir"]) / f"xgboost_label_free_final_seed{seed}.json"
        if path.exists():
            raise FileExistsError(f"{path} exists; refusing to overwrite")
        clf = fit(data, winner["params"], cfg["fixed"], seed)
        clf.save_model(str(path))
        val_pr = float(
            average_precision_score(data["y_val"], clf.predict_proba(data["X_val"])[:, 1])
        )
        finals.append({"seed": seed, "model": path.as_posix(), "val_pr_auc": val_pr})
        print(f"final seed {seed}: val PR-AUC {val_pr:.4f} -> {path}")
    doc["final_models"] = finals
    write_json(Path(cfg["results_out"]), doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
