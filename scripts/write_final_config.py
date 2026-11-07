"""Write configs/hetero_gat_final.yaml from the finished hyperparameter search (protocol §5).

Refuses unless every search configuration has finished. The final config uses the winner's
hyperparameters, ALL training targets, max 30 epochs, patience 6, the isolated loader and a
per-epoch resume state on Google Drive. Seeds are given at run time with
``train_gnn --seed 42|43|44`` (each seed gets its own checkpoint and model file).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.hparam_search import print_selection, relative_costs, select_winner


def build_final(
    base: dict[str, Any], winner: dict[str, Any], final: dict[str, Any]
) -> dict[str, Any]:
    """Final YAML document: base config + winner params + final-run budget."""
    doc = json.loads(json.dumps(base))  # deep copy of plain YAML data
    p = winner["params"]
    doc["model"]["hidden_dim"] = p["hidden_dim"]
    doc["model"]["dropout"] = p["dropout"]
    t = doc["training"]
    t.update(
        learning_rate=p["learning_rate"],
        weight_decay=p["weight_decay"],
        num_neighbors=p["num_neighbors"],
        loss_type=p["loss_type"],
        epochs=final["max_epochs"],
        patience=final["patience"],
        seed=final["seeds"][0],
        train_target_fraction=None,
        model_save_path=final["model_save_path"],
        checkpoint_path=final["checkpoint_path"],
        mlflow_experiment_name=final["mlflow_experiment"],
    )
    return doc


def main(argv: list[str] | None = None) -> int:
    """Write the final config."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--search-config", default="configs/hparam_search.yaml")
    parser.add_argument(
        "--results", required=True, help="hparam_search.json with every configuration finished"
    )
    parser.add_argument("--out", default="configs/hetero_gat_final.yaml")
    args = parser.parse_args(argv)

    search_cfg = yaml.safe_load(Path(args.search_config).read_text(encoding="utf-8"))
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    unfinished = [e["index"] for e in results["configs"] if e.get("status") != "finished"]
    if unfinished:
        raise SystemExit(
            f"Search not finished (configs {unfinished}); refusing to write {args.out}"
        )
    winner, report = select_winner(
        results["configs"], search_cfg["selection"]["margin"], relative_costs(search_cfg)
    )
    assert winner is not None
    print_selection(report)
    base = yaml.safe_load(Path(search_cfg["base_config"]).read_text(encoding="utf-8"))
    doc = build_final(base, winner, search_cfg["final"])
    header = (
        "# ── Final HeteroGAT runs (docs/protocol.md §5) ── written by scripts/write_final_config.py\n"
        f"# Search winner: config {winner['index']} (best validation PR-AUC "
        f"{winner['best_val_pr_auc']:.4f} at its search budget).\n"
        f"# Selection: {report['reason']}.\n"
        "# Run once per seed:\n"
        f"#   python -m fraudgnn.training.train_gnn --config {args.out} --causal --device cuda "
        f"--resume --seed <{'|'.join(str(s) for s in search_cfg['final']['seeds'])}>\n"
    )
    Path(args.out).write_text(header + yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    print(Path(args.out).read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
