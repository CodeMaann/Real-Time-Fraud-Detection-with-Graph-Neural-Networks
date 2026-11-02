"""Evaluate explanations on random true positives, one model per explanation.

For each model in ``configs/explain.yaml`` and each fraud type, sample up to ``n_per_type``
random true positives (``is_fraud == 1`` and the model's test score >= its
validation-selected threshold), explain each with that model only, and run a deletion test:

- XGBoost (SHAP): replace the top-k risk-raising features with their training-split median.
- HeteroGAT (perturbation): remove the top-k risk-raising components together.

Each deletion is compared with removing k random features/components of the same model. An
explanation counts as faithful for a case when removing its top-k lowers the score more than
the random removals do on average.

Writes ``docs/examples/README.md`` (generated, do not edit), a few interactive neighborhood
pages, and the per-case results to ``data/processed/explanation_eval.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from fraudgnn.config import load_baseline_config, load_gnn_config
from fraudgnn.data.features import compute_tabular_features, get_feature_column_names
from fraudgnn.explain.gnn_explainer import HeteroGATExplainer
from fraudgnn.explain.reason_codes import build_reason_codes
from fraudgnn.explain.viz import render_causal_neighborhood_html, render_neighborhood_png
from fraudgnn.explain.xgb_explainer import XGBoostExplainer
from fraudgnn.graph.temporal_loader import CausalLinkNeighborLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_unified import TARGET_REL, split_targets
from scripts.train_label_free_xgb import compute_label_free_graph_features


def explain_xgb_case(
    explainer: XGBoostExplainer,
    row: pd.Series,
    tx_id: str,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """SHAP explanation plus deletion test (top-k vs random-k features to median)."""
    t0 = time.perf_counter()
    exp = explainer.explain_transaction(row, transaction_id=tx_id, top_k=cfg["top_k"])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    ranked = sorted(exp.all_shap_values.items(), key=lambda kv: kv[1], reverse=True)
    top = [name for name, val in ranked if val > 0][: cfg["faithfulness_k"]]
    names = explainer.feature_names
    variants = [top] + [
        list(rng.choice(names, size=len(top), replace=False))
        for _ in range(cfg["random_baselines"])
    ]
    rows = pd.DataFrame([row[names].to_dict()] * (len(variants) + 1))
    for i, removed in enumerate(variants, start=1):
        for name in removed:
            rows.loc[i, name] = explainer.reference_values[name]
    proba = explainer.predict(rows)
    return {
        "explanation": exp,
        "score_explained": float(proba[0]),
        "top_attribution": ranked[0][0] if ranked[0][1] > 0 else None,
        "removed_top": top,
        "delta_top": float(proba[0] - proba[1]) if top else None,
        "delta_random": float(np.mean(proba[0] - proba[2:])) if top else None,
        "elapsed_ms": elapsed_ms,
    }


def explain_gnn_case(
    explainer: HeteroGATExplainer,
    batch: Any,
    tx_id: str,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Perturbation explanation plus deletion test (top-k vs random-k components)."""
    exp = explainer.explain_subgraph(
        batch, transaction_id=tx_id, max_edge_ablations=cfg["max_edge_ablations"]
    )
    impacts = exp.component_impacts
    ranked = sorted(impacts.items(), key=lambda kv: kv[1], reverse=True)
    top = [name for name, val in ranked if val > 0][: cfg["faithfulness_k"]]
    result: dict[str, Any] = {
        "explanation": exp,
        "score_explained": exp.base_probability,
        "top_attribution": ranked[0][0] if ranked[0][1] > 0 else None,
        "removed_top": top,
        "delta_top": None,
        "delta_random": None,
        "elapsed_ms": exp.elapsed_time_ms,
    }
    if top:
        names = list(impacts)
        result["delta_top"] = exp.base_probability - explainer.predict_without(batch, top)
        random_deltas = [
            exp.base_probability
            - explainer.predict_without(
                batch, list(rng.choice(names, size=len(top), replace=False))
            )
            for _ in range(cfg["random_baselines"])
        ]
        result["delta_random"] = float(np.mean(random_deltas))
    return result


def summarize(cases: list[dict[str, Any]], available: int) -> dict[str, Any]:
    """Aggregate per (model, fraud type) results."""
    tested = [c for c in cases if c["delta_top"] is not None]
    faithful = sum(c["delta_top"] > c["delta_random"] for c in tested)
    tops = Counter(c["top_attribution"] for c in cases if c["top_attribution"])
    most, most_n = tops.most_common(1)[0] if tops else ("none", 0)
    return {
        "explained": len(cases),
        "available": available,
        "tested": len(tested),
        "faithful": int(faithful),
        "mean_delta_top": float(np.mean([c["delta_top"] for c in tested])) if tested else None,
        "mean_delta_random": float(np.mean([c["delta_random"] for c in tested]))
        if tested
        else None,
        "median_ms": float(np.median([c["elapsed_ms"] for c in cases])) if cases else None,
        "most_common_top": most,
        "most_common_top_n": most_n,
    }


def fmt(value: float | None, spec: str) -> str:
    """Format an optional number."""
    return "not run" if value is None else format(value, spec)


def write_readme(
    out_dir: Path,
    exp_cfg: dict[str, Any],
    model_names: dict[str, str],
    summaries: dict[str, dict[str, dict[str, Any]]],
    examples: dict[str, dict[str, dict[str, Any]]],
) -> None:
    """Write the generated docs/examples/README.md."""
    k = exp_cfg["faithfulness_k"]
    md = [
        "# Explanation Evaluation",
        "",
        "Generated by `scripts/make_explanation_examples.py` (settings in `configs/explain.yaml`). "
        "Do not edit by hand.",
        "",
        f"For each model and fraud type, up to {exp_cfg['n_per_type']} random true positives "
        f"(seed {exp_cfg['seed']}) are explained **with that model only**. Labels and fraud types "
        "are used only to pick and group the cases, never as explainer inputs.",
        "",
        f"**Deletion test:** remove the explanation's top {k} risk-raising attributions together "
        f"and compare the drop in score with removing {k} random attributions of the same model "
        f"(mean of {exp_cfg['random_baselines']} draws). XGBoost features are replaced by their "
        "training-split median; HeteroGAT components (device/IP embedding, edge-feature groups) "
        "are removed. *Faithful* = the top-k drop is larger than the random drop. Cases with no "
        "risk-raising attribution are not tested.",
        "",
    ]
    for key, per_type in summaries.items():
        md += [
            f"## {model_names[key]}",
            "",
            "| Fraud type | Explained / TPs available | Faithful (k/n tested) | Mean drop, top-k | "
            "Mean drop, random-k | Most common top attribution | Median ms per explanation |",
            "|---|---|---|---|---|---|---|",
        ]
        for ft, s in per_type.items():
            md.append(
                f"| {ft} | {s['explained']}/{s['available']} | {s['faithful']}/{s['tested']} | "
                f"{fmt(s['mean_delta_top'], '.4f')} | {fmt(s['mean_delta_random'], '.4f')} | "
                f"`{s['most_common_top']}` ({s['most_common_top_n']}/{s['explained']}) | "
                f"{fmt(s['median_ms'], '.1f')} |"
            )
        md.append("")

    png_cfg = exp_cfg.get("ring_png")
    if png_cfg:
        md += [
            "## Static example",
            "",
            f"![2-hop neighborhood of a flagged {png_cfg['fraud_type']} transaction]"
            f"({Path(png_cfg['path']).name})",
            "",
            "The fraud type is a post-hoc label shown for auditing only; it is not a model input.",
            "",
        ]
    md += ["## One example per model and fraud type", ""]
    for key, per_type in examples.items():
        for ft, ex in per_type.items():
            md.append(f"### {model_names[key]}: `{ex['tx_id']}` ({ft}, post-hoc label)")
            md.append("")
            md += [f"> {s}" for s in ex["reasons"]]
            if ex.get("html"):
                md += ["", f"Neighborhood view: [{ex['html']}]({ex['html']})"]
            md.append("")

    md += [
        "## Limitations",
        "",
        "- The deletion test checks that the named attributions matter *to the model*; it does "
        "not show that they match the simulator's fraud mechanism.",
        "- SHAP deletion uses the training median as the 'removed' value; other reference "
        "choices can give different drops.",
        "- HeteroGAT deltas are one-at-a-time and not additive. Each explanation scores the "
        "transaction on its own causal subgraph, the same input as the isolated batched test "
        "scores used to select true positives.",
        "- Timings are single-process CPU wall-clock on this machine.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(md), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the explanation evaluation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/explain.yaml")
    args = parser.parse_args(argv)

    exp_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    eval_cfg = yaml.safe_load(Path(exp_cfg["eval_config"]).read_text(encoding="utf-8"))
    base_cfg = load_baseline_config(eval_cfg["baseline_config"])
    entries = {e["key"]: e for e in eval_cfg["models"]}
    rng = np.random.default_rng(exp_cfg["seed"])

    df_raw = pd.read_parquet(base_cfg.data_path).sort_values("timestamp", ignore_index=True)
    ts = pd.to_datetime(df_raw["timestamp"])
    t_min, dur = ts.min(), ts.max() - ts.min()
    train_mask = ts <= t_min + dur * base_cfg.split.train_ratio
    test_mask = ts > t_min + dur * (base_cfg.split.train_ratio + base_cfg.split.val_ratio)
    test_offset = int(np.flatnonzero(test_mask.to_numpy())[0])
    df_test = df_raw[test_mask].reset_index(drop=True)

    with np.load(eval_cfg["predictions_out"]) as npz:
        preds = {k: npz[k] for k in npz.files}
    thresholds = json.loads(Path(eval_cfg["thresholds_out"]).read_text(encoding="utf-8"))
    if not np.array_equal(preds["y_true"], df_test["is_fraud"].to_numpy()):
        raise ValueError("Saved predictions are not aligned with the test split.")

    out_dir = Path(exp_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("tx_*.html"):
        stale.unlink()

    model_names = {m["key"]: entries[m["key"]]["name"] for m in exp_cfg["models"]}
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    examples: dict[str, dict[str, dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []

    for model in exp_cfg["models"]:
        key, entry = model["key"], entries[model["key"]]
        threshold = float(thresholds[key]["val_selected_threshold"])
        proba = preds[key]
        print(f"\n=== {entry['name']} (threshold {threshold:.4f}) ===")

        if model["explainer"] == "shap":
            feats = compute_tabular_features(df_raw, include_label_features=False)
            graph_feats = compute_label_free_graph_features(df_raw)
            feats = pd.concat([feats, graph_feats], axis=1)
            cols = get_feature_column_names(include_label_features=False) + list(
                graph_feats.columns
            )
            medians = feats.loc[train_mask, cols].median()
            xgb_explainer = XGBoostExplainer(
                model_path=entry["checkpoint"],
                feature_names=cols,
                reference_values={c: float(medians[c]) for c in cols},
            )
        else:
            test_graph = torch.load(
                Path(eval_cfg["processed_dir"]) / "test_graph.pt", weights_only=False
            )
            targets = split_targets(test_graph, "test_mask")
            if not np.array_equal(targets["edge_label"].numpy(), preds["y_true"]):
                raise ValueError("Test graph targets are not aligned with the test split.")
            gnn_cfg = load_gnn_config(entry["gnn_config"])
            gnn_explainer = HeteroGATExplainer.from_checkpoint(
                entry["checkpoint"],
                gnn_cfg,
                test_graph.node_types,
                test_graph.edge_types,
                conv_type=entry.get("conv_type"),
                use_device_ip=bool(entry["use_device_ip"]),
            )
            num_neighbors = gnn_cfg.training.num_neighbors[: gnn_cfg.model.num_layers]

        summaries[key], examples[key] = {}, {}
        for ft in exp_cfg["fraud_types"]:
            is_tp = (
                (df_test["fraud_type"] == ft).to_numpy()
                & (df_test["is_fraud"] == 1).to_numpy()
                & (proba >= threshold)
            )
            candidates = np.flatnonzero(is_tp)
            chosen = rng.choice(
                candidates, size=min(exp_cfg["n_per_type"], len(candidates)), replace=False
            )
            cases: list[dict[str, Any]] = []
            for test_idx in sorted(int(i) for i in chosen):
                tx_id = str(df_test.at[test_idx, "transaction_id"])
                if model["explainer"] == "shap":
                    case = explain_xgb_case(
                        xgb_explainer, feats.iloc[test_offset + test_idx], tx_id, exp_cfg, rng
                    )
                    reasons = build_reason_codes(
                        entry["name"],
                        threshold,
                        xgb_explanation=case["explanation"],
                        transaction_id=tx_id,
                    )
                else:
                    sl = slice(test_idx, test_idx + 1)
                    loader = CausalLinkNeighborLoader(
                        data=test_graph,
                        num_neighbors=num_neighbors,
                        edge_label_index=(TARGET_REL, targets["edge_index"][:, sl]),
                        edge_label=targets["edge_label"][sl],
                        edge_time=targets["edge_time"][sl],
                        edge_attr=targets["edge_attr"][sl],
                        device_index=targets["device_index"][sl],
                        ip_index=targets["ip_index"][sl],
                        batch_size=1,
                        shuffle=False,
                    )
                    case = explain_gnn_case(gnn_explainer, next(iter(loader)), tx_id, exp_cfg, rng)
                    reasons = build_reason_codes(
                        entry["name"],
                        threshold,
                        gnn_explanation=case["explanation"],
                        transaction_id=tx_id,
                    )
                case.pop("explanation")
                case.update(
                    model=key,
                    fraud_type=ft,
                    tx_id=tx_id,
                    test_idx=test_idx,
                    score_batched=float(proba[test_idx]),
                    reasons=reasons,
                )
                cases.append(case)

            summaries[key][ft] = summarize(cases, len(candidates))
            records += cases
            s = summaries[key][ft]
            print(
                f"  {ft}: explained {s['explained']}/{s['available']}, faithful "
                f"{s['faithful']}/{s['tested']}"
            )
            if cases:
                ex = cases[0]
                html = None
                if exp_cfg["html_examples"]:
                    html = f"tx_{ex['tx_id']}_{key}_{ft}.html"
                    render_causal_neighborhood_html(
                        df=df_raw,
                        target_tx_id=ex["tx_id"],
                        output_path=str(out_dir / html),
                        reason_codes=ex["reasons"],
                        risk_score=ex["score_explained"],
                        threshold=threshold,
                        post_hoc_fraud_type=ft,
                        is_fraud_ground_truth=True,
                        model_name=entry["name"],
                        cdn_resources=exp_cfg["viz_resources"],
                    )
                examples[key][ft] = {
                    "tx_id": ex["tx_id"],
                    "reasons": ex["reasons"],
                    "html": html,
                    "score": ex["score_explained"],
                    "threshold": threshold,
                }

    png_cfg = exp_cfg.get("ring_png")
    if png_cfg:
        ex = examples[png_cfg["model"]][png_cfg["fraud_type"]]
        render_neighborhood_png(
            df=df_raw,
            target_tx_id=ex["tx_id"],
            output_path=png_cfg["path"],
            model_name=model_names[png_cfg["model"]],
            risk_score=ex["score"],
            threshold=ex["threshold"],
            post_hoc_fraud_type=png_cfg["fraud_type"],
            width_in=png_cfg["width_in"],
            height_in=png_cfg["height_in"],
            dpi=png_cfg["dpi"],
            layout_seed=png_cfg["layout_seed"],
        )
    write_readme(out_dir, exp_cfg, model_names, summaries, examples)
    Path(exp_cfg["results_out"]).write_text(
        json.dumps({"summaries": summaries, "cases": records}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nWrote {out_dir / 'README.md'} and {exp_cfg['results_out']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
