"""Unit tests for explainability.

Validates:
1. XGBoost SHAP explainer output structure and data-derived reason codes.
2. HeteroGAT perturbation explainer output structure and the shared predict path.
3. Reason codes: exactly one model per explanation, 3-5 sentences.
4. Causal future-injection invariance of features and SHAP explanations.
5. Single-sample CPU explanation latency stays under 2 seconds.

Tests that need the local dataset, graphs or checkpoints are skipped when those are absent.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from scripts.train_label_free_xgb import compute_label_free_graph_features

from fraudgnn.config import load_gnn_config
from fraudgnn.data.features import compute_tabular_features, get_feature_column_names
from fraudgnn.explain.gnn_explainer import GNNExplanation, HeteroGATExplainer
from fraudgnn.explain.reason_codes import build_reason_codes
from fraudgnn.explain.xgb_explainer import (
    FeatureExplanation,
    TransactionExplanation,
    XGBoostExplainer,
)
from fraudgnn.graph.temporal_loader import CausalLinkNeighborLoader

RAW = Path("data/raw/transactions.parquet")
TEST_GRAPH = Path("data/processed/test_graph.pt")
XGB_MODEL = Path("models/xgboost_label_free.json")
GAT_MODEL = Path("models/best_hetero_gat_pna_causal.pt")
GAT_CONFIG = "configs/hetero_gat.yaml"
TARGET_REL = ("user", "transacts_at", "merchant")
THRESHOLD = 0.5  # arbitrary display value for sentence-structure tests

needs_data = pytest.mark.skipif(
    not all(p.exists() for p in (RAW, TEST_GRAPH, XGB_MODEL, GAT_MODEL)),
    reason="local dataset, graphs or checkpoints not available",
)


@pytest.fixture(scope="module")
def real_data_context() -> dict[str, Any]:
    """Load features, the test graph and a single-target batch for the first test edge."""
    df_raw = pd.read_parquet(RAW).sort_values("timestamp", ignore_index=True)
    timestamps = pd.to_datetime(df_raw["timestamp"])
    t_min, t_max = timestamps.min(), timestamps.max()
    val_cut = t_min + (t_max - t_min) * 0.80
    first_test_idx = int(np.flatnonzero((timestamps > val_cut).to_numpy())[0])

    df_tab = compute_tabular_features(df_raw, include_label_features=False)
    df_graph = compute_label_free_graph_features(df_raw)
    cols = get_feature_column_names(include_label_features=False) + list(df_graph.columns)
    df_features = pd.concat([df_tab, df_graph], axis=1)

    test_graph = torch.load(TEST_GRAPH, weights_only=False)
    mask = test_graph[TARGET_REL].test_mask
    loader = CausalLinkNeighborLoader(
        data=test_graph,
        num_neighbors=[10, 10],
        edge_label_index=(TARGET_REL, test_graph[TARGET_REL].edge_index[:, mask][:, 0:1]),
        edge_label=test_graph[TARGET_REL].edge_label[mask][0:1],
        edge_time=test_graph[TARGET_REL].edge_time[mask][0:1],
        edge_attr=test_graph[TARGET_REL].edge_attr[mask][0:1],
        device_index=test_graph["user", "uses_device", "device"].edge_index[1][mask][0:1],
        ip_index=test_graph["user", "uses_ip", "ip"].edge_index[1][mask][0:1],
        batch_size=1,
        shuffle=False,
    )
    medians = df_features[cols].median()
    return {
        "df_raw": df_raw,
        "df_features": df_features,
        "feature_cols": cols,
        "medians": {c: float(medians[c]) for c in cols},
        "test_graph": test_graph,
        "batch": next(iter(loader)),
        "first_test_raw_idx": first_test_idx,
    }


def _gnn_explainer(ctx: dict[str, Any]) -> HeteroGATExplainer:
    graph = ctx["test_graph"]
    return HeteroGATExplainer.from_checkpoint(
        GAT_MODEL,
        load_gnn_config(GAT_CONFIG),
        graph.node_types,
        graph.edge_types,
        conv_type="pna",
        use_device_ip=True,
    )


@needs_data
def test_xgb_explainer_structure(real_data_context: dict[str, Any]) -> None:
    """XGBoostExplainer returns top-5 SHAP attributions with data-derived references."""
    explainer = XGBoostExplainer(
        model_path=str(XGB_MODEL),
        feature_names=real_data_context["feature_cols"],
        reference_values=real_data_context["medians"],
    )
    idx = real_data_context["first_test_raw_idx"]
    feat_row = real_data_context["df_features"].iloc[idx]
    tx_id = str(real_data_context["df_raw"].iloc[idx]["transaction_id"])

    explanation = explainer.explain_transaction(feat_row, transaction_id=tx_id, top_k=5)

    assert isinstance(explanation, TransactionExplanation)
    assert explanation.transaction_id == tx_id
    assert 0.0 <= explanation.predicted_probability <= 1.0
    assert len(explanation.top_features) == 5
    for item in explanation.top_features:
        assert isinstance(item, FeatureExplanation)
        assert item.feature_name in real_data_context["feature_cols"]
        assert np.isfinite(item.shap_value)
        assert "training median" in item.reason_code
        assert ("raises" in item.reason_code) == (item.shap_value > 0)

    batch_proba = explainer.predict(real_data_context["df_features"].iloc[[idx]])
    assert np.isclose(batch_proba[0], explanation.predicted_probability)


@needs_data
def test_gnn_explainer_structure(real_data_context: dict[str, Any]) -> None:
    """HeteroGATExplainer deltas are consistent with its own predict path."""
    explainer = _gnn_explainer(real_data_context)
    batch = real_data_context["batch"]

    explanation = explainer.explain_subgraph(batch, transaction_id="tx", max_edge_ablations=5)

    assert isinstance(explanation, GNNExplanation)
    assert explanation.explanation_method == "perturbation_ablation"
    assert set(explanation.edge_group_impacts) == set(explainer.EDGE_FEATURE_GROUPS)
    assert len(explanation.top_neighbor_edges) <= 5
    base = explainer.predict(batch)
    assert np.isclose(base, explanation.base_probability)
    assert np.isclose(
        base - explainer.predict_without(batch, ["device_binding"]), explanation.device_impact
    )
    assert np.isclose(
        base - explainer.predict_without(batch, ["amount"]),
        explanation.edge_group_impacts["amount"],
    )


@needs_data
def test_reason_codes_one_model_each(real_data_context: dict[str, Any]) -> None:
    """Each explanation is built from exactly one model and has 3-5 sentences."""
    xgb_explainer = XGBoostExplainer(
        model_path=str(XGB_MODEL),
        feature_names=real_data_context["feature_cols"],
        reference_values=real_data_context["medians"],
    )
    idx = real_data_context["first_test_raw_idx"]
    xgb_exp = xgb_explainer.explain_transaction(real_data_context["df_features"].iloc[idx])
    gnn_exp = _gnn_explainer(real_data_context).explain_subgraph(
        real_data_context["batch"], max_edge_ablations=5
    )

    for reasons in (
        build_reason_codes("XGBoost", THRESHOLD, xgb_explanation=xgb_exp),
        build_reason_codes("HeteroGAT", THRESHOLD, gnn_explanation=gnn_exp),
    ):
        assert 3 <= len(reasons) <= 5, reasons
        assert all(r.strip().endswith(".") for r in reasons)

    xgb_text = " ".join(build_reason_codes("XGBoost", THRESHOLD, xgb_explanation=xgb_exp))
    assert "HeteroGAT" not in xgb_text and "Removing" not in xgb_text

    with pytest.raises(ValueError):
        build_reason_codes("both", THRESHOLD, xgb_explanation=xgb_exp, gnn_explanation=gnn_exp)
    with pytest.raises(ValueError):
        build_reason_codes("none", THRESHOLD)


@needs_data
def test_causal_future_injection_invariance() -> None:
    """Injecting future transactions after t does not change features or SHAP for t."""
    df_raw = pd.read_parquet(RAW).sort_values("timestamp", ignore_index=True).head(500).copy()
    timestamps = pd.to_datetime(df_raw["timestamp"])
    target_idx = 250
    t_target = timestamps.iloc[target_idx]
    target_tx_id = str(df_raw.iloc[target_idx]["transaction_id"])

    tab_cols = get_feature_column_names(include_label_features=False)

    def features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        tab = compute_tabular_features(df, include_label_features=False)
        graph = compute_label_free_graph_features(df)
        cols = tab_cols + list(graph.columns)
        return pd.concat([tab[tab_cols], graph], axis=1)[cols], cols

    feat_base, cols = features(df_raw)
    explainer = XGBoostExplainer(model_path=str(XGB_MODEL), feature_names=cols)
    base_exp = explainer.explain_transaction(feat_base.iloc[target_idx], top_k=5)

    target = df_raw.iloc[target_idx]
    future_time = pd.Timestamp(t_target) + pd.Timedelta(days=2)
    future_rows = [
        {
            "transaction_id": f"tx_future_fake_{i}",
            "timestamp": future_time + pd.Timedelta(minutes=i),
            "user_id": target["user_id"],
            "merchant_id": target["merchant_id"],
            "card_id": target["card_id"],
            "device_id": target["device_id"],
            "ip_address": target["ip_address"],
            "amount": 99999.0 + i * 1000.0,
            "channel": "online",
            "card_present": 0,
            "is_fraud": 1,
            "fraud_type": "fraud_ring",
        }
        for i in range(10)
    ]
    df_inj = pd.concat([df_raw, pd.DataFrame(future_rows)], ignore_index=True)
    df_inj["timestamp"] = pd.to_datetime(df_inj["timestamp"])
    df_inj = df_inj.sort_values("timestamp").reset_index(drop=True)
    inj_idx = int(np.flatnonzero(df_inj["transaction_id"] == target_tx_id)[0])
    assert inj_idx == target_idx

    feat_inj, _ = features(df_inj)
    diff = np.max(
        np.abs(
            feat_base.iloc[target_idx].to_numpy(np.float32)
            - feat_inj.iloc[inj_idx].to_numpy(np.float32)
        )
    )
    assert diff == 0.0, f"Future injection leaked: max feature difference {diff}"

    inj_exp = explainer.explain_transaction(feat_inj.iloc[inj_idx], top_k=5)
    assert np.isclose(base_exp.predicted_probability, inj_exp.predicted_probability, atol=1e-6)
    for b, i in zip(base_exp.top_features, inj_exp.top_features, strict=True):
        assert b.feature_name == i.feature_name
        assert np.isclose(b.shap_value, i.shap_value, atol=1e-5)
        assert b.reason_code == i.reason_code


@needs_data
def test_single_sample_explanation_latency_cpu(real_data_context: dict[str, Any]) -> None:
    """One XGBoost and one HeteroGAT explanation each finish in under 2 seconds on CPU."""
    xgb_explainer = XGBoostExplainer(
        model_path=str(XGB_MODEL), feature_names=real_data_context["feature_cols"]
    )
    gnn_explainer = _gnn_explainer(real_data_context)
    feat_row = real_data_context["df_features"].iloc[real_data_context["first_test_raw_idx"]]

    start = time.perf_counter()
    xgb_explainer.explain_transaction(feat_row, top_k=5)
    xgb_s = time.perf_counter() - start

    start = time.perf_counter()
    gnn_explainer.explain_subgraph(real_data_context["batch"])
    gnn_s = time.perf_counter() - start

    assert xgb_s < 2.0, f"XGBoost explanation took {xgb_s:.3f}s"
    assert gnn_s < 2.0, f"HeteroGAT explanation took {gnn_s:.3f}s"
