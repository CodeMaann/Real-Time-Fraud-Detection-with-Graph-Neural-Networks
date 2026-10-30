"""Model explainability: SHAP, GNN perturbation, reason codes, and causal visualization."""

from fraudgnn.explain.gnn_explainer import GNNExplanation, HeteroGATExplainer
from fraudgnn.explain.reason_codes import build_reason_codes
from fraudgnn.explain.viz import extract_2hop_causal_subgraph, render_causal_neighborhood_html
from fraudgnn.explain.xgb_explainer import (
    FeatureExplanation,
    TransactionExplanation,
    XGBoostExplainer,
)

__all__ = [
    "FeatureExplanation",
    "GNNExplanation",
    "HeteroGATExplainer",
    "TransactionExplanation",
    "XGBoostExplainer",
    "build_reason_codes",
    "extract_2hop_causal_subgraph",
    "render_causal_neighborhood_html",
]
