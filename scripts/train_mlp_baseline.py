"""Plain MLP baseline on ONLY the 27 per-transaction edge features (no graph).

Answers: Does the GNN add anything over edge features alone?
Uses:
- 27 per-transaction edge features (StandardScaler fit on train only)
- WeightedBCEWithLogitsLoss (same pos_weight as GNN training)
- Adam optimizer (lr=0.001, weight_decay=1e-5)
- Early stopping on validation PR-AUC (patience 5)
- Threshold strictly selected on validation set only, evaluated on test set
- Sliced per-fraud-type recall across all 5 typologies
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import HeteroData

from fraudgnn.models.losses import WeightedBCEWithLogitsLoss
from fraudgnn.training.evaluate import compute_metrics


class PlainMLP(nn.Module):
    """3-layer Multi-Layer Perceptron for tabular transaction classification."""

    def __init__(self, in_dim: int = 27, hidden_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting raw logit."""
        return self.net(x).squeeze(-1)


def train_and_eval_mlp(
    data_dir: str = "data/processed",
    raw_data_path: str = "data/raw/transactions.parquet",
    batch_size: int = 2048,
    max_epochs: int = 30,
    patience: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Train MLP on edge features and evaluate on test set."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    print("=" * 70)
    print("  DIAGNOSTIC 1: PLAIN MLP ON 27 EDGE FEATURES ONLY (NO GRAPH)")
    print("=" * 70)

    # 1. Load precomputed edge features and labels from graph splits
    p_dir = Path(data_dir)
    train_graph: HeteroData = torch.load(p_dir / "train_graph.pt", weights_only=False)
    val_graph: HeteroData = torch.load(p_dir / "val_graph.pt", weights_only=False)
    test_graph: HeteroData = torch.load(p_dir / "test_graph.pt", weights_only=False)

    target_rel = ("user", "transacts_at", "merchant")

    X_train = train_graph[target_rel].edge_attr.float()
    y_train = train_graph[target_rel].edge_label.float()

    val_mask = val_graph[target_rel].val_mask
    X_val = val_graph[target_rel].edge_attr[val_mask].float()
    y_val = val_graph[target_rel].edge_label[val_mask].float()

    test_mask = test_graph[target_rel].test_mask
    X_test = test_graph[target_rel].edge_attr[test_mask].float()
    y_test = test_graph[target_rel].edge_label[test_mask].float()

    # Load raw metadata for test evaluation slicing
    df_raw = pd.read_parquet(raw_data_path)
    timestamps = pd.to_datetime(df_raw["timestamp"])
    t_min = timestamps.min()
    t_max = timestamps.max()
    dur = t_max - t_min
    train_cut = t_min + dur * 0.60
    val_cut = t_min + dur * 0.80

    df_val = df_raw[(timestamps > train_cut) & (timestamps <= val_cut)].reset_index(drop=True)
    df_test = df_raw[timestamps > val_cut].reset_index(drop=True)

    val_amounts = df_val["amount"].to_numpy()
    val_fraud_types = df_val["fraud_type"].to_numpy()

    test_amounts = df_test["amount"].to_numpy()
    test_fraud_types = df_test["fraud_type"].to_numpy()

    print(f"Data shapes: Train={X_train.shape[0]:,}, Val={X_val.shape[0]:,}, Test={X_test.shape[0]:,}")
    print(f"Features: {X_train.shape[1]} dims")

    # DataLoader
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Model and Loss
    model = PlainMLP(in_dim=X_train.shape[1], hidden_dim=64, dropout=0.2).to(device)
    n_pos = int((y_train == 1).sum().item())
    n_neg = int((y_train == 0).sum().item())
    pos_weight = float(n_neg / max(1, n_pos))
    loss_fn = WeightedBCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    best_val_pr_auc = -1.0
    patience_cnt = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_val_thresh = 0.5
    best_epoch = 0

    print("\nTraining MLP...")
    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total_loss = 0.0
        n_b = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_b += 1

        tr_loss = total_loss / max(1, n_b)

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            v_logits = model(X_val)
            v_probs = torch.sigmoid(v_logits).numpy()

        val_eval = compute_metrics(
            y_true=y_val.numpy(),
            y_pred_proba=v_probs,
            amounts=val_amounts,
            review_cost=10.0,
            default_fraud_amount=500.0,
            fraud_types=val_fraud_types,
        )
        elapsed = time.perf_counter() - t0

        print(
            f"Epoch {epoch:02d}/{max_epochs:02d} [{elapsed:.1f}s] - "
            f"Loss: {tr_loss:.4f} | Val PR-AUC: {val_eval.pr_auc:.4f} | "
            f"Val F1: {val_eval.f1:.4f} (thresh={val_eval.best_threshold:.4f}) | "
            f"Val Recall@1%FPR: {val_eval.recall_at_1pct_fpr:.4f}"
        )

        if val_eval.pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_eval.pr_auc
            best_val_thresh = val_eval.best_threshold
            best_epoch = epoch
            patience_cnt = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, best val PR-AUC: {best_val_pr_auc:.4f})")
                break

    assert best_state is not None
    model.load_state_dict(best_state)

    # Test Evaluation
    print("\n" + "=" * 70)
    print("  EVALUATING ON UNSEEN TEST PARTITION (Threshold from Validation)")
    print("=" * 70)
    model.eval()
    t0_score = time.perf_counter()
    with torch.no_grad():
        t_logits = model(X_test)
        test_probs = torch.sigmoid(t_logits).numpy()
    score_time = time.perf_counter() - t0_score
    latency_ms = (score_time / len(test_probs)) * 1000.0

    test_eval = compute_metrics(
        y_true=y_test.numpy(),
        y_pred_proba=test_probs,
        amounts=test_amounts,
        review_cost=10.0,
        default_fraud_amount=500.0,
        threshold_override=best_val_thresh,
        fraud_types=test_fraud_types,
        compute_ci=True,
    )

    # Log to MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fraud-gnn-baseline")
    with mlflow.start_run(run_name="mlp_edge_features_only") as run:
        mlflow.log_params(
            {
                "model_type": "plain_mlp",
                "epochs": max_epochs,
                "patience": patience,
                "batch_size": batch_size,
                "lr": 0.001,
                "pos_weight": pos_weight,
                "causal": True,
            }
        )
        mlflow.log_metrics(
            {
                "test_pr_auc": test_eval.pr_auc,
                "test_pr_auc_ci_lower": test_eval.pr_auc_ci_lower,
                "test_pr_auc_ci_upper": test_eval.pr_auc_ci_upper,
                "test_roc_auc": test_eval.roc_auc,
                "test_f1": test_eval.f1,
                "test_recall_at_1pct_fpr": test_eval.recall_at_1pct_fpr,
                "test_business_cost": test_eval.total_cost,
                "val_selected_threshold": best_val_thresh,
                "inference_latency_ms": latency_ms,
            }
        )
        for ft, rec in test_eval.per_type_recall.items():
            mlflow.log_metric(f"test_recall_{ft}", rec)

        run_id = run.info.run_id

    print(f"Validation PR-AUC (Primary) : {best_val_pr_auc:.4f}")
    print(f"Validation Selected Thresh  : {best_val_thresh:.4f}")
    print("-" * 70)
    print(
        f"TEST PR-AUC (Primary)       : {test_eval.pr_auc:.4f} [95% CI: {test_eval.pr_auc_ci_lower:.4f} - {test_eval.pr_auc_ci_upper:.4f}]"
    )
    print(f"TEST ROC-AUC                : {test_eval.roc_auc:.4f}")
    print(f"TEST Recall @ 1% FPR        : {test_eval.recall_at_1pct_fpr:.4f}")
    print(f"TEST F1 @ Val Threshold     : {test_eval.f1:.4f}")
    print(f"TEST Business Cost          : ${test_eval.total_cost:,.2f}")
    print(f"Inference Latency           : {latency_ms:.6f} ms/sample")
    print(f"MLflow Run ID               : {run_id}")
    print("-" * 70)
    print("PER-FRAUD-TYPE TEST RECALL:")
    for ft, rec in sorted(test_eval.per_type_recall.items()):
        print(f"  - {ft:<24s}: {rec:.4f}")
    print("=" * 70)

    return {
        "model_name": "Plain MLP (27 Edge Features Only, No Graph)",
        "run_id": run_id,
        "pr_auc": test_eval.pr_auc,
        "pr_auc_ci_lower": test_eval.pr_auc_ci_lower,
        "pr_auc_ci_upper": test_eval.pr_auc_ci_upper,
        "roc_auc": test_eval.roc_auc,
        "recall_1pct_fpr": test_eval.recall_at_1pct_fpr,
        "f1": test_eval.f1,
        "business_cost": test_eval.total_cost,
        "latency_ms": latency_ms,
        "threshold": best_val_thresh,
        "best_epoch": best_epoch,
        "per_type_recall": test_eval.per_type_recall,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train plain MLP on edge features only.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    train_and_eval_mlp(max_epochs=args.epochs, patience=args.patience)
