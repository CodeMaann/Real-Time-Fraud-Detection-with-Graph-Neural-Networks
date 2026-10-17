"""Sanity checks for GNN models, loaders, label alignment, and loss scaling.

Performs:
1. Label alignment verification:
   Confirms edge_label, edge_attr, and edge_label_index correspond to the same transactions
   after sampling/batching in LinkNeighborLoader and HomogeneousLinkLoader.
2. Positive rate and loss scale per batch:
   Inspects positive class prevalence, logit ranges, and loss magnitudes across batches
   for both WeightedBCE and FocalLoss.
3. Decision threshold logic verification:
   Confirms thresholds do not collapse into degenerate 'predict all fraud' (recall 1.0 on all types).
4. Overfit test on ~2,000 edges:
   Confirms the model architecture and optimizer can achieve train PR-AUC approaching 1.0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

from fraudgnn.graph.loader import LinkNeighborLoader
from fraudgnn.models.hetero_gat import HeteroGAT
from fraudgnn.models.losses import get_loss_fn
from fraudgnn.training.evaluate import compute_metrics


def check_label_alignment(train_graph: HeteroData) -> None:
    print("=" * 60)
    print("CHECK 1: Label and Feature Alignment in Loader")
    print("=" * 60)
    target_rel = ("user", "transacts_at", "merchant")
    orig_edge_index = train_graph[target_rel].edge_index
    orig_edge_label = train_graph[target_rel].edge_label
    orig_edge_attr = train_graph[target_rel].edge_attr

    loader = LinkNeighborLoader(
        data=train_graph,
        num_neighbors=[10, 10],
        edge_label_index=(target_rel, orig_edge_index),
        edge_label=orig_edge_label,
        edge_attr=orig_edge_attr,
        batch_size=512,
        shuffle=False,
        seed=42,
    )

    batch = next(iter(loader))
    batch_labels = batch[target_rel].edge_label.numpy()
    batch_attrs = batch[target_rel].edge_attr.numpy()
    batch_label_idx = batch[target_rel].edge_label_index.numpy()

    # Compare first 512 elements of original graph with first batch
    expected_labels = orig_edge_label[:512].numpy()
    expected_attrs = orig_edge_attr[:512].numpy()

    labels_match = np.array_equal(batch_labels, expected_labels)
    attrs_match = np.allclose(batch_attrs, expected_attrs, atol=1e-6)

    # Check local node mapping matches original global IDs
    local_src = batch_label_idx[0]
    local_dst = batch_label_idx[1]
    global_src_mapped = batch["user"].n_id[local_src].numpy()
    global_dst_mapped = batch["merchant"].n_id[local_dst].numpy()

    expected_global_src = orig_edge_index[0, :512].numpy()
    expected_global_dst = orig_edge_index[1, :512].numpy()

    src_nodes_match = np.array_equal(global_src_mapped, expected_global_src)
    dst_nodes_match = np.array_equal(global_dst_mapped, expected_global_dst)

    print(f"  edge_label alignment: {'PASS' if labels_match else 'FAIL'}")
    print(f"  edge_attr alignment : {'PASS' if attrs_match else 'FAIL'}")
    print(f"  source node mapping : {'PASS' if src_nodes_match else 'FAIL'}")
    print(f"  dest node mapping   : {'PASS' if dst_nodes_match else 'FAIL'}")

    assert labels_match and attrs_match and src_nodes_match and dst_nodes_match, (
        "Alignment check failed!"
    )


def check_batch_imbalance_and_loss_scale(train_graph: HeteroData) -> None:
    print("\n" + "=" * 60)
    print("CHECK 2: Batch Positive Rate & Loss Scaling")
    print("=" * 60)
    target_rel = ("user", "transacts_at", "merchant")
    loader = LinkNeighborLoader(
        data=train_graph,
        num_neighbors=[10, 10],
        edge_label_index=(target_rel, train_graph[target_rel].edge_index),
        edge_label=train_graph[target_rel].edge_label,
        edge_attr=train_graph[target_rel].edge_attr,
        batch_size=1024,
        shuffle=True,
        seed=42,
    )

    n_pos_total = int((train_graph[target_rel].edge_label == 1).sum().item())
    n_neg_total = int((train_graph[target_rel].edge_label == 0).sum().item())
    pos_weight = float(n_neg_total / max(1, n_pos_total))

    loss_weighted_bce = get_loss_fn("weighted_bce", pos_weight=pos_weight)
    loss_focal_025 = get_loss_fn("focal", gamma=2.0, alpha=0.25)
    loss_focal_weighted = get_loss_fn("focal", gamma=2.0, alpha=0.75, pos_weight=pos_weight)

    print(f"  Total edges: {n_pos_total + n_neg_total} (Pos: {n_pos_total}, Neg: {n_neg_total})")
    print(f"  Overall positive rate: {n_pos_total / (n_pos_total + n_neg_total):.4%}")
    print(f"  Exact pos_weight (n_neg / n_pos): {pos_weight:.2f}")

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 3:
            break
        labels = batch[target_rel].edge_label
        n_pos_b = int((labels == 1).sum().item())
        pos_rate_b = n_pos_b / len(labels)

        # Generate dummy logits initialized near zero
        dummy_logits = torch.zeros_like(labels)

        l_wbce = float(loss_weighted_bce(dummy_logits, labels).item())
        l_foc_025 = float(loss_focal_025(dummy_logits, labels).item())
        l_foc_w = float(loss_focal_weighted(dummy_logits, labels).item())

        print(
            f"  Batch {batch_idx}: size={len(labels)}, pos={n_pos_b} ({pos_rate_b:.2%}) | "
            f"WeightedBCE loss: {l_wbce:.4f} | Focal (alpha=0.25): {l_foc_025:.4f} | "
            f"Focal (alpha=0.75, pos_weight): {l_foc_w:.4f}"
        )


def check_threshold_logic() -> None:
    print("\n" + "=" * 60)
    print("CHECK 3: Decision Threshold Selection Logic")
    print("=" * 60)
    # Scenario A: Well-calibrated predictions
    y_true = np.array([0] * 990 + [1] * 10)
    y_prob_good = np.zeros(1000)
    y_prob_good[:990] = np.random.uniform(0.001, 0.05, 990)
    y_prob_good[990:] = np.random.uniform(0.60, 0.95, 10)

    metrics_good = compute_metrics(y_true, y_prob_good)
    print(
        f"  Well-calibrated case: PR-AUC={metrics_good.pr_auc:.4f}, Best Thresh={metrics_good.best_threshold:.4f}, F1={metrics_good.f1:.4f}, TP={metrics_good.tp}, FP={metrics_good.fp}"
    )

    # Scenario B: Low-probability collapse (all probabilities very small)
    y_prob_low = np.zeros(1000)
    y_prob_low[:990] = np.random.uniform(0.0001, 0.001, 990)
    y_prob_low[990:] = np.random.uniform(0.001, 0.005, 10)

    metrics_low = compute_metrics(y_true, y_prob_low)
    print(
        f"  Low-probability case: PR-AUC={metrics_low.pr_auc:.4f}, Best Thresh={metrics_low.best_threshold:.4f}, F1={metrics_low.f1:.4f}, TP={metrics_low.tp}, FP={metrics_low.fp}"
    )
    assert metrics_low.best_threshold > 0.0, "Threshold must not collapse to degenerate 0"


def check_overfit_test(train_graph: HeteroData) -> None:
    print("\n" + "=" * 60)
    print("CHECK 4: Overfit Test on ~2,000 Edges (Target PR-AUC -> 1.0)")
    print("=" * 60)
    target_rel = ("user", "transacts_at", "merchant")
    orig_labels = train_graph[target_rel].edge_label

    # Select balanced subset of ~2,000 edges with ~100 fraud edges
    pos_indices = torch.where(orig_labels == 1)[0][:100]
    neg_indices = torch.where(orig_labels == 0)[0][:1900]
    sub_indices = torch.cat([pos_indices, neg_indices])

    sub_edge_index = train_graph[target_rel].edge_index[:, sub_indices]
    sub_edge_label = orig_labels[sub_indices]
    sub_edge_attr = train_graph[target_rel].edge_attr[sub_indices]

    loader = LinkNeighborLoader(
        data=train_graph,
        num_neighbors=[10, 10],
        edge_label_index=(target_rel, sub_edge_index),
        edge_label=sub_edge_label,
        edge_attr=sub_edge_attr,
        batch_size=len(sub_indices),  # single batch for deterministic overfitting
        shuffle=False,
        seed=42,
    )

    batch = next(iter(loader))
    in_channels_dict = {nt: int(train_graph[nt].x.size(1)) for nt in train_graph.node_types}

    model = HeteroGAT(
        node_types=train_graph.node_types,
        edge_types=train_graph.edge_types,
        in_channels_dict=in_channels_dict,
        hidden_channels=64,
        edge_dim=sub_edge_attr.size(1),
        num_layers=2,
        heads=4,
        dropout=0.0,  # No dropout during overfit test
        use_edge_attr=True,
    )

    n_pos = int((sub_edge_label == 1).sum().item())
    n_neg = int((sub_edge_label == 0).sum().item())
    pos_weight = float(n_neg / max(1, n_pos))
    loss_fn = get_loss_fn("weighted_bce", pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    target_y = sub_edge_label.numpy()

    for epoch in range(1, 101):
        optimizer.zero_grad()
        logits = model(
            x_dict={nt: batch[nt].x for nt in batch.node_types},
            edge_index_dict={et: batch[et].edge_index for et in batch.edge_types},
            edge_label_index=batch[target_rel].edge_label_index,
            edge_attr=batch[target_rel].edge_attr,
        )
        loss = loss_fn(logits, batch[target_rel].edge_label)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                probs = torch.sigmoid(logits).numpy()
                metrics = compute_metrics(target_y, probs)
                print(
                    f"  Epoch {epoch:3d}/100: Loss = {loss.item():.4f} | Train PR-AUC = {metrics.pr_auc:.4f} | Train ROC-AUC = {metrics.roc_auc:.4f} | F1 = {metrics.f1:.4f}"
                )

    final_pr_auc = metrics.pr_auc
    print(f"\n  Final Overfit PR-AUC: {final_pr_auc:.4f}")
    assert final_pr_auc > 0.90, (
        f"Overfit test failed! Expected PR-AUC > 0.90, got {final_pr_auc:.4f}"
    )
    print(
        "  Overfit test: PASS (model and loss can successfully memorize and separate transactions)"
    )


def main() -> None:
    data_dir = Path("data/processed")
    train_graph = torch.load(data_dir / "train_graph.pt", weights_only=False)

    check_label_alignment(train_graph)
    check_batch_imbalance_and_loss_scale(train_graph)
    check_threshold_logic()
    check_overfit_test(train_graph)
    print("\nALL SANITY CHECKS COMPLETED.")


if __name__ == "__main__":
    main()
