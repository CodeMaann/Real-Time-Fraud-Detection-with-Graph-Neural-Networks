"""Leakage battery for the causal HeteroGAT pipeline on REAL validation data.

a) Brute-force reference: for random validation targets, physically delete every edge with
   edge_time >= the target's timestamp, rebuild the 2-hop neighborhood and node features
   from what remains (own code, not the loader's sampler), score, and compare with the
   production loader (isolated targets, batched).
b) Future injection: add transactions at time T (inside a block of validation targets) on
   the entities of every earlier target; earlier targets' scores must be bit-identical at
   batch sizes 1, 2, 64, 2048. Later targets must change (control: the injection is live).
c) Timestamp ties: count targets sharing an exact edge_time with another transaction on the
   same user, merchant, device or IP; check no sampled edge has time >= its target's time
   and the target's own edges (all relations, both directions) are absent.
d) Alignment: every val/test target's user, merchant, card, device, IP, time and label
   match the raw transaction row.
e) Message-passing ablation: validation PR-AUC with neighbor messages removed.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score
from torch_geometric.data import HeteroData

from fraudgnn.config import load_baseline_config, load_gnn_config
from fraudgnn.graph.builder import extract_entity_mappings
from fraudgnn.graph.temporal_loader import CausalLinkNeighborLoader
from fraudgnn.models.hetero_gat import HeteroGAT
from fraudgnn.training.evaluate import compute_metrics

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_unified import TARGET_REL, split_targets

NODE_TYPE_ORDER = ["user", "card", "merchant", "device", "ip"]  # feature layout, own copy
SECONDS_PER_HOUR = 3600.0
FEATURE_DIM = 17


# ── a) Independent brute-force reference ───────────────────────────────────────────────


class BruteForceReference:
    """Rebuild one target's causal subgraph by deleting all edges at or after its time."""

    def __init__(self, graph: HeteroData, num_neighbors: list[int]) -> None:
        self.graph = graph
        self.num_neighbors = num_neighbors
        self.raw: dict[tuple[str, str, str], tuple[np.ndarray, ...]] = {}
        for et in graph.edge_types:
            ei = graph[et].edge_index.numpy()
            tm = graph[et].edge_time.numpy().astype(np.int64)
            attr = getattr(graph[et], "edge_attr", None)
            amt = (
                attr[:, 0].numpy().astype(np.float64)
                if attr is not None and attr.size(1) > 0
                else np.zeros(ei.shape[1], dtype=np.float64)
            )
            self.raw[et] = (ei[0], ei[1], tm, amt)

    def _past(self, t: int) -> dict[tuple[str, str, str], dict[str, np.ndarray]]:
        """Physically delete every edge with time >= t, then group the rest by source."""
        past = {}
        for et, (src, dst, tm, amt) in self.raw.items():
            keep = np.flatnonzero(tm < t)
            order = np.argsort(src[keep], kind="stable")
            kept = keep[order]
            past[et] = {
                "src": src[kept],
                "dst": dst[kept],
                "time": tm[kept],
                "amt": amt[kept],
                "eid": kept,
            }
        return past

    @staticmethod
    def _edges_of(p: dict[str, np.ndarray], node: int) -> slice:
        lo = int(np.searchsorted(p["src"], node, side="left"))
        hi = int(np.searchsorted(p["src"], node, side="right"))
        return slice(lo, hi)

    def build(
        self,
        user: int,
        merchant: int,
        device: int | None,
        ip: int | None,
        t: int,
        target_attr: torch.Tensor,
    ) -> HeteroData:
        """Single-target HeteroData equivalent to the loader's causal subgraph."""
        past = self._past(t)
        sampled: dict[str, set[int]] = {nt: set() for nt in self.graph.node_types}
        sampled["user"].add(user)
        sampled["merchant"].add(merchant)
        if device is not None:
            sampled["device"].add(device)
        if ip is not None:
            sampled["ip"].add(ip)
        frontier = {nt: sorted(s) for nt, s in sampled.items()}
        picked: dict[tuple[str, str, str], set[int]] = {et: set() for et in self.raw}

        for k in self.num_neighbors:
            found: dict[str, set[int]] = {nt: set() for nt in self.graph.node_types}
            for et, p in past.items():
                s_t, _, d_t = et
                for node in frontier[s_t]:
                    seg = self._edges_of(p, node)
                    if seg.stop == seg.start:
                        continue
                    # most recent k: order by (time, edge id), keep the last k
                    order = np.lexsort((p["eid"][seg], p["time"][seg]))[-k:]
                    picked[et].update(p["eid"][seg][order].tolist())
                    found[d_t].update(p["dst"][seg][order].tolist())
            for nt in self.graph.node_types:
                new = found[nt] - sampled[nt]
                sampled[nt] |= new
                frontier[nt] = sorted(new)

        batch = HeteroData()
        local: dict[str, dict[int, int]] = {}
        for nt in self.graph.node_types:
            nodes = sorted(sampled[nt])
            local[nt] = {n: i for i, n in enumerate(nodes)}
            batch[nt].n_id = torch.tensor(nodes, dtype=torch.long)
            batch[nt].x = torch.from_numpy(self._features(nt, nodes, t, past))
        for et in self.raw:
            s_t, _, d_t = et
            eids = sorted(picked[et])
            src, dst = self.raw[et][0], self.raw[et][1]
            batch[et].edge_index = torch.tensor(
                [[local[s_t][int(src[e])] for e in eids], [local[d_t][int(dst[e])] for e in eids]],
                dtype=torch.long,
            ).reshape(2, -1)
            batch[et].e_id = torch.tensor(eids, dtype=torch.long)
        store = batch[TARGET_REL]
        store.edge_label_index = torch.tensor(
            [[local["user"][user]], [local["merchant"][merchant]]]
        )
        store.edge_attr = target_attr.reshape(1, -1).to(torch.float32)
        if device is not None:
            store.device_index = torch.tensor([local["device"][device]])
        if ip is not None:
            store.ip_index = torch.tensor([local["ip"][ip]])
        return batch

    def _features(
        self,
        nt: str,
        nodes: list[int],
        t: int,
        past: dict[tuple[str, str, str], dict[str, np.ndarray]],
    ) -> np.ndarray:
        n = len(nodes)
        distinct = np.zeros((n, 5), dtype=np.float64)
        degree = np.zeros((n, 5), dtype=np.float64)
        amount = np.zeros(n, dtype=np.float64)
        earliest = np.full(n, np.inf)
        for et, p in past.items():
            s_t, _, d_t = et
            if s_t != nt:
                continue
            col = NODE_TYPE_ORDER.index(d_t)
            for i, node in enumerate(nodes):
                seg = self._edges_of(p, node)
                if seg.stop == seg.start:
                    continue
                distinct[i, col] += len(np.unique(p["dst"][seg]))
                degree[i, col] += seg.stop - seg.start
                amount[i] += float(p["amt"][seg].sum())
                earliest[i] = min(earliest[i], float(p["time"][seg].min()))
        age = np.where(
            np.isfinite(earliest), np.maximum(0.0, (t - earliest) / SECONDS_PER_HOUR), 0.0
        )
        feats = np.zeros((n, FEATURE_DIM), dtype=np.float32)
        feats[:, NODE_TYPE_ORDER.index(nt)] = 1.0
        feats[:, 5:10] = np.log1p(distinct)
        feats[:, 10:15] = np.log1p(degree)
        feats[:, 15] = np.log1p(np.maximum(0.0, amount))
        feats[:, 16] = np.log1p(age)
        return feats


# ── shared helpers ────────────────────────────────────────────────────────────────────


def build_model(cfg: dict[str, Any], graph: HeteroData, weights: bool = True) -> HeteroGAT:
    """HeteroGAT from its training config; checkpoint loaded unless ``weights=False``."""
    gnn_cfg = load_gnn_config(cfg["gnn_config"])
    model = HeteroGAT(
        node_types=graph.node_types,
        edge_types=graph.edge_types,
        in_channels_dict={nt: FEATURE_DIM for nt in graph.node_types},
        hidden_channels=gnn_cfg.model.hidden_dim,
        edge_dim=gnn_cfg.model.edge_dim,
        num_layers=gnn_cfg.model.num_layers,
        heads=gnn_cfg.model.heads,
        dropout=gnn_cfg.model.dropout,
        conv_type=cfg["conv_type"],
        use_edge_attr=gnn_cfg.model.use_edge_attr,
        use_device_ip=bool(cfg["use_device_ip"]),
    )
    if weights:
        model.load_state_dict(torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def score(
    model: HeteroGAT, batch: HeteroData, drop_edges: bool = False, zero_x: bool = False
) -> torch.Tensor:
    """Probabilities for every target in a batch (optionally without message passing)."""
    store = batch[TARGET_REL]
    edges = {
        et: (torch.empty((2, 0), dtype=torch.long) if drop_edges else batch[et].edge_index)
        for et in batch.edge_types
    }
    x = {nt: (torch.zeros_like(batch[nt].x) if zero_x else batch[nt].x) for nt in batch.node_types}
    logits = model(
        x_dict=x,
        edge_index_dict=edges,
        edge_label_index=store.edge_label_index,
        edge_attr=getattr(store, "edge_attr", None),
        device_index=getattr(store, "device_index", None),
        ip_index=getattr(store, "ip_index", None),
    )
    return torch.sigmoid(logits)


def make_loader(
    graph: HeteroData,
    tgt: dict[str, torch.Tensor],
    num_neighbors: list[int],
    batch_size: int,
    isolate: bool = True,
) -> CausalLinkNeighborLoader:
    """Production loader over the given targets."""
    return CausalLinkNeighborLoader(
        data=graph,
        num_neighbors=num_neighbors,
        edge_label_index=(TARGET_REL, tgt["edge_index"]),
        edge_label=tgt["edge_label"],
        edge_time=tgt["edge_time"],
        edge_attr=tgt["edge_attr"],
        device_index=tgt["device_index"],
        ip_index=tgt["ip_index"],
        batch_size=batch_size,
        shuffle=False,
        isolate_targets=isolate,
    )


def batched_scores(
    model: HeteroGAT, loader: CausalLinkNeighborLoader, positions: np.ndarray
) -> torch.Tensor:
    """Scores for target positions, built in loader-sized chunks in the given order."""
    out = []
    for i in range(0, len(positions), loader.batch_size):
        out.append(score(model, loader.build_batch(positions[i : i + loader.batch_size])))
    return torch.cat(out)


# ── checks ────────────────────────────────────────────────────────────────────────────


def check_brute_force(
    model: HeteroGAT,
    graph: HeteroData,
    tgt: dict[str, torch.Tensor],
    num_neighbors: list[int],
    positions: np.ndarray,
    production_batch: int,
) -> dict[str, Any]:
    """3a: reference subgraph vs loader (structure at batch size 1, scores batched)."""
    ref = BruteForceReference(graph, num_neighbors)
    loader1 = make_loader(graph, tgt, num_neighbors, 1)
    production = batched_scores(
        model, make_loader(graph, tgt, num_neighbors, production_batch), positions
    )
    diffs, struct_mismatch = [], 0
    col_diff = np.zeros(FEATURE_DIM, dtype=np.float64)
    for j, pos in enumerate(positions.tolist()):
        b_ref = ref.build(
            int(tgt["edge_index"][0, pos]),
            int(tgt["edge_index"][1, pos]),
            int(tgt["device_index"][pos]),
            int(tgt["ip_index"][pos]),
            int(tgt["edge_time"][pos]),
            tgt["edge_attr"][pos],
        )
        b_loader = loader1.build_batch(np.array([pos]))
        same = all(
            torch.equal(b_ref[nt].n_id, b_loader[nt].n_id) for nt in graph.node_types
        ) and all(torch.equal(b_ref[et].e_id, b_loader[et].e_id) for et in graph.edge_types)
        struct_mismatch += 0 if same else 1
        if same:
            for nt in graph.node_types:
                if b_ref[nt].x.numel():
                    d = (b_ref[nt].x - b_loader[nt].x).abs().amax(dim=0).double().numpy()
                    col_diff = np.maximum(col_diff, d)
        diffs.append(abs(float(score(model, b_ref)[0]) - float(production[j])))
    return {
        "targets": len(positions),
        "structure_mismatches": struct_mismatch,
        "max_node_feature_abs_diff": float(col_diff.max()),
        "max_node_feature_abs_diff_by_column": {int(c): float(v) for c, v in enumerate(col_diff)},
        "max_score_abs_diff": max(diffs),
        "mean_score_abs_diff": float(np.mean(diffs)),
    }


def inject_future(
    graph: HeteroData,
    tgt: dict[str, torch.Tensor],
    positions: np.ndarray,
    t_inject: int,
    scale: float,
) -> HeteroData:
    """Copy of the graph with one fake transaction at t_inject per given target's entities."""
    g = copy.deepcopy(graph)
    own = tgt["global_eid"][positions]
    ids = {
        "user": tgt["edge_index"][0, positions],
        "merchant": tgt["edge_index"][1, positions],
        "card": graph["user", "uses_card", "card"].edge_index[1][own],
        "device": tgt["device_index"][positions],
        "ip": tgt["ip_index"][positions],
    }
    n = len(positions)
    times = torch.full((n,), t_inject, dtype=torch.long)
    for et in graph.edge_types:
        s_t, _, d_t = et
        add = torch.stack([ids[s_t], ids[d_t]])
        g[et].edge_index = torch.cat([g[et].edge_index, add], dim=1)
        g[et].edge_time = torch.cat([g[et].edge_time, times])
        attr = getattr(g[et], "edge_attr", None)
        if attr is not None:
            fake = tgt["edge_attr"][positions].clone()
            fake[:, 0] = fake[:, 0].abs() * scale + scale
            g[et].edge_attr = torch.cat([attr, fake.to(attr.dtype)])
        for key in ("edge_label", "val_mask", "test_mask"):
            val = getattr(g[et], key, None)
            if val is not None:
                g[et][key] = torch.cat([val, torch.zeros(n, dtype=val.dtype)])
    return g


def check_injection(
    model: HeteroGAT,
    graph: HeteroData,
    tgt: dict[str, torch.Tensor],
    num_neighbors: list[int],
    block: int,
    batch_sizes: list[int],
    scale: float,
) -> dict[str, Any]:
    """3b: earlier targets bit-identical after injecting later transactions; later ones change."""
    positions = np.arange(block)
    times = tgt["edge_time"][:block]
    t_inject = int(times.median())
    early = (times < t_inject).numpy()
    injected = inject_future(graph, tgt, positions[early], t_inject, scale)
    result: dict[str, Any] = {
        "block": block,
        "t_inject": t_inject,
        "early_targets": int(early.sum()),
    }
    for bs in batch_sizes:
        clean = batched_scores(model, make_loader(graph, tgt, num_neighbors, bs), positions)
        dirty = batched_scores(model, make_loader(injected, tgt, num_neighbors, bs), positions)
        e, late = torch.from_numpy(early), torch.from_numpy(~early)
        result[f"bs{bs}"] = {
            "early_bit_identical": bool(torch.equal(clean[e], dirty[e])),
            "early_max_abs_diff": float((clean[e] - dirty[e]).abs().max()),
            "late_changed": int((clean[late] != dirty[late]).sum()),
            "late_targets": int(late.sum()),
        }
    shared_clean = batched_scores(
        model, make_loader(graph, tgt, num_neighbors, block, isolate=False), positions
    )
    shared_dirty = batched_scores(
        model, make_loader(injected, tgt, num_neighbors, block, isolate=False), positions
    )
    e = torch.from_numpy(early)
    result["legacy_shared_contrast"] = {
        "early_changed": int((shared_clean[e] != shared_dirty[e]).sum()),
        "early_max_abs_diff": float((shared_clean[e] - shared_dirty[e]).abs().max()),
    }
    return result


def check_ties_and_own_edges(
    graph: HeteroData,
    tgt: dict[str, torch.Tensor],
    df_split: pd.DataFrame,
    df_history: pd.DataFrame,
    num_neighbors: list[int],
    batch_size: int,
    limit: int | None = None,
) -> dict[str, Any]:
    """3c: tie counts and per-target absence of own/later edges in the sampled subgraph."""
    sec = pd.to_datetime(df_history["timestamp"]).astype("int64") // 10**9
    hist = df_history.assign(_sec=sec.to_numpy())
    target_sec = pd.to_datetime(df_split["timestamp"]).astype("int64") // 10**9
    ties: dict[str, int] = {}
    any_tie = np.zeros(len(df_split), dtype=bool)
    for col in ("user_id", "merchant_id", "device_id", "ip_address"):
        counts = hist.groupby([col, "_sec"]).size()
        key = pd.MultiIndex.from_arrays([df_split[col].to_numpy(), target_sec.to_numpy()])
        tied = counts.reindex(key).to_numpy() > 1
        ties[col] = int(tied.sum())
        any_tie |= tied
    n = len(df_split) if limit is None else min(limit, len(df_split))
    loader = make_loader(graph, tgt, num_neighbors, batch_size)
    later_or_equal, own_present, edges_checked = 0, 0, 0
    for start in range(0, n, batch_size):
        pos = np.arange(start, min(start + batch_size, n))
        b = loader.build_batch(pos)
        t_target = tgt["edge_time"][pos]
        own = tgt["global_eid"][pos]
        for et in graph.edge_types:
            if b[et].e_id.numel() == 0:
                continue
            slot = b[et].target_slot
            e_time = graph[et].edge_time[b[et].e_id]
            later_or_equal += int((e_time >= t_target[slot]).sum())
            own_present += int((b[et].e_id == own[slot]).sum())
            edges_checked += int(b[et].e_id.numel())
    return {
        "targets": len(df_split),
        "targets_with_exact_time_tie": int(any_tie.sum()),
        "ties_by_entity": ties,
        "subgraph_targets_checked": n,
        "sampled_edges_checked": edges_checked,
        "edges_at_or_after_target_time": later_or_equal,
        "own_edges_present": own_present,
    }


def check_alignment(
    graph: HeteroData, mask_name: str, df_split: pd.DataFrame, mappings: Any
) -> dict[str, int]:
    """3d: per-target bound entities, time and label match the raw rows (mismatch counts)."""
    t = split_targets(graph, mask_name)
    mask = t["mask"]
    card = graph["user", "uses_card", "card"].edge_index[1][mask].numpy()
    expected = {
        "user": [mappings.user_to_idx[u] for u in df_split["user_id"]],
        "merchant": [mappings.merchant_to_idx[m] for m in df_split["merchant_id"]],
        "card": [mappings.card_to_idx[c] for c in df_split["card_id"]],
        "device": [mappings.device_to_idx[d] for d in df_split["device_id"]],
        "ip": [mappings.ip_to_idx[i] for i in df_split["ip_address"]],
    }
    actual = {
        "user": t["edge_index"][0].numpy(),
        "merchant": t["edge_index"][1].numpy(),
        "card": card,
        "device": t["device_index"].numpy(),
        "ip": t["ip_index"].numpy(),
    }
    out = {k: int((np.asarray(expected[k]) != actual[k]).sum()) for k in expected}
    sec = (pd.to_datetime(df_split["timestamp"]).astype("int64") // 10**9).to_numpy()
    out["time"] = int((sec != t["edge_time"].numpy()).sum())
    out["label"] = int((df_split["is_fraud"].to_numpy() != t["edge_label"].numpy()).sum())
    out["rows"] = len(df_split)
    return out


def check_ablation(
    model: HeteroGAT,
    graph: HeteroData,
    tgt: dict[str, torch.Tensor],
    num_neighbors: list[int],
    batch_size: int,
) -> dict[str, Any]:
    """3e: validation PR-AUC with and without neighbor messages."""
    loader = make_loader(graph, tgt, num_neighbors, batch_size)
    full, no_mp, no_mp_no_x = [], [], []
    for start in range(0, loader.num_edges, batch_size):
        b = loader.build_batch(np.arange(start, min(start + batch_size, loader.num_edges)))
        full.append(score(model, b))
        no_mp.append(score(model, b, drop_edges=True))
        no_mp_no_x.append(score(model, b, drop_edges=True, zero_x=True))
    y = tgt["edge_label"].numpy()
    full_vec = torch.cat(full).numpy()
    ev = compute_metrics(y_true=y.astype(np.int64), y_pred_proba=full_vec, compute_ci=True)
    return {
        "full_model_compute_metrics": ev.pr_auc,
        "full_model_ci": [ev.pr_auc_ci_lower, ev.pr_auc_ci_upper],
        "targets_scored": len(full_vec),
        "full_model": float(average_precision_score(y, full_vec)),
        "no_message_passing": float(average_precision_score(y, torch.cat(no_mp).numpy())),
        "no_message_passing_no_node_features": float(
            average_precision_score(y, torch.cat(no_mp_no_x).numpy())
        ),
    }


def load_context(eval_config: str) -> dict[str, Any]:
    """Raw rows (file order), split rows, graphs and targets with their graph edge ids."""
    eval_cfg = yaml.safe_load(Path(eval_config).read_text(encoding="utf-8"))
    base = load_baseline_config(eval_cfg["baseline_config"])
    df = pd.read_parquet(base.data_path)
    ts = pd.to_datetime(df["timestamp"])
    if not ts.is_monotonic_increasing:
        raise ValueError("raw transactions are not in time order")
    t0, dur = ts.min(), ts.max() - ts.min()
    train_cut = t0 + dur * base.split.train_ratio
    val_cut = t0 + dur * (base.split.train_ratio + base.split.val_ratio)
    processed = Path(eval_cfg["processed_dir"])
    ctx: dict[str, Any] = {"df": df}
    for name, mask, sel in (
        ("val", "val_mask", (ts > train_cut) & (ts <= val_cut)),
        ("test", "test_mask", ts > val_cut),
    ):
        graph = torch.load(processed / f"{name}_graph.pt", weights_only=False)
        tgt = split_targets(graph, mask)
        tgt["global_eid"] = torch.from_numpy(np.flatnonzero(tgt["mask"].numpy()))
        ctx[name] = {
            "graph": graph,
            "tgt": tgt,
            "rows": df[sel].reset_index(drop=True),
            "history": df[ts <= (val_cut if name == "val" else ts.max())],
        }
    return ctx


SPLITS = {  # split -> (world, graph file, target mask)
    "val": ("original", "val_graph.pt", "val_mask"),
    "test": ("original", "test_graph.pt", "test_mask"),
    "holdout": ("holdout", "test_graph.pt", "test_mask"),
}


def load_split(cfg: dict[str, Any], split: str) -> dict[str, Any]:
    """Raw rows, graph and targets of one split (original val/test or the sealed holdout)."""
    world_name, graph_file, mask = SPLITS[split]
    world = cfg["worlds"][world_name]
    eval_cfg = yaml.safe_load(Path(cfg["eval_config"]).read_text(encoding="utf-8"))
    base = load_baseline_config(eval_cfg["baseline_config"])
    df = pd.read_parquet(world["raw"])
    ts = pd.to_datetime(df["timestamp"])
    if not ts.is_monotonic_increasing:
        raise ValueError(f"{world['raw']} is not in time order")
    t0, dur = ts.min(), ts.max() - ts.min()
    train_cut = t0 + dur * base.split.train_ratio
    val_cut = t0 + dur * (base.split.train_ratio + base.split.val_ratio)
    sel = (ts > train_cut) & (ts <= val_cut) if split == "val" else ts > val_cut
    graph = torch.load(Path(world["processed"]) / graph_file, weights_only=False)
    tgt = split_targets(graph, mask)
    tgt["global_eid"] = torch.from_numpy(np.flatnonzero(tgt["mask"].numpy()))
    return {
        "df": df,
        "mask_name": mask,
        "graph": graph,
        "tgt": tgt,
        "rows": df[sel].reset_index(drop=True),
        "history": df[ts <= (val_cut if split == "val" else ts.max())],
    }


def run_split(cfg: dict[str, Any], model_cfg: dict[str, Any], split: str) -> dict[str, Any]:
    """Checks a-d on one split for one checkpoint; e only off the holdout (it is a metric)."""
    gnn_cfg = load_gnn_config(model_cfg["gnn_config"])
    nn_ = gnn_cfg.training.num_neighbors[: gnn_cfg.model.num_layers]
    bs = gnn_cfg.training.batch_size
    data = load_split(cfg, split)
    model = build_model(model_cfg, data["graph"])
    rng = np.random.default_rng(cfg["seed"])
    report: dict[str, Any] = {"split": split, "model": model_cfg, "config": cfg}
    n_targets = data["tgt"]["edge_index"].size(1)
    positions = np.sort(rng.choice(n_targets, cfg["brute_force"]["targets"], replace=False))
    a = check_brute_force(model, data["graph"], data["tgt"], nn_, positions, bs)
    a["pass"] = (
        a["structure_mismatches"] == 0
        and a["max_score_abs_diff"] < cfg["brute_force"]["tolerance"]
    )
    inj = cfg["injection"]
    b = check_injection(
        model,
        data["graph"],
        data["tgt"],
        nn_,
        inj["block_size"],
        inj["batch_sizes"],
        inj["amount_scale"],
    )
    b["pass"] = all(b[f"bs{k}"]["early_bit_identical"] for k in inj["batch_sizes"]) and all(
        b[f"bs{k}"]["late_changed"] > 0 for k in inj["batch_sizes"]
    )
    c = check_ties_and_own_edges(
        data["graph"], data["tgt"], data["rows"], data["history"], nn_, bs
    )
    c["pass"] = c["edges_at_or_after_target_time"] == 0 and c["own_edges_present"] == 0
    d_split = check_alignment(
        data["graph"], data["mask_name"], data["rows"], extract_entity_mappings(data["df"])
    )
    d = {split: d_split, "pass": all(v == 0 for k, v in d_split.items() if k != "rows")}
    report.update(a_brute_force=a, b_injection=b, c_ties=c, d_alignment=d)
    if split == "holdout":
        report["e_ablation"] = "not run on the holdout (computes a metric; protocol section 3)"
    else:
        report["e_ablation"] = check_ablation(model, data["graph"], data["tgt"], nn_, bs)
    report["gate_pass"] = all(x["pass"] for x in (a, b, c, d))
    return report


def main(argv: list[str] | None = None) -> int:
    """Run all checks and write the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/leakage_battery.yaml")
    parser.add_argument(
        "--split",
        choices=list(SPLITS),
        default=None,
        help="Per-checkpoint mode: run checks on one split",
    )
    parser.add_argument("--key", default=None, help="Name used in the per-checkpoint output file")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gnn-config", default=None)
    parser.add_argument("--conv-type", default=None)
    parser.add_argument("--no-device-ip", action="store_true")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.split is not None:
        if not (args.key and args.checkpoint and args.gnn_config):
            raise SystemExit("--split needs --key, --checkpoint and --gnn-config")
        model_cfg = {
            "checkpoint": args.checkpoint,
            "gnn_config": args.gnn_config,
            "conv_type": args.conv_type or load_gnn_config(args.gnn_config).model.conv_type,
            "use_device_ip": not args.no_device_ip,
        }
        report = run_split(cfg, model_cfg, args.split)
        out = Path(cfg["per_checkpoint_out"].format(key=args.key, split=args.split))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("split", "gate_pass")}), f"-> {out}")
        return 0
    gnn_cfg = load_gnn_config(cfg["model"]["gnn_config"])
    nn_ = gnn_cfg.training.num_neighbors[: gnn_cfg.model.num_layers]
    bs = gnn_cfg.training.batch_size

    ctx = load_context(cfg["eval_config"])
    val = ctx["val"]
    model = build_model(cfg["model"], val["graph"])
    rng = np.random.default_rng(cfg["seed"])
    report: dict[str, Any] = {"config": cfg}

    t0 = time.perf_counter()
    positions = np.sort(
        rng.choice(val["tgt"]["edge_index"].size(1), cfg["brute_force"]["targets"], replace=False)
    )
    report["a_brute_force"] = check_brute_force(
        model, val["graph"], val["tgt"], nn_, positions, bs
    )
    report["a_brute_force"]["pass"] = (
        report["a_brute_force"]["structure_mismatches"] == 0
        and report["a_brute_force"]["max_score_abs_diff"] < cfg["brute_force"]["tolerance"]
    )
    print("a)", report["a_brute_force"], f"{time.perf_counter() - t0:.0f}s", flush=True)

    t0 = time.perf_counter()
    inj = cfg["injection"]
    report["b_injection"] = check_injection(
        model,
        val["graph"],
        val["tgt"],
        nn_,
        inj["block_size"],
        inj["batch_sizes"],
        inj["amount_scale"],
    )
    report["b_injection"]["pass"] = all(
        report["b_injection"][f"bs{b}"]["early_bit_identical"] for b in inj["batch_sizes"]
    ) and all(report["b_injection"][f"bs{b}"]["late_changed"] > 0 for b in inj["batch_sizes"])
    print("b)", report["b_injection"], f"{time.perf_counter() - t0:.0f}s", flush=True)

    t0 = time.perf_counter()
    report["c_ties"] = check_ties_and_own_edges(
        val["graph"], val["tgt"], val["rows"], val["history"], nn_, bs
    )
    report["c_ties"]["pass"] = (
        report["c_ties"]["edges_at_or_after_target_time"] == 0
        and report["c_ties"]["own_edges_present"] == 0
    )
    print("c)", report["c_ties"], f"{time.perf_counter() - t0:.0f}s", flush=True)

    mappings = extract_entity_mappings(ctx["df"])
    report["d_alignment"] = {
        name: check_alignment(ctx[name]["graph"], mask, ctx[name]["rows"], mappings)
        for name, mask in (("val", "val_mask"), ("test", "test_mask"))
    }
    report["d_alignment"]["pass"] = all(
        v == 0
        for split in ("val", "test")
        for k, v in report["d_alignment"][split].items()
        if k != "rows"
    )
    print("d)", report["d_alignment"], flush=True)

    t0 = time.perf_counter()
    report["e_ablation"] = check_ablation(model, val["graph"], val["tgt"], nn_, bs)
    print("e)", report["e_ablation"], f"{time.perf_counter() - t0:.0f}s", flush=True)

    report["gate_pass"] = all(
        report[k]["pass"] for k in ("a_brute_force", "b_injection", "c_ties", "d_alignment")
    )
    Path(cfg["results_out"]).write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print("GATE PASS" if report["gate_pass"] else "GATE FAIL", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
