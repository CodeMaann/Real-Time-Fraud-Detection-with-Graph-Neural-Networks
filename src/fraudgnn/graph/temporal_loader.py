"""Causal Temporal Link and Neighbor Loaders for Graph Neural Networks.

Guarantees 100% strict temporal causality (zero lookahead leakage within windows):
1. When sampling multi-hop neighborhoods for a target transaction edge occurring at
   timestamp t_target, only edges with edge_time < t_target are eligible to be
   sampled. No future edge within the same split window can ever enter the
   computational tree of a transaction.
2. Node features are dynamically computed on-the-fly strictly from history before
   each target transaction cutoff, eliminating window-aggregation leakage (degree,
   volume, age) while preserving essential inductive structural signals.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData

STANDARD_NODE_TYPES = ["user", "card", "merchant", "device", "ip"]


class FastTemporalAdjacency:
    """Pre-indexed temporal CSR adjacency structure for rapid causal neighbor querying.

    Edges are sorted by (source, time). Queries are vectorized: a combined
    ``source * span + time`` key turns "edges of u strictly before t" into one
    ``np.searchsorted`` over all queried nodes.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_time: torch.Tensor,
        num_src_nodes: int,
        edge_amount: torch.Tensor | None = None,
    ) -> None:
        self.num_src_nodes = num_src_nodes
        self.keys: np.ndarray | None = None
        self.t_min = 0
        self.span = 1
        if edge_index.size(1) == 0:
            self.ptrs = np.zeros(num_src_nodes + 1, dtype=np.int64)
            self.dst = np.empty(0, dtype=np.int64)
            self.times = np.empty(0, dtype=np.int64)
            self.edge_ids = np.empty(0, dtype=np.int64)
            self.amounts = np.empty(0, dtype=np.float64)
            self.node_cum_amounts = np.empty(0, dtype=np.float64)
            self.cum_distinct = np.zeros(1, dtype=np.int64)
            return

        src_arr = edge_index[0].cpu().numpy()
        dst_arr = edge_index[1].cpu().numpy()
        time_arr = edge_time.cpu().numpy().astype(np.int64)
        edge_ids = np.arange(len(src_arr), dtype=np.int64)

        if edge_amount is not None:
            amt_arr = edge_amount.cpu().numpy().astype(np.float64)
        else:
            amt_arr = np.zeros(len(src_arr), dtype=np.float64)

        # Lexsort: primary key src_arr, secondary key time_arr (ascending order)
        sort_idx = np.lexsort((time_arr, src_arr))
        src_sorted = src_arr[sort_idx]
        self.dst = dst_arr[sort_idx]
        self.times = time_arr[sort_idx]
        self.edge_ids = edge_ids[sort_idx]
        self.amounts = amt_arr[sort_idx]

        counts = np.bincount(src_sorted, minlength=num_src_nodes)
        self.ptrs = np.zeros(num_src_nodes + 1, dtype=np.int64)
        self.ptrs[1:] = np.cumsum(counts)

        # Running amount sum restarted at every source node (float64). A single cumulative
        # sum over all nodes' edges would make each node's prefix depend, through rounding,
        # on other nodes' edges, including later ones (leakage battery check 3b).
        self.node_cum_amounts = (
            pd.Series(self.amounts).groupby(src_sorted).cumsum().to_numpy(dtype=np.float64)
        )

        # Prefix count of distinct destinations per source: an edge counts if it is the
        # first (earliest) edge from its source to that destination.
        pair = src_sorted.astype(np.int64) * (int(self.dst.max()) + 1) + self.dst
        _, first_pos = np.unique(pair, return_index=True)
        first_flag = np.zeros(len(pair), dtype=np.int64)
        first_flag[first_pos] = 1
        self.cum_distinct = np.zeros(len(pair) + 1, dtype=np.int64)
        self.cum_distinct[1:] = np.cumsum(first_flag)

        self.t_min = int(self.times.min())
        self.span = int(self.times.max()) - self.t_min + 2
        if (num_src_nodes + 1) * self.span < 2**62:
            self.keys = src_sorted.astype(np.int64) * self.span + (self.times - self.t_min)

    def _prefix_bounds(
        self, nodes: np.ndarray, cutoffs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (start, end) so that sorted edges[start:end] of each node are before its cutoff.

        Nodes outside the source range get an empty range.
        """
        valid = nodes < self.num_src_nodes
        safe = np.where(valid, nodes, 0)
        start = self.ptrs[safe]
        stop = self.ptrs[safe + 1]
        if self.keys is not None:
            rel = np.clip(cutoffs - self.t_min, 0, self.span - 1)
            end = np.searchsorted(self.keys, safe * self.span + rel, side="left")
            end = np.clip(end, start, stop)
        else:
            end = np.array(
                [
                    s + int(np.searchsorted(self.times[s:e], t, side="left"))
                    for s, e, t in zip(start, stop, cutoffs, strict=True)
                ],
                dtype=np.int64,
            )
        return start, np.where(valid, end, start)

    def sample_causal_neighbors(
        self,
        src_nodes: np.ndarray | list[int],
        time_cutoffs: np.ndarray | list[int],
        max_neighbors: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return up to ``max_neighbors`` most recent edges strictly BEFORE each cutoff.

        Selection is deterministic (most recent first); ``rng`` is accepted for API
        compatibility and not used.

        Args:
            src_nodes: Array of source node indices.
            time_cutoffs: Array of timestamps. For src_nodes[k], only edges with
                edge_time < time_cutoffs[k] are valid.
            max_neighbors: Maximum neighbors to sample per source node.
            rng: Unused.

        Returns:
            Tuple of (sampled_dst_nodes, sampled_original_edge_ids, sampled_parent_cutoffs),
            grouped by source node in input order, each group in ascending time.
        """
        cuts = np.asarray(time_cutoffs, dtype=np.int64)
        dst, eids, row = self.sample_causal_neighbors_indexed(src_nodes, cuts, max_neighbors)
        return dst, eids, cuts[row]

    def sample_causal_neighbors_indexed(
        self,
        src_nodes: np.ndarray | list[int],
        time_cutoffs: np.ndarray | list[int],
        max_neighbors: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Like ``sample_causal_neighbors`` but return the query row of each sampled edge.

        Returns:
            Tuple of (sampled_dst_nodes, sampled_original_edge_ids, query_row_index).
        """
        empty = np.empty(0, dtype=np.int64)
        if len(self.dst) == 0 or len(src_nodes) == 0:
            return empty, empty, empty

        cuts = np.asarray(time_cutoffs, dtype=np.int64)
        start, end = self._prefix_bounds(np.asarray(src_nodes, dtype=np.int64), cuts)
        take = np.minimum(end - start, max_neighbors)
        total = int(take.sum())
        if total == 0:
            return empty, empty, empty
        # Positions end[k]-take[k] .. end[k]-1 for every node k, concatenated in order.
        group_start = np.repeat(np.cumsum(take) - take, take)
        pos = np.repeat(end - take, take) + (np.arange(total) - group_start)
        row = np.repeat(np.arange(len(take), dtype=np.int64), take)
        return self.dst[pos], self.edge_ids[pos], row

    def get_causal_node_stats(
        self,
        nodes: np.ndarray | list[int],
        time_cutoffs: np.ndarray | list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute causal (degree, distinct_dst_count, amount_sum, first_time) strictly BEFORE time_cutoffs.

        Args:
            nodes: Sequence of node IDs.
            time_cutoffs: Sequence of timestamps corresponding to each node.

        Returns:
            Tuple of (degrees, distinct_dst_counts, amount_sums, first_times) as numpy arrays.
        """
        n = len(nodes)
        out_first_time = np.full(n, np.inf, dtype=np.float64)
        if len(self.dst) == 0 or n == 0:
            return (
                np.zeros(n, dtype=np.float32),
                np.zeros(n, dtype=np.float32),
                np.zeros(n, dtype=np.float32),
                out_first_time,
            )

        start, end = self._prefix_bounds(
            np.asarray(nodes, dtype=np.int64), np.asarray(time_cutoffs, dtype=np.int64)
        )
        deg = (end - start).astype(np.float32)
        distinct = (self.cum_distinct[end] - self.cum_distinct[start]).astype(np.float32)
        has = end > start
        amt = np.zeros(n, dtype=np.float32)
        amt[has] = self.node_cum_amounts[end[has] - 1].astype(np.float32)
        out_first_time[has] = self.times[start[has]].astype(np.float64)
        return deg, distinct, amt, out_first_time


class CausalLinkNeighborLoader:
    """Strictly causal multi-hop neighborhood sampler for HeteroData graphs.

    Enforces that message passing for a target transaction edge at time t only
    uses relational edges with edge_time < t, guaranteeing zero within-window
    lookahead bias.

    Args:
        data: HeteroData graph object.
        num_neighbors: Sample limits per hop (e.g. [10, 10]).
        edge_label_index: Target relation and edge indices.
        edge_label: Ground truth labels.
        edge_time: Target edge timestamps.
        edge_attr: Optional edge attributes.
        device_index: Optional device node per target edge (decoder binding). Never inferred.
        ip_index: Optional IP node per target edge (decoder binding). Never inferred.
        batch_size: Target edges per batch.
        shuffle: Whether to shuffle target edges.
        seed: Seed for the shuffle permutation. Neighbor selection is deterministic
            (most recent ``k`` edges before the cutoff), so with ``shuffle=False`` the
            seed has no effect on the output.
        compute_causal_node_features: Whether to dynamically compute 17-dim causal node features.
        isolate_targets: If True (default), each target gets its own subgraph and node
            features with its own timestamp as cutoff, so a batched score equals scoring the
            target alone. If False, all targets in a batch share one induced subgraph; edges
            sampled for a later target can then reach an earlier target's embedding (legacy
            behaviour, kept only to measure that leak).
    """

    def __init__(
        self,
        data: HeteroData,
        num_neighbors: list[int] | dict[Any, list[int]],
        edge_label_index: tuple[tuple[str, str, str], torch.Tensor] | torch.Tensor,
        edge_label: torch.Tensor,
        edge_time: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        device_index: torch.Tensor | None = None,
        ip_index: torch.Tensor | None = None,
        batch_size: int = 512,
        shuffle: bool = True,
        seed: int | None = 42,
        compute_causal_node_features: bool = True,
        isolate_targets: bool = True,
    ) -> None:
        self.data = data
        self.isolate_targets = isolate_targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.compute_causal_node_features = compute_causal_node_features

        if isinstance(num_neighbors, list):
            self.num_neighbors = num_neighbors
        elif isinstance(num_neighbors, dict):
            first_val = next(iter(num_neighbors.values()))
            self.num_neighbors = first_val if isinstance(first_val, list) else [10, 10]
        else:
            self.num_neighbors = [10, 10]

        if isinstance(edge_label_index, tuple):
            self.target_edge_type = edge_label_index[0]
            self.target_edge_index = edge_label_index[1]
        else:
            self.target_edge_type = ("user", "transacts_at", "merchant")
            self.target_edge_index = edge_label_index

        self.edge_label = edge_label
        self.edge_time = edge_time
        self.edge_attr = edge_attr
        self.num_edges = self.target_edge_index.size(1)
        self._target_src = self.target_edge_index[0].cpu().numpy()
        self._target_dst = self.target_edge_index[1].cpu().numpy()
        self._labels = edge_label.cpu().numpy()
        self._times = edge_time.cpu().numpy()
        self._attr = edge_attr.cpu().numpy() if edge_attr is not None else None

        # Target transaction device and ip indices. Only explicit arguments are used: an
        # implicit fallback (graph-level attribute or a prefix slice of the uses_device /
        # uses_ip edges) silently seeds devices/IPs for models trained without binding, and
        # the prefix slice is misaligned whenever the targets are a masked subset (val/test).
        if device_index is not None and device_index.numel() != self.num_edges:
            raise ValueError("device_index must have one entry per target edge")
        if ip_index is not None and ip_index.numel() != self.num_edges:
            raise ValueError("ip_index must have one entry per target edge")
        self.device_index: np.ndarray | None = (
            device_index.cpu().numpy() if device_index is not None else None
        )
        self.ip_index: np.ndarray | None = ip_index.cpu().numpy() if ip_index is not None else None

        # Build temporal CSR index for all relational edges
        self.adj_dict: dict[tuple[str, str, str], FastTemporalAdjacency] = {}
        for edge_type in data.edge_types:
            src_t, _, _ = edge_type
            n_src = data[src_t].num_nodes
            e_idx = data[edge_type].edge_index
            e_time = getattr(data[edge_type], "edge_time", None)
            if e_time is None:
                # Fallback to zero timestamp if not provided
                e_time = torch.zeros(e_idx.size(1), dtype=torch.long)

            e_amt = None
            if hasattr(data[edge_type], "edge_attr") and data[edge_type].edge_attr is not None:
                attr = data[edge_type].edge_attr
                if attr.size(1) > 0:
                    e_amt = attr[:, 0]

            self.adj_dict[edge_type] = FastTemporalAdjacency(
                edge_index=e_idx,
                edge_time=e_time,
                num_src_nodes=n_src,
                edge_amount=e_amt,
            )

    def __len__(self) -> int:
        return math.ceil(self.num_edges / self.batch_size)

    def __iter__(self) -> Iterator[HeteroData]:
        perm = self.rng.permutation(self.num_edges) if self.shuffle else np.arange(self.num_edges)
        for start_idx in range(0, self.num_edges, self.batch_size):
            yield self.build_batch(perm[start_idx : start_idx + self.batch_size])

    def build_batch(self, batch_indices: np.ndarray) -> HeteroData:
        """Build the mini-batch for the given target positions.

        With ``isolate_targets`` every target gets its own subgraph and node features,
        sampled with its own timestamp as cutoff, and the per-target subgraphs are stacked
        block-diagonally. Each target's output is then identical to scoring it alone.
        Without it all targets share one induced subgraph (legacy behaviour, leaks edges
        sampled for later targets into earlier targets' embeddings).

        Args:
            batch_indices: Positions into the loader's target edges.

        Returns:
            HeteroData mini-batch.
        """
        batch_indices = np.asarray(batch_indices, dtype=np.int64)
        if self.isolate_targets:
            return self._build_isolated(batch_indices)
        return self._build_shared(batch_indices)

    def _node_features(self, nt: str, nodes: np.ndarray, cuts: np.ndarray) -> np.ndarray:
        """Strictly causal 17-dim node features, each node using its own cutoff.

        [0:5]   One-hot node type (user, card, merchant, device, ip)
        [5:10]  log1p(distinct neighbors by target type: user, card, merchant, device, ip)
        [10:15] log1p(total edges by target type: user, card, merchant, device, ip)
        [15]    log1p(total_amount)
        [16]    log1p(age_hours)
        """
        type_idx = STANDARD_NODE_TYPES.index(nt) if nt in STANDARD_NODE_TYPES else 0
        n_nodes = len(nodes)
        distinct_by_target = np.zeros((n_nodes, 5), dtype=np.float32)
        deg_by_target = np.zeros((n_nodes, 5), dtype=np.float32)
        total_amt = np.zeros(n_nodes, dtype=np.float32)
        earliest_time = np.full(n_nodes, np.inf, dtype=np.float64)

        for et, adj in self.adj_dict.items():
            s_t, _, d_t = et
            if s_t == nt:
                d, dist, a, ft = adj.get_causal_node_stats(nodes, cuts)
                if d_t in STANDARD_NODE_TYPES:
                    d_idx = STANDARD_NODE_TYPES.index(d_t)
                    distinct_by_target[:, d_idx] += dist
                    deg_by_target[:, d_idx] += d
                total_amt += a
                mask = d > 0
                earliest_time[mask] = np.minimum(earliest_time[mask], ft[mask])

        ages_hours = np.zeros(n_nodes, dtype=np.float32)
        has_prior = earliest_time < np.inf
        ages_hours[has_prior] = np.maximum(
            0.0,
            (cuts[has_prior] - earliest_time[has_prior]) / 3600.0,
        )

        feats = np.zeros((n_nodes, 17), dtype=np.float32)
        if type_idx < 5:
            feats[:, type_idx] = 1.0
        feats[:, 5:10] = np.log1p(distinct_by_target)
        feats[:, 10:15] = np.log1p(deg_by_target)
        feats[:, 15] = np.log1p(np.maximum(0.0, total_amt))
        feats[:, 16] = np.log1p(ages_hours)
        return feats

    def _build_isolated(self, batch_indices: np.ndarray) -> HeteroData:
        """Build one independent subgraph per target and stack them block-diagonally.

        Every sampled node and edge is keyed by (target slot, global id), so targets never
        share nodes. All nodes in a target's tree use that target's timestamp as cutoff,
        which reproduces the subgraph the target gets when scored alone (batch size 1).
        """
        src_t, _, dst_t = self.target_edge_type
        node_types = self.data.node_types
        n_nodes = {nt: int(self.data[nt].num_nodes) for nt in node_types}
        slots = np.arange(len(batch_indices), dtype=np.int64)
        b_times = self._times[batch_indices].astype(np.int64)
        b_src = self._target_src[batch_indices].astype(np.int64)
        b_dst = self._target_dst[batch_indices].astype(np.int64)
        b_dev = (
            self.device_index[batch_indices].astype(np.int64)
            if self.device_index is not None
            else None
        )
        b_ip = self.ip_index[batch_indices].astype(np.int64) if self.ip_index is not None else None

        empty = np.empty(0, dtype=np.int64)
        seeds: dict[str, list[np.ndarray]] = {nt: [] for nt in node_types}
        seeds[src_t].append(slots * n_nodes[src_t] + b_src)
        seeds[dst_t].append(slots * n_nodes[dst_t] + b_dst)
        if b_dev is not None and "device" in n_nodes:
            seeds["device"].append(slots * n_nodes["device"] + b_dev)
        if b_ip is not None and "ip" in n_nodes:
            seeds["ip"].append(slots * n_nodes["ip"] + b_ip)
        sampled = {
            nt: np.unique(np.concatenate(seeds[nt])) if seeds[nt] else empty for nt in node_types
        }
        frontier = dict(sampled)
        edge_keys: dict[tuple[str, str, str], list[np.ndarray]] = {
            et: [] for et in self.data.edge_types
        }

        # 1. Multi-hop expansion for all targets at once; every node uses its target's time
        for k_limit in self.num_neighbors:
            found: dict[str, list[np.ndarray]] = {nt: [] for nt in node_types}
            for et, adj in self.adj_dict.items():
                s_t, _, d_t = et
                keys = frontier[s_t]
                if len(keys) == 0:
                    continue
                f_slot = keys // n_nodes[s_t]
                dst, eids, row = adj.sample_causal_neighbors_indexed(
                    keys % n_nodes[s_t], b_times[f_slot], k_limit
                )
                if len(eids) == 0:
                    continue
                e_slot = f_slot[row]
                edge_keys[et].append(e_slot * self.data[et].edge_index.size(1) + eids)
                found[d_t].append(e_slot * n_nodes[d_t] + dst)
            for nt in node_types:
                cand = np.unique(np.concatenate(found[nt])) if found[nt] else empty
                new = np.setdiff1d(cand, sampled[nt], assume_unique=True)
                sampled[nt] = np.union1d(sampled[nt], new)
                frontier[nt] = new

        # 2. Nodes, ordered by (target slot, global id), with causal node features
        batch = HeteroData()
        for nt in node_types:
            keys = sampled[nt]
            node_ids = keys % n_nodes[nt] if len(keys) else empty
            batch[nt].num_nodes = len(keys)
            batch[nt].n_id = torch.from_numpy(node_ids)
            if self.compute_causal_node_features:
                cuts = b_times[keys // n_nodes[nt]] if len(keys) else empty
                batch[nt].x = torch.from_numpy(self._node_features(nt, node_ids, cuts))
            elif hasattr(self.data[nt], "x") and self.data[nt].x is not None:
                batch[nt].x = self.data[nt].x[node_ids]

        # 3. Relational edges, each inside its own target's subgraph
        for et in self.data.edge_types:
            s_t, _, d_t = et
            if not edge_keys[et]:
                batch[et].edge_index = torch.empty((2, 0), dtype=torch.long)
                continue
            n_e = self.data[et].edge_index.size(1)
            keys = np.unique(np.concatenate(edge_keys[et]))
            e_slot, eids = keys // n_e, keys % n_e
            orig = self.data[et].edge_index
            g_src = orig[0, eids].cpu().numpy()
            g_dst = orig[1, eids].cpu().numpy()
            loc_src = np.searchsorted(sampled[s_t], e_slot * n_nodes[s_t] + g_src)
            loc_dst = np.searchsorted(sampled[d_t], e_slot * n_nodes[d_t] + g_dst)
            batch[et].edge_index = torch.from_numpy(np.stack([loc_src, loc_dst]).astype(np.int64))
            if hasattr(self.data[et], "edge_time"):
                batch[et].edge_time = self.data[et].edge_time[eids]
            # Provenance for audits (not model inputs): original edge id and owning target.
            batch[et].e_id = torch.from_numpy(eids.astype(np.int64))
            batch[et].target_slot = torch.from_numpy(e_slot.astype(np.int64))

        # 4. Target edges in local coordinates
        def local(nt: str, ids: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(
                np.searchsorted(sampled[nt], slots * n_nodes[nt] + ids).astype(np.int64)
            )

        tgt = batch[self.target_edge_type]
        tgt.edge_label_index = torch.stack([local(src_t, b_src), local(dst_t, b_dst)])
        tgt.edge_label = torch.from_numpy(self._labels[batch_indices]).to(dtype=torch.float32)
        tgt.edge_time = torch.from_numpy(self._times[batch_indices])
        if self._attr is not None:
            tgt.edge_attr = torch.from_numpy(self._attr[batch_indices]).to(dtype=torch.float32)
        if b_dev is not None and "device" in n_nodes:
            tgt.device_index = local("device", b_dev)
        if b_ip is not None and "ip" in n_nodes:
            tgt.ip_index = local("ip", b_ip)
        tgt.batch_size = len(batch_indices)
        return batch

    def _build_shared(self, batch_indices: np.ndarray) -> HeteroData:
        """Build one induced subgraph shared by all targets in ``batch_indices``."""
        src_t, _, dst_t = self.target_edge_type
        b_src = self._target_src[batch_indices]
        b_dst = self._target_dst[batch_indices]
        b_labels = self._labels[batch_indices]
        b_times = self._times[batch_indices]
        b_attr = self._attr[batch_indices] if self._attr is not None else None
        b_dev = self.device_index[batch_indices] if self.device_index is not None else None
        b_ip = self.ip_index[batch_indices] if self.ip_index is not None else None

        # 1. Multi-hop causal neighborhood expansion
        sampled_nodes: dict[str, set[int]] = {nt: set() for nt in self.data.node_types}
        sampled_nodes[src_t].update(b_src)
        sampled_nodes[dst_t].update(b_dst)
        if b_dev is not None and "device" in sampled_nodes:
            sampled_nodes["device"].update(b_dev)
        if b_ip is not None and "ip" in sampled_nodes:
            sampled_nodes["ip"].update(b_ip)

        sampled_eids: dict[tuple[str, str, str], list[np.ndarray]] = {
            et: [] for et in self.data.edge_types
        }

        # Map active nodes to their respective temporal cutoff limits
        frontier_nodes: dict[str, list[int]] = {nt: [] for nt in self.data.node_types}
        frontier_times: dict[str, list[int]] = {nt: [] for nt in self.data.node_types}
        node_cutoffs: dict[str, dict[int, int]] = {nt: {} for nt in self.data.node_types}

        for u_id, t_cut in zip(b_src, b_times, strict=False):
            if u_id not in node_cutoffs[src_t] or t_cut < node_cutoffs[src_t][u_id]:
                node_cutoffs[src_t][int(u_id)] = int(t_cut)
        for v_id, t_cut in zip(b_dst, b_times, strict=False):
            if v_id not in node_cutoffs[dst_t] or t_cut < node_cutoffs[dst_t][v_id]:
                node_cutoffs[dst_t][int(v_id)] = int(t_cut)
        if b_dev is not None and "device" in node_cutoffs:
            for d_id, t_cut in zip(b_dev, b_times, strict=False):
                if d_id not in node_cutoffs["device"] or t_cut < node_cutoffs["device"][d_id]:
                    node_cutoffs["device"][int(d_id)] = int(t_cut)
        if b_ip is not None and "ip" in node_cutoffs:
            for ip_id, t_cut in zip(b_ip, b_times, strict=False):
                if ip_id not in node_cutoffs["ip"] or t_cut < node_cutoffs["ip"][ip_id]:
                    node_cutoffs["ip"][int(ip_id)] = int(t_cut)

        frontier_nodes[src_t].extend(b_src.tolist())
        frontier_times[src_t].extend(b_times.tolist())
        frontier_nodes[dst_t].extend(b_dst.tolist())
        frontier_times[dst_t].extend(b_times.tolist())
        if b_dev is not None and "device" in frontier_nodes:
            frontier_nodes["device"].extend(b_dev.tolist())
            frontier_times["device"].extend(b_times.tolist())
        if b_ip is not None and "ip" in frontier_nodes:
            frontier_nodes["ip"].extend(b_ip.tolist())
            frontier_times["ip"].extend(b_times.tolist())

        for hop_k in range(len(self.num_neighbors)):
            k_limit = self.num_neighbors[hop_k]
            next_frontier_nodes: dict[str, list[int]] = {nt: [] for nt in self.data.node_types}
            next_frontier_times: dict[str, list[int]] = {nt: [] for nt in self.data.node_types}

            for edge_type, adj in self.adj_dict.items():
                s_t, _, d_t = edge_type
                seeds = frontier_nodes[s_t]
                cutoffs = frontier_times[s_t]
                if not seeds:
                    continue

                s_dst, s_eids, s_cuts = adj.sample_causal_neighbors(
                    src_nodes=seeds,
                    time_cutoffs=cutoffs,
                    max_neighbors=k_limit,
                    rng=self.rng,
                )
                if len(s_eids) > 0:
                    sampled_eids[edge_type].append(s_eids)
                    for d_node, d_cut in zip(s_dst, s_cuts, strict=False):
                        d_int = int(d_node)
                        d_c_int = int(d_cut)
                        if d_int not in node_cutoffs[d_t] or d_c_int < node_cutoffs[d_t][d_int]:
                            node_cutoffs[d_t][d_int] = d_c_int

                        if d_int not in sampled_nodes[d_t]:
                            sampled_nodes[d_t].add(d_int)
                            next_frontier_nodes[d_t].append(d_int)
                            next_frontier_times[d_t].append(d_c_int)

            frontier_nodes = next_frontier_nodes
            frontier_times = next_frontier_times

        # 2. Build Mini-Batch HeteroData
        batch = HeteroData()
        global_to_local: dict[str, dict[int, int]] = {}

        for nt in self.data.node_types:
            nodes_sorted = np.array(sorted(sampled_nodes[nt]), dtype=np.int64)
            global_to_local[nt] = {gid: lid for lid, gid in enumerate(nodes_sorted)}
            batch[nt].num_nodes = len(nodes_sorted)
            batch[nt].n_id = torch.from_numpy(nodes_sorted)

            if self.compute_causal_node_features:
                def_cut = int(b_times.min()) if len(b_times) > 0 else 0
                cuts_arr = np.array(
                    [node_cutoffs[nt].get(int(gid), def_cut) for gid in nodes_sorted],
                    dtype=np.int64,
                )
                batch[nt].x = torch.from_numpy(self._node_features(nt, nodes_sorted, cuts_arr))
            elif hasattr(self.data[nt], "x") and self.data[nt].x is not None:
                batch[nt].x = self.data[nt].x[nodes_sorted]

        # 3. Induce causal relational edges
        for edge_type in self.data.edge_types:
            s_t, _, d_t = edge_type
            e_chunks = sampled_eids[edge_type]
            if not e_chunks:
                batch[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
                continue

            all_eids = np.unique(np.concatenate(e_chunks))
            orig_edge_index = self.data[edge_type].edge_index
            sub_e_src = orig_edge_index[0, all_eids].cpu().numpy()
            sub_e_dst = orig_edge_index[1, all_eids].cpu().numpy()

            g_to_l_s = global_to_local[s_t]
            g_to_l_d = global_to_local[d_t]

            loc_src = np.array([g_to_l_s[u] for u in sub_e_src], dtype=np.int64)
            loc_dst = np.array([g_to_l_d[v] for v in sub_e_dst], dtype=np.int64)

            batch[edge_type].edge_index = torch.stack(
                [torch.from_numpy(loc_src), torch.from_numpy(loc_dst)], dim=0
            )

            if hasattr(self.data[edge_type], "edge_time"):
                batch[edge_type].edge_time = self.data[edge_type].edge_time[all_eids]

        # 4. Map target edges to local contiguous node coordinates
        local_target_src = np.array([global_to_local[src_t][u] for u in b_src], dtype=np.int64)
        local_target_dst = np.array([global_to_local[dst_t][v] for v in b_dst], dtype=np.int64)

        target_edge_label_index = torch.stack(
            [torch.from_numpy(local_target_src), torch.from_numpy(local_target_dst)],
            dim=0,
        )
        batch[self.target_edge_type].edge_label_index = target_edge_label_index
        batch[self.target_edge_type].edge_label = torch.from_numpy(b_labels).to(
            dtype=torch.float32
        )
        batch[self.target_edge_type].edge_time = torch.from_numpy(b_times)
        if b_attr is not None:
            batch[self.target_edge_type].edge_attr = torch.from_numpy(b_attr).to(
                dtype=torch.float32
            )
        if b_dev is not None and "device" in global_to_local:
            local_dev = np.array([global_to_local["device"][d] for d in b_dev], dtype=np.int64)
            batch[self.target_edge_type].device_index = torch.from_numpy(local_dev)
        if b_ip is not None and "ip" in global_to_local:
            local_ip = np.array([global_to_local["ip"][i] for i in b_ip], dtype=np.int64)
            batch[self.target_edge_type].ip_index = torch.from_numpy(local_ip)
        batch[self.target_edge_type].batch_size = len(batch_indices)

        return batch


class CausalHomogeneousLinkLoader:
    """Strictly causal homogeneous neighborhood sampler for GraphSAGE.

    Sampled neighbors satisfy edge_time < t for the target that sampled them, with dynamic
    causal node features.

    Args:
        data: Homogeneous graph.
        edge_label_index: Target edges (local to ``data``).
        edge_label: Target labels.
        edge_time: Target timestamps.
        edge_attr: Optional target edge attributes.
        batch_size: Targets per batch.
        num_neighbors: Sample limits per hop.
        shuffle: Whether to shuffle target edges.
        seed: Seed for the shuffle permutation (neighbor selection is deterministic).
        compute_causal_node_features: Whether to compute the 8-dim causal node features.
        isolate_targets: If True (default), each target gets its own subgraph and node
            features with its own timestamp as cutoff, so a batched score equals scoring the
            target alone. If False, all targets in a batch share one induced subgraph
            (legacy behaviour, kept to reproduce earlier GraphSAGE numbers).
    """

    def __init__(
        self,
        data: Data,
        edge_label_index: torch.Tensor,
        edge_label: torch.Tensor,
        edge_time: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch_size: int = 512,
        num_neighbors: list[int] | None = None,
        shuffle: bool = True,
        seed: int | None = 42,
        compute_causal_node_features: bool = True,
        isolate_targets: bool = True,
    ) -> None:
        self.data = data
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors if num_neighbors is not None else [10, 10]
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.compute_causal_node_features = compute_causal_node_features
        self.isolate_targets = isolate_targets

        self.target_edge_index = edge_label_index
        self.edge_label = edge_label
        self.edge_time = edge_time
        self.edge_attr = edge_attr
        self.num_edges = self.target_edge_index.size(1)
        self._target_src = self.target_edge_index[0].cpu().numpy().astype(np.int64)
        self._target_dst = self.target_edge_index[1].cpu().numpy().astype(np.int64)
        self._labels = edge_label.cpu().numpy()
        self._times = edge_time.cpu().numpy().astype(np.int64)
        self._attr = edge_attr.cpu().numpy() if edge_attr is not None else None

        self.n_nodes = int(data.x.size(0))
        e_time = getattr(data, "edge_time", None)
        if e_time is None:
            e_time = torch.zeros(data.edge_index.size(1), dtype=torch.long)

        e_amt = None
        if (
            hasattr(data, "edge_attr")
            and data.edge_attr is not None
            and data.edge_attr.size(1) > 0
        ):
            e_amt = data.edge_attr[:, 0]

        self.adj = FastTemporalAdjacency(
            edge_index=data.edge_index,
            edge_time=e_time,
            num_src_nodes=self.n_nodes,
            edge_amount=e_amt,
        )

    def __len__(self) -> int:
        return math.ceil(self.num_edges / self.batch_size)

    def __iter__(self) -> Iterator[Data]:
        perm = self.rng.permutation(self.num_edges) if self.shuffle else np.arange(self.num_edges)
        for start_idx in range(0, self.num_edges, self.batch_size):
            yield self.build_batch(perm[start_idx : start_idx + self.batch_size])

    def build_batch(self, batch_indices: np.ndarray) -> Data:
        """Build the mini-batch for the given target positions (isolated or shared)."""
        batch_indices = np.asarray(batch_indices, dtype=np.int64)
        if self.isolate_targets:
            return self._build_isolated(batch_indices)
        return self._build_shared(batch_indices)

    def _node_features(self, nodes: np.ndarray, cuts: np.ndarray) -> np.ndarray:
        """8-dim causal node features: bias, distinct neighbors, degree, amount, age."""
        d, dist, a, ft = self.adj.get_causal_node_stats(nodes, cuts)
        ages_hours = np.zeros(len(nodes), dtype=np.float32)
        has_prior = ft < np.inf
        ages_hours[has_prior] = np.maximum(
            0.0,
            (cuts[has_prior] - ft[has_prior]) / 3600.0,
        )
        feats = np.zeros((len(nodes), 8), dtype=np.float32)
        feats[:, 0] = 1.0
        feats[:, 1] = np.log1p(dist)
        feats[:, 5] = np.log1p(d)
        feats[:, 6] = np.log1p(np.maximum(0.0, a))
        feats[:, 7] = np.log1p(ages_hours)
        return feats

    def _finish(
        self, batch: Data, batch_indices: np.ndarray, loc_src: np.ndarray, loc_dst: np.ndarray
    ) -> Data:
        """Attach target edges, labels, times and attributes."""
        batch.edge_label_index = torch.from_numpy(np.stack([loc_src, loc_dst]).astype(np.int64))
        batch.edge_label = torch.from_numpy(self._labels[batch_indices]).to(dtype=torch.float32)
        batch.edge_time = torch.from_numpy(self._times[batch_indices])
        if self._attr is not None:
            batch.edge_attr = torch.from_numpy(self._attr[batch_indices]).to(dtype=torch.float32)
        batch.batch_size = len(batch_indices)
        return batch

    def _build_isolated(self, batch_indices: np.ndarray) -> Data:
        """One independent subgraph per target, keyed by (target slot, node id)."""
        n = self.n_nodes
        n_e = self.data.edge_index.size(1)
        slots = np.arange(len(batch_indices), dtype=np.int64)
        b_times = self._times[batch_indices]
        b_src = self._target_src[batch_indices]
        b_dst = self._target_dst[batch_indices]

        sampled = np.unique(np.concatenate([slots * n + b_src, slots * n + b_dst]))
        frontier = sampled
        edge_keys: list[np.ndarray] = []
        for k_limit in self.num_neighbors:
            if len(frontier) == 0:
                continue
            f_slot = frontier // n
            dst, eids, row = self.adj.sample_causal_neighbors_indexed(
                frontier % n, b_times[f_slot], k_limit
            )
            if len(eids) == 0:
                frontier = np.empty(0, dtype=np.int64)
                continue
            e_slot = f_slot[row]
            edge_keys.append(e_slot * n_e + eids)
            cand = np.unique(e_slot * n + dst)
            frontier = np.setdiff1d(cand, sampled, assume_unique=True)
            sampled = np.union1d(sampled, frontier)

        node_ids = sampled % n
        batch = Data()
        batch.num_nodes = len(sampled)
        batch.n_id = torch.from_numpy(node_ids)
        if self.compute_causal_node_features:
            batch.x = torch.from_numpy(self._node_features(node_ids, b_times[sampled // n]))
        else:
            batch.x = self.data.x[node_ids]

        if edge_keys:
            keys = np.unique(np.concatenate(edge_keys))
            e_slot, eids = keys // n_e, keys % n_e
            g_src = self.data.edge_index[0, eids].cpu().numpy()
            g_dst = self.data.edge_index[1, eids].cpu().numpy()
            batch.edge_index = torch.from_numpy(
                np.stack(
                    [
                        np.searchsorted(sampled, e_slot * n + g_src),
                        np.searchsorted(sampled, e_slot * n + g_dst),
                    ]
                ).astype(np.int64)
            )
            # Provenance for audits (not model inputs): original edge id and owning target.
            batch.e_id = torch.from_numpy(eids.astype(np.int64))
            batch.target_slot = torch.from_numpy(e_slot.astype(np.int64))
        else:
            batch.edge_index = torch.empty((2, 0), dtype=torch.long)
            batch.e_id = torch.empty(0, dtype=torch.long)
            batch.target_slot = torch.empty(0, dtype=torch.long)

        return self._finish(
            batch,
            batch_indices,
            np.searchsorted(sampled, slots * n + b_src),
            np.searchsorted(sampled, slots * n + b_dst),
        )

    def _build_shared(self, batch_indices: np.ndarray) -> Data:
        """One induced subgraph shared by all targets in the batch (legacy behaviour)."""
        b_src = self._target_src[batch_indices]
        b_dst = self._target_dst[batch_indices]
        b_times = self._times[batch_indices]

        sampled_nodes = set(b_src.tolist()) | set(b_dst.tolist())
        sampled_eids: list[np.ndarray] = []

        frontier_nodes: list[int] = list(b_src) + list(b_dst)
        frontier_times: list[int] = list(b_times) + list(b_times)
        node_cutoffs: dict[int, int] = {}

        for u_id, t_cut in zip(b_src, b_times, strict=False):
            u_int = int(u_id)
            t_int = int(t_cut)
            if u_int not in node_cutoffs or t_int < node_cutoffs[u_int]:
                node_cutoffs[u_int] = t_int
        for v_id, t_cut in zip(b_dst, b_times, strict=False):
            v_int = int(v_id)
            t_int = int(t_cut)
            if v_int not in node_cutoffs or t_int < node_cutoffs[v_int]:
                node_cutoffs[v_int] = t_int

        for hop_k in range(len(self.num_neighbors)):
            k_limit = self.num_neighbors[hop_k]
            if not frontier_nodes:
                continue

            s_dst, s_eids, s_cuts = self.adj.sample_causal_neighbors(
                src_nodes=frontier_nodes,
                time_cutoffs=frontier_times,
                max_neighbors=k_limit,
                rng=self.rng,
            )

            next_fn: list[int] = []
            next_ft: list[int] = []

            if len(s_eids) > 0:
                sampled_eids.append(s_eids)
                for d_node, d_cut in zip(s_dst, s_cuts, strict=False):
                    d_int = int(d_node)
                    d_c_int = int(d_cut)
                    if d_int not in node_cutoffs or d_c_int < node_cutoffs[d_int]:
                        node_cutoffs[d_int] = d_c_int
                    if d_int not in sampled_nodes:
                        sampled_nodes.add(d_int)
                        next_fn.append(d_int)
                        next_ft.append(d_c_int)

            frontier_nodes = next_fn
            frontier_times = next_ft

        nodes_sorted = np.array(sorted(sampled_nodes), dtype=np.int64)
        global_to_local = {gid: lid for lid, gid in enumerate(nodes_sorted)}

        batch = Data()
        batch.num_nodes = len(nodes_sorted)
        batch.n_id = torch.from_numpy(nodes_sorted)

        if self.compute_causal_node_features:
            def_cut = int(b_times.min()) if len(b_times) > 0 else 0
            cuts_arr = np.array(
                [node_cutoffs.get(int(gid), def_cut) for gid in nodes_sorted],
                dtype=np.int64,
            )
            batch.x = torch.from_numpy(self._node_features(nodes_sorted, cuts_arr))
        else:
            batch.x = self.data.x[nodes_sorted]

        if sampled_eids:
            all_eids = np.unique(np.concatenate(sampled_eids))
            orig_sub_src = self.data.edge_index[0, all_eids].cpu().numpy()
            orig_sub_dst = self.data.edge_index[1, all_eids].cpu().numpy()

            loc_src = np.array([global_to_local[u] for u in orig_sub_src], dtype=np.int64)
            loc_dst = np.array([global_to_local[v] for v in orig_sub_dst], dtype=np.int64)
            batch.edge_index = torch.stack(
                [torch.from_numpy(loc_src), torch.from_numpy(loc_dst)], dim=0
            )
        else:
            batch.edge_index = torch.empty((2, 0), dtype=torch.long)

        return self._finish(
            batch,
            batch_indices,
            np.array([global_to_local[u] for u in b_src], dtype=np.int64),
            np.array([global_to_local[v] for v in b_dst], dtype=np.int64),
        )
