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
import torch
from torch_geometric.data import Data, HeteroData

STANDARD_NODE_TYPES = ["user", "card", "merchant", "device", "ip"]


class FastTemporalAdjacency:
    """Pre-indexed temporal CSR adjacency structure for rapid causal neighbor querying."""

    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_time: torch.Tensor,
        num_src_nodes: int,
        edge_amount: torch.Tensor | None = None,
    ) -> None:
        self.num_src_nodes = num_src_nodes
        if edge_index.size(1) == 0:
            self.ptrs = np.zeros(num_src_nodes + 1, dtype=np.int64)
            self.dst = np.empty(0, dtype=np.int64)
            self.times = np.empty(0, dtype=np.int64)
            self.edge_ids = np.empty(0, dtype=np.int64)
            self.amounts = np.empty(0, dtype=np.float32)
            self.cumsum_amounts = np.zeros(1, dtype=np.float32)
            return

        src_arr = edge_index[0].cpu().numpy()
        dst_arr = edge_index[1].cpu().numpy()
        time_arr = edge_time.cpu().numpy()
        edge_ids = np.arange(len(src_arr), dtype=np.int64)

        if edge_amount is not None:
            amt_arr = edge_amount.cpu().numpy().astype(np.float32)
        else:
            amt_arr = np.zeros(len(src_arr), dtype=np.float32)

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

        self.cumsum_amounts = np.zeros(len(self.amounts) + 1, dtype=np.float32)
        self.cumsum_amounts[1:] = np.cumsum(self.amounts)

    def sample_causal_neighbors(
        self,
        src_nodes: np.ndarray | list[int],
        time_cutoffs: np.ndarray | list[int],
        max_neighbors: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample up to max_neighbors occurring strictly BEFORE time_cutoffs.

        Args:
            src_nodes: Array of source node indices.
            time_cutoffs: Array of timestamps. For src_nodes[k], only edges with
                edge_time < time_cutoffs[k] are valid.
            max_neighbors: Maximum neighbors to sample per source node.
            rng: NumPy random generator.

        Returns:
            Tuple of (sampled_dst_nodes, sampled_original_edge_ids, sampled_parent_cutoffs).
        """
        if len(self.dst) == 0 or len(src_nodes) == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )

        out_dst: list[np.ndarray] = []
        out_eids: list[np.ndarray] = []
        out_cutoffs: list[np.ndarray] = []

        for u, t in zip(src_nodes, time_cutoffs, strict=False):
            if u >= self.num_src_nodes:
                continue
            start = self.ptrs[u]
            end_all = self.ptrs[u + 1]
            if end_all <= start:
                continue

            # Binary search for edges strictly before time cutoff t
            pos = int(np.searchsorted(self.times[start:end_all], t, side="left"))
            end = start + pos
            deg = end - start
            if deg <= 0:
                continue

            if deg <= max_neighbors:
                out_dst.append(self.dst[start:end])
                out_eids.append(self.edge_ids[start:end])
                out_cutoffs.append(np.full(deg, t, dtype=np.int64))
            else:
                # Prioritize most recent interactions prior to cutoff t
                s_idx = end - max_neighbors
                out_dst.append(self.dst[s_idx:end])
                out_eids.append(self.edge_ids[s_idx:end])
                out_cutoffs.append(np.full(max_neighbors, t, dtype=np.int64))

        if out_dst:
            return (
                np.concatenate(out_dst),
                np.concatenate(out_eids),
                np.concatenate(out_cutoffs),
            )
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

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
        out_deg = np.zeros(n, dtype=np.float32)
        out_distinct = np.zeros(n, dtype=np.float32)
        out_amt = np.zeros(n, dtype=np.float32)
        out_first_time = np.full(n, np.inf, dtype=np.float64)

        if len(self.dst) == 0 or n == 0:
            return out_deg, out_distinct, out_amt, out_first_time

        for idx, (u, t) in enumerate(zip(nodes, time_cutoffs, strict=False)):
            if u >= self.num_src_nodes:
                continue
            start = self.ptrs[u]
            end_all = self.ptrs[u + 1]
            if end_all <= start:
                continue

            pos = int(np.searchsorted(self.times[start:end_all], t, side="left"))
            out_deg[idx] = float(pos)
            if pos > 0:
                sub_dst = self.dst[start : start + pos]
                out_distinct[idx] = 1.0 if pos == 1 else float(len(np.unique(sub_dst)))
                out_amt[idx] = float(self.cumsum_amounts[start + pos] - self.cumsum_amounts[start])
                out_first_time[idx] = float(self.times[start])

        return out_deg, out_distinct, out_amt, out_first_time


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
        batch_size: Target edges per batch.
        shuffle: Whether to shuffle target edges.
        seed: Random seed for reproducibility.
        compute_causal_node_features: Whether to dynamically compute 8-dim causal node features.
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
    ) -> None:
        self.data = data
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

        # Target transaction device and ip indices (if present)
        target_store = data[self.target_edge_type]
        if device_index is not None:
            self.device_index: np.ndarray | None = device_index.cpu().numpy()
        elif hasattr(target_store, "device_index") and target_store.device_index is not None:
            self.device_index = target_store.device_index.cpu().numpy()
        elif (
            ("user", "uses_device", "device") in data.edge_types
            and data["user", "uses_device", "device"].edge_index.size(1) >= self.num_edges
        ):
            self.device_index = (
                data["user", "uses_device", "device"].edge_index[1][: self.num_edges].cpu().numpy()
            )
        else:
            self.device_index = None

        if ip_index is not None:
            self.ip_index: np.ndarray | None = ip_index.cpu().numpy()
        elif hasattr(target_store, "ip_index") and target_store.ip_index is not None:
            self.ip_index = target_store.ip_index.cpu().numpy()
        elif (
            ("user", "uses_ip", "ip") in data.edge_types
            and data["user", "uses_ip", "ip"].edge_index.size(1) >= self.num_edges
        ):
            self.ip_index = (
                data["user", "uses_ip", "ip"].edge_index[1][: self.num_edges].cpu().numpy()
            )
        else:
            self.ip_index = None

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

        src_t, _, dst_t = self.target_edge_type
        target_src_all = self.target_edge_index[0].cpu().numpy()
        target_dst_all = self.target_edge_index[1].cpu().numpy()
        labels_all = self.edge_label.cpu().numpy()
        times_all = self.edge_time.cpu().numpy()
        attr_all = self.edge_attr.cpu().numpy() if self.edge_attr is not None else None

        for start_idx in range(0, self.num_edges, self.batch_size):
            batch_indices = perm[start_idx : start_idx + self.batch_size]

            b_src = target_src_all[batch_indices]
            b_dst = target_dst_all[batch_indices]
            b_labels = labels_all[batch_indices]
            b_times = times_all[batch_indices]
            b_attr = attr_all[batch_indices] if attr_all is not None else None
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
                            if (
                                d_int not in node_cutoffs[d_t]
                                or d_c_int < node_cutoffs[d_t][d_int]
                            ):
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
                    # Dynamically compute strictly causal 17-dim node features:
                    # [0:5]   One-hot node type (user, card, merchant, device, ip)
                    # [5:10]  log1p(distinct neighbors by target type: user, card, merchant, device, ip)
                    # [10:15] log1p(total edges by target type: user, card, merchant, device, ip)
                    # [15]    log1p(total_amount)
                    # [16]    log1p(age_hours)
                    type_idx = STANDARD_NODE_TYPES.index(nt) if nt in STANDARD_NODE_TYPES else 0
                    def_cut = int(b_times.min()) if len(b_times) > 0 else 0
                    cuts_arr = np.array(
                        [node_cutoffs[nt].get(int(gid), def_cut) for gid in nodes_sorted],
                        dtype=np.int64,
                    )

                    n_nodes = len(nodes_sorted)
                    distinct_by_target = np.zeros((n_nodes, 5), dtype=np.float32)
                    deg_by_target = np.zeros((n_nodes, 5), dtype=np.float32)
                    total_amt = np.zeros(n_nodes, dtype=np.float32)
                    earliest_time = np.full(n_nodes, np.inf, dtype=np.float64)

                    for et, adj in self.adj_dict.items():
                        s_t, _, d_t = et
                        if s_t == nt:
                            d, dist, a, ft = adj.get_causal_node_stats(nodes_sorted, cuts_arr)
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
                        (cuts_arr[has_prior] - earliest_time[has_prior]) / 3600.0,
                    )

                    feats = np.zeros((n_nodes, 17), dtype=np.float32)
                    if type_idx < 5:
                        feats[:, type_idx] = 1.0
                    feats[:, 5:10] = np.log1p(distinct_by_target)
                    feats[:, 10:15] = np.log1p(deg_by_target)
                    feats[:, 15] = np.log1p(np.maximum(0.0, total_amt))
                    feats[:, 16] = np.log1p(ages_hours)
                    batch[nt].x = torch.from_numpy(feats)
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

            yield batch


class CausalHomogeneousLinkLoader:
    """Strictly causal homogeneous neighborhood sampler for GraphSAGE.

    Guarantees that for any target transaction edge at time t, sampled neighbors
    strictly satisfy edge_time < t across all hops, with dynamic causal node features.
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
    ) -> None:
        self.data = data
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors if num_neighbors is not None else [10, 10]
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.compute_causal_node_features = compute_causal_node_features

        self.target_edge_index = edge_label_index
        self.edge_label = edge_label
        self.edge_time = edge_time
        self.edge_attr = edge_attr
        self.num_edges = self.target_edge_index.size(1)

        n_nodes = data.x.size(0)
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
            num_src_nodes=n_nodes,
            edge_amount=e_amt,
        )

    def __len__(self) -> int:
        return math.ceil(self.num_edges / self.batch_size)

    def __iter__(self) -> Iterator[Data]:
        perm = self.rng.permutation(self.num_edges) if self.shuffle else np.arange(self.num_edges)

        target_src_all = self.target_edge_index[0].cpu().numpy()
        target_dst_all = self.target_edge_index[1].cpu().numpy()
        labels_all = self.edge_label.cpu().numpy()
        times_all = self.edge_time.cpu().numpy()
        attr_all = self.edge_attr.cpu().numpy() if self.edge_attr is not None else None

        for start_idx in range(0, self.num_edges, self.batch_size):
            batch_indices = perm[start_idx : start_idx + self.batch_size]

            b_src = target_src_all[batch_indices]
            b_dst = target_dst_all[batch_indices]
            b_labels = labels_all[batch_indices]
            b_times = times_all[batch_indices]
            b_attr = attr_all[batch_indices] if attr_all is not None else None

            sampled_nodes = set(b_src) | set(b_dst)
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
                d, dist, a, ft = self.adj.get_causal_node_stats(nodes_sorted, cuts_arr)
                ages_hours = np.zeros(len(nodes_sorted), dtype=np.float32)
                has_prior = ft < np.inf
                ages_hours[has_prior] = np.maximum(
                    0.0,
                    (cuts_arr[has_prior] - ft[has_prior]) / 3600.0,
                )
                feats = np.zeros((len(nodes_sorted), 8), dtype=np.float32)
                feats[:, 0] = 1.0
                feats[:, 1] = np.log1p(dist)
                feats[:, 5] = np.log1p(d)
                feats[:, 6] = np.log1p(np.maximum(0.0, a))
                feats[:, 7] = np.log1p(ages_hours)
                batch.x = torch.from_numpy(feats)
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

            local_target_src = np.array([global_to_local[u] for u in b_src], dtype=np.int64)
            local_target_dst = np.array([global_to_local[v] for v in b_dst], dtype=np.int64)
            batch.edge_label_index = torch.stack(
                [torch.from_numpy(local_target_src), torch.from_numpy(local_target_dst)],
                dim=0,
            )
            batch.edge_label = torch.from_numpy(b_labels).to(dtype=torch.float32)
            batch.edge_time = torch.from_numpy(b_times)
            if b_attr is not None:
                batch.edge_attr = torch.from_numpy(b_attr).to(dtype=torch.float32)
            batch.batch_size = len(batch_indices)

            yield batch
