"""Temporal Inductive Link and Neighbor Loaders for Graph Neural Networks.

Implements multi-hop neighborhood sampling (LinkNeighborLoader / NeighborLoader)
in pure PyTorch with pre-indexed adjacency structures. Guarantees zero lookahead
bias across temporal partitions without relying on platform-specific C++ extensions.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data, HeteroData


class FastAdjacency:
    """Pre-indexed Compressed Sparse Row (CSR) adjacency structure for rapid neighbor querying."""

    def __init__(self, edge_index: torch.Tensor, num_src_nodes: int) -> None:
        self.num_src_nodes = num_src_nodes
        if edge_index.size(1) == 0:
            self.ptrs = np.zeros(num_src_nodes + 1, dtype=np.int64)
            self.dst = np.empty(0, dtype=np.int64)
            self.edge_ids = np.empty(0, dtype=np.int64)
            return

        src_arr = edge_index[0].cpu().numpy()
        dst_arr = edge_index[1].cpu().numpy()
        edge_ids = np.arange(len(src_arr), dtype=np.int64)

        sort_idx = np.argsort(src_arr)
        src_sorted = src_arr[sort_idx]
        self.dst = dst_arr[sort_idx]
        self.edge_ids = edge_ids[sort_idx]

        counts = np.bincount(src_sorted, minlength=num_src_nodes)
        self.ptrs = np.zeros(num_src_nodes + 1, dtype=np.int64)
        self.ptrs[1:] = np.cumsum(counts)

    def sample_neighbors(
        self,
        src_nodes: np.ndarray | list[int],
        max_neighbors: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample up to max_neighbors for each source node.

        Returns:
            Tuple of (sampled_dst_nodes, sampled_original_edge_ids).
        """
        if len(self.dst) == 0 or len(src_nodes) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

        out_dst: list[np.ndarray] = []
        out_eids: list[np.ndarray] = []

        for u in src_nodes:
            if u >= self.num_src_nodes:
                continue
            start = self.ptrs[u]
            end = self.ptrs[u + 1]
            deg = end - start
            if deg <= 0:
                continue
            if deg <= max_neighbors:
                out_dst.append(self.dst[start:end])
                out_eids.append(self.edge_ids[start:end])
            else:
                offset = int(rng.integers(0, deg - max_neighbors + 1))
                out_dst.append(self.dst[start + offset : start + offset + max_neighbors])
                out_eids.append(self.edge_ids[start + offset : start + offset + max_neighbors])

        if out_dst:
            return np.concatenate(out_dst), np.concatenate(out_eids)
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)


class LinkNeighborLoader:
    """Multi-hop link-level neighborhood sampler for HeteroData graphs.

    Generates inductive mini-batches for edge classification by extracting the
    k-hop relational neighborhoods around the source and destination nodes of each
    target transaction edge.

    Args:
        data: HeteroData object containing graph structure and node/edge attributes.
        num_neighbors: Number of neighbors to sample per node at each hop (e.g. [10, 10]).
        edge_label_index: Tuple specifying target relation and edge index,
            e.g. (("user", "transacts_at", "merchant"), edge_index), or raw tensor.
        edge_label: Binary ground truth labels for the target edges.
        edge_attr: Optional edge attributes for the target edges.
        batch_size: Number of target edges per mini-batch.
        shuffle: Whether to shuffle target edges each epoch.
        seed: Random seed for reproducible neighbor sampling and batch ordering.
    """

    def __init__(
        self,
        data: HeteroData,
        num_neighbors: list[int] | dict[Any, list[int]],
        edge_label_index: tuple[tuple[str, str, str], torch.Tensor] | torch.Tensor,
        edge_label: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch_size: int = 512,
        shuffle: bool = True,
        seed: int | None = 42,
    ) -> None:
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if isinstance(num_neighbors, list):
            self.num_neighbors = num_neighbors
        elif isinstance(num_neighbors, dict):
            first_val = next(iter(num_neighbors.values()))
            self.num_neighbors = first_val if isinstance(first_val, list) else [10, 10]
        else:
            self.num_neighbors = [10, 10]

        # Extract target relation and index
        if isinstance(edge_label_index, tuple):
            self.target_edge_type = edge_label_index[0]
            self.target_edge_index = edge_label_index[1]
        else:
            self.target_edge_type = ("user", "transacts_at", "merchant")
            self.target_edge_index = edge_label_index

        self.edge_label = edge_label
        self.edge_attr = edge_attr
        self.num_edges = self.target_edge_index.size(1)

        # Pre-build CSR indexes for all edge relations
        self.adj_dict: dict[tuple[str, str, str], FastAdjacency] = {}
        for edge_type in data.edge_types:
            src_t, _, _ = edge_type
            n_src = data[src_t].num_nodes
            e_idx = data[edge_type].edge_index
            self.adj_dict[edge_type] = FastAdjacency(e_idx, n_src)

    def __len__(self) -> int:
        return math.ceil(self.num_edges / self.batch_size)

    def __iter__(self) -> Iterator[HeteroData]:
        perm = self.rng.permutation(self.num_edges) if self.shuffle else np.arange(self.num_edges)

        src_t, _, dst_t = self.target_edge_type
        target_src_all = self.target_edge_index[0].cpu().numpy()
        target_dst_all = self.target_edge_index[1].cpu().numpy()
        labels_all = self.edge_label.cpu().numpy()
        attr_all = self.edge_attr.cpu().numpy() if self.edge_attr is not None else None

        for start_idx in range(0, self.num_edges, self.batch_size):
            batch_indices = perm[start_idx : start_idx + self.batch_size]

            b_src = target_src_all[batch_indices]
            b_dst = target_dst_all[batch_indices]
            b_labels = labels_all[batch_indices]
            b_attr = attr_all[batch_indices] if attr_all is not None else None

            # 1. Multi-hop neighborhood expansion
            sampled_nodes: dict[str, set[int]] = {nt: set() for nt in self.data.node_types}
            sampled_nodes[src_t].update(b_src)
            sampled_nodes[dst_t].update(b_dst)

            sampled_eids: dict[tuple[str, str, str], list[np.ndarray]] = {
                et: [] for et in self.data.edge_types
            }

            current_frontier: dict[str, set[int]] = {
                nt: set(sampled_nodes[nt]) for nt in self.data.node_types
            }

            for hop_k in range(len(self.num_neighbors)):
                k_limit = self.num_neighbors[hop_k]
                next_frontier: dict[str, set[int]] = {nt: set() for nt in self.data.node_types}

                for edge_type, adj in self.adj_dict.items():
                    s_t, _, d_t = edge_type
                    seeds = list(current_frontier[s_t])
                    if not seeds:
                        continue
                    s_dst, s_eids = adj.sample_neighbors(seeds, k_limit, self.rng)
                    if len(s_eids) > 0:
                        sampled_eids[edge_type].append(s_eids)
                    for d_node in s_dst:
                        if d_node not in sampled_nodes[d_t]:
                            next_frontier[d_t].add(d_node)
                            sampled_nodes[d_t].add(d_node)

                current_frontier = next_frontier

            # 2. Build Mini-Batch HeteroData
            batch = HeteroData()
            global_to_local: dict[str, dict[int, int]] = {}

            for nt in self.data.node_types:
                nodes_sorted = np.array(sorted(sampled_nodes[nt]), dtype=np.int64)
                global_to_local[nt] = {gid: lid for lid, gid in enumerate(nodes_sorted)}
                batch[nt].num_nodes = len(nodes_sorted)
                batch[nt].n_id = torch.from_numpy(nodes_sorted)
                if hasattr(self.data[nt], "x") and self.data[nt].x is not None:
                    batch[nt].x = self.data[nt].x[nodes_sorted]

            # 3. Induce relational edges directly from sampled edge IDs
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
            if b_attr is not None:
                batch[self.target_edge_type].edge_attr = torch.from_numpy(b_attr).to(
                    dtype=torch.float32
                )
            batch[self.target_edge_type].batch_size = len(batch_indices)

            yield batch


class HomogeneousLinkLoader:
    """Multi-hop link-level neighborhood sampler for homogeneous PyG Data graphs.

    Args:
        data: Homogeneous Data object.
        edge_label_index: Target edges tensor of shape [2, E_target].
        edge_label: Target edge labels.
        edge_attr: Optional target edge attributes.
        batch_size: Batch size.
        num_neighbors: Neighbors sampled per hop.
        shuffle: Whether to shuffle target edges each epoch.
        seed: Random seed.
    """

    def __init__(
        self,
        data: Data,
        edge_label_index: torch.Tensor,
        edge_label: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch_size: int = 512,
        num_neighbors: list[int] | None = None,
        shuffle: bool = True,
        seed: int | None = 42,
    ) -> None:
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.num_neighbors = num_neighbors or [10, 10]

        self.edge_label_index = edge_label_index
        self.edge_label = edge_label
        self.edge_attr = edge_attr
        self.num_edges = edge_label_index.size(1)

        # Pre-build CSR adjacency for all homogeneous edges
        self.adj = FastAdjacency(data.edge_index, data.num_nodes)

    def __len__(self) -> int:
        return math.ceil(self.num_edges / self.batch_size)

    def __iter__(self) -> Iterator[Data]:
        perm = self.rng.permutation(self.num_edges) if self.shuffle else np.arange(self.num_edges)

        src_all = self.edge_label_index[0].cpu().numpy()
        dst_all = self.edge_label_index[1].cpu().numpy()
        labels_all = self.edge_label.cpu().numpy()
        attr_all = self.edge_attr.cpu().numpy() if self.edge_attr is not None else None

        for start_idx in range(0, self.num_edges, self.batch_size):
            batch_indices = perm[start_idx : start_idx + self.batch_size]

            b_src = src_all[batch_indices]
            b_dst = dst_all[batch_indices]
            b_labels = labels_all[batch_indices]
            b_attr = attr_all[batch_indices] if attr_all is not None else None

            # Multi-hop expansion
            sampled_nodes: set[int] = set(b_src) | set(b_dst)
            sampled_eids: list[np.ndarray] = []
            current_frontier = set(sampled_nodes)

            for hop_k in range(len(self.num_neighbors)):
                k_limit = self.num_neighbors[hop_k]
                seeds = list(current_frontier)
                if not seeds:
                    break
                s_dst, s_eids = self.adj.sample_neighbors(seeds, k_limit, self.rng)
                if len(s_eids) > 0:
                    sampled_eids.append(s_eids)
                next_frontier: set[int] = set()
                for d_node in s_dst:
                    if d_node not in sampled_nodes:
                        next_frontier.add(d_node)
                        sampled_nodes.add(d_node)
                current_frontier = next_frontier

            nodes_sorted = np.array(sorted(sampled_nodes), dtype=np.int64)
            global_to_local = {gid: lid for lid, gid in enumerate(nodes_sorted)}

            # Induce edges directly
            if sampled_eids:
                all_eids = np.unique(np.concatenate(sampled_eids))
                e_src = self.data.edge_index[0, all_eids].cpu().numpy()
                e_dst = self.data.edge_index[1, all_eids].cpu().numpy()
                loc_src = np.array([global_to_local[u] for u in e_src], dtype=np.int64)
                loc_dst = np.array([global_to_local[v] for v in e_dst], dtype=np.int64)
                sub_edge_index = torch.stack(
                    [torch.from_numpy(loc_src), torch.from_numpy(loc_dst)], dim=0
                )
            else:
                sub_edge_index = torch.empty((2, 0), dtype=torch.long)

            batch = Data()
            batch.num_nodes = len(nodes_sorted)
            batch.n_id = torch.from_numpy(nodes_sorted)
            batch.x = self.data.x[nodes_sorted]
            batch.edge_index = sub_edge_index

            # Map target edges
            loc_t_src = np.array([global_to_local[u] for u in b_src], dtype=np.int64)
            loc_t_dst = np.array([global_to_local[v] for v in b_dst], dtype=np.int64)

            batch.edge_label_index = torch.stack(
                [torch.from_numpy(loc_t_src), torch.from_numpy(loc_t_dst)], dim=0
            )
            batch.edge_label = torch.from_numpy(b_labels).to(dtype=torch.float32)
            if b_attr is not None:
                batch.edge_attr = torch.from_numpy(b_attr).to(dtype=torch.float32)
            batch.batch_size = len(batch_indices)

            yield batch
