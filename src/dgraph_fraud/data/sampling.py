"""Resource-aware directed neighbor sampling without native extensions."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class NeighborSamplingPolicy:
    num_neighbors: tuple[int, ...] = (15, 10)
    batch_size: int = 1024
    num_workers: int = 0
    seed: int = 42

    def validate(self) -> None:
        if not self.num_neighbors or any(value <= 0 for value in self.num_neighbors):
            raise ValueError("num_neighbors phải gồm các số nguyên dương")
        if self.batch_size <= 0:
            raise ValueError("batch_size phải dương")
        if self.num_workers < 0:
            raise ValueError("num_workers không được âm")


@dataclass(frozen=True, slots=True)
class SampledSubgraph:
    nodes: np.ndarray
    edge_index: np.ndarray
    seed_nodes: np.ndarray
    hop_edge_counts: tuple[int, ...]


class DirectedNeighborSampler:
    """Sample incoming neighbors for message flow ``source -> target``.

    The CSR index is built by destination node. This preserves the directed
    graph semantics and avoids creating a doubled, symmetrized edge list.
    """

    def __init__(
        self, edge_index: np.ndarray, num_nodes: int, policy: NeighborSamplingPolicy
    ) -> None:
        policy.validate()
        if edge_index.ndim != 2 or edge_index.shape[1] != 2:
            raise ValueError("edge_index phải có shape (E, 2)")
        self.policy = policy
        self.num_nodes = num_nodes
        source, target = edge_index[:, 0], edge_index[:, 1]
        order = np.argsort(target, kind="stable")
        self.neighbors = np.asarray(source[order], dtype=np.int64)
        counts = np.bincount(target, minlength=num_nodes)
        self.indptr = np.empty(num_nodes + 1, dtype=np.int64)
        self.indptr[0] = 0
        np.cumsum(counts, out=self.indptr[1:])

    @property
    def index_nbytes(self) -> int:
        return int(self.neighbors.nbytes + self.indptr.nbytes)

    def sample(self, seed_nodes: np.ndarray, *, seed: int | None = None) -> SampledSubgraph:
        seeds = np.unique(np.asarray(seed_nodes, dtype=np.int64))
        if seeds.size == 0:
            raise ValueError("seed_nodes không được rỗng")
        if int(seeds.min()) < 0 or int(seeds.max()) >= self.num_nodes:
            raise ValueError("seed_nodes chứa ID ngoài miền hợp lệ")

        rng = np.random.default_rng(self.policy.seed if seed is None else seed)
        frontier = seeds
        all_sources: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        hop_edge_counts: list[int] = []

        for fanout in self.policy.num_neighbors:
            hop_sources: list[np.ndarray] = []
            hop_targets: list[np.ndarray] = []
            for target in frontier:
                start, end = int(self.indptr[target]), int(self.indptr[target + 1])
                candidates = self.neighbors[start:end]
                if candidates.size > fanout:
                    chosen = candidates[rng.choice(candidates.size, fanout, replace=False)]
                else:
                    chosen = candidates
                if chosen.size:
                    hop_sources.append(chosen)
                    hop_targets.append(np.full(chosen.size, target, dtype=np.int64))

            if not hop_sources:
                hop_edge_counts.append(0)
                frontier = np.empty(0, dtype=np.int64)
                continue
            sources = np.concatenate(hop_sources)
            targets = np.concatenate(hop_targets)
            all_sources.append(sources)
            all_targets.append(targets)
            hop_edge_counts.append(int(sources.size))
            frontier = np.unique(sources)

        if all_sources:
            sources = np.concatenate(all_sources)
            targets = np.concatenate(all_targets)
            nodes = np.unique(np.concatenate((seeds, sources, targets)))
            local_sources = np.searchsorted(nodes, sources)
            local_targets = np.searchsorted(nodes, targets)
            local_edges = np.stack((local_sources, local_targets))
        else:
            nodes = seeds
            local_edges = np.empty((2, 0), dtype=np.int64)

        return SampledSubgraph(
            nodes=nodes,
            edge_index=local_edges,
            seed_nodes=seeds,
            hop_edge_counts=tuple(hop_edge_counts),
        )
