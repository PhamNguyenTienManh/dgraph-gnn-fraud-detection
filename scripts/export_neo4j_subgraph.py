"""Export a connected, fraud-centered DGraphFin sample for Neo4j Aura.

The source NPZ file is opened read-only. The exporter selects three Fraud seed
nodes that share a 1-hop neighbor, keeps every incoming and outgoing 1-hop
neighbor of those seeds, then adds genuine nearby Normal nodes from the next
hop until the Normal-to-Fraud ratio is approximately balanced.

The final sample is a weakly connected induced subgraph. Edge direction, type,
and timestamp are preserved. Exact duplicate edge records are removed without
inventing or rewiring data.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = ("y", "edge_index", "edge_type", "edge_timestamp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a connected DGraphFin subgraph for Neo4j Aura."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/dgraphfin.npz"),
        help="Path to the original DGraphFin NPZ file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/neo4j"),
        help="Directory for nodes.csv and edges.csv.",
    )
    parser.add_argument(
        "--fraud-seeds",
        type=int,
        default=3,
        help="Number of nearby Fraud seed nodes (default: 3).",
    )
    parser.add_argument(
        "--target-normal-ratio",
        type=float,
        default=1.5,
        help="Target Normal-to-Fraud ratio between 1.0 and 2.0 (default: 1.5).",
    )
    return parser.parse_args()


def load_dataset(dataset_path: Path) -> tuple[np.ndarray, ...]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with np.load(dataset_path, allow_pickle=False) as data:
        missing = [field for field in REQUIRED_FIELDS if field not in data.files]
        if missing:
            raise ValueError(f"Dataset is missing required fields: {missing}")

        labels = data["y"]
        edge_index = data["edge_index"]
        edge_types = data["edge_type"]
        timestamps = data["edge_timestamp"]

    if labels.ndim != 1:
        raise ValueError(f"Expected y to be one-dimensional, got {labels.shape}")
    if edge_index.ndim != 2 or edge_index.shape[1] != 2:
        raise ValueError(f"Expected edge_index shape (E, 2), got {edge_index.shape}")
    if edge_types.shape != (edge_index.shape[0],):
        raise ValueError("edge_type length does not match edge_index")
    if timestamps.shape != (edge_index.shape[0],):
        raise ValueError("edge_timestamp length does not match edge_index")
    if edge_index.size and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= labels.size
    ):
        raise ValueError("edge_index contains a node ID outside the label array")

    return labels, edge_index, edge_types, timestamps


def distinct_fraud_neighbor_pairs(
    labels: np.ndarray, edge_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return unique (neighbor ID, Fraud node ID) pairs in either direction."""
    sources = edge_index[:, 0]
    targets = edge_index[:, 1]
    source_is_fraud = labels[sources] == 1
    target_is_fraud = labels[targets] == 1

    neighbor_ids = np.concatenate(
        (targets[source_is_fraud], sources[target_is_fraud])
    )
    fraud_ids = np.concatenate((sources[source_is_fraud], targets[target_is_fraud]))
    if neighbor_ids.size == 0:
        raise ValueError("No edges incident to Fraud nodes were found")

    order = np.lexsort((fraud_ids, neighbor_ids))
    neighbor_ids = neighbor_ids[order]
    fraud_ids = fraud_ids[order]
    keep = np.ones(neighbor_ids.size, dtype=bool)
    keep[1:] = (neighbor_ids[1:] != neighbor_ids[:-1]) | (
        fraud_ids[1:] != fraud_ids[:-1]
    )
    return neighbor_ids[keep], fraud_ids[keep]


def select_close_fraud_seeds(
    labels: np.ndarray, edge_index: np.ndarray, num_seeds: int
) -> tuple[np.ndarray, int]:
    """Select Fraud seeds sharing the strongest common 1-hop hub."""
    if num_seeds <= 0:
        raise ValueError("fraud_seeds must be positive")

    neighbor_ids, fraud_ids = distinct_fraud_neighbor_pairs(labels, edge_index)
    unique_neighbors, starts, counts = np.unique(
        neighbor_ids, return_index=True, return_counts=True
    )
    eligible = np.flatnonzero(counts >= num_seeds)
    if eligible.size == 0:
        raise ValueError(
            f"No node is shared by at least {num_seeds} distinct Fraud nodes"
        )

    largest_group_size = int(counts[eligible].max())
    largest_groups = eligible[counts[eligible] == largest_group_size]
    group_index = int(
        largest_groups[np.argmin(unique_neighbors[largest_groups])]
    )
    shared_hub = int(unique_neighbors[group_index])
    start = int(starts[group_index])
    candidate_ids = fraud_ids[start : start + largest_group_size]

    distinct_neighbor_counts = np.bincount(
        fraud_ids, minlength=labels.size
    )[candidate_ids]
    rank = np.lexsort((candidate_ids, -distinct_neighbor_counts))
    seeds = candidate_ids[rank[:num_seeds]].astype(np.int64, copy=False)
    return np.sort(seeds), shared_hub


def collect_one_hop_nodes(
    seeds: np.ndarray, edge_index: np.ndarray, num_nodes: int
) -> np.ndarray:
    """Keep every node one incoming or outgoing edge away from a seed."""
    sources = edge_index[:, 0]
    targets = edge_index[:, 1]
    seed_mask = np.zeros(num_nodes, dtype=bool)
    seed_mask[seeds] = True
    incident = seed_mask[sources] | seed_mask[targets]
    return np.union1d(np.unique(edge_index[incident].reshape(-1)), seeds)


def expand_with_nearby_normal_nodes(
    base_nodes: np.ndarray,
    seeds: np.ndarray,
    labels: np.ndarray,
    edge_index: np.ndarray,
    target_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Add connected Normal nodes adjacent to the original 1-hop frontier."""
    if not 1.0 <= target_ratio <= 2.0:
        raise ValueError("target_normal_ratio must be between 1.0 and 2.0")

    base_label_counts = np.bincount(labels[base_nodes], minlength=4)
    num_normal = int(base_label_counts[0])
    num_fraud = int(base_label_counts[1])
    if num_fraud == 0:
        raise ValueError("The 1-hop sample unexpectedly contains no Fraud nodes")
    if num_normal > 2 * num_fraud:
        raise ValueError(
            "The complete 1-hop neighborhood already exceeds a 2:1 "
            "Normal-to-Fraud ratio; nodes will not be removed to force balance"
        )

    target_normal_count = math.ceil(target_ratio * num_fraud)
    normals_needed = max(0, target_normal_count - num_normal)
    if normals_needed == 0:
        return base_nodes, np.empty(0, dtype=np.int64)

    num_nodes = labels.size
    sources = edge_index[:, 0]
    targets = edge_index[:, 1]
    base_mask = np.zeros(num_nodes, dtype=bool)
    base_mask[base_nodes] = True

    frontier = np.setdiff1d(base_nodes, seeds, assume_unique=True)
    frontier_mask = np.zeros(num_nodes, dtype=bool)
    frontier_mask[frontier] = True

    outgoing_candidates = (
        frontier_mask[sources] & ~base_mask[targets] & (labels[targets] == 0)
    )
    incoming_candidates = (
        frontier_mask[targets] & ~base_mask[sources] & (labels[sources] == 0)
    )
    candidate_occurrences = np.concatenate(
        (targets[outgoing_candidates], sources[incoming_candidates])
    )
    candidate_ids, connection_counts = np.unique(
        candidate_occurrences, return_counts=True
    )
    if candidate_ids.size < normals_needed:
        raise ValueError(
            f"Only {candidate_ids.size} nearby Normal nodes are available, but "
            f"{normals_needed} are required to reach the target ratio"
        )

    rank = np.lexsort((candidate_ids, -connection_counts))
    added_normals = np.sort(candidate_ids[rank[:normals_needed]])
    return np.union1d(base_nodes, added_normals), added_normals


def induced_edge_ids(
    node_ids: np.ndarray, edge_index: np.ndarray, num_nodes: int
) -> np.ndarray:
    selected_mask = np.zeros(num_nodes, dtype=bool)
    selected_mask[node_ids] = True
    sources = edge_index[:, 0]
    targets = edge_index[:, 1]
    return np.flatnonzero(selected_mask[sources] & selected_mask[targets])


def remove_duplicate_edges(
    edge_ids: np.ndarray,
    edge_index: np.ndarray,
    edge_types: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    """Remove exact duplicate edge rows while retaining their original order."""
    if edge_ids.size == 0:
        return edge_ids

    edge_rows = np.column_stack(
        (
            edge_index[edge_ids, 0],
            edge_index[edge_ids, 1],
            edge_types[edge_ids],
            timestamps[edge_ids],
        )
    )
    _, first_positions = np.unique(edge_rows, axis=0, return_index=True)
    return edge_ids[np.sort(first_positions)]


def is_weakly_connected(
    node_ids: np.ndarray, edge_ids: np.ndarray, edge_index: np.ndarray
) -> bool:
    """Check connectivity after ignoring edge directions."""
    if node_ids.size <= 1:
        return True

    adjacency = {int(node_id): [] for node_id in node_ids}
    for edge_id in edge_ids:
        source = int(edge_index[edge_id, 0])
        target = int(edge_index[edge_id, 1])
        adjacency[source].append(target)
        adjacency[target].append(source)

    visited = {int(node_ids[0])}
    stack = [int(node_ids[0])]
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return len(visited) == node_ids.size


def write_nodes_csv(
    path: Path, node_ids: np.ndarray, labels: np.ndarray, seeds: np.ndarray
) -> None:
    seed_set = {int(node_id) for node_id in seeds}
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("node_id", "label", "is_seed"))
        writer.writerows(
            (
                int(node_id),
                int(labels[node_id]),
                "true" if int(node_id) in seed_set else "false",
            )
            for node_id in node_ids
        )


def write_edges_csv(
    path: Path,
    edge_ids: np.ndarray,
    edge_index: np.ndarray,
    edge_types: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("source", "target", "edge_type", "timestamp"))
        writer.writerows(
            (
                int(edge_index[edge_id, 0]),
                int(edge_index[edge_id, 1]),
                int(edge_types[edge_id]),
                int(timestamps[edge_id]),
            )
            for edge_id in edge_ids
        )


def main() -> None:
    args = parse_args()
    labels, edge_index, edge_types, timestamps = load_dataset(args.dataset)

    seeds, shared_hub = select_close_fraud_seeds(
        labels, edge_index, args.fraud_seeds
    )
    base_nodes = collect_one_hop_nodes(seeds, edge_index, labels.size)
    selected_nodes, added_normals = expand_with_nearby_normal_nodes(
        base_nodes,
        seeds,
        labels,
        edge_index,
        args.target_normal_ratio,
    )
    selected_edges = induced_edge_ids(selected_nodes, edge_index, labels.size)
    selected_edges = remove_duplicate_edges(
        selected_edges, edge_index, edge_types, timestamps
    )

    connected = is_weakly_connected(selected_nodes, selected_edges, edge_index)
    if not connected:
        raise ValueError("Final sample is not weakly connected")

    label_counts = np.bincount(labels[selected_nodes], minlength=4)
    normal_to_fraud_ratio = float(label_counts[0] / label_counts[1])
    if not 1.0 <= normal_to_fraud_ratio <= 2.0:
        raise ValueError(
            "Final Normal-to-Fraud ratio is outside the required 1:1 to 2:1 range"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = args.output_dir / "nodes.csv"
    edges_path = args.output_dir / "edges.csv"
    write_nodes_csv(nodes_path, selected_nodes, labels, seeds)
    write_edges_csv(
        edges_path, selected_edges, edge_index, edge_types, timestamps
    )

    base_counts = np.bincount(labels[base_nodes], minlength=4)
    print(f"Fraud seeds (shared 1-hop hub {shared_hub}): {seeds.tolist()}")
    print(f"Complete 1-hop base labels 0/1/2/3: {base_counts[:4].tolist()}")
    print(f"Added nearby Normal nodes: {added_normals.tolist()}")
    print(f"Final nodes: {selected_nodes.size}")
    print(f"Final node labels 0/1/2/3: {label_counts[:4].tolist()}")
    print(f"Normal-to-Fraud ratio: {normal_to_fraud_ratio:.3f}:1")
    print(f"Final directed edges: {selected_edges.size}")
    print("Weakly connected: yes")
    print(f"Nodes CSV: {nodes_path}")
    print(f"Edges CSV: {edges_path}")


if __name__ == "__main__":
    main()
