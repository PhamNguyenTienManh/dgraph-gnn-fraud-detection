"""Canonical graph views and optional PyTorch Geometric conversion."""

from dataclasses import dataclass

import numpy as np

from .loader import DGraphDataset


def build_target_background_edge_type(labels, edge_index):
    """Encode paper-style T->T, T->B, B->T and B->B relations.

    Labels 0/1 are target nodes and labels 2/3 are background nodes. Fraud
    identity is not exposed: normal and fraud nodes share the same target type.
    """

    import torch

    if labels.ndim != 1:
        raise ValueError("labels phải có shape [N]")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index phải có shape [2, E]")
    source_target = labels[edge_index[0]] <= 1
    destination_target = labels[edge_index[1]] <= 1
    # 0=T->T, 1=T->B, 2=B->T, 3=B->B.
    return ((~source_target).to(torch.long) * 2 + (~destination_target).to(torch.long))


@dataclass(frozen=True, slots=True)
class GraphArrays:
    x: np.ndarray
    y: np.ndarray
    edge_index: np.ndarray
    edge_type: np.ndarray
    edge_timestamp: np.ndarray
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray
    directed: bool = True


def build_graph_arrays(dataset: DGraphDataset) -> GraphArrays:
    """Create the [2, E] edge view expected by GNN frameworks without symmetrizing."""

    return GraphArrays(
        x=dataset.x,
        y=dataset.y,
        edge_index=dataset.edge_index.T,
        edge_type=dataset.edge_type,
        edge_timestamp=dataset.edge_timestamp,
        train_idx=dataset.train_mask,
        valid_idx=dataset.valid_mask,
        test_idx=dataset.test_mask,
    )


def to_pyg_data(graph: GraphArrays):
    """Convert canonical arrays to PyG Data when Sprint 2 extras are installed."""

    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:
        raise RuntimeError(
            "Cần cài optional dependency 'graph' để chuyển sang PyTorch Geometric"
        ) from exc

    return Data(
        x=torch.from_numpy(graph.x),
        y=torch.from_numpy(graph.y),
        edge_index=torch.from_numpy(np.ascontiguousarray(graph.edge_index)),
        edge_type=torch.from_numpy(graph.edge_type),
        edge_timestamp=torch.from_numpy(graph.edge_timestamp),
        train_idx=torch.from_numpy(graph.train_idx),
        valid_idx=torch.from_numpy(graph.valid_idx),
        test_idx=torch.from_numpy(graph.test_idx),
        directed=graph.directed,
    )


def to_torch_tensors(graph: GraphArrays) -> dict[str, object]:
    """Create shared-memory CPU tensors without requiring PyG."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Cần cài PyTorch để tạo tensor graph") from exc

    return {
        "x": torch.from_numpy(graph.x),
        "y": torch.from_numpy(graph.y),
        "edge_index": torch.from_numpy(np.ascontiguousarray(graph.edge_index)),
        "edge_type": torch.from_numpy(graph.edge_type),
        "edge_timestamp": torch.from_numpy(graph.edge_timestamp),
        "train_idx": torch.from_numpy(graph.train_idx),
        "valid_idx": torch.from_numpy(graph.valid_idx),
        "test_idx": torch.from_numpy(graph.test_idx),
    }
