"""Dataset statistics used as Sprint 1 acceptance evidence."""

import numpy as np

from .loader import DGraphDataset


def _counts(values: np.ndarray) -> dict[str, int]:
    keys, counts = np.unique(values, return_counts=True)
    return {str(int(key)): int(count) for key, count in zip(keys, counts, strict=True)}


def profile_dataset(dataset: DGraphDataset) -> dict[str, object]:
    """Produce JSON-serializable core graph, label, split, and feature statistics."""

    split_profiles: dict[str, object] = {}
    for name in ("train_mask", "valid_mask", "test_mask"):
        indices = getattr(dataset, name)
        split_profiles[name.removesuffix("_mask")] = {
            "node_count": int(indices.size),
            "label_counts": _counts(dataset.y[indices]),
        }

    source = dataset.edge_index[:, 0]
    target = dataset.edge_index[:, 1]
    out_degree = np.bincount(source, minlength=dataset.num_nodes)
    in_degree = np.bincount(target, minlength=dataset.num_nodes)

    feature_profiles = []
    for feature_index in range(dataset.x.shape[1]):
        values = dataset.x[:, feature_index]
        feature_profiles.append(
            {
                "feature": feature_index,
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean(dtype=np.float64)),
                "std": float(values.std(dtype=np.float64)),
                "minus_one_count": int(np.count_nonzero(values == -1)),
            }
        )

    return {
        "num_nodes": dataset.num_nodes,
        "num_edges": dataset.num_edges,
        "feature_dimension": int(dataset.x.shape[1]),
        "feature_dtype": str(dataset.x.dtype),
        "feature_min": float(dataset.x.min()),
        "feature_max": float(dataset.x.max()),
        "features": feature_profiles,
        "label_counts": _counts(dataset.y),
        "splits": split_profiles,
        "edge_type_counts": _counts(dataset.edge_type),
        "timestamp": {
            "min": int(dataset.edge_timestamp.min()),
            "max": int(dataset.edge_timestamp.max()),
            "unique_count": int(np.unique(dataset.edge_timestamp).size),
        },
        "directed_graph": {
            "self_loop_count": int(np.count_nonzero(source == target)),
            "isolated_node_count": int(np.count_nonzero((in_degree + out_degree) == 0)),
            "max_in_degree": int(in_degree.max()),
            "max_out_degree": int(out_degree.max()),
            "mean_in_degree": float(in_degree.mean()),
            "mean_out_degree": float(out_degree.mean()),
        },
    }
