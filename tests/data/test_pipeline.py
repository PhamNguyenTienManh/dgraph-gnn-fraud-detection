import numpy as np

from dgraph_fraud.data.graph import (
    build_graph_arrays,
    build_target_background_edge_type,
    to_torch_tensors,
)
from dgraph_fraud.data.loader import load_dgraphfin
from dgraph_fraud.data.profiler import profile_dataset
from dgraph_fraud.data.sampling import DirectedNeighborSampler, NeighborSamplingPolicy
from dgraph_fraud.data.validator import validate_dataset


def _write_tiny_dataset(path):
    np.savez(
        path,
        x=np.arange(102, dtype=np.float64).reshape(6, 17),
        y=np.array([0, 1, 2, 3, 0, 1]),
        edge_index=np.array([[0, 1], [1, 2], [2, 0], [4, 5]]),
        edge_type=np.array([0, 1, 1, 0]),
        edge_timestamp=np.array([10, 20, 30, 40]),
        train_mask=np.array([0, 1]),
        valid_mask=np.array([4]),
        test_mask=np.array([5]),
    )


def test_load_validate_profile_and_graph_view(tmp_path):
    path = tmp_path / "tiny.npz"
    _write_tiny_dataset(path)

    dataset = load_dgraphfin(path)
    validation = validate_dataset(dataset)
    profile = profile_dataset(dataset)
    graph = build_graph_arrays(dataset)

    assert validation.is_valid
    assert dataset.x.dtype == np.float32
    assert profile["label_counts"] == {"0": 2, "1": 2, "2": 1, "3": 1}
    assert len(profile["features"]) == 17
    assert graph.edge_index.shape == (2, 4)
    assert graph.directed is True


def test_validator_rejects_overlapping_splits(tmp_path):
    path = tmp_path / "bad.npz"
    _write_tiny_dataset(path)
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["valid_mask"] = np.array([1])
    np.savez(path, **arrays)

    report = validate_dataset(load_dgraphfin(path))

    assert not report.is_valid
    assert any("chồng lấn" in error for error in report.errors)


def test_directed_neighbor_sampler_uses_incoming_edges(tmp_path):
    path = tmp_path / "tiny.npz"
    _write_tiny_dataset(path)
    dataset = load_dgraphfin(path)
    policy = NeighborSamplingPolicy(num_neighbors=(2,), batch_size=1, seed=42)
    sampler = DirectedNeighborSampler(dataset.edge_index, dataset.num_nodes, policy)

    sampled = sampler.sample(np.array([2]))

    assert sampled.hop_edge_counts == (1,)
    assert sampled.nodes.tolist() == [1, 2]
    assert sampled.edge_index.tolist() == [[0], [1]]


def test_torch_conversion_preserves_graph_shape(tmp_path):
    pytest = __import__("pytest")
    pytest.importorskip("torch")
    path = tmp_path / "tiny.npz"
    _write_tiny_dataset(path)
    graph = build_graph_arrays(load_dgraphfin(path))

    tensors = to_torch_tensors(graph)

    assert tuple(tensors["x"].shape) == (6, 17)
    assert tuple(tensors["edge_index"].shape) == (2, 4)


def test_target_background_relations_do_not_expose_fraud_identity():
    torch = __import__("torch")
    labels = torch.tensor([0, 1, 2, 3])
    edge_index = torch.tensor(
        [[0, 1, 0, 2, 2, 3], [1, 0, 2, 0, 3, 2]], dtype=torch.long
    )

    relation = build_target_background_edge_type(labels, edge_index)

    assert relation.tolist() == [0, 0, 1, 2, 3, 3]
