import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from dgraph_fraud.models import GCN, GraphSAGE, RGCN, GNNConfig, build_model


@pytest.fixture
def tiny_graph() -> Data:
    return Data(
        x=torch.arange(8 * 17, dtype=torch.float32).reshape(8, 17) / 100,
        edge_index=torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 1, 4],
                [1, 2, 3, 4, 5, 6, 7, 0, 5, 2],
            ],
            dtype=torch.long,
        ),
    )


@pytest.mark.parametrize("model_class", [GCN, GraphSAGE])
def test_forward_returns_one_finite_logit_per_node(model_class, tiny_graph):
    model = model_class(GNNConfig(dropout=0.0))
    model.eval()

    logits = model(tiny_graph.x, tiny_graph.edge_index)

    assert logits.shape == (tiny_graph.num_nodes,)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("model_name", ["gcn", "graphsage", "sage"])
def test_factory_builds_supported_models(model_name):
    model = build_model(model_name, GNNConfig())

    assert isinstance(model, (GCN, GraphSAGE))


@pytest.mark.parametrize("model_class", [GCN, GraphSAGE])
def test_model_accepts_neighbor_loader_batch(model_class, tiny_graph):
    loader = NeighborLoader(
        tiny_graph,
        input_nodes=torch.arange(tiny_graph.num_nodes),
        num_neighbors=[3, 2],
        batch_size=3,
        shuffle=False,
        num_workers=0,
        subgraph_type="directional",
    )
    batch = next(iter(loader))
    model = model_class(GNNConfig(dropout=0.0))
    model.eval()

    logits = model(batch.x, batch.edge_index)

    assert logits.shape == (batch.num_nodes,)
    assert logits[: batch.batch_size].shape == (3,)


def test_config_supports_missing_indicator_features():
    config = GNNConfig(in_channels=34)

    assert config.in_channels == 34


def test_rgcn_uses_sampled_relation_types():
    data = Data(
        x=torch.randn(6, 34),
        y=torch.tensor([0, 1, 2, 3, 0, 2]),
        edge_index=torch.tensor(
            [[0, 2, 1, 3, 4, 5], [1, 0, 2, 5, 3, 4]], dtype=torch.long
        ),
        edge_type=torch.tensor([0, 2, 1, 3, 1, 2], dtype=torch.long),
    )
    loader = NeighborLoader(
        data,
        input_nodes=torch.tensor([0, 1, 4]),
        num_neighbors=[3, 2],
        batch_size=2,
        shuffle=False,
        num_workers=0,
        subgraph_type="directional",
    )
    batch = next(iter(loader))
    model = RGCN(
        GNNConfig(
            in_channels=34,
            hidden_channels=13,
            dropout=0.0,
            num_relations=4,
        )
    )

    logits = model(batch.x, batch.edge_index, batch.edge_type)

    assert logits.shape == (batch.num_nodes,)
    assert torch.isfinite(logits).all()
    assert sum(parameter.numel() for parameter in model.parameters()) == 2289


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("in_channels", 0),
        ("hidden_channels", 0),
        ("num_layers", 1),
        ("dropout", 1.0),
    ],
)
def test_config_rejects_invalid_values(field, value):
    values = {field: value}

    with pytest.raises(ValueError):
        GNNConfig(**values)


def test_model_rejects_invalid_edge_index_dtype(tiny_graph):
    model = GCN(GNNConfig())

    with pytest.raises(TypeError, match="torch.long"):
        model(tiny_graph.x, tiny_graph.edge_index.to(torch.int32))
