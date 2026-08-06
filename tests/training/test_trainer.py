import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from dgraph_fraud.models import GNNConfig, GraphSAGE
from dgraph_fraud.training.config import TrainingConfig
from dgraph_fraud.training.runner import _fixed_random_subset
from dgraph_fraud.training.trainer import BaselineTrainer


def _loader(data, input_nodes, *, shuffle=False):
    return NeighborLoader(
        data,
        input_nodes=input_nodes,
        num_neighbors=[2, 2],
        batch_size=4,
        shuffle=shuffle,
        num_workers=0,
        subgraph_type="directional",
    )


def test_trainer_runs_one_epoch_and_returns_metrics():
    torch.manual_seed(42)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    data = Data(
        x=torch.randn(8, 17),
        y=labels,
        edge_index=torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 1, 3], [1, 2, 3, 4, 5, 6, 7, 0, 5, 7]],
            dtype=torch.long,
        ),
    )
    model = GraphSAGE(GNNConfig(hidden_channels=8, dropout=0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = BaselineTrainer(
        model,
        optimizer,
        nn.BCEWithLogitsLoss(),
        TrainingConfig(
            epochs=1,
            learning_rate=0.01,
            weight_decay=0.0,
            early_stopping_patience=1,
            max_train_batches=1,
            max_eval_batches=1,
        ),
        seed=42,
        device=torch.device("cpu"),
    )

    result = trainer.fit(
        _loader(data, torch.tensor([0, 1, 2, 3]), shuffle=True),
        _loader(data, torch.tensor([0, 1, 2, 3])),
        _loader(data, torch.tensor([4, 5, 6, 7])),
    )

    assert result.epochs_completed == 1
    assert result.best_epoch == 1
    assert 0.0 <= result.best_validation.roc_auc <= 1.0
    assert 0.0 <= result.test.average_precision <= 1.0


def test_fixed_evaluation_subset_is_reproducible_and_not_a_prefix():
    indices = torch.arange(100)

    first = _fixed_random_subset(indices, 10, seed=42)
    second = _fixed_random_subset(indices, 10, seed=42)

    assert torch.equal(first, second)
    assert not torch.equal(first, indices[:10])
    assert torch.unique(first).numel() == 10
