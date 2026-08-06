import torch

from dgraph_fraud.models import GNNConfig, build_model
from dgraph_fraud.training.checkpoint import load_model_checkpoint


def test_checkpoint_rebuilds_model_and_predictions(tmp_path):
    torch.manual_seed(42)
    config = GNNConfig(in_channels=17, hidden_channels=8, num_layers=2, dropout=0.0)
    model = build_model("gcn", config).eval()
    x = torch.randn(5, 17)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    expected = model(x, edge_index)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "model": "gcn",
            "model_config": {
                "in_channels": 17,
                "hidden_channels": 8,
                "num_layers": 2,
                "dropout": 0.0,
            },
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    loaded, _ = load_model_checkpoint(checkpoint_path)

    assert torch.equal(expected, loaded(x, edge_index))
