import json

import pytest

from dgraph_fraud.training.config import load_experiment_config


def _payload():
    return {
        "run_name": "test",
        "mode": "full",
        "data_path": "tiny.npz",
        "output_root": "runs",
        "device": "cpu",
        "feature_mode": "raw",
        "seeds": [42],
        "models": ["gcn", "graphsage"],
        "model": {
            "in_channels": 17,
            "hidden_channels": 8,
            "num_layers": 2,
            "dropout": 0.0
        },
        "sampling": {"num_neighbors": [2, 2], "batch_size": 4, "num_workers": 0},
        "training": {
            "epochs": 1,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "early_stopping_patience": 1,
            "use_pos_weight": True,
            "gradient_clip_norm": 2.0,
            "max_train_batches": 1,
            "max_eval_batches": 1
        }
    }


def test_load_experiment_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    config = load_experiment_config(path)

    assert config.sampling.num_neighbors == (2, 2)
    assert config.models == ("gcn", "graphsage")
    assert config.feature_standardization == "none"
    assert config.graph_direction == "directed"


def test_config_accepts_ablation_controls(tmp_path):
    payload = _payload()
    payload["feature_standardization"] = "global_zscore"
    payload["graph_direction"] = "undirected"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_experiment_config(path)

    assert config.feature_standardization == "global_zscore"
    assert config.graph_direction == "undirected"


def test_config_requires_feature_dimension_match(tmp_path):
    payload = _payload()
    payload["feature_mode"] = "zero_indicator"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="in_channels=34"):
        load_experiment_config(path)


def test_config_accepts_paper_style_rgcn(tmp_path):
    payload = _payload()
    payload["feature_mode"] = "zero_indicator"
    payload["relation_mode"] = "target_background"
    payload["models"] = ["rgcn"]
    payload["model"] = {
        "in_channels": 34,
        "hidden_channels": 13,
        "num_layers": 2,
        "dropout": 0.5,
        "num_relations": 4,
    }
    path = tmp_path / "rgcn.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_experiment_config(path)

    assert config.models == ("rgcn",)
    assert config.relation_mode == "target_background"
    assert config.model.num_relations == 4
