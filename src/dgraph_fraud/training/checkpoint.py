"""Safe loading of Sprint 2 model checkpoints."""

from pathlib import Path
from typing import Any

import torch

from dgraph_fraud.models import GNNConfig, build_model
from dgraph_fraud.models.base import BaseNodeClassifier


def load_model_checkpoint(
    path: str | Path, *, device: torch.device | str = "cpu"
) -> tuple[BaseNodeClassifier, dict[str, Any]]:
    """Rebuild a baseline model and load tensor weights from a trusted checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    required = {"model", "model_config", "state_dict"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint thiếu trường bắt buộc: {sorted(missing)}")

    model = build_model(
        str(checkpoint["model"]),
        GNNConfig(**checkpoint["model_config"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint
