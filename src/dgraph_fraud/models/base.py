"""Common interface shared by the Sprint 2 GNN baselines."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from .config import GNNConfig


class BaseNodeClassifier(nn.Module, ABC):
    """Base class for sampled-subgraph binary node classifiers.

    Implementations return one raw fraud logit per input node. Loss,
    probability conversion, and seed-node slicing belong to the future
    training/inference pipeline rather than the model itself.
    """

    def __init__(self, config: GNNConfig) -> None:
        super().__init__()
        self.config = config

    @staticmethod
    def _validate_inputs(x: Tensor, edge_index: Tensor) -> None:
        if x.ndim != 2:
            raise ValueError(f"x phải có 2 chiều [N, F], nhận shape {tuple(x.shape)}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index phải có shape [2, E], "
                f"nhận shape {tuple(edge_index.shape)}"
            )
        if edge_index.dtype != torch.long:
            raise TypeError(f"edge_index phải có dtype torch.long, nhận {edge_index.dtype}")

    @abstractmethod
    def forward(
        self, x: Tensor, edge_index: Tensor, edge_type: Tensor | None = None
    ) -> Tensor:
        """Return one raw binary-classification logit per node."""
