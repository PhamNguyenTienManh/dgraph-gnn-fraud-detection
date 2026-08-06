"""GraphSAGE baseline."""

import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import SAGEConv

from .base import BaseNodeClassifier
from .config import GNNConfig


class GraphSAGE(BaseNodeClassifier):
    """Mean-aggregation GraphSAGE returning one fraud logit per node."""

    def __init__(self, config: GNNConfig) -> None:
        super().__init__(config)
        channels = (
            [config.in_channels]
            + [config.hidden_channels] * (config.num_layers - 1)
            + [1]
        )
        self.convs = nn.ModuleList(
            SAGEConv(
                channels[index],
                channels[index + 1],
                aggr="mean",
                normalize=False,
                root_weight=True,
            )
            for index in range(config.num_layers)
        )

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_type: Tensor | None = None
    ) -> Tensor:
        self._validate_inputs(x, edge_index)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.config.dropout, training=self.training)
        logits = self.convs[-1](x, edge_index)
        return logits.squeeze(-1)
