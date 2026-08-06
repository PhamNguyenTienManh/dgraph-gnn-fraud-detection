"""Paper-style RGCN using target/background endpoint relations."""

import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import RGCNConv

from .base import BaseNodeClassifier
from .config import GNNConfig


class RGCN(BaseNodeClassifier):
    """Relation-aware GCN returning one fraud logit per node.

    Relation IDs encode the four directed endpoint combinations T->T, T->B,
    B->T and B->B. The hidden width is configured independently so the model
    can be parameter-matched to the homogeneous GCN baseline.
    """

    def __init__(self, config: GNNConfig) -> None:
        super().__init__(config)
        if config.num_relations is None:
            raise ValueError("RGCN yêu cầu num_relations")
        channels = (
            [config.in_channels]
            + [config.hidden_channels] * (config.num_layers - 1)
            + [1]
        )
        self.convs = nn.ModuleList(
            RGCNConv(
                channels[index],
                channels[index + 1],
                num_relations=config.num_relations,
                root_weight=True,
                bias=True,
            )
            for index in range(config.num_layers)
        )

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_type: Tensor | None = None
    ) -> Tensor:
        self._validate_inputs(x, edge_index)
        if edge_type is None:
            raise ValueError("RGCN yêu cầu edge_type")
        if edge_type.ndim != 1 or edge_type.numel() != edge_index.shape[1]:
            raise ValueError("edge_type phải có shape [E] khớp với edge_index")
        if edge_type.dtype != edge_index.dtype:
            raise TypeError("edge_type phải có dtype torch.long")
        if edge_type.numel() and (
            int(edge_type.min()) < 0
            or int(edge_type.max()) >= int(self.config.num_relations)
        ):
            raise ValueError("edge_type nằm ngoài miền relation đã cấu hình")

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.config.dropout, training=self.training)
        logits = self.convs[-1](x, edge_index, edge_type)
        return logits.squeeze(-1)
