"""Small model factory used by future experiment configuration."""

from typing import Literal

from .base import BaseNodeClassifier
from .config import GNNConfig
from .gcn import GCN
from .graphsage import GraphSAGE
from .rgcn import RGCN

ModelName = Literal["gcn", "graphsage", "rgcn"]


def build_model(name: ModelName | str, config: GNNConfig) -> BaseNodeClassifier:
    """Build a supported Sprint 2 baseline by its stable lowercase name."""

    normalized_name = name.strip().lower()
    if normalized_name == "gcn":
        return GCN(config)
    if normalized_name in {"graphsage", "sage"}:
        return GraphSAGE(config)
    if normalized_name == "rgcn":
        return RGCN(config)
    raise ValueError(
        f"Model không được hỗ trợ: {name!r}. Chọn 'gcn', 'graphsage' hoặc 'rgcn'"
    )
