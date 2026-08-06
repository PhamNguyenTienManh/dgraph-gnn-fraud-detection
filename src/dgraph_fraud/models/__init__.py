"""GNN baselines for binary node classification."""

from .config import GNNConfig
from .factory import build_model
from .gcn import GCN
from .graphsage import GraphSAGE
from .rgcn import RGCN

__all__ = ["GCN", "GraphSAGE", "RGCN", "GNNConfig", "build_model"]
