"""Validated configuration shared by node-classification GNNs."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GNNConfig:
    """Architecture-only configuration for a binary GNN baseline."""

    in_channels: int = 17
    hidden_channels: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    num_relations: int | None = None

    def __post_init__(self) -> None:
        if self.in_channels <= 0:
            raise ValueError("in_channels phải là số nguyên dương")
        if self.hidden_channels <= 0:
            raise ValueError("hidden_channels phải là số nguyên dương")
        if self.num_layers < 2:
            raise ValueError("num_layers phải từ 2 trở lên")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout phải thuộc khoảng [0, 1)")
        if self.num_relations is not None and self.num_relations <= 0:
            raise ValueError("num_relations phải là số nguyên dương hoặc null")
