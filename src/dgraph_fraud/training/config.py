"""Validated JSON configuration for full baseline experiments."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from dgraph_fraud.models import GNNConfig


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    num_neighbors: tuple[int, ...]
    batch_size: int
    num_workers: int = 0

    def __post_init__(self) -> None:
        if not self.num_neighbors or any(value <= 0 for value in self.num_neighbors):
            raise ValueError("num_neighbors phải gồm các số nguyên dương")
        if self.batch_size <= 0:
            raise ValueError("batch_size phải dương")
        if self.num_workers < 0:
            raise ValueError("num_workers không được âm")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    use_pos_weight: bool = True
    gradient_clip_norm: float | None = 2.0
    max_train_batches: int | None = None
    max_eval_batches: int | None = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs phải dương")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate phải dương")
        if self.weight_decay < 0:
            raise ValueError("weight_decay không được âm")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience phải dương")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm phải dương hoặc null")
        for name in ("max_train_batches", "max_eval_batches"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} phải dương hoặc null")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    run_name: str
    mode: str
    data_path: str
    output_root: str
    device: str
    feature_mode: str
    feature_standardization: str
    graph_direction: str
    relation_mode: str
    seeds: tuple[int, ...]
    models: tuple[str, ...]
    model: GNNConfig
    sampling: SamplingConfig
    training: TrainingConfig

    def __post_init__(self) -> None:
        if self.mode != "full":
            raise ValueError("mode phải là 'full'")
        if self.device != "cpu":
            raise ValueError("Môi trường hiện tại chỉ hỗ trợ device='cpu'")
        if self.feature_mode not in {"raw", "zero_indicator"}:
            raise ValueError("feature_mode phải là 'raw' hoặc 'zero_indicator'")
        if self.feature_standardization not in {"none", "global_zscore"}:
            raise ValueError(
                "feature_standardization must be 'none' or 'global_zscore'"
            )
        if self.graph_direction not in {"directed", "undirected"}:
            raise ValueError("graph_direction must be 'directed' or 'undirected'")
        if not self.seeds:
            raise ValueError("Cần ít nhất một seed")
        if not self.models or any(
            name not in {"gcn", "graphsage", "rgcn"} for name in self.models
        ):
            raise ValueError("models chỉ hỗ trợ 'gcn', 'graphsage' và 'rgcn'")
        if self.relation_mode not in {"original", "target_background"}:
            raise ValueError("relation_mode phải là 'original' hoặc 'target_background'")
        if "rgcn" in self.models:
            if self.relation_mode != "target_background":
                raise ValueError("RGCN paper-style yêu cầu relation_mode='target_background'")
            if self.model.num_relations != 4:
                raise ValueError("RGCN paper-style yêu cầu num_relations=4")
        elif self.relation_mode != "original":
            raise ValueError("target_background chỉ dùng cho RGCN")
        if self.model.num_layers != len(self.sampling.num_neighbors):
            raise ValueError("Số GNN layer phải bằng số hop trong num_neighbors")
        expected_channels = 17 if self.feature_mode == "raw" else 34
        if self.model.in_channels != expected_channels:
            raise ValueError(
                f"feature_mode={self.feature_mode!r} yêu cầu in_channels={expected_channels}"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return ExperimentConfig(
        run_name=str(payload["run_name"]),
        mode=str(payload["mode"]),
        data_path=str(payload["data_path"]),
        output_root=str(payload["output_root"]),
        device=str(payload.get("device", "cpu")),
        feature_mode=str(payload["feature_mode"]),
        feature_standardization=str(payload.get("feature_standardization", "none")),
        graph_direction=str(payload.get("graph_direction", "directed")),
        relation_mode=str(payload.get("relation_mode", "original")),
        seeds=tuple(int(seed) for seed in payload["seeds"]),
        models=tuple(str(name).lower() for name in payload["models"]),
        model=GNNConfig(**payload["model"]),
        sampling=SamplingConfig(
            num_neighbors=tuple(int(value) for value in payload["sampling"]["num_neighbors"]),
            batch_size=int(payload["sampling"]["batch_size"]),
            num_workers=int(payload["sampling"].get("num_workers", 0)),
        ),
        training=TrainingConfig(**payload["training"]),
    )
