"""End-to-end orchestration and persistent run logging for Sprint 2."""

from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any

import psutil
import torch
from torch import nn
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected

from dgraph_fraud.data.features import standardize_features, transform_features
from dgraph_fraud.data.graph import (
    build_graph_arrays,
    build_target_background_edge_type,
    to_pyg_data,
)
from dgraph_fraud.data.loader import load_dgraphfin
from dgraph_fraud.models import build_model

from .config import ExperimentConfig
from .reproducibility import environment_versions, set_seed
from .trainer import BaselineTrainer, TrainingResult


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fixed_random_subset(indices: torch.Tensor, limit: int | None, seed: int) -> torch.Tensor:
    """Choose one reproducible subset without depending on split index ordering."""
    if limit is None or indices.numel() <= limit:
        return indices
    generator = torch.Generator().manual_seed(seed)
    positions = torch.randperm(indices.numel(), generator=generator)[:limit]
    return indices[positions]


def _build_loaders(data, config: ExperimentConfig, seed: int):
    common = {
        "data": data,
        "num_neighbors": list(config.sampling.num_neighbors),
        "batch_size": config.sampling.batch_size,
        "num_workers": config.sampling.num_workers,
        "subgraph_type": "directional",
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = NeighborLoader(
        input_nodes=data.train_idx,
        shuffle=True,
        generator=generator,
        **common,
    )
    eval_limit = (
        None
        if config.training.max_eval_batches is None
        else config.training.max_eval_batches * config.sampling.batch_size
    )
    # Partial evaluations use exactly the same nodes for every model and seed.
    evaluation_seed = config.seeds[0]
    valid_nodes = _fixed_random_subset(data.valid_idx, eval_limit, evaluation_seed + 10_000)
    test_nodes = _fixed_random_subset(data.test_idx, eval_limit, evaluation_seed + 20_000)
    valid_loader = NeighborLoader(input_nodes=valid_nodes, shuffle=False, **common)
    test_loader = NeighborLoader(input_nodes=test_nodes, shuffle=False, **common)
    return train_loader, valid_loader, test_loader


def _result_payload(
    result: TrainingResult,
    *,
    model_name: str,
    seed: int,
    parameter_count: int,
    positive_weight: float,
    process_rss_mib_at_end: float,
    run_elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "seed": seed,
        "parameter_count": parameter_count,
        "positive_weight": positive_weight,
        "best_epoch": result.best_epoch,
        "epochs_completed": result.epochs_completed,
        "stopped_early": result.stopped_early,
        "validation": result.best_validation.as_dict(),
        "test": result.test.as_dict(),
        "history": [record.as_dict() for record in result.history],
        "process_rss_mib_at_end": process_rss_mib_at_end,
        "run_elapsed_seconds": run_elapsed_seconds,
    }


def _aggregate(results: list[dict[str, Any]], split: str, metric: str) -> dict[str, float]:
    values = [float(result[split][metric]) for result in results]
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def run_experiment(config: ExperimentConfig) -> Path:
    started = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_root) / f"{config.run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    _json_dump(run_dir / "config.json", config.as_dict())

    data_path = Path(config.data_path)
    dataset = load_dgraphfin(data_path)
    transformed_x = transform_features(dataset.x, config.feature_mode)
    if config.feature_standardization == "global_zscore":
        standardized_x = standardize_features(transformed_x)
        del transformed_x
        transformed_x = standardized_x
    graph = replace(build_graph_arrays(dataset), x=transformed_x)
    pyg_data = to_pyg_data(graph)
    del dataset, graph, transformed_x
    if config.graph_direction == "undirected":
        pyg_data.edge_index, edge_attributes = to_undirected(
            pyg_data.edge_index,
            edge_attr=[pyg_data.edge_type, pyg_data.edge_timestamp],
            num_nodes=pyg_data.num_nodes,
            reduce="min",
        )
        pyg_data.edge_type, pyg_data.edge_timestamp = edge_attributes
        pyg_data.directed = False
    if config.relation_mode == "target_background":
        pyg_data.edge_type = build_target_background_edge_type(
            pyg_data.y, pyg_data.edge_index
        )
    unique_relations, relation_counts = torch.unique(
        pyg_data.edge_type, return_counts=True
    )
    relation_type_counts = {
        str(int(relation)): int(count)
        for relation, count in zip(unique_relations, relation_counts, strict=True)
    }

    train_labels = pyg_data.y[pyg_data.train_idx]
    negative_count = int((train_labels == 0).sum())
    positive_count = int((train_labels == 1).sum())
    if positive_count == 0:
        raise ValueError("Train split không có fraud node")
    positive_weight = negative_count / positive_count if config.training.use_pos_weight else 1.0

    process = psutil.Process()
    all_results: dict[str, list[dict[str, Any]]] = {name: [] for name in config.models}
    for model_name in config.models:
        for seed in config.seeds:
            run_started = time.perf_counter()
            set_seed(seed)
            model = build_model(model_name, config.model)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(positive_weight, dtype=torch.float32)
            )
            train_loader, valid_loader, test_loader = _build_loaders(pyg_data, config, seed)
            trainer = BaselineTrainer(
                model,
                optimizer,
                criterion,
                config.training,
                seed=seed,
                device=torch.device(config.device),
            )
            result = trainer.fit(train_loader, valid_loader, test_loader)
            model_dir = run_dir / f"{model_name}_seed{seed}"
            model_dir.mkdir()
            checkpoint = {
                "model": model_name,
                "seed": seed,
                "model_config": config.as_dict()["model"],
                "best_epoch": result.best_epoch,
                "best_validation_ap": result.best_validation.average_precision,
                "state_dict": result.best_state_dict,
            }
            torch.save(checkpoint, model_dir / "best.pt")
            payload = _result_payload(
                result,
                model_name=model_name,
                seed=seed,
                parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                positive_weight=positive_weight,
                process_rss_mib_at_end=process.memory_info().rss / 1024 / 1024,
                run_elapsed_seconds=time.perf_counter() - run_started,
            )
            _json_dump(model_dir / "metrics.json", payload)
            all_results[model_name].append(payload)

    comparison: dict[str, Any] = {
        "run_name": config.run_name,
        "mode": config.mode,
        "is_partial_evaluation": config.training.max_eval_batches is not None,
        "partial_evaluation_subset": {
            "method": "fixed_random_without_replacement",
            "seed": config.seeds[0],
            "maximum_nodes_per_split": (
                None
                if config.training.max_eval_batches is None
                else config.training.max_eval_batches * config.sampling.batch_size
            ),
        },
        "selection_metric": "validation.average_precision",
        "test_used_for_selection": False,
        "relation_mode": config.relation_mode,
        "feature_mode": config.feature_mode,
        "feature_standardization": config.feature_standardization,
        "graph_direction": config.graph_direction,
        "relation_type_counts": relation_type_counts,
        "data_sha256": _sha256(data_path),
        "environment": environment_versions(),
        "elapsed_seconds": time.perf_counter() - started,
        "models": {},
    }
    for model_name, results in all_results.items():
        comparison["models"][model_name] = {
            "runs": results,
            "aggregate": {
                "validation_roc_auc": _aggregate(results, "validation", "roc_auc"),
                "validation_average_precision": _aggregate(
                    results, "validation", "average_precision"
                ),
                "test_roc_auc": _aggregate(results, "test", "roc_auc"),
                "test_average_precision": _aggregate(results, "test", "average_precision"),
            },
        }
    _json_dump(run_dir / "comparison.json", comparison)
    return run_dir
