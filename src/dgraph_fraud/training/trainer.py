"""Training loop for GCN/GraphSAGE sampled-subgraph baselines."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import time
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from dgraph_fraud.models.base import BaseNodeClassifier

from .config import TrainingConfig
from .metrics import BinaryMetrics, compute_binary_metrics
from .reproducibility import set_seed


@dataclass(frozen=True, slots=True)
class EpochRecord:
    epoch: int
    train_loss: float
    train_batches: int
    train_seed_nodes: int
    validation: BinaryMetrics
    seconds: float

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["validation"] = self.validation.as_dict()
        return payload


@dataclass(frozen=True, slots=True)
class TrainingResult:
    best_epoch: int
    best_validation: BinaryMetrics
    test: BinaryMetrics
    epochs_completed: int
    stopped_early: bool
    history: tuple[EpochRecord, ...]
    best_state_dict: dict[str, Tensor]


class BaselineTrainer:
    """Train one model while selecting checkpoints only by validation AP."""

    def __init__(
        self,
        model: BaseNodeClassifier,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config: TrainingConfig,
        *,
        seed: int,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.seed = seed
        self.device = device

    def _train_epoch(self, loader: Iterable, epoch: int) -> tuple[float, int, int]:
        self.model.train()
        set_seed(self.seed + epoch)
        loss_sum = 0.0
        batch_count = 0
        seed_count_total = 0
        for batch_index, batch in enumerate(loader):
            if (
                self.config.max_train_batches is not None
                and batch_index >= self.config.max_train_batches
            ):
                break
            batch = batch.to(self.device)
            seed_count = int(batch.batch_size)
            targets = batch.y[:seed_count].to(torch.float32)
            if not torch.isin(targets.to(torch.int64), torch.tensor([0, 1], device=self.device)).all():
                raise ValueError("Train seed chứa nhãn ngoài 0/1")

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(
                batch.x, batch.edge_index, getattr(batch, "edge_type", None)
            )[:seed_count]
            loss = self.criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Loss không hữu hạn")
            loss.backward()
            if self.config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
            self.optimizer.step()

            loss_sum += float(loss.detach()) * seed_count
            seed_count_total += seed_count
            batch_count += 1
        if batch_count == 0 or seed_count_total == 0:
            raise RuntimeError("Không có train batch nào được xử lý")
        return loss_sum / seed_count_total, batch_count, seed_count_total

    @torch.no_grad()
    def evaluate(self, loader: Iterable, *, evaluation_seed: int) -> BinaryMetrics:
        self.model.eval()
        set_seed(evaluation_seed)
        labels: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        for batch_index, batch in enumerate(loader):
            if (
                self.config.max_eval_batches is not None
                and batch_index >= self.config.max_eval_batches
            ):
                break
            batch = batch.to(self.device)
            seed_count = int(batch.batch_size)
            logits = self.model(
                batch.x, batch.edge_index, getattr(batch, "edge_type", None)
            )[:seed_count]
            labels.append(batch.y[:seed_count].cpu().numpy())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
        if not labels:
            raise RuntimeError("Không có evaluation batch nào được xử lý")
        return compute_binary_metrics(np.concatenate(labels), np.concatenate(probabilities))

    def fit(self, train_loader: Iterable, valid_loader: Iterable, test_loader: Iterable) -> TrainingResult:
        history: list[EpochRecord] = []
        best_ap = float("-inf")
        best_epoch = 0
        best_validation: BinaryMetrics | None = None
        best_state: dict[str, Tensor] | None = None
        epochs_without_improvement = 0

        for epoch in range(1, self.config.epochs + 1):
            started = time.perf_counter()
            train_loss, train_batches, train_seed_nodes = self._train_epoch(train_loader, epoch)
            validation = self.evaluate(valid_loader, evaluation_seed=self.seed + 10_000)
            history.append(
                EpochRecord(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_batches=train_batches,
                    train_seed_nodes=train_seed_nodes,
                    validation=validation,
                    seconds=time.perf_counter() - started,
                )
            )
            if validation.average_precision > best_ap:
                best_ap = validation.average_precision
                best_epoch = epoch
                best_validation = validation
                best_state = deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self.config.early_stopping_patience:
                break

        if best_state is None or best_validation is None:
            raise RuntimeError("Không tạo được best checkpoint")
        self.model.load_state_dict(best_state)
        test_metrics = self.evaluate(test_loader, evaluation_seed=self.seed + 20_000)
        return TrainingResult(
            best_epoch=best_epoch,
            best_validation=best_validation,
            test=test_metrics,
            epochs_completed=len(history),
            stopped_early=len(history) < self.config.epochs,
            history=tuple(history),
            best_state_dict=best_state,
        )
