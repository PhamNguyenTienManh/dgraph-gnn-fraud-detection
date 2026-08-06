"""Integrity checks for canonical DGraphFin arrays."""

from dataclasses import dataclass, field

import numpy as np

from .loader import DGraphDataset
from .schema import FEATURE_DIMENSION


@dataclass(slots=True)
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, object]:
        return {"is_valid": self.is_valid, "errors": self.errors, "warnings": self.warnings}


def _validate_split_indices(
    name: str, indices: np.ndarray, num_nodes: int, seen: np.ndarray, report: ValidationReport
) -> None:
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        report.errors.append(f"{name} phải là mảng chỉ số nguyên một chiều")
        return
    if indices.size == 0:
        report.errors.append(f"{name} không được rỗng")
        return
    if int(indices.min()) < 0 or int(indices.max()) >= num_nodes:
        report.errors.append(f"{name} chứa node ID ngoài miền [0, {num_nodes})")
        return
    unique = np.unique(indices)
    if unique.size != indices.size:
        report.errors.append(f"{name} chứa node ID trùng lặp")
    if np.any(seen[unique]):
        report.errors.append(f"{name} chồng lấn với split đã kiểm tra trước đó")
    seen[unique] = True


def validate_dataset(dataset: DGraphDataset, *, feature_chunk_size: int = 250_000) -> ValidationReport:
    """Validate shapes, ranges, split isolation, labels, and finite features."""

    report = ValidationReport()
    n, e = dataset.num_nodes, dataset.num_edges

    if dataset.x.ndim != 2 or dataset.x.shape[1] != FEATURE_DIMENSION:
        report.errors.append(f"x phải có shape (N, {FEATURE_DIMENSION}), nhận {dataset.x.shape}")
    if dataset.y.shape != (n,):
        report.errors.append(f"y phải có shape ({n},), nhận {dataset.y.shape}")
    if dataset.edge_index.ndim != 2 or dataset.edge_index.shape[1] != 2:
        report.errors.append(f"edge_index phải có shape (E, 2), nhận {dataset.edge_index.shape}")
    if dataset.edge_type.shape != (e,):
        report.errors.append(f"edge_type phải có shape ({e},), nhận {dataset.edge_type.shape}")
    if dataset.edge_timestamp.shape != (e,):
        report.errors.append(f"edge_timestamp phải có shape ({e},), nhận {dataset.edge_timestamp.shape}")

    for start in range(0, n, feature_chunk_size):
        if not np.isfinite(dataset.x[start : start + feature_chunk_size]).all():
            report.errors.append("x chứa NaN hoặc Inf")
            break

    labels = np.unique(dataset.y)
    if not np.isin(labels, [0, 1, 2, 3]).all():
        report.errors.append(f"y chứa nhãn ngoài miền 0..3: {labels.tolist()}")

    if e:
        edge_min, edge_max = int(dataset.edge_index.min()), int(dataset.edge_index.max())
        if edge_min < 0 or edge_max >= n:
            report.errors.append(f"edge_index chứa node ID ngoài miền [0, {n})")
        if int(dataset.edge_type.min()) < 0:
            report.errors.append("edge_type chứa giá trị âm")

    seen = np.zeros(n, dtype=np.bool_)
    for name in ("train_mask", "valid_mask", "test_mask"):
        _validate_split_indices(name, getattr(dataset, name), n, seen, report)

    split_indices = np.concatenate((dataset.train_mask, dataset.valid_mask, dataset.test_mask))
    if split_indices.size and not np.isin(dataset.y[split_indices], [0, 1]).all():
        report.errors.append("Split dự đoán chứa node không thuộc lớp 0/1")
    labeled_count = int(np.count_nonzero(np.isin(dataset.y, [0, 1])))
    if int(seen.sum()) != labeled_count:
        report.warnings.append(
            f"Ba split bao phủ {int(seen.sum())}/{labeled_count} node thuộc lớp 0/1"
        )

    return report

# x                 : (N, 17), toàn bộ giá trị hữu hạn
# y                 : (N,), chỉ chứa 0/1/2/3
# edge_index        : (E, 2), node ID thuộc [0, N)
# edge_type         : (E,), không âm
# edge_timestamp    : (E,)
# train/valid/test  : 1D, không rỗng, không trùng, không overlap
# split labels      : chỉ chứa 0/1
# split coverage    : nên bao phủ toàn bộ node 0/1