"""Binary fraud metrics with explicit input validation."""

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    roc_auc: float
    average_precision: float
    sample_count: int
    positive_count: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> BinaryMetrics:
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    y_score = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y_true.shape != y_score.shape:
        raise ValueError("labels và probabilities phải có cùng shape")
    if y_true.size == 0:
        raise ValueError("Không thể tính metric trên tập rỗng")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Metric chỉ nhận nhãn nhị phân 0/1")
    if np.unique(y_true).size != 2:
        raise ValueError("ROC-AUC/AP yêu cầu cả lớp 0 và lớp 1 trong tập đánh giá")
    if not np.isfinite(y_score).all():
        raise ValueError("probabilities chứa NaN hoặc Inf")
    return BinaryMetrics(
        roc_auc=float(roc_auc_score(y_true, y_score)),
        average_precision=float(average_precision_score(y_true, y_score)),
        sample_count=int(y_true.size),
        positive_count=int(y_true.sum()),
    )
