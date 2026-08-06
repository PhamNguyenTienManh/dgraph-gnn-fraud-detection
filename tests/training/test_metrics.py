import numpy as np
import pytest

from dgraph_fraud.training.metrics import compute_binary_metrics


def test_binary_metrics_match_known_example():
    metrics = compute_binary_metrics(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])
    )

    assert metrics.roc_auc == pytest.approx(0.75)
    assert metrics.average_precision == pytest.approx(5 / 6)
    assert metrics.positive_count == 2


def test_binary_metrics_require_both_classes():
    with pytest.raises(ValueError, match="cả lớp"):
        compute_binary_metrics(np.array([0, 0]), np.array([0.1, 0.2]))
