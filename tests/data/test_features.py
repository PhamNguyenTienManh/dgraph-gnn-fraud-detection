import numpy as np
import pytest

from dgraph_fraud.data.features import standardize_features, transform_features


def test_zero_indicator_replaces_missing_and_appends_flags():
    x = np.array([[-1.0, 2.0], [3.0, -1.0]], dtype=np.float32)

    transformed = transform_features(x, "zero_indicator")

    np.testing.assert_array_equal(
        transformed,
        np.array([[0.0, 2.0, 1.0, 0.0], [3.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    )


def test_transform_rejects_unknown_mode():
    with pytest.raises(ValueError):
        transform_features(np.ones((2, 2), dtype=np.float32), "mean")


def test_standardize_features_uses_column_zscore_and_handles_constants():
    x = np.array([[1.0, 5.0], [3.0, 5.0]], dtype=np.float32)

    standardized = standardize_features(x)

    np.testing.assert_allclose(
        standardized,
        np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(x, np.array([[1.0, 5.0], [3.0, 5.0]]))
