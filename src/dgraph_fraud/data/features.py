"""Leakage-free feature transforms for DGraphFin missing values."""

from typing import Literal

import numpy as np

FeatureMode = Literal["raw", "zero_indicator"]


def standardize_features(x: np.ndarray) -> np.ndarray:
    """Return a global per-column z-score transform in float32.

    Constant columns are only centered (their divisor is set to one), avoiding
    NaN/Inf values while keeping the transformation deterministic.
    """

    if x.ndim != 2:
        raise ValueError(f"x must have shape [N, F], got {x.shape}")
    source = np.asarray(x, dtype=np.float32)
    mean = source.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = source.std(axis=0, dtype=np.float64).astype(np.float32)
    safe_std = np.where(std > 0.0, std, 1.0).astype(np.float32, copy=False)
    standardized = source.copy()
    np.subtract(standardized, mean, out=standardized)
    np.divide(standardized, safe_std, out=standardized)
    return standardized


def transform_features(x: np.ndarray, mode: FeatureMode | str) -> np.ndarray:
    """Transform node features without learning statistics from any split.

    ``zero_indicator`` follows the DGraph paper's Trick B: replace the ``-1``
    missing sentinel with zero and append one binary missing flag per feature.
    """

    if x.ndim != 2:
        raise ValueError(f"x phải có shape [N, F], nhận {x.shape}")
    normalized_mode = mode.strip().lower()
    source = np.asarray(x, dtype=np.float32)
    if normalized_mode == "raw":
        return source
    if normalized_mode == "zero_indicator":
        missing = source == -1
        filled = np.where(missing, 0.0, source).astype(np.float32, copy=False)
        return np.concatenate((filled, missing.astype(np.float32)), axis=1)
    raise ValueError("feature mode phải là 'raw' hoặc 'zero_indicator'")
