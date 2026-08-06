"""Schema contract for the DGraphFin NPZ file."""

from dataclasses import dataclass


REQUIRED_KEYS = (
    "x",
    "y",
    "edge_index",
    "edge_type",
    "edge_timestamp",
    "train_mask",
    "valid_mask",
    "test_mask",
)

PREDICTION_LABELS = (0, 1)
BACKGROUND_LABELS = (2, 3)
POSITIVE_LABEL = 1
FEATURE_DIMENSION = 17


@dataclass(frozen=True)
class DatasetContract:
    """Stable facts and modeling policy for DGraphFin."""

    directed: bool = True
    feature_dimension: int = FEATURE_DIMENSION
    prediction_labels: tuple[int, int] = PREDICTION_LABELS
    background_labels: tuple[int, int] = BACKGROUND_LABELS
    positive_label: int = POSITIVE_LABEL
    split_fields_are_indices: bool = True
