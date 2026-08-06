"""Memory-conscious loader for DGraphFin."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .schema import REQUIRED_KEYS


@dataclass(slots=True)
class DGraphDataset:
    """In-memory canonical arrays.

    The three fields named ``*_mask`` in the source file contain node indices,
    not boolean masks. Their original names are retained for dataset fidelity.
    """

    x: NDArray[np.float32]
    y: NDArray[np.int64]
    edge_index: NDArray[np.int64]
    edge_type: NDArray[np.int64]
    edge_timestamp: NDArray[np.int64]
    train_mask: NDArray[np.int64]
    valid_mask: NDArray[np.int64]
    test_mask: NDArray[np.int64]

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[0])


def inspect_keys(path: str | Path) -> tuple[str, ...]:
    """Return NPZ keys without materializing all arrays."""

    with np.load(Path(path), allow_pickle=False) as archive:
        return tuple(archive.files)


def load_dgraphfin(path: str | Path, *, feature_dtype: str = "float32") -> DGraphDataset:
    """Load canonical DGraphFin arrays and reduce feature memory to float32."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dataset: {dataset_path}")
    if feature_dtype != "float32":
        raise ValueError("Sprint 1 chỉ hỗ trợ feature_dtype='float32'")

    with np.load(dataset_path, allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_KEYS) - set(archive.files))
        if missing:
            raise ValueError(f"Dataset thiếu các trường bắt buộc: {missing}")

        return DGraphDataset(
            x=np.asarray(archive["x"], dtype=np.float32),
            y=np.asarray(archive["y"], dtype=np.int64),
            edge_index=np.asarray(archive["edge_index"], dtype=np.int64),
            edge_type=np.asarray(archive["edge_type"], dtype=np.int64),
            edge_timestamp=np.asarray(archive["edge_timestamp"], dtype=np.int64),
            train_mask=np.asarray(archive["train_mask"], dtype=np.int64),
            valid_mask=np.asarray(archive["valid_mask"], dtype=np.int64),
            test_mask=np.asarray(archive["test_mask"], dtype=np.int64),
        )
