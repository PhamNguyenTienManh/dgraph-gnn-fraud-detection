"""Random seed and environment helpers."""

import platform
import random

import numpy as np
import sklearn
import torch
import torch_geometric


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def environment_versions() -> dict[str, str | bool]:
    try:
        import pyg_lib

        pyg_lib_version = pyg_lib.__version__
    except ImportError:
        pyg_lib_version = "not-installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "pyg_lib": pyg_lib_version,
        "scikit_learn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
