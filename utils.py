import random
import numpy as np
import torch
import os


# ----------------------------------------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Method to allow global reproducibility across Python, NumPy, and PyTorch (CPU/CUDA/MPS)."""

    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed=seed)

    # PyTorch (CPU)
    torch.manual_seed(seed=seed)

    # PyTorch (GPU)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)

    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------------------------------------------------
def seed_worker(
        worker_id: int
) -> None:
    """Seed worker by worker ID."""

    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
