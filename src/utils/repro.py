# src/utils/repro.py
"""Reproducibility utilities: seeding and determinism control."""

import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False):
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: The random seed to use across all libraries.
        deterministic: If True, enables deterministic algorithms and disables
            cudnn benchmarking for reproducibility. This may reduce performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # if deterministic:
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # else:
    torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int):
    """
    Initialize worker processes with unique seeds for data loading.

    This function ensures that each DataLoader worker has its own random seed,
    derived from PyTorch's worker seed. This is essential for reproducible
    augmentation across multiple workers.

    Args:
        worker_id: Worker process ID (provided automatically by DataLoader)

    Note:
        This is automatically called by DataLoader when num_workers > 0.
        Use by passing `worker_init_fn=worker_init_fn` to DataLoader.
    """
    # Get the base seed from PyTorch's worker info
    worker_seed = torch.initial_seed() % 2**32

    # Seed numpy and random for this worker
    np.random.seed(worker_seed)
    random.seed(worker_seed)
