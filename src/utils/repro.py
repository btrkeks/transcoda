# src/utils/repro.py
"""Reproducibility utilities: seeding and determinism control."""

import random

import numpy as np
import torch
from lightning.pytorch import seed_everything as _pl_seed_everything


def seed_everything(seed: int, deterministic: bool = False):
    """Set random seeds for Python, NumPy, and PyTorch.

    Delegates to Lightning's seed_everything so each DDP rank receives a
    rank-aware seed and DataLoader workers get distinct seeds (workers=True).
    Without rank-aware seeding, every DDP subprocess would draw identical
    augmentations on identically-numbered workers.
    """
    _pl_seed_everything(seed, workers=True, verbose=False)
    # Lightning's seed_everything sets PL_GLOBAL_SEED and python/numpy/torch RNGs;
    # we still need to apply the optional cudnn benchmark policy.
    torch.backends.cudnn.benchmark = True
    # Defensive: ensure stdlib random and numpy are seeded even if a future
    # Lightning version drops one of them.
    random.seed(seed)
    np.random.seed(seed)


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
