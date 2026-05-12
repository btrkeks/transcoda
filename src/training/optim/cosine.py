"""Cosine LR schedulers with per-group floors.

PyTorch's ``CosineAnnealingLR`` / ``CosineAnnealingWarmRestarts`` accept only a
scalar ``eta_min``, applied identically to every parameter group. With
layer-wise learning-rate decay (LLRD) the param groups can have very different
initial LRs, so a single global ``eta_min`` is wrong: groups whose initial LR
is below the global floor would have their LR rise during cosine decay.

The fix is to give each group a floor that is the same fraction of its own
initial LR. The cosine formula

    lr(t) = eta_min + (base_lr - eta_min) * (1 + cos(pi * t / T)) / 2

with ``eta_min = base_lr * f`` factors as

    lr(t) = base_lr * (f + (1 - f) * (1 + cos(pi * t / T)) / 2)

i.e. a shared multiplier applied to each group's ``base_lr`` — exactly what
``torch.optim.lr_scheduler.LambdaLR`` does.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _cosine_multiplier(progress: float, eta_min_factor: float) -> float:
    """Cosine multiplier on [0, 1]: 1.0 at progress=0, ``eta_min_factor`` at progress=1."""
    f = eta_min_factor
    return f + (1.0 - f) * (1.0 + math.cos(math.pi * progress)) / 2.0


def make_cosine_annealing_lambda_lr(
    optimizer: Optimizer,
    *,
    T_max: int,
    eta_min_factor: float,
) -> LambdaLR:
    """Single-cycle cosine annealing with per-group floor at ``base_lr * eta_min_factor``."""
    if T_max <= 0:
        raise ValueError(f"T_max must be positive, got {T_max}")
    t_max = int(T_max)

    def lr_lambda(step: int) -> float:
        progress = min(1.0, step / t_max)
        return _cosine_multiplier(progress, eta_min_factor)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def make_cosine_warm_restarts_lambda_lr(
    optimizer: Optimizer,
    *,
    T_0: int,
    T_mult: int,
    eta_min_factor: float,
) -> LambdaLR:
    """Cosine annealing with warm restarts; per-group floor at ``base_lr * eta_min_factor``.

    Mirrors the cycle logic of ``CosineAnnealingWarmRestarts``: cycle ``i`` has
    length ``T_0 * T_mult ** i``.
    """
    if T_0 <= 0:
        raise ValueError(f"T_0 must be positive, got {T_0}")
    if T_mult < 1:
        raise ValueError(f"T_mult must be >= 1, got {T_mult}")

    def lr_lambda(step: int) -> float:
        t_cur = step
        t_i = T_0
        while t_cur >= t_i:
            t_cur -= t_i
            t_i *= T_mult
        progress = t_cur / t_i
        return _cosine_multiplier(progress, eta_min_factor)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
