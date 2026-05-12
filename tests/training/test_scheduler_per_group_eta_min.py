"""Tests for cosine schedulers with per-group floors.

The schedulers in ``src.training.optim.cosine`` apply a single shared
multiplier to each param group's ``base_lr``, so each group decays toward its
own floor at ``base_lr * eta_min_factor``. This is the property we rely on for
LLRD: a group whose initial LR is much smaller than another group should
*not* have its LR rise during cosine decay (which is what would happen with
PyTorch's stock ``CosineAnnealingLR(eta_min=<global float>)``).
"""

from __future__ import annotations

import math

import pytest
import torch

from src.training.optim.cosine import (
    make_cosine_annealing_lambda_lr,
    make_cosine_warm_restarts_lambda_lr,
)


def _build_two_group_optimizer() -> torch.optim.Optimizer:
    p_big = torch.nn.Parameter(torch.zeros(1))
    p_small = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD(
        [
            {"params": [p_big], "lr": 1e-3, "name": "big"},
            {"params": [p_small], "lr": 1e-5, "name": "small"},
        ],
        lr=1e-3,
    )


def test_cosine_annealing_floors_each_group_at_its_own_eta_min():
    optimizer = _build_two_group_optimizer()
    eta_min_factor = 0.1
    t_max = 100
    scheduler = make_cosine_annealing_lambda_lr(
        optimizer, T_max=t_max, eta_min_factor=eta_min_factor
    )

    # Initial step (epoch 0): full base_lr.
    big_initial = optimizer.param_groups[0]["lr"]
    small_initial = optimizer.param_groups[1]["lr"]
    assert math.isclose(big_initial, 1e-3, rel_tol=1e-12)
    assert math.isclose(small_initial, 1e-5, rel_tol=1e-12)

    big_lrs = [big_initial]
    small_lrs = [small_initial]
    for _ in range(t_max):
        scheduler.step()
        big_lrs.append(optimizer.param_groups[0]["lr"])
        small_lrs.append(optimizer.param_groups[1]["lr"])

    # Both groups must be monotonically non-increasing through the cosine
    # phase. This is the property that fails with a global `eta_min` when
    # group_initial_lr < eta_min (the small group would heat up).
    for prev, curr in zip(big_lrs, big_lrs[1:]):
        assert curr <= prev + 1e-12
    for prev, curr in zip(small_lrs, small_lrs[1:]):
        assert curr <= prev + 1e-12

    # Final step (t = t_max): each group lands on its own floor.
    assert math.isclose(big_lrs[-1], 1e-3 * eta_min_factor, rel_tol=1e-9)
    assert math.isclose(small_lrs[-1], 1e-5 * eta_min_factor, rel_tol=1e-9)

    # Ratios across groups are preserved across the schedule.
    for big, small in zip(big_lrs, small_lrs):
        assert math.isclose(small / big, 1e-5 / 1e-3, rel_tol=1e-9)


def test_cosine_warm_restarts_resets_each_cycle_per_group():
    optimizer = _build_two_group_optimizer()
    eta_min_factor = 0.0
    T_0 = 10
    T_mult = 2
    scheduler = make_cosine_warm_restarts_lambda_lr(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min_factor=eta_min_factor
    )

    # Step through enough to cross at least one restart boundary.
    seen_after_restart = False
    for step in range(1, T_0 * 3):
        scheduler.step()
        big = optimizer.param_groups[0]["lr"]
        small = optimizer.param_groups[1]["lr"]
        # Ratio is preserved at every step.
        assert math.isclose(small / big, 1e-5 / 1e-3, rel_tol=1e-9)
        if step == T_0:
            # Right at the restart boundary, both groups jump back near base_lr.
            assert big > 0.9 * 1e-3
            assert small > 0.9 * 1e-5
            seen_after_restart = True
    assert seen_after_restart


@pytest.mark.parametrize("invalid_T_max", [0, -1])
def test_cosine_annealing_rejects_non_positive_T_max(invalid_T_max):
    optimizer = _build_two_group_optimizer()
    with pytest.raises(ValueError):
        make_cosine_annealing_lambda_lr(
            optimizer, T_max=invalid_T_max, eta_min_factor=0.1
        )


def test_cosine_warm_restarts_validates_args():
    optimizer = _build_two_group_optimizer()
    with pytest.raises(ValueError):
        make_cosine_warm_restarts_lambda_lr(
            optimizer, T_0=0, T_mult=2, eta_min_factor=0.1
        )
    with pytest.raises(ValueError):
        make_cosine_warm_restarts_lambda_lr(
            optimizer, T_0=10, T_mult=0, eta_min_factor=0.1
        )
