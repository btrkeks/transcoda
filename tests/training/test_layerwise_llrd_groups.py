"""Tests for build_llrd_param_groups_for_convnextv2.

Uses a synthetic module that mirrors ConvNeXtV2's parameter-name structure
(``embeddings.*`` and ``encoder.stages.<i>.*``). This avoids downloading
weights while still exercising the real stage-mapping and LR-scaling logic.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from src.training.optim.layerwise import build_llrd_param_groups_for_convnextv2


class _Embeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 8, kernel_size=2, bias=True)
        self.norm = nn.LayerNorm(8)


class _Block(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(c, c, kernel_size=1, groups=c, bias=True)
        self.norm = nn.LayerNorm(c)
        self.pwconv1 = nn.Linear(c, c * 2, bias=True)
        # The layerwise utility treats any name containing 'grn' as no-decay.
        self.grn = nn.Parameter(torch.zeros(c * 2))
        self.pwconv2 = nn.Linear(c * 2, c, bias=True)


class _InnerEncoder(nn.Module):
    def __init__(self, num_stages: int = 4) -> None:
        super().__init__()
        self.stages = nn.ModuleList([_Block(8) for _ in range(num_stages)])


class _SyntheticConvNextV2(nn.Module):
    """Mirrors HuggingFace ConvNeXtV2Model attribute layout."""

    def __init__(self, num_stages: int = 4) -> None:
        super().__init__()
        self.embeddings = _Embeddings()
        self.encoder = _InnerEncoder(num_stages=num_stages)


def _group_by_name(groups: list[dict]) -> dict[str, dict]:
    return {g["name"]: g for g in groups}


def test_llrd_groups_have_expected_lr_ratios():
    encoder = _SyntheticConvNextV2()
    base_lr = 1e-4
    gamma = 0.6

    groups = build_llrd_param_groups_for_convnextv2(
        encoder=encoder,
        base_encoder_lr=base_lr,
        weight_decay=0.05,
        gamma=gamma,
        name_prefix="encoder",
    )

    by_name = _group_by_name(groups)

    num_stages = 4
    expected_stages = list(range(num_stages + 1))  # 0 (embeddings) + 1..4
    for stage_id in expected_stages:
        decay_name = f"encoder.stage{stage_id}.decay"
        no_decay_name = f"encoder.stage{stage_id}.no_decay"
        assert decay_name in by_name, f"missing group {decay_name}"
        assert no_decay_name in by_name, f"missing group {no_decay_name}"

        expected_lr = base_lr * (gamma ** (num_stages - stage_id))
        assert math.isclose(by_name[decay_name]["lr"], expected_lr, rel_tol=1e-9)
        assert math.isclose(by_name[no_decay_name]["lr"], expected_lr, rel_tol=1e-9)

    # Deepest stage matches the base encoder LR exactly.
    assert math.isclose(by_name[f"encoder.stage{num_stages}.decay"]["lr"], base_lr, rel_tol=1e-12)


def test_llrd_groups_split_decay_and_no_decay_correctly():
    encoder = _SyntheticConvNextV2()
    weight_decay = 0.05

    groups = build_llrd_param_groups_for_convnextv2(
        encoder=encoder,
        base_encoder_lr=1e-4,
        weight_decay=weight_decay,
        gamma=0.6,
    )

    decay_ids: set[int] = set()
    no_decay_ids: set[int] = set()
    for g in groups:
        if g["name"].endswith(".decay"):
            assert g["weight_decay"] == weight_decay
            decay_ids.update(id(p) for p in g["params"])
        else:
            assert g["name"].endswith(".no_decay")
            assert g["weight_decay"] == 0.0
            no_decay_ids.update(id(p) for p in g["params"])

    # decay and no_decay are disjoint
    assert decay_ids.isdisjoint(no_decay_ids)

    # Spot checks: norms / biases / grn -> no_decay; conv & linear weights -> decay.
    block0 = encoder.encoder.stages[0]
    assert id(block0.dwconv.weight) in decay_ids
    assert id(block0.dwconv.bias) in no_decay_ids
    assert id(block0.norm.weight) in no_decay_ids  # norm name routes to no_decay
    assert id(block0.norm.bias) in no_decay_ids
    assert id(block0.pwconv1.weight) in decay_ids
    assert id(block0.pwconv1.bias) in no_decay_ids
    assert id(block0.grn) in no_decay_ids
    assert id(encoder.embeddings.patch_embed.weight) in decay_ids
    assert id(encoder.embeddings.patch_embed.bias) in no_decay_ids


def test_llrd_groups_assign_params_to_correct_stage():
    encoder = _SyntheticConvNextV2()

    groups = build_llrd_param_groups_for_convnextv2(
        encoder=encoder,
        base_encoder_lr=1e-4,
        weight_decay=0.05,
        gamma=0.6,
    )

    by_name = _group_by_name(groups)

    # Embeddings -> stage 0
    stage0_ids = {
        id(p)
        for g in groups
        if g["name"].startswith("encoder.stage0.")
        for p in g["params"]
    }
    for _, p in encoder.embeddings.named_parameters():
        assert id(p) in stage0_ids

    # encoder.stages[i] -> stage i+1
    for i, stage in enumerate(encoder.encoder.stages):
        stage_id = i + 1
        decay = by_name[f"encoder.stage{stage_id}.decay"]["params"]
        no_decay = by_name[f"encoder.stage{stage_id}.no_decay"]["params"]
        ids_in_group = {id(p) for p in (*decay, *no_decay)}
        for _, p in stage.named_parameters():
            assert id(p) in ids_in_group


def test_llrd_groups_exclude_frozen_parameters():
    encoder = _SyntheticConvNextV2()
    # Freeze all of stage 0 (embeddings) and the entire first encoder block.
    for p in encoder.embeddings.parameters():
        p.requires_grad = False
    for p in encoder.encoder.stages[0].parameters():
        p.requires_grad = False

    groups = build_llrd_param_groups_for_convnextv2(
        encoder=encoder,
        base_encoder_lr=1e-4,
        weight_decay=0.05,
        gamma=0.6,
    )

    by_name = _group_by_name(groups)
    # No groups should exist for stage 0 (embeddings) or stage 1 (encoder.stages[0])
    assert "encoder.stage0.decay" not in by_name
    assert "encoder.stage0.no_decay" not in by_name
    assert "encoder.stage1.decay" not in by_name
    assert "encoder.stage1.no_decay" not in by_name

    # Trainable param count must match
    total_in_groups = sum(
        sum(p.numel() for p in g["params"]) for g in groups
    )
    expected_trainable = sum(
        p.numel() for p in encoder.parameters() if p.requires_grad
    )
    assert total_in_groups == expected_trainable


def test_llrd_param_count_matches_encoder_trainable_total():
    encoder = _SyntheticConvNextV2()

    groups = build_llrd_param_groups_for_convnextv2(
        encoder=encoder,
        base_encoder_lr=1e-4,
        weight_decay=0.05,
        gamma=0.6,
    )

    total_in_groups = sum(
        sum(p.numel() for p in g["params"]) for g in groups
    )
    expected = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    assert total_in_groups == expected
