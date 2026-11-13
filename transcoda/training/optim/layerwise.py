"""Layer-wise learning rate decay (LLRD) utilities for ConvNeXtV2 encoder.

This module provides functionality to create parameter groups with layer-wise learning rate decay,
where earlier layers (closer to input) receive lower learning rates than later layers (closer to task).

ConvNeXtV2 Stage Mapping:
    - Stage 0: embeddings (patch_embeddings)
    - Stage 1: encoder.stages[0]
    - Stage 2: encoder.stages[1]
    - Stage 3: encoder.stages[2]
    - Stage 4: encoder.stages[3]

LR Formula:
    lr_stage_i = base_encoder_lr × (gamma^(num_stages - stage_i))

    Where:
    - gamma is the decay factor (e.g., 0.75)
    - num_stages = 4 (the last stage ID)
    - Earlier stages get lower LR (embeddings gets base_lr × gamma^4)
    - Last stage gets base LR (stage 3 gets base_lr × gamma^0 = base_lr)

Example:
    With base_lr=1e-4 and gamma=0.75:
    - embeddings: 1e-4 × 0.75^4 ≈ 3.16e-5
    - stage 0: 1e-4 × 0.75^3 ≈ 4.22e-5
    - stage 1: 1e-4 × 0.75^2 ≈ 5.63e-5
    - stage 2: 1e-4 × 0.75^1 ≈ 7.50e-5
    - stage 3: 1e-4 × 0.75^0 = 1.00e-4
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import nn


# Hardcoded no-decay heuristics (standard practice from BERT/transformers)
_NO_DECAY_KEYWORDS = ("bias", "layernorm", "norm", "grn")


def _is_no_decay_param(param_name: str, param: torch.nn.Parameter) -> bool:
    """Check if a parameter should have zero weight decay.

    Args:
        param_name: Full parameter name (e.g., "encoder.stages.0.layers.0.conv.weight")
        param: The parameter tensor

    Returns:
        True if parameter should have weight_decay=0 (bias, norm layers, etc.)
    """
    if not param.requires_grad:
        return True

    # Check if name ends with .bias
    if param_name.endswith(".bias"):
        return True

    # Check for normalization/GRN layers (case-insensitive)
    name_lower = param_name.lower()
    return any(keyword in name_lower for keyword in _NO_DECAY_KEYWORDS)


def _get_convnextv2_stage_id(param_name: str, num_stages: int = 4) -> int:
    """Map a ConvNeXtV2 parameter name to its stage ID (0=embeddings, 1-4=encoder stages).

    Args:
        param_name: Full parameter name from named_parameters()
        num_stages: Number of encoder stages (default: 4 for ConvNeXtV2-Tiny/Base)

    Returns:
        Stage ID: 0 for embeddings, 1-4 for encoder.stages[0-3], num_stages for unknown params
    """
    # Check for embeddings first
    if param_name.startswith("embeddings."):
        return 0

    # Check for encoder stages
    for i in range(num_stages):
        if param_name.startswith(f"encoder.stages.{i}."):
            return i + 1

    # Unknown parameters default to the last stage (highest LR)
    # This is a safe fallback for any unexpected parameter names
    return num_stages


def split_named_params_for_weight_decay(
    named_params: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    lr: float,
    weight_decay: float,
    name_prefix: str,
) -> list[dict]:
    """Split named parameters into decay/no-decay optimizer groups.

    Parameters that are frozen are excluded automatically.
    """
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for param_name, param in named_params:
        if not param.requires_grad:
            continue
        if _is_no_decay_param(param_name, param):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups: list[dict] = []
    if decay_params:
        groups.append(
            {
                "params": decay_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": f"{name_prefix}.decay",
            }
        )
    if no_decay_params:
        groups.append(
            {
                "params": no_decay_params,
                "lr": lr,
                "weight_decay": 0.0,
                "name": f"{name_prefix}.no_decay",
            }
        )
    return groups


def build_llrd_param_groups_for_convnextv2(
    encoder: nn.Module,
    base_encoder_lr: float,
    weight_decay: float,
    gamma: float,
    *,
    name_prefix: str = "encoder",
) -> list[dict]:
    """Build parameter groups with layer-wise learning rate decay for ConvNeXtV2 encoder.

    Creates stage-level parameter groups where earlier stages receive lower learning rates.
    Each stage is split into two groups: one with weight decay (conv/linear weights) and
    one without (bias/norm parameters).

    Args:
        encoder: ConvNeXtV2 encoder module (typically model.encoder)
        base_encoder_lr: Base learning rate for the deepest encoder stage
        weight_decay: Weight decay value for parameters that should have it
        gamma: Decay factor for layer-wise LR (e.g., 0.75)
        name_prefix: Prefix for group names in logging (default: "encoder")

    Returns:
        List of parameter group dicts compatible with PyTorch optimizers.
        Each dict has: {"params": [...], "lr": float, "weight_decay": float, "name": str}

    Example:
        >>> groups = build_llrd_param_groups_for_convnextv2(
        ...     encoder=model.encoder,
        ...     base_encoder_lr=1e-4,
        ...     weight_decay=0.003,
        ...     gamma=0.75,
        ... )
        >>> optimizer = torch.optim.AdamW(groups)
    """
    # Probe number of stages from encoder structure
    # ConvNeXtV2Model has .encoder.stages (ModuleList)
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "stages"):
        num_stages = len(encoder.encoder.stages)
    else:
        # Fallback to default ConvNeXtV2-Tiny/Base structure
        num_stages = 4

    # Bucket parameters by stage and decay status
    # Structure: buckets[stage_id]["decay"|"no_decay"] = [param, param, ...]
    buckets: dict[int, dict[str, list[torch.nn.Parameter]]] = defaultdict(
        lambda: {"decay": [], "no_decay": []}
    )

    for name, param in encoder.named_parameters():
        # Skip frozen parameters (respects freeze_encoder_stages)
        if not param.requires_grad:
            continue

        # Determine which stage this parameter belongs to
        stage_id = _get_convnextv2_stage_id(name, num_stages)

        # Route to decay or no-decay bucket
        if _is_no_decay_param(name, param):
            buckets[stage_id]["no_decay"].append(param)
        else:
            buckets[stage_id]["decay"].append(param)

    # Build parameter groups with layer-wise LR
    param_groups: list[dict] = []

    for stage_id in sorted(buckets.keys()):
        # Calculate LR for this stage using LLRD formula
        # Earlier stages (lower stage_id) get lower LR
        lr = base_encoder_lr * (gamma ** (num_stages - stage_id))

        stage_buckets = buckets[stage_id]

        # Add decay group if it has parameters
        if stage_buckets["decay"]:
            param_groups.append(
                {
                    "params": stage_buckets["decay"],
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "name": f"{name_prefix}.stage{stage_id}.decay",
                }
            )

        # Add no-decay group if it has parameters
        if stage_buckets["no_decay"]:
            param_groups.append(
                {
                    "params": stage_buckets["no_decay"],
                    "lr": lr,
                    "weight_decay": 0.0,
                    "name": f"{name_prefix}.stage{stage_id}.no_decay",
                }
            )

    return param_groups
