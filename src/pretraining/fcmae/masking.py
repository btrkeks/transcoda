from __future__ import annotations

import torch

BACKGROUND_HEAVY_INK_DENSITY_THRESHOLD = 0.01
FOREGROUND_HEAVY_INK_DENSITY_THRESHOLD = 0.03


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert images to flattened rectangular patch vectors.

    Ported from: docs/external/ConvNeXt-V2/models/fcmae.py:91-103.
    Adapted for rectangular inputs.
    # Ported from facebookresearch/ConvNeXt-V2, fcmae.py:91-103, commit
    # 2553895753323c6fe0b2bf390683f5ea358a42b9. Licensed under the upstream LICENSE.
    """
    if imgs.ndim != 4:
        raise ValueError(f"imgs must have shape (B, C, H, W), got {tuple(imgs.shape)}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    batch, channels, height, width = imgs.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"image size {(height, width)} must be divisible by patch_size={patch_size}"
        )

    grid_h = height // patch_size
    grid_w = width // patch_size
    x = imgs.reshape(batch, channels, grid_h, patch_size, grid_w, patch_size)
    x = torch.einsum("nchpwq->nhwpqc", x)
    return x.reshape(batch, grid_h * grid_w, patch_size * patch_size * channels)


def unpatchify(x: torch.Tensor, grid_h: int, grid_w: int, patch_size: int) -> torch.Tensor:
    """Convert flattened rectangular patch vectors back to images.

    Ported from: docs/external/ConvNeXt-V2/models/fcmae.py:105-117.
    Adapted to take rectangular grid dimensions explicitly.
    # Ported from facebookresearch/ConvNeXt-V2, fcmae.py:105-117, commit
    # 2553895753323c6fe0b2bf390683f5ea358a42b9. Licensed under the upstream LICENSE.
    """
    if x.ndim != 3:
        raise ValueError(f"x must have shape (B, L, patch_dim), got {tuple(x.shape)}")
    if grid_h <= 0 or grid_w <= 0 or patch_size <= 0:
        raise ValueError("grid_h, grid_w, and patch_size must be positive")
    if x.shape[1] != grid_h * grid_w:
        raise ValueError(f"x has {x.shape[1]} patches, expected {grid_h * grid_w}")
    channels_times_patch = x.shape[2]
    patch_area = patch_size * patch_size
    if channels_times_patch % patch_area != 0:
        raise ValueError("last dimension must be divisible by patch_size**2")
    channels = channels_times_patch // patch_area

    x = x.reshape(x.shape[0], grid_h, grid_w, patch_size, patch_size, channels)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(x.shape[0], channels, grid_h * patch_size, grid_w * patch_size)


def random_patch_mask(
    batch_size: int,
    grid_h: int,
    grid_w: int,
    mask_ratio: float,
    device: torch.device,
    valid_patch_mask: torch.Tensor | None = None,
    patch_weights: torch.Tensor | None = None,
    bias_strength: float = 0.0,
    foreground_mask_ratio: float | None = None,
    medium_mask_ratio: float | None = None,
    background_mask_ratio: float | None = None,
    background_threshold: float = BACKGROUND_HEAVY_INK_DENSITY_THRESHOLD,
    foreground_threshold: float = FOREGROUND_HEAVY_INK_DENSITY_THRESHOLD,
) -> torch.Tensor:
    """Return a boolean mask where True marks selected reconstruction patches.

    With quota ratios and ``patch_weights`` in ``[0, 1]``, sampling first draws
    from foreground/medium/background ink-density bins, then backfills from
    remaining valid patches. Without quota ratios, ``bias_strength > 0`` keeps
    the legacy weighted random sampler via ``noise = rand - bias * weight``.

    Ported from: docs/external/ConvNeXt-V2/models/fcmae.py:119-135.
    Adapted for rectangular grids, optional valid-patch sampling, and weighted
    sampling without replacement.
    # Ported from facebookresearch/ConvNeXt-V2, fcmae.py:119-135, commit
    # 2553895753323c6fe0b2bf390683f5ea358a42b9. Licensed under the upstream LICENSE.
    """
    if batch_size <= 0 or grid_h <= 0 or grid_w <= 0:
        raise ValueError("batch_size, grid_h, and grid_w must be positive")
    if not 0 < mask_ratio < 1:
        raise ValueError("mask_ratio must satisfy 0 < mask_ratio < 1")
    if bias_strength < 0:
        raise ValueError("bias_strength must be >= 0")
    quota_values = (foreground_mask_ratio, medium_mask_ratio, background_mask_ratio)
    use_quotas = any(value is not None for value in quota_values)
    if use_quotas:
        if patch_weights is None:
            raise ValueError("patch_weights are required for foreground-aware quota masking")
        if any(value is None for value in quota_values):
            raise ValueError("all foreground/medium/background mask ratios are required")
        quota_sum = (
            float(foreground_mask_ratio)
            + float(medium_mask_ratio)
            + float(background_mask_ratio)
        )
        if (
            min(
                float(foreground_mask_ratio),
                float(medium_mask_ratio),
                float(background_mask_ratio),
            )
            < 0
        ):
            raise ValueError("foreground/medium/background mask ratios must be >= 0")
        if abs(quota_sum - 1.0) > 1.0e-6:
            raise ValueError("foreground/medium/background mask ratios must sum to 1.0")
        if background_threshold < 0 or foreground_threshold < 0:
            raise ValueError("ink-density thresholds must be >= 0")
        if background_threshold >= foreground_threshold:
            raise ValueError("background_threshold must be less than foreground_threshold")

    num_patches = grid_h * grid_w
    if valid_patch_mask is None:
        valid = torch.ones(batch_size, num_patches, dtype=torch.bool, device=device)
    else:
        if valid_patch_mask.shape != (batch_size, grid_h, grid_w):
            raise ValueError(
                "valid_patch_mask must have shape "
                f"{(batch_size, grid_h, grid_w)}, got {tuple(valid_patch_mask.shape)}"
            )
        valid = valid_patch_mask.to(device=device, dtype=torch.bool).reshape(batch_size, num_patches)

    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    noise = torch.rand(batch_size, num_patches, device=device)
    weights = None
    if patch_weights is not None:
        if patch_weights.shape != (batch_size, grid_h, grid_w):
            raise ValueError(
                "patch_weights must have shape "
                f"{(batch_size, grid_h, grid_w)}, got {tuple(patch_weights.shape)}"
            )
        weights = patch_weights.to(device=device, dtype=noise.dtype).reshape(batch_size, num_patches)
    if bias_strength > 0 and weights is not None and not use_quotas:
        noise = noise - bias_strength * weights
    noise = noise.masked_fill(~valid, float("inf"))

    for sample_idx in range(batch_size):
        num_valid = int(valid[sample_idx].sum().item())
        if num_valid == 0:
            continue
        num_mask = max(1, int(round(mask_ratio * num_valid)))
        if use_quotas:
            assert weights is not None
            sample_weights = weights[sample_idx]
            sample_valid = valid[sample_idx]
            selected = torch.zeros(num_patches, dtype=torch.bool, device=device)
            fg_target = int(round(num_mask * float(foreground_mask_ratio)))
            medium_target = int(round(num_mask * float(medium_mask_ratio)))
            background_target = max(0, num_mask - fg_target - medium_target)
            bins = [
                (sample_valid & (sample_weights >= foreground_threshold), fg_target),
                (
                    sample_valid
                    & (sample_weights >= background_threshold)
                    & (sample_weights < foreground_threshold),
                    medium_target,
                ),
                (sample_valid & (sample_weights < background_threshold), background_target),
            ]
            for bin_mask, target in bins:
                if target <= 0:
                    continue
                available = bin_mask & ~selected
                ids = torch.nonzero(available, as_tuple=False).flatten()
                if ids.numel() == 0:
                    continue
                order = torch.argsort(noise[sample_idx, ids])
                selected[ids[order[:target]]] = True

            shortfall = num_mask - int(selected.sum().item())
            if shortfall > 0:
                remaining = sample_valid & ~selected
                ids = torch.nonzero(remaining, as_tuple=False).flatten()
                if ids.numel() > 0:
                    order = torch.argsort(noise[sample_idx, ids])
                    selected[ids[order[:shortfall]]] = True
            candidate_ids = torch.nonzero(selected, as_tuple=False).flatten()
        else:
            candidate_ids = torch.argsort(noise[sample_idx])[:num_mask]
        mask[sample_idx, candidate_ids] = True

    return mask.reshape(batch_size, grid_h, grid_w)


def upsample_mask(mask: torch.Tensor, scale: int) -> torch.Tensor:
    """Upsample a rectangular patch mask to image-space mask pixels.

    Ported from: docs/external/ConvNeXt-V2/models/fcmae.py:137-142.
    Adapted to accept `(B, grid_h, grid_w)` directly.
    # Ported from facebookresearch/ConvNeXt-V2, fcmae.py:137-142, commit
    # 2553895753323c6fe0b2bf390683f5ea358a42b9. Licensed under the upstream LICENSE.
    """
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (B, grid_h, grid_w), got {tuple(mask.shape)}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    return mask.repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)
