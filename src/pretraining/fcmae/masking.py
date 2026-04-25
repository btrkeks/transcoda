from __future__ import annotations

import torch


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
) -> torch.Tensor:
    """Return a boolean mask where True marks selected reconstruction patches.

    With ``bias_strength > 0`` and ``patch_weights`` in ``[0, 1]``, sampling is
    biased toward higher-weighted patches via ``noise = rand - bias * weight``
    before argsort. ``bias_strength=0`` recovers the original uniform sampling.

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
    if bias_strength > 0 and patch_weights is not None:
        if patch_weights.shape != (batch_size, grid_h, grid_w):
            raise ValueError(
                "patch_weights must have shape "
                f"{(batch_size, grid_h, grid_w)}, got {tuple(patch_weights.shape)}"
            )
        weights = patch_weights.to(device=device, dtype=noise.dtype).reshape(batch_size, num_patches)
        noise = noise - bias_strength * weights
    noise = noise.masked_fill(~valid, float("inf"))

    for sample_idx in range(batch_size):
        num_valid = int(valid[sample_idx].sum().item())
        if num_valid == 0:
            continue
        num_mask = max(1, int(round(mask_ratio * num_valid)))
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
