"""Dense masked image modeling inspired by ConvNeXt V2 FCMAE.

This module uses a Hugging Face dense ConvNeXtV2 encoder with learned pixel
mask tokens. It is therefore a SimMIM-style masked image modeling objective,
not Meta's sparse FCMAE implementation. Small patch/mask/loss helpers are
adapted from `docs/external/ConvNeXt-V2/models/fcmae.py`; sparse encoder
modules, MinkowskiEngine, and upstream training code are intentionally replaced.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel, ConvNextV2Config
from transformers.models.convnextv2.modeling_convnextv2 import ConvNextV2Layer

from src.pretraining.fcmae.config import FCMAEModelConfig
from src.pretraining.fcmae.masking import patchify, random_patch_mask, upsample_mask


@dataclass
class MaskedImageModelingOutput:
    loss: torch.Tensor
    pred_patches: torch.Tensor
    target_patches: torch.Tensor
    mask: torch.Tensor
    valid_patch_mask: torch.Tensor | None
    masked_foreground_ratio: torch.Tensor
    samples_skipped_no_valid_patches: torch.Tensor


def _detect_encoder_output_dim(encoder: nn.Module) -> int:
    config = getattr(encoder, "config", None)
    hidden_sizes = getattr(config, "hidden_sizes", None)
    if hidden_sizes:
        return int(hidden_sizes[-1])
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)
    raise ValueError("encoder_output_dim must be provided when it cannot be detected")


def _extract_feature_map(encoder_output: object, *, expected_channels: int) -> torch.Tensor:
    features = getattr(encoder_output, "last_hidden_state", encoder_output)
    if isinstance(features, tuple):
        features = features[0]
    if not torch.is_tensor(features):
        raise TypeError("encoder output must be a tensor or expose last_hidden_state")
    if features.ndim != 4:
        raise ValueError(f"encoder feature map must be 4D, got {tuple(features.shape)}")
    if features.shape[1] == expected_channels:
        return features
    if features.shape[-1] != expected_channels:
        raise ValueError(
            "encoder feature map channel dimension is ambiguous: "
            f"got {tuple(features.shape)}, expected_channels={expected_channels}"
        )
    return features.permute(0, 3, 1, 2).contiguous()


class DenseMaskedImageModelingConvNeXtV2(nn.Module):
    def __init__(
        self,
        config: FCMAEModelConfig,
        encoder: nn.Module | None = None,
        encoder_output_dim: int | None = None,
        encoder_stride: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.patch_size = int(config.patch_size)
        self.mask_ratio = float(config.mask_ratio)
        self.norm_pix_loss = bool(config.norm_pix_loss)

        if encoder is None:
            self.encoder = AutoModel.from_pretrained(config.encoder_model_name_or_path)
            self.encoder_output_dim = _detect_encoder_output_dim(self.encoder)
            self.encoder_stride = int(encoder_stride or self.patch_size)
        else:
            if encoder_output_dim is None or encoder_stride is None:
                raise ValueError(
                    "encoder_output_dim and encoder_stride are required when encoder is injected"
                )
            self.encoder = encoder
            self.encoder_output_dim = int(encoder_output_dim)
            self.encoder_stride = int(encoder_stride)

        if self.encoder_stride != self.patch_size:
            raise ValueError("v1 requires encoder_stride to equal patch_size")

        self.mask_token = nn.Parameter(torch.empty(1, 3, self.patch_size, self.patch_size))
        nn.init.normal_(self.mask_token, std=0.02)

        self.proj = nn.Conv2d(self.encoder_output_dim, config.decoder_dim, kernel_size=1)
        decoder_config = ConvNextV2Config(
            hidden_sizes=[config.decoder_dim],
            depths=[config.decoder_depth],
            drop_path_rate=0.0,
        )
        self.decoder = nn.Sequential(
            *[
                ConvNextV2Layer(decoder_config, dim=config.decoder_dim, drop_path=0.0)
                for _ in range(config.decoder_depth)
            ]
        )
        self.pred = nn.Conv2d(
            config.decoder_dim,
            self.patch_size * self.patch_size * 3,
            kernel_size=1,
        )

    def _apply_pixel_mask(self, pixel_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        grid_h, grid_w = mask.shape[-2:]
        tiled_token = self.mask_token.repeat(pixel_values.shape[0], 1, grid_h, grid_w)
        pixel_mask = upsample_mask(mask, self.patch_size).unsqueeze(1)
        return torch.where(pixel_mask, tiled_token.to(pixel_values.dtype), pixel_values)

    def _decode_features(self, features: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        if features.shape[-2:] != (grid_h, grid_w):
            raise ValueError(
                "encoder feature map shape does not match patch grid: "
                f"{tuple(features.shape[-2:])} vs {(grid_h, grid_w)}"
            )
        x = self.proj(features)
        x = self.decoder(x)
        pred = self.pred(x)
        pred = pred.reshape(pred.shape[0], pred.shape[1], grid_h * grid_w)
        return torch.einsum("ncl->nlc", pred)

    def _reconstruction_loss(
        self,
        imgs: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        valid_patch_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute masked patch reconstruction loss.

        Ported from: docs/external/ConvNeXt-V2/models/fcmae.py:164-184.
        Adapted to exclude invalid padded patches and avoid zero denominators.
        # Ported from facebookresearch/ConvNeXt-V2, fcmae.py:164-184, commit
        # 2553895753323c6fe0b2bf390683f5ea358a42b9. Licensed under the upstream LICENSE.
        """
        target = patchify(imgs, self.patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6).sqrt()

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        flat_mask = mask.reshape(mask.shape[0], -1).to(dtype=loss.dtype)
        if valid_patch_mask is None:
            flat_valid = torch.ones_like(flat_mask)
        else:
            flat_valid = valid_patch_mask.reshape(valid_patch_mask.shape[0], -1).to(
                device=loss.device,
                dtype=loss.dtype,
            )
        loss_mask = flat_mask * flat_valid
        denominator = loss_mask.sum().clamp_min(1.0)
        skipped = (flat_valid.sum(dim=1) == 0).sum()
        return (loss * loss_mask).sum() / denominator, target, denominator, skipped

    def forward(
        self,
        pixel_values: torch.Tensor,
        valid_patch_mask: torch.Tensor | None = None,
    ) -> MaskedImageModelingOutput:
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must have shape (B, 3, H, W)")
        batch_size, _channels, height, width = pixel_values.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("pixel_values height and width must be divisible by patch_size")

        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        mask = random_patch_mask(
            batch_size,
            grid_h,
            grid_w,
            self.mask_ratio,
            pixel_values.device,
            valid_patch_mask=valid_patch_mask,
        )
        masked_pixels = self._apply_pixel_mask(pixel_values, mask)
        features = _extract_feature_map(
            self.encoder(masked_pixels),
            expected_channels=self.encoder_output_dim,
        )
        pred = self._decode_features(features, grid_h, grid_w)
        loss, target, _denominator, skipped = self._reconstruction_loss(
            pixel_values,
            pred,
            mask,
            valid_patch_mask,
        )

        if valid_patch_mask is None:
            masked_foreground_ratio = mask.float().mean()
        else:
            masked_valid = (mask & valid_patch_mask.to(device=mask.device)).sum()
            valid_total = valid_patch_mask.to(device=mask.device).sum().clamp_min(1)
            masked_foreground_ratio = masked_valid.float() / valid_total.float()

        return MaskedImageModelingOutput(
            loss=loss,
            pred_patches=pred,
            target_patches=target,
            mask=mask,
            valid_patch_mask=valid_patch_mask,
            masked_foreground_ratio=masked_foreground_ratio,
            samples_skipped_no_valid_patches=skipped.to(device=loss.device),
        )
