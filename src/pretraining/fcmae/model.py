"""Dense masked image modeling inspired by ConvNeXt V2 FCMAE.

This module uses a Hugging Face dense ConvNeXtV2 encoder with learned
feature-space mask tokens after the patch embedding. It is therefore a
SimMIM-style masked image modeling objective, not Meta's sparse FCMAE
implementation. Small patch/mask/loss helpers are adapted from
`docs/external/ConvNeXt-V2/models/fcmae.py`; sparse encoder modules,
MinkowskiEngine, and upstream training code are intentionally replaced.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel, ConvNextV2Config
from transformers.models.convnextv2.modeling_convnextv2 import ConvNextV2Layer

from src.pretraining.fcmae.config import FCMAEModelConfig
from src.pretraining.fcmae.masking import (
    BACKGROUND_HEAVY_INK_DENSITY_THRESHOLD,
    FOREGROUND_HEAVY_INK_DENSITY_THRESHOLD,
    patchify,
    random_patch_mask,
)


@dataclass
class MaskedImageModelingOutput:
    loss: torch.Tensor
    pred_patches: torch.Tensor
    target_patches: torch.Tensor
    mask: torch.Tensor
    valid_patch_mask: torch.Tensor | None
    masked_foreground_ratio: torch.Tensor
    samples_skipped_no_valid_patches: torch.Tensor
    masked_ink_density: torch.Tensor | None
    masked_background_loss: torch.Tensor | None
    masked_foreground_loss: torch.Tensor | None
    masked_background_patch_ratio: torch.Tensor | None
    masked_foreground_patch_ratio: torch.Tensor | None
    loss_weight_mean_masked: torch.Tensor | None


@dataclass
class ReconstructionLossOutput:
    loss: torch.Tensor
    target_patches: torch.Tensor
    per_patch_loss: torch.Tensor
    loss_mask: torch.Tensor
    denominator: torch.Tensor
    samples_skipped_no_valid_patches: torch.Tensor
    loss_weight_mean_masked: torch.Tensor | None


def _detect_encoder_output_dim(encoder: nn.Module) -> int:
    config = getattr(encoder, "config", None)
    hidden_sizes = getattr(config, "hidden_sizes", None)
    if hidden_sizes:
        return int(hidden_sizes[-1])
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)
    raise ValueError("encoder_output_dim must be provided when it cannot be detected")


def _detect_encoder_embedding_dim(encoder: nn.Module) -> int:
    config = getattr(encoder, "config", None)
    hidden_sizes = getattr(config, "hidden_sizes", None)
    if hidden_sizes:
        return int(hidden_sizes[0])
    embeddings = getattr(encoder, "embeddings", None)
    patch_embeddings = getattr(embeddings, "patch_embeddings", None)
    out_channels = getattr(patch_embeddings, "out_channels", None)
    if out_channels is None:
        out_channels = getattr(embeddings, "out_channels", None)
    if out_channels is not None:
        return int(out_channels)
    raise ValueError("encoder embedding dimension could not be detected")


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
        self.ink_bias_strength = float(config.ink_bias_strength)

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

        self.encoder_embedding_dim = _detect_encoder_embedding_dim(self.encoder)
        self.mask_token = nn.Parameter(torch.empty(1, self.encoder_embedding_dim, 1, 1))
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

    def _encode_with_feature_mask(self, pixel_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embeddings = getattr(self.encoder, "embeddings", None)
        encoder = getattr(self.encoder, "encoder", None)
        if embeddings is None or encoder is None:
            raise TypeError("feature-space FCMAE masking requires ConvNeXtV2-style embeddings and encoder modules")

        hidden_states = embeddings(pixel_values)
        if hidden_states.ndim != 4:
            raise ValueError(f"encoder embeddings must be 4D, got {tuple(hidden_states.shape)}")
        grid_h, grid_w = mask.shape[-2:]
        embed_h, embed_w = hidden_states.shape[-2:]
        if embed_h % grid_h != 0 or embed_w % grid_w != 0:
            raise ValueError(
                "embedding grid must be an integer multiple of reconstruction grid: "
                f"{(embed_h, embed_w)} vs {(grid_h, grid_w)}"
            )
        scale_h = embed_h // grid_h
        scale_w = embed_w // grid_w
        feature_mask = mask.repeat_interleave(scale_h, dim=1).repeat_interleave(scale_w, dim=2)
        feature_mask = feature_mask.unsqueeze(1)
        mask_token = self.mask_token.to(dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = torch.where(feature_mask, mask_token, hidden_states)
        return _extract_feature_map(
            encoder(hidden_states),
            expected_channels=self.encoder_output_dim,
        )

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
        ink_density: torch.Tensor | None,
    ) -> ReconstructionLossOutput:
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
        if ink_density is None:
            patch_loss_weight = torch.ones_like(loss_mask)
            loss_weight_mean_masked: torch.Tensor | None = None
        else:
            ink = ink_density.to(device=loss.device, dtype=loss.dtype).reshape(loss_mask.shape)
            patch_loss_weight = 1.0 + self.config.ink_loss_weight_alpha * (
                ink / self.config.ink_loss_weight_target_density
            ).clamp(0.0, 1.0)
            weighted_mask_total = loss_mask.sum()
            loss_weight_mean_masked = torch.where(
                weighted_mask_total > 0,
                (patch_loss_weight * loss_mask).sum() / weighted_mask_total.clamp_min(1.0),
                torch.ones((), device=loss.device, dtype=loss.dtype),
            )
        weighted_loss_mask = loss_mask * patch_loss_weight
        denominator = weighted_loss_mask.sum().clamp_min(1.0)
        skipped = (flat_valid.sum(dim=1) == 0).sum()
        return ReconstructionLossOutput(
            loss=(loss * weighted_loss_mask).sum() / denominator,
            target_patches=target,
            per_patch_loss=loss,
            loss_mask=loss_mask,
            denominator=denominator,
            samples_skipped_no_valid_patches=skipped,
            loss_weight_mean_masked=loss_weight_mean_masked,
        )

    def _masked_ink_bin_diagnostics(
        self,
        per_patch_loss: torch.Tensor,
        loss_mask: torch.Tensor,
        ink_density: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if ink_density is None:
            return None, None, None, None

        ink = ink_density.to(device=per_patch_loss.device, dtype=torch.float32)
        if ink.shape != loss_mask.shape:
            ink = ink.reshape(loss_mask.shape)

        background_mask = loss_mask * (
            ink < BACKGROUND_HEAVY_INK_DENSITY_THRESHOLD
        ).to(dtype=loss_mask.dtype)
        foreground_mask = loss_mask * (
            ink >= FOREGROUND_HEAVY_INK_DENSITY_THRESHOLD
        ).to(dtype=loss_mask.dtype)

        masked_total = loss_mask.sum().clamp_min(1.0)
        background_count = background_mask.sum()
        foreground_count = foreground_mask.sum()

        nan = torch.full((), float("nan"), device=per_patch_loss.device, dtype=per_patch_loss.dtype)
        background_loss = torch.where(
            background_count > 0,
            (per_patch_loss * background_mask).sum() / background_count.clamp_min(1.0),
            nan,
        )
        foreground_loss = torch.where(
            foreground_count > 0,
            (per_patch_loss * foreground_mask).sum() / foreground_count.clamp_min(1.0),
            nan,
        )
        background_ratio = background_count.to(dtype=per_patch_loss.dtype) / masked_total
        foreground_ratio = foreground_count.to(dtype=per_patch_loss.dtype) / masked_total
        return background_loss, foreground_loss, background_ratio, foreground_ratio

    def forward(
        self,
        pixel_values: torch.Tensor,
        valid_patch_mask: torch.Tensor | None = None,
        ink_density: torch.Tensor | None = None,
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
            patch_weights=ink_density,
            bias_strength=self.ink_bias_strength,
            foreground_mask_ratio=(
                self.config.foreground_mask_ratio if ink_density is not None else None
            ),
            medium_mask_ratio=self.config.medium_mask_ratio if ink_density is not None else None,
            background_mask_ratio=(
                self.config.background_mask_ratio if ink_density is not None else None
            ),
        )
        features = self._encode_with_feature_mask(pixel_values, mask)
        pred = self._decode_features(features, grid_h, grid_w)
        reconstruction = self._reconstruction_loss(
            pixel_values,
            pred,
            mask,
            valid_patch_mask,
            ink_density,
        )
        (
            masked_background_loss,
            masked_foreground_loss,
            masked_background_patch_ratio,
            masked_foreground_patch_ratio,
        ) = self._masked_ink_bin_diagnostics(
            reconstruction.per_patch_loss,
            reconstruction.loss_mask,
            ink_density,
        )

        if valid_patch_mask is None:
            masked_foreground_ratio = mask.float().mean()
        else:
            masked_valid = (mask & valid_patch_mask.to(device=mask.device)).sum()
            valid_total = valid_patch_mask.to(device=mask.device).sum().clamp_min(1)
            masked_foreground_ratio = masked_valid.float() / valid_total.float()

        if ink_density is None:
            masked_ink_density: torch.Tensor | None = None
        else:
            ink = ink_density.to(device=mask.device, dtype=torch.float32)
            mask_f = mask.float()
            denom = mask_f.sum().clamp_min(1.0)
            masked_ink_density = (ink * mask_f).sum() / denom

        return MaskedImageModelingOutput(
            loss=reconstruction.loss,
            pred_patches=pred,
            target_patches=reconstruction.target_patches,
            mask=mask,
            valid_patch_mask=valid_patch_mask,
            masked_foreground_ratio=masked_foreground_ratio,
            samples_skipped_no_valid_patches=reconstruction.samples_skipped_no_valid_patches.to(
                device=reconstruction.loss.device
            ),
            masked_ink_density=masked_ink_density,
            masked_background_loss=masked_background_loss,
            masked_foreground_loss=masked_foreground_loss,
            masked_background_patch_ratio=masked_background_patch_ratio,
            masked_foreground_patch_ratio=masked_foreground_patch_ratio,
            loss_weight_mean_masked=reconstruction.loss_weight_mean_masked,
        )
