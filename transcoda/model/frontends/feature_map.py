from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from ..configuration_transcoda import TranscodaConfig
from ..encoder import EncoderLoader
from ..vision_frontend import PositionalEncoding2D, ProjectorMLP, VisionFrontendOutput


class FeatureMapVisionFrontend(nn.Module):
    """Base frontend for encoders that expose a final 2D feature map."""

    def __init__(self, config: TranscodaConfig) -> None:
        super().__init__()
        self.encoder, encoder_output_dim = EncoderLoader.load(config)
        self.encoder_output_dim = encoder_output_dim
        self.projector = ProjectorMLP(
            in_dim=encoder_output_dim,
            out_dim=config.d_model,
            hidden_mult=float(config.projector_hidden_mult),
        )
        self.pos2d: PositionalEncoding2D | None = None
        self._last_encoder_hw: tuple[int, int] | None = None

    def extract_feature_map(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def bridge_named_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
        return self.projector.named_parameters()

    def forward(
        self,
        images: torch.Tensor,
        image_sizes: torch.Tensor | None = None,
    ) -> VisionFrontendOutput:
        images = images.contiguous(memory_format=torch.channels_last)
        encoder_features = self._coerce_feature_map_channels(
            self.extract_feature_map(images),
            expected_channels=self.encoder_output_dim,
        )

        if encoder_features.ndim != 4:
            raise ValueError(
                "vision frontend expected a 4D encoder feature map, "
                f"got shape {tuple(encoder_features.shape)}"
            )

        batch_size, _, enc_h, enc_w = encoder_features.shape
        self._last_encoder_hw = (enc_h, enc_w)

        encoder_tokens = torch.flatten(encoder_features, start_dim=2, end_dim=3).transpose(1, 2)
        encoder_tokens = encoder_tokens.contiguous()

        projected_tokens = self.projector(encoder_tokens)

        projected_map = projected_tokens.transpose(1, 2).reshape(batch_size, -1, enc_h, enc_w)
        if self.pos2d is None or enc_h > self.pos2d.h_max or enc_w > self.pos2d.w_max:
            self.pos2d = PositionalEncoding2D(
                dim=projected_map.shape[1],
                h_max=enc_h,
                w_max=enc_w,
            ).to(projected_map.device)

        projected_map_pos = self.pos2d(projected_map)
        projected_tokens_pos = torch.flatten(
            projected_map_pos, start_dim=2, end_dim=3
        ).transpose(1, 2)
        projected_tokens_pos = projected_tokens_pos.contiguous()

        attention_mask = None
        if image_sizes is not None:
            attention_mask = self.build_memory_key_padding_mask(
                image_sizes=image_sizes,
                encoder_hw=(enc_h, enc_w),
                input_hw=(images.shape[-2], images.shape[-1]),
                device=images.device,
            )

        return VisionFrontendOutput(
            encoder_tokens_raw=projected_tokens,
            encoder_tokens_pos=projected_tokens_pos,
            encoder_attention_mask=attention_mask,
        )

    @staticmethod
    def _coerce_feature_map_channels(
        encoder_features: torch.Tensor,
        *,
        expected_channels: int,
    ) -> torch.Tensor:
        if (
            encoder_features.ndim == 4
            and encoder_features.shape[1] != expected_channels
            and encoder_features.shape[-1] == expected_channels
        ):
            return encoder_features.permute(0, 3, 1, 2).contiguous()
        return encoder_features

    @staticmethod
    def build_memory_key_padding_mask(
        image_sizes: torch.Tensor,
        encoder_hw: tuple[int, int],
        input_hw: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Build a boolean mask for valid encoder token positions."""
        if not isinstance(image_sizes, torch.Tensor):
            sizes = torch.tensor(image_sizes, device=device)
        else:
            sizes = image_sizes.to(device)

        if sizes.ndim == 1:
            sizes = sizes.view(-1, 2)

        batch_size = sizes.shape[0]
        enc_h, enc_w = encoder_hw
        padded_h, padded_w = input_hw

        orig_h = sizes[:, 0]
        orig_w = sizes[:, 1]

        valid_h = (orig_h * enc_h + padded_h - 1) // padded_h
        valid_w = (orig_w * enc_w + padded_w - 1) // padded_w

        h_range = torch.arange(enc_h, device=device).view(1, enc_h, 1)
        w_range = torch.arange(enc_w, device=device).view(1, 1, enc_w)

        mask_h = h_range < valid_h.view(batch_size, 1, 1)
        mask_w = w_range < valid_w.view(batch_size, 1, 1)
        mask = mask_h & mask_w
        return mask.reshape(batch_size, -1)
