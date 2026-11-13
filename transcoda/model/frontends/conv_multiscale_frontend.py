from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_map import FeatureMapVisionFrontend


def _resolve_hidden_state_channels(encoder: nn.Module, hidden_state_index: int) -> int:
    if hidden_state_index >= 0:
        raise ValueError("conv_multiscale requires a negative multiscale_hidden_state_index")

    encoder_config = getattr(encoder, "config", None)
    hidden_sizes = getattr(encoder_config, "hidden_sizes", None)
    if not hidden_sizes:
        raise ValueError(
            "conv_multiscale requires encoder.config.hidden_sizes to infer "
            "the selected hidden-state channel count"
        )

    hidden_sizes = [int(size) for size in hidden_sizes]
    hidden_state_channels = [hidden_sizes[0], *hidden_sizes]
    try:
        return hidden_state_channels[hidden_state_index]
    except IndexError as exc:
        raise ValueError(
            "multiscale_hidden_state_index is out of range for ConvNeXt hidden states: "
            f"index={hidden_state_index}, available={len(hidden_state_channels)}"
        ) from exc


class MultiScaleFeatureFusion(nn.Module):
    """Gated residual fusion from a higher-resolution detail map into final features."""

    def __init__(
        self,
        detail_channels: int,
        final_channels: int,
        init_logit: float,
    ) -> None:
        super().__init__()
        self.detail_projection = nn.Conv2d(detail_channels, final_channels, kernel_size=1)
        self.gate_logit = nn.Parameter(torch.tensor(float(init_logit)))

    def forward(self, final_map: torch.Tensor, detail_map: torch.Tensor) -> torch.Tensor:
        if final_map.ndim != 4 or detail_map.ndim != 4:
            raise ValueError(
                "multiscale fusion expects 4D feature maps, "
                f"got final={tuple(final_map.shape)}, detail={tuple(detail_map.shape)}"
            )

        detail_map = self.detail_projection(detail_map)
        detail_map = F.adaptive_avg_pool2d(detail_map, output_size=final_map.shape[-2:])
        gate = torch.sigmoid(self.gate_logit).to(dtype=final_map.dtype, device=final_map.device)
        return final_map + gate * detail_map.to(dtype=final_map.dtype, device=final_map.device)


class ConvMultiScaleVisionFrontend(FeatureMapVisionFrontend):
    """ConvNeXt frontend that fuses penultimate-stage detail into final-stage tokens."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.multiscale_hidden_state_index = int(config.multiscale_hidden_state_index)
        self.multiscale_detail_dim = _resolve_hidden_state_channels(
            self.encoder,
            self.multiscale_hidden_state_index,
        )
        self.fusion = MultiScaleFeatureFusion(
            detail_channels=self.multiscale_detail_dim,
            final_channels=self.encoder_output_dim,
            init_logit=float(config.multiscale_fusion_init_logit),
        )

    def bridge_named_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
        yield from self.projector.named_parameters()
        for name, param in self.fusion.named_parameters():
            yield f"fusion.{name}", param

    def extract_feature_map(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(pixel_values=images, output_hidden_states=True)
        final_map = getattr(outputs, "last_hidden_state", None)
        if final_map is None:
            raise ValueError("ConvMultiScaleVisionFrontend requires last_hidden_state")

        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise ValueError(
                "ConvMultiScaleVisionFrontend requires encoder outputs with hidden_states. "
                "Use a ConvNeXt-style encoder that honors output_hidden_states=True."
            )

        try:
            detail_map = hidden_states[self.multiscale_hidden_state_index]
        except IndexError as exc:
            raise ValueError(
                "multiscale_hidden_state_index is out of range for returned hidden_states: "
                f"index={self.multiscale_hidden_state_index}, available={len(hidden_states)}"
            ) from exc

        if not torch.is_tensor(final_map) or not torch.is_tensor(detail_map):
            raise TypeError("multiscale frontend expected tensor hidden states")

        final_map = self._coerce_feature_map_channels(
            final_map,
            expected_channels=self.encoder_output_dim,
        )
        detail_map = self._coerce_feature_map_channels(
            detail_map,
            expected_channels=self.multiscale_detail_dim,
        )
        return self.fusion(final_map, detail_map)
