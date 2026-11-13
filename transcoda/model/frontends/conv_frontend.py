from __future__ import annotations

import torch

from .feature_map import FeatureMapVisionFrontend


class ConvVisionFrontend(FeatureMapVisionFrontend):
    """Conv encoder frontend with token-space projector MLP."""

    def extract_feature_map(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values=images).last_hidden_state
