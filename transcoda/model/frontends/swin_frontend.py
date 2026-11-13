from __future__ import annotations

import torch

from .feature_map import FeatureMapVisionFrontend


class SwinVisionFrontend(FeatureMapVisionFrontend):
    """Swin/SwinV2 frontend using Hugging Face reshaped hidden states."""

    def extract_feature_map(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(pixel_values=images, output_hidden_states=True)
        reshaped_hidden_states = getattr(outputs, "reshaped_hidden_states", None)
        if not reshaped_hidden_states:
            raise ValueError(
                "SwinVisionFrontend requires encoder outputs with reshaped_hidden_states. "
                "Use a Hugging Face Swin/SwinV2-style vision encoder."
            )
        return reshaped_hidden_states[-1]
