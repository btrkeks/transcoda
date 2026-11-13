from __future__ import annotations

from ..configuration_transcoda import TranscodaConfig
from ..vision_frontend import VisionFrontend
from .conv_frontend import ConvVisionFrontend
from .conv_multiscale_frontend import ConvMultiScaleVisionFrontend
from .swin_frontend import SwinVisionFrontend


def build_vision_frontend(config: TranscodaConfig) -> VisionFrontend:
    if config.vision_frontend == "conv":
        return ConvVisionFrontend(config)
    if config.vision_frontend == "conv_multiscale":
        return ConvMultiScaleVisionFrontend(config)
    if config.vision_frontend == "swin":
        return SwinVisionFrontend(config)
    raise ValueError(
        f"Unsupported vision_frontend: {config.vision_frontend}. "
        "Supported: ['conv', 'conv_multiscale', 'swin']"
    )
