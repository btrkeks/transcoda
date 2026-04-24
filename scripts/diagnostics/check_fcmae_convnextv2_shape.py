from __future__ import annotations

import argparse

import torch
from transformers import AutoModel


def _feature_map(output: object, *, expected_channels: int) -> torch.Tensor:
    features = getattr(output, "last_hidden_state", output)
    if features.shape[1] == expected_channels:
        return features
    if features.shape[-1] != expected_channels:
        raise ValueError(
            f"ambiguous feature layout {tuple(features.shape)} for channels={expected_channels}"
        )
    return features.permute(0, 3, 1, 2).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe ConvNeXtV2 FCMAE feature-map shape")
    parser.add_argument("--model", default="facebook/convnextv2-base-22k-224")
    parser.add_argument("--image-height", type=int, default=768)
    parser.add_argument("--image-width", type=int, default=544)
    parser.add_argument("--stride", type=int, default=32)
    args = parser.parse_args()

    encoder = AutoModel.from_pretrained(args.model)
    hidden_sizes = getattr(encoder.config, "hidden_sizes", None)
    expected_channels = int(hidden_sizes[-1]) if hidden_sizes else int(encoder.config.hidden_size)
    x = torch.zeros(1, 3, args.image_height, args.image_width)
    with torch.no_grad():
        features = _feature_map(encoder(x), expected_channels=expected_channels)
    assert features.ndim == 4
    expected = (args.image_height // args.stride, args.image_width // args.stride)
    assert features.shape[-2:] == expected, (features.shape, expected)
    print(
        f"channels={features.shape[1]} feature_hw={tuple(features.shape[-2:])} "
        f"stride={args.stride}"
    )


if __name__ == "__main__":
    main()
