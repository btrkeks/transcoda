"""Validation/test generation helpers for Transcoda models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import transformers

from transcoda.model import VisionFrontendOutput
from transcoda.model.generation_policy import (
    GenerationSettings,
    build_generate_kwargs,
)


@dataclass(frozen=True)
class DecodePolicy:
    """Named generation policy used for validation prediction records."""

    name: str
    generation_settings: GenerationSettings
    configured_max_length: int | None = None


class ValidationDecoder:
    """Decode validation/test batches with an explicit generation policy."""

    def __init__(
        self,
        *,
        model: Any,
        constraint_factory: Any,
        policy: DecodePolicy,
    ) -> None:
        self._model = model
        self._constraint_factory = constraint_factory
        self._policy = policy

    @property
    def policy_name(self) -> str:
        return self._policy.name

    def effective_max_length(self, max_length: int) -> int:
        effective_max_length = int(max_length)
        if self._policy.configured_max_length is not None:
            effective_max_length = min(
                effective_max_length,
                int(self._policy.configured_max_length),
            )
        return effective_max_length

    def generate(
        self,
        *,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor | None,
        max_length: int,
    ) -> torch.Tensor:
        """Generate token IDs for a validation/test batch."""
        effective_max_length = self.effective_max_length(max_length)
        bundle = self._constraint_factory.build(self._policy.generation_settings)
        logits_processor = bundle.logits_processors
        stopping_criteria = bundle.stopping_criteria
        generation_settings = bundle.generation_settings
        _require_single_return_sequence(generation_settings)

        encoder_outputs = self._model.forward_encoder(pixel_values, image_sizes=image_sizes)
        expand_size = _encoder_expand_size(generation_settings)
        if expand_size > 1:
            encoder_outputs = _expand_encoder_outputs(encoder_outputs, expand_size)

        generate_kwargs = build_generate_kwargs(
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            max_length=effective_max_length,
            settings=generation_settings,
            logits_processor=logits_processor,
        )
        if stopping_criteria is not None:
            generate_kwargs["stopping_criteria"] = transformers.StoppingCriteriaList(
                stopping_criteria
            )
        generate_kwargs.pop("pixel_values", None)
        generate_kwargs.pop("image_sizes", None)
        generate_kwargs["input_ids"] = torch.full(
            (pixel_values.shape[0], 1),
            self._model.config.bos_token_id,
            dtype=torch.long,
            device=pixel_values.device,
        )
        generate_kwargs["encoder_outputs"] = encoder_outputs
        return self._model.generate(**generate_kwargs)


def validation_generation_cap(
    target_lengths: torch.Tensor,
    *,
    absolute_max_length: int | None,
    multiplier: float,
    buffer: int,
    maxlen: int,
    configured_max_length: int | None,
) -> int:
    """Validation decode cap as Hugging Face max_length (BOS and EOS included).

    An absolute cap ignores targets, so a page decodes the same way whatever its
    batch neighbours are. Without one, fall back to the target-relative cap.
    """
    if absolute_max_length is None:
        return target_relative_generation_cap(
            target_lengths,
            multiplier=multiplier,
            buffer=buffer,
            maxlen=maxlen,
            configured_max_length=configured_max_length,
        )
    caps = [int(absolute_max_length), int(maxlen)]
    if configured_max_length is not None:
        caps.append(int(configured_max_length))
    return max(1, min(caps))


def target_relative_generation_cap(
    target_lengths: torch.Tensor,
    *,
    multiplier: float,
    buffer: int,
    maxlen: int,
    configured_max_length: int | None,
) -> int:
    """Cap validation generation relative to target length and model max length."""
    max_gt_length = int(target_lengths.max().item())
    relative_cap = int(math.ceil(max_gt_length * multiplier)) + int(buffer)
    caps = [relative_cap, int(maxlen)]
    if configured_max_length is not None:
        caps.append(int(configured_max_length))
    return max(1, min(caps))


def _encoder_expand_size(generation_settings: GenerationSettings) -> int:
    num_beams = max(1, int(getattr(generation_settings, "num_beams", 1)))
    _require_single_return_sequence(generation_settings)
    if num_beams > 1:
        return num_beams
    return 1


def _require_single_return_sequence(generation_settings: GenerationSettings) -> None:
    num_return_sequences = int(getattr(generation_settings, "num_return_sequences", 1))
    if num_return_sequences != 1:
        raise ValueError(
            "validation decoding requires num_return_sequences=1; metrics and prediction "
            "records expect exactly one generated sequence per input sample."
        )


def _expand_encoder_outputs(
    encoder_outputs: VisionFrontendOutput,
    expand_size: int,
) -> VisionFrontendOutput:
    return VisionFrontendOutput(
        encoder_tokens_raw=encoder_outputs.encoder_tokens_raw.repeat_interleave(
            expand_size,
            dim=0,
        ),
        encoder_tokens_pos=encoder_outputs.encoder_tokens_pos.repeat_interleave(
            expand_size,
            dim=0,
        ),
        encoder_attention_mask=(
            None
            if encoder_outputs.encoder_attention_mask is None
            else encoder_outputs.encoder_attention_mask.repeat_interleave(expand_size, dim=0)
        ),
    )
