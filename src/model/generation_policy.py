from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from src.artifacts import DecodingSpec
from src.config import Generation as GenerationConfig


@dataclass(frozen=True)
class GenerationSettings:
    """Normalized decoding settings shared across all generation call sites."""

    strategy: str
    num_beams: int
    length_penalty: float
    repetition_penalty: float
    early_stopping: bool | str
    num_return_sequences: int
    use_cache: bool
    do_sample: bool


def _validate_and_build(
    *,
    strategy: str,
    num_beams: int,
    length_penalty: float,
    repetition_penalty: float,
    early_stopping: bool | str,
    num_return_sequences: int,
    use_cache: bool,
    do_sample: bool,
) -> GenerationSettings:
    if strategy not in {"greedy", "beam"}:
        raise ValueError(f"Unsupported generation strategy: {strategy}")
    if do_sample:
        raise ValueError("Sampling is disabled for deterministic OMR decoding")
    if strategy == "greedy":
        num_beams = 1
    elif num_beams <= 1:
        raise ValueError("num_beams must be > 1 for beam search")
    if num_return_sequences < 1:
        raise ValueError("num_return_sequences must be >= 1")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be > 0")

    return GenerationSettings(
        strategy=strategy,
        num_beams=int(num_beams),
        length_penalty=float(length_penalty),
        repetition_penalty=float(repetition_penalty),
        early_stopping=early_stopping,
        num_return_sequences=int(num_return_sequences),
        use_cache=bool(use_cache),
        do_sample=bool(do_sample),
    )


def settings_from_generation_config(config: GenerationConfig) -> GenerationSettings:
    return _validate_and_build(
        strategy=config.strategy,
        num_beams=config.num_beams,
        length_penalty=config.length_penalty,
        repetition_penalty=config.repetition_penalty,
        early_stopping=config.early_stopping,
        num_return_sequences=config.num_return_sequences,
        use_cache=config.use_cache,
        do_sample=config.do_sample,
    )


def settings_from_decoding_spec(decoding: DecodingSpec) -> GenerationSettings:
    strategy = decoding.strategy
    num_beams = decoding.num_beams
    if num_beams is None:
        num_beams = 1 if strategy == "greedy" else 4

    length_penalty = 1.0 if decoding.length_penalty is None else decoding.length_penalty

    return _validate_and_build(
        strategy=strategy,
        num_beams=int(num_beams),
        length_penalty=float(length_penalty),
        repetition_penalty=float(decoding.repetition_penalty),
        early_stopping=decoding.early_stopping,
        num_return_sequences=decoding.num_return_sequences,
        use_cache=decoding.use_cache,
        do_sample=decoding.do_sample,
    )


def apply_generation_overrides(
    settings: GenerationSettings,
    *,
    strategy: str | None = None,
    num_beams: int | None = None,
    length_penalty: float | None = None,
    repetition_penalty: float | None = None,
    early_stopping: bool | str | None = None,
    num_return_sequences: int | None = None,
    use_cache: bool | None = None,
) -> GenerationSettings:
    updated = replace(
        settings,
        strategy=settings.strategy if strategy is None else strategy,
        num_beams=settings.num_beams if num_beams is None else num_beams,
        length_penalty=settings.length_penalty if length_penalty is None else length_penalty,
        repetition_penalty=(
            settings.repetition_penalty if repetition_penalty is None else repetition_penalty
        ),
        early_stopping=settings.early_stopping if early_stopping is None else early_stopping,
        num_return_sequences=(
            settings.num_return_sequences if num_return_sequences is None else num_return_sequences
        ),
        use_cache=settings.use_cache if use_cache is None else use_cache,
    )
    return _validate_and_build(
        strategy=updated.strategy,
        num_beams=updated.num_beams,
        length_penalty=updated.length_penalty,
        repetition_penalty=updated.repetition_penalty,
        early_stopping=updated.early_stopping,
        num_return_sequences=updated.num_return_sequences,
        use_cache=updated.use_cache,
        do_sample=updated.do_sample,
    )


def enforce_constraint_safe_settings(
    settings: GenerationSettings,
    *,
    has_constraints: bool,
) -> GenerationSettings:
    """Force constrained decoding into a beam-safe deterministic mode."""
    if not has_constraints:
        return settings
    if (
        settings.strategy == "greedy"
        and settings.num_beams == 1
        and settings.num_return_sequences == 1
    ):
        return settings

    return _validate_and_build(
        strategy="greedy",
        num_beams=1,
        length_penalty=settings.length_penalty,
        repetition_penalty=settings.repetition_penalty,
        early_stopping=settings.early_stopping,
        num_return_sequences=1,
        use_cache=settings.use_cache,
        do_sample=settings.do_sample,
    )


def enforce_grammar_safe_settings(settings: GenerationSettings) -> GenerationSettings:
    """Backward-compatible alias for constrained decoding safety."""
    return enforce_constraint_safe_settings(settings, has_constraints=True)


def build_generate_kwargs(
    *,
    pixel_values: Any,
    image_sizes: Any,
    max_length: int | None,
    settings: GenerationSettings,
    logits_processor: list[Any] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "pixel_values": pixel_values,
        "image_sizes": image_sizes,
        "do_sample": settings.do_sample,
        "use_cache": settings.use_cache,
        "num_beams": settings.num_beams,
        "repetition_penalty": settings.repetition_penalty,
    }
    if settings.strategy == "beam":
        kwargs["length_penalty"] = settings.length_penalty
        kwargs["early_stopping"] = settings.early_stopping
        kwargs["num_return_sequences"] = settings.num_return_sequences
    if max_length is not None:
        kwargs["max_length"] = int(max_length)
    if logits_processor is not None:
        kwargs["logits_processor"] = logits_processor
    return kwargs
