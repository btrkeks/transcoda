import pytest

from src.artifacts import DecodingSpec
from src.config import Generation
from src.model.generation_policy import (
    GenerationSettings,
    apply_generation_overrides,
    build_generate_kwargs,
    enforce_grammar_safe_settings,
    settings_from_decoding_spec,
    settings_from_generation_config,
)


def test_settings_from_generation_config_beam_defaults():
    cfg = Generation()
    settings = settings_from_generation_config(cfg)

    assert settings == GenerationSettings(
        strategy="beam",
        num_beams=4,
        length_penalty=1.0,
        repetition_penalty=1.1,
        early_stopping=True,
        num_return_sequences=1,
        use_cache=True,
        do_sample=False,
    )


def test_settings_from_decoding_spec_greedy():
    decoding = DecodingSpec(strategy="greedy", max_len=128, eos_token="</eos>")
    settings = settings_from_decoding_spec(decoding)

    assert settings.strategy == "greedy"
    assert settings.num_beams == 1
    assert settings.repetition_penalty == pytest.approx(1.1)


def test_apply_overrides_switches_to_beam():
    decoding = DecodingSpec(strategy="greedy", max_len=128, eos_token="</eos>")
    settings = settings_from_decoding_spec(decoding)
    updated = apply_generation_overrides(
        settings,
        strategy="beam",
        num_beams=4,
        length_penalty=0.8,
        repetition_penalty=1.25,
    )

    assert updated.strategy == "beam"
    assert updated.num_beams == 4
    assert updated.length_penalty == pytest.approx(0.8)
    assert updated.repetition_penalty == pytest.approx(1.25)


def test_build_generate_kwargs_for_beam():
    settings = GenerationSettings(
        strategy="beam",
        num_beams=4,
        length_penalty=1.2,
        repetition_penalty=1.1,
        early_stopping=True,
        num_return_sequences=1,
        use_cache=True,
        do_sample=False,
    )

    kwargs = build_generate_kwargs(
        pixel_values="pixel",
        image_sizes="sizes",
        max_length=64,
        settings=settings,
    )

    assert kwargs["num_beams"] == 4
    assert kwargs["length_penalty"] == pytest.approx(1.2)
    assert kwargs["repetition_penalty"] == pytest.approx(1.1)
    assert kwargs["max_length"] == 64


def test_invalid_override_raises():
    settings = GenerationSettings(
        strategy="greedy",
        num_beams=1,
        length_penalty=1.0,
        repetition_penalty=1.1,
        early_stopping=True,
        num_return_sequences=1,
        use_cache=True,
        do_sample=False,
    )
    with pytest.raises(ValueError, match="num_beams must be > 1"):
        apply_generation_overrides(settings, strategy="beam", num_beams=1)


def test_enforce_grammar_safe_settings_forces_greedy_from_beam():
    settings = GenerationSettings(
        strategy="beam",
        num_beams=4,
        length_penalty=1.0,
        repetition_penalty=1.1,
        early_stopping=True,
        num_return_sequences=1,
        use_cache=True,
        do_sample=False,
    )

    safe = enforce_grammar_safe_settings(settings)

    assert safe.strategy == "greedy"
    assert safe.num_beams == 1
    assert safe.num_return_sequences == 1
    assert safe.repetition_penalty == pytest.approx(1.1)


def test_enforce_grammar_safe_settings_noop_for_greedy_single():
    settings = GenerationSettings(
        strategy="greedy",
        num_beams=1,
        length_penalty=1.0,
        repetition_penalty=1.1,
        early_stopping=True,
        num_return_sequences=1,
        use_cache=True,
        do_sample=False,
    )

    assert enforce_grammar_safe_settings(settings) == settings


def test_generation_config_default_repetition_penalty():
    cfg = Generation()
    assert cfg.repetition_penalty == pytest.approx(1.1)


def test_invalid_repetition_penalty_raises():
    with pytest.raises(ValueError, match="repetition_penalty must be > 0"):
        Generation(repetition_penalty=0)
