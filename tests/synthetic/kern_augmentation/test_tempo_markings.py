"""Tests for tempo marking augmentation."""

from __future__ import annotations

import pytest

from scripts.dataset_generation.augmentation.tempo_markings import (
    apply_tempo_markings,
    has_tempo_markings,
)


def test_apply_tempo_markings_adds_omd_and_mm(simple_kern: str) -> None:
    result = apply_tempo_markings(simple_kern, include_mm_probability=1.0)

    assert result.startswith("!!!OMD:")
    mm_line = next(line for line in result.splitlines() if "*MM" in line)
    assert len(mm_line.split("\t")) == 2
    assert mm_line.split("\t")[1] == "*"


def test_apply_tempo_markings_adds_only_omd_when_mm_probability_zero(simple_kern: str) -> None:
    result = apply_tempo_markings(simple_kern, include_mm_probability=0.0)

    assert result.startswith("!!!OMD:")
    assert "*MM" not in result


def test_apply_tempo_markings_skips_when_omd_already_exists(simple_kern: str) -> None:
    source = "!!!OMD: Allegro\n" + simple_kern
    result = apply_tempo_markings(source, include_mm_probability=1.0)

    assert result == source


def test_apply_tempo_markings_skips_when_mm_already_exists(simple_kern: str) -> None:
    source = simple_kern.replace("*M4/4\t*M4/4", "*M4/4\t*M4/4\n*MM90\t*")
    result = apply_tempo_markings(source, include_mm_probability=1.0)

    assert result == source


def test_has_tempo_markings_detects_omd_and_textual_mm(simple_kern: str) -> None:
    with_omd = "!!!OMD: Andante\n" + simple_kern
    with_text_mm = simple_kern.replace("*M4/4\t*M4/4", "*M4/4\t*M4/4\n*MMAllegro\t*")

    assert has_tempo_markings(with_omd) is True
    assert has_tempo_markings(with_text_mm) is True
    assert has_tempo_markings(simple_kern) is False


@pytest.mark.parametrize("invalid_probability", [-0.1, 1.1])
def test_apply_tempo_markings_rejects_invalid_probability(
    simple_kern: str,
    invalid_probability: float,
) -> None:
    with pytest.raises(ValueError, match="include_mm_probability must be in \\[0.0, 1.0\\]"):
        apply_tempo_markings(simple_kern, include_mm_probability=invalid_probability)
