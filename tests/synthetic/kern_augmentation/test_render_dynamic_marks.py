"""Tests for render-only terraced dynamic-mark augmentation."""

from __future__ import annotations

from scripts.dataset_generation.augmentation.render_dynamic_marks import apply_render_dynamic_marks

_DYNAMIC_MARKS = {"pp", "p", "mp", "mf", "f", "ff", "sf", "z"}


def test_apply_render_dynamic_marks_appends_trailing_dynam(simple_kern: str) -> None:
    result = apply_render_dynamic_marks(
        simple_kern,
        sample_probability=1.0,
        min_marks=1,
        max_marks=1,
    )
    lines = result.splitlines()
    assert lines[0].endswith("\t**dynam")

    dynamic_tokens = [line.split("\t")[-1] for line in lines if line and not line.startswith(("*", "=", "!"))]
    assert any(token in _DYNAMIC_MARKS for token in dynamic_tokens)


def test_apply_render_dynamic_marks_noop_when_probability_is_zero(simple_kern: str) -> None:
    result = apply_render_dynamic_marks(
        simple_kern,
        sample_probability=0.0,
        min_marks=1,
        max_marks=2,
    )
    assert result == simple_kern


def test_apply_render_dynamic_marks_uses_existing_trailing_dynam() -> None:
    krn = "\n".join(
        [
            "**kern\t**dynam",
            "*M4/4\t*",
            "4c\t<",
            "4d\t.",
            "4e\t.",
            "=\t=",
            "*-\t*-",
        ]
    )
    result = apply_render_dynamic_marks(
        krn,
        sample_probability=1.0,
        min_marks=2,
        max_marks=2,
    )
    lines = result.splitlines()
    assert lines[0].count("**dynam") == 1
    assert "4c\t<" in result
    assert any(
        line.endswith("\tpp")
        or line.endswith("\tp")
        or line.endswith("\tmp")
        or line.endswith("\tmf")
        or line.endswith("\tf")
        or line.endswith("\tff")
        or line.endswith("\tsf")
        or line.endswith("\tz")
        for line in lines
    )


def test_apply_render_dynamic_marks_noop_without_eligible_data() -> None:
    krn = "\n".join(
        [
            "**kern",
            "*M4/4",
            ".",
            "=",
            "*-",
        ]
    )
    result = apply_render_dynamic_marks(
        krn,
        sample_probability=1.0,
        min_marks=1,
        max_marks=2,
    )
    assert result == krn
