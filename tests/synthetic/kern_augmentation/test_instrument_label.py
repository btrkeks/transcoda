"""Tests for instrument label augmentation."""

from __future__ import annotations

from scripts.dataset_generation.augmentation.instrument_label import apply_instrument_label_piano


def test_inserts_piano_line_after_kern_header() -> None:
    krn = "\n".join(
        [
            "**kern\t**kern",
            "*clefF4\t*clefG2",
            "*M4/4\t*M4/4",
            "4c\t4e",
            "*-\t*-",
        ]
    )

    result = apply_instrument_label_piano(krn)
    lines = result.splitlines()

    assert lines[0] == "**kern\t**kern"
    assert lines[1] == '*I"Piano\t*I"Piano'
    assert lines[2] == "*clefF4\t*clefG2"


def test_inserts_piano_line_at_top_when_kern_header_absent() -> None:
    krn = "\n".join(
        [
            "*clefF4\t*clefG2",
            "*M4/4\t*M4/4",
            "4c\t4e",
            "*-\t*-",
        ]
    )

    result = apply_instrument_label_piano(krn)
    lines = result.splitlines()

    assert lines[0] == '*I"Piano\t*I"Piano'
    assert lines[1] == "*clefF4\t*clefG2"


def test_noop_when_instrument_interpretation_already_present() -> None:
    krn = "\n".join(
        [
            "**kern\t**kern",
            '*I"Piano\t*I"Piano',
            "*clefF4\t*clefG2",
            "4c\t4e",
            "*-\t*-",
        ]
    )

    assert apply_instrument_label_piano(krn) == krn


def test_empty_input_is_unchanged() -> None:
    assert apply_instrument_label_piano("") == ""
