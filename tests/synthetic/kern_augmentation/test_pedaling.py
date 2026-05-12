"""Tests for pedaling augmentation."""

from __future__ import annotations

import random

from scripts.dataset_generation.augmentation.pedaling import apply_pedaling


def _has_local_spine_mismatch(krn: str) -> bool:
    """Check inserted pedal lines against neighboring active spine widths."""
    lines = krn.splitlines()
    for i, line in enumerate(lines):
        if not (line.startswith("*ped") or line.startswith("*Xped")):
            continue
        count = line.count("\t") + 1
        prev = next(
            (lines[j] for j in range(i - 1, -1, -1) if lines[j].strip() and not lines[j].startswith("!")),
            None,
        )
        nxt = next(
            (
                lines[j]
                for j in range(i + 1, len(lines))
                if lines[j].strip() and not lines[j].startswith("!")
            ),
            None,
        )
        if prev is not None and count != prev.count("\t") + 1:
            return True
        if nxt is not None and count != nxt.count("\t") + 1:
            return True
    return False


def test_apply_pedaling_uses_local_spine_count_with_spine_splits() -> None:
    krn = """*clefF4\t*clefG2
*M4/4\t*M4/4
*\t*^
4C\t4e\t4g
=\t=\t=
4D\t4f\t4a
=\t=\t=
*-\t*-\t*-
"""
    result = apply_pedaling(krn, measures_probability=1.0)
    assert "*ped" in result
    assert "*Xped" in result
    assert not _has_local_spine_mismatch(result)


def test_apply_pedaling_preserves_constant_spine_layout(simple_kern: str) -> None:
    result = apply_pedaling(simple_kern, measures_probability=1.0)
    assert "*ped" in result
    assert "*Xped" in result
    assert not _has_local_spine_mismatch(result)


def test_apply_pedaling_is_deterministic_with_seeded_rng(simple_kern: str) -> None:
    first = apply_pedaling(simple_kern, measures_probability=0.6, rng=random.Random(77))
    second = apply_pedaling(simple_kern, measures_probability=0.6, rng=random.Random(77))
    assert first == second
