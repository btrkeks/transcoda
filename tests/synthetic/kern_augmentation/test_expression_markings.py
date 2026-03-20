"""Tests for expression-marking augmentation."""

from __future__ import annotations

import random

from scripts.dataset_generation.augmentation.expression_markings import apply_expression_markings


def _has_local_spine_mismatch(krn: str) -> bool:
    """Check inserted expression lines against neighboring active spine widths."""
    lines = krn.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("*>"):
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


def test_apply_expression_markings_uses_local_spine_count_with_spine_splits() -> None:
    krn = """*clefF4\t*clefG2
*M4/4\t*M4/4
*\t*^
4C\t4e\t4g
=\t=\t=
4D\t4f\t4a
=\t=\t=
*-\t*-\t*-
"""
    result = apply_expression_markings(krn, rng=random.Random(0))
    assert "*>" in result
    assert not _has_local_spine_mismatch(result)


def test_apply_expression_markings_is_deterministic_with_seeded_rng(simple_kern: str) -> None:
    first = apply_expression_markings(simple_kern, rng=random.Random(1234))
    second = apply_expression_markings(simple_kern, rng=random.Random(1234))
    assert first == second
