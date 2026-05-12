from __future__ import annotations

from pathlib import Path

import pytest

from src.grammar.rhythm_rule import RhythmRule

_FIXTURE_DIR = Path("tests/fixtures/rhythm_rule_polish_regressions")


def _load_fixture(path: Path) -> tuple[int, list[str]]:
    expected_failure_line: int | None = None
    content_lines: list[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.startswith("!! expected_failure_line="):
            expected_failure_line = int(raw_line.split("=", 1)[1])
            continue
        if raw_line.startswith("!!"):
            continue
        content_lines.append(raw_line)

    if expected_failure_line is None:
        raise AssertionError(f"Fixture {path} is missing expected failure metadata")

    return expected_failure_line, content_lines


@pytest.mark.parametrize(
    "fixture_name",
    sorted(path.name for path in _FIXTURE_DIR.glob("*.krn")),
)
def test_polish_regression_excerpt_becomes_unclosable_at_expected_line(
    fixture_name: str,
) -> None:
    expected_failure_line, lines = _load_fixture(_FIXTURE_DIR / fixture_name)
    rule = RhythmRule()

    for line_number, line in enumerate(lines, start=1):
        fields = tuple(line.split("\t"))
        can_close = rule.can_close_line(fields)

        if line_number == expected_failure_line:
            assert can_close is False, (
                f"{fixture_name} should first fail at line {expected_failure_line}, "
                f"but accepted: {line!r}"
            )
            return

        assert can_close is True, (
            f"{fixture_name} failed too early at line {line_number}: {line!r}"
        )
        rule.on_line_closed(fields)

    raise AssertionError(f"{fixture_name} never hit its expected failing line")
