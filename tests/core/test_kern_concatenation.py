from pathlib import Path

import pytest

from src.core.kern_concatenation import (
    restore_terminal_spine_count_before_final_barline,
    summarize_spine_topology,
)
from src.core.kern_utils import is_bar_line, is_spinemerge_line

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "kern_concatenation"


def _fixture_case_dirs() -> list[Path]:
    if not _FIXTURE_ROOT.exists():
        return []
    case_dirs = sorted(path for path in _FIXTURE_ROOT.iterdir() if path.is_dir())
    for case_dir in case_dirs:
        assert (case_dir / "input.krn").exists(), f"Missing input fixture in {case_dir.name}"
        assert (case_dir / "expected.krn").exists(), f"Missing expected fixture in {case_dir.name}"
    return case_dirs


def _read_case_pair(case_dir: Path) -> tuple[str, str]:
    original = (case_dir / "input.krn").read_text(encoding="utf-8")
    expected = (case_dir / "expected.krn").read_text(encoding="utf-8")
    return original, expected


def _last_non_empty_lines(text: str, count: int) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-count:]


def test_summarize_spine_topology_tracks_terminal_split_width():
    text = "\n".join(
        [
            "**kern",
            "*clefG2",
            "=1",
            "4c",
            "*^",
            "*\t*",
            "=2\t=2",
            "4d\t4f",
            "*-\t*-",
        ]
    )

    topology = summarize_spine_topology(text)

    assert topology.initial_spine_count == 1
    assert topology.terminal_spine_count == 2


@pytest.mark.parametrize("case_dir", _fixture_case_dirs(), ids=lambda path: path.name)
def test_kern_concatenation_fixtures_document_expected_contract(case_dir: Path):
    original, expected = _read_case_pair(case_dir)

    original_topology = summarize_spine_topology(original)
    expected_topology = summarize_spine_topology(expected)
    expected_tail = _last_non_empty_lines(expected, 2)

    assert original_topology.initial_spine_count is not None
    assert original_topology.terminal_spine_count is not None
    assert original_topology.initial_spine_count < original_topology.terminal_spine_count
    assert expected_topology.initial_spine_count == expected_topology.terminal_spine_count
    assert len(expected_tail) == 2
    assert is_spinemerge_line(expected_tail[0])
    assert is_bar_line(expected_tail[1])


@pytest.mark.parametrize("case_dir", _fixture_case_dirs(), ids=lambda path: path.name)
def test_restore_terminal_spine_count_matches_golden_master_examples(case_dir: Path):
    original, expected = _read_case_pair(case_dir)

    actual = restore_terminal_spine_count_before_final_barline(original)

    assert actual.rstrip("\n") == expected.rstrip("\n")


def test_restore_terminal_spine_count_collapses_trailing_terminator_width():
    original = "\n".join(
        [
            "**kern",
            "*clefG2",
            "=1",
            "4c",
            "*^",
            "*\t*",
            "=2\t=2",
            "4d\t4f",
            "==\t==",
            "*-\t*-",
        ]
    )
    expected = "\n".join(
        [
            "**kern",
            "*clefG2",
            "=1",
            "4c",
            "*^",
            "*\t*",
            "=2\t=2",
            "4d\t4f",
            "*v\t*v",
            "==",
            "*-",
        ]
    )

    actual = restore_terminal_spine_count_before_final_barline(original)

    assert actual == expected
