"""Tests for the header clef filter."""

import tempfile
from pathlib import Path

import pytest

from scripts.dataset_generation.filtering.base import FilterContext
from scripts.dataset_generation.filtering.filters.header_clef import HeaderClefFilter


@pytest.fixture
def filter():
    return HeaderClefFilter()


def _check(filter, content: str):
    """Write content to a temp file and run the filter."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".krn", delete=False) as f:
        f.write(content)
        f.flush()
        return filter.check(Path(f.name), FilterContext())


class TestHeaderClefFilter:
    def test_pass_basic(self, filter):
        content = "**kern\n*clefG2\n*M4/4\n4c\n=1\n*-\n"
        result = _check(filter, content)
        assert result.passed

    def test_pass_two_spines(self, filter):
        content = "**kern\t**kern\n*clefF4\t*clefG2\n*M4/4\t*M4/4\n4C\t4c\n=1\t=1\n*-\t*-\n"
        result = _check(filter, content)
        assert result.passed

    def test_pass_clef_with_keysig_timesig(self, filter):
        content = "**kern\n*clefG2\n*k[f#c#]\n*M3/4\n4c\n=1\n*-\n"
        result = _check(filter, content)
        assert result.passed

    def test_pass_ekern(self, filter):
        content = "**ekern\t**ekern\n*clefF4\t*clefG2\n4C\t4c\n*-\t*-\n"
        result = _check(filter, content)
        assert result.passed

    def test_fail_no_clef_barline_first(self, filter):
        content = "**kern\n=1\n4c\n4d\n*-\n"
        result = _check(filter, content)
        assert not result.passed
        assert result.reason == "Missing header clef declaration"
        assert result.details["missing_spines"] == [0]

    def test_fail_no_clef_notes_first(self, filter):
        content = "**kern\n4c\n4d\n4e\n*-\n"
        result = _check(filter, content)
        assert not result.passed
        assert result.reason == "Missing header clef declaration"

    def test_fail_partial_clef(self, filter):
        content = "**kern\t**kern\n*clefG2\t*\n4c\t4e\n*-\t*-\n"
        result = _check(filter, content)
        assert not result.passed
        assert result.details["missing_spines"] == [1]

    def test_fail_no_spine_declaration(self, filter):
        content = "*clefG2\n4c\n*-\n"
        result = _check(filter, content)
        assert not result.passed
        assert result.reason == "No spine declaration found"

    def test_fail_no_kern_spine_declaration(self, filter):
        content = "**dynam\t**text\n*\t*\nmf\t.\n*-\t*-\n"
        result = _check(filter, content)
        assert not result.passed
        assert result.reason == "No spine declaration found"

    def test_fail_empty_file(self, filter):
        result = _check(filter, "")
        assert not result.passed

    def test_pass_clef_after_other_tandem(self, filter):
        """Clef can appear after other tandem interpretations like stria."""
        content = "**kern\n*stria5\n*clefG2\n*M4/4\n4c\n*-\n"
        result = _check(filter, content)
        assert result.passed

    def test_pass_mixed_spines_kern_clefs_present(self, filter):
        content = (
            "**kern\t**dynam\t**kern\t**text\n"
            "*\t*\t*\t*\n"
            "*clefF4\t*\t*clefG2\t*\n"
            "*M4/4\t*\t*M4/4\t*\n"
            "4C\tmf\t4c\tlyric\n"
            "*-\t*-\t*-\t*-\n"
        )
        result = _check(filter, content)
        assert result.passed

    def test_fail_mixed_spines_missing_kern_clef(self, filter):
        content = (
            "**kern\t**dynam\t**kern\t**text\n"
            "*\t*\t*\t*\n"
            "*clefF4\t*\t*\t*\n"
            "4C\tmf\t4c\tlyric\n"
            "*-\t*-\t*-\t*-\n"
        )
        result = _check(filter, content)
        assert not result.passed
        # Missing clef on the second **kern column (index 2 in full token row).
        assert result.details["missing_spines"] == [2]
