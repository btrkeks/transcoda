"""Tests for parallel filtering pipeline."""

import tempfile
from pathlib import Path

import pytest

from scripts.dataset_generation.filtering import FilterPipeline
from scripts.dataset_generation.filtering.filters import (
    RationalDurationFilter,
    TerminationFilter,
    UTF8Filter,
)
from scripts.dataset_generation.filtering.stats import FileCheckResult


# Valid kern content that passes all filters
VALID_KERN = """\
**kern
*clefG2
*M4/4
4c
4d
4e
4f
=1
*-
"""

# Invalid UTF-8 (will be written as bytes)
INVALID_UTF8_BYTES = b"\xff\xfe invalid utf8"

# Missing terminator
MISSING_TERMINATOR = """\
**kern
*clefG2
4c
4d
"""

# Contains rational duration
RATIONAL_DURATION = """\
**kern
*clefG2
2%5r
*-
"""


@pytest.fixture
def temp_kern_dir():
    """Create a temporary directory with kern files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create valid files
        (tmpdir_path / "valid1.krn").write_text(VALID_KERN, encoding="utf-8")
        (tmpdir_path / "valid2.krn").write_text(VALID_KERN, encoding="utf-8")
        (tmpdir_path / "valid3.krn").write_text(VALID_KERN, encoding="utf-8")

        # Create file that fails termination check
        (tmpdir_path / "bad_termination.krn").write_text(
            MISSING_TERMINATOR, encoding="utf-8"
        )

        # Create file with rational duration
        (tmpdir_path / "rational.krn").write_text(RATIONAL_DURATION, encoding="utf-8")

        yield tmpdir_path


@pytest.fixture
def temp_kern_dir_with_invalid_utf8():
    """Create a temporary directory with a mix of files including invalid UTF-8."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create valid files
        (tmpdir_path / "valid1.krn").write_text(VALID_KERN, encoding="utf-8")

        # Create invalid UTF-8 file
        (tmpdir_path / "bad_utf8.krn").write_bytes(INVALID_UTF8_BYTES)

        yield tmpdir_path


class TestFileCheckResult:
    """Tests for the FileCheckResult dataclass."""

    def test_path_property(self):
        """Test that path property returns a Path object."""
        result = FileCheckResult(
            path_str="/some/path/file.krn",
            passed=True,
            rejecting_filter=None,
            filter_outcomes={},
        )
        assert result.path == Path("/some/path/file.krn")
        assert isinstance(result.path, Path)

    def test_passed_true(self):
        """Test FileCheckResult with passing file."""
        result = FileCheckResult(
            path_str="/file.krn",
            passed=True,
            rejecting_filter=None,
            filter_outcomes={"utf8": ("passed", None), "termination": ("passed", None)},
        )
        assert result.passed is True
        assert result.rejecting_filter is None

    def test_passed_false(self):
        """Test FileCheckResult with failing file."""
        result = FileCheckResult(
            path_str="/file.krn",
            passed=False,
            rejecting_filter="termination",
            filter_outcomes={
                "utf8": ("passed", None),
                "termination": ("failed", "Missing spine terminators"),
            },
        )
        assert result.passed is False
        assert result.rejecting_filter == "termination"


class TestParallelVsSequential:
    """Tests that parallel and sequential produce identical results."""

    def test_same_passing_files(self, temp_kern_dir):
        """Parallel and sequential should yield the same passing files."""
        filters = [UTF8Filter(), TerminationFilter(), RationalDurationFilter()]
        paths = sorted(temp_kern_dir.glob("*.krn"))

        # Sequential
        pipeline_seq = FilterPipeline(filters)
        passing_seq = set(pipeline_seq.filter_files(paths))

        # Parallel
        pipeline_par = FilterPipeline(filters)
        passing_par = set(pipeline_par.filter_files_parallel(paths, workers=2))

        assert passing_seq == passing_par

    def test_same_stats_counts(self, temp_kern_dir):
        """Parallel and sequential should have the same stats counts."""
        filters = [UTF8Filter(), TerminationFilter(), RationalDurationFilter()]
        paths = sorted(temp_kern_dir.glob("*.krn"))

        # Sequential - must consume iterator to populate stats
        pipeline_seq = FilterPipeline(filters)
        list(pipeline_seq.filter_files(paths))
        stats_seq = pipeline_seq.stats

        # Parallel
        pipeline_par = FilterPipeline(filters)
        list(pipeline_par.filter_files_parallel(paths, workers=2))
        stats_par = pipeline_par.stats

        assert stats_seq.total_input == stats_par.total_input
        assert stats_seq.total_passed == stats_par.total_passed
        assert stats_seq.total_failed == stats_par.total_failed

    def test_same_rejection_counts(self, temp_kern_dir):
        """Parallel and sequential should have the same rejection by filter counts."""
        filters = [UTF8Filter(), TerminationFilter(), RationalDurationFilter()]
        paths = sorted(temp_kern_dir.glob("*.krn"))

        # Sequential
        pipeline_seq = FilterPipeline(filters)
        list(pipeline_seq.filter_files(paths))
        rejections_seq = pipeline_seq.stats.rejection_by_filter

        # Parallel
        pipeline_par = FilterPipeline(filters)
        list(pipeline_par.filter_files_parallel(paths, workers=2))
        rejections_par = pipeline_par.stats.rejection_by_filter

        assert rejections_seq == rejections_par


class TestStatsAggregation:
    """Tests for stats aggregation in parallel mode."""

    def test_per_filter_stats_populated(self, temp_kern_dir):
        """Per-filter stats should be populated correctly."""
        filters = [UTF8Filter(), TerminationFilter()]
        paths = sorted(temp_kern_dir.glob("*.krn"))

        pipeline = FilterPipeline(filters)
        list(pipeline.filter_files_parallel(paths, workers=2))

        stats = pipeline.stats
        filter_names = {f.name for f in stats.per_filter}
        assert filter_names == {"utf8", "termination"}

    def test_failure_reasons_aggregated(self, temp_kern_dir):
        """Failure reasons should be aggregated across workers."""
        filters = [TerminationFilter()]
        paths = sorted(temp_kern_dir.glob("*.krn"))

        pipeline = FilterPipeline(filters)
        list(pipeline.filter_files_parallel(paths, workers=2))

        # Check that the termination filter recorded a failure
        term_stats = next(f for f in pipeline.stats.per_filter if f.name == "termination")
        assert term_stats.files_failed > 0
        assert len(term_stats.failure_reasons) > 0


class TestErrorHandling:
    """Tests for error handling in parallel mode."""

    def test_invalid_utf8_handled(self, temp_kern_dir_with_invalid_utf8):
        """Invalid UTF-8 files should be handled gracefully."""
        filters = [UTF8Filter(), TerminationFilter()]
        paths = sorted(temp_kern_dir_with_invalid_utf8.glob("*.krn"))

        pipeline = FilterPipeline(filters)
        passing = list(pipeline.filter_files_parallel(paths, workers=2))

        # Only valid1.krn should pass
        assert len(passing) == 1
        assert passing[0].name == "valid1.krn"

        # Stats should show the failure
        stats = pipeline.stats
        assert stats.total_failed == 1

    def test_continues_after_file_error(self, temp_kern_dir):
        """Pipeline should continue processing after a file fails."""
        filters = [UTF8Filter(), TerminationFilter()]
        paths = sorted(temp_kern_dir.glob("*.krn"))

        pipeline = FilterPipeline(filters)
        passing = list(pipeline.filter_files_parallel(paths, workers=2))

        # Should have processed all files
        assert pipeline.stats.total_input == len(paths)
        # Some should pass (the valid ones)
        assert len(passing) >= 1


class TestFilterConfigs:
    """Tests for the _get_filter_configs method."""

    def test_simple_filters(self):
        """Simple filters should return empty params."""
        filters = [UTF8Filter(), TerminationFilter()]
        pipeline = FilterPipeline(filters)

        configs = pipeline._get_filter_configs()

        assert configs == [("utf8", {}), ("termination", {})]

    def test_rhythm_filter_params(self):
        """RhythmFilter should include its params in config."""
        # Skip if rhythm_checker binary doesn't exist
        from scripts.dataset_generation.filtering.filters import RhythmFilter

        filters = [
            RhythmFilter(
                binary_path="/custom/path",
                allow_anacrusis=False,
                allow_incomplete_final=True,
            )
        ]
        pipeline = FilterPipeline(filters)

        configs = pipeline._get_filter_configs()

        assert len(configs) == 1
        name, params = configs[0]
        assert name == "rhythm"
        assert params["binary_path"] == "/custom/path"
        assert params["allow_anacrusis"] is False
        assert params["allow_incomplete_final"] is True


class TestParallelWithManyFiles:
    """Tests with larger numbers of files."""

    def test_many_files(self):
        """Test parallel processing with many files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create 20 valid files
            for i in range(20):
                (tmpdir_path / f"valid{i:02d}.krn").write_text(VALID_KERN, encoding="utf-8")

            # Create 5 invalid files
            for i in range(5):
                (tmpdir_path / f"bad{i:02d}.krn").write_text(
                    MISSING_TERMINATOR, encoding="utf-8"
                )

            paths = sorted(tmpdir_path.glob("*.krn"))
            filters = [UTF8Filter(), TerminationFilter()]

            pipeline = FilterPipeline(filters)
            passing = list(pipeline.filter_files_parallel(paths, workers=4))

            assert len(passing) == 20
            assert pipeline.stats.total_passed == 20
            assert pipeline.stats.total_failed == 5

    def test_single_worker_works(self, temp_kern_dir):
        """Test that workers=1 still works correctly in parallel mode."""
        filters = [UTF8Filter(), TerminationFilter()]
        paths = sorted(temp_kern_dir.glob("*.krn"))

        pipeline = FilterPipeline(filters)
        list(pipeline.filter_files_parallel(paths, workers=1))

        # Should still work
        assert pipeline.stats.total_input == len(paths)
