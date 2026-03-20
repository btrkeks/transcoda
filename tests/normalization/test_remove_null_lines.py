"""Tests for RemoveNullLines pass."""

import sys
from pathlib import Path

# Add parent directory to path for imports when running as a test
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.dataset_generation.normalization import NormalizationContext
from scripts.dataset_generation.normalization.passes import RemoveNullLines


class TestRemoveNullLines:
    """Tests for RemoveNullLines pass."""

    def test_pass_exists(self):
        """RemoveNullLines pass should be instantiable."""
        pass_obj = RemoveNullLines()
        assert pass_obj.name == "remove_null_lines"

    def test_removes_single_spine_null_line(self):
        """Should remove a line containing only a single dot."""
        pass_obj = RemoveNullLines()
        ctx = NormalizationContext()

        input_text = "."
        expected = ""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_removes_multi_spine_all_null_line(self):
        """Should remove a line where all spines are null tokens."""
        pass_obj = RemoveNullLines()
        ctx = NormalizationContext()

        input_text = ".\t.\t."
        expected = ""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_mixed_spine_line(self):
        """Should preserve lines that have at least one non-null token."""
        pass_obj = RemoveNullLines()
        ctx = NormalizationContext()

        input_text = ".\t4c\t."
        expected = ".\t4c\t."

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_removes_null_only_lines_from_multiline(self):
        """Should remove null-only lines while preserving other lines."""
        pass_obj = RemoveNullLines()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern
*clefF4\t*clefG2
.\t.
4c 4e 4g\t4cc 4ee
.\t.\t.
4F# 4A\t4f# 4a"""

        expected = """**kern\t**kern
*clefF4\t*clefG2
4c 4e 4g\t4cc 4ee
4F# 4A\t4f# 4a"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_lines_with_content(self):
        """Should preserve all lines that contain actual musical content."""
        pass_obj = RemoveNullLines()
        ctx = NormalizationContext()

        input_text = """4c\t4e
4d\t4f
4e\t4g"""

        expected = """4c\t4e
4d\t4f
4e\t4g"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_handles_empty_input(self):
        """Should handle empty string input."""
        pass_obj = RemoveNullLines()
        ctx = NormalizationContext()

        input_text = ""
        expected = ""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_removes_all_lines_if_all_null(self):
        """Should result in empty string if all lines are null-only."""
        pass_obj = RemoveNullLines()
        ctx = NormalizationContext()

        input_text = """.\t.
.
.\t.\t."""

        expected = ""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_interpretations(self):
        """Should preserve interpretation lines (starting with *)."""
        pass_obj = RemoveNullLines()
        ctx = NormalizationContext()

        input_text = """*clefF4\t*clefG2
.
*k[b-]\t*k[b-]"""

        expected = """*clefF4\t*clefG2
*k[b-]\t*k[b-]"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_barlines(self):
        """Should preserve barline tokens."""
        pass_obj = RemoveNullLines()
        ctx = NormalizationContext()

        input_text = """4c\t4e
=\t=
.
4d\t4f"""

        expected = """4c\t4e
=\t=
4d\t4f"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_handles_consecutive_null_lines(self):
        """Should remove multiple consecutive null-only lines."""
        pass_obj = RemoveNullLines()
        ctx = NormalizationContext()

        input_text = """4c\t4e
.
.\t.
.\t.\t.
4d\t4f"""

        expected = """4c\t4e
4d\t4f"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_real_world_example(self):
        """Should handle real-world example from buggy_samples."""
        pass_obj = RemoveNullLines()
        ctx = NormalizationContext()

        input_text = """*clefF4\t*clefG2\t*clefG2
*k[b-e-a-d-]\t*k[b-e-a-d-]\t*k[b-e-a-d-]
*M2/2\t*M2/2\t*M2/2
.\t.\t.
6D-\t2dd-\t1a-]
8F-\t.\t.
6D-\t.\t."""

        expected = """*clefF4\t*clefG2\t*clefG2
*k[b-e-a-d-]\t*k[b-e-a-d-]\t*k[b-e-a-d-]
*M2/2\t*M2/2\t*M2/2
6D-\t2dd-\t1a-]
8F-\t.\t.
6D-\t.\t."""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_single_note_in_spine(self):
        """Should preserve lines where at least one spine has a note."""
        pass_obj = RemoveNullLines()
        ctx = NormalizationContext()

        input_text = ".\t.\t4c"
        expected = ".\t.\t4c"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected
