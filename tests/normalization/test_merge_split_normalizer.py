"""Tests for MergeSplitNormalizer pass."""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports when running as a test
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.dataset_generation.normalization import NormalizationContext
from scripts.dataset_generation.normalization.passes import MergeSplitNormalizer


class TestMergeSplitNormalizer:
    """Tests for MergeSplitNormalizer pass."""

    def test_pass_exists(self):
        """MergeSplitNormalizer pass should be instantiable."""
        pass_obj = MergeSplitNormalizer()
        assert pass_obj.name == "merge_split_normalizer"

    def test_normalizes_pattern1_basic(self):
        """Should normalize pattern 1: *v\\t*v\\t*\\n=\\t=\\n*\\t*^ → =\\t=\\t=."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        input_text = "*v\t*v\t*\n=\t=\n*\t*^"
        expected = "=\t=\t="

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_normalizes_pattern2_basic(self):
        """Should normalize pattern 2: *\\t*v\\t*v\\n=\\t=\\n*-\\t*- → =\\t=\\t=\\n*-\\t*-\\t*-."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        input_text = "*\t*v\t*v\n=\t=\n*-\t*-"
        expected = "=\t=\t=\n*-\t*-\t*-"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_normalizes_both_patterns(self):
        """Should normalize both patterns when present in the same text."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern\t**kern
*v\t*v\t*
=\t=
*\t*^
4c\t4e\t4g
*\t*v\t*v
=\t=
*-\t*-"""

        expected = """**kern\t**kern\t**kern
=\t=\t=
4c\t4e\t4g
=\t=\t=
*-\t*-\t*-"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_text_without_patterns(self):
        """Should not modify text that doesn't contain the target patterns."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern
*clefF4\t*clefG2
4c\t4e
=\t=
4d\t4f
*-\t*-"""

        expected = input_text

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_handles_empty_input(self):
        """Should handle empty string input."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        input_text = ""
        expected = ""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_normalizes_pattern1_in_context(self):
        """Should normalize pattern 1 when surrounded by other kern notation."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern\t**kern
*clefF4\t*clefG2\t*clefG2
4c\t4e\t4g
*v\t*v\t*
=\t=
*\t*^
4d\t4f\t4a
*-\t*-\t*-"""

        expected = """**kern\t**kern\t**kern
*clefF4\t*clefG2\t*clefG2
4c\t4e\t4g
=\t=\t=
4d\t4f\t4a
*-\t*-\t*-"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_normalizes_pattern2_in_context(self):
        """Should normalize pattern 2 when surrounded by other kern notation."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern\t**kern
*clefF4\t*clefG2\t*clefG2
4c\t4e\t4g
*\t*v\t*v
=\t=
*-\t*-"""

        expected = """**kern\t**kern\t**kern
*clefF4\t*clefG2\t*clefG2
4c\t4e\t4g
=\t=\t=
*-\t*-\t*-"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_handles_multiple_occurrences_pattern1(self):
        """Should normalize multiple occurrences of pattern 1."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        input_text = """*v\t*v\t*
=\t=
*\t*^
4c\t4e\t4g
*v\t*v\t*
=\t=
*\t*^
4d\t4f\t4a"""

        expected = """=\t=\t=
4c\t4e\t4g
=\t=\t=
4d\t4f\t4a"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_handles_multiple_occurrences_pattern2(self):
        """Should normalize multiple occurrences of pattern 2."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        input_text = """4c\t4e\t4g
*\t*v\t*v
=\t=
*-\t*-"""

        expected = """4c\t4e\t4g
=\t=\t=
*-\t*-\t*-"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_2_spine_merge_split(self):
        """Should preserve 2-spine merge/split (doesn't restore original count)."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        # 2 spines merge to 1, barline, split to 2 - but original was 2, not 1
        # This is actually a valid normalization case: 2→1→2
        input_text = """*v\t*v
=
*^"""

        expected = "=\t="

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_real_world_example_pattern1(self):
        """Should handle realistic kern notation with pattern 1."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern\t**kern
*clefF4\t*clefG2\t*clefG2
*k[b-e-a-]\t*k[b-e-a-]\t*k[b-e-a-]
*M4/4\t*M4/4\t*M4/4
4C\t4e-\t4g
4D\t4f\t4a-
*v\t*v\t*
=\t=
*\t*^
2E-\t2g\t2b-
4F\t4a-\t4cc
*-\t*-\t*-"""

        expected = """**kern\t**kern\t**kern
*clefF4\t*clefG2\t*clefG2
*k[b-e-a-]\t*k[b-e-a-]\t*k[b-e-a-]
*M4/4\t*M4/4\t*M4/4
4C\t4e-\t4g
4D\t4f\t4a-
=\t=\t=
2E-\t2g\t2b-
4F\t4a-\t4cc
*-\t*-\t*-"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_real_world_example_pattern2(self):
        """Should handle realistic kern notation with pattern 2."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern\t**kern
*clefF4\t*clefG2\t*clefG2
*k[]\t*k[]\t*k[]
*M3/4\t*M3/4\t*M3/4
4C\t4e\t4g
4D\t4f\t4a
4E\t4g\t4b
*\t*v\t*v
=\t=
*-\t*-"""

        expected = """**kern\t**kern\t**kern
*clefF4\t*clefG2\t*clefG2
*k[]\t*k[]\t*k[]
*M3/4\t*M3/4\t*M3/4
4C\t4e\t4g
4D\t4f\t4a
4E\t4g\t4b
=\t=\t=
*-\t*-\t*-"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_users_original_example(self):
        """Should normalize the user's original example: *v *v * / = = / *^ *."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        # User's example: split before continue (*^ *)
        input_text = "*v\t*v\t*\n=\t=\n*^\t*"
        expected = "=\t=\t="

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    # New tests for N-spine patterns and edge cases

    def test_4_spine_merge_split(self):
        """Should handle 4-spine patterns."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        # 4 spines: merge first two, then merge result with third → 2 spines
        # Then barline, then split both → 4 spines
        input_text = "*v\t*v\t*\t*\n*v\t*v\n=\t=\n*^\t*^"
        expected = "=\t=\t=\t="

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_no_substring_matching_bug(self):
        """Should not create =^ tokens from substring matching."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        # The problematic pattern that caused the bug
        input_text = "*\t*v\t*v\n=\t=\n*^\t*"
        result = pass_obj.transform(input_text, ctx)

        assert "=^" not in result
        assert result == "=\t=\t="

    def test_chained_merges_4_to_2(self):
        """Should handle chained merges: 4→3→2→4."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        # 4 spines merge to 3, then 3 to 2, barline, split to 4
        input_text = "*v\t*v\t*\t*\n*\t*v\t*v\n=\t=\n*^\t*^"
        expected = "=\t=\t=\t="

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_mismatched_split_preserved(self):
        """Should preserve sequence when split doesn't restore original count."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        # 4 spines merge to 1, barline, split to 2 (not 4), so it should not collapse.
        input_text = "*v\t*v\t*v\t*v\n=\t=\n*^\t*"
        expected = "*v\t*v\t*v\t*v\n=\n*^\t*"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_triple_merge_when_following_split_does_not_restore_original_width(self):
        """Should use post-merge width for preserved triple-merge barlines."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        input_text = "*v\t*v\t*v\t*\n=\t=\n*\t*^"
        expected = "*v\t*v\t*v\t*\n=\t=\n*\t*^"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_preserves_trailing_triple_merge_before_non_split_content(self):
        """Should preserve a 5-spine triple-merge with a 3-field barline."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        input_text = "*\t*\t*v\t*v\t*v\n=\t=\t=\n4c\t4e\t4g"
        expected = input_text

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_normalizes_triple_merge_before_terminator(self):
        """Should restore original width for merge→barline→terminator."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        input_text = "*v\t*v\t*v\t*\n=\t=\n*-\t*-"
        expected = "=\t=\t=\t=\n*-\t*-\t*-\t*-"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_merge_without_barline_preserved(self):
        """Should preserve merge lines that aren't followed by barline."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        input_text = "*v\t*v\t*\n4c\t4e"
        expected = "*v\t*v\t*\n4c\t4e"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)

        assert result == expected

    def test_5_spine_merge_split(self):
        """Should handle 5-spine patterns."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        # 5 spines merge to 3, barline, split back to 5
        input_text = "*v\t*v\t*\t*v\t*v\n=\t=\t=\n*^\t*\t*^"
        expected = "=\t=\t=\t=\t="

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_merge_barline_no_split_preserved(self):
        """Should preserve merge→barline followed by non-split content."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        input_text = "*v\t*v\t*\n=\t=\n4c\t4e"
        expected = "*v\t*v\t*\n=\t=\n4c\t4e"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)

        assert result == expected

    def test_all_merge_split_variations(self):
        """Should normalize all 3-spine merge/split pattern variations."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        patterns = [
            # *v *v * (rightmost continues)
            ("*v\t*v\t*\n=\t=\n*\t*^", "=\t=\t="),
            ("*v\t*v\t*\n=\t=\n*^\t*", "=\t=\t="),
            # *v * *v (middle continues)
            ("*v\t*\t*v\n=\t=\n*\t*^", "=\t=\t="),
            ("*v\t*\t*v\n=\t=\n*^\t*", "=\t=\t="),
            # * *v *v (leftmost continues)
            ("*\t*v\t*v\n=\t=\n*\t*^", "=\t=\t="),
            ("*\t*v\t*v\n=\t=\n*^\t*", "=\t=\t="),
        ]

        for pattern, expected in patterns:
            pass_obj.prepare(pattern, ctx)
            result = pass_obj.transform(pattern, ctx)
            pass_obj.validate(result, ctx)
            assert result == expected, f"Failed for pattern: {pattern!r}"

    def test_all_merge_terminate_variations(self):
        """Should normalize all 3-spine merge/terminate pattern variations."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        patterns = [
            ("*v\t*v\t*\n=\t=\n*-\t*-", "=\t=\t=\n*-\t*-\t*-"),
            ("*v\t*\t*v\n=\t=\n*-\t*-", "=\t=\t=\n*-\t*-\t*-"),
            ("*\t*v\t*v\n=\t=\n*-\t*-", "=\t=\t=\n*-\t*-\t*-"),
        ]

        for pattern, expected in patterns:
            pass_obj.prepare(pattern, ctx)
            result = pass_obj.transform(pattern, ctx)
            pass_obj.validate(result, ctx)
            assert result == expected, f"Failed for pattern: {pattern!r}"

    def test_numbered_barline_preserved(self):
        """Should preserve barline numbers when normalizing."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        input_text = "*v\t*v\t*\n=5\t=5\n*\t*^"
        expected = "=5\t=5\t=5"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_merge_then_terminate_no_barline(self):
        """Should handle merge followed directly by terminate (no barline)."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        input_text = "*\t*v\t*v\n*-\t*-"
        expected = "*-\t*-\t*-"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_merge_then_terminate_no_barline_in_context(self):
        """Should handle merge→terminate in context of a full piece."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern\t**kern
*clefF4\t*clefG2\t*clefG2
4c\t4e\t4g
*\t*v\t*v
*-\t*-"""

        expected = """**kern\t**kern\t**kern
*clefF4\t*clefG2\t*clefG2
4c\t4e\t4g
*-\t*-\t*-"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_all_merge_terminate_no_barline_variations(self):
        """Should normalize all 3-spine merge→terminate (no barline) variations."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        patterns = [
            ("*v\t*v\t*\n*-\t*-", "*-\t*-\t*-"),
            ("*v\t*\t*v\n*-\t*-", "*-\t*-\t*-"),
            ("*\t*v\t*v\n*-\t*-", "*-\t*-\t*-"),
        ]

        for pattern, expected in patterns:
            pass_obj.prepare(pattern, ctx)
            result = pass_obj.transform(pattern, ctx)
            pass_obj.validate(result, ctx)
            assert result == expected, f"Failed for pattern: {pattern!r}"

    def test_final_barline_preserved(self):
        """Should preserve final barline markers when normalizing."""
        pass_obj = MergeSplitNormalizer()
        ctx = NormalizationContext()

        input_text = "*v\t*v\t*\n==\t==\n*-\t*-"
        expected = "==\t==\t==\n*-\t*-\t*-"

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected
