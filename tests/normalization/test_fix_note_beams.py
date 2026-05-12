"""Tests for FixNoteBeams normalization pass."""

from pathlib import Path

import pytest

from scripts.dataset_generation.normalization import NormalizationContext
from scripts.dataset_generation.normalization.passes import FixNoteBeams


class TestFixNoteBeams:
    """Tests for the FixNoteBeams pass."""

    def setup_method(self):
        """Set up test fixtures."""
        self.pass_obj = FixNoteBeams()
        self.ctx = NormalizationContext()

    def _apply_pass(self, text: str) -> str:
        """Helper to apply the pass with context."""
        self.pass_obj.prepare(text, self.ctx)
        result = self.pass_obj.transform(text, self.ctx)
        self.pass_obj.validate(result, self.ctx)
        return result

    def test_beam_fixing_with_spine_splits(self):
        """Test that beam fixing handles spine splits (*^) correctly.

        This test reproduces a bug where beam fixing fails on real-world
        **kern files that use spine manipulations (splits/merges) within measures.

        Example from Ballade_No1_p02.krn shows spine splitting mid-measure.
        """
        # Minimal reproduction of the bug - a measure with spine split
        krn_with_spine_split = """**kern	**kern	**dynam
*clefF4	*clefG2	*
*k[b-e-]	*k[b-e-]	*
*M6/4	*M6/4	*
*	*^	*
2.CC 2.C	2.a	4e- 4g	.
.	.	4e- 4g	.
.	.	4e- 4g	.
=	=	=	=
4CC#:) 4C#:	[2.a	4A: 4en:	[ pp
*	*	*^	*
.	.	.	1a	.
(4En' 4A 4en	.	4ryy@	.	.
4E' 4A 4e	.	4ryy@	.	.
4E' 4A 4e	2a]	4ryy@	.	.
4E' 4A 4e	.	4ryy@	.	.
*	*v	*v	*v	*
4E') 4A 4e	[4a^	.
=	=	=
*-	*-	*-
"""

        # This should not raise an AssertionError about inconsistent spines
        result = self._apply_pass(krn_with_spine_split)

        # The result should still be valid **kern with the same structure
        assert "**kern" in result
        assert "*^" in result  # Spine split should be preserved
        assert "*v" in result  # Spine merge should be preserved

    def test_beam_fixing_with_multiple_spine_manipulations(self):
        """Test beam fixing with complex spine manipulations from real score.

        This is a more comprehensive test using a larger excerpt that includes
        multiple spine splits and merges across several measures.
        """
        # Excerpt from Ballade_No1_p02.krn with multiple spine manipulations
        krn_complex = """**kern	**kern	**dynam
*clefF4	*clefG2	*
*k[b-e-]	*k[b-e-]	*
*M6/4	*M6/4	*
*	*^	*
4DD'	2.b-	.	.
*^	*	*	*
2G	(4B- 4e-	.	.	.
.	2B- 2d	.	.	.
(2.D	.	2.f#	.	.
.	4B- 4c#	.	.	.
.	4A) 4cn	.	.	.
*v	*v	*	*	*
=	=	=	=
4GG')	4g)	]	]
*^	*^	*	*
2G 2B-	(4f	4bb-	(8b-L	.
.	.	.	8dd	.
.	4e-)	8ryy	8ff	[
.	.	8ryy	[8ee-J	.
*v	*v	*	*	*
4FF#'	4ryy	4ee-])	]
*-	*-	*-	*-
"""

        result = self._apply_pass(krn_complex)

        # Verify the result maintains the structure
        assert "**kern" in result
        assert result.count("*^") >= 2  # At least 2 spine splits
        assert result.count("*v") >= 2  # At least 2 spine merges

    def test_beam_fixing_simple_case(self):
        """Test that beam fixing still works on simple cases without spine manipulations."""
        simple_krn = """**kern	**kern
*clefF4	*clefG2
*k[]	*k[]
*M4/4	*M4/4
4c	4cc
4d	4dd
4e	4ee
4f	4ff
=	=
*-	*-
"""

        result = self._apply_pass(simple_krn)

        # Should work without errors
        assert "**kern" in result
        assert "4c" in result

    def test_beam_fixing_with_tie_markers(self):
        """Test that beam fixing handles tie markers correctly.

        Bug #1: Tokens like [8ee-J (tie start + duration + beam end) should parse correctly.
        """
        krn_with_ties = """**kern	**kern
*clefF4	*clefG2
*k[]	*k[]
*M4/4	*M4/4
4c	[8ccL
4d	8dd]
4e	8ee]J
4f	[8ffL
=	=
*-	*-
"""

        # This should not raise AssertionError about parsing duration
        result = self._apply_pass(krn_with_ties)

        # Verify result is valid
        assert "**kern" in result
        assert "8cc" in result or "[8cc" in result  # Duration should be preserved

    def test_beam_fixing_with_real_ballade_excerpt(self):
        """Test with actual excerpt from Ballade_No1_p02.krn that triggered the bug.

        This reproduces the AssertionError: Inconsistent number of spines in measure part
        """
        # Load the actual file that causes the error
        ballade_path = Path("data/raw/manual/Ballade_No1_p02.krn")

        if ballade_path.exists():
            krn_content = ballade_path.read_text()

            # This should not raise AssertionError about inconsistent spines
            result = self._apply_pass(krn_content)

            assert "**kern" in result
        else:
            pytest.skip("Ballade_No1_p02.krn not found in data/raw/manual/")

    def test_beam_fixing_with_trailing_q(self):
        krn_with_trailing_q = "f##Lq"
        result = self._apply_pass(krn_with_trailing_q)
        assert result == "f##q"
