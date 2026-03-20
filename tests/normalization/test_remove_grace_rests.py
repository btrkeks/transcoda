"""Tests for RemoveGraceRests pass."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.dataset_generation.normalization import NormalizationContext
from scripts.dataset_generation.normalization.passes import RemoveGraceRests


class TestRemoveGraceRests:
    """Tests for RemoveGraceRests pass."""

    def test_pass_exists(self):
        """Pass should be instantiable with correct name."""
        p = RemoveGraceRests()
        assert p.name == "remove_grace_rests"

    def test_bare_grace_rest(self):
        """Should replace bare qr with null token."""
        p = RemoveGraceRests()
        ctx = NormalizationContext()
        result = p.transform("2e\t.\tqr\t2a", ctx)
        assert result == "2e\t.\t.\t2a"

    def test_grace_rest_with_trailing_at(self):
        """Should replace qr@ (after CleanupKern strips yy) with null token."""
        p = RemoveGraceRests()
        ctx = NormalizationContext()
        result = p.transform("2e\t.\tqr@\t2a", ctx)
        assert result == "2e\t.\t.\t2a"

    def test_grace_rest_with_editorial(self):
        """Should replace qryy@ with null token."""
        p = RemoveGraceRests()
        ctx = NormalizationContext()
        result = p.transform("2e\t.\tqryy@\t2a", ctx)
        assert result == "2e\t.\t.\t2a"

    def test_grace_rest_with_duration(self):
        """Should replace grace rest with duration prefix."""
        p = RemoveGraceRests()
        ctx = NormalizationContext()
        result = p.transform("4qr\t8eL", ctx)
        assert result == ".\t8eL"

    def test_preserves_normal_rests(self):
        """Should not modify normal rests with duration."""
        p = RemoveGraceRests()
        ctx = NormalizationContext()
        input_text = "4r\t8r\t2r"
        result = p.transform(input_text, ctx)
        assert result == input_text

    def test_preserves_grace_notes(self):
        """Should not modify grace notes (q after pitch)."""
        p = RemoveGraceRests()
        ctx = NormalizationContext()
        input_text = "8eq\t4cq\teq"
        result = p.transform(input_text, ctx)
        assert result == input_text

    def test_preserves_null_tokens(self):
        """Should not modify null tokens."""
        p = RemoveGraceRests()
        ctx = NormalizationContext()
        input_text = ".\t.\t."
        result = p.transform(input_text, ctx)
        assert result == input_text

    def test_preserves_interpretations(self):
        """Should not modify interpretation lines."""
        p = RemoveGraceRests()
        ctx = NormalizationContext()
        input_text = "*clefG2\t*clefF4"
        result = p.transform(input_text, ctx)
        assert result == input_text

    def test_preserves_barlines(self):
        """Should not modify barlines."""
        p = RemoveGraceRests()
        ctx = NormalizationContext()
        input_text = "=\t="
        result = p.transform(input_text, ctx)
        assert result == input_text

    def test_empty_input(self):
        """Should handle empty string."""
        p = RemoveGraceRests()
        ctx = NormalizationContext()
        result = p.transform("", ctx)
        assert result == ""

    def test_multiline(self):
        """Should handle multiple lines with grace rests."""
        p = RemoveGraceRests()
        ctx = NormalizationContext()
        input_text = "4e\tqr\t4g\n8fL\t8r\t8aL"
        expected = "4e\t.\t4g\n8fL\t8r\t8aL"
        result = p.transform(input_text, ctx)
        assert result == expected

    def test_real_world_context(self):
        """Should handle the actual failing line from the dataset."""
        p = RemoveGraceRests()
        ctx = NormalizationContext()
        # After CleanupKern: qryy@ becomes qr@ (y stripped, @ remains)
        input_text = "2e\t.\t2e;\t.\tqr@\t2a\t2cc#;"
        expected = "2e\t.\t2e;\t.\t.\t2a\t2cc#;"
        result = p.transform(input_text, ctx)
        assert result == expected
