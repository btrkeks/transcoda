"""Tests for NormalizeRScale pass."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.dataset_generation.normalization import NormalizationContext
from scripts.dataset_generation.normalization.passes import NormalizeRScale


class TestNormalizeRScale:
    """Tests for active *rscale rewriting."""

    def test_rewrites_single_spine_rscale_two_and_drops_star_only_lines(self):
        p = NormalizeRScale()
        ctx = NormalizationContext()

        input_text = "*clefG2\n*rscale:2\n4c\n*rscale:1\n2d"
        expected = "*clefG2\n2c\n2d"

        p.prepare(input_text, ctx)
        result = p.transform(input_text, ctx)

        assert result == expected

    def test_rewrites_single_spine_rscale_half(self):
        p = NormalizeRScale()
        ctx = NormalizationContext()

        input_text = "*clefG2\n*rscale:1/2\n4c\n*rscale:1\n2d"
        expected = "*clefG2\n8c\n2d"

        p.prepare(input_text, ctx)
        result = p.transform(input_text, ctx)

        assert result == expected

    def test_rewrites_only_active_spine(self):
        p = NormalizeRScale()
        ctx = NormalizationContext()

        input_text = "*clefF4\t*clefG2\n*\t*rscale:2\n4C\t4c\n*\t*rscale:1\n2D\t2d"
        expected = "*clefF4\t*clefG2\n4C\t2c\n2D\t2d"

        p.prepare(input_text, ctx)
        result = p.transform(input_text, ctx)

        assert result == expected

    def test_rewrites_chords_and_rests(self):
        p = NormalizeRScale()
        ctx = NormalizationContext()

        input_text = "*rscale:1/2\n4c 4e 4g\n2r\n*rscale:1\n4a"
        expected = "8c 8e 8g\n4r\n4a"

        p.prepare(input_text, ctx)
        result = p.transform(input_text, ctx)

        assert result == expected

    def test_preserves_null_tokens_and_grace_notes(self):
        p = NormalizeRScale()
        ctx = NormalizationContext()

        input_text = "*rscale:2\n.\n8cq\n16qqcL\n*rscale:1\n4d"
        expected = ".\n8cq\n16qqcL\n4d"

        p.prepare(input_text, ctx)
        result = p.transform(input_text, ctx)

        assert result == expected

    def test_propagates_scale_across_split_and_merge(self):
        p = NormalizeRScale()
        ctx = NormalizationContext()

        input_text = "*clefG2\n*rscale:2\n*^\n4c\t4e\n*v\t*v\n2g\n*rscale:1\n4a"
        expected = "*clefG2\n*^\n2c\t2e\n*v\t*v\n1g\n4a"

        p.prepare(input_text, ctx)
        result = p.transform(input_text, ctx)

        assert result == expected

    def test_emits_newly_supported_reciprocal_duration(self):
        p = NormalizeRScale()
        ctx = NormalizationContext()

        input_text = "*rscale:2\n76aa(L\n*rscale:1\n4b"
        expected = "38aa(L\n4b"

        p.prepare(input_text, ctx)
        result = p.transform(input_text, ctx)

        assert result == expected

    def test_rejects_unsupported_future_ratio(self):
        p = NormalizeRScale()
        ctx = NormalizationContext()

        input_text = "*rscale:3/2\n4c"

        p.prepare(input_text, ctx)
        with pytest.raises(ValueError, match=r"unsupported \*rscale value"):
            p.transform(input_text, ctx)

    def test_rejects_merge_of_different_active_rscale_values(self):
        p = NormalizeRScale()
        ctx = NormalizationContext()

        input_text = "*\t*rscale:2\n4c\t4e\n*v\t*v\n2g"

        p.prepare(input_text, ctx)
        with pytest.raises(ValueError, match=r"different active \*rscale values"):
            p.transform(input_text, ctx)
