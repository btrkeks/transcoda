"""Tests for RemoveNonKernSpines pass."""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports when running as a test
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.dataset_generation.normalization import NormalizationContext
from scripts.dataset_generation.normalization.passes import (
    RemoveNonKernSpines,
    UnsafeSpineStructureError,
)


class TestRemoveNonKernSpines:
    """Behavioral tests for non-kern spine removal with spine manipulators."""

    def test_pass_exists(self):
        pass_obj = RemoveNonKernSpines()
        assert pass_obj.name == "remove_non_kern_spines"

    def test_removes_non_kern_spines_with_split_merge(self):
        pass_obj = RemoveNonKernSpines()
        ctx = NormalizationContext()

        input_text = """**kern\t**text\t**kern
*clefG2\t*\t*clefF4
*\t*^\t*
4c\tlyrA\tlyrB\t4E
*\t*v\t*v\t*
4d\tlyr\t4F"""

        expected = """**kern\t**kern
*clefG2\t*clefF4
*\t*
4c\t4E
*\t*
4d\t4F"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == expected

    def test_no_non_kern_spines_is_passthrough(self):
        pass_obj = RemoveNonKernSpines()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern
*clefF4\t*clefG2
4C\t4e
*-\t*-"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert result == input_text

    def test_unsupported_spine_manipulators_raise(self):
        pass_obj = RemoveNonKernSpines()
        ctx = NormalizationContext()

        input_text = """**kern\t**text\t**kern
*\t*x\t*
4c\tlyr\t4e"""

        pass_obj.prepare(input_text, ctx)
        with pytest.raises(UnsafeSpineStructureError, match=r"unsupported spine manipulators"):
            pass_obj.transform(input_text, ctx)

