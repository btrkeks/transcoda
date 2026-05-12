"""Tests for RepairInterpretationSpacing pass."""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports when running as a test
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.dataset_generation.normalization import NormalizationContext
from scripts.dataset_generation.normalization.passes import RepairInterpretationSpacing


class TestRepairInterpretationSpacing:
    """Coverage for shared spine-state integration."""

    def test_repairs_interpretation_tabs_after_split(self):
        pass_obj = RepairInterpretationSpacing()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern
*^\t*
*\t*Xtuplet    *
4c\t4e\t4g"""

        pass_obj.prepare(input_text, ctx)
        result = pass_obj.transform(input_text, ctx)
        pass_obj.validate(result, ctx)

        assert "*\t*Xtuplet\t*" in result

    def test_unsupported_spine_manipulator_raises(self):
        pass_obj = RepairInterpretationSpacing()
        ctx = NormalizationContext()

        input_text = """**kern\t**kern
*x\t*x"""

        pass_obj.prepare(input_text, ctx)
        with pytest.raises(ValueError, match=r"Unsupported spine manipulator"):
            pass_obj.transform(input_text, ctx)

