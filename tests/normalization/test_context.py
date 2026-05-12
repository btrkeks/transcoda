"""Tests for NormalizationContext."""

import sys
from pathlib import Path

# Add parent directory to path for imports when running as a test
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.dataset_generation.normalization import NormalizationContext


class TestNormalizationContext:
    """Tests for NormalizationContext."""

    def test_context_is_dict(self):
        """Context should behave like a dictionary."""
        ctx = NormalizationContext()
        ctx["key"] = "value"
        assert ctx["key"] == "value"
        assert "key" in ctx

    def test_context_stores_pass_data(self):
        """Context should allow passes to store and retrieve data."""
        ctx = NormalizationContext()
        ctx["pass1"] = {"data": [1, 2, 3]}
        ctx["pass2"] = {"count": 5}

        assert ctx["pass1"]["data"] == [1, 2, 3]
        assert ctx["pass2"]["count"] == 5
