"""
Kern notation normalization system.

This package provides a composable pipeline architecture for normalizing
humdrum kern transcriptions. Normalization passes can be composed, reordered,
and configured to transform kern strings in a predictable, maintainable way.

Quick Start:
    from normalization import Pipeline
    from normalization.passes import IdentityPass, OrderNotes

    # Create a pipeline with ordered passes
    pipeline = Pipeline([OrderNotes()])

    # Normalize a kern string
    result = pipeline("4g 4e 4c")

Configuration-based usage:
    from normalization import Pipeline

    # Load pipeline from JSON config
    pipeline = Pipeline.from_config("config/normalization/default.json")
    result = pipeline(kern_string)

Implementing a new pass:
    from normalization.base import NormalizationContext

    class MyPass:
        name = "my_pass"

        def prepare(self, text: str, ctx: NormalizationContext) -> None:
            # Parse/analyze the input
            pass

        def transform(self, text: str, ctx: NormalizationContext) -> str:
            # Transform the kern string
            return text

        def validate(self, text: str, ctx: NormalizationContext) -> None:
            # Validate the output
            pass
"""

from .base import NormalizationContext, NormalizationPass
from .pipeline import Pipeline
from .presets import normalize_kern_transcription

__all__ = [
    "NormalizationContext",
    "NormalizationPass",
    "Pipeline",
    "normalize_kern_transcription",
]

__version__ = "0.1.0"
