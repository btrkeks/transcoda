"""Cleanup passes for pre-parser kern repair."""

from .musescore_palette_template import RemoveMuseScorePaletteTemplate
from .remove_exact_interpretations import RemoveExactInterpretations
from .remove_grandstaff_single_y_from_notes import RemoveGrandstaffSingleYFromNotes
from .rest_decorations import RemoveRestDecorations

PASS_REGISTRY: dict[str, type] = {
    "remove_musescore_palette_template": RemoveMuseScorePaletteTemplate,
    "remove_grandstaff_single_y_from_notes": RemoveGrandstaffSingleYFromNotes,
    "remove_exact_interpretations": RemoveExactInterpretations,
    "remove_rest_decorations": RemoveRestDecorations,
}

__all__ = [
    "PASS_REGISTRY",
    "RemoveGrandstaffSingleYFromNotes",
    "RemoveExactInterpretations",
    "RemoveMuseScorePaletteTemplate",
    "RemoveRestDecorations",
]
