"""Preset normalization pipelines for **kern transcriptions.

This module provides reusable normalization functions that can be applied
consistently across different parts of the codebase (e.g., synthetic generation,
validation dataset preprocessing, inference).
"""

from __future__ import annotations

from . import Pipeline
from .base import NormalizationContext
from .passes import (
    CanonicalizeBarlines,
    CanonicalizeHeaderOrder,
    CanonicalizeNoteOrder,
    CapSlurs,
    CleanupKern,
    FixNoteBeams,
    FixSlurs,
    FixTies,
    MergeHeaderClefLines,
    MergeSplitNormalizer,
    NormalizeNullKeysig,
    OrderNotes,
    RemoveConflictingBowings,
    RemoveContradictoryAccidentals,
    RemoveGraceRests,
    RemoveLeadingBarlines,
    RemoveNonKernSpines,
    RemoveNullLines,
    RemoveNullTies,
    RemoveRedundantStria,
    RemoveRedundantTimeSignatures,
    RepairInterpretationSpacing,
    StripTerminalTerminator,
    SymbolsBeforeSplit,
    ValidateSpineOperations,
)

# Full preprocessing pipeline for raw external data
# Includes cleanup preprocessing followed by standard normalization
_FULL_PIPELINE = Pipeline(
    [
        # == Removal ==
        # IMPORTANT: RemoveNonKernSpines must run BEFORE CleanupKern because it needs
        # the **kern header line to identify which columns to keep
        RemoveNonKernSpines(),  # Remove **mxhm, **dynam, **text and other non-kern spines
        CleanupKern(),  # Remove comments, **kern headers, unwanted tokens, etc.
        RemoveGraceRests(),  # Replace grace rests (qr) with null tokens - must be before CanonicalizeNoteOrder
        RemoveRedundantStria(),  # Remove redundant mid-piece stria lines
        RemoveLeadingBarlines(),
        RemoveNullLines(),
        CanonicalizeBarlines(),  # Normalize non-standard barline tokens (=|! → =, ==:|! → =:|!)
        MergeSplitNormalizer(),
        StripTerminalTerminator(),  # Remove terminal *- for page-faithful targets
        # == Repairing ==
        CapSlurs(),  # Cap slurs at 2 per note before other fixes
        RemoveRedundantTimeSignatures(),  # Remove meter when equivalent mensuration exists
        # Exclude reparation for new dataset
        # RepairInterpretationSpacing(),  # Fix malformed tabs in external data
        # FixNoteBeams(),
        # FixTies(),
        # FixSlurs(),
        # == Reordering ==
        CanonicalizeHeaderOrder(),  # Reorder header lines to canonical order
        MergeHeaderClefLines(),  # Merge consecutive clef lines (after reordering brings them together)
        NormalizeNullKeysig(),  # Replace * with *k[] in key signature lines
        CanonicalizeNoteOrder(),  # Reorder note components to canonical order
        OrderNotes(
            ascending=True
        ),
        SymbolsBeforeSplit(),
        # == Removal ==
        RemoveContradictoryAccidentals(),  # Resolve #n, -n, etc. - must be after CanonicalizeNoteOrder
        RemoveConflictingBowings(),  # Remove down-bow when both up-bow and down-bow present
        RemoveNullTies(),  # Remove null ties (4G[]) - must be after CanonicalizeNoteOrder
        ValidateSpineOperations(),  # Enforce semantic spine-op validity in final output
    ]
)


def normalize_kern_transcription(transcription: str) -> str:
    normalized, _ = normalize_kern_transcription_with_context(transcription)
    return normalized


def normalize_kern_transcription_with_context(
    transcription: str,
) -> tuple[str, NormalizationContext]:
    """Normalize transcription and return both output and pass context."""
    ctx = NormalizationContext()
    normalized = _FULL_PIPELINE(transcription, ctx=ctx)
    return normalized, ctx


def preprocess_and_normalize_kern(transcription: str) -> str:
    return _FULL_PIPELINE(transcription)
