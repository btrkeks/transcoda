"""Preset cleanup pipelines for raw converted kern data."""

from __future__ import annotations

from .base import CleanupContext
from .passes import (
    RemoveExactInterpretations,
    RemoveGrandstaffSingleYFromNotes,
    RemoveMuseScorePaletteTemplate,
    RemoveRestDecorations,
)
from .pipeline import CleanupPipeline

_STANDARD_PRE_PARSER_CLEANUP_PASSES = [
    RemoveMuseScorePaletteTemplate(),
    RemoveRestDecorations(),
    RemoveExactInterpretations(),
]

_PRE_PARSER_CLEANUP_PIPELINE = CleanupPipeline(
    _STANDARD_PRE_PARSER_CLEANUP_PASSES
)

_GRANDSTAFF_PRE_PARSER_CLEANUP_PIPELINE = CleanupPipeline(
    [
        *_STANDARD_PRE_PARSER_CLEANUP_PASSES,
        RemoveGrandstaffSingleYFromNotes(),
    ]
)


def clean_kern_for_parser(
    transcription: str,
    source_dataset: str | None = None,
) -> str:
    cleaned, _ = clean_kern_for_parser_with_context(
        transcription,
        source_dataset=source_dataset,
    )
    return cleaned


def clean_kern_for_parser_with_context(
    transcription: str,
    source_dataset: str | None = None,
) -> tuple[str, CleanupContext]:
    """Clean transcription and return both output and pass context."""
    ctx = CleanupContext()
    pipeline = (
        _GRANDSTAFF_PRE_PARSER_CLEANUP_PIPELINE
        if source_dataset == "grandstaff"
        else _PRE_PARSER_CLEANUP_PIPELINE
    )
    cleaned = pipeline(transcription, ctx=ctx)
    return cleaned, ctx
