"""Truncation policy and prefix candidate generation."""

from __future__ import annotations

from scripts.dataset_generation.dataset_generation.truncation import (
    PrefixTruncationCandidate,
    build_prefix_truncation_candidates as _build_prefix_truncation_candidates,
)
from scripts.dataset_generation.dataset_generation_new.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation_new.types import SvgLayoutDiagnostics, TruncationMode


def classify_truncation_mode(
    diagnostics: SvgLayoutDiagnostics,
    recipe: ProductionRecipe,
) -> TruncationMode:
    if diagnostics.page_count > 1:
        return "required"
    if diagnostics.system_count > recipe.truncation.required_over_systems:
        return "required"
    if (
        recipe.truncation.preferred_min_systems
        <= diagnostics.system_count
        <= recipe.truncation.preferred_max_systems
    ):
        return "preferred"
    return "forbidden"


def build_prefix_candidates(
    label_transcription: str,
    recipe: ProductionRecipe,
) -> list[PrefixTruncationCandidate]:
    return _build_prefix_truncation_candidates(
        label_transcription,
        max_trials=recipe.truncation.max_candidate_trials,
    )
