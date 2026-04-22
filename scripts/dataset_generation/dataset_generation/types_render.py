"""Render-stage types: SVG diagnostics, render results, acceptance, samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.dataset_generation.dataset_generation.image_generation.types import RenderedPage
from scripts.dataset_generation.dataset_generation.types_domain import AcceptanceAction
from scripts.dataset_generation.dataset_generation.types_events import (
    AugmentationTraceEvent,
    VerovioDiagnostic,
)


@dataclass(frozen=True)
class SvgLayoutDiagnostics:
    system_count: int
    page_count: int
    system_bbox_stats: dict[str, float] | None = None


@dataclass(frozen=True)
class RenderResult:
    image: np.ndarray | None
    render_layers: RenderedPage | None
    svg_diagnostics: SvgLayoutDiagnostics
    bottom_whitespace_ratio: float | None
    vertical_fill_ratio: float | None
    render_height_px: int | None = None
    bottom_padding_px: int | None = None
    bottom_whitespace_px: int | None = None
    top_whitespace_px: int | None = None
    content_height_px: int | None = None
    rejection_reason: str | None = None
    metadata_prefix: str = ""
    verovio_diagnostics: tuple[VerovioDiagnostic, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.image is not None and self.rejection_reason is None


@dataclass(frozen=True)
class AcceptanceDecision:
    action: AcceptanceAction
    reason: str | None = None


@dataclass(frozen=True)
class AcceptedSample:
    sample_id: str
    label_transcription: str
    image_bytes: bytes
    initial_kern_spine_count: int
    segment_count: int
    source_ids: tuple[str, ...]
    source_measure_count: int
    source_non_empty_line_count: int
    system_count: int
    truncation_applied: bool
    truncation_reason: str | None
    truncation_ratio: float | None
    bottom_whitespace_ratio: float | None
    vertical_fill_ratio: float | None
    bottom_whitespace_px: int | None = None
    top_whitespace_px: int | None = None
    content_height_px: int | None = None


@dataclass(frozen=True)
class AugmentedRenderResult:
    final_image: np.ndarray | bytes
    trace: AugmentationTraceEvent | None
    base_image: np.ndarray | None = None
    pre_augraphy_image: np.ndarray | None = None
