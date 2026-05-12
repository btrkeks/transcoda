# src/synth/exceptions.py
from __future__ import annotations


class GeneratorError(Exception):
    """Base class for synthetic score generator failures."""


class RenderingError(GeneratorError):
    """Low-level rendering or rasterization failure (SVG/PNG decode, etc.)."""


class LayoutOverflowError(RenderingError):
    """The rendered layout exceeded requested limits (pages or systems)."""

    def __init__(
        self,
        message: str,
        *,
        page_count: int | None = None,
        allowed_pages: int | None = None,
        system_count: int | None = None,
        max_systems: int | None = None,
    ) -> None:
        details: list[str] = [message]
        if page_count is not None and allowed_pages is not None:
            details.append(f"pages={page_count} allowed={allowed_pages}")
        if system_count is not None and max_systems is not None:
            details.append(f"systems={system_count} max={max_systems}")
        super().__init__(" | ".join(details))
