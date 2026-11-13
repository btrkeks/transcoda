"""Shared VeroSim metric option normalization."""

from __future__ import annotations

DEFAULT_VEROSIM_MODE = "active"
VEROSIM_MODES = frozenset({"active", "experimental"})
LEGACY_VEROSIM_DETAIL_TO_MODE = {
    "tierA": "active",
    "tierAB": "active",
    "tierAB_dir": "experimental",
}


def normalize_verosim_mode(
    *,
    mode: str | None = None,
    detail: str | None = None,
) -> str:
    """Return a supported VeroSim mode, accepting legacy detail names."""
    if detail is not None and detail not in LEGACY_VEROSIM_DETAIL_TO_MODE:
        raise ValueError("legacy VeroSim detail must be one of tierA, tierAB, tierAB_dir")

    resolved_mode = mode
    if resolved_mode is None:
        resolved_mode = (
            LEGACY_VEROSIM_DETAIL_TO_MODE[detail]
            if detail is not None
            else DEFAULT_VEROSIM_MODE
        )

    if resolved_mode not in VEROSIM_MODES:
        raise ValueError("VeroSim mode must be one of active, experimental")
    return resolved_mode
