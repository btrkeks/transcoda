"""Collection of file filters for kern notation."""

from .header_clef import HeaderClefFilter
from .rational_duration import RationalDurationFilter
from .rhythm import RhythmFilter
from .staves import StavesFilter

# Registry of all available filters
FILTER_REGISTRY: dict[str, type] = {
    "header_clef": HeaderClefFilter,
    "staves": StavesFilter,
    "rational_duration": RationalDurationFilter,
    "rhythm": RhythmFilter,
}

__all__ = [
    "HeaderClefFilter",
    "StavesFilter",
    "RationalDurationFilter",
    "RhythmFilter",
    "FILTER_REGISTRY",
]
