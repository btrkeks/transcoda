"""Collection of file filters for kern notation."""

from .accidentals import AccidentalsFilter
from .excessive_octave import ExcessiveOctaveFilter
from .header_clef import HeaderClefFilter
from .rational_duration import RationalDurationFilter
from .rhythm import RhythmFilter
from .termination import TerminationFilter
from .utf8 import UTF8Filter

# Registry of all available filters
FILTER_REGISTRY: dict[str, type] = {
    "utf8": UTF8Filter,
    "termination": TerminationFilter,
    "header_clef": HeaderClefFilter,
    "rational_duration": RationalDurationFilter,
    "rhythm": RhythmFilter,
    "accidentals": AccidentalsFilter,
    "excessive_octave": ExcessiveOctaveFilter,
}

__all__ = [
    "AccidentalsFilter",
    "ExcessiveOctaveFilter",
    "HeaderClefFilter",
    "UTF8Filter",
    "TerminationFilter",
    "RationalDurationFilter",
    "RhythmFilter",
    "FILTER_REGISTRY",
]
