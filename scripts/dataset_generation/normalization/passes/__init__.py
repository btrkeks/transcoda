"""Collection of normalization passes for kern notation."""

from .canonicalize_barlines import CanonicalizeBarlines
from .canonicalize_header_order import CanonicalizeHeaderOrder
from .canonicalize_note_order import CanonicalizeNoteOrder
from .cap_slurs import CapSlurs
from .cleanup_kern import CleanupKern
from .fix_note_beams import FixNoteBeams
from .fix_slurs import FixSlurs
from .fix_ties import FixTies
from .merge_header_clef_lines import MergeHeaderClefLines
from .merge_split_normalizer import MergeSplitNormalizer
from .normalize_final_barline import NormalizeFinalBarline
from .normalize_null_keysig import NormalizeNullKeysig
from .normalize_rscale import NormalizeRScale
from .order_notes import OrderNotes
from .remove_conflicting_bowings import RemoveConflictingBowings
from .remove_grace_rests import RemoveGraceRests
from .remove_contradictory_accidentals import RemoveContradictoryAccidentals
from .remove_leading_barlines import RemoveLeadingBarlines
from .remove_non_kern_spines import RemoveNonKernSpines, UnsafeSpineStructureError
from .remove_null_lines import RemoveNullLines
from .remove_null_ties import RemoveNullTies
from .remove_redundant_stria import RemoveRedundantStria
from .remove_redundant_time_signatures import RemoveRedundantTimeSignatures
from .repair_interpretation_spacing import RepairInterpretationSpacing
from .strip_terminal_terminator import StripTerminalTerminator
from .symbols_before_split import SymbolsBeforeSplit
from .validate_spine_operations import ValidateSpineOperations

# Registry of all available passes
# Add new passes here as you implement them
PASS_REGISTRY: dict[str, type] = {
    "canonicalize_barlines": CanonicalizeBarlines,
    "canonicalize_header_order": CanonicalizeHeaderOrder,
    "canonicalize_note_order": CanonicalizeNoteOrder,
    "cap_slurs": CapSlurs,
    "cleanup_kern": CleanupKern,
    "fix_note_beams": FixNoteBeams,
    "fix_slurs": FixSlurs,
    "fix_ties": FixTies,
    "merge_header_clef_lines": MergeHeaderClefLines,
    "merge_split_normalizer": MergeSplitNormalizer,
    "normalize_final_barline": NormalizeFinalBarline,
    "normalize_null_keysig": NormalizeNullKeysig,
    "normalize_rscale": NormalizeRScale,
    "order_notes": OrderNotes,
    "remove_conflicting_bowings": RemoveConflictingBowings,
    "remove_grace_rests": RemoveGraceRests,
    "remove_contradictory_accidentals": RemoveContradictoryAccidentals,
    "remove_leading_barlines": RemoveLeadingBarlines,
    "remove_non_kern_spines": RemoveNonKernSpines,
    "remove_null_lines": RemoveNullLines,
    "remove_null_ties": RemoveNullTies,
    "remove_redundant_stria": RemoveRedundantStria,
    "remove_redundant_time_signatures": RemoveRedundantTimeSignatures,
    "repair_interpretation_spacing": RepairInterpretationSpacing,
    "strip_terminal_terminator": StripTerminalTerminator,
    "symbols_before_split": SymbolsBeforeSplit,
    "validate_spine_operations": ValidateSpineOperations,
}

__all__ = [
    "CanonicalizeBarlines",
    "CanonicalizeHeaderOrder",
    "CanonicalizeNoteOrder",
    "CapSlurs",
    "CleanupKern",
    "FixNoteBeams",
    "FixSlurs",
    "FixTies",
    "MergeHeaderClefLines",
    "MergeSplitNormalizer",
    "NormalizeFinalBarline",
    "NormalizeNullKeysig",
    "NormalizeRScale",
    "OrderNotes",
    "RemoveConflictingBowings",
    "RemoveGraceRests",
    "RemoveContradictoryAccidentals",
    "RemoveLeadingBarlines",
    "RemoveNonKernSpines",
    "RemoveNullLines",
    "RemoveNullTies",
    "RemoveRedundantStria",
    "RemoveRedundantTimeSignatures",
    "RepairInterpretationSpacing",
    "StripTerminalTerminator",
    "SymbolsBeforeSplit",
    "ValidateSpineOperations",
    "UnsafeSpineStructureError",
    "PASS_REGISTRY",
]
