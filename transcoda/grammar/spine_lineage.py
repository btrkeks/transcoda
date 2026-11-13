"""Original-spine lineage and track-scoped tandem state for **kern decoding.

A finished transcoda-v2 prediction must still resolve once kern_parser lowers
it to Humdrum **kern. These rules mirror kern_parser rev 05663e3:

- `score/topology.rs` `Builder::manipulate`: every `*v` of one merge run must
  descend from the same original spine (track).
- `score/mod.rs` `Resolver::tandems`: track-scoped tandems are keyed by track,
  so sibling subspines on one line must agree.
- `score/state.rs` `apply_span`: an ottava closes only the kind that is open
  and opens only when none, or the same kind, is open. Ottava state is
  track-scoped, so splits and merges carry it unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from transcoda.core.spine_state import advance_tracks

MERGE_FIELD = "*v"
OTTAVA_KINDS = ("8va", "8ba", "15ma", "15ba")
OTTAVA_FIELDS = frozenset(
    [f"*{kind}" for kind in OTTAVA_KINDS] + [f"*X{kind}" for kind in OTTAVA_KINDS]
)
LINEAGE_FIELDS = OTTAVA_FIELDS | {MERGE_FIELD}
# Proper prefixes that ottava and merge fields share with other legal fields
# (`*`, `*^`, `*Xped`, `*Xcue`, ...). Masking them would block those fields.
SHARED_FIELD_PREFIXES = frozenset({"*", "*X"})
_SPINE_OP_FIELDS = frozenset({"*", "*^", MERGE_FIELD})
# `*cue` is layer-scoped in kern_parser, so siblings may disagree on it.
_NON_TRACK_FIELDS = _SPINE_OP_FIELDS | {"*cue", "*Xcue"}
_INVISIBLE_PREFIXES = ("*clef", "*M")
_INVISIBLE_SUFFIX = "yy"
_FORBIDDEN_OTTAVAS: dict[str | None, frozenset[str]] = {
    None: frozenset(f"*X{kind}" for kind in OTTAVA_KINDS),
    **{kind: OTTAVA_FIELDS - {f"*{kind}", f"*X{kind}"} for kind in OTTAVA_KINDS},
}


def _is_track_tandem(field: str) -> bool:
    return field.startswith("*") and field not in _NON_TRACK_FIELDS


def _visible(field: str) -> str:
    """Drop the `yy` marker kern_parser ignores when comparing siblings (`state::visible`)."""
    if field.startswith(_INVISIBLE_PREFIXES):
        return field.removesuffix(_INVISIBLE_SUFFIX)
    return field


@dataclass(frozen=True)
class SpineLineage:
    """Original spine of each active column and the ottava open on each spine."""

    tracks: tuple[int, ...]
    open_ottavas: tuple[str | None, ...]

    @classmethod
    def initial(cls, width: int) -> SpineLineage:
        return cls(tracks=tuple(range(width)), open_ottavas=(None,) * width)

    def advance(self, fields: Sequence[str]) -> SpineLineage:
        """Apply one closed line. Raise ValueError when kern_parser would reject it."""
        if all(field in _SPINE_OP_FIELDS for field in fields):
            return SpineLineage(advance_tracks(self.tracks, fields), self.open_ottavas)
        if not all(field.startswith("*") for field in fields):
            return self

        chosen: dict[int, str] = {}
        for column, field in enumerate(fields):
            if not _is_track_tandem(field):
                continue
            first = chosen.setdefault(self.tracks[column], field)
            if _visible(first) != _visible(field):
                raise ValueError(
                    f"sibling subspines set different values on one line: {first} and {field}"
                )

        open_ottavas = list(self.open_ottavas)
        for track, field in chosen.items():
            if field not in OTTAVA_FIELDS:
                continue
            if field in _FORBIDDEN_OTTAVAS[open_ottavas[track]]:
                raise ValueError(f"{field} does not match the open ottava {open_ottavas[track]}")
            kind = field.removeprefix("*").removeprefix("X")
            open_ottavas[track] = None if field.startswith("*X") else kind
        return SpineLineage(self.tracks, tuple(open_ottavas))

    def sibling_spellings(
        self, column: int, completed_fields: Sequence[str]
    ) -> tuple[str, ...] | None:
        """Fields that agree with a tandem an earlier sibling subspine set on this line."""
        track = self.tracks[column]
        for earlier, field in enumerate(completed_fields):
            if self.tracks[earlier] == track and _is_track_tandem(field):
                visible = _visible(field)
                if visible.startswith(_INVISIBLE_PREFIXES):
                    return ("*", visible, visible + _INVISIBLE_SUFFIX)
                return ("*", visible)
        return None

    def forbidden_fields(self, column: int, completed_fields: Sequence[str]) -> frozenset[str]:
        """Ottava and merge fields that would make this column unresolvable."""
        forbidden = _FORBIDDEN_OTTAVAS[self.open_ottavas[self.tracks[column]]]
        if not self.merge_allowed(column, completed_fields):
            forbidden = forbidden | {MERGE_FIELD}
        return forbidden

    def merge_allowed(self, column: int, completed_fields: Sequence[str]) -> bool:
        tracks = self.tracks
        if completed_fields and completed_fields[-1] == MERGE_FIELD:
            return tracks[column] == tracks[column - 1]
        return column + 1 < len(tracks) and tracks[column + 1] == tracks[column]
