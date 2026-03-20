"""Pass to canonicalize the order of components within note tokens.

This pass reorders the components of each note token to match the canonical
order defined in the kern grammar:

Regular notes:
    duration pitch accidental? pause? ornament* harmonic? articulation? arpeggio? slur? breath? tie-mark? beam? editorial?

Grace notes:
    duration? pitch accidental? "q" slur? tie-mark? beam?

Groupetto:
    duration "qq" pitch accidental? beam?
"""

from __future__ import annotations

import re

from ..base import NormalizationContext

# Character categories for classification
ACCIDENTALS = {"#", "-", "n"}
PAUSE = {";"}
ORNAMENTS = {"T", "t", "M", "m", "W", "w", "S", "$", "R", "O"}
HARMONIC = {"o"}
ARTICULATIONS = {"'", "`", "~", "^", "i", "I", "v", "u"}
ARPEGGIO = {":"}
SLURS = {"(", ")"}
BREATH = {","}
TIES = {"[", "]", "_"}
BEAMS = {"L", "J", "K", "k"}
EDITORIAL = {"y", "x"}

# Canonical order for ornaments when multiple are present
# Trills → Mordents → Inverted mordents → Turns → Concluding turn → Generic
ORNAMENT_ORDER = {
    "T": 0,  # whole-tone trill
    "t": 1,  # semitone trill
    "M": 2,  # whole-tone mordent
    "m": 3,  # semitone mordent
    "W": 4,  # whole-tone inverted mordent
    "w": 5,  # semitone inverted mordent
    "S": 6,  # turn
    "$": 7,  # inverted turn
    "R": 8,  # concluding turn
    "O": 9,  # generic ornament
}


def _count_unknown_char(unknown_counter: dict[str, int] | None, char: str) -> None:
    """Increment unknown-char counter when enabled."""
    if unknown_counter is None:
        return
    unknown_counter[char] = unknown_counter.get(char, 0) + 1


def _is_note_token(token: str) -> bool:
    """Check if token contains a pitch (a-g or A-G) or is a rest."""
    if not token or token == ".":
        return False
    if token.startswith("=") or token.startswith("*") or token.startswith("!"):
        return False
    # Check for pitch letters or rest
    return bool(re.search(r"[a-gA-Gr]", token))


def _classify_char(char: str, next_char: str | None = None) -> tuple[str, int]:
    """
    Classify a character and return (category, consume_count).

    Returns:
        Tuple of (category_name, number_of_chars_to_consume)
    """
    if char in ACCIDENTALS:
        # Handle double accidentals (## or --)
        if next_char == char and char in {"#", "-"}:
            return ("accidental", 2)
        return ("accidental", 1)
    elif char in PAUSE:
        return ("pause", 1)
    elif char in ORNAMENTS:
        return ("ornament", 1)
    elif char in HARMONIC:
        return ("harmonic", 1)
    elif char in ARTICULATIONS:
        return ("articulation", 1)
    elif char in ARPEGGIO:
        return ("arpeggio", 1)
    elif char in SLURS:
        # Handle double slurs (( or ))
        if next_char == char:
            return ("slur", 2)
        return ("slur", 1)
    elif char in BREATH:
        return ("breath", 1)
    elif char in TIES:
        # Handle double ties [[ or ]] or __
        if next_char == char:
            return ("tie", 2)
        return ("tie", 1)
    elif char in BEAMS:
        return ("beam", 1)
    elif char in EDITORIAL:
        # Handle yy
        if char == "y" and next_char == "y":
            return ("editorial", 2)
        return ("editorial", 1)
    elif char == "q":
        return ("grace_q", 1)
    return ("unknown", 1)


def _parse_note_components(
    token: str,
    unknown_counter: dict[str, int] | None = None,
) -> dict[str, str] | None:
    """
    Parse a note token into its component parts.

    Returns a dictionary with keys:
        duration, pitch, accidental, pause, ornament, articulation, arpeggio,
        slur, breath, tie, beam, editorial, grace_q, is_groupetto, is_rest

    Returns None if the token is not a valid note.
    """
    components = {
        "duration": "",
        "pitch": "",
        "accidental": "",
        "pause": "",
        "ornament": "",
        "harmonic": "",
        "articulation": "",
        "arpeggio": "",
        "slur": "",
        "breath": "",
        "tie": "",
        "beam": "",
        "editorial": "",
        "grace_q": "",  # Will be "q" for grace notes
        "is_groupetto": False,
        "is_rest": False,
    }

    remaining = token

    # Check for groupetto pattern (qq before pitch) anywhere in token
    # Groupetto: duration "qq" pitch accidental? beam?
    groupetto_match = re.search(r"(\d+\.*)qq([a-gA-G]+)", remaining)
    if groupetto_match:
        components["duration"] = groupetto_match.group(1)
        components["pitch"] = groupetto_match.group(2)
        components["is_groupetto"] = True
        # Get remaining after the groupetto core
        remaining = remaining[groupetto_match.end() :]

        # Parse remaining (accidental, beam only for groupetto)
        for char in remaining:
            if char in ACCIDENTALS:
                components["accidental"] += char
            elif char in BEAMS:
                components["beam"] += char
            else:
                _count_unknown_char(unknown_counter, char)
            # Groupetto doesn't have other modifiers per grammar

        return components

    # Step 1: Consume any leading modifiers (misplaced ties, slurs, etc.)
    # These come before duration and pitch but should be moved after
    i = 0
    while i < len(remaining):
        char = remaining[i]
        next_char = remaining[i + 1] if i + 1 < len(remaining) else None

        # Stop at duration digits
        if char.isdigit():
            break
        # Stop at pitch letters
        if char in "abcdefgABCDEFG":
            break
        # Stop at rest
        if char == "r":
            break

        # Classify and consume the modifier
        category, consume = _classify_char(char, next_char)
        if category != "unknown" and category != "grace_q":
            components[category] += remaining[i : i + consume]
            i += consume
        elif category == "grace_q":
            # q before pitch means grace note format: q pitch
            # But we need to check if next char is a pitch
            if next_char and next_char in "abcdefgABCDEFG":
                components["grace_q"] = "q"
                i += 1
            else:
                # Unknown q, skip it
                i += 1
        else:
            # Unknown character at start, skip
            _count_unknown_char(unknown_counter, char)
            i += 1

    remaining = remaining[i:]

    # Step 2: Parse duration (digits + dots)
    duration_match = re.match(r"^(\d+\.*)", remaining)
    if duration_match:
        components["duration"] = duration_match.group(1)
        remaining = remaining[len(components["duration"]) :]

    # Step 3: Check for rest
    if remaining.startswith("r"):
        components["is_rest"] = True
        components["pitch"] = "r"
        remaining = remaining[1:]
        # Rests can have editorial markers (yy)
        for char in remaining:
            if char in EDITORIAL:
                components["editorial"] += char
        return components

    # Step 4: Check for grace note marker "q" before pitch (alternate format)
    if remaining.startswith("q") and len(remaining) > 1 and remaining[1] in "abcdefgABCDEFG":
        components["grace_q"] = "q"
        remaining = remaining[1:]

    # Step 5: Parse pitch (consecutive same letters, case-sensitive)
    pitch_match = re.match(r"^([a-gA-G])\1*", remaining)
    if pitch_match:
        components["pitch"] = pitch_match.group(0)
        remaining = remaining[len(components["pitch"]) :]
    else:
        # No valid pitch found
        if not components["pitch"]:
            return None

    # Step 6: Parse remaining characters by category
    i = 0
    while i < len(remaining):
        char = remaining[i]
        next_char = remaining[i + 1] if i + 1 < len(remaining) else None

        category, consume = _classify_char(char, next_char)
        if category == "grace_q":
            components["grace_q"] = "q"
            i += consume
        elif category != "unknown":
            components[category] += remaining[i : i + consume]
            i += consume
        else:
            # Fail-safe: drop unknown characters instead of preserving them.
            _count_unknown_char(unknown_counter, char)
            i += 1

    return components


def _sort_ornaments(ornaments: str) -> str:
    """Sort ornament characters according to canonical order."""
    if len(ornaments) <= 1:
        return ornaments
    return "".join(sorted(ornaments, key=lambda c: ORNAMENT_ORDER.get(c, 99)))


def _reconstruct_canonical(components: dict[str, str]) -> str:
    """
    Reconstruct a note token from components in canonical order.

    Canonical order for regular notes:
        duration pitch accidental pause ornament harmonic articulation arpeggio slur breath tie beam editorial

    Canonical order for grace notes:
        duration pitch accidental q slur tie beam

    Groupetto:
        duration qq pitch accidental beam
    """
    if components["is_rest"]:
        return components["duration"] + "r" + components["editorial"]

    if components["is_groupetto"]:
        return (
            components["duration"]
            + "qq"
            + components["pitch"]
            + components["accidental"]
            + components["beam"]
        )

    if components["grace_q"]:
        # Grace note canonical order: duration pitch accidental q slur tie beam
        return (
            components["duration"]
            + components["pitch"]
            + components["accidental"]
            + "q"
            + components["slur"]
            + components["tie"]
            + components["beam"]
        )

    # Regular note canonical order (with sorted ornaments)
    return (
        components["duration"]
        + components["pitch"]
        + components["accidental"]
        + components["pause"]
        + _sort_ornaments(components["ornament"])
        + components["harmonic"]
        + components["articulation"]
        + components["arpeggio"]
        + components["slur"]
        + components["breath"]
        + components["tie"]
        + components["beam"]
        + components["editorial"]
    )


def _canonicalize_token(
    token: str,
    unknown_counter: dict[str, int] | None = None,
) -> str:
    """Canonicalize a single token if it's a note."""
    if not _is_note_token(token):
        return token

    components = _parse_note_components(token, unknown_counter=unknown_counter)
    if components is None:
        return token

    return _reconstruct_canonical(components)


def _process_spine(
    spine: str,
    unknown_counter: dict[str, int] | None = None,
) -> str:
    """Process a single spine, which may contain space-separated notes (chord)."""
    # Split by spaces to get individual tokens
    tokens = spine.split(" ")
    result_tokens = [
        _canonicalize_token(token, unknown_counter=unknown_counter)
        for token in tokens
    ]
    return " ".join(result_tokens)


class CanonicalizeNoteOrder:
    """
    Reorder note token components to canonical order.

    This pass ensures that all note tokens have their components in the
    canonical order defined by the kern grammar. This is important for:
    1. Grammar validation to succeed
    2. Consistent model training data
    3. Deterministic output format

    Example transformations:
        [8fJ -> 8f[J      (tie moved after pitch)
        (4c#L -> 4c#(L    (slur moved after accidental)
        8qc -> 8cq        (grace note: q moved after pitch)
        8cTmw;vL -> 8c;TmwvL  (pause before ornaments, ornaments sorted: T<m<w)
        4c[,L -> 4c,[L    (breath moved before tie)
        8d)o -> 8do)       (harmonic moved before slur)
    """

    name = "canonicalize_note_order"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """Initialize per-pass unknown-character drop counters."""
        ctx[self.name] = {
            "unknown_char_drop_counts": {},
            "unknown_char_drops_total": 0,
        }

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Transform by canonicalizing the order of note components."""
        pass_stats = ctx.setdefault(
            self.name,
            {
                "unknown_char_drop_counts": {},
                "unknown_char_drops_total": 0,
            },
        )
        unknown_counter = pass_stats.setdefault("unknown_char_drop_counts", {})

        lines = text.split("\n")
        result_lines = []

        for line in lines:
            # Split by tabs to get spines
            spines = line.split("\t")
            result_spines = [
                _process_spine(spine, unknown_counter=unknown_counter)
                for spine in spines
            ]
            result_lines.append("\t".join(result_spines))

        pass_stats["unknown_char_drops_total"] = int(sum(unknown_counter.values()))
        return "\n".join(result_lines)

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """No validation needed."""
        pass
