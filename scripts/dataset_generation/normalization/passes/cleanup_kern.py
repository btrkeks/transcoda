"""CleanupKern pass - removes unwanted tokens and artifacts from raw **kern data."""

from __future__ import annotations

import re

from ..base import NormalizationContext

# Precompiled regex patterns for performance
DEFAULT_AVOID_TOKENS: tuple[str, ...] = (
    "*cue",
    "*Xcue",
    "*kcancel",
    "*below",
    "*ped",
    "*Xped",
    "*tuplet",
    "*Xtuplet",
    "*Alternative",
)

# Build a single regex that matches any forbidden token as a whole field.
# Accept both tab and space separators to handle malformed interpretation
# lines from external data (e.g., "*\t*Xtuplet    *").
_ALT = "|".join(re.escape(t) for t in DEFAULT_AVOID_TOKENS)
_TOKEN_FIELD_RE = re.compile(rf"(^|[ \t])({_ALT})(?=[ \t]|$)", re.MULTILINE)

_BAR_NUMBERS_RE = re.compile(r"(?<=\=)\d+")
_X_TOKEN_RE = re.compile(r"(?<!clef)(?<!\*)X")

# e.g., "*", "*\t*", "*    *", "*\t*    *" — and removes the whole line.
_ALL_STAR_LINE = re.compile(r"(?m)^[ \t]*\*(?:[ \t]+\*)*[ \t]*(?:\r?\n|$)")

_REMOVE_CHARS = {ord("<"): None, ord(">"): None, ord("y"): None, ord("j"): None, ord("R"): None, ord("Z"): None}
_USER_DEFINED_MARK_TRANS = str.maketrans({"N": None})


# Matches @...@ or @@...@@ tremolo shorthand blocks (e.g., @16@ or @@8@@)
_TREMOLO_BLOCK_RE = re.compile(r"@+[^@]+@+")

_STEM_DIRECTION_TRANS = str.maketrans({"/": None, "\\": None})
_ORNAMENT_TRANS = str.maketrans({"'": None, "^": None})
_UNSUPPORTED_NOTE_SYMBOL_TRANS = str.maketrans({
    "S": None,
    "$": None,
    "&": None,
    "p": None,
})
_BEAM_SUFFIX_S_RE = re.compile(r"([LJKk]+)s$")

# Matches lines that contain **kern/**ekern declarations, including versioned
# variants such as **kern_1.0 or **ekern_1.0.
_KERN_DECLARATION_LINE = re.compile(
    r"^(?:[^\t\n]*\t)*\*\*e?kern(?:_[^\t\n]+)?(?:\t[^\t\n]*)*(?:\r?\n|$)",
    re.MULTILINE,
)

# Matches any comment line (starting with !) - includes local comments, layout options, and reference records
_COMMENT_LINE = re.compile(r"^!.*(?:\r?\n|$)", re.MULTILINE)

# Matches key interpretation tokens like *A:, *a:, *F#:, *b-: (analytical, not visible on score)
_KEY_INTERPRETATION_RE = re.compile(r"(^|\t)\*[a-gA-G][#-]?:(?=\t|$)", re.MULTILINE)

# Matches instrument designation tokens like *Ioboe, *Ipiano, *Ivox, etc.
_INSTRUMENT_TOKEN_RE = re.compile(r"(^|\t)\*I[^\t\n]*(?=\t|$)", re.MULTILINE)

# Matches metronome/tempo marking tokens like *MM88, *MM120, etc.
_METRONOME_TOKEN_RE = re.compile(r"(^|\t)\*MM\d+(?:\.\d+)?(?=\t|$)", re.MULTILINE)

# Matches transposition interpretation tokens like *Trd0c0, *Trd1c2, *Trd-3c-5, etc.
_TRANSPOSITION_TOKEN_RE = re.compile(r"(^|\t)\*Trd-?\d+c-?\d+(?=\t|$)", re.MULTILINE)

# Matches part designation tokens like *part1, *part2, etc.
_PART_TOKEN_RE = re.compile(r"(^|\t)\*part\d+(?=\t|$)", re.MULTILINE)

# Matches staff assignment tokens like *staff0, *staff1, *staff2, etc.
_STAFF_TOKEN_RE = re.compile(r"(^|\t)\*staff\d+(?=\t|$)", re.MULTILINE)


def _remove_unwanted_tokens(krn: str) -> str:
    """Replace unwanted tokens with '*' when they appear as whole fields."""
    return _TOKEN_FIELD_RE.sub(r"\1*", krn)


def _remove_all_star_lines(krn: str) -> str:
    """Remove lines containing only * tokens."""
    return _ALL_STAR_LINE.sub("", krn)


def _remove_x_tokens(krn: str) -> str:
    """Remove standalone X characters (but not *X interpretations or *clefX)."""
    return _X_TOKEN_RE.sub("", krn)


def _remove_unwanted_chars(krn: str) -> str:
    """Remove <, >, y, j, R characters."""
    return krn.translate(_REMOVE_CHARS)


def _remove_user_defined_marks(krn: str) -> str:
    """Remove user-defined N marks from note fields.

    Preserves these characters in interpretation tokens (starting with *)
    and barlines (starting with =).
    """
    lines = krn.split("\n")
    result_lines = []

    for line in lines:
        fields = line.split("\t")
        result_fields = []

        for field in fields:
            if field.startswith("*") or field.startswith("="):
                result_fields.append(field)
            else:
                result_fields.append(field.translate(_USER_DEFINED_MARK_TRANS))

        result_lines.append("\t".join(result_fields))

    return "\n".join(result_lines)


def _remove_bar_numbers(krn: str) -> str:
    """Remove digits from bar lines (e.g., =42 -> =)."""
    return _BAR_NUMBERS_RE.sub("", krn)


def _remove_tremolo_shorthand(krn: str) -> str:
    """Remove @...@ and @@...@@ tremolo shorthand blocks (e.g., 4c@16@ -> 4c)."""
    return _TREMOLO_BLOCK_RE.sub("", krn)


def _remove_kern_declaration_lines(krn: str) -> str:
    """Remove lines containing **kern or **ekern spine declarations."""
    return _KERN_DECLARATION_LINE.sub("", krn)


def _remove_comments(krn: str) -> str:
    """Remove all comment lines (lines starting with !)."""
    return _COMMENT_LINE.sub("", krn)


def _remove_key_interpretations(krn: str) -> str:
    """Replace key interpretation tokens (*A:, *f#:, etc.) with '*'."""
    return _KEY_INTERPRETATION_RE.sub(r"\1*", krn)


def _remove_instrument_tokens(krn: str) -> str:
    """Replace instrument designation tokens (*Ioboe, *Ipiano, etc.) with '*'."""
    return _INSTRUMENT_TOKEN_RE.sub(r"\1*", krn)


def _remove_metronome_markings(krn: str) -> str:
    """Replace metronome/tempo marking tokens (*MM88, *MM120, etc.) with '*'."""
    return _METRONOME_TOKEN_RE.sub(r"\1*", krn)


def _remove_transposition_tokens(krn: str) -> str:
    """Replace transposition interpretation tokens (*Trd0c0, *Trd1c2, etc.) with '*'."""
    return _TRANSPOSITION_TOKEN_RE.sub(r"\1*", krn)


def _remove_part_tokens(krn: str) -> str:
    """Replace part designation tokens (*part1, *part2, etc.) with '*'."""
    return _PART_TOKEN_RE.sub(r"\1*", krn)


def _remove_staff_tokens(krn: str) -> str:
    """Replace staff assignment tokens (*staff0, *staff1, *staff2, etc.) with '*'."""
    return _STAFF_TOKEN_RE.sub(r"\1*", krn)


def _remove_stem_directions(krn: str) -> str:
    """Remove stem direction markers (/ and \\) from note tokens.

    Preserves these characters in interpretation tokens (starting with *)
    and barlines (starting with =).
    """
    lines = krn.split("\n")
    result_lines = []

    for line in lines:
        fields = line.split("\t")
        result_fields = []

        for field in fields:
            if field.startswith("*") or field.startswith("="):
                result_fields.append(field)
            else:
                result_fields.append(field.translate(_STEM_DIRECTION_TRANS))

        result_lines.append("\t".join(result_fields))

    return "\n".join(result_lines)


def _remove_ornaments(krn: str) -> str:
    """Remove ornament markers (' and ^) from note tokens.

    Preserves these characters in interpretation tokens (starting with *)
    and barlines (starting with =).
    """
    lines = krn.split("\n")
    result_lines = []

    for line in lines:
        fields = line.split("\t")
        result_fields = []

        for field in fields:
            if field.startswith("*") or field.startswith("="):
                result_fields.append(field)
            else:
                result_fields.append(field.translate(_ORNAMENT_TRANS))

        result_lines.append("\t".join(result_fields))

    return "\n".join(result_lines)


def _strip_unsupported_note_symbols(krn: str) -> str:
    """Strip unsupported note symbols from note fields.

    Removes unsupported symbols (S, $, &, p) and strips stray trailing lowercase
    s when it follows beam markers (e.g., Ls, JJs) in note/chord tokens.

    Preserves interpretation and barline fields.
    """
    lines = krn.split("\n")
    result_lines = []

    for line in lines:
        fields = line.split("\t")
        result_fields = []

        for field in fields:
            if field.startswith("*") or field.startswith("="):
                result_fields.append(field)
                continue

            tokens = field.split(" ")
            cleaned_tokens = []
            for token in tokens:
                cleaned = token.translate(_UNSUPPORTED_NOTE_SYMBOL_TRANS)
                cleaned = _BEAM_SUFFIX_S_RE.sub(r"\1", cleaned)
                cleaned_tokens.append(cleaned)

            result_fields.append(" ".join(cleaned_tokens))

        result_lines.append("\t".join(result_fields))

    return "\n".join(result_lines)


class CleanupKern:
    """
    Applies cleanup preprocessing to raw **kern transcriptions from external sources.

    Operations performed in order:
    1. Remove comment lines (lines starting with !)
    2. Remove unwanted tokens (*cue, *ped, etc.)
    3. Remove key interpretation tokens (*A:, *f#:, etc.)
    4. Remove instrument designation tokens (*Ioboe, *Ipiano, etc.)
    5. Remove metronome/tempo markings (*MM88, *MM120, etc.)
    6. Remove transposition interpretations (*Trd0c0, *Trd1c2, etc.)
    7. Remove part designation tokens (*part1, *part2, etc.)
    8. Remove staff assignment tokens (*staff0, *staff1, *staff2, etc.)
    9. Remove lines with only * tokens
    10. Remove X characters (not *X interpretations or *clefX percussion clef)
    11. Remove @...@ tremolo shorthand blocks
    12. Remove unsupported note symbols (S, $, &, p, and beam-suffix s forms)
    13. Remove unwanted characters (<, >, y, j, R, Z)
    14. Remove user-defined N marks from note fields
    15. Remove bar numbers (=42 -> =)
    16. Remove stem directions (/ and \\) from note fields
    17. Remove ornaments (' and ^) from note fields
    18. Remove lines containing **kern or **ekern spine declarations

    This pass is intended for raw **kern from external sources like polish-scores.
    For already-preprocessed data (e.g., synthetic data), use normalize_kern_transcription
    without this pass.
    """

    name = "cleanup_kern"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No preparation needed."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Apply all cleanup operations in sequence."""
        text = _remove_comments(text)
        text = _remove_unwanted_tokens(text)
        text = _remove_key_interpretations(text)
        text = _remove_instrument_tokens(text)
        text = _remove_metronome_markings(text)
        text = _remove_transposition_tokens(text)
        text = _remove_part_tokens(text)
        text = _remove_staff_tokens(text)
        text = _remove_all_star_lines(text)
        text = _remove_x_tokens(text)
        text = _remove_tremolo_shorthand(text)
        text = _strip_unsupported_note_symbols(text)
        text = _remove_unwanted_chars(text)
        text = _remove_user_defined_marks(text)
        text = _remove_bar_numbers(text)
        text = _remove_stem_directions(text)
        text = _remove_ornaments(text)
        text = _remove_kern_declaration_lines(text)
        return text.strip()

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """No validation needed."""
        pass
