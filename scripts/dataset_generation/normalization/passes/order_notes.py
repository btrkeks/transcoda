"""Pass to order notes in chords by pitch."""

import re

from ..base import NormalizationContext


class OrderNotes:
    """
    Orders notes within chords by pitch (lowest to highest).

    In kern notation, chords are represented by multiple note tokens with spaces.
    This pass reorders them to follow a canonical pitch ordering.

    Example:
        Input:  "4g 4e 4c"  (chord with G, E, C)
        Output: "4c 4e 4g"  (ordered from lowest to highest)

    TODO: Implement the actual pitch parsing and ordering logic
    """

    name = "order_notes"

    def __init__(self, ascending: bool = True):
        """
        Initialize the pass.

        Args:
            ascending: If True, order from lowest to highest pitch.
                      If False, order from highest to lowest.
        """
        self.ascending = ascending

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """
        Parse the kern string to identify chords.

        TODO: Implement chord detection and parsing
        - Identify chord boundaries
        - Parse individual notes within chords
        - Store parsed data in ctx["order_notes"] for reuse
        """
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """
        Reorder notes within each chord by pitch.
        """
        lines = text.split("\n")
        result_lines = []

        for line in lines:
            # Split by tabs to get spines
            spines = line.split("\t")
            result_spines = []

            for spine in spines:
                result_spines.append(self._process_spine(spine))

            result_lines.append("\t".join(result_spines))

        return "\n".join(result_lines)

    def _process_spine(self, spine: str) -> str:
        """Process a single spine (token)."""
        # Check if spine should be processed
        if not self._should_process(spine):
            return spine

        # Split by spaces to get individual notes
        notes_strs = spine.split(" ")

        # If only one note, no need to sort
        if len(notes_strs) == 1:
            return spine

        # Parse all notes
        notes = []
        for note_str in notes_strs:
            parsed = parse_kern_note(note_str)
            if parsed and not parsed["is_rest"]:
                notes.append(parsed)

        # If no valid notes, return original
        if not notes:
            return spine

        # Sort notes
        sorted_notes = sort_notes_by_pitch(notes, self.ascending)

        # Normalize beam markers: collect from all notes, place only on last
        _BEAM_CHARS = set("LJKk")
        beam_str = ""
        for note in sorted_notes:
            mods = note["_modifiers"]
            note_beams = "".join(c for c in mods if c in _BEAM_CHARS)
            if note_beams and not beam_str:
                beam_str = note_beams
            note["_modifiers"] = "".join(c for c in mods if c not in _BEAM_CHARS)

        if beam_str:
            sorted_notes[-1]["_modifiers"] += beam_str

        # Reconstruct
        reconstructed = [reconstruct_kern_note(note) for note in sorted_notes]

        return " ".join(reconstructed)

    def _should_process(self, spine: str) -> bool:
        """Check if spine should be processed."""
        # Skip barlines, interpretations, null tokens, and reference records
        if spine.startswith("=") or spine.startswith("*") or spine.startswith("!") or spine == ".":
            return False

        # Skip if it's just a rest
        if "r" in spine and not any(c in spine for c in "abcdefgABCDEFG"):
            return False

        # Check if it contains notes
        return bool(re.search(r"[a-gA-G]", spine))

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """
        Validate that the output is still valid kern notation.

        TODO: Implement validation
        - Check that no notes were lost
        - Verify kern syntax is still valid
        - Optionally verify ordering is correct
        """
        pass


# Helper functions to implement:


def parse_kern_note(note_token: str) -> dict:
    """
    Parse a single kern note token into components.

    Args:
        note_token: A kern note like "4c#" or "8ee-"

    Returns:
        Dictionary with keys: pitch, octave, duration, accidental, etc.
    """
    result = {
        "duration": "",
        "pitch": "",  # normalized lowercase letter
        "octave": 0,
        "accidental": "",
        "original": note_token,
        "is_rest": False,
        "midi": 0,
        "articulations": [],
        "ties": [],
        "_pitch_str": "",  # original pitch representation for reconstruction
        "_modifiers": "",  # other modifiers for reconstruction
    }

    remaining = note_token

    # Parse duration (digits followed by optional dots)
    duration_match = re.match(r"^(\d+(?:%\d+)?\.*)", remaining)
    if duration_match:
        result["duration"] = duration_match.group(1)
        remaining = remaining[len(result["duration"]) :]

    # Check if it's a rest
    if remaining.startswith("r"):
        result["is_rest"] = True
        result["pitch"] = "r"
        return result

    # Parse pitch (consecutive same letters)
    pitch_match = re.match(r"^([a-gA-G]+)", remaining)
    if not pitch_match:
        # Not a valid note
        return None

    pitch_str = pitch_match.group(1)
    result["_pitch_str"] = pitch_str
    remaining = remaining[len(pitch_str) :]

    # Determine octave and pitch letter
    pitch_letter = pitch_str[0]  # Keep original case
    pitch_letter_normalized = pitch_letter.lower()  # For MIDI calculation
    count = len(pitch_str)

    if pitch_str[0].islower():
        # Lowercase: base octave 4, each additional letter adds 1
        octave = 3 + count
    else:
        # Uppercase: base octave 3, each additional letter subtracts 1
        octave = 4 - count

    result["pitch"] = pitch_letter
    result["octave"] = octave

    # Parse accidental
    if remaining and remaining[0] in "#-":
        result["accidental"] = remaining[0]
        remaining = remaining[1:]

    # Parse remaining modifiers (articulations, ties, etc.)
    result["_modifiers"] = remaining
    for char in remaining:
        if char in "LJk":
            result["articulations"].append(char)
        elif char in "[]{}":
            result["ties"].append(char)

    # Calculate MIDI (use normalized lowercase pitch)
    result["midi"] = kern_pitch_to_midi(pitch_letter_normalized, octave, result["accidental"])

    return result


def kern_pitch_to_midi(pitch: str, octave: int, accidental: str = "") -> int:
    """
    Convert kern pitch notation to MIDI note number.

    Args:
        pitch: Base pitch ('c', 'd', 'e', etc.)
        octave: Octave number
        accidental: Accidental ('#', '-', 'n', etc.)

    Returns:
        MIDI note number (0-127)
    """
    # Pitch class to semitones from C
    pitch_class = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}

    # Base MIDI number (C-1 = 0, so we add 1 to octave)
    midi = 12 * (octave + 1) + pitch_class[pitch.lower()]

    # Apply accidental
    if accidental == "#":
        midi += 1
    elif accidental == "-":
        midi -= 1

    return midi


def sort_notes_by_pitch(notes: list[dict], ascending: bool = True) -> list[dict]:
    """
    Sort a list of parsed notes by pitch.

    Args:
        notes: List of parsed note dictionaries
        ascending: Sort order (True for low to high)

    Returns:
        Sorted list of notes
    """
    return sorted(notes, key=lambda n: n["midi"], reverse=not ascending)


def reconstruct_kern_note(note_dict: dict) -> str:
    """
    Reconstruct a kern note string from parsed components.

    Args:
        note_dict: Parsed note dictionary

    Returns:
        Reconstructed kern note string
    """
    if note_dict["is_rest"]:
        return note_dict["original"]

    return (
        note_dict["duration"]
        + note_dict["_pitch_str"]
        + note_dict["accidental"]
        + note_dict["_modifiers"]
    )
