"""FixNoteBeams pass - fixes beams that extend over multiple bar lines.

This module handles fixing beams that extend over multiple bar lines in kern format.
Beams that cross bar lines need to be properly closed with 'J' before each bar line
and reopened with 'L' after.
"""

from __future__ import annotations

import math

from src.core.kern_utils import (
    get_duration_of_token,
    is_grace_note,
    is_note_token,
    split_into_same_spine_nr_chunks_and_measures,
)

from ..base import NormalizationContext


def _increment_beam_count(lst: list[float], spine_nr: int, count: int = 1) -> None:
    if spine_nr >= len(lst):
        lst.extend([0] * (spine_nr + 1 - len(lst)))
    lst[spine_nr] += count


def _decrement_beam_count(lst: list[float], spine_nr: int, count: int = 1) -> None:
    if spine_nr >= len(lst):
        lst.extend([0] * (spine_nr + 1 - len(lst)))
    lst[spine_nr] -= count


def _get_num_beams_for_duration(duration: int) -> int:
    num_beams = max(0, int(math.log2(duration)) - 2)
    return num_beams


def _get_num_of_open_beams(counts: list[float], j: int) -> int:
    return max(0, int(counts[j]) if j < len(counts) else 0)


def _get_beam_count(counts: list[float], j: int) -> float:
    return counts[j] if j < len(counts) else 0


def _replace_trailing_ls_with_ks(s: str, num: int) -> str:
    assert s.endswith("L" * num), f"String does not have enough trailing Ls to replace: {s}"
    return s[:-num] + "K" * num


def _is_opening_a_beam(field: str) -> bool:
    return "L" in field and is_note_token(field)


def _fix_beams_in_snippet(snippet: str) -> str:
    """Fix beams in a snippet with same number of spines and no measure boundaries."""
    grid: list[list[str]] = [line.split("\t") for line in snippet.splitlines()]
    beam_counts: list[float] = []

    for i, row in enumerate(grid):
        for j, field in enumerate(row):
            if ("L" in field or "K" in field) and is_note_token(field):
                duration = get_duration_of_token(field)

                if duration is None and is_grace_note(field):
                    # Grace notes are durationless per humdrum specification, but verovio needs a duration
                    # We simply assign them a duration of 8
                    row[j] = "8" + field
                    duration = 8

                assert duration is not None, (
                    f"Could not parse duration of token {field} in snippet:\n{snippet}"
                )

                num_beams = _get_num_beams_for_duration(duration)
                num_open_beams = _get_num_of_open_beams(beam_counts, j)

                if num_open_beams > num_beams:
                    # Need to replace some Ls with Ks again
                    need_to_convert = num_open_beams - num_beams
                    _decrement_beam_count(beam_counts, j, need_to_convert)
                    num_open_beams = _get_num_of_open_beams(beam_counts, j)
                    assert num_open_beams == num_beams, (
                        "Open beam count should match after conversion of L to K"
                    )

                assert num_open_beams <= num_beams, (
                    "Cannot have more open beams than allowed by duration"
                )
                beams_to_open = num_beams - num_open_beams

                # Open beams
                row[j] = field.replace("L", "").replace("K", "") + beams_to_open * "L"
                _increment_beam_count(beam_counts, j, beams_to_open)

            elif "J" in field or ("k" in field and is_note_token(field)):
                duration = get_duration_of_token(field)

                if duration is None and is_grace_note(field):
                    # Grace notes are durationless per humdrum specification, but verovio needs a duration
                    # We simply assign them a duration of 8
                    row[j] = "8" + field
                    duration = 8

                assert duration is not None, (
                    f"Could not parse duration of token {field} in snippet:\n{snippet}"
                )

                num_beams = _get_num_beams_for_duration(duration)
                num_open_beams = _get_num_of_open_beams(beam_counts, j)

                # Fix number of open beams if more are open than should be
                if num_open_beams > num_beams:
                    # Cannot close more beams than are open
                    # Some of the previous Ls need to be converted to Ks

                    need_to_convert = num_open_beams - num_beams
                    # Go lines up until we find the field containing the L to convert
                    found_L = False
                    for i_ in range(i - 1, -1, -1):
                        if "L" in grid[i_][j]:
                            found_L = True
                            # Found the L field to convert
                            # Replace the last @need_to_convert number of Ls with Ks
                            field_to_modify = grid[i_][j]

                            # -- Sanity check --
                            l_count = field_to_modify.count("L")
                            assert l_count >= need_to_convert, "Not enough Ls to convert to Ks"
                            # --

                            grid[i_][j] = _replace_trailing_ls_with_ks(
                                field_to_modify, need_to_convert
                            )
                            _decrement_beam_count(beam_counts, j, need_to_convert)
                            break
                    assert found_L, (
                        f"Could not find L to convert to K for spine {j + 1} from line {i + 1} "
                        f"in snippet:\n{[row[j] for row in grid[: i + 1]]}"
                    )

                    num_open_beams = _get_num_of_open_beams(beam_counts, j)
                    assert num_open_beams == num_beams, (
                        f"Open beam count {num_open_beams} should match num beams {num_beams} "
                        "after conversion of L to K"
                    )

                assert num_open_beams <= num_beams, "Cannot close more beams than are open"
                num_full_beams = num_open_beams  # Close all open beams from the left
                num_partial_beams = num_beams - num_full_beams  # Close off any remaining beams

                # Close beams
                row[j] = (
                    field.replace("J", "").replace("k", "")
                    + num_full_beams * "J"
                    + num_partial_beams * "k"
                )
                _decrement_beam_count(beam_counts, j, num_full_beams)

                assert _get_beam_count(beam_counts, j) == 0, (
                    "Correct number of beams should be closed after J/k processing."
                )

    # Only open beams should remain at this point
    assert all(count >= 0 for count in beam_counts), "Too many closed beams."

    # Close off any trailing open beams at the end of the snippet
    for j in range(len(beam_counts)):
        if _get_num_of_open_beams(beam_counts, j) > 0:
            num_open_beams = _get_num_of_open_beams(beam_counts, j)

            for i in range(len(grid) - 1, -1, -1):
                field = grid[i][j]
                if is_note_token(field):
                    if _is_opening_a_beam(field):
                        # Case: The beam was opened by the last note in the measure
                        # Remove the correct number of opened beams
                        new_field = field.replace("L", "", num_open_beams)
                        removed_ls = len(field) - len(new_field)
                        assert removed_ls == num_open_beams, "Not enough Ls to remove."
                        grid[i][j] = new_field
                        # TODO: There is the small possibility that some of the beams
                        # were also opened by some earlier tokens
                    else:
                        # Close the remaining open beams
                        # We can safely assume that there are no 'k' chars here,
                        # as this would imply that we were in the above 'J' processing loop already
                        grid[i][j] = grid[i][j] + "J" * num_open_beams
                    _decrement_beam_count(beam_counts, j, num_open_beams)
                    break

    # All beams should be closed now
    assert all(count == 0 for count in beam_counts), (
        "Not all beams were closed at the end of snippet."
    )

    return "\n".join("\t".join(row) for row in grid)


class FixNoteBeams:
    """
    Fixes beams that extend over multiple bar lines in **kern notation.

    In Humdrum kern, beams that cross bar lines need to be properly closed
    with 'J' before each bar line and reopened with 'L' after. This pass
    ensures syntactically valid beam groupings.

    The algorithm works by:
    1. Splitting the transcription into chunks with uniform spine count
    2. Processing each chunk independently to fix beam markers
    3. Rejoining the chunks

    Beam markers:
    - L: Start a beam (connects note to next note)
    - J: End a beam (connects note to previous note)
    - K: Partial beam start (doesn't connect, just extends right)
    - k: Partial beam end (doesn't connect, just extends left)
    """

    name = "fix_note_beams"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        """No preparation needed."""
        pass

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        """Fix beams that extend over multiple bars."""
        measures = split_into_same_spine_nr_chunks_and_measures(text)
        fixed_measures = [_fix_beams_in_snippet(measure) for measure in measures]
        result = "\n".join(fixed_measures)
        return result

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        """No validation needed."""
        pass
