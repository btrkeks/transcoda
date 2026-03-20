"""Invisible rest augmentation for multi-voice regions."""

from __future__ import annotations

import random

from scripts.dataset_generation.dataset_generation.image_generation.kern_ops import is_spinemerge_line, is_spinesplit_line

from .kern_utils import append_to_token

__all__ = ["apply_invisible_rests"]

# The suffix marking a rest as invisible
INVISIBLE_REST_SUFFIX = "yy"


def apply_invisible_rests(
    krn: str,
    per_rest_probability: float = 0.5,
) -> str:
    """Add 'yy' suffix to rests in multi-voice regions.

    In multi-voice music (between *^ and *v), rests are often hidden
    to reduce visual clutter. This augmentation trains the model to
    recognize invisible rests marked with 'yy'.

    Args:
        krn: The kern string to modify.
        per_rest_probability: Probability of marking each eligible rest
            as invisible. Defaults to 0.5.

    Returns:
        Modified kern string with 'yy' added to selected rests.
    """
    if not krn:
        return krn

    lines = krn.splitlines()
    result_lines: list[str] = []
    in_multivoice_region = False

    for line in lines:
        # Check for region boundaries
        if is_spinesplit_line(line):
            in_multivoice_region = True
            result_lines.append(line)
            continue

        if is_spinemerge_line(line):
            in_multivoice_region = False
            result_lines.append(line)
            continue

        # Only process data lines in multi-voice regions
        if not in_multivoice_region or _is_structural_line(line):
            result_lines.append(line)
            continue

        # Process data line: apply yy to rests (unless all are rests)
        new_line = _process_data_line(line, per_rest_probability)
        result_lines.append(new_line)

    return "\n".join(result_lines)


def _is_structural_line(line: str) -> bool:
    """Check if line is structural (interpretation, barline, empty)."""
    if not line:
        return True
    if line.startswith("*") or line.startswith("="):
        return True
    return False


def _is_rest_token(token: str) -> bool:
    """Check if a token represents a rest."""
    return "r" in token.lower()


def _all_tokens_are_rests(tokens: list[str]) -> bool:
    """Check if all tokens in a line are rests."""
    return all(_is_rest_token(t) for t in tokens)


def _process_data_line(line: str, probability: float) -> str:
    """Process a data line, adding yy to selected rests.

    Skips lines where ALL tokens are rests (everyone resting).
    """
    tokens = line.split("\t")

    # Skip if all tokens are rests
    if _all_tokens_are_rests(tokens):
        return line

    # Apply yy to rests with probability
    new_tokens: list[str] = []
    for token in tokens:
        if _is_rest_token(token) and random.random() < probability:
            new_tokens.append(append_to_token(token, INVISIBLE_REST_SUFFIX))
        else:
            new_tokens.append(token)

    return "\t".join(new_tokens)
