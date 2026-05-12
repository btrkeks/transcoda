"""Generate random metadata (title/author) for scores."""

from __future__ import annotations

import names
from wonderwords import RandomSentence

_sentence_generator = RandomSentence()


def generate_metadata_prefix(include_title: bool, include_author: bool) -> str:
    """Generate random title and/or author metadata for a score.

    Args:
        include_title: Whether to include a random title.
        include_author: Whether to include a random author name.

    Returns:
        Humdrum reference record string (may be empty).
    """
    parts: list[str] = []

    if include_title:
        title = _sentence_generator.sentence()
        parts.append(f"!!!OTL:{title}\n")

    if include_author:
        author = names.get_full_name()
        parts.append(f"!!!COM:{author}\n")

    return "".join(parts)
