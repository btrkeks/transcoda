"""Stored page transcriptions carry no end-of-piece footer.

kern_parser's humdrum-kern serializer appends the `*-` terminator and its default
`!!!RDF**kern:` signifier lines whenever it renders, merges or exports, and drops any it
reads. A stored page therefore holds neither, so pages look the same whichever tool wrote
them. A final barline (`==`) is music and stays.
"""

from __future__ import annotations

# The signifier lines kern_parser's humdrum-kern preset emits on every serialization.
SYNTHESIZED_SIGNIFIERS = frozenset(
    {
        "!!!RDF**kern: > = above",
        "!!!RDF**kern: < = below",
        "!!!RDF**kern: N = linked",
        "!!!RDF**kern: Z = editorial accidental, parentheses",
        "!!!RDF**kern: j = editorial accidental, brackets",
    }
)


def _is_terminator(line: str) -> bool:
    return set(line.rstrip("\r").split("\t")) == {"*-"}


def strip_footer(text: str) -> str:
    """Drop synthesized signifier lines and the final all-spine `*-` record.

    A `*-` record is the footer only when nothing but global comments or blank lines
    follows it; a terminator followed by more data is left alone.
    """
    lines = [line for line in text.split("\n") if line.rstrip("\r") not in SYNTHESIZED_SIGNIFIERS]
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index]
        if not line.strip() or line.startswith("!!"):
            continue
        if _is_terminator(line):
            del lines[index]
        break
    return "\n".join(lines)


def has_footer(text: str) -> bool:
    return strip_footer(text) != text
