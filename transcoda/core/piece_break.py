"""Continuous-spine page text and standalone publication parts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from .kern_concatenation import (
    _advance_lineage,
    _initial_lineage,
    restore_terminal_spine_count_before_final_barline,
    summarize_spine_topology,
)
from .spine_state import is_interpretation_record

PIECE_BREAK = "!!piece-break"
# A new piece starts a new system: the earlier piece's final barline ends one.
BOUNDARY_LINE_BREAK = "!!LO:LB:g=original"


def _is_line_break(line: str) -> bool:
    stripped = line.rstrip("\r\n")
    return stripped == "!!LO:LB" or stripped.startswith("!!LO:LB:")


def _final_barline_index(lines: Sequence[str]) -> int | None:
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].startswith("="):
            return index
    return None


def _with_boundary_break(lines: list[str]) -> None:
    index = _final_barline_index(lines)
    if index is not None and not (index and _is_line_break(lines[index - 1])):
        lines.insert(index, BOUNDARY_LINE_BREAK + "\n")


def _without_boundary_break(chunk: str) -> str:
    lines = chunk.splitlines(keepends=True)
    index = _final_barline_index(lines)
    if index and _is_line_break(lines[index - 1]):
        del lines[index - 1]
    return "".join(lines)


def strip_piece_breaks(text: str) -> str:
    return "".join(
        line for line in text.splitlines(keepends=True) if line.rstrip("\r\n") != PIECE_BREAK
    )


def _chunks(text: str) -> list[str]:
    chunks = [""]
    for line in text.splitlines(keepends=True):
        if line.rstrip("\r\n") == PIECE_BREAK:
            chunks.append("")
        else:
            chunks[-1] += line
    return chunks


def _header(text: str) -> tuple[str, ...]:
    for line in text.splitlines():
        if line.startswith("**"):
            fields = tuple(line.split("\t"))
            if not all(field.startswith("**") for field in fields):
                raise ValueError("Piece headers must name every spine type.")
            return fields
    return ()


def _records(text: str, types: tuple[str, ...]):
    lineage = _initial_lineage(len(types))
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        if not stripped or stripped.startswith("!!"):
            yield line, None, lineage
            continue
        fields = stripped.split("\t")
        if len(fields) != len(lineage):
            raise ValueError(f"Spine width mismatch in {stripped!r}")
        yield line, fields, lineage
        if is_interpretation_record(fields):
            lineage = _advance_lineage(lineage, fields)


def _standalone(chunk: str, types: tuple[str, ...]) -> str:
    if not types:
        if chunk.strip():
            raise ValueError("Piece text has no exclusive interpretation header.")
        return ""
    text = chunk if _header(chunk) else "\t".join(types) + "\n" + chunk
    used: set[int] = set()
    terminal = _initial_lineage(len(types))
    for _, fields, lineage in _records(text, types):
        if fields is None:
            continue
        if not fields[0].startswith(("*", "!", "=")):
            for field, roots in zip(fields, lineage, strict=True):
                if field != ".":
                    used.update(roots)
        terminal = (
            _advance_lineage(lineage, fields) if is_interpretation_record(fields) else lineage
        )
    if not used:
        return ""
    result = []
    for line, fields, lineage in _records(text, types):
        if fields is None:
            result.append(line)
        else:
            kept = [
                field
                for field, roots in zip(fields, lineage, strict=True)
                if any(root in used for root in roots)
            ]
            if kept:
                result.append("\t".join(kept) + ("\n" if line.endswith("\n") else ""))
    if terminal:
        if result and not result[-1].endswith("\n"):
            result[-1] += "\n"
        result.append(
            "\t".join("*-" for roots in terminal if any(root in used for root in roots)) + "\n"
        )
    return "".join(result)


def split_parts(text: str) -> list[str]:
    chunks = _chunks(text)
    if len(chunks) == 1:
        return [text]
    types = _header(text)
    parts = []
    for position, chunk in enumerate(chunks):
        last = position == len(chunks) - 1
        parts.append(_standalone(chunk if last else _without_boundary_break(chunk), types))
        for _, fields, lineage in _records(chunk, types):
            if fields is not None:
                terminal = (
                    _advance_lineage(lineage, fields)
                    if is_interpretation_record(fields)
                    else lineage
                )
                next_types = tuple(types[roots[0]] for roots in terminal)
        if chunk.strip():
            types = next_types
    return parts


def part_line_ranges(text: str) -> tuple[tuple[int, int], ...]:
    """Inclusive 1-based page lines of each chunk `split_parts` yields; a marker line opens its
    chunk."""
    lines = text.splitlines()
    starts = [1] + [number for number, line in enumerate(lines, 1) if line == PIECE_BREAK]
    ends = [start - 1 for start in starts[1:]] + [len(lines)]
    return tuple(zip(starts, ends, strict=True))


def _keys(types: tuple[str, ...]) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    keys = []
    for kind in types:
        counts[kind] += 1
        keys.append((kind, counts[kind]))
    return keys


def _close_boundary(text: str) -> str:
    types = _header(text)
    if not types:
        return text
    lines = text.splitlines(keepends=True)
    last = None
    boundary = len(lines)
    width = len(types)
    for index, (_, fields, lineage) in enumerate(_records(text, types)):
        if fields is None:
            continue
        if all(field == "*-" for field in fields):
            boundary, width = index, len(lineage)
            break
        last = fields
        width = (
            len(_advance_lineage(lineage, fields))
            if is_interpretation_record(fields)
            else len(lineage)
        )
    if last is not None and not last[0].startswith("="):
        lines.insert(boundary, "\t".join("==" for _ in range(width)) + "\n")
    return "".join(lines)


def join_parts(parts: Sequence[str]) -> str:
    if not parts:
        raise ValueError("A page needs at least one part.")
    if len(parts) == 1:
        return parts[0]
    prepared = [
        restore_terminal_spine_count_before_final_barline(_close_boundary(part)) for part in parts
    ]
    keys = [_keys(_header(part)) for part in prepared]
    union: list[tuple[str, int]] = []
    for part_keys in keys:
        cursor = 0
        for key in part_keys:
            if key not in union:
                union.insert(cursor, key)
            position = union.index(key)
            if position < cursor:
                raise ValueError("Spine ordering cannot be reconciled by padding.")
            cursor = position + 1
    if not union:
        return (PIECE_BREAK + "\n") * (len(parts) - 1)
    result = ["\t".join(kind for kind, _ in union) + "\n"]
    for index, (part, part_keys) in enumerate(zip(prepared, keys, strict=True)):
        if index:
            _with_boundary_break(result)
            result.append(PIECE_BREAK + "\n")
        types = _header(part)
        for line, fields, lineage in _records(part, types):
            if fields is None:
                result.append(line if line.endswith("\n") else line + "\n")
                continue
            if all(field.startswith("**") for field in fields) or all(
                field == "*-" for field in fields
            ):
                continue
            default = (
                fields[0]
                if fields[0].startswith("=")
                else "*"
                if fields[0].startswith("*")
                else "!"
                if fields[0].startswith("!")
                else "."
            )
            grouped = {key: [] for key in union}
            for field, roots in zip(fields, lineage, strict=True):
                if len(roots) != 1:
                    raise ValueError("Joined independent spines cannot be padded safely.")
                grouped[part_keys[roots[0]]].append("*" if field == "*-" else field)
            result.append(
                "\t".join(field for key in union for field in (grouped[key] or [default])) + "\n"
            )
        # A split remaining at the boundary cannot continue into the next header.
        if part and index < len(parts) - 1:
            summary = summarize_spine_topology(part)
            if summary.terminal_spine_count is not None and summary.terminal_spine_count > len(
                types
            ):
                raise ValueError("Split spines need a final barline before the piece break.")
    result.append("\t".join("*-" for _ in union) + "\n")
    return "".join(result)


def replace_part(text: str, index: int, part: str) -> str:
    parts = split_parts(text)
    if index < 0 or index >= len(parts):
        raise IndexError("Piece part index is outside the page.")
    if parts[index] == part:
        return text
    parts[index] = part
    joined = join_parts(parts)
    old_chunks, new_chunks = _chunks(text), _chunks(joined)
    # Preserve authored bytes when the replacement does not change boundary topology.
    if _header(text) == _header(joined):
        for other in range(len(parts)):
            if other != index:
                new_chunks[other] = old_chunks[other]
        return (PIECE_BREAK + "\n").join(new_chunks)
    return joined
