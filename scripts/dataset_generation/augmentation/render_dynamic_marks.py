"""Render-only terraced dynamic marks using a trailing ``**dynam`` spine."""

from __future__ import annotations

import random

from .dynam_spine import (
    default_dynam_token,
    find_trailing_dynam_spine_index,
    infer_trailing_spine_index,
    is_eligible_data_line,
    line_has_writable_dynam_null,
)

__all__ = ["apply_render_dynamic_marks"]

_DYNAMIC_MARK_TOKENS: tuple[str, ...] = ("pp", "p", "mp", "mf", "f", "ff", "sf", "z")


def apply_render_dynamic_marks(
    krn: str,
    sample_probability: float = 0.15,
    min_marks: int = 1,
    max_marks: int = 2,
    assume_trailing_dynam: bool = False,
) -> str:
    """Inject visual-only terraced dynamic markings in a ``**dynam`` spine.

    The function either appends a trailing dynamics spine (when none exists)
    or writes into an existing trailing ``**dynam`` column. Existing non-null
    dynamics tokens are never overwritten.
    """
    if not krn:
        return krn
    if not 0.0 <= sample_probability <= 1.0:
        raise ValueError(
            "sample_probability must be in [0.0, 1.0], "
            f"got {sample_probability}"
        )
    if min_marks < 1:
        raise ValueError(f"min_marks must be >= 1, got {min_marks}")
    if max_marks < min_marks:
        raise ValueError(
            f"max_marks must be >= min_marks, got min_marks={min_marks}, max_marks={max_marks}"
        )
    if random.random() >= sample_probability:
        return krn

    lines = krn.splitlines()
    candidate_lines = [i for i, line in enumerate(lines) if is_eligible_data_line(line)]
    if not candidate_lines:
        return krn

    trailing_dynam_idx = find_trailing_dynam_spine_index(lines)
    if trailing_dynam_idx is None and assume_trailing_dynam:
        trailing_dynam_idx = infer_trailing_spine_index(lines)

    if trailing_dynam_idx is None:
        available_lines = candidate_lines
    else:
        available_lines = [
            i
            for i in candidate_lines
            if line_has_writable_dynam_null(lines[i], trailing_dynam_idx)
        ]
    if not available_lines:
        return krn

    target_mark_count = min(random.randint(min_marks, max_marks), len(available_lines))
    if target_mark_count < 1:
        return krn
    selected_lines = random.sample(available_lines, k=target_mark_count)

    if trailing_dynam_idx is None:
        dynam_tokens = [default_dynam_token(line) for line in lines]
        for idx in selected_lines:
            dynam_tokens[idx] = random.choice(_DYNAMIC_MARK_TOKENS)

        out_lines: list[str] = []
        for line, dynam_token in zip(lines, dynam_tokens, strict=True):
            if dynam_token is None:
                out_lines.append(line)
                continue
            if line:
                out_lines.append(f"{line}\t{dynam_token}")
            else:
                out_lines.append(dynam_token)
        return "\n".join(out_lines)

    out_lines: list[str] = list(lines)
    for idx in selected_lines:
        fields = out_lines[idx].split("\t")
        if len(fields) <= trailing_dynam_idx:
            continue
        if fields[trailing_dynam_idx].strip() != ".":
            continue
        fields[trailing_dynam_idx] = random.choice(_DYNAMIC_MARK_TOKENS)
        out_lines[idx] = "\t".join(fields)

    return "\n".join(out_lines)
