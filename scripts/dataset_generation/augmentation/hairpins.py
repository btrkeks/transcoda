"""Render-only hairpin augmentation using a temporary trailing ``**dynam`` spine."""

from __future__ import annotations

import random

from .dynam_spine import default_dynam_token, is_eligible_data_line

__all__ = ["apply_render_hairpins"]

_MAX_SPAN_STEPS = 8


def apply_render_hairpins(
    krn: str,
    sample_probability: float = 0.25,
    max_spans: int = 2,
) -> str:
    """Inject wedge hairpins in a temporary trailing ``**dynam`` spine.

    The function is intended for render-only augmentation: it appends a dynamics
    column and places hairpins there while leaving the original **kern content
    untouched.
    """
    if not krn:
        return krn
    if not 0.0 <= sample_probability <= 1.0:
        raise ValueError(
            "sample_probability must be in [0.0, 1.0], "
            f"got {sample_probability}"
        )
    if max_spans < 1:
        raise ValueError(f"max_spans must be >= 1, got {max_spans}")
    if random.random() >= sample_probability:
        return krn

    lines = krn.splitlines()
    candidate_lines = [i for i, line in enumerate(lines) if is_eligible_data_line(line)]
    if len(candidate_lines) < 2:
        return krn

    spans = _sample_non_overlapping_spans(num_steps=len(candidate_lines), max_spans=max_spans)
    if not spans:
        return krn

    dynam_tokens = [default_dynam_token(line) for line in lines]
    for start_step, end_step in spans:
        is_crescendo = random.random() < 0.5
        start_token, continuation_token, end_token = (
            ("<", "(", "[") if is_crescendo else (">", ")", "]")
        )

        dynam_tokens[candidate_lines[start_step]] = start_token
        for step_idx in range(start_step + 1, end_step):
            dynam_tokens[candidate_lines[step_idx]] = continuation_token
        dynam_tokens[candidate_lines[end_step]] = end_token

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
def _sample_non_overlapping_spans(
    *,
    num_steps: int,
    max_spans: int,
) -> list[tuple[int, int]]:
    max_possible = min(max_spans, num_steps // 2)
    if max_possible < 1:
        return []

    target_span_count = random.randint(1, max_possible)
    available_ranges: list[tuple[int, int]] = [(0, num_steps - 1)]
    spans: list[tuple[int, int]] = []

    for _ in range(target_span_count):
        candidate_ranges = [r for r in available_ranges if (r[1] - r[0] + 1) >= 2]
        if not candidate_ranges:
            break

        range_start, range_end = random.choice(candidate_ranges)
        range_len = range_end - range_start + 1
        span_len = random.randint(2, min(_MAX_SPAN_STEPS, range_len))
        span_start = random.randint(range_start, range_end - span_len + 1)
        span_end = span_start + span_len - 1
        spans.append((span_start, span_end))

        updated_ranges: list[tuple[int, int]] = []
        for current_start, current_end in available_ranges:
            if current_end < span_start or current_start > span_end:
                updated_ranges.append((current_start, current_end))
                continue
            if current_start <= span_start - 1:
                updated_ranges.append((current_start, span_start - 1))
            if span_end + 1 <= current_end:
                updated_ranges.append((span_end + 1, current_end))
        available_ranges = updated_ranges

    spans.sort()
    return spans
