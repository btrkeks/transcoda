"""Shared confidence summary helpers for generated prediction records."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

SUSPECT_LOGPROB_THRESHOLD = math.log(0.5)


def numeric_token_logprobs(confidence: Mapping[str, Any]) -> list[float] | None:
    values = confidence.get("token_logprobs")
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        return None
    logprobs: list[float] = []
    for value in values:
        if not isinstance(value, int | float):
            return None
        logprobs.append(float(value))
    return logprobs


def suspect_token_count(
    confidence: Mapping[str, Any],
    *,
    threshold: float = SUSPECT_LOGPROB_THRESHOLD,
) -> int | None:
    logprobs = numeric_token_logprobs(confidence)
    if logprobs is None:
        return None
    return sum(1 for value in logprobs if value < threshold)


def priority_score(confidence: Mapping[str, Any]) -> float | None:
    count = suspect_token_count(confidence)
    if count is None:
        return None

    num_scored = confidence.get("num_scored_tokens")
    if isinstance(num_scored, int | float):
        token_count = max(0.0, float(num_scored))
    else:
        logprobs = numeric_token_logprobs(confidence)
        token_count = float(len(logprobs or []))
    return count / (1.0 + token_count / 500.0)


def confidence_summary_fields(confidence: Mapping[str, Any]) -> dict[str, int | float]:
    count = suspect_token_count(confidence)
    score = priority_score(confidence)
    fields: dict[str, int | float] = {}
    if count is not None:
        fields["suspect_token_count"] = count
    if score is not None:
        fields["priority_score"] = score
    return fields
