"""Strict identities shared by dataset transforms and validation records."""

from __future__ import annotations

import re
from typing import Any

SampleId = str | int

_REVISION_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def validate_sample_id(value: Any) -> SampleId:
    if isinstance(value, bool):
        raise ValueError("sample_id must be a non-empty string or integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value:
        return value
    raise ValueError("sample_id must be a non-empty string or integer")


def sample_id_as_string(value: Any) -> str:
    """Return the canonical live dataset identity.

    Persisted validation records still accept integers for backward compatibility
    and for index-only loaders, but named dataset/eval-set members are strings.
    """
    return str(validate_sample_id(value))


def validate_sample_revision_digest(
    value: Any,
    *,
    allow_none: bool = True,
) -> str | None:
    if value is None and allow_none:
        return None
    if isinstance(value, str) and _REVISION_DIGEST.fullmatch(value):
        return value
    raise ValueError(
        "sample_revision_digest must be a lowercase sha256 digest"
    )
