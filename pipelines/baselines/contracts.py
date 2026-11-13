"""Versioned, runtime-neutral contracts for external baseline evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import PurePosixPath
from typing import Any, Literal, TypeVar

from pipelines.baselines.sample_id import validate_sample_id

SCHEMA_VERSION = 1
EXPECTED_IMAGE_WIDTH = 1050
EXPECTED_IMAGE_HEIGHT = 1485
BASELINE_MODEL_IDS = {
    "legato": "guangyangmusic/legato",
    "smt": "PRAIG/smt-fp-grandstaff",
}
BASELINE_MODEL_SLUGS = {
    "legato": "guangyangmusic__legato",
    "smt": "praig__smt-fp-grandstaff",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
_RUN_ID_RE = re.compile(r"^[0-9a-f]{20}$")


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def validate_sha256(value: str, *, field_name: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def validate_revision(value: str, *, field_name: str) -> None:
    if _REVISION_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be an immutable lowercase revision hash")


def validate_relative_posix_path(value: str, *, field_name: str) -> None:
    if not value or "\\" in value:
        raise ValueError(f"{field_name} must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field_name} must be a normalized relative POSIX path")
    if path.as_posix() != value:
        raise ValueError(f"{field_name} must be a normalized relative POSIX path")


@dataclass(frozen=True)
class InputManifestRecord:
    schema_version: int
    sample_id: str
    image_path: str
    image_sha256: str
    width: int
    height: int

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        validate_sample_id(self.sample_id)
        validate_relative_posix_path(self.image_path, field_name="image_path")
        validate_sha256(self.image_sha256, field_name="image_sha256")
        if (self.width, self.height) != (EXPECTED_IMAGE_WIDTH, EXPECTED_IMAGE_HEIGHT):
            raise ValueError(
                f"baseline images must be exactly {EXPECTED_IMAGE_WIDTH}x{EXPECTED_IMAGE_HEIGHT}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> InputManifestRecord:
        return _strict_dataclass_from_dict(cls, payload)


@dataclass(frozen=True)
class GroundTruthManifestRecord:
    schema_version: int
    sample_id: str
    kern_path: str
    kern_sha256: str

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        validate_sample_id(self.sample_id)
        validate_relative_posix_path(self.kern_path, field_name="kern_path")
        validate_sha256(self.kern_sha256, field_name="kern_sha256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GroundTruthManifestRecord:
        return _strict_dataclass_from_dict(cls, payload)


@dataclass(frozen=True)
class RunSpec:
    model_id: str
    model_revision: str
    processor_revision: str | None
    manifest_digest: str
    generation_config: dict[str, Any]
    dtype: str
    clone_commit: str
    attempt_id: str
    parent_run_id: str | None
    adapter_version: str
    predictor_source_digest: str
    environment_digest: str
    component_revisions: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        validate_revision(self.model_revision, field_name="model_revision")
        if self.processor_revision is not None:
            validate_revision(self.processor_revision, field_name="processor_revision")
        validate_sha256(self.manifest_digest, field_name="manifest_digest")
        validate_revision(self.clone_commit, field_name="clone_commit")
        validate_sample_id(self.attempt_id)
        if self.parent_run_id is not None and _RUN_ID_RE.fullmatch(self.parent_run_id) is None:
            raise ValueError("parent_run_id must be a 20-character lowercase run id")
        if self.attempt_id != "initial" and self.parent_run_id is None:
            raise ValueError("non-initial attempt_id requires parent_run_id")
        if not self.adapter_version or not self.dtype:
            raise ValueError("adapter_version and dtype must be non-empty")
        validate_sha256(self.predictor_source_digest, field_name="predictor_source_digest")
        validate_sha256(self.environment_digest, field_name="environment_digest")
        if not isinstance(self.generation_config, dict):
            raise ValueError("generation_config must be a JSON object")
        canonical_json_bytes(self.generation_config)
        if not isinstance(self.component_revisions, dict):
            raise ValueError("component_revisions must be a JSON object")
        for component, revision in self.component_revisions.items():
            validate_sample_id(component)
            validate_revision(revision, field_name=f"component_revisions.{component}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunSpec:
        return _strict_dataclass_from_dict(cls, payload)

    @property
    def run_id(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()[:20]


TerminationReason = Literal["eos", "max_length", "constraint_dead_end", "error"]


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    image_sha256: str
    manifest_digest: str
    model_id: str
    model_revision: str
    native_format: Literal["bekern", "transcoda-kern", "abc"]
    output_text: str
    output_tokens: list[str | int]
    termination_reason: TerminationReason
    ended_with_eos: bool
    timing_s: float
    error: str | None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        validate_sample_id(self.sample_id)
        validate_sha256(self.image_sha256, field_name="image_sha256")
        validate_sha256(self.manifest_digest, field_name="manifest_digest")
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        validate_revision(self.model_revision, field_name="model_revision")
        if self.native_format not in {"bekern", "transcoda-kern", "abc"}:
            raise ValueError("native_format must be bekern, transcoda-kern, or abc")
        if self.termination_reason not in {
            "eos",
            "max_length",
            "constraint_dead_end",
            "error",
        }:
            raise ValueError("invalid termination_reason")
        if self.ended_with_eos != (self.termination_reason == "eos"):
            raise ValueError("ended_with_eos must agree with termination_reason")
        if (self.error is not None) != (self.termination_reason == "error"):
            raise ValueError("error must be present exactly for termination_reason=error")
        if not math.isfinite(self.timing_s) or self.timing_s < 0:
            raise ValueError("timing_s must be a finite non-negative number")
        if not isinstance(self.output_tokens, list) or not all(
            isinstance(token, (str, int)) and not isinstance(token, bool)
            for token in self.output_tokens
        ):
            raise ValueError("output_tokens must contain only model-native strings or integers")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PredictionRecord:
        return _strict_dataclass_from_dict(cls, payload)


def input_manifest_digest(records: list[InputManifestRecord]) -> str:
    return _records_digest(records)


def ground_truth_manifest_digest(records: list[GroundTruthManifestRecord]) -> str:
    return _records_digest(records)


def _records_digest(records: list[InputManifestRecord] | list[GroundTruthManifestRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.sample_id):
        digest.update(canonical_json_bytes(record.to_dict()))
    return digest.hexdigest()


def _validate_schema_version(value: int) -> None:
    if value != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")


T = TypeVar("T")


def _strict_dataclass_from_dict(cls: type[T], payload: dict[str, Any]) -> T:
    if not isinstance(payload, dict):
        raise ValueError(f"{cls.__name__} must be a JSON object")
    expected = {field.name for field in fields(cls)}
    actual = set(payload)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise ValueError(f"{cls.__name__} has unknown fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{cls.__name__} is missing fields: {', '.join(missing)}")
    try:
        return cls(**payload)
    except TypeError as exc:
        raise ValueError(f"invalid {cls.__name__}: {exc}") from exc
