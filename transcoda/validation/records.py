"""Shared validation prediction record model and JSONL helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transcoda.core.sample_identity import (
    SampleId,
    validate_sample_id,
    validate_sample_revision_digest,
)
from transcoda.core.text_processing import token_ids_to_string
from transcoda.grammar.semantic_sequence_finalizer import finalize_generated_kern_sequence

PredictionSampleId = SampleId
PredictionDedupeKey = tuple[str, PredictionSampleId, str | None]


def _prediction_sample_id(value: Any) -> PredictionSampleId:
    return validate_sample_id(value)


@dataclass(frozen=True)
class DecodedSample:
    """One decoded validation sample in the persisted prediction JSONL schema."""

    val_set: str
    sample_id: PredictionSampleId
    source: Any
    cer: float
    prediction: str
    prediction_hit_max_length: bool
    prediction_truncated: bool
    target: str
    decode_policy_name: str | None = None
    generation_max_length: int | None = None
    sample_revision_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", _prediction_sample_id(self.sample_id))
        object.__setattr__(
            self,
            "sample_revision_digest",
            validate_sample_revision_digest(self.sample_revision_digest),
        )

    @property
    def dedupe_key(self) -> PredictionDedupeKey:
        return (self.val_set, self.sample_id, self.sample_revision_digest)

    def to_json_record(self) -> dict[str, Any]:
        """Return the stable JSON object consumed by downstream validation tools."""
        record = {
            "val_set": self.val_set,
            "sample_id": self.sample_id,
            "source": self.source,
            "cer": self.cer,
            "prediction": self.prediction,
            "prediction_hit_max_length": self.prediction_hit_max_length,
            "prediction_truncated": self.prediction_truncated,
            "target": self.target,
        }
        if self.decode_policy_name is not None:
            record["decode_policy_name"] = self.decode_policy_name
        if self.generation_max_length is not None:
            record["generation_max_length"] = self.generation_max_length
        if self.sample_revision_digest is not None:
            record["sample_revision_digest"] = self.sample_revision_digest
        return record

    def to_json_line(self) -> str:
        return json.dumps(self.to_json_record(), ensure_ascii=False)

    @classmethod
    def from_json_record(cls, record: Mapping[str, Any]) -> DecodedSample:
        return cls(
            val_set=str(record["val_set"]),
            sample_id=_prediction_sample_id(record["sample_id"]),
            source=record.get("source"),
            cer=float(record["cer"]),
            prediction=str(record["prediction"]),
            prediction_hit_max_length=bool(record.get("prediction_hit_max_length", False)),
            prediction_truncated=bool(record.get("prediction_truncated", False)),
            target=str(record["target"]),
            decode_policy_name=(
                None
                if record.get("decode_policy_name") is None
                else str(record["decode_policy_name"])
            ),
            generation_max_length=(
                None
                if record.get("generation_max_length") is None
                else int(record["generation_max_length"])
            ),
            sample_revision_digest=record.get("sample_revision_digest"),
        )

    @classmethod
    def from_json_line(cls, line: str) -> DecodedSample:
        return cls.from_json_record(json.loads(line))


@dataclass(frozen=True)
class DecodedBatch:
    """Decoded samples emitted for one validation batch."""

    samples: tuple[DecodedSample, ...]

    def __iter__(self):
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)


@dataclass(frozen=True)
class PredictionSet:
    """A collection of decoded validation samples."""

    samples: tuple[DecodedSample, ...]

    def __iter__(self):
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def records_by_set(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sample in self.samples:
            counts[sample.val_set] = counts.get(sample.val_set, 0) + 1
        return counts

    def deduped(self) -> PredictionSet:
        """Deduplicate distributed padding without merging distinct revisions."""
        seen: set[PredictionDedupeKey] = set()
        samples: list[DecodedSample] = []
        for sample in self.samples:
            if sample.dedupe_key in seen:
                continue
            seen.add(sample.dedupe_key)
            samples.append(sample)
        return PredictionSet(tuple(samples))

    def write_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as out:
            for sample in self.samples:
                out.write(sample.to_json_line() + "\n")

    @classmethod
    def from_samples(cls, samples: Iterable[DecodedSample]) -> PredictionSet:
        return cls(tuple(samples))

    @classmethod
    def read_jsonl(cls, path: str | Path) -> PredictionSet:
        samples: list[DecodedSample] = []
        with Path(path).open("r", encoding="utf-8") as in_file:
            for line in in_file:
                if line.strip():
                    samples.append(DecodedSample.from_json_line(line))
        return cls(tuple(samples))


def decoded_batch_from_validation_outputs(
    outputs: Mapping[str, Any],
    *,
    i2w: dict[int, str],
    pad_token_id: int,
    bos_token_id: int | None,
    eos_token_id: int | None,
) -> DecodedBatch:
    """Build shared decoded records from ``TranscodaTrainer`` validation outputs."""
    val_set = str(outputs["val_set_name"])
    sample_ids = _as_list(outputs["sample_ids"])
    dataset_sample_ids = outputs.get("dataset_sample_ids")
    persisted_sample_ids = (
        sample_ids if dataset_sample_ids is None else _as_list(dataset_sample_ids)
    )
    sample_revision_digests = outputs.get("sample_revision_digests")
    revision_digests = (
        [None] * len(sample_ids)
        if sample_revision_digests is None
        else _as_list(sample_revision_digests)
    )
    if len(persisted_sample_ids) != len(sample_ids):
        raise ValueError(
            "dataset_sample_ids length does not match sample_ids: "
            f"{len(persisted_sample_ids)} != {len(sample_ids)}"
        )
    if len(revision_digests) != len(sample_ids):
        raise ValueError(
            "sample_revision_digests length does not match sample_ids: "
            f"{len(revision_digests)} != {len(sample_ids)}"
        )
    sources = outputs.get("sources") or [None] * len(sample_ids)
    cers = outputs["cers"]
    pred_ids = outputs["pred_ids"]
    gt_ids = outputs["gt_ids"]
    decode_policy_name = outputs.get("decode_policy_name")
    if decode_policy_name is not None:
        decode_policy_name = str(decode_policy_name)
    generation_max_length = outputs.get("generation_max_length")
    if generation_max_length is not None:
        generation_max_length = int(generation_max_length)

    samples: list[DecodedSample] = []
    for index, sample_id in enumerate(persisted_sample_ids):
        finalized_prediction = finalize_generated_kern_sequence(
            token_ids=_token_ids_at(pred_ids, index),
            i2w=i2w,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            max_length=generation_max_length,
        )
        samples.append(
            DecodedSample(
                val_set=val_set,
                sample_id=_prediction_sample_id(sample_id),
                source=sources[index],
                cer=float(cers[index]),
                prediction=finalized_prediction.text,
                prediction_hit_max_length=finalized_prediction.hit_max_length,
                prediction_truncated=finalized_prediction.truncated,
                target=token_ids_to_string(
                    _token_ids_at(gt_ids, index),
                    i2w,
                    pad_token_id,
                    add_header=False,
                ),
                decode_policy_name=decode_policy_name,
                generation_max_length=generation_max_length,
                sample_revision_digest=revision_digests[index],
            )
        )

    return DecodedBatch(tuple(samples))


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return list(value)


def _token_ids_at(rows: Sequence[Any], index: int) -> list[int]:
    row = rows[index]
    if hasattr(row, "tolist"):
        row = row.tolist()
    return [int(token_id) for token_id in row]
