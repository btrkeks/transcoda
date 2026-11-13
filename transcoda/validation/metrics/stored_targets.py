"""Stored labels as OMR-NED targets (``validation.metrics.verosim.target_text: stored``).

Validation records keep the target the tokenizer decodes from the label ids,
which loses any text the vocabulary cannot spell (``*cue`` becomes
``*c<unk>e``). A suite that scores the stored label instead swaps each record's
target for the dataset's ``transcription`` before kern_parser and VeroSim see
it. The swap is keyed by (validation set, sample id) and checked against the
record's ``sample_revision_digest``, so it can only put back the label the
prediction was made for.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from transcoda.metrics_schema import VAL_SUBSET_SUFFIX

TARGET_TEXT_ROUND_TRIP = "round_trip"
TARGET_TEXT_STORED = "stored"
TARGET_TEXTS = frozenset({TARGET_TEXT_ROUND_TRIP, TARGET_TEXT_STORED})


class StoredTargetError(ValueError):
    """A record's stored label is missing or belongs to another revision."""


class StoredTargets:
    """The ``transcription`` column of each validation dataset, by sample id."""

    def __init__(self, validation_paths: Mapping[str, str]) -> None:
        self._paths = dict(validation_paths)
        self._labels: dict[str, dict[str, tuple[str, str | None]]] = {}

    def target(self, record: Mapping[str, Any]) -> str:
        val_set = str(record["val_set"])
        name = val_set.removesuffix(VAL_SUBSET_SUFFIX) if val_set not in self._paths else val_set
        labels = self._load(name)
        sample_id = str(record["sample_id"])
        if sample_id not in labels:
            raise StoredTargetError(f"no stored label for {val_set}/{sample_id}")
        text, revision = labels[sample_id]
        recorded = record.get("sample_revision_digest")
        if recorded is None or revision is None:
            raise StoredTargetError(
                f"{val_set}/{sample_id} cannot be matched to its stored label: the "
                f"{'prediction record' if recorded is None else 'dataset'} has no "
                "sample_revision_digest"
            )
        if recorded != revision:
            raise StoredTargetError(
                f"{val_set}/{sample_id} was predicted for revision {recorded}, "
                f"but the stored label is revision {revision}"
            )
        return text

    def _load(self, name: str) -> dict[str, tuple[str, str | None]]:
        if name in self._labels:
            return self._labels[name]
        if name not in self._paths:
            raise StoredTargetError(f"validation set {name!r} has no dataset path")
        from datasets import load_from_disk

        dataset = load_from_disk(self._paths[name])
        columns = ["sample_id", "transcription", "sample_revision_digest"]
        missing = [column for column in columns if column not in dataset.column_names]
        if missing:
            raise StoredTargetError(f"validation set {name!r} has no {missing} column")
        rows = dataset.select_columns(columns)
        labels = {
            str(row["sample_id"]): (str(row["transcription"]), row["sample_revision_digest"])
            for row in rows
        }
        self._labels[name] = labels
        return labels


def target_lookup_for(
    target_text: str | None, validation_paths: Mapping[str, str]
) -> StoredTargets | None:
    """The OMR-NED target lookup a ``target_text`` policy needs (None: the round trip)."""
    if target_text in (None, TARGET_TEXT_ROUND_TRIP):
        return None
    if target_text != TARGET_TEXT_STORED:
        raise ValueError(f"target_text must be one of {sorted(TARGET_TEXTS)}: {target_text!r}")
    return StoredTargets(validation_paths)
