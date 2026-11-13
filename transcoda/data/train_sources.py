"""Several train releases read as one training set (``data.train_sources``).

Each release is loaded and tokenized on its own, so a release keeps its own
``Dataset.map`` cache and no two releases need the same row schema, only the
same fields in a loaded sample. Every sample carries the name of its release
as ``train_source``, and the sampler is the capped mixture sampler with one
source per release, selected on that column.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from transcoda.config import TrainSource
from transcoda.core.dataset_release import RELEASE_FILENAME, parse_release_file
from transcoda.data.mixture import (
    LoadedMixture,
    MixtureSource,
    MixtureSpec,
    RowSelector,
    mixture_digest,
)

TRAIN_SOURCE_COLUMN = "train_source"
TRAIN_SOURCES_MIXTURE_ID = "data.train_sources"
DEFAULT_RELEASES_ROOT = Path("data/releases")


def source_dataset_path(source: TrainSource, releases_root: Path = DEFAULT_RELEASES_ROOT) -> str:
    """The HF snapshot a source reads: its ``path``, else its release's ``dataset_path``."""
    if source.path is not None:
        return source.path
    release_json = releases_root / source.release / RELEASE_FILENAME
    try:
        manifest, _digest = parse_release_file(release_json.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"train source {source.release!r} has no path and {release_json} is unreadable: {exc}"
        ) from exc
    if manifest.role != "train":
        raise ValueError(f"train source {source.release!r} is a {manifest.role} release")
    return manifest.dataset_path


def train_sources_mixture(sources: Sequence[TrainSource], *, epoch_size: int) -> LoadedMixture:
    """The mixture that draws each release by its weight, under its repeat cap."""
    spec = MixtureSpec(
        mixture_id=TRAIN_SOURCES_MIXTURE_ID,
        epoch_size=epoch_size,
        sources=tuple(
            MixtureSource(
                name=source.release,
                selector=RowSelector(match={TRAIN_SOURCE_COLUMN: source.release}),
                weight=source.weight,
                max_repeats_per_epoch=source.max_repeats_per_epoch,
            )
            for source in sources
        ),
    )
    return LoadedMixture(
        path=Path(TRAIN_SOURCES_MIXTURE_ID), spec=spec, digest=mixture_digest(spec)
    )


class MultiSourceDataset(Dataset):
    """Train releases end to end; a global index maps to (release, row in release)."""

    def __init__(self, parts: Sequence[tuple[str, Dataset]]) -> None:
        if not parts:
            raise ValueError("a multi-source dataset needs at least one source")
        names = [name for name, _dataset in parts]
        if len(names) != len(set(names)):
            raise ValueError(f"train source names must be unique: {names}")
        self.parts = tuple(parts)
        self._starts: list[int] = []
        total = 0
        for _name, dataset in self.parts:
            self._starts.append(total)
            total += len(dataset)
        self._length = total

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        part = bisect.bisect_right(self._starts, index) - 1
        name, dataset = self.parts[part]
        sample = dataset[index - self._starts[part]]
        sample[TRAIN_SOURCE_COLUMN] = name
        return sample

    def source_lengths(self) -> dict[str, int]:
        return {name: len(dataset) for name, dataset in self.parts}

    def column(self, column: str) -> list | None:
        """One metadata column over all sources, or None when a source lacks it."""
        if column == TRAIN_SOURCE_COLUMN:
            return [name for name, dataset in self.parts for _ in range(len(dataset))]
        values: list = []
        for _name, dataset in self.parts:
            part = dataset_column(dataset, column)
            if part is None:
                return None
            values.extend(part)
        return values


def dataset_column(dataset: Any, column: str) -> list | None:
    """A metadata column of a loaded train dataset without decoding images."""
    if isinstance(dataset, MultiSourceDataset):
        return dataset.column(column)
    inner = getattr(dataset, "dataset", dataset)
    column_names = getattr(inner, "column_names", None)
    if column_names is not None and column in column_names:
        metadata = inner.with_format(None) if hasattr(inner, "with_format") else inner
        return list(metadata[column])
    if isinstance(inner, list):
        if not inner or not isinstance(inner[0], dict) or column not in inner[0]:
            return None
        return [sample.get(column) for sample in inner]
    try:
        values = inner[column]
    except (KeyError, TypeError, AttributeError, IndexError):
        return None
    return list(values)


def check_sample_fields(parts: Sequence[tuple[str, Any]]) -> None:
    """Refuse sources whose samples carry different fields.

    The collator takes the batch's fields from its first sample and reads them
    from every other one, so mixed batches need the same fields from every
    source. The fields are those of a loaded sample, after the image/label
    transform (which turns ``source`` and ``source_ids`` alike into
    ``source``), not the raw dataset columns.
    """
    seen = {name: frozenset(dataset[0]) for name, dataset in parts if len(dataset)}
    if len(set(seen.values())) > 1:
        detail = "; ".join(f"{name}: {sorted(fields)}" for name, fields in seen.items())
        raise ValueError(f"train sources disagree on sample fields: {detail}")
