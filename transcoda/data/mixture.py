"""Versioned training-mixture compiler and deterministic capped sampler."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Annotated, Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
SelectorValue = str | bool | int

MIXTURE_SELECTOR_FIELDS = frozenset(
    {
        "source_dataset",
        "source_split",
        "source_domain",
        "curation_stage",
        "label_tier",
        "has_tuplets",
        "has_ottava",
        "has_chords",
        "has_degradation",
        # Set by the datamodule from data.train_sources: the row's release.
        "train_source",
    }
)


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RowSelector(FrozenModel):
    match: dict[str, tuple[SelectorValue, ...]] = Field(default_factory=dict)
    not_match: dict[str, tuple[SelectorValue, ...]] = Field(default_factory=dict)

    @field_validator("match", "not_match", mode="before")
    @classmethod
    def _normalize_values(cls, value: object) -> object:
        if value is None:
            return {}
        if not isinstance(value, dict):
            return value
        invalid_keys = sorted(repr(key) for key in value if not isinstance(key, str))
        if invalid_keys:
            raise ValueError(
                f"mixture selector field names must be strings: {invalid_keys}"
            )
        return {key: raw if isinstance(raw, (list, tuple)) else [raw] for key, raw in value.items()}

    @model_validator(mode="after")
    def _validate_fields(self) -> RowSelector:
        fields = set(self.match) | set(self.not_match)
        unknown = sorted(fields - MIXTURE_SELECTOR_FIELDS)
        if unknown:
            raise ValueError(f"mixture selector uses unsupported fields: {unknown}")
        for field, values in (*self.match.items(), *self.not_match.items()):
            if not values:
                raise ValueError(f"mixture selector values must be non-empty: {field}")
            if len(values) != len({(type(value), value) for value in values}):
                raise ValueError(f"mixture selector values must be unique: {field}")
        return self

    @property
    def required_fields(self) -> frozenset[str]:
        return frozenset(self.match) | frozenset(self.not_match)

    def matches(self, row: Mapping[str, Any]) -> bool:
        return all(
            _contains_typed(values, row[field]) for field, values in self.match.items()
        ) and all(
            not _contains_typed(values, row[field]) for field, values in self.not_match.items()
        )


class MixtureSource(FrozenModel):
    name: str
    selector: RowSelector
    # Zero keeps a source's rows matched (and pinned) but never draws them.
    weight: float = Field(ge=0, le=1)
    max_repeats_per_epoch: int | None = Field(default=None, ge=1)

    @field_validator("name")
    @classmethod
    def _non_empty_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("mixture source names must be non-empty")
        return value

    @field_validator("weight", mode="before")
    @classmethod
    def _finite_weight(cls, value: Any) -> Any:
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            return value
        if not finite:
            raise ValueError("mixture source weights must be finite")
        return value


class OversampleRule(FrozenModel):
    name: str
    selector: RowSelector
    factor: float = Field(gt=1)

    @field_validator("name")
    @classmethod
    def _non_empty_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("mixture oversample names must be non-empty")
        return value

    @field_validator("factor")
    @classmethod
    def _finite_factor(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("mixture oversample factors must be finite")
        return value


class MixtureSpec(FrozenModel):
    schema_version: Literal[1] = 1
    mixture_id: str
    epoch_size: int | Literal["dataset"] = "dataset"
    sources: tuple[MixtureSource, ...]
    oversample: tuple[OversampleRule, ...] = ()

    @field_validator("mixture_id")
    @classmethod
    def _non_empty_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("mixture_id must be non-empty")
        return value

    @field_validator("epoch_size")
    @classmethod
    def _positive_epoch_size(cls, value: int | str) -> int | str:
        if isinstance(value, int) and value < 1:
            raise ValueError("mixture epoch_size must be positive")
        return value

    @model_validator(mode="after")
    def _validate_rules(self) -> MixtureSpec:
        if not self.sources:
            raise ValueError("mixtures require at least one source")
        source_names = [source.name for source in self.sources]
        oversample_names = [rule.name for rule in self.oversample]
        if len(source_names) != len(set(source_names)):
            raise ValueError("mixture source names must be unique")
        if len(oversample_names) != len(set(oversample_names)):
            raise ValueError("mixture oversample names must be unique")
        total = sum(Decimal(str(source.weight)) for source in self.sources)
        if total != Decimal("1"):
            raise ValueError(f"mixture source weights must sum to 1, got {total}")
        return self

    @property
    def required_fields(self) -> tuple[str, ...]:
        fields = {
            field
            for selector in (
                *(source.selector for source in self.sources),
                *(rule.selector for rule in self.oversample),
            )
            for field in selector.required_fields
        }
        return tuple(sorted(fields))


@dataclass(frozen=True)
class LoadedMixture:
    path: Path
    spec: MixtureSpec
    digest: str


@dataclass(frozen=True)
class ResolvedMixtureSource:
    name: str
    quota: int
    indices: tuple[int, ...]
    row_weights: tuple[float, ...]
    max_repeats_per_epoch: int | None


@dataclass(frozen=True)
class ResolvedMixture:
    mixture_id: str
    digest: str
    epoch_size: int
    sources: tuple[ResolvedMixtureSource, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "mixture_id": self.mixture_id,
            "digest": self.digest,
            "epoch_size": self.epoch_size,
            "sources": [
                {
                    "name": source.name,
                    "row_count": len(source.indices),
                    "quota": source.quota,
                    "max_repeats_per_epoch": source.max_repeats_per_epoch,
                }
                for source in self.sources
            ],
        }

    def epoch_summary(self, indices: Sequence[int]) -> dict[str, Any]:
        payload = self.summary()
        source_payloads = {source["name"]: source for source in payload["sources"]}
        for source in self.sources:
            source_indices = set(source.indices)
            counts = Counter(index for index in indices if index in source_indices)
            source_payloads[source.name].update(
                {
                    "observed_samples": sum(counts.values()),
                    "observed_unique_rows": len(counts),
                    "observed_max_repeats": max(counts.values(), default=0),
                }
            )
        return payload


def canonical_mixture_bytes(spec: MixtureSpec) -> bytes:
    payload = spec.model_dump(mode="json", exclude_none=True)
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def mixture_digest(spec: MixtureSpec) -> str:
    return "sha256:" + hashlib.sha256(canonical_mixture_bytes(spec)).hexdigest()


def write_mixture(path: str | Path, spec: MixtureSpec) -> None:
    Path(path).write_bytes(canonical_mixture_bytes(spec))


def load_mixture(path: str | Path) -> LoadedMixture:
    mixture_path = Path(path)
    raw = mixture_path.read_bytes()
    spec = MixtureSpec.model_validate_json(raw)
    canonical = canonical_mixture_bytes(spec)
    if raw != canonical:
        raise ValueError(
            f"mixture file is not canonical: {mixture_path}; rewrite with write_mixture"
        )
    return LoadedMixture(
        path=mixture_path,
        spec=spec,
        digest="sha256:" + hashlib.sha256(raw).hexdigest(),
    )


def compile_mixture_rows(
    loaded: LoadedMixture,
    rows: Sequence[Mapping[str, Any]],
) -> ResolvedMixture:
    fields = loaded.spec.required_fields
    columns = {field: [row.get(field, _MISSING) for row in rows] for field in fields}
    return compile_mixture_columns(loaded, columns=columns, row_count=len(rows))


def compile_mixture_columns(
    loaded: LoadedMixture,
    *,
    columns: Mapping[str, Sequence[Any]],
    row_count: int,
    epoch_size_override: int | None = None,
) -> ResolvedMixture:
    if row_count < 1:
        raise ValueError("cannot compile a mixture for an empty dataset")
    required = loaded.spec.required_fields
    for field in required:
        values = columns.get(field)
        if values is None:
            raise ValueError(f"training dataset is missing mixture selector field: {field}")
        if len(values) != row_count:
            raise ValueError(
                f"mixture selector column length mismatch for {field}: {len(values)} != {row_count}"
            )
        if any(
            value is _MISSING or value is None or type(value) not in {str, bool, int}
            for value in values
        ):
            raise ValueError(f"training dataset has missing values for selector field: {field}")

    source_indices = {source.name: [] for source in loaded.spec.sources}
    source_weights = {source.name: [] for source in loaded.spec.sources}
    for index in range(row_count):
        row = {field: columns[field][index] for field in required}
        matched = [source for source in loaded.spec.sources if source.selector.matches(row)]
        if not matched:
            raise ValueError(f"mixture source rules leave row {index} unmatched")
        if len(matched) > 1:
            raise ValueError(
                f"mixture source rules overlap for row {index}: "
                f"{[source.name for source in matched]}"
            )
        source = matched[0]
        weight = 1.0
        for rule in loaded.spec.oversample:
            if rule.selector.matches(row):
                weight *= rule.factor
                if not math.isfinite(weight):
                    raise ValueError(f"mixture oversample weight is non-finite for row {index}")
        source_indices[source.name].append(index)
        source_weights[source.name].append(weight)

    empty = [name for name, indices in source_indices.items() if not indices]
    if empty:
        raise ValueError(f"mixture source rules match no rows: {empty}")

    epoch_size = epoch_size_override
    if epoch_size is None:
        epoch_size = (
            row_count if loaded.spec.epoch_size == "dataset" else int(loaded.spec.epoch_size)
        )
    if epoch_size < 1:
        raise ValueError("resolved mixture epoch size must be positive")
    quotas = largest_remainder_quotas(loaded.spec.sources, epoch_size)
    resolved_sources: list[ResolvedMixtureSource] = []
    for source in sorted(loaded.spec.sources, key=lambda item: item.name):
        indices = tuple(source_indices[source.name])
        weights = tuple(source_weights[source.name])
        if not math.isfinite(sum(weights)):
            raise ValueError(f"mixture source {source.name!r} has non-finite total row weight")
        quota = quotas[source.name]
        if (
            source.max_repeats_per_epoch is not None
            and quota > len(indices) * source.max_repeats_per_epoch
        ):
            raise ValueError(
                f"mixture source {source.name!r} quota {quota} exceeds "
                f"{len(indices)} rows x cap {source.max_repeats_per_epoch}"
            )
        resolved_sources.append(
            ResolvedMixtureSource(
                name=source.name,
                quota=quota,
                indices=indices,
                row_weights=weights,
                max_repeats_per_epoch=source.max_repeats_per_epoch,
            )
        )
    return ResolvedMixture(
        mixture_id=loaded.spec.mixture_id,
        digest=loaded.digest,
        epoch_size=epoch_size,
        sources=tuple(resolved_sources),
    )


def _contains_typed(values: Sequence[SelectorValue], candidate: object) -> bool:
    return any(type(candidate) is type(value) and candidate == value for value in values)


def largest_remainder_quotas(
    sources: Sequence[MixtureSource],
    epoch_size: int,
) -> dict[str, int]:
    if epoch_size < 1:
        raise ValueError("mixture epoch size must be positive")
    exact = {source.name: Decimal(str(source.weight)) * epoch_size for source in sources}
    quotas = {
        name: int(value.to_integral_value(rounding=ROUND_FLOOR)) for name, value in exact.items()
    }
    remaining = epoch_size - sum(quotas.values())
    order = sorted(
        sources,
        key=lambda source: (-(exact[source.name] - quotas[source.name]), source.name),
    )
    for source in order[:remaining]:
        quotas[source.name] += 1
    return quotas


class CappedMixtureSampler(torch.utils.data.Sampler[int]):
    def __init__(self, resolved: ResolvedMixture, *, seed: int) -> None:
        self.resolved = resolved
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.resolved.epoch_size

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("mixture sampler epoch must be a non-negative integer")
        self.epoch = epoch

    def state_dict(self) -> dict[str, int | str]:
        return {
            "epoch": self.epoch,
            "mixture_digest": self.resolved.digest,
            "seed": self.seed,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        epoch = state_dict.get("epoch")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("mixture sampler state requires a non-negative integer epoch")
        mixture_digest = state_dict.get("mixture_digest")
        if mixture_digest != self.resolved.digest:
            raise ValueError(
                "mixture sampler checkpoint digest does not match the resolved mixture"
            )
        seed = state_dict.get("seed")
        if type(seed) is not int or seed != self.seed:
            raise ValueError("mixture sampler checkpoint seed does not match training seed")
        self.epoch = epoch

    def __iter__(self):
        # Lightning calls set_epoch() once at the logical epoch boundary.
        # Iterator recreation inside an epoch must replay the same sequence.
        return iter(self.epoch_indices())

    def epoch_indices(self) -> tuple[int, ...]:
        sampled: list[int] = []
        for source in self.resolved.sources:
            rng = random.Random(
                _derived_seed(
                    self.resolved.digest,
                    self.seed,
                    self.epoch,
                    source.name,
                )
            )
            if source.max_repeats_per_epoch is None:
                sampled.extend(
                    rng.choices(
                        source.indices,
                        weights=source.row_weights,
                        k=source.quota,
                    )
                )
            else:
                sampled.extend(_sample_capped(source, rng))
        shuffle = random.Random(
            _derived_seed(self.resolved.digest, self.seed, self.epoch, "shuffle")
        )
        shuffle.shuffle(sampled)
        return tuple(sampled)


def _sample_capped(
    source: ResolvedMixtureSource,
    rng: random.Random,
) -> list[int]:
    assert source.max_repeats_per_epoch is not None
    slots: list[tuple[float, int, int]] = []
    for index, weight in zip(source.indices, source.row_weights, strict=True):
        for slot in range(source.max_repeats_per_epoch):
            draw = max(rng.random(), sys.float_info.min)
            priority = -math.log(draw) / weight
            slots.append((priority, index, slot))
    slots.sort()
    selected = [index for _priority, index, _slot in slots[: source.quota]]
    counts = Counter(selected)
    if counts and max(counts.values()) > source.max_repeats_per_epoch:
        raise AssertionError("capped mixture sampler exceeded a per-row cap")
    return selected


def _derived_seed(digest: str, seed: int, epoch: int, scope: str) -> int:
    payload = f"{digest}\n{seed}\n{epoch}\n{scope}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"metadata row must be an object: {path}:{line_number}")
        rows.append(payload)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m transcoda.data.mixture")
    parser.add_argument("resolve", choices=("resolve",))
    parser.add_argument("--mixture", type=Path, required=True)
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        loaded = load_mixture(args.mixture)
        resolved = compile_mixture_rows(loaded, _read_jsonl_rows(args.rows_jsonl))
        sampler = CappedMixtureSampler(resolved, seed=args.seed)
        sampler.set_epoch(args.epoch)
        indices = sampler.epoch_indices()
        payload = resolved.epoch_summary(indices)
        payload["epoch"] = args.epoch
        print(json.dumps(payload, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps({"error": type(exc).__name__, "detail": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2


_MISSING = object()


if __name__ == "__main__":
    raise SystemExit(main())
