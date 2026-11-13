"""Source registry parsing for synthetic dataset materialization.

The registry lists synthetic sources only. Real pages are built from the corpus
as their own releases (``python -m pipelines.datasets.releases build-real``), so
a real source kind in a registry is a configuration error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SourceKind = Literal["synthetic_svg"]
REMOVED_REAL_SOURCE_KINDS = frozenset({"real_scan_directory", "real_corpus"})


@dataclass(frozen=True)
class SourceConfig:
    name: str
    split: str
    kind: SourceKind
    source_dataset: str
    render_settings_jsonl: Path | None = None
    layout_targets_report_jsonl: Path | None = None
    target_dir: Path | None = None


@dataclass(frozen=True)
class FinalDatasetRegistry:
    sources: tuple[SourceConfig, ...]


def load_registry(path: str | Path) -> FinalDatasetRegistry:
    registry_path = Path(path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Registry must be a JSON object: {registry_path}")

    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("Registry must include a non-empty 'sources' list")

    sources = tuple(_parse_source(item, registry_path=registry_path) for item in raw_sources)
    names = [source.name for source in sources]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate source name(s) in registry: {duplicates}")
    return FinalDatasetRegistry(sources=sources)


def _parse_source(payload: object, *, registry_path: Path) -> SourceConfig:
    if not isinstance(payload, dict):
        raise ValueError("Each registry source must be an object")

    name = _required_str(payload, "name")
    split = str(payload.get("split") or "train")
    kind = _required_str(payload, "kind")
    source_dataset = str(payload.get("source_dataset") or name)

    if kind == "synthetic_svg":
        has_render_settings = _has_nonempty_str(payload, "render_settings_jsonl")
        has_layout_report = _has_nonempty_str(payload, "layout_targets_report_jsonl")
        if has_render_settings == has_layout_report:
            raise ValueError(
                "Synthetic source must set exactly one of "
                "render_settings_jsonl or layout_targets_report_jsonl"
            )

        target_dir_value = payload.get("target_dir")
        if has_layout_report:
            if isinstance(target_dir_value, str) and target_dir_value.strip():
                raise ValueError(
                    "Synthetic source using layout_targets_report_jsonl must not set target_dir"
                )
            layout_targets_report_jsonl = _resolve_path(
                _required_str(payload, "layout_targets_report_jsonl"),
                registry_path=registry_path,
            )
            return SourceConfig(
                name=name,
                split=split,
                kind="synthetic_svg",
                source_dataset=source_dataset,
                layout_targets_report_jsonl=layout_targets_report_jsonl,
            )

        if not isinstance(target_dir_value, str) or not target_dir_value.strip():
            raise ValueError("Synthetic source using render_settings_jsonl must set target_dir")
        render_settings_jsonl = _resolve_path(
            _required_str(payload, "render_settings_jsonl"),
            registry_path=registry_path,
        )
        target_dir = _resolve_path(target_dir_value, registry_path=registry_path)
        return SourceConfig(
            name=name,
            split=split,
            kind="synthetic_svg",
            source_dataset=source_dataset,
            render_settings_jsonl=render_settings_jsonl,
            target_dir=target_dir,
        )

    if kind in REMOVED_REAL_SOURCE_KINDS:
        raise ValueError(
            f"registry source {name!r} has kind {kind!r}: real sources were removed "
            "from dataset registries; build real pages with "
            "`python -m pipelines.datasets.releases build-real`"
        )

    raise ValueError(f"Unsupported source kind for {name!r}: {kind!r}")


def _required_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Registry source missing required string field: {key}")
    return value


def _has_nonempty_str(payload: dict[str, object], key: str) -> bool:
    value = payload.get(key)
    return isinstance(value, str) and bool(value.strip())


def _resolve_path(value: str, *, registry_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    base = (
        registry_path.parent.parent
        if registry_path.parent.name == "config"
        else registry_path.parent
    )
    return (base / path).resolve()
