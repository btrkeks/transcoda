"""Convert a materialization manifest (JSONL rows plus image files) to a Hugging Face dataset.

Synthetic rows come from ``svg_raster_probe``; real rows come from
``real_rows.write_real_rows``. ``FINAL_FEATURES`` is the row schema of every
release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import datasets
from datasets import Dataset, Features, Value

from pipelines.datasets.materialization.identity import EffectiveLabelTier, content_digest
from transcoda.core.sample_identity import validate_sample_revision_digest

CAPABILITY_FIELDS = (
    "has_tuplets",
    "has_ottava",
    "has_chords",
    "has_degradation",
)

FINAL_FEATURES = Features(
    {
        "image": datasets.Image(mode="RGB"),
        "transcription": Value("string"),
        "sample_id": Value("string"),
        "source": Value("string"),
        "source_dataset": Value("string"),
        "source_split": Value("string"),
        "source_domain": Value("string"),
        "curation_stage": Value("string"),
        "svg_system_count": Value("int32"),
        "render_settings_path": Value("string"),
        "svg_path": Value("string"),
        "target_path": Value("string"),
        "augraphy_outcome": Value("string"),
        "document_id": Value("string"),
        "label_tier": Value("string"),
        "sample_revision_digest": Value("string"),
        **{name: Value("bool") for name in CAPABILITY_FIELDS},
    }
)
SOURCE_DOMAINS = frozenset({"synth"})


@dataclass(frozen=True)
class ConversionSummary:
    manifest_path: Path
    output_dir: Path
    row_count: int


def convert_rust_manifest_to_hf(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    image_root: str | Path | None = None,
    hf_cache_dir: str | Path | None = None,
    direct_single_shard: bool = False,
    fingerprint: str | None = None,
) -> ConversionSummary:
    """``fingerprint`` fixes the dataset's ``_fingerprint``; without it, datasets
    hashes the generator arguments, which include this manifest's path."""
    manifest = Path(manifest_path)
    output = Path(output_dir)
    resolved_image_root = Path(image_root) if image_root is not None else manifest.parent
    cache_parent = Path(hf_cache_dir) if hf_cache_dir is not None else output.parent
    cache_parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    _validate_manifest(manifest, resolved_image_root)

    progress_was_enabled = datasets.is_progress_bar_enabled()
    datasets.disable_progress_bar()
    try:
        with tempfile.TemporaryDirectory(prefix="rust_manifest_hf_", dir=cache_parent) as cache_dir:
            dataset = Dataset.from_generator(
                _iter_manifest_rows,
                features=FINAL_FEATURES,
                cache_dir=cache_dir,
                keep_in_memory=False,
                gen_kwargs={
                    "manifest_path": manifest,
                    "image_root": resolved_image_root,
                },
                fingerprint=fingerprint,
            )
            if len(dataset) == 0:
                raise ValueError(f"No rows converted from manifest: {manifest}")
            if direct_single_shard:
                _move_cached_dataset_to_disk(dataset, output)
            else:
                dataset.save_to_disk(str(output))
    finally:
        if progress_was_enabled:
            datasets.enable_progress_bar()

    return ConversionSummary(manifest_path=manifest, output_dir=output, row_count=len(dataset))


def _move_cached_dataset_to_disk(dataset: Dataset, output_dir: Path) -> None:
    """Save a generated dataset without rewriting its Arrow cache."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    cache_files = [Path(cache_file["filename"]) for cache_file in dataset.cache_files]
    if not cache_files:
        raise ValueError("Cannot direct-save dataset without Arrow cache files")

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_count = len(cache_files)
    data_files: list[dict[str, str]] = []
    for shard_idx, cache_file in enumerate(cache_files):
        shard_name = f"data-{shard_idx:05d}-of-{shard_count:05d}.arrow"
        shutil.move(str(cache_file), output_dir / shard_name)
        data_files.append({"filename": shard_name})

    state = {
        key: dataset.__dict__[key]
        for key in [
            "_fingerprint",
            "_format_columns",
            "_format_kwargs",
            "_format_type",
            "_output_all_columns",
        ]
    }
    state["_split"] = str(dataset.split) if dataset.split is not None else dataset.split
    state["_data_files"] = data_files
    (output_dir / "state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    dataset_info = asdict(dataset._info)
    sorted_dataset_info = {key: dataset_info[key] for key in sorted(dataset_info)}
    (output_dir / "dataset_info.json").write_text(
        json.dumps(sorted_dataset_info, indent=2),
        encoding="utf-8",
    )


def _validate_manifest(manifest_path: Path, image_root: Path) -> None:
    row_count = 0
    for line_no, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row_count += 1
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{manifest_path}:{line_no}: manifest row must be an object")
        image_value = row.get("image")
        if not isinstance(image_value, str) or not image_value:
            raise ValueError(f"{manifest_path}:{line_no}: missing image path")
        image_path = Path(image_value)
        if not image_path.is_absolute():
            image_path = image_root / image_path
        if not image_path.exists():
            raise ValueError(f"{manifest_path}:{line_no}: image file does not exist: {image_path}")
        _resolve_label_tier(
            row,
            source_domain=str(row.get("source_domain") or ""),
            manifest_path=manifest_path,
            line_no=line_no,
        )
        if row.get("source_domain") == "real":
            for key in ("document_id", "sample_revision_digest"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{manifest_path}:{line_no}: real rows require {key}")
        if row.get("sample_revision_digest") is not None:
            try:
                validate_sample_revision_digest(row["sample_revision_digest"], allow_none=False)
            except ValueError as exc:
                raise ValueError(f"{manifest_path}:{line_no}: {exc}") from exc
    if row_count == 0:
        raise ValueError(f"No rows converted from manifest: {manifest_path}")


def _iter_manifest_rows(manifest_path: Path, image_root: Path) -> Iterator[dict[str, Any]]:
    for line_no, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{manifest_path}:{line_no}: manifest row must be an object")
        image_value = row.get("image")
        if not isinstance(image_value, str) or not image_value:
            raise ValueError(f"{manifest_path}:{line_no}: missing image path")
        image_path = Path(image_value)
        if not image_path.is_absolute():
            image_path = image_root / image_path
        image_bytes = image_path.read_bytes()
        converted = dict(row)
        converted["image"] = {"path": None, "bytes": image_bytes}
        source_domain = str(converted.get("source_domain") or "")
        sample_id = str(converted.get("sample_id") or "")
        if converted.get("document_id") is None:
            converted["document_id"] = (
                f"synthetic:{converted.get('source_dataset', 'unknown')}:{sample_id}"
            )
        elif not isinstance(converted["document_id"], str) or not converted["document_id"].strip():
            raise ValueError(f"{manifest_path}:{line_no}: document_id must be a non-empty string")
        converted["label_tier"] = _resolve_label_tier(
            converted,
            source_domain=source_domain,
            manifest_path=manifest_path,
            line_no=line_no,
        )
        if converted.get("sample_revision_digest") is None:
            converted["sample_revision_digest"] = content_digest(
                {
                    "sample_id": sample_id,
                    "image_sha256": "sha256:" + hashlib.sha256(image_bytes).hexdigest(),
                    "transcription_sha256": hashlib.sha256(
                        str(converted.get("transcription") or "").encode("utf-8")
                    ).hexdigest(),
                }
            )
        else:
            try:
                converted["sample_revision_digest"] = validate_sample_revision_digest(
                    converted["sample_revision_digest"], allow_none=False
                )
            except ValueError as exc:
                raise ValueError(f"{manifest_path}:{line_no}: {exc}") from exc
        for name in CAPABILITY_FIELDS:
            if converted.get(name) is None:
                converted[name] = False
            elif type(converted[name]) is not bool:
                raise ValueError(f"{manifest_path}:{line_no}: {name} must be a boolean")
        yield converted


def _resolve_label_tier(
    row: dict[str, Any],
    *,
    source_domain: str,
    manifest_path: Path,
    line_no: int,
) -> str:
    if source_domain not in SOURCE_DOMAINS:
        raise ValueError(
            f"{manifest_path}:{line_no}: source_domain must be one of "
            f"{sorted(SOURCE_DOMAINS)}, got {source_domain!r}"
        )
    value = row.get("label_tier")
    if source_domain == "synth" and value is None:
        return "synthetic"
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{manifest_path}:{line_no}: real manifest rows require an explicit label_tier"
        )
    allowed = {tier.value for tier in EffectiveLabelTier}
    if value not in allowed:
        raise ValueError(f"{manifest_path}:{line_no}: unknown label_tier: {value!r}")
    if source_domain == "synth" and value != EffectiveLabelTier.SYNTHETIC.value:
        raise ValueError(
            f"{manifest_path}:{line_no}: synthetic rows require label_tier 'synthetic'"
        )
    if source_domain != "synth" and value == EffectiveLabelTier.SYNTHETIC.value:
        raise ValueError(f"{manifest_path}:{line_no}: real rows cannot use label_tier 'synthetic'")
    if source_domain != "synth" and value != EffectiveLabelTier.GOLD.value:
        raise ValueError(f"{manifest_path}:{line_no}: real rows require label_tier 'gold'")
    return value


def main(
    *,
    manifest_path: str,
    output_dir: str,
    image_root: str | None = None,
    hf_cache_dir: str | None = None,
    direct_single_shard: bool = False,
) -> dict[str, object]:
    summary = convert_rust_manifest_to_hf(
        manifest_path=manifest_path,
        output_dir=output_dir,
        image_root=image_root,
        hf_cache_dir=hf_cache_dir,
        direct_single_shard=direct_single_shard,
    )
    return {
        "manifest_path": str(summary.manifest_path),
        "output_dir": str(summary.output_dir),
        "row_count": summary.row_count,
    }


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Rust manifest.jsonl materialization output to a HF dataset."
    )
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument(
        "--direct_single_shard",
        action="store_true",
        help="Move the generated Arrow cache into the output directory instead of rewriting it.",
    )
    args = parser.parse_args(argv)
    print(json.dumps(main(**vars(args)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
