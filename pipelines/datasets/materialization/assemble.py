"""Materialize a synthetic Hugging Face dataset in Python (legacy path).

Real pages are not built here; ``python -m pipelines.datasets.releases
build-real`` builds them as their own releases.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import asdict, dataclass, is_dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import datasets
from datasets import Dataset
from PIL import Image
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from pipelines.datasets.materialization.identity import content_digest
from pipelines.datasets.data_spec import (
    DEFAULT_DATA_SPEC_PATH,
    resolve_image_size_from_spec,
)
from pipelines.datasets.materialization.images import (
    SYNTHETIC_BACKGROUND_MODES,
    rasterize_synthetic_svg,
    stable_seed,
)
from pipelines.datasets.materialization.registry import SourceConfig, load_registry
from pipelines.datasets.materialization.rust_manifest_to_hf import (
    CAPABILITY_FIELDS,
    FINAL_FEATURES,
)

SYNTHETIC_RENDER_DIR_NAME = "6_concatenated"
DEFAULT_WRITER_BATCH_SIZE = 8
DATASET_IMAGE_FORMAT = "JPEG"
DATASET_IMAGE_QUALITY = 95


@dataclass(frozen=True)
class AssemblySummary:
    output_dir: Path
    row_count: int
    source_counts: dict[str, int]
    skipped_counts: dict[str, int]
    skipped_sample_ids: dict[str, list[str]]


@dataclass(frozen=True)
class _SyntheticWorkItem:
    sequence: int
    source_name: str
    source_dataset: str
    source_split: str
    render_settings_path: Path
    render_path: Path
    svg_path: Path
    target_path: Path
    transcription: str
    sample_id: str
    svg_system_count: int
    seed: int
    capabilities: dict[str, bool] | None = None


_WorkItem = _SyntheticWorkItem


def materialize_dataset(
    *,
    registry: str | Path,
    output_dir: str | Path,
    data_spec_path: str | Path = DEFAULT_DATA_SPEC_PATH,
    strict_data_spec: bool = True,
    preview_dir: str | Path | None = None,
    max_synthetic_rows: int | None = None,
    num_workers: int = 1,
    writer_batch_size: int = DEFAULT_WRITER_BATCH_SIZE,
    hf_cache_dir: str | Path | None = None,
    synthetic_augraphy: bool = True,
    synthetic_background: str = "current",
    quiet: bool = False,
) -> AssemblySummary:
    _validate_synthetic_background(synthetic_background)
    _configure_worker_environment()
    width, height = resolve_image_size_from_spec(
        image_width=None,
        image_height=None,
        data_spec_path=data_spec_path,
        strict_data_spec=strict_data_spec,
    )
    loaded_registry = load_registry(registry)
    preview_path = Path(preview_dir) if preview_dir is not None else None
    if preview_path is not None:
        preview_path.mkdir(parents=True, exist_ok=True)

    skipped: Counter[str] = Counter()
    skipped_sample_ids: dict[str, list[str]] = {}
    source_counts: Counter[str] = Counter()
    resolved_num_workers = max(1, int(num_workers))
    resolved_writer_batch_size = max(1, int(writer_batch_size))
    progress_total = (
        None
        if quiet
        else _resolve_progress_total(
            loaded_registry.sources,
            max_synthetic_rows=max_synthetic_rows,
        )
    )
    progress = None if quiet else _build_progress()
    output_path = Path(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_parent = Path(hf_cache_dir) if hf_cache_dir is not None else output_path.parent
    cache_parent.mkdir(parents=True, exist_ok=True)

    progress_was_enabled = datasets.is_progress_bar_enabled()
    datasets.disable_progress_bar()
    try:
        with (
            tempfile.TemporaryDirectory(prefix="final_dataset_hf_", dir=cache_parent) as cache_dir,
            progress if progress is not None else nullcontext() as active_progress,
        ):
            task_id = None
            if progress is not None:
                task_id = active_progress.add_task("Assembling final dataset", total=progress_total)

            dataset = Dataset.from_generator(
                _iter_dataset_rows,
                features=FINAL_FEATURES,
                cache_dir=cache_dir,
                keep_in_memory=False,
                gen_kwargs={
                    "sources": loaded_registry.sources,
                    "width": width,
                    "height": height,
                    "num_workers": resolved_num_workers,
                    "skipped": skipped,
                    "skipped_sample_ids": skipped_sample_ids,
                    "source_counts": source_counts,
                    "preview_path": preview_path,
                    "max_synthetic_rows": max_synthetic_rows,
                    "synthetic_augraphy": synthetic_augraphy,
                    "synthetic_background": synthetic_background,
                    "progress": active_progress,
                    "task_id": task_id,
                },
                fingerprint=_materialization_fingerprint(
                    sources=loaded_registry.sources,
                    width=width,
                    height=height,
                    num_workers=resolved_num_workers,
                    writer_batch_size=resolved_writer_batch_size,
                    image_encoding=f"{DATASET_IMAGE_FORMAT.lower()}_q{DATASET_IMAGE_QUALITY}",
                    max_synthetic_rows=max_synthetic_rows,
                    synthetic_augraphy=synthetic_augraphy,
                    synthetic_background=synthetic_background,
                ),
                writer_batch_size=resolved_writer_batch_size,
            )

            if len(dataset) == 0:
                raise ValueError("No rows assembled; check registry inputs and source availability")

            dataset.save_to_disk(str(output_path))
    finally:
        if progress_was_enabled:
            datasets.enable_progress_bar()

    summary = AssemblySummary(
        output_dir=output_path,
        row_count=len(dataset),
        source_counts=dict(source_counts),
        skipped_counts=dict(skipped),
        skipped_sample_ids=skipped_sample_ids,
    )
    if not quiet:
        print(json.dumps(_summary_to_json(summary), sort_keys=True))
    return summary


def _materialization_fingerprint(
    *,
    sources: tuple[SourceConfig, ...],
    width: int,
    height: int,
    num_workers: int,
    writer_batch_size: int,
    image_encoding: str,
    max_synthetic_rows: int | None,
    synthetic_augraphy: bool,
    synthetic_background: str,
) -> str:
    payload = {
        "sources": [_json_safe(source) for source in sources],
        "width": width,
        "height": height,
        "num_workers": num_workers,
        "writer_batch_size": writer_batch_size,
        "image_encoding": image_encoding,
        "max_synthetic_rows": max_synthetic_rows,
        "synthetic_augraphy": synthetic_augraphy,
        "synthetic_background": synthetic_background,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"final_dataset_materialization_{digest}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _validate_synthetic_background(synthetic_background: str) -> None:
    if synthetic_background not in SYNTHETIC_BACKGROUND_MODES:
        choices = ", ".join(sorted(SYNTHETIC_BACKGROUND_MODES))
        raise ValueError(
            f"synthetic_background must be one of: {choices}; got {synthetic_background!r}"
        )


def _iter_dataset_rows(
    sources: tuple[SourceConfig, ...],
    *,
    width: int,
    height: int,
    num_workers: int,
    skipped: Counter[str],
    skipped_sample_ids: dict[str, list[str]],
    source_counts: Counter[str],
    preview_path: Path | None,
    max_synthetic_rows: int | None,
    synthetic_augraphy: bool,
    synthetic_background: str,
    progress: Progress | None,
    task_id: int | None,
) -> Iterator[dict[str, Any]]:
    if num_workers == 1:
        yield from _assemble_rows_sequential(
            sources,
            width=width,
            height=height,
            num_workers=num_workers,
            skipped=skipped,
            skipped_sample_ids=skipped_sample_ids,
            source_counts=source_counts,
            preview_path=preview_path,
            max_synthetic_rows=max_synthetic_rows,
            synthetic_augraphy=synthetic_augraphy,
            synthetic_background=synthetic_background,
            progress=progress,
            task_id=task_id,
        )
    else:
        yield from _assemble_rows_parallel(
            sources,
            width=width,
            height=height,
            num_workers=num_workers,
            skipped=skipped,
            skipped_sample_ids=skipped_sample_ids,
            source_counts=source_counts,
            preview_path=preview_path,
            max_synthetic_rows=max_synthetic_rows,
            synthetic_augraphy=synthetic_augraphy,
            synthetic_background=synthetic_background,
            progress=progress,
            task_id=task_id,
        )


def _assemble_rows_sequential(
    sources: tuple[SourceConfig, ...],
    *,
    width: int,
    height: int,
    num_workers: int,
    skipped: Counter[str],
    skipped_sample_ids: dict[str, list[str]],
    source_counts: Counter[str],
    preview_path: Path | None,
    max_synthetic_rows: int | None,
    synthetic_augraphy: bool,
    synthetic_background: str,
    progress: Progress | None,
    task_id: int | None,
) -> Iterator[dict[str, Any]]:
    synthetic_rows = 0
    for source in sources:
        if source.kind == "synthetic_svg":
            if max_synthetic_rows is not None and synthetic_rows >= max_synthetic_rows:
                continue
            for row in iter_synthetic_rows(
                source,
                width=width,
                height=height,
                skipped=skipped,
                synthetic_augraphy=synthetic_augraphy,
                synthetic_background=synthetic_background,
            ):
                if max_synthetic_rows is not None and synthetic_rows >= max_synthetic_rows:
                    break
                source_counts[source.name] += 1
                synthetic_rows += 1
                _maybe_write_preview(row, preview_path=preview_path)
                _advance_progress(progress, task_id)
                yield _prepare_row_for_dataset(row)


def _assemble_rows_parallel(
    sources: tuple[SourceConfig, ...],
    *,
    width: int,
    height: int,
    num_workers: int,
    skipped: Counter[str],
    skipped_sample_ids: dict[str, list[str]],
    source_counts: Counter[str],
    preview_path: Path | None,
    max_synthetic_rows: int | None,
    synthetic_augraphy: bool,
    synthetic_background: str,
    progress: Progress | None,
    task_id: int | None,
) -> Iterator[dict[str, Any]]:
    work_items = _iter_work_items(
        sources,
        skipped=skipped,
        skipped_sample_ids=skipped_sample_ids,
        max_synthetic_rows=max_synthetic_rows,
        workers=num_workers,
    )

    next_sequence_to_write = 0
    completed: dict[int, tuple[str, dict[str, Any]] | None] = {}
    max_in_flight = max(num_workers * 4, num_workers)
    encode_in_worker = preview_path is None
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        pending_items = iter(work_items)
        in_flight: dict[Future[dict[str, Any]], _WorkItem] = {}

        def submit_until_full() -> None:
            while len(in_flight) + len(completed) < max_in_flight:
                try:
                    item = next(pending_items)
                except StopIteration:
                    break
                in_flight[
                    executor.submit(
                        _build_row_from_work_item,
                        item,
                        width,
                        height,
                        encode_in_worker,
                        synthetic_augraphy,
                        synthetic_background,
                    )
                ] = item

        submit_until_full()
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                item = in_flight.pop(future)
                try:
                    row = future.result()
                except Exception:
                    skipped[f"{_work_item_source_name(item)}:worker_error"] += 1
                    completed[item.sequence] = None
                else:
                    completed[item.sequence] = (_work_item_source_name(item), row)
                _advance_progress(progress, task_id)
                while next_sequence_to_write in completed:
                    ready = completed.pop(next_sequence_to_write)
                    if ready is not None:
                        source_name, ready_row = ready
                        source_counts[source_name] += 1
                        _maybe_write_preview(ready_row, preview_path=preview_path)
                        yield _prepare_row_for_dataset(ready_row)
                    next_sequence_to_write += 1
            submit_until_full()


def _advance_progress(progress: Progress | None, task_id: int | None) -> None:
    if progress is not None and task_id is not None:
        progress.update(task_id, advance=1)


def _build_row_from_work_item(
    item: _WorkItem,
    width: int,
    height: int,
    encode_image: bool,
    synthetic_augraphy: bool,
    synthetic_background: str = "current",
) -> dict[str, Any]:
    _configure_worker_environment()
    image, augraphy_outcome = rasterize_synthetic_svg(
        item.svg_path.read_text(encoding="utf-8"),
        width=width,
        height=height,
        seed=item.seed,
        apply_augraphy=synthetic_augraphy,
        synthetic_background=synthetic_background,
    )
    emitted_image_sha256 = _encoded_image_sha256(image)
    curation_stage = "synthetic_svg_augraphy" if synthetic_augraphy else "synthetic_svg"
    row = {
        "image": image,
        "transcription": item.transcription,
        "sample_id": item.sample_id,
        "source": str(item.render_path),
        "source_dataset": item.source_dataset,
        "source_split": item.source_split,
        "source_domain": "synth",
        "curation_stage": curation_stage,
        "svg_system_count": item.svg_system_count,
        "render_settings_path": str(item.render_settings_path),
        "svg_path": str(item.svg_path),
        "target_path": str(item.target_path),
        "augraphy_outcome": augraphy_outcome,
        "document_id": f"synthetic:{item.source_name}:{item.sample_id}",
        "label_tier": "synthetic",
        "sample_revision_digest": content_digest(
            {
                "sample_id": item.sample_id,
                "image_sha256": emitted_image_sha256,
                "transcription_sha256": hashlib.sha256(
                    item.transcription.encode("utf-8")
                ).hexdigest(),
            }
        ),
        **_capability_fields(item.capabilities or {}),
    }
    return _prepare_row_for_dataset(row) if encode_image else row


def _configure_worker_environment() -> None:
    mpl_config_dir = os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "transcoda_matplotlib"),
    )
    Path(mpl_config_dir).mkdir(parents=True, exist_ok=True)


def _prepare_row_for_dataset(row: dict[str, Any]) -> dict[str, Any]:
    image = row.get("image")
    if not isinstance(image, Image.Image):
        return row

    encoded = dict(row)
    encoded["image"] = _encode_image_for_dataset(image)
    return encoded


def _encode_image_for_dataset(image: Image.Image) -> dict[str, bytes | None]:
    buffer = BytesIO()
    image.convert("RGB").save(
        buffer,
        format=DATASET_IMAGE_FORMAT,
        quality=DATASET_IMAGE_QUALITY,
    )
    return {"path": None, "bytes": buffer.getvalue()}


def _encoded_image_sha256(image: Image.Image) -> str:
    encoded = _encode_image_for_dataset(image)["bytes"]
    if not isinstance(encoded, bytes):
        raise TypeError("encoded dataset image has no bytes")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _work_item_source_name(item: _WorkItem) -> str:
    return item.source_name


def _build_progress() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console(stderr=True),
    )


def _resolve_progress_total(
    sources: tuple[SourceConfig, ...],
    *,
    max_synthetic_rows: int | None,
) -> int:
    return _estimate_total_rows(
        sources,
        max_synthetic_rows=max_synthetic_rows,
    )


def _estimate_total_rows(
    sources: tuple[SourceConfig, ...],
    *,
    max_synthetic_rows: int | None,
) -> int:
    total = 0
    synthetic_rows = 0
    for source in sources:
        if source.kind == "synthetic_svg":
            if max_synthetic_rows is not None and synthetic_rows >= max_synthetic_rows:
                continue
            remaining = None if max_synthetic_rows is None else max_synthetic_rows - synthetic_rows
            count = _estimate_synthetic_rows(source, limit=remaining)
            synthetic_rows += count
            total += count
    return total


def _estimate_synthetic_rows(source: SourceConfig, *, limit: int | None) -> int:
    if source.layout_targets_report_jsonl is not None:
        return _estimate_layout_target_report_rows(source, limit=limit)

    assert source.render_settings_jsonl is not None
    assert source.target_dir is not None
    render_settings_path = source.render_settings_jsonl
    if not render_settings_path.exists():
        return 0
    render_dir = render_settings_path.parent.parent / SYNTHETIC_RENDER_DIR_NAME

    count = 0
    for line in render_settings_path.read_text(encoding="utf-8").splitlines():
        if limit is not None and count >= limit:
            break
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "success":
            continue
        render_path = Path(str(payload.get("file") or ""))
        svg_path = Path(str(payload.get("selected_svg_file") or ""))
        if not _path_exists_from_cwd(render_path) or not svg_path.exists():
            continue
        target_path = _target_path_for_render_path(
            render_path,
            render_dir=render_dir,
            target_dir=source.target_dir,
        )
        if target_path is None or not target_path.exists():
            continue
        if not target_path.read_text(encoding="utf-8").strip():
            continue
        count += 1
    return count


def _estimate_layout_target_report_rows(source: SourceConfig, *, limit: int | None) -> int:
    assert source.layout_targets_report_jsonl is not None
    report_path = source.layout_targets_report_jsonl
    if not report_path.exists():
        return 0

    count = 0
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if limit is not None and count >= limit:
            break
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "success":
            continue
        render_value = payload.get("file")
        svg_value = payload.get("selected_svg_file")
        target_value = payload.get("output_file")
        if not (
            isinstance(render_value, str)
            and isinstance(svg_value, str)
            and isinstance(target_value, str)
        ):
            continue
        render_path = _layout_report_render_path(render_value, report_path)
        svg_path = Path(svg_value)
        target_path = Path(target_value)
        if not (
            _path_exists_from_cwd(render_path)
            and _path_exists_from_cwd(svg_path)
            and _path_exists_from_cwd(target_path)
        ):
            continue
        if not target_path.read_text(encoding="utf-8").strip():
            continue
        count += 1
    return count


def _iter_work_items(
    sources: tuple[SourceConfig, ...],
    *,
    skipped: Counter[str],
    skipped_sample_ids: dict[str, list[str]],
    max_synthetic_rows: int | None,
    workers: int,
):
    sequence = 0
    synthetic_rows = 0
    for source in sources:
        if source.kind == "synthetic_svg":
            if max_synthetic_rows is not None and synthetic_rows >= max_synthetic_rows:
                continue
            for item in _iter_synthetic_work_items(
                source, skipped=skipped, start_sequence=sequence
            ):
                if max_synthetic_rows is not None and synthetic_rows >= max_synthetic_rows:
                    break
                yield item
                sequence += 1
                synthetic_rows += 1


def _iter_synthetic_work_items(
    source: SourceConfig,
    *,
    skipped: Counter[str],
    start_sequence: int,
):
    if source.layout_targets_report_jsonl is not None:
        yield from _iter_layout_target_report_work_items(
            source,
            skipped=skipped,
            start_sequence=start_sequence,
        )
        return

    assert source.render_settings_jsonl is not None
    assert source.target_dir is not None
    render_settings_path = source.render_settings_jsonl
    if not render_settings_path.exists():
        skipped[f"{source.name}:missing_render_settings"] += 1
        return
    render_dir = render_settings_path.parent.parent / SYNTHETIC_RENDER_DIR_NAME
    sequence = start_sequence

    for line_no, line in enumerate(
        render_settings_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            skipped[f"{source.name}:invalid_jsonl"] += 1
            continue

        if payload.get("status") != "success":
            skipped[f"{source.name}:non_success_status"] += 1
            continue

        render_path = Path(str(payload.get("file") or ""))
        svg_path = Path(str(payload.get("selected_svg_file") or ""))
        if not _path_exists_from_cwd(render_path):
            skipped[f"{source.name}:missing_render_source"] += 1
            continue
        if not svg_path.exists():
            skipped[f"{source.name}:missing_svg"] += 1
            continue
        target_path = _target_path_for_render_path(
            render_path,
            render_dir=render_dir,
            target_dir=source.target_dir,
        )
        if target_path is None:
            skipped[f"{source.name}:render_path_outside_render_dir"] += 1
            continue
        if not target_path.exists():
            skipped[f"{source.name}:missing_target"] += 1
            continue
        transcription = target_path.read_text(encoding="utf-8")
        if not transcription.strip():
            skipped[f"{source.name}:empty_target"] += 1
            continue

        sample_id = render_path.stem
        yield _SyntheticWorkItem(
            sequence=sequence,
            source_name=source.name,
            source_dataset=source.source_dataset,
            source_split=source.split,
            render_settings_path=render_settings_path,
            render_path=render_path,
            svg_path=svg_path,
            target_path=target_path,
            transcription=transcription,
            sample_id=sample_id,
            svg_system_count=int(payload.get("selected_system_count") or 0),
            seed=stable_seed(f"{source.name}:{sample_id}:{line_no}"),
            capabilities=_capability_fields(
                payload.get("capabilities")
                if isinstance(payload.get("capabilities"), dict)
                else payload
            ),
        )
        sequence += 1


def _iter_layout_target_report_work_items(
    source: SourceConfig,
    *,
    skipped: Counter[str],
    start_sequence: int,
):
    assert source.layout_targets_report_jsonl is not None
    report_path = source.layout_targets_report_jsonl
    if not report_path.exists():
        skipped[f"{source.name}:missing_layout_targets_report"] += 1
        return

    sequence = start_sequence
    for line_no, line in enumerate(report_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            skipped[f"{source.name}:invalid_jsonl"] += 1
            continue

        if payload.get("status") != "success":
            skipped[f"{source.name}:non_success_status"] += 1
            continue

        render_value = payload.get("file")
        if not isinstance(render_value, str) or not render_value:
            skipped[f"{source.name}:missing_render_file_field"] += 1
            continue
        svg_value = payload.get("selected_svg_file")
        if not isinstance(svg_value, str) or not svg_value:
            skipped[f"{source.name}:missing_svg_field"] += 1
            continue
        target_value = payload.get("output_file")
        if not isinstance(target_value, str) or not target_value:
            skipped[f"{source.name}:missing_target_field"] += 1
            continue

        render_path = _layout_report_render_path(render_value, report_path)
        if not _path_exists_from_cwd(render_path):
            skipped[f"{source.name}:missing_render_source"] += 1
            continue
        svg_path = Path(svg_value)
        if not _path_exists_from_cwd(svg_path):
            skipped[f"{source.name}:missing_svg"] += 1
            continue
        target_path = Path(target_value)
        if not _path_exists_from_cwd(target_path):
            skipped[f"{source.name}:missing_target"] += 1
            continue
        transcription = target_path.read_text(encoding="utf-8")
        if not transcription.strip():
            skipped[f"{source.name}:empty_target"] += 1
            continue

        sample_id = render_path.stem
        yield _SyntheticWorkItem(
            sequence=sequence,
            source_name=source.name,
            source_dataset=source.source_dataset,
            source_split=source.split,
            render_settings_path=report_path,
            render_path=render_path,
            svg_path=svg_path,
            target_path=target_path,
            transcription=transcription,
            sample_id=sample_id,
            svg_system_count=int(payload.get("selected_system_count") or 0),
            seed=stable_seed(f"{source.name}:{sample_id}:{line_no}"),
            capabilities=_capability_fields(
                payload.get("capabilities")
                if isinstance(payload.get("capabilities"), dict)
                else payload
            ),
        )
        sequence += 1


def iter_synthetic_rows(
    source: SourceConfig,
    *,
    width: int,
    height: int,
    skipped: Counter[str],
    synthetic_augraphy: bool = True,
    synthetic_background: str = "current",
):
    _validate_synthetic_background(synthetic_background)
    for item in _iter_synthetic_work_items(source, skipped=skipped, start_sequence=0):
        yield _build_row_from_work_item(
            item,
            width=width,
            height=height,
            encode_image=False,
            synthetic_augraphy=synthetic_augraphy,
            synthetic_background=synthetic_background,
        )


def _path_exists_from_cwd(path: Path) -> bool:
    return path.exists() if path.is_absolute() else path.resolve().exists()


def _capability_fields(values: dict[str, Any]) -> dict[str, bool]:
    return {name: values.get(name) is True for name in CAPABILITY_FIELDS}


def _layout_report_render_path(value: str, report_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or _path_exists_from_cwd(path):
        return path
    return report_path.parent.parent / SYNTHETIC_RENDER_DIR_NAME / path


def _target_path_for_render_path(
    render_path: Path,
    *,
    render_dir: Path,
    target_dir: Path,
) -> Path | None:
    normalized_render_path = render_path if render_path.is_absolute() else render_path.resolve()
    normalized_render_dir = render_dir if render_dir.is_absolute() else render_dir.resolve()
    try:
        relative = normalized_render_path.relative_to(normalized_render_dir)
    except ValueError:
        return None
    return target_dir / relative


def _maybe_write_preview(row: dict[str, Any], *, preview_path: Path | None) -> None:
    if preview_path is None:
        return
    existing = len(list(preview_path.glob("*.png")))
    if existing >= 25:
        return
    image = row["image"]
    if isinstance(image, Image.Image):
        safe_id = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in row["sample_id"]
        )
        image.save(preview_path / f"{existing:04d}_{safe_id}.png")
    elif isinstance(image, dict) and image.get("bytes") is not None:
        safe_id = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in row["sample_id"]
        )
        with Image.open(BytesIO(image["bytes"])) as decoded:
            decoded.save(preview_path / f"{existing:04d}_{safe_id}.png")


def _summary_to_json(summary: AssemblySummary) -> dict[str, object]:
    return {
        "output_dir": str(summary.output_dir),
        "row_count": summary.row_count,
        "source_counts": summary.source_counts,
        "skipped_counts": summary.skipped_counts,
        "skipped_sample_ids": summary.skipped_sample_ids,
    }
