"""Resumable shard writer and manifest store for dataset generation."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import DatasetInfo, Features
from datasets.arrow_writer import ArrowWriter
from datasets.splits import SplitInfo

from scripts.dataset_generation.dataset_generation.config import GenerationRunConfig
from scripts.dataset_generation.dataset_generation.run_context import RunContext


@dataclass(frozen=True)
class GeneratorResumeState:
    """Minimal persisted generator state required to resume scheduling."""

    file_states: dict[str, dict[str, Any]]
    successful_samples: int
    failure_reason_counts: dict[str, int]
    truncation_counts: dict[str, int]
    accepted_system_histogram: dict[str, int]
    accepted_unknown_system_count_successes: int
    accepted_bottom_whitespace_ratios: list[float]
    accepted_vertical_fill_ratios: list[float]


def compute_generation_config_fingerprint(
    *,
    config: GenerationRunConfig,
    resolved_kern_dirs: list[Path],
) -> str:
    """Hash content-affecting generation inputs for resume compatibility."""
    payload = {
        "dataset_preset": config.dataset_preset,
        "kern_dirs": [str(path) for path in resolved_kern_dirs],
        "num_samples": int(config.effective_num_samples),
        "target_accepted_samples": config.effective_target_accepted_samples,
        "max_scheduled_tasks": config.effective_max_scheduled_tasks,
        "variants_per_file": int(config.variants_per_file),
        "adaptive_variants_enabled": bool(config.adaptive_variants_enabled),
        "overflow_truncation_enabled": bool(config.overflow_truncation_enabled),
        "overflow_truncation_max_trials": int(config.overflow_truncation_max_trials),
        "augment_seed": config.augment_seed,
        "data_spec_path": str(config.data_spec_path),
        "strict_data_spec": bool(config.strict_data_spec),
        "image_width": config.image_width,
        "image_height": config.image_height,
        "render_pedals_enabled": bool(config.render_pedals_enabled),
        "render_pedals_probability": float(config.render_pedals_probability),
        "render_pedals_measures_probability": float(config.render_pedals_measures_probability),
        "render_instrument_piano_enabled": bool(config.render_instrument_piano_enabled),
        "render_instrument_piano_probability": float(config.render_instrument_piano_probability),
        "render_sforzando_enabled": bool(config.render_sforzando_enabled),
        "render_sforzando_probability": float(config.render_sforzando_probability),
        "render_sforzando_per_note_probability": float(
            config.render_sforzando_per_note_probability
        ),
        "render_accent_enabled": bool(config.render_accent_enabled),
        "render_accent_probability": float(config.render_accent_probability),
        "render_accent_per_note_probability": float(config.render_accent_per_note_probability),
        "render_tempo_enabled": bool(config.render_tempo_enabled),
        "render_tempo_probability": float(config.render_tempo_probability),
        "render_tempo_include_mm_probability": float(config.render_tempo_include_mm_probability),
        "render_hairpins_enabled": bool(config.render_hairpins_enabled),
        "render_hairpins_probability": float(config.render_hairpins_probability),
        "render_hairpins_max_spans": int(config.render_hairpins_max_spans),
        "render_dynamic_marks_enabled": bool(config.render_dynamic_marks_enabled),
        "render_dynamic_marks_probability": float(config.render_dynamic_marks_probability),
        "render_dynamic_marks_min_count": int(config.render_dynamic_marks_min_count),
        "render_dynamic_marks_max_count": int(config.render_dynamic_marks_max_count),
        "courtesy_naturals_probability": float(config.courtesy_naturals_probability),
        "disable_offline_image_augmentations": bool(
            config.disable_offline_image_augmentations
        ),
        "geom_x_squeeze_prob": float(config.geom_x_squeeze_prob),
        "geom_x_squeeze_min_scale": float(config.geom_x_squeeze_min_scale),
        "geom_x_squeeze_max_scale": float(config.geom_x_squeeze_max_scale),
        "geom_x_squeeze_apply_in_conservative": bool(
            config.geom_x_squeeze_apply_in_conservative
        ),
        "geom_x_squeeze_preview_force_scale": config.geom_x_squeeze_preview_force_scale,
        "target_min_systems": config.target_min_systems,
        "target_max_systems": config.target_max_systems,
        "render_layout_profile": str(config.render_layout_profile),
        "prefilter_min_non_empty_lines": config.prefilter_min_non_empty_lines,
        "prefilter_max_non_empty_lines": config.prefilter_max_non_empty_lines,
        "prefilter_min_measure_count": config.prefilter_min_measure_count,
        "prefilter_max_measure_count": config.prefilter_max_measure_count,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def derive_sample_seed(*, sample_id: str, salt: str) -> int:
    """Return a stable per-sample NumPy seed."""
    digest = hashlib.sha256(f"{salt}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**32 - 1)


class ResumableDatasetRunStore:
    """Persist accepted samples as Arrow shards and resume state in SQLite."""

    def __init__(
        self,
        *,
        run_context: RunContext,
        features: Features,
        config_fingerprint: str,
        resume_mode: str,
        flush_every_terminal_tasks: int = 100,
    ) -> None:
        self.run_context = run_context
        self.features = features
        self.config_fingerprint = config_fingerprint
        self.resume_mode = str(resume_mode).strip().lower()
        if self.resume_mode not in {"auto", "never", "must"}:
            raise ValueError("resume_mode must be one of: auto, never, must")
        self.flush_every_terminal_tasks = max(1, int(flush_every_terminal_tasks))
        self.resume_session_id = f"{int(time.time())}-{os.getpid()}"
        self.resumed = False
        self.resumed_from_session_id: str | None = None

        self._dirty_file_states: dict[str, dict[str, Any]] = {}
        self._pending_samples: list[dict[str, Any]] = []
        self._pending_sample_metrics: list[dict[str, Any]] = []
        self._terminal_events_since_flush = 0

    @property
    def output_path(self) -> Path:
        return self.run_context.output_path

    @property
    def incomplete_marker_path(self) -> Path:
        return self.run_context.incomplete_marker_path

    @property
    def resume_db_path(self) -> Path:
        return self.run_context.resume_db_path

    @property
    def resume_dir(self) -> Path:
        return self.run_context.resume_dir

    @property
    def staged_shards_dir(self) -> Path:
        return self.run_context.staged_shards_dir

    def prepare(self) -> GeneratorResumeState | None:
        """Create or load resumable state for the current output path."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.resume_dir.mkdir(parents=True, exist_ok=True)
        self.staged_shards_dir.mkdir(parents=True, exist_ok=True)

        has_state = self.resume_db_path.exists()
        has_incomplete_marker = self.incomplete_marker_path.exists()
        has_complete_dataset = (
            (self.output_path / "state.json").exists()
            and (self.output_path / "dataset_info.json").exists()
        )

        if has_complete_dataset and not has_incomplete_marker:
            raise RuntimeError(
                f"Output directory already contains a completed dataset: {self.output_path}"
            )

        if has_state or has_incomplete_marker:
            if self.resume_mode == "never":
                raise RuntimeError(
                    f"Refusing to overwrite resumable dataset state in {self.output_path}; "
                    "set resume_mode=auto|must to continue it."
                )
            if not has_state or not has_incomplete_marker:
                raise RuntimeError(
                    f"Resumable dataset state is inconsistent in {self.output_path}."
                )
            snapshot = self._load_snapshot()
            if snapshot is None:
                if self.resume_mode == "must":
                    raise RuntimeError(
                        f"resume_mode=must but no resumable snapshot exists in {self.output_path}"
                    )
                self._initialize_manifest()
                self.incomplete_marker_path.write_text("incomplete\n", encoding="utf-8")
                return None
            self.resumed = True
            self.incomplete_marker_path.write_text("incomplete\n", encoding="utf-8")
            with self._connect() as conn:
                self._set_meta(conn, "active_resume_session_id", self.resume_session_id)
                self._set_meta(conn, "last_run_artifacts_dir", str(self.run_context.run_artifacts_dir))
                self._set_meta(conn, "last_progress_path", str(self.run_context.progress_path))
            return snapshot

        if self.resume_mode == "must":
            raise RuntimeError(
                f"resume_mode=must but no resumable state exists for {self.output_path}"
            )
        self._initialize_manifest()
        self.incomplete_marker_path.write_text("incomplete\n", encoding="utf-8")
        return None

    def observe_terminal_task(
        self,
        *,
        generator: Any,
        file_path: Path,
        file_state: Any,
        sample: dict[str, Any] | None,
    ) -> None:
        """Buffer state after one terminal task completion."""
        self._dirty_file_states[str(file_path)] = {
            "path": str(file_path),
            "next_variant_idx": int(file_state.next_variant_idx),
            "attempted": int(file_state.attempted),
            "successful": int(file_state.successful),
            "failed": int(file_state.failed),
            "quarantined": int(bool(file_state.quarantined)),
            "early_stopped": int(bool(file_state.early_stopped)),
            "total_variants": int(file_state.total_variants),
        }
        if sample is not None:
            self._pending_samples.append(sample)
            self._pending_sample_metrics.append(
                {
                    "sample_id": str(sample["sample_id"]),
                    "actual_system_count": sample.get("actual_system_count"),
                    "truncation_applied": int(bool(sample.get("truncation_applied", False))),
                    "bottom_whitespace_ratio": sample.get("bottom_whitespace_ratio"),
                    "vertical_fill_ratio": sample.get("vertical_fill_ratio"),
                }
            )
        self._terminal_events_since_flush += 1
        if self._terminal_events_since_flush >= self.flush_every_terminal_tasks:
            self.flush(generator=generator)

    def flush(self, *, generator: Any) -> None:
        """Persist all buffered samples and generator state."""
        if not self._dirty_file_states and not self._pending_samples and self._terminal_events_since_flush == 0:
            return

        shard_row: dict[str, Any] | None = None
        if self._pending_samples:
            shard_row = self._write_pending_shard(self._pending_samples)

        with self._connect() as conn:
            if shard_row is not None:
                conn.execute(
                    """
                    INSERT INTO shards (shard_idx, temp_filename, num_examples, num_bytes, committed_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        int(shard_row["shard_idx"]),
                        str(shard_row["temp_filename"]),
                        int(shard_row["num_examples"]),
                        int(shard_row["num_bytes"]),
                        float(time.time()),
                    ),
                )
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO accepted_samples (
                        sample_id,
                        actual_system_count,
                        truncation_applied,
                        bottom_whitespace_ratio,
                        vertical_fill_ratio
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            row["sample_id"],
                            row["actual_system_count"],
                            row["truncation_applied"],
                            row["bottom_whitespace_ratio"],
                            row["vertical_fill_ratio"],
                        )
                        for row in self._pending_sample_metrics
                    ],
                )
            conn.executemany(
                """
                INSERT INTO file_state (
                    path,
                    next_variant_idx,
                    attempted,
                    successful,
                    failed,
                    quarantined,
                    early_stopped,
                    total_variants
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    next_variant_idx=excluded.next_variant_idx,
                    attempted=excluded.attempted,
                    successful=excluded.successful,
                    failed=excluded.failed,
                    quarantined=excluded.quarantined,
                    early_stopped=excluded.early_stopped,
                    total_variants=excluded.total_variants
                """,
                [
                    (
                        row["path"],
                        row["next_variant_idx"],
                        row["attempted"],
                        row["successful"],
                        row["failed"],
                        row["quarantined"],
                        row["early_stopped"],
                        row["total_variants"],
                    )
                    for row in self._dirty_file_states.values()
                ],
            )
            snapshot = self._build_generator_snapshot(generator)
            self._set_meta(conn, "failure_reason_counts", snapshot["failure_reason_counts"])
            self._set_meta(conn, "truncation_counts", snapshot["truncation_counts"])
            self._set_meta(
                conn,
                "accepted_system_histogram",
                snapshot["accepted_system_histogram"],
            )
            self._set_meta(
                conn,
                "accepted_unknown_system_count_successes",
                snapshot["accepted_unknown_system_count_successes"],
            )
            self._set_meta(
                conn,
                "successful_samples",
                snapshot["successful_samples"],
            )
            self._set_meta(
                conn,
                "last_run_artifacts_dir",
                str(self.run_context.run_artifacts_dir),
            )
            self._set_meta(conn, "last_progress_path", str(self.run_context.progress_path))
            self._set_meta(conn, "last_flush_at", time.time())

        self._dirty_file_states.clear()
        self._pending_samples.clear()
        self._pending_sample_metrics.clear()
        self._terminal_events_since_flush = 0

    def mark_terminal_status(self, *, generator: Any, status: str) -> None:
        """Persist current state and mark the resume manifest terminal."""
        self.flush(generator=generator)
        with self._connect() as conn:
            self._set_meta(conn, "terminal_status", status)
            self._set_meta(conn, "last_flush_at", time.time())

    def finalize(self, *, generator: Any) -> dict[str, Any]:
        """Promote staged shards into a completed HF dataset directory."""
        self.flush(generator=generator)

        with self._connect() as conn:
            shard_rows = list(
                conn.execute(
                    "SELECT shard_idx, temp_filename, num_examples, num_bytes FROM shards ORDER BY shard_idx"
                )
            )

        if not shard_rows:
            raise RuntimeError("Cannot finalize dataset with zero committed shards")

        for stale_file in self.output_path.glob("data-*.arrow"):
            stale_file.unlink()

        shard_lengths: list[int] = []
        data_files: list[dict[str, str]] = []
        total_size_bytes = 0
        total_examples = 0
        total_shards = len(shard_rows)
        for output_index, (_shard_idx, temp_filename, num_examples, num_bytes) in enumerate(shard_rows):
            source_path = self.staged_shards_dir / temp_filename
            target_name = f"data-{output_index:05d}-of-{total_shards:05d}.arrow"
            target_path = self.output_path / target_name
            if not source_path.exists():
                raise RuntimeError(f"Missing committed shard file: {source_path}")
            shutil.move(str(source_path), str(target_path))
            data_files.append({"filename": target_name})
            shard_lengths.append(int(num_examples))
            total_size_bytes += int(num_bytes)
            total_examples += int(num_examples)

        state_payload = {
            "_fingerprint": self.config_fingerprint,
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": "train",
            "_data_files": data_files,
        }
        (self.output_path / "state.json").write_text(
            json.dumps(state_payload, indent=2),
            encoding="utf-8",
        )

        info = DatasetInfo(
            builder_name="generator",
            dataset_name="generator",
            config_name="default",
            features=self.features,
            splits={
                "train": {
                    "name": "train",
                    "num_bytes": total_size_bytes,
                    "num_examples": total_examples,
                    "shard_lengths": shard_lengths,
                    "dataset_name": "generator",
                }
            },
            dataset_size=total_size_bytes,
            size_in_bytes=total_size_bytes,
            download_size=0,
        )
        info.write_to_directory(str(self.output_path))

        with self._connect() as conn:
            self._set_meta(conn, "terminal_status", "completed")
            self._set_meta(conn, "completed", True)
            self._set_meta(conn, "completed_at", time.time())

        with contextlib.suppress(FileNotFoundError):
            self.incomplete_marker_path.unlink()
        with contextlib.suppress(OSError):
            if self.staged_shards_dir.exists() and not any(self.staged_shards_dir.iterdir()):
                self.staged_shards_dir.rmdir()

        return {
            "total_examples": total_examples,
            "total_size_bytes": total_size_bytes,
            "shard_lengths": shard_lengths,
        }

    def _initialize_manifest(self) -> None:
        if self.resume_db_path.exists():
            self.resume_db_path.unlink()
        with self._connect() as conn:
            self._init_schema(conn)
            self._set_meta(conn, "config_fingerprint", self.config_fingerprint)
            self._set_meta(conn, "created_at", time.time())
            self._set_meta(conn, "active_resume_session_id", self.resume_session_id)
            self._set_meta(conn, "output_dir", str(self.output_path))
            self._set_meta(conn, "completed", False)
            self._set_meta(conn, "terminal_status", "running")
            self._set_meta(conn, "last_run_artifacts_dir", str(self.run_context.run_artifacts_dir))
            self._set_meta(conn, "last_progress_path", str(self.run_context.progress_path))

    def _load_snapshot(self) -> GeneratorResumeState | None:
        if not self.resume_db_path.exists():
            return None
        with self._connect() as conn:
            self._init_schema(conn)
            stored_fingerprint = self._get_meta(conn, "config_fingerprint")
            if stored_fingerprint != self.config_fingerprint:
                raise RuntimeError(
                    "Resumable dataset state exists but the generation config fingerprint changed"
                )
            self.resumed_from_session_id = self._get_meta(conn, "active_resume_session_id")
            file_states = {
                row["path"]: {
                    "next_variant_idx": row["next_variant_idx"],
                    "attempted": row["attempted"],
                    "successful": row["successful"],
                    "failed": row["failed"],
                    "quarantined": bool(row["quarantined"]),
                    "early_stopped": bool(row["early_stopped"]),
                    "total_variants": row["total_variants"],
                }
                for row in conn.execute(
                    """
                    SELECT path, next_variant_idx, attempted, successful, failed,
                           quarantined, early_stopped, total_variants
                    FROM file_state
                    """
                )
            }
            accepted_metrics_rows = list(
                conn.execute(
                    """
                    SELECT actual_system_count, bottom_whitespace_ratio, vertical_fill_ratio
                    FROM accepted_samples
                    ORDER BY sample_id
                    """
                )
            )
            return GeneratorResumeState(
                file_states=file_states,
                successful_samples=int(self._get_meta(conn, "successful_samples", 0)),
                failure_reason_counts=dict(self._get_meta(conn, "failure_reason_counts", {})),
                truncation_counts=dict(self._get_meta(conn, "truncation_counts", {})),
                accepted_system_histogram=dict(self._get_meta(conn, "accepted_system_histogram", {})),
                accepted_unknown_system_count_successes=int(
                    self._get_meta(conn, "accepted_unknown_system_count_successes", 0)
                ),
                accepted_bottom_whitespace_ratios=[
                    float(row["bottom_whitespace_ratio"])
                    for row in accepted_metrics_rows
                    if row["bottom_whitespace_ratio"] is not None
                ],
                accepted_vertical_fill_ratios=[
                    float(row["vertical_fill_ratio"])
                    for row in accepted_metrics_rows
                    if row["vertical_fill_ratio"] is not None
                ],
            )

    def _write_pending_shard(self, pending_samples: list[dict[str, Any]]) -> dict[str, Any]:
        shard_idx = self._next_shard_index()
        temp_filename = f"shard-{shard_idx:06d}.arrow"
        shard_path = self.staged_shards_dir / temp_filename
        writer = ArrowWriter(path=str(shard_path), features=self.features)
        for sample in pending_samples:
            writer.write(sample)
        num_examples, num_bytes = writer.finalize()
        return {
            "shard_idx": shard_idx,
            "temp_filename": temp_filename,
            "num_examples": int(num_examples),
            "num_bytes": int(num_bytes),
        }

    def _next_shard_index(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COALESCE(MAX(shard_idx), -1) AS max_shard_idx FROM shards").fetchone()
            assert row is not None
            return int(row["max_shard_idx"]) + 1

    def _build_generator_snapshot(self, generator: Any) -> dict[str, Any]:
        return {
            "successful_samples": int(generator.stats.successful),
            "failure_reason_counts": dict(generator._run_failure_reason_counts),
            "truncation_counts": dict(generator._run_truncation_counts),
            "accepted_system_histogram": {
                str(key): int(value)
                for key, value in generator._accepted_system_count_histogram.items()
            },
            "accepted_unknown_system_count_successes": int(
                generator._accepted_unknown_system_count_successes
            ),
        }

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.resume_db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema(conn)
        return conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS file_state (
                path TEXT PRIMARY KEY,
                next_variant_idx INTEGER NOT NULL,
                attempted INTEGER NOT NULL,
                successful INTEGER NOT NULL,
                failed INTEGER NOT NULL,
                quarantined INTEGER NOT NULL,
                early_stopped INTEGER NOT NULL,
                total_variants INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS accepted_samples (
                sample_id TEXT PRIMARY KEY,
                actual_system_count INTEGER,
                truncation_applied INTEGER NOT NULL,
                bottom_whitespace_ratio REAL,
                vertical_fill_ratio REAL
            );

            CREATE TABLE IF NOT EXISTS shards (
                shard_idx INTEGER PRIMARY KEY,
                temp_filename TEXT NOT NULL UNIQUE,
                num_examples INTEGER NOT NULL,
                num_bytes INTEGER NOT NULL,
                committed_at REAL NOT NULL
            );
            """
        )

    def _set_meta(self, conn: sqlite3.Connection, key: str, value: Any) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value_json) VALUES (?, ?)",
            (key, json.dumps(value)),
        )

    def _get_meta(self, conn: sqlite3.Connection, key: str, default: Any | None = None) -> Any:
        row = conn.execute("SELECT value_json FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return json.loads(row["value_json"])
