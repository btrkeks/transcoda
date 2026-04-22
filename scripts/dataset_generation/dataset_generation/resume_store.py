"""Resumable HF shard store for the production rewrite."""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import DatasetInfo, Features
from datasets.arrow_writer import ArrowWriter

from scripts.dataset_generation.dataset_generation.io import write_json
from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.run_context import RunContext
from scripts.dataset_generation.dataset_generation.types import ResumeSnapshot


def compute_config_fingerprint(
    *,
    input_dirs: tuple[str, ...],
    base_seed: int,
    recipe: ProductionRecipe,
    extra_config: dict[str, Any] | None = None,
) -> str:
    payload = {
        "input_dirs": list(input_dirs),
        "base_seed": int(base_seed),
        "recipe": asdict(recipe),
        "extra_config": extra_config or {},
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RuntimeSnapshot:
    next_sample_idx: int
    accepted_samples: int
    rejected_samples: int
    failure_reason_counts: dict[str, int]
    truncation_counts: dict[str, int]
    full_render_system_histogram: dict[str, int]
    accepted_system_histogram: dict[str, dict[str, int]]
    truncated_output_system_histogram: dict[str, int]
    preferred_5_6_counts: dict[str, int]
    bottom_whitespace_px_histogram: dict[str, int]
    top_whitespace_px_histogram: dict[str, int]
    content_height_px_histogram: dict[str, int]
    terminal_timeout_crash_artifacts: int
    terminal_process_expired_crash_artifacts: int
    requested_target_bucket_histogram: dict[str, int]
    candidate_hit_counts: dict[str, int]
    retry_counts: dict[str, int]
    quarantined_sources: tuple[str, ...]
    augmentation_outcome_counts: dict[str, int]
    augmentation_band_counts: dict[str, int]
    augmentation_branch_counts: dict[str, int]
    final_geometry_counts: dict[str, int]
    oob_failure_reason_counts: dict[str, int]
    outer_gate_failure_reason_counts: dict[str, int]


class ResumableShardStore:
    def __init__(
        self,
        *,
        run_context: RunContext,
        features: Features,
        config_fingerprint: str,
        resume_mode: str,
    ) -> None:
        self.run_context = run_context
        self.features = features
        self.config_fingerprint = config_fingerprint
        normalized_resume_mode = str(resume_mode).strip().lower()
        if normalized_resume_mode not in {"auto", "off", "require"}:
            raise ValueError("resume_mode must be one of: auto, off, require")
        self.resume_mode = normalized_resume_mode

    @property
    def output_path(self) -> Path:
        return self.run_context.output_path

    @property
    def resume_db_path(self) -> Path:
        return self.run_context.resume_db_path

    @property
    def staged_shards_dir(self) -> Path:
        return self.run_context.staged_shards_dir

    @property
    def incomplete_marker_path(self) -> Path:
        return self.run_context.incomplete_marker_path

    def prepare(self) -> ResumeSnapshot | None:
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.run_context.resume_dir.mkdir(parents=True, exist_ok=True)
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
            if self.resume_mode == "off":
                self._clear_incomplete_state()
                self._initialize_manifest()
                self.incomplete_marker_path.write_text("incomplete\n", encoding="utf-8")
                return None
            if not has_state or not has_incomplete_marker:
                raise RuntimeError(
                    f"Resumable dataset state is inconsistent in {self.output_path}"
                )
            snapshot = self._load_snapshot()
            if snapshot is None:
                raise RuntimeError(f"No resumable snapshot exists in {self.output_path}")
            self.incomplete_marker_path.write_text("incomplete\n", encoding="utf-8")
            return snapshot

        if self.resume_mode == "require":
            raise RuntimeError(
                f"resume_mode=require but no resumable state exists for {self.output_path}"
            )
        self._initialize_manifest()
        self.incomplete_marker_path.write_text("incomplete\n", encoding="utf-8")
        return None

    def commit(
        self,
        *,
        snapshot: RuntimeSnapshot,
        sample_rows: list[dict[str, Any]],
    ) -> None:
        shard_row: dict[str, Any] | None = None
        if sample_rows:
            shard_row = self._write_pending_shard(sample_rows)

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
            self._set_meta(conn, "next_sample_idx", snapshot.next_sample_idx)
            self._set_meta(conn, "accepted_samples", snapshot.accepted_samples)
            self._set_meta(conn, "rejected_samples", snapshot.rejected_samples)
            self._set_meta(conn, "failure_reason_counts", snapshot.failure_reason_counts)
            self._set_meta(conn, "truncation_counts", snapshot.truncation_counts)
            self._set_meta(conn, "full_render_system_histogram", snapshot.full_render_system_histogram)
            self._set_meta(conn, "accepted_system_histogram", snapshot.accepted_system_histogram)
            self._set_meta(
                conn,
                "truncated_output_system_histogram",
                snapshot.truncated_output_system_histogram,
            )
            self._set_meta(conn, "preferred_5_6_counts", snapshot.preferred_5_6_counts)
            self._set_meta(
                conn,
                "bottom_whitespace_px_histogram",
                snapshot.bottom_whitespace_px_histogram,
            )
            self._set_meta(
                conn,
                "top_whitespace_px_histogram",
                snapshot.top_whitespace_px_histogram,
            )
            self._set_meta(
                conn,
                "content_height_px_histogram",
                snapshot.content_height_px_histogram,
            )
            self._set_meta(
                conn,
                "terminal_timeout_crash_artifacts",
                snapshot.terminal_timeout_crash_artifacts,
            )
            self._set_meta(
                conn,
                "terminal_process_expired_crash_artifacts",
                snapshot.terminal_process_expired_crash_artifacts,
            )
            self._set_meta(
                conn,
                "requested_target_bucket_histogram",
                snapshot.requested_target_bucket_histogram,
            )
            self._set_meta(conn, "candidate_hit_counts", snapshot.candidate_hit_counts)
            self._set_meta(conn, "retry_counts", snapshot.retry_counts)
            self._set_meta(conn, "quarantined_sources", list(snapshot.quarantined_sources))
            self._set_meta(conn, "augmentation_outcome_counts", snapshot.augmentation_outcome_counts)
            self._set_meta(conn, "augmentation_band_counts", snapshot.augmentation_band_counts)
            self._set_meta(conn, "augmentation_branch_counts", snapshot.augmentation_branch_counts)
            self._set_meta(conn, "final_geometry_counts", snapshot.final_geometry_counts)
            self._set_meta(conn, "oob_failure_reason_counts", snapshot.oob_failure_reason_counts)
            self._set_meta(conn, "outer_gate_failure_reason_counts", snapshot.outer_gate_failure_reason_counts)
            self._set_meta(conn, "terminal_status", "running")
            self._set_meta(conn, "last_flush_at", time.time())

    def mark_terminal_status(self, *, status: str) -> None:
        with self._connect() as conn:
            self._set_meta(conn, "terminal_status", status)
            self._set_meta(conn, "last_flush_at", time.time())

    def finalize(self) -> dict[str, Any]:
        with self._connect() as conn:
            shard_rows = list(
                conn.execute(
                    "SELECT shard_idx, temp_filename, num_examples, num_bytes FROM shards ORDER BY shard_idx"
                )
            )
            snapshot = ResumeSnapshot(
                next_sample_idx=int(self._get_meta(conn, "next_sample_idx", 0)),
                accepted_samples=int(self._get_meta(conn, "accepted_samples", 0)),
                rejected_samples=int(self._get_meta(conn, "rejected_samples", 0)),
                failure_reason_counts=dict(self._get_meta(conn, "failure_reason_counts", {})),
                truncation_counts=dict(self._get_meta(conn, "truncation_counts", {})),
                full_render_system_histogram=dict(
                    self._get_meta(conn, "full_render_system_histogram", {})
                ),
                accepted_system_histogram=dict(
                    self._get_meta(conn, "accepted_system_histogram", {})
                ),
                truncated_output_system_histogram=dict(
                    self._get_meta(conn, "truncated_output_system_histogram", {})
                ),
                preferred_5_6_counts=dict(self._get_meta(conn, "preferred_5_6_counts", {})),
                bottom_whitespace_px_histogram=dict(
                    self._get_meta(conn, "bottom_whitespace_px_histogram", {})
                ),
                top_whitespace_px_histogram=dict(
                    self._get_meta(conn, "top_whitespace_px_histogram", {})
                ),
                content_height_px_histogram=dict(
                    self._get_meta(conn, "content_height_px_histogram", {})
                ),
                terminal_timeout_crash_artifacts=int(
                    self._get_meta(conn, "terminal_timeout_crash_artifacts", 0)
                ),
                terminal_process_expired_crash_artifacts=int(
                    self._get_meta(conn, "terminal_process_expired_crash_artifacts", 0)
                ),
                requested_target_bucket_histogram=dict(
                    self._get_meta(conn, "requested_target_bucket_histogram", {})
                ),
                candidate_hit_counts=dict(self._get_meta(conn, "candidate_hit_counts", {})),
                retry_counts=dict(self._get_meta(conn, "retry_counts", {})),
                quarantined_sources=tuple(self._get_meta(conn, "quarantined_sources", [])),
                augmentation_outcome_counts=dict(
                    self._get_meta(conn, "augmentation_outcome_counts", {})
                ),
                augmentation_band_counts=dict(
                    self._get_meta(conn, "augmentation_band_counts", {})
                ),
                augmentation_branch_counts=dict(
                    self._get_meta(conn, "augmentation_branch_counts", {})
                ),
                final_geometry_counts=dict(self._get_meta(conn, "final_geometry_counts", {})),
                oob_failure_reason_counts=dict(
                    self._get_meta(conn, "oob_failure_reason_counts", {})
                ),
                outer_gate_failure_reason_counts=dict(
                    self._get_meta(conn, "outer_gate_failure_reason_counts", {})
                ),
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
            total_examples += int(num_examples)
            total_size_bytes += int(num_bytes)

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
            builder_name="dataset_generation",
            dataset_name="dataset_generation",
            config_name="default",
            features=self.features,
            splits={
                "train": {
                    "name": "train",
                    "num_bytes": total_size_bytes,
                    "num_examples": total_examples,
                    "shard_lengths": shard_lengths,
                    "dataset_name": "dataset_generation",
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
            "snapshot": snapshot,
        }

    def _clear_incomplete_state(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.incomplete_marker_path.unlink()
        with contextlib.suppress(FileNotFoundError):
            self.resume_db_path.unlink()
        if self.run_context.resume_dir.exists():
            shutil.rmtree(self.run_context.resume_dir, ignore_errors=True)
        for stale_file in self.output_path.glob("data-*.arrow"):
            stale_file.unlink()
        for filename in ("state.json", "dataset_info.json"):
            with contextlib.suppress(FileNotFoundError):
                (self.output_path / filename).unlink()

    def _initialize_manifest(self) -> None:
        if self.resume_db_path.exists():
            self.resume_db_path.unlink()
        with self._connect() as conn:
            self._init_schema(conn)
            self._set_meta(conn, "config_fingerprint", self.config_fingerprint)
            self._set_meta(conn, "created_at", time.time())
            self._set_meta(conn, "next_sample_idx", 0)
            self._set_meta(conn, "accepted_samples", 0)
            self._set_meta(conn, "rejected_samples", 0)
            self._set_meta(conn, "failure_reason_counts", {})
            self._set_meta(conn, "truncation_counts", {})
            self._set_meta(conn, "full_render_system_histogram", {})
            self._set_meta(conn, "accepted_system_histogram", {})
            self._set_meta(conn, "truncated_output_system_histogram", {})
            self._set_meta(
                conn,
                "preferred_5_6_counts",
                {
                    "preferred_5_6_accepted_full": 0,
                    "preferred_5_6_rescued": 0,
                    "preferred_5_6_truncated": 0,
                    "preferred_5_6_failed": 0,
                },
            )
            self._set_meta(conn, "bottom_whitespace_px_histogram", {})
            self._set_meta(conn, "top_whitespace_px_histogram", {})
            self._set_meta(conn, "content_height_px_histogram", {})
            self._set_meta(conn, "terminal_timeout_crash_artifacts", 0)
            self._set_meta(conn, "terminal_process_expired_crash_artifacts", 0)
            self._set_meta(conn, "requested_target_bucket_histogram", {})
            self._set_meta(conn, "candidate_hit_counts", {})
            self._set_meta(conn, "retry_counts", {})
            self._set_meta(conn, "quarantined_sources", [])
            self._set_meta(conn, "augmentation_outcome_counts", {})
            self._set_meta(conn, "augmentation_band_counts", {})
            self._set_meta(conn, "augmentation_branch_counts", {})
            self._set_meta(conn, "final_geometry_counts", {})
            self._set_meta(conn, "oob_failure_reason_counts", {})
            self._set_meta(conn, "outer_gate_failure_reason_counts", {})
            self._set_meta(conn, "completed", False)
            self._set_meta(conn, "terminal_status", "running")

    def _load_snapshot(self) -> ResumeSnapshot | None:
        if not self.resume_db_path.exists():
            return None
        with self._connect() as conn:
            stored_fingerprint = self._get_meta(conn, "config_fingerprint")
            if stored_fingerprint != self.config_fingerprint:
                raise RuntimeError(
                    "Resumable dataset state exists but the generation config fingerprint changed"
                )
            return ResumeSnapshot(
                next_sample_idx=int(self._get_meta(conn, "next_sample_idx", 0)),
                accepted_samples=int(self._get_meta(conn, "accepted_samples", 0)),
                rejected_samples=int(self._get_meta(conn, "rejected_samples", 0)),
                failure_reason_counts=dict(self._get_meta(conn, "failure_reason_counts", {})),
                truncation_counts=dict(self._get_meta(conn, "truncation_counts", {})),
                full_render_system_histogram=dict(
                    self._get_meta(conn, "full_render_system_histogram", {})
                ),
                accepted_system_histogram=dict(
                    self._get_meta(conn, "accepted_system_histogram", {})
                ),
                truncated_output_system_histogram=dict(
                    self._get_meta(conn, "truncated_output_system_histogram", {})
                ),
                preferred_5_6_counts=dict(self._get_meta(conn, "preferred_5_6_counts", {})),
                bottom_whitespace_px_histogram=dict(
                    self._get_meta(conn, "bottom_whitespace_px_histogram", {})
                ),
                top_whitespace_px_histogram=dict(
                    self._get_meta(conn, "top_whitespace_px_histogram", {})
                ),
                content_height_px_histogram=dict(
                    self._get_meta(conn, "content_height_px_histogram", {})
                ),
                terminal_timeout_crash_artifacts=int(
                    self._get_meta(conn, "terminal_timeout_crash_artifacts", 0)
                ),
                terminal_process_expired_crash_artifacts=int(
                    self._get_meta(conn, "terminal_process_expired_crash_artifacts", 0)
                ),
                requested_target_bucket_histogram=dict(
                    self._get_meta(conn, "requested_target_bucket_histogram", {})
                ),
                candidate_hit_counts=dict(self._get_meta(conn, "candidate_hit_counts", {})),
                retry_counts=dict(self._get_meta(conn, "retry_counts", {})),
                quarantined_sources=tuple(self._get_meta(conn, "quarantined_sources", [])),
                augmentation_outcome_counts=dict(
                    self._get_meta(conn, "augmentation_outcome_counts", {})
                ),
                augmentation_band_counts=dict(
                    self._get_meta(conn, "augmentation_band_counts", {})
                ),
                augmentation_branch_counts=dict(
                    self._get_meta(conn, "augmentation_branch_counts", {})
                ),
                final_geometry_counts=dict(self._get_meta(conn, "final_geometry_counts", {})),
                oob_failure_reason_counts=dict(
                    self._get_meta(conn, "oob_failure_reason_counts", {})
                ),
                outer_gate_failure_reason_counts=dict(
                    self._get_meta(conn, "outer_gate_failure_reason_counts", {})
                ),
            )

    def _write_pending_shard(self, pending_rows: list[dict[str, Any]]) -> dict[str, Any]:
        shard_idx = self._next_shard_index()
        temp_filename = f"shard-{shard_idx:06d}.arrow"
        shard_path = self.staged_shards_dir / temp_filename
        writer = ArrowWriter(path=str(shard_path), features=self.features)
        for row in pending_rows:
            writer.write(row)
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

    def _connect(self) -> sqlite3.Connection:
        self.run_context.resume_dir.mkdir(parents=True, exist_ok=True)
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


def write_run_info(
    *,
    run_context: RunContext,
    recipe: ProductionRecipe,
    input_dirs: tuple[str, ...],
    output_dir: str,
    num_workers: int,
    target_samples: int,
    max_attempts: int,
    base_seed: int,
    resume_mode: str,
    failure_policy: dict[str, int | str] | None = None,
    capture_verovio_diagnostics: bool = True,
    quarantine_in: str | None = None,
    quarantine_out: str | None = None,
    system_balance: dict[str, object] | None = None,
    runtime_seconds: dict[str, float] | None = None,
    layout_summary: dict[str, Any] | None = None,
    snapshot: RuntimeSnapshot | ResumeSnapshot | None = None,
    finalization: dict[str, Any] | None = None,
    auto_quarantined_source_count: int | None = None,
    invalid_source_examples: list[dict[str, Any]] | None = None,
) -> None:
    serialized_finalization = finalization
    if finalization is not None:
        serialized_finalization = dict(finalization)
        if "snapshot" in serialized_finalization and serialized_finalization["snapshot"] is not None:
            serialized_finalization["snapshot"] = asdict(serialized_finalization["snapshot"])
    payload = {
        "run_id": run_context.run_id,
        "run_started_at": run_context.run_started_at,
        "input_dirs": list(input_dirs),
        "output_dir": output_dir,
        "num_workers": num_workers,
        "target_samples": target_samples,
        "max_attempts": max_attempts,
        "base_seed": base_seed,
        "resume_mode": resume_mode,
        "failure_policy": failure_policy,
        "capture_verovio_diagnostics": capture_verovio_diagnostics,
        "quarantine_in": quarantine_in,
        "quarantine_out": quarantine_out,
        "verovio_events_path": str(run_context.verovio_events_path),
        "failure_events_path": str(run_context.failure_events_path),
        "augmentation_events_path": str(run_context.augmentation_events_path),
        "augmentation_previews_dir": str(run_context.augmentation_previews_dir),
        "system_balance": system_balance,
        "recipe_version": recipe.version,
        "runtime_seconds": runtime_seconds,
        "layout_summary": layout_summary,
        "snapshot": asdict(snapshot) if snapshot is not None else None,
        "finalization": serialized_finalization,
        "auto_quarantined_source_count": auto_quarantined_source_count,
        "invalid_source_examples": invalid_source_examples,
    }
    write_json(run_context.info_path, payload)
