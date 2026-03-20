from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import datasets
import pytest
from PIL import Image

from scripts.dataset_generation.dataset_generation.base import GenerationStats
from scripts.dataset_generation.dataset_generation.config import GenerationRunConfig
from scripts.dataset_generation.dataset_generation.service import run_generation


def _jpeg_bytes() -> bytes:
    image = Image.new("RGB", (8, 8), color=(255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


@dataclass
class _ObservedFileState:
    next_variant_idx: int
    attempted: int
    successful: int
    failed: int
    quarantined: bool
    early_stopped: bool
    total_variants: int


class _FakeStreamingGenerator:
    fail_after_accepts: int | None = None

    def __init__(
        self,
        *,
        kern_dirs,
        resume_state=None,
        resume_observer=None,
        deterministic_seed_salt=None,
        **_kwargs,
    ) -> None:
        self.file_paths = [Path(kern_dirs[0]) / "000.krn", Path(kern_dirs[0]) / "001.krn"]
        self.resume_state = resume_state
        self.resume_observer = resume_observer
        self.deterministic_seed_salt = deterministic_seed_salt
        self.stats = GenerationStats(successful=resume_state.successful_samples if resume_state else 0)
        self.variant_policy_summary = {
            "policy": "fixed",
            "enabled": False,
            "mean_variants_per_file": 1.0,
            "total_available_tasks": 2,
        }
        self.prefilter_summary = {
            "min_non_empty_lines": None,
            "max_non_empty_lines": None,
            "min_measure_count": None,
            "max_measure_count": None,
            "original_file_count": 2,
            "retained_file_count": 2,
            "filtered_out_file_count": 0,
        }
        self.last_quarantine_summary = {
            "files": [],
            "files_count": 0,
            "skipped_tasks": 0,
            "dropped_pending_tasks": 0,
            "preloaded_files_count": 0,
        }
        self.last_failure_summary = {
            "requested_tasks": 2,
            "submitted_tasks": 0,
            "successful_samples": self.stats.successful,
            "failed_samples_total": 0,
            "skipped_samples_total": 0,
            "accepted_target": 2,
            "max_scheduled_tasks": 2,
            "stop_condition": "running",
            "truncation": {"attempted": 0, "rescued": 0, "failed": 0},
            "failure_reason_counts": {},
            "quarantine": {
                "skipped_tasks": 0,
                "dropped_pending_tasks": 0,
                "skipped_tasks_by_reason": {},
                "dropped_pending_tasks_by_reason": {},
            },
        }
        self.last_layout_summary = {
            "accepted_samples_total": self.stats.successful,
            "systems_histogram": {"5": self.stats.successful},
            "with_known_system_count": self.stats.successful,
            "with_unknown_system_count": 0,
            "accepted_ge_6_systems": 0,
            "accepted_ge_6_systems_rate": 0.0,
            "target_system_band": None,
            "accepted_in_target_band": 0,
            "accepted_in_target_band_rate": 0.0,
            "bottom_whitespace_ratio_stats": {"count": self.stats.successful, "mean": 0.2},
            "vertical_fill_ratio_stats": {"count": self.stats.successful, "mean": 0.7},
        }
        self.last_scheduler_summary = {
            "version": 1,
            "stop_condition": "running",
            "accepted_target": 2,
            "max_scheduled_tasks": 2,
            "total_available_tasks": 2,
            "submitted_tasks": self.stats.successful,
            "terminal_completed_tasks": self.stats.successful,
            "successful_samples": self.stats.successful,
            "file_counts": {
                "total": 2,
                "schedulable_remaining": 2 - self.stats.successful,
                "with_at_least_one_success": self.stats.successful,
                "early_stopped": 0,
                "quarantined": 0,
            },
            "per_file_attempts": {"count": 2},
            "per_file_successes": {"count": 2},
            "top_high_yield_files": [],
            "top_low_yield_files": [],
            "timing_by_outcome": {},
            "worker_stage_timing_by_outcome": {},
        }
        self.profile_report = None
        self._run_failure_reason_counts = {}
        self._run_truncation_counts = {}
        self._accepted_system_count_histogram = {5: self.stats.successful}
        self._accepted_unknown_system_count_successes = 0
        self._accepted_bottom_whitespace_ratios = [0.2] * self.stats.successful
        self._accepted_vertical_fill_ratios = [0.7] * self.stats.successful

    def emit_external_progress(self, _status: str) -> None:
        return None

    def _sample(self, idx: int) -> dict[str, object]:
        return {
            "image": _jpeg_bytes(),
            "transcription": f"transcription-{idx}",
            "source": f"{idx:03}.krn",
            "source_dataset": "fake",
            "source_split": "train",
            "sample_id": f"piece_{idx}__v0",
            "curation_stage": "synthetic",
            "source_domain": "synth",
            "actual_system_count": 5,
            "truncation_applied": False,
            "truncation_ratio": 0.0,
            "render_layout_profile": "default",
            "bottom_whitespace_ratio": 0.2,
            "vertical_fill_ratio": 0.7,
            "source_non_empty_line_count": 12,
            "source_measure_count": 4,
        }

    def _observe_success(self, idx: int) -> dict[str, object]:
        sample = self._sample(idx)
        file_path = self.file_paths[idx]
        state = _ObservedFileState(
            next_variant_idx=1,
            attempted=1,
            successful=1,
            failed=0,
            quarantined=False,
            early_stopped=False,
            total_variants=1,
        )
        if self.resume_observer is not None:
            self.resume_observer.observe_terminal_task(
                generator=self,
                file_path=file_path,
                file_state=state,
                sample=sample,
            )
        self.stats.successful += 1
        self.last_failure_summary["submitted_tasks"] = self.stats.successful
        self.last_failure_summary["successful_samples"] = self.stats.successful
        self.last_layout_summary["accepted_samples_total"] = self.stats.successful
        self.last_layout_summary["systems_histogram"] = {"5": self.stats.successful}
        self.last_layout_summary["with_known_system_count"] = self.stats.successful
        self.last_layout_summary["bottom_whitespace_ratio_stats"] = {
            "count": self.stats.successful,
            "mean": 0.2,
        }
        self.last_layout_summary["vertical_fill_ratio_stats"] = {
            "count": self.stats.successful,
            "mean": 0.7,
        }
        self.last_scheduler_summary["submitted_tasks"] = self.stats.successful
        self.last_scheduler_summary["terminal_completed_tasks"] = self.stats.successful
        self.last_scheduler_summary["successful_samples"] = self.stats.successful
        self.last_scheduler_summary["file_counts"]["with_at_least_one_success"] = self.stats.successful
        self._accepted_system_count_histogram = {5: self.stats.successful}
        self._accepted_bottom_whitespace_ratios = [0.2] * self.stats.successful
        self._accepted_vertical_fill_ratios = [0.7] * self.stats.successful
        return sample

    def generate(self, num_samples: int, target_accepted_samples=None, max_scheduled_tasks=None):
        del num_samples, target_accepted_samples, max_scheduled_tasks
        start_idx = self.resume_state.successful_samples if self.resume_state else 0
        accepted_this_session = 0
        for idx in range(start_idx, 2):
            sample = self._observe_success(idx)
            accepted_this_session += 1
            yield sample
            if (
                self.resume_state is None
                and self.fail_after_accepts is not None
                and accepted_this_session >= self.fail_after_accepts
            ):
                raise RuntimeError("simulated interruption")
        self.last_failure_summary["stop_condition"] = "accepted_target_reached"
        self.last_scheduler_summary["stop_condition"] = "accepted_target_reached"


def _build_config(tmp_path: Path, *, output_dir: Path, **kwargs) -> GenerationRunConfig:
    kern_dir = tmp_path / "kern"
    kern_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(2):
        (kern_dir / f"{idx:03}.krn").write_text("4c\n*-\n", encoding="utf-8")
    return GenerationRunConfig(
        kern_dirs=(str(kern_dir),),
        output_dir=str(output_dir),
        target_accepted_samples=2,
        max_scheduled_tasks=2,
        quiet=True,
        **kwargs,
    )


def test_run_generation_finalizes_loadable_dataset(monkeypatch, tmp_path):
    output_dir = tmp_path / "datasets" / "train"
    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.service.FileDataGenerator",
        _FakeStreamingGenerator,
    )
    _FakeStreamingGenerator.fail_after_accepts = None

    result = run_generation(_build_config(tmp_path, output_dir=output_dir))

    dataset = datasets.Dataset.load_from_disk(str(output_dir))
    assert result.total_samples == 2
    assert len(dataset) == 2
    assert dataset["sample_id"] == ["piece_0__v0", "piece_1__v0"]
    assert not (output_dir / "INCOMPLETE").exists()
    assert (output_dir / "state.json").exists()
    assert (output_dir / ".resume" / "manifest.sqlite").exists()


def test_run_generation_resumes_incomplete_dataset(monkeypatch, tmp_path):
    output_dir = tmp_path / "datasets" / "train"
    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.service.FileDataGenerator",
        _FakeStreamingGenerator,
    )
    _FakeStreamingGenerator.fail_after_accepts = 1
    config = _build_config(tmp_path, output_dir=output_dir)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_generation(config)

    assert (output_dir / "INCOMPLETE").exists()
    with pytest.raises(FileNotFoundError):
        datasets.Dataset.load_from_disk(str(output_dir))

    _FakeStreamingGenerator.fail_after_accepts = None
    result = run_generation(config)

    dataset = datasets.Dataset.load_from_disk(str(output_dir))
    assert result.total_samples == 2
    assert len(dataset) == 2
    assert sorted(dataset["sample_id"]) == ["piece_0__v0", "piece_1__v0"]
    assert not (output_dir / "INCOMPLETE").exists()


def test_run_generation_rejects_resume_config_fingerprint_mismatch(monkeypatch, tmp_path):
    output_dir = tmp_path / "datasets" / "train"
    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.service.FileDataGenerator",
        _FakeStreamingGenerator,
    )
    _FakeStreamingGenerator.fail_after_accepts = 1
    base_config = _build_config(tmp_path, output_dir=output_dir)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_generation(base_config)

    changed_config = _build_config(
        tmp_path,
        output_dir=output_dir,
        render_pedals_probability=0.9,
    )
    _FakeStreamingGenerator.fail_after_accepts = None
    with pytest.raises(RuntimeError, match="config fingerprint changed"):
        run_generation(changed_config)
