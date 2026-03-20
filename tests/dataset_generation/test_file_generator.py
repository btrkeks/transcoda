from collections import Counter
from pathlib import Path
from concurrent.futures import TimeoutError

import pytest

from scripts.dataset_generation.dataset_generation.file_generator import FileDataGenerator
from scripts.dataset_generation.dataset_generation.worker_models import (
    PROFILE_STAGE_NAMES,
    SampleFailure,
    SampleProfile,
    SampleSuccess,
    WorkerInitConfig,
)


class _FakeFuture:
    def __init__(self, value):
        self._value = value

    def done(self):
        return True

    def result(self):
        return self._value


class _TimeoutOnceFuture:
    def __init__(self, outcome, *, should_timeout: bool = False):
        self._outcome = outcome
        self._should_timeout = should_timeout

    def done(self):
        return True

    def result(self, timeout=None):
        if self._should_timeout:
            raise TimeoutError()
        return self._outcome


class _NonBlockingDelayFuture:
    def __init__(self, outcome, *, nonblocking_pending: bool = False):
        self._outcome = outcome
        self._pending_probes = 1 if nonblocking_pending else 0

    def done(self):
        return self._pending_probes == 0

    def result(self, timeout=None):
        if timeout == 0.0 and self._pending_probes > 0:
            self._pending_probes -= 1
            raise TimeoutError()
        return self._outcome


class _InterruptFuture:
    def done(self):
        return True

    def result(self, timeout=None):
        raise KeyboardInterrupt()


class _OOMOnMapPool:
    def __init__(self, *args, **kwargs):
        self._scheduled = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def map(self, fn, iterable, timeout=None):
        queued = list(iterable)
        if len(queued) >= 50:
            raise MemoryError("simulated OOM from queueing too many tasks")
        return _FakeFuture(iter(()))

    def schedule(self, fn, args=(), timeout=None):
        self._scheduled += 1
        file_path = Path(args[0])
        return _FakeFuture((b"jpeg-bytes", "transcription", file_path.name))


class _CapturingPool:
    last_instance = None

    def __init__(self, *args, **kwargs):
        self.scheduled_args = []
        self.initargs = kwargs.get("initargs")
        _CapturingPool.last_instance = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def schedule(self, fn, args=(), timeout=None):
        self.scheduled_args.append(args)
        file_path = Path(args[0])
        return _FakeFuture((b"jpeg-bytes", "transcription", file_path.name))


class _YieldAwarePool:
    last_instance = None

    def __init__(self, *args, **kwargs):
        self.scheduled_args = []
        _YieldAwarePool.last_instance = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def schedule(self, fn, args=(), timeout=None):
        self.scheduled_args.append(args)
        file_path = Path(args[0])
        variant_idx = int(args[1])
        success = file_path.name.startswith("good")
        if success:
            return _FakeFuture(
                SampleSuccess(
                    image=b"jpeg-bytes",
                    transcription=f"tx-{file_path.name}-{variant_idx}",
                    filename=file_path.name,
                    actual_system_count=5,
                )
            )
        return _FakeFuture(
            SampleFailure(
                code="system_band_truncation_exhausted",
                filename=file_path.name,
                truncation_attempted=True,
            )
        )


class _FailureMixPool:
    def __init__(self, *args, **kwargs):
        self._calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def schedule(self, fn, args=(), timeout=None):
        self._calls += 1
        file_path = Path(args[0])
        outcomes = [
            (None, "Reject:multi_page", file_path.name),
            (None, "Reject:sparse_render:black_ratio=0.002300", file_path.name),
            (None, "Reject:render_fit:bottom_clearance", file_path.name),
            (None, "Reject:invalid_kern:rational duration", file_path.name),
            (None, "Error processing file", file_path.name),
        ]
        idx = min(self._calls - 1, len(outcomes) - 1)
        return _FakeFuture(outcomes[idx])


class _ProfileOutcomePool:
    def __init__(self, *args, **kwargs):
        self._calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def schedule(self, fn, args=(), timeout=None):
        self._calls += 1
        file_path = Path(args[0])
        base_profile = SampleProfile(
            stages_ms={
                stage_name: float(idx + self._calls)
                for idx, stage_name in enumerate(PROFILE_STAGE_NAMES)
            },
            failure_stage="render" if self._calls == 2 else None,
        )
        if self._calls == 1:
            return _FakeFuture(
                SampleSuccess(
                    image=b"jpeg-bytes",
                    transcription="transcription",
                    filename=file_path.name,
                    profile=base_profile,
                )
            )
        return _FakeFuture(
            SampleFailure(
                code="render_fit",
                filename=file_path.name,
                detail="bottom_clearance",
                profile=base_profile,
            )
        )


class _RetryThenSuccessPool:
    def __init__(self, *args, **kwargs):
        self._attempts: dict[tuple[str, int], int] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def schedule(self, fn, args=(), timeout=None):
        file_path = Path(args[0])
        variant_idx = int(args[1])
        key = (str(file_path), variant_idx)
        attempt = self._attempts.get(key, 0)
        self._attempts[key] = attempt + 1
        if attempt == 0:
            return _TimeoutOnceFuture(None, should_timeout=True)
        return _TimeoutOnceFuture((b"jpeg-bytes", "transcription", file_path.name))


class _NonBlockingOrderPool:
    def __init__(self, *args, **kwargs):
        self._calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def schedule(self, fn, args=(), timeout=None):
        self._calls += 1
        file_path = Path(args[0])
        transcription = f"tx-{file_path.name}"
        if self._calls == 1:
            return _NonBlockingDelayFuture(
                (b"jpeg-bytes", transcription, file_path.name),
                nonblocking_pending=True,
            )
        return _NonBlockingDelayFuture((b"jpeg-bytes", transcription, file_path.name))


class _LayoutSummaryPool:
    def __init__(self, *args, **kwargs):
        self._calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def schedule(self, fn, args=(), timeout=None):
        self._calls += 1
        file_path = Path(args[0])
        outcomes = [
            SampleSuccess(
                image=b"jpeg-bytes",
                transcription="tx-0",
                filename=file_path.name,
                actual_system_count=6,
                bottom_whitespace_ratio=0.12,
                vertical_fill_ratio=0.78,
            ),
            SampleSuccess(
                image=b"jpeg-bytes",
                transcription="tx-1",
                filename=file_path.name,
                actual_system_count=4,
                bottom_whitespace_ratio=0.24,
                vertical_fill_ratio=0.66,
            ),
            SampleSuccess(
                image=b"jpeg-bytes",
                transcription="tx-2",
                filename=file_path.name,
                actual_system_count=6,
                bottom_whitespace_ratio=0.18,
                vertical_fill_ratio=0.72,
            ),
        ]
        idx = min(self._calls - 1, len(outcomes) - 1)
        return _FakeFuture(outcomes[idx])


class _LegacySuccessOnlyPool:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def schedule(self, fn, args=(), timeout=None):
        file_path = Path(args[0])
        return _FakeFuture((b"jpeg-bytes", "transcription", file_path.name))


class _InterruptPool:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def schedule(self, fn, args=(), timeout=None):
        return _InterruptFuture()


class _TruncationOutcomePool:
    def __init__(self, *args, **kwargs):
        self._calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def schedule(self, fn, args=(), timeout=None):
        self._calls += 1
        file_path = Path(args[0])
        if self._calls == 1:
            return _FakeFuture(
                SampleSuccess(
                    image=b"jpeg-bytes",
                    transcription="tx-truncated",
                    filename=file_path.name,
                    truncation_applied=True,
                    truncation_ratio=0.67,
                )
            )
        return _FakeFuture(
            SampleFailure(
                code="multi_page",
                filename=file_path.name,
                truncation_attempted=True,
            )
        )


class _TargetBandTruncationPool:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def schedule(self, fn, args=(), timeout=None):
        file_path = Path(args[0])
        return _FakeFuture(
            SampleSuccess(
                image=b"jpeg-bytes",
                transcription="tx-truncated",
                filename=file_path.name,
                actual_system_count=6,
                truncation_applied=True,
                truncation_ratio=0.75,
            )
        )


class _TargetBandRejectPool:
    def __init__(self, *args, **kwargs):
        self._calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def schedule(self, fn, args=(), timeout=None):
        self._calls += 1
        file_path = Path(args[0])
        return _FakeFuture(
            SampleSuccess(
                image=b"jpeg-bytes",
                transcription=f"tx-{self._calls}",
                filename=file_path.name,
                actual_system_count=4,
            )
        )


class _FailedTruncationPool:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def schedule(self, fn, args=(), timeout=None):
        file_path = Path(args[0])
        return _FakeFuture(
            SampleFailure(
                code="system_band_truncation_exhausted",
                filename=file_path.name,
                detail="target_band=5-6",
                truncation_attempted=True,
            )
        )


def _write_kern_with_line_count(path: Path, line_count: int) -> None:
    path.write_text("".join("4c\n" for _ in range(line_count)), encoding="utf-8")


def _write_kern_with_measure_count(path: Path, measure_count: int) -> None:
    lines = ["*clefG2", "*M4/4"]
    for idx in range(1, measure_count + 1):
        lines.append(f"={idx}")
        lines.append("4c")
    lines.append("*-")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_generate_avoids_large_map_queue(monkeypatch, tmp_path):
    # 60 files would trigger the simulated map OOM if map-based queueing is used.
    for i in range(60):
        (tmp_path / f"{i:03}.krn").write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _OOMOnMapPool,
    )

    generator = FileDataGenerator(kern_dirs=tmp_path, num_workers=2)
    samples = list(generator.generate(num_samples=60))

    assert len(samples) == 60
    assert generator.stats.successful == 60
    assert generator.stats.total_failures == 0


def test_generate_expands_variants_per_file(monkeypatch, tmp_path):
    for i in range(2):
        (tmp_path / f"{i:03}.krn").write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _CapturingPool,
    )

    generator = FileDataGenerator(kern_dirs=tmp_path, num_workers=2, variants_per_file=3)
    samples = list(generator.generate(num_samples=10))

    assert len(samples) == 6
    assert generator.stats.successful == 6
    assert _CapturingPool.last_instance is not None
    variant_indices = {int(args[1]) for args in _CapturingPool.last_instance.scheduled_args}
    assert variant_indices == {0, 1, 2}
    assert samples[0]["source_dataset"] == tmp_path.parent.name
    assert samples[0]["source_split"] == "train"
    assert samples[0]["curation_stage"] == "synthetic"
    assert samples[0]["source_domain"] == "synth"
    assert samples[0]["render_layout_profile"] == "default"
    assert samples[0]["sample_id"].endswith("__v0")


def test_generate_adaptive_variants_by_line_count(monkeypatch, tmp_path):
    _write_kern_with_line_count(tmp_path / "000.krn", 100)
    _write_kern_with_line_count(tmp_path / "001.krn", 300)
    _write_kern_with_line_count(tmp_path / "002.krn", 900)
    _write_kern_with_line_count(tmp_path / "003.krn", 1300)

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _CapturingPool,
    )

    generator = FileDataGenerator(
        kern_dirs=tmp_path,
        num_workers=2,
        variants_per_file=3,
        adaptive_variants_enabled=True,
    )
    samples = list(generator.generate(num_samples=20))

    assert len(samples) == 12
    assert generator.total_available_tasks == 12
    assert generator.variant_policy_summary["enabled"] is True
    assert generator.last_failure_summary["requested_tasks"] == 12
    assert _CapturingPool.last_instance is not None
    scheduled_counts = Counter(
        Path(args[0]).name for args in _CapturingPool.last_instance.scheduled_args
    )
    assert scheduled_counts == {
        "000.krn": 1,
        "001.krn": 2,
        "002.krn": 4,
        "003.krn": 5,
    }


def test_generate_stops_at_target_accepted_samples(monkeypatch, tmp_path):
    for i in range(3):
        (tmp_path / f"{i:03}.krn").write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _CapturingPool,
    )

    generator = FileDataGenerator(kern_dirs=tmp_path, num_workers=1, variants_per_file=3)
    samples = list(generator.generate(num_samples=100, target_accepted_samples=2))

    assert len(samples) == 2
    assert generator.last_failure_summary["stop_condition"] == "accepted_target_reached"
    assert generator.last_failure_summary["accepted_target"] == 2
    assert generator.last_run_result is not None
    assert generator.last_run_result.scheduler_summary["stop_condition"] == "accepted_target_reached"


def test_generate_prefers_accepted_target_over_task_budget(monkeypatch, tmp_path):
    for i in range(2):
        (tmp_path / f"{i:03}.krn").write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _CapturingPool,
    )

    generator = FileDataGenerator(kern_dirs=tmp_path, num_workers=1, variants_per_file=5)
    samples = list(
        generator.generate(
            num_samples=100,
            target_accepted_samples=1,
            max_scheduled_tasks=6,
        )
    )

    assert len(samples) == 1
    assert generator.last_failure_summary["stop_condition"] == "accepted_target_reached"


def test_generate_bootstraps_one_variant_per_file_before_second(monkeypatch, tmp_path):
    for i in range(2):
        (tmp_path / f"{i:03}.krn").write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _CapturingPool,
    )

    generator = FileDataGenerator(kern_dirs=tmp_path, num_workers=2, variants_per_file=2)
    list(generator.generate(num_samples=4))

    assert _CapturingPool.last_instance is not None
    first_two = [(Path(args[0]).name, int(args[1])) for args in _CapturingPool.last_instance.scheduled_args[:2]]
    assert first_two == [("000.krn", 0), ("001.krn", 0)]


def test_generate_yield_aware_scheduler_deprioritizes_and_early_stops_zero_success_files(
    monkeypatch,
    tmp_path,
):
    (tmp_path / "good.krn").write_text("4c\n*-\n", encoding="utf-8")
    (tmp_path / "bad.krn").write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _YieldAwarePool,
    )

    generator = FileDataGenerator(kern_dirs=tmp_path, num_workers=1, variants_per_file=10)
    samples = list(generator.generate(num_samples=100))

    assert len(samples) == 10
    assert _YieldAwarePool.last_instance is not None
    scheduled_names = [Path(args[0]).name for args in _YieldAwarePool.last_instance.scheduled_args]
    assert scheduled_names[:2] == ["bad.krn", "good.krn"]
    assert scheduled_names[3] == "good.krn"
    assert scheduled_names.count("bad.krn") == 8
    assert scheduled_names.count("good.krn") == 10
    assert generator.last_scheduler_summary["file_counts"]["early_stopped"] == 1


def test_generate_prefilters_by_min_non_empty_lines(monkeypatch, tmp_path):
    _write_kern_with_line_count(tmp_path / "000.krn", 100)
    _write_kern_with_line_count(tmp_path / "001.krn", 513)
    _write_kern_with_line_count(tmp_path / "002.krn", 900)

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _CapturingPool,
    )

    generator = FileDataGenerator(
        kern_dirs=tmp_path,
        num_workers=2,
        variants_per_file=1,
        prefilter_min_non_empty_lines=513,
    )
    samples = list(generator.generate(num_samples=10))

    assert len(samples) == 2
    assert generator.prefilter_summary == {
        "min_non_empty_lines": 513,
        "max_non_empty_lines": None,
        "min_measure_count": None,
        "max_measure_count": None,
        "original_file_count": 3,
        "retained_file_count": 2,
        "filtered_out_file_count": 1,
    }
    assert _CapturingPool.last_instance is not None
    scheduled = {Path(args[0]).name for args in _CapturingPool.last_instance.scheduled_args}
    assert scheduled == {"001.krn", "002.krn"}


def test_generate_prefilters_by_non_empty_line_band(monkeypatch, tmp_path):
    _write_kern_with_line_count(tmp_path / "000.krn", 100)
    _write_kern_with_line_count(tmp_path / "001.krn", 320)
    _write_kern_with_line_count(tmp_path / "002.krn", 430)
    _write_kern_with_line_count(tmp_path / "003.krn", 700)

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _CapturingPool,
    )

    generator = FileDataGenerator(
        kern_dirs=tmp_path,
        num_workers=2,
        variants_per_file=1,
        prefilter_min_non_empty_lines=300,
        prefilter_max_non_empty_lines=450,
    )
    samples = list(generator.generate(num_samples=10))

    assert len(samples) == 2
    assert generator.prefilter_summary == {
        "min_non_empty_lines": 300,
        "max_non_empty_lines": 450,
        "min_measure_count": None,
        "max_measure_count": None,
        "original_file_count": 4,
        "retained_file_count": 2,
        "filtered_out_file_count": 2,
    }
    assert _CapturingPool.last_instance is not None
    scheduled = {Path(args[0]).name for args in _CapturingPool.last_instance.scheduled_args}
    assert scheduled == {"001.krn", "002.krn"}


def test_generate_prefilters_by_measure_count_band(monkeypatch, tmp_path):
    _write_kern_with_measure_count(tmp_path / "000.krn", 8)
    _write_kern_with_measure_count(tmp_path / "001.krn", 14)
    _write_kern_with_measure_count(tmp_path / "002.krn", 18)
    _write_kern_with_measure_count(tmp_path / "003.krn", 26)

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _CapturingPool,
    )

    generator = FileDataGenerator(
        kern_dirs=tmp_path,
        num_workers=2,
        variants_per_file=1,
        prefilter_min_measure_count=12,
        prefilter_max_measure_count=20,
    )
    samples = list(generator.generate(num_samples=10))

    assert len(samples) == 2
    assert generator.prefilter_summary == {
        "min_non_empty_lines": None,
        "max_non_empty_lines": None,
        "min_measure_count": 12,
        "max_measure_count": 20,
        "original_file_count": 4,
        "retained_file_count": 2,
        "filtered_out_file_count": 2,
    }
    assert _CapturingPool.last_instance is not None
    scheduled = {Path(args[0]).name for args in _CapturingPool.last_instance.scheduled_args}
    assert scheduled == {"001.krn", "002.krn"}


def test_generate_tracks_truncation_rescue_counters(monkeypatch, tmp_path):
    for i in range(2):
        (tmp_path / f"{i:03}.krn").write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _TruncationOutcomePool,
    )

    generator = FileDataGenerator(kern_dirs=tmp_path, num_workers=2, variants_per_file=1)
    samples = list(generator.generate(num_samples=2))

    assert len(samples) == 1
    assert generator.last_failure_summary["truncation"] == {
        "attempted": 2,
        "rescued": 1,
        "failed": 1,
    }


def test_generate_target_band_accepts_only_rescued_in_band(monkeypatch, tmp_path):
    (tmp_path / "000.krn").write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _TargetBandTruncationPool,
    )

    generator = FileDataGenerator(
        kern_dirs=tmp_path,
        num_workers=1,
        variants_per_file=1,
        target_min_systems=5,
        target_max_systems=6,
    )
    samples = list(generator.generate(num_samples=1))

    assert len(samples) == 1
    assert samples[0]["actual_system_count"] == 6
    assert samples[0]["truncation_applied"] is True
    assert samples[0]["truncation_ratio"] == pytest.approx(0.75)
    assert generator.last_layout_summary["target_system_band"] == {"min": 5, "max": 6}
    assert generator.last_layout_summary["accepted_in_target_band"] == 1
    assert generator.last_layout_summary["accepted_in_target_band_rate"] == pytest.approx(1.0)


def test_generate_target_band_rejects_underfull_success(monkeypatch, tmp_path):
    for i in range(2):
        (tmp_path / f"{i:03}.krn").write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _TargetBandRejectPool,
    )

    generator = FileDataGenerator(
        kern_dirs=tmp_path,
        num_workers=1,
        variants_per_file=1,
        target_min_systems=5,
        target_max_systems=6,
    )
    samples = list(generator.generate(num_samples=2))

    assert samples == []
    assert generator.last_failure_summary["failure_reason_counts"]["system_band_rejected"] == 2
    assert generator.last_failure_summary["failure_reason_counts"]["system_band_below_min"] == 2


def test_generate_counts_failed_truncation_for_system_band_rejection(monkeypatch, tmp_path):
    (tmp_path / "000.krn").write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _FailedTruncationPool,
    )

    generator = FileDataGenerator(
        kern_dirs=tmp_path,
        num_workers=1,
        variants_per_file=1,
        target_min_systems=5,
        target_max_systems=6,
    )
    samples = list(generator.generate(num_samples=1))

    assert samples == []
    assert generator.last_failure_summary["failure_reason_counts"]["system_band_rejected"] == 1
    assert (
        generator.last_failure_summary["failure_reason_counts"][
            "system_band_truncation_exhausted"
        ]
        == 1
    )
    assert generator._run_truncation_counts["attempted"] == 1
    assert generator._run_truncation_counts["failed"] == 1


def test_generate_tracks_structured_rejection_reasons(monkeypatch, tmp_path):
    for i in range(5):
        (tmp_path / f"{i:03}.krn").write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _FailureMixPool,
    )

    generator = FileDataGenerator(kern_dirs=tmp_path, num_workers=2, variants_per_file=1)
    samples = list(generator.generate(num_samples=5))

    assert samples == []
    assert generator.stats.overflows == 1
    assert generator.stats.rejected_sparse == 1
    assert generator.stats.rejected_render_fit == 1
    assert generator.stats.invalid == 1
    assert generator.stats.errors == 1


def test_generate_collects_profile_report_when_enabled(monkeypatch, tmp_path):
    for i in range(2):
        (tmp_path / f"{i:03}.krn").write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _ProfileOutcomePool,
    )

    generator = FileDataGenerator(
        kern_dirs=tmp_path,
        num_workers=2,
        variants_per_file=1,
        profile_enabled=True,
        profile_log_every=1,
        profile_capture_per_sample=True,
    )
    samples = list(generator.generate(num_samples=2))

    assert len(samples) == 1
    assert generator.profile_report is not None
    report = generator.profile_report
    assert report["outcome_counts"]["success"] == 1
    assert report["outcome_counts"]["failure"] == 1
    assert report["failure_code_counts"]["render_fit"] == 1
    assert report["worker_stage_stats"]["read_kern_ms"]["count"] == 2
    assert isinstance(report["per_sample"], list)
    assert len(report["per_sample"]) == 2
    assert generator.last_run_result is not None
    assert generator.last_run_result.stats["successful"] == 1
    assert generator.last_run_result.failure_summary["failed_samples_total"] == 1


def test_generate_retries_once_after_timeout_and_recovers(monkeypatch, tmp_path):
    (tmp_path / "000.krn").write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _RetryThenSuccessPool,
    )

    generator = FileDataGenerator(
        kern_dirs=tmp_path,
        num_workers=1,
        variants_per_file=1,
        profile_enabled=True,
    )
    samples = list(generator.generate(num_samples=1))

    assert len(samples) == 1
    assert generator.stats.successful == 1
    assert generator.stats.timeouts == 0
    assert generator.profile_report is not None
    assert generator.profile_report["retry_counts"]["timeout"] == 1


def test_generate_timeout_fail_fast_quarantines_file(monkeypatch, tmp_path):
    (tmp_path / "000.krn").write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _RetryThenSuccessPool,
    )

    generator = FileDataGenerator(
        kern_dirs=tmp_path,
        num_workers=1,
        variants_per_file=1,
        max_task_retries_timeout=0,
        profile_enabled=True,
    )
    samples = list(generator.generate(num_samples=1))

    assert samples == []
    assert generator.stats.successful == 0
    assert generator.stats.timeouts == 1
    assert generator.last_quarantine_summary["files_count"] == 1
    assert str(tmp_path / "000.krn") in generator.last_quarantine_summary["files"]
    assert generator.profile_report is not None
    assert generator.profile_report["retry_counts"].get("timeout", 0) == 0


def test_generate_applies_preloaded_quarantine(monkeypatch, tmp_path):
    for i in range(3):
        (tmp_path / f"{i:03}.krn").write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _CapturingPool,
    )

    preloaded = tmp_path / "001.krn"
    generator = FileDataGenerator(
        kern_dirs=tmp_path,
        num_workers=2,
        variants_per_file=1,
        initial_quarantined_files=[preloaded],
    )
    samples = list(generator.generate(num_samples=3))

    assert len(samples) == 2
    assert _CapturingPool.last_instance is not None
    scheduled = {str(Path(args[0])) for args in _CapturingPool.last_instance.scheduled_args}
    assert str(preloaded) not in scheduled
    assert generator.last_quarantine_summary["preloaded_files_count"] == 1
    assert generator.last_quarantine_summary["files_count"] == 1
    assert str(preloaded) in generator.last_quarantine_summary["files"]


def test_generate_prefers_ready_future_over_fifo_blocking(monkeypatch, tmp_path):
    for i in range(2):
        (tmp_path / f"{i:03}.krn").write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _NonBlockingOrderPool,
    )

    generator = FileDataGenerator(kern_dirs=tmp_path, num_workers=2, variants_per_file=1)
    samples = list(generator.generate(num_samples=2))

    assert len(samples) == 2
    # First scheduled task is nonblocking-pending, so second task should complete first.
    assert samples[0]["transcription"] == "tx-001.krn"


def test_generate_tracks_layout_summary_histogram_for_structured_success(monkeypatch, tmp_path):
    for i in range(3):
        (tmp_path / f"{i:03}.krn").write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _LayoutSummaryPool,
    )

    generator = FileDataGenerator(kern_dirs=tmp_path, num_workers=2, variants_per_file=1)
    samples = list(generator.generate(num_samples=3))

    assert len(samples) == 3
    assert generator.stats.successful == 3
    assert generator.last_layout_summary["version"] == 1
    assert generator.last_layout_summary["population"] == "accepted_samples"
    assert generator.last_layout_summary["accepted_samples_total"] == 3
    assert generator.last_layout_summary["with_known_system_count"] == 3
    assert generator.last_layout_summary["with_unknown_system_count"] == 0
    assert generator.last_layout_summary["systems_histogram"] == {"4": 1, "6": 2}
    assert generator.last_layout_summary["accepted_ge_6_systems"] == 2
    assert generator.last_layout_summary["accepted_ge_6_systems_rate"] == pytest.approx(2 / 3)
    assert generator.last_layout_summary["target_system_band"] is None
    assert generator.last_layout_summary["accepted_in_target_band"] == 0
    assert generator.last_layout_summary["bottom_whitespace_ratio_stats"]["mean"] == pytest.approx(0.18)
    assert generator.last_layout_summary["bottom_whitespace_ratio_stats"]["p50"] == pytest.approx(0.18)
    assert generator.last_layout_summary["vertical_fill_ratio_stats"]["mean"] == pytest.approx(0.72)
    assert generator.last_run_result is not None
    assert generator.last_run_result.layout_summary["accepted_ge_6_systems"] == 2


def test_generate_tracks_layout_summary_unknown_for_legacy_success(monkeypatch, tmp_path):
    for i in range(2):
        (tmp_path / f"{i:03}.krn").write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _LegacySuccessOnlyPool,
    )

    generator = FileDataGenerator(kern_dirs=tmp_path, num_workers=2, variants_per_file=1)
    samples = list(generator.generate(num_samples=2))

    assert len(samples) == 2
    assert generator.stats.successful == 2
    assert generator.last_layout_summary["accepted_samples_total"] == 2
    assert generator.last_layout_summary["with_known_system_count"] == 0
    assert generator.last_layout_summary["with_unknown_system_count"] == 2
    assert generator.last_layout_summary["systems_histogram"] == {}
    assert generator.last_layout_summary["accepted_ge_6_systems"] == 0
    assert generator.last_layout_summary["accepted_ge_6_systems_rate"] == 0.0
    assert generator.last_layout_summary["accepted_in_target_band"] == 0
    assert generator.last_layout_summary["bottom_whitespace_ratio_stats"]["count"] == 0
    assert generator.last_layout_summary["vertical_fill_ratio_stats"]["count"] == 0


def test_generate_emits_progress_log_and_heartbeat(monkeypatch, tmp_path, capsys):
    for i in range(2):
        (tmp_path / f"{i:03}.krn").write_text("4c\n*-\n", encoding="utf-8")

    progress_path = tmp_path / "progress.json"
    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _LegacySuccessOnlyPool,
    )

    generator = FileDataGenerator(
        kern_dirs=tmp_path,
        num_workers=2,
        variants_per_file=1,
        progress_enabled=True,
        progress_update_interval_seconds=1,
        progress_path=progress_path,
        progress_run_id="run-1",
        progress_output_dir="data/datasets/train",
        progress_run_artifacts_dir="data/datasets/_runs/train/run-1",
    )
    samples = list(generator.generate(num_samples=2))

    assert len(samples) == 2
    output = capsys.readouterr().out
    assert "[progress]" in output
    assert "accepted=2/2" in output
    assert progress_path.exists()
    payload = progress_path.read_text(encoding="utf-8")
    assert '"status": "completed"' in payload
    assert '"accepted_samples": 2' in payload


def test_generate_writes_interrupted_progress_heartbeat(monkeypatch, tmp_path):
    (tmp_path / "000.krn").write_text("4c\n*-\n", encoding="utf-8")

    progress_path = tmp_path / "progress.json"
    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _InterruptPool,
    )

    generator = FileDataGenerator(
        kern_dirs=tmp_path,
        num_workers=1,
        variants_per_file=1,
        progress_enabled=True,
        progress_update_interval_seconds=1,
        progress_path=progress_path,
        progress_run_id="run-2",
        progress_output_dir="data/datasets/train",
        progress_run_artifacts_dir="data/datasets/_runs/train/run-2",
    )

    with pytest.raises(KeyboardInterrupt):
        list(generator.generate(num_samples=1))

    assert progress_path.exists()
    payload = progress_path.read_text(encoding="utf-8")
    assert '"status": "interrupted"' in payload


def test_generator_passes_render_pedal_settings_to_worker_init(monkeypatch, tmp_path):
    (tmp_path / "000.krn").write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.file_generator.ProcessPool",
        _CapturingPool,
    )

    generator = FileDataGenerator(
        kern_dirs=tmp_path,
        num_workers=2,
        render_pedals_enabled=False,
        render_pedals_probability=0.35,
        render_pedals_measures_probability=0.15,
        render_instrument_piano_enabled=True,
        render_instrument_piano_probability=0.25,
        render_sforzando_enabled=True,
        render_sforzando_probability=0.30,
        render_sforzando_per_note_probability=0.04,
        render_accent_enabled=True,
        render_accent_probability=0.12,
        render_accent_per_note_probability=0.02,
        render_tempo_enabled=True,
        render_tempo_probability=0.22,
        render_tempo_include_mm_probability=0.44,
        render_hairpins_enabled=True,
        render_hairpins_probability=0.40,
        render_hairpins_max_spans=3,
        render_dynamic_marks_enabled=True,
        render_dynamic_marks_probability=0.15,
        render_dynamic_marks_min_count=1,
        render_dynamic_marks_max_count=2,
    )
    samples = list(generator.generate(num_samples=1))

    assert len(samples) == 1
    assert _CapturingPool.last_instance is not None
    assert _CapturingPool.last_instance.initargs == (
        WorkerInitConfig(
            image_width=1050,
            image_height=None,
            augment_seed=None,
            render_pedals_enabled=False,
            render_pedals_probability=0.35,
            render_pedals_measures_probability=0.15,
            render_instrument_piano_enabled=True,
            render_instrument_piano_probability=0.25,
            render_sforzando_enabled=True,
            render_sforzando_probability=0.30,
            render_sforzando_per_note_probability=0.04,
            render_accent_enabled=True,
            render_accent_probability=0.12,
            render_accent_per_note_probability=0.02,
            render_tempo_enabled=True,
            render_tempo_probability=0.22,
            render_tempo_include_mm_probability=0.44,
            render_hairpins_enabled=True,
            render_hairpins_probability=0.40,
            render_hairpins_max_spans=3,
            render_dynamic_marks_enabled=True,
            render_dynamic_marks_probability=0.15,
            render_dynamic_marks_min_count=1,
            render_dynamic_marks_max_count=2,
            geom_x_squeeze_prob=0.45,
            geom_x_squeeze_min_scale=0.70,
            geom_x_squeeze_max_scale=0.95,
            geom_x_squeeze_apply_in_conservative=True,
            geom_x_squeeze_preview_force_scale=None,
        ),
    )


@pytest.mark.parametrize("invalid_probability", [-0.01, 1.01])
def test_generator_rejects_invalid_render_pedal_sample_probability(tmp_path, invalid_probability):
    (tmp_path / "000.krn").write_text("4c\n*-\n", encoding="utf-8")

    with pytest.raises(ValueError, match="render_pedals_probability must be in \\[0.0, 1.0\\]"):
        FileDataGenerator(
            kern_dirs=tmp_path,
            render_pedals_probability=invalid_probability,
        )


@pytest.mark.parametrize("invalid_probability", [-0.01, 1.01])
def test_generator_rejects_invalid_render_pedal_probability(tmp_path, invalid_probability):
    (tmp_path / "000.krn").write_text("4c\n*-\n", encoding="utf-8")

    with pytest.raises(ValueError, match="render_pedals_measures_probability must be in \\[0.0, 1.0\\]"):
        FileDataGenerator(
            kern_dirs=tmp_path,
            render_pedals_measures_probability=invalid_probability,
        )


@pytest.mark.parametrize("invalid_probability", [-0.01, 1.01])
def test_generator_rejects_invalid_render_piano_probability(tmp_path, invalid_probability):
    (tmp_path / "000.krn").write_text("4c\n*-\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match="render_instrument_piano_probability must be in \\[0.0, 1.0\\]"
    ):
        FileDataGenerator(
            kern_dirs=tmp_path,
            render_instrument_piano_probability=invalid_probability,
        )


@pytest.mark.parametrize("invalid_probability", [-0.01, 1.01])
def test_generator_rejects_invalid_render_sforzando_probability(tmp_path, invalid_probability):
    (tmp_path / "000.krn").write_text("4c\n*-\n", encoding="utf-8")

    with pytest.raises(ValueError, match="render_sforzando_probability must be in \\[0.0, 1.0\\]"):
        FileDataGenerator(
            kern_dirs=tmp_path,
            render_sforzando_probability=invalid_probability,
        )


@pytest.mark.parametrize("invalid_probability", [-0.01, 1.01])
def test_generator_rejects_invalid_render_sforzando_per_note_probability(tmp_path, invalid_probability):
    (tmp_path / "000.krn").write_text("4c\n*-\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match="render_sforzando_per_note_probability must be in \\[0.0, 1.0\\]"
    ):
        FileDataGenerator(
            kern_dirs=tmp_path,
            render_sforzando_per_note_probability=invalid_probability,
        )


@pytest.mark.parametrize("invalid_probability", [-0.01, 1.01])
def test_generator_rejects_invalid_render_accent_probability(tmp_path, invalid_probability):
    (tmp_path / "000.krn").write_text("4c\n*-\n", encoding="utf-8")

    with pytest.raises(ValueError, match="render_accent_probability must be in \\[0.0, 1.0\\]"):
        FileDataGenerator(
            kern_dirs=tmp_path,
            render_accent_probability=invalid_probability,
        )


@pytest.mark.parametrize("invalid_probability", [-0.01, 1.01])
def test_generator_rejects_invalid_render_accent_per_note_probability(tmp_path, invalid_probability):
    (tmp_path / "000.krn").write_text("4c\n*-\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match="render_accent_per_note_probability must be in \\[0.0, 1.0\\]"
    ):
        FileDataGenerator(
            kern_dirs=tmp_path,
            render_accent_per_note_probability=invalid_probability,
        )


@pytest.mark.parametrize("invalid_probability", [-0.01, 1.01])
def test_generator_rejects_invalid_render_tempo_probability(tmp_path, invalid_probability):
    (tmp_path / "000.krn").write_text("4c\n*-\n", encoding="utf-8")

    with pytest.raises(ValueError, match="render_tempo_probability must be in \\[0.0, 1.0\\]"):
        FileDataGenerator(
            kern_dirs=tmp_path,
            render_tempo_probability=invalid_probability,
        )


@pytest.mark.parametrize("invalid_probability", [-0.01, 1.01])
def test_generator_rejects_invalid_render_tempo_include_mm_probability(tmp_path, invalid_probability):
    (tmp_path / "000.krn").write_text("4c\n*-\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match="render_tempo_include_mm_probability must be in \\[0.0, 1.0\\]"
    ):
        FileDataGenerator(
            kern_dirs=tmp_path,
            render_tempo_include_mm_probability=invalid_probability,
        )


@pytest.mark.parametrize("invalid_probability", [-0.01, 1.01])
def test_generator_rejects_invalid_render_hairpins_probability(tmp_path, invalid_probability):
    (tmp_path / "000.krn").write_text("4c\n*-\n", encoding="utf-8")

    with pytest.raises(ValueError, match="render_hairpins_probability must be in \\[0.0, 1.0\\]"):
        FileDataGenerator(
            kern_dirs=tmp_path,
            render_hairpins_probability=invalid_probability,
        )


@pytest.mark.parametrize("invalid_max_spans", [0, -1])
def test_generator_rejects_invalid_render_hairpins_max_spans(tmp_path, invalid_max_spans):
    (tmp_path / "000.krn").write_text("4c\n*-\n", encoding="utf-8")

    with pytest.raises(ValueError, match="render_hairpins_max_spans must be >= 1"):
        FileDataGenerator(
            kern_dirs=tmp_path,
            render_hairpins_max_spans=invalid_max_spans,
        )


@pytest.mark.parametrize("invalid_probability", [-0.01, 1.01])
def test_generator_rejects_invalid_render_dynamic_marks_probability(tmp_path, invalid_probability):
    (tmp_path / "000.krn").write_text("4c\n*-\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match="render_dynamic_marks_probability must be in \\[0.0, 1.0\\]"
    ):
        FileDataGenerator(
            kern_dirs=tmp_path,
            render_dynamic_marks_probability=invalid_probability,
        )


@pytest.mark.parametrize("invalid_min_count", [0, -1])
def test_generator_rejects_invalid_render_dynamic_marks_min_count(tmp_path, invalid_min_count):
    (tmp_path / "000.krn").write_text("4c\n*-\n", encoding="utf-8")

    with pytest.raises(ValueError, match="render_dynamic_marks_min_count must be >= 1"):
        FileDataGenerator(
            kern_dirs=tmp_path,
            render_dynamic_marks_min_count=invalid_min_count,
        )


def test_generator_rejects_invalid_render_dynamic_marks_count_range(tmp_path):
    (tmp_path / "000.krn").write_text("4c\n*-\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="render_dynamic_marks_max_count must be >= render_dynamic_marks_min_count",
    ):
        FileDataGenerator(
            kern_dirs=tmp_path,
            render_dynamic_marks_min_count=2,
            render_dynamic_marks_max_count=1,
        )
