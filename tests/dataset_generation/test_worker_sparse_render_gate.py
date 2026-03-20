# ruff: noqa: E402, I001

import contextlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

try:
    import cv2  # noqa: F401
except ImportError:
    sys.modules.setdefault("cv2", types.SimpleNamespace(resize=lambda image, size: image))
sys.modules.setdefault("names", types.SimpleNamespace(get_full_name=lambda: "Test Author"))


class _DummyRandomSentence:
    def sentence(self):
        return "Test Title"


sys.modules.setdefault("wonderwords", types.SimpleNamespace(RandomSentence=_DummyRandomSentence))

from scripts.dataset_generation.dataset_generation.image_generation.types import GeneratedScore
from scripts.dataset_generation.dataset_generation import worker
from scripts.dataset_generation.dataset_generation.truncation import PrefixTruncationSpace
from scripts.dataset_generation.dataset_generation.worker_models import (
    PROFILE_STAGE_NAMES,
    SampleFailure,
    SampleSuccess,
    WorkerInitConfig,
)


def _make_rgb_image(*, height=120, width=200, black_rows=0):
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    if black_rows > 0:
        img[:black_rows, :] = 0
    return img


def test_is_sparse_render_detects_very_low_black_ratio():
    img = _make_rgb_image(height=200, width=200, black_rows=0)
    img[0, :100] = 0  # 100 / 40000 = 0.0025

    is_sparse, black_ratio = worker._is_sparse_render(img)

    assert is_sparse is True
    assert black_ratio < 0.005


def test_is_sparse_render_accepts_normal_density():
    img = _make_rgb_image(height=200, width=200, black_rows=3)  # 0.015

    is_sparse, black_ratio = worker._is_sparse_render(img)

    assert is_sparse is False
    assert black_ratio > 0.005


def test_generate_sample_from_path_outcome_uses_deterministic_sample_seed(
    monkeypatch, tmp_path
):
    kern_path = Path(tmp_path) / "seeded.krn"
    kern_path.write_text("4c\n*-\n", encoding="utf-8")

    seed_calls: list[int] = []
    monkeypatch.setattr(worker, "_file_renderer", object())
    monkeypatch.setattr(
        worker,
        "_current_worker_config",
        lambda: WorkerInitConfig(
            image_width=1050,
            deterministic_seed_salt="salt-123",
        ),
    )
    monkeypatch.setattr(worker.np.random, "seed", lambda value: seed_calls.append(int(value)))
    monkeypatch.setattr(worker, "_build_render_transcription", lambda content, config: content)
    monkeypatch.setattr(worker, "_build_label_transcription", lambda content, config: content)
    monkeypatch.setattr(
        worker,
        "_render_kern_for_sample",
        lambda *args, **kwargs: (
            _make_rgb_image(height=20, width=20, black_rows=2),
            None,
            4,
            0.1,
            0.8,
            None,
        ),
    )
    monkeypatch.setattr(
        worker,
        "_postprocess_rendered_image",
        lambda *args, **kwargs: (b"jpeg", 0.0, 0.0, None),
    )

    outcome = worker.generate_sample_from_path_outcome(
        kern_path,
        variant_idx=0,
        sample_id="piece__v0",
    )

    assert isinstance(outcome, SampleSuccess)
    assert seed_calls == [worker.derive_sample_seed(sample_id="piece__v0", salt="salt-123")]


def test_generate_sample_from_path_outcome_emits_profile_payload(monkeypatch, tmp_path):
    kern_path = Path(tmp_path) / "sample.krn"
    kern_path.write_text("4c\n*-\n", encoding="utf-8")

    dense_img = np.full((1485, 1050, 3), 255, dtype=np.uint8)
    dense_img[20:80, 20:500] = 0

    monkeypatch.setattr(worker, "_image_width", 1050)
    monkeypatch.setattr(worker, "_image_height", 1485)
    monkeypatch.setattr(worker, "_augment_seed", None)
    monkeypatch.setattr(worker, "_file_renderer", object())
    monkeypatch.setattr(worker, "_render_pedals_enabled", False)
    monkeypatch.setattr(worker, "_render_pedals_measures_probability", 0.0)
    monkeypatch.setattr(worker, "_render_instrument_piano_enabled", False)
    monkeypatch.setattr(worker, "_render_instrument_piano_probability", 0.0)
    monkeypatch.setattr(worker, "_render_sforzando_enabled", False)
    monkeypatch.setattr(worker, "_render_sforzando_probability", 0.0)
    monkeypatch.setattr(worker, "_render_sforzando_per_note_probability", 0.0)
    monkeypatch.setattr(worker, "_render_accent_enabled", False)
    monkeypatch.setattr(worker, "_render_accent_probability", 0.0)
    monkeypatch.setattr(worker, "_render_accent_per_note_probability", 0.0)
    monkeypatch.setattr(worker, "_render_tempo_enabled", False)
    monkeypatch.setattr(worker, "_render_tempo_probability", 0.0)
    monkeypatch.setattr(worker, "_render_tempo_include_mm_probability", 0.0)
    monkeypatch.setattr(worker, "_render_hairpins_enabled", False)
    monkeypatch.setattr(worker, "_render_hairpins_probability", 0.0)
    monkeypatch.setattr(worker, "_render_hairpins_max_spans", 1)
    monkeypatch.setattr(worker, "_render_dynamic_marks_enabled", False)
    monkeypatch.setattr(worker, "_render_dynamic_marks_probability", 0.0)
    monkeypatch.setattr(worker, "_render_dynamic_marks_min_count", 1)
    monkeypatch.setattr(worker, "_render_dynamic_marks_max_count", 2)
    monkeypatch.setattr(worker, "_profile_enabled", True)
    monkeypatch.setattr(worker, "_worker_config", None)
    monkeypatch.setattr(worker, "_capture_stderr_fd", lambda: contextlib.nullcontext([]))

    def fake_generate_score_with_diagnostics(kern_for_verovio, renderer, config):
        return (
            GeneratedScore(
                image=dense_img,
                transcription=kern_for_verovio,
                actual_system_count=4,
                metadata_prefix="",
            ),
            None,
        )

    monkeypatch.setattr(worker, "generate_score_with_diagnostics", fake_generate_score_with_diagnostics)

    offline_mod = types.SimpleNamespace(
        offline_augment=lambda image, **kwargs: image,
    )
    monkeypatch.setitem(
        sys.modules,
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment",
        offline_mod,
    )

    monkeypatch.setattr(worker, "_worker_config", None)
    monkeypatch.setattr(worker, "_geom_x_squeeze_prob", 0.0)
    monkeypatch.setattr(worker, "_geom_x_squeeze_min_scale", 0.8)
    monkeypatch.setattr(worker, "_geom_x_squeeze_max_scale", 0.9)
    monkeypatch.setattr(worker, "_geom_x_squeeze_apply_in_conservative", True)
    monkeypatch.setattr(worker, "_geom_x_squeeze_preview_force_scale", None)
    monkeypatch.setattr(worker, "_render_pedals_enabled", False)

    outcome = worker.generate_sample_from_path_outcome(kern_path, variant_idx=0)

    assert isinstance(outcome, SampleSuccess)
    assert outcome.profile is not None
    for stage_name in PROFILE_STAGE_NAMES:
        assert stage_name in outcome.profile.stages_ms
    assert outcome.profile.stages_ms["read_kern_ms"] > 0.0
    assert outcome.profile.stages_ms["worker_total_ms"] > 0.0


def test_generate_sample_from_path_outcome_uses_augmented_label_transcription(monkeypatch, tmp_path):
    kern_path = Path(tmp_path) / "sample.krn"
    kern_path.write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(worker, "_file_renderer", object())
    monkeypatch.setattr(
        worker,
        "_current_worker_config",
        lambda: WorkerInitConfig(
            image_width=1050,
            image_height=1485,
            courtesy_naturals_probability=1.0,
            profile_enabled=False,
        ),
    )
    monkeypatch.setattr(worker, "_build_label_transcription", lambda text, _config: "4cn\n*-\n")
    monkeypatch.setattr(worker, "_build_render_transcription", lambda text, _config: text)

    render_inputs: list[str] = []

    def fake_render(kern_for_verovio: str, *, filename: str, config: WorkerInitConfig):
        render_inputs.append(kern_for_verovio)
        return np.zeros((32, 32, 3), dtype=np.uint8), None, 2, None, None, None

    monkeypatch.setattr(worker, "_render_kern_for_sample", fake_render)
    monkeypatch.setattr(
        worker,
        "_postprocess_rendered_image",
        lambda image, **kwargs: (b"jpeg", 0.0, 0.0, None),
    )

    outcome = worker.generate_sample_from_path_outcome(kern_path, variant_idx=0)

    assert isinstance(outcome, SampleSuccess)
    assert outcome.transcription == "4cn\n*-\n"
    assert render_inputs == ["4cn\n*-\n"]


def test_generate_sample_from_path_outcome_rescues_multi_page_with_truncation(monkeypatch, tmp_path):
    kern_path = Path(tmp_path) / "sample.krn"
    original = "\n".join(
        [
            "*clefG2",
            "*M4/4",
            "=1",
            "4c",
            "=2",
            "4d",
            "=3",
            "4e",
        ]
    )
    kern_path.write_text(original, encoding="utf-8")

    monkeypatch.setattr(worker, "_file_renderer", object())
    monkeypatch.setattr(
        worker,
        "_current_worker_config",
        lambda: WorkerInitConfig(
            image_width=1050,
            image_height=1485,
            overflow_truncation_enabled=True,
            overflow_truncation_max_trials=8,
            profile_enabled=False,
        ),
    )
    monkeypatch.setattr(worker, "_build_render_transcription", lambda text, _config: text)

    def fake_render(
        kern_for_verovio: str,
        *,
        filename: str,
        config: WorkerInitConfig,
    ):
        if "4e" in kern_for_verovio:
            return None, None, SampleFailure(code="multi_page", filename=filename)
        return np.zeros((32, 32, 3), dtype=np.uint8), 3, None

    monkeypatch.setattr(worker, "_render_kern_for_sample", fake_render)
    monkeypatch.setattr(
        worker,
        "_postprocess_rendered_image",
        lambda image, **kwargs: (b"jpeg", 0.0, 0.0, None),
    )

    outcome = worker.generate_sample_from_path_outcome(kern_path, variant_idx=0)

    assert isinstance(outcome, SampleSuccess)
    assert outcome.truncation_applied is True
    assert outcome.truncation_ratio is not None and outcome.truncation_ratio < 1.0
    assert "4e" not in outcome.transcription


def test_generate_sample_from_path_outcome_marks_truncation_attempted_on_exhaustion(
    monkeypatch, tmp_path
):
    kern_path = Path(tmp_path) / "sample.krn"
    kern_path.write_text(
        "\n".join(["*clefG2", "*M4/4", "=1", "4c", "=2", "4d", "=3", "4e"]),
        encoding="utf-8",
    )

    monkeypatch.setattr(worker, "_file_renderer", object())
    monkeypatch.setattr(
        worker,
        "_current_worker_config",
        lambda: WorkerInitConfig(
            image_width=1050,
            image_height=1485,
            overflow_truncation_enabled=True,
            overflow_truncation_max_trials=4,
            profile_enabled=False,
        ),
    )
    monkeypatch.setattr(worker, "_build_render_transcription", lambda text, _config: text)
    monkeypatch.setattr(
        worker,
        "_render_kern_for_sample",
        lambda kern_for_verovio, *, filename, config: (
            None,
            None,
            SampleFailure(code="multi_page", filename=filename),
        ),
    )

    outcome = worker.generate_sample_from_path_outcome(kern_path, variant_idx=0)

    assert isinstance(outcome, SampleFailure)
    assert outcome.code == "multi_page"
    assert outcome.truncation_attempted is True


@pytest.mark.parametrize("system_count", [5, 6])
def test_generate_sample_from_path_outcome_accepts_target_band_success(
    monkeypatch, tmp_path, system_count
):
    kern_path = Path(tmp_path) / "sample.krn"
    kern_path.write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(worker, "_file_renderer", object())
    monkeypatch.setattr(
        worker,
        "_current_worker_config",
        lambda: WorkerInitConfig(
            image_width=1050,
            image_height=1485,
            target_min_systems=5,
            target_max_systems=6,
            overflow_truncation_enabled=True,
            profile_enabled=False,
        ),
    )
    monkeypatch.setattr(worker, "_build_render_transcription", lambda text, _config: text)
    monkeypatch.setattr(
        worker,
        "_render_kern_for_sample",
        lambda kern_for_verovio, *, filename, config: (
            np.zeros((32, 32, 3), dtype=np.uint8),
            system_count,
            None,
        ),
    )
    monkeypatch.setattr(
        worker,
        "_postprocess_rendered_image",
        lambda image, **kwargs: (b"jpeg", 0.0, 0.0, None),
    )

    outcome = worker.generate_sample_from_path_outcome(kern_path, variant_idx=0)

    assert isinstance(outcome, SampleSuccess)
    assert outcome.actual_system_count == system_count
    assert outcome.truncation_applied is False


def test_generate_sample_from_path_outcome_rejects_underfull_target_band_without_truncation(
    monkeypatch, tmp_path
):
    kern_path = Path(tmp_path) / "sample.krn"
    kern_path.write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(worker, "_file_renderer", object())
    monkeypatch.setattr(
        worker,
        "_current_worker_config",
        lambda: WorkerInitConfig(
            image_width=1050,
            image_height=1485,
            target_min_systems=5,
            target_max_systems=6,
            overflow_truncation_enabled=True,
            profile_enabled=False,
        ),
    )
    monkeypatch.setattr(worker, "_build_render_transcription", lambda text, _config: text)
    monkeypatch.setattr(
        worker,
        "_render_kern_for_sample",
        lambda kern_for_verovio, *, filename, config: (
            np.zeros((32, 32, 3), dtype=np.uint8),
            4,
            None,
        ),
    )
    truncation_called = {"value": False}

    def _unexpected_truncation_space(*args, **kwargs):
        truncation_called["value"] = True
        return PrefixTruncationSpace(chunks=("unexpected\n",))

    monkeypatch.setattr(
        worker,
        "build_prefix_truncation_space",
        _unexpected_truncation_space,
    )

    outcome = worker.generate_sample_from_path_outcome(kern_path, variant_idx=0)

    assert isinstance(outcome, SampleFailure)
    assert outcome.code == "system_band_below_min"
    assert outcome.detail == "system_count=4"
    assert truncation_called["value"] is False


def test_generate_sample_from_path_outcome_rejects_overfull_target_band_without_truncation_when_disabled(
    monkeypatch, tmp_path
):
    kern_path = Path(tmp_path) / "sample.krn"
    kern_path.write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(worker, "_file_renderer", object())
    monkeypatch.setattr(
        worker,
        "_current_worker_config",
        lambda: WorkerInitConfig(
            image_width=1050,
            image_height=1485,
            target_min_systems=5,
            target_max_systems=6,
            overflow_truncation_enabled=False,
            profile_enabled=False,
        ),
    )
    monkeypatch.setattr(worker, "_build_render_transcription", lambda text, _config: text)
    monkeypatch.setattr(
        worker,
        "_render_kern_for_sample",
        lambda kern_for_verovio, *, filename, config: (
            np.zeros((32, 32, 3), dtype=np.uint8),
            7,
            None,
        ),
    )

    outcome = worker.generate_sample_from_path_outcome(kern_path, variant_idx=0)

    assert isinstance(outcome, SampleFailure)
    assert outcome.code == "system_band_above_max"
    assert outcome.detail == "system_count=7"
    assert outcome.truncation_attempted is False


def test_generate_sample_from_path_outcome_rescues_overfull_target_band_with_truncation(
    monkeypatch, tmp_path
):
    kern_path = Path(tmp_path) / "sample.krn"
    kern_path.write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(worker, "_file_renderer", object())
    monkeypatch.setattr(
        worker,
        "_current_worker_config",
        lambda: WorkerInitConfig(
            image_width=1050,
            image_height=1485,
            target_min_systems=5,
            target_max_systems=6,
            courtesy_naturals_probability=0.0,
            overflow_truncation_enabled=True,
            overflow_truncation_max_trials=4,
            profile_enabled=False,
        ),
    )
    monkeypatch.setattr(worker, "_build_render_transcription", lambda text, _config: text)
    monkeypatch.setattr(
        worker,
        "build_prefix_truncation_space",
        lambda kern_text: PrefixTruncationSpace(chunks=("cand-1\n", "cand-2\n", "cand-3\n")),
    )

    def fake_render(kern_for_verovio: str, *, filename: str, config: WorkerInitConfig):
        if kern_for_verovio == "4c\n*-\n":
            return np.zeros((32, 32, 3), dtype=np.uint8), 7, None
        if kern_for_verovio == "cand-1\ncand-2":
            return np.zeros((32, 32, 3), dtype=np.uint8), 6, None
        raise AssertionError(f"unexpected truncation probe: {kern_for_verovio!r}")

    monkeypatch.setattr(worker, "_render_kern_for_sample", fake_render)
    monkeypatch.setattr(
        worker,
        "_postprocess_rendered_image",
        lambda image, **kwargs: (b"jpeg", 0.0, 0.0, None),
    )

    outcome = worker.generate_sample_from_path_outcome(kern_path, variant_idx=0)

    assert isinstance(outcome, SampleSuccess)
    assert outcome.actual_system_count == 6
    assert outcome.truncation_applied is True
    assert outcome.truncation_ratio == pytest.approx(2 / 3)


def test_generate_sample_from_path_outcome_searches_longer_after_under_min_candidate(
    monkeypatch, tmp_path
):
    kern_path = Path(tmp_path) / "sample.krn"
    kern_path.write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(worker, "_file_renderer", object())
    monkeypatch.setattr(
        worker,
        "_current_worker_config",
        lambda: WorkerInitConfig(
            image_width=1050,
            image_height=1485,
            target_min_systems=5,
            target_max_systems=6,
            overflow_truncation_enabled=True,
            overflow_truncation_max_trials=4,
            profile_enabled=False,
        ),
    )
    monkeypatch.setattr(worker, "_build_render_transcription", lambda text, _config: text)
    monkeypatch.setattr(
        worker,
        "build_prefix_truncation_space",
        lambda kern_text: PrefixTruncationSpace(
            chunks=("chunk-1\n", "chunk-2\n", "chunk-3\n", "chunk-4\n")
        ),
    )
    render_calls: list[str] = []

    def fake_render(kern_for_verovio: str, *, filename: str, config: WorkerInitConfig):
        render_calls.append(kern_for_verovio)
        if kern_for_verovio == "4c\n*-\n":
            return np.zeros((32, 32, 3), dtype=np.uint8), 8, None
        if kern_for_verovio == "chunk-1\nchunk-2":
            return np.zeros((32, 32, 3), dtype=np.uint8), 4, None
        if kern_for_verovio == "chunk-1\nchunk-2\nchunk-3":
            return np.zeros((32, 32, 3), dtype=np.uint8), 5, None
        raise AssertionError(f"unexpected truncation probe: {kern_for_verovio!r}")

    monkeypatch.setattr(worker, "_render_kern_for_sample", fake_render)
    monkeypatch.setattr(
        worker,
        "_postprocess_rendered_image",
        lambda image, **kwargs: (b"jpeg", 0.0, 0.0, None),
    )

    outcome = worker.generate_sample_from_path_outcome(kern_path, variant_idx=0)

    assert isinstance(outcome, SampleSuccess)
    assert outcome.truncation_applied is True
    assert outcome.actual_system_count == 5
    assert render_calls == ["4c\n*-\n", "chunk-1\nchunk-2", "chunk-1\nchunk-2\nchunk-3"]


def test_generate_sample_from_path_outcome_marks_truncation_exhaustion_diagnostic(
    monkeypatch, tmp_path
):
    kern_path = Path(tmp_path) / "sample.krn"
    kern_path.write_text("4c\n*-\n", encoding="utf-8")

    monkeypatch.setattr(worker, "_file_renderer", object())
    monkeypatch.setattr(
        worker,
        "_current_worker_config",
        lambda: WorkerInitConfig(
            image_width=1050,
            image_height=1485,
            target_min_systems=5,
            target_max_systems=7,
            overflow_truncation_enabled=True,
            overflow_truncation_max_trials=4,
            profile_enabled=False,
        ),
    )
    monkeypatch.setattr(worker, "_build_render_transcription", lambda text, _config: text)
    monkeypatch.setattr(
        worker,
        "build_prefix_truncation_space",
        lambda kern_text: PrefixTruncationSpace(
            chunks=("chunk-1\n", "chunk-2\n", "chunk-3\n", "chunk-4\n")
        ),
    )

    def fake_render(kern_for_verovio: str, *, filename: str, config: WorkerInitConfig):
        if kern_for_verovio == "4c\n*-\n":
            return np.zeros((32, 32, 3), dtype=np.uint8), 9, None
        if kern_for_verovio == "chunk-1\nchunk-2":
            return np.zeros((32, 32, 3), dtype=np.uint8), 4, None
        return None, None, SampleFailure(code="render_fit", filename=filename, detail="bottom_clearance")

    monkeypatch.setattr(worker, "_render_kern_for_sample", fake_render)

    outcome = worker.generate_sample_from_path_outcome(kern_path, variant_idx=0)

    assert isinstance(outcome, SampleFailure)
    assert outcome.code == "system_band_truncation_exhausted"
    assert outcome.truncation_attempted is True
    assert outcome.detail == "target_band=5-7;diagnostic=mixed_gap"


def test_generate_sample_from_path_allows_sparse_render_and_runs_offline_augment(monkeypatch, tmp_path):
    kern_path = Path(tmp_path) / "sample.krn"
    kern_path.write_text("4c\n*-\n", encoding="utf-8")

    sparse_img = np.full((1485, 1050, 3), 255, dtype=np.uint8)
    sparse_img[10, 10:200] = 0  # Very sparse black pixels

    monkeypatch.setattr(worker, "_image_width", 1050)
    monkeypatch.setattr(worker, "_image_height", 1485)
    monkeypatch.setattr(worker, "_augment_seed", None)
    monkeypatch.setattr(worker, "_file_renderer", object())
    monkeypatch.setattr(worker, "_render_hairpins_enabled", False)
    monkeypatch.setattr(worker, "_render_hairpins_probability", 0.0)
    monkeypatch.setattr(worker, "_render_hairpins_max_spans", 1)
    monkeypatch.setattr(worker, "_render_dynamic_marks_enabled", False)
    monkeypatch.setattr(worker, "_render_dynamic_marks_probability", 0.0)
    monkeypatch.setattr(worker, "_render_dynamic_marks_min_count", 1)
    monkeypatch.setattr(worker, "_render_dynamic_marks_max_count", 2)
    monkeypatch.setattr(worker, "_courtesy_naturals_probability", 0.0)
    monkeypatch.setattr(worker, "_capture_stderr_fd", lambda: contextlib.nullcontext([]))

    def fake_generate_score_with_diagnostics(kern_for_verovio, renderer, config):
        return (
            GeneratedScore(
                image=sparse_img,
                transcription=kern_for_verovio,
                actual_system_count=4,
                metadata_prefix="",
            ),
            None,
        )

    monkeypatch.setattr(worker, "generate_score_with_diagnostics", fake_generate_score_with_diagnostics)

    offline_mod = types.SimpleNamespace(
        offline_augment=lambda image, **kwargs: image,
    )
    monkeypatch.setitem(
        sys.modules,
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment",
        offline_mod,
    )

    result = worker.generate_sample_from_path(kern_path, variant_idx=0)

    assert result[0] is not None
    assert isinstance(result[0], bytes)
    assert result[2] == "sample.krn"
    assert result[1] == "4c\n*-\n"


def test_generate_sample_from_path_injects_render_only_pedals(monkeypatch, tmp_path):
    kern_path = Path(tmp_path) / "sample.krn"
    original_transcription = "\n".join(
        [
            "*clefG2",
            "*k[]",
            "*M4/4",
            "=1",
            "4c",
            "=2",
            "4d",
            "==",
            "*-",
        ]
    )
    kern_path.write_text(original_transcription, encoding="utf-8")

    dense_img = np.full((1485, 1050, 3), 255, dtype=np.uint8)
    dense_img[20:80, 20:500] = 0

    monkeypatch.setattr(worker, "_image_width", 1050)
    monkeypatch.setattr(worker, "_image_height", 1485)
    monkeypatch.setattr(worker, "_augment_seed", None)
    monkeypatch.setattr(worker, "_file_renderer", object())
    monkeypatch.setattr(worker, "_render_pedals_enabled", True)
    monkeypatch.setattr(worker, "_render_pedals_probability", 1.0)
    monkeypatch.setattr(worker, "_render_pedals_measures_probability", 1.0)
    monkeypatch.setattr(worker, "_render_instrument_piano_enabled", False)
    monkeypatch.setattr(worker, "_render_instrument_piano_probability", 0.0)
    monkeypatch.setattr(worker, "_render_sforzando_enabled", False)
    monkeypatch.setattr(worker, "_render_sforzando_probability", 0.0)
    monkeypatch.setattr(worker, "_render_sforzando_per_note_probability", 0.0)
    monkeypatch.setattr(worker, "_render_accent_enabled", False)
    monkeypatch.setattr(worker, "_render_accent_probability", 0.0)
    monkeypatch.setattr(worker, "_render_accent_per_note_probability", 0.0)
    monkeypatch.setattr(worker, "_render_tempo_enabled", False)
    monkeypatch.setattr(worker, "_render_tempo_probability", 0.0)
    monkeypatch.setattr(worker, "_render_tempo_include_mm_probability", 0.0)
    monkeypatch.setattr(worker, "_render_hairpins_enabled", False)
    monkeypatch.setattr(worker, "_render_hairpins_probability", 0.0)
    monkeypatch.setattr(worker, "_render_hairpins_max_spans", 1)
    monkeypatch.setattr(worker, "_render_dynamic_marks_enabled", False)
    monkeypatch.setattr(worker, "_render_dynamic_marks_probability", 0.0)
    monkeypatch.setattr(worker, "_render_dynamic_marks_min_count", 1)
    monkeypatch.setattr(worker, "_render_dynamic_marks_max_count", 2)
    monkeypatch.setattr(worker, "_courtesy_naturals_probability", 0.0)
    monkeypatch.setattr(worker, "_capture_stderr_fd", lambda: contextlib.nullcontext([]))

    captured = {}

    def fake_generate_score_with_diagnostics(kern_for_verovio, renderer, config):
        captured["render_input"] = kern_for_verovio
        return (
            GeneratedScore(
                image=dense_img,
                transcription=kern_for_verovio,
                actual_system_count=4,
                metadata_prefix="",
            ),
            None,
        )

    monkeypatch.setattr(worker, "generate_score_with_diagnostics", fake_generate_score_with_diagnostics)

    offline_mod = types.SimpleNamespace(
        offline_augment=lambda image, **kwargs: image,
    )
    monkeypatch.setitem(
        sys.modules,
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment",
        offline_mod,
    )

    result = worker.generate_sample_from_path(kern_path, variant_idx=0)

    assert result[0] is not None
    assert isinstance(result[0], bytes)
    assert result[1] == original_transcription
    assert "*ped" not in result[1]
    assert "*Xped" not in result[1]
    assert captured["render_input"].startswith("**kern\n")
    assert "*ped" in captured["render_input"]
    assert "*Xped" in captured["render_input"]


def test_generate_sample_from_path_skips_render_only_pedals_when_sample_gate_misses(
    monkeypatch, tmp_path
):
    kern_path = Path(tmp_path) / "sample.krn"
    original_transcription = "\n".join(
        [
            "*clefG2",
            "*k[]",
            "*M4/4",
            "=1",
            "4c",
            "=2",
            "4d",
            "==",
            "*-",
        ]
    )
    kern_path.write_text(original_transcription, encoding="utf-8")

    dense_img = np.full((1485, 1050, 3), 255, dtype=np.uint8)
    dense_img[20:80, 20:500] = 0

    monkeypatch.setattr(worker, "_image_width", 1050)
    monkeypatch.setattr(worker, "_image_height", 1485)
    monkeypatch.setattr(worker, "_augment_seed", None)
    monkeypatch.setattr(worker, "_file_renderer", object())
    monkeypatch.setattr(worker, "_render_pedals_enabled", True)
    monkeypatch.setattr(worker, "_render_pedals_probability", 0.0)
    monkeypatch.setattr(worker, "_render_pedals_measures_probability", 1.0)
    monkeypatch.setattr(worker, "_render_instrument_piano_enabled", False)
    monkeypatch.setattr(worker, "_render_instrument_piano_probability", 0.0)
    monkeypatch.setattr(worker, "_render_sforzando_enabled", False)
    monkeypatch.setattr(worker, "_render_sforzando_probability", 0.0)
    monkeypatch.setattr(worker, "_render_sforzando_per_note_probability", 0.0)
    monkeypatch.setattr(worker, "_render_accent_enabled", False)
    monkeypatch.setattr(worker, "_render_accent_probability", 0.0)
    monkeypatch.setattr(worker, "_render_accent_per_note_probability", 0.0)
    monkeypatch.setattr(worker, "_render_tempo_enabled", False)
    monkeypatch.setattr(worker, "_render_tempo_probability", 0.0)
    monkeypatch.setattr(worker, "_render_tempo_include_mm_probability", 0.0)
    monkeypatch.setattr(worker, "_render_hairpins_enabled", False)
    monkeypatch.setattr(worker, "_render_hairpins_probability", 0.0)
    monkeypatch.setattr(worker, "_render_hairpins_max_spans", 1)
    monkeypatch.setattr(worker, "_render_dynamic_marks_enabled", False)
    monkeypatch.setattr(worker, "_render_dynamic_marks_probability", 0.0)
    monkeypatch.setattr(worker, "_render_dynamic_marks_min_count", 1)
    monkeypatch.setattr(worker, "_render_dynamic_marks_max_count", 2)
    monkeypatch.setattr(worker, "_courtesy_naturals_probability", 0.0)
    monkeypatch.setattr(worker, "_capture_stderr_fd", lambda: contextlib.nullcontext([]))

    captured = {}

    def fake_generate_score_with_diagnostics(kern_for_verovio, renderer, config):
        captured["render_input"] = kern_for_verovio
        return (
            GeneratedScore(
                image=dense_img,
                transcription=kern_for_verovio,
                actual_system_count=4,
                metadata_prefix="",
            ),
            None,
        )

    monkeypatch.setattr(worker, "generate_score_with_diagnostics", fake_generate_score_with_diagnostics)

    offline_mod = types.SimpleNamespace(
        offline_augment=lambda image, **kwargs: image,
    )
    monkeypatch.setitem(
        sys.modules,
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment",
        offline_mod,
    )

    result = worker.generate_sample_from_path(kern_path, variant_idx=0)

    assert result[0] is not None
    assert isinstance(result[0], bytes)
    assert result[1] == original_transcription
    assert "*ped" not in captured["render_input"]
    assert "*Xped" not in captured["render_input"]


def test_generate_sample_from_path_injects_render_only_piano_label(monkeypatch, tmp_path):
    kern_path = Path(tmp_path) / "sample.krn"
    original_transcription = "\n".join(
        [
            "*clefF4\t*clefG2",
            "*k[]\t*k[]",
            "*M4/4\t*M4/4",
            "=1\t=1",
            "4c\t4e",
            "==\t==",
            "*-\t*-",
        ]
    )
    kern_path.write_text(original_transcription, encoding="utf-8")

    dense_img = np.full((1485, 1050, 3), 255, dtype=np.uint8)
    dense_img[20:80, 20:500] = 0

    monkeypatch.setattr(worker, "_image_width", 1050)
    monkeypatch.setattr(worker, "_image_height", 1485)
    monkeypatch.setattr(worker, "_augment_seed", None)
    monkeypatch.setattr(worker, "_file_renderer", object())
    monkeypatch.setattr(worker, "_render_pedals_enabled", False)
    monkeypatch.setattr(worker, "_render_pedals_measures_probability", 0.0)
    monkeypatch.setattr(worker, "_render_instrument_piano_enabled", True)
    monkeypatch.setattr(worker, "_render_instrument_piano_probability", 1.0)
    monkeypatch.setattr(worker, "_render_sforzando_enabled", False)
    monkeypatch.setattr(worker, "_render_sforzando_probability", 0.0)
    monkeypatch.setattr(worker, "_render_sforzando_per_note_probability", 0.0)
    monkeypatch.setattr(worker, "_render_accent_enabled", False)
    monkeypatch.setattr(worker, "_render_accent_probability", 0.0)
    monkeypatch.setattr(worker, "_render_accent_per_note_probability", 0.0)
    monkeypatch.setattr(worker, "_render_tempo_enabled", False)
    monkeypatch.setattr(worker, "_render_tempo_probability", 0.0)
    monkeypatch.setattr(worker, "_render_tempo_include_mm_probability", 0.0)
    monkeypatch.setattr(worker, "_render_hairpins_enabled", False)
    monkeypatch.setattr(worker, "_render_hairpins_probability", 0.0)
    monkeypatch.setattr(worker, "_render_hairpins_max_spans", 1)
    monkeypatch.setattr(worker, "_render_dynamic_marks_enabled", False)
    monkeypatch.setattr(worker, "_render_dynamic_marks_probability", 0.0)
    monkeypatch.setattr(worker, "_render_dynamic_marks_min_count", 1)
    monkeypatch.setattr(worker, "_render_dynamic_marks_max_count", 2)
    monkeypatch.setattr(worker, "_courtesy_naturals_probability", 0.0)
    monkeypatch.setattr(worker, "_capture_stderr_fd", lambda: contextlib.nullcontext([]))

    captured = {}

    def fake_generate_score_with_diagnostics(kern_for_verovio, renderer, config):
        captured["render_input"] = kern_for_verovio
        return (
            GeneratedScore(
                image=dense_img,
                transcription=kern_for_verovio,
                actual_system_count=4,
                metadata_prefix="",
            ),
            None,
        )

    monkeypatch.setattr(worker, "generate_score_with_diagnostics", fake_generate_score_with_diagnostics)

    offline_mod = types.SimpleNamespace(
        offline_augment=lambda image, **kwargs: image,
    )
    monkeypatch.setitem(
        sys.modules,
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment",
        offline_mod,
    )

    result = worker.generate_sample_from_path(kern_path, variant_idx=0)

    assert result[0] is not None
    assert isinstance(result[0], bytes)
    assert result[1] == original_transcription
    assert '*I"Piano' not in result[1]
    assert captured["render_input"].startswith("**kern\t**kern\n")
    assert '*I"Piano\t*I"Piano' in captured["render_input"]


def test_generate_sample_from_path_injects_render_only_sforzandos(monkeypatch, tmp_path):
    kern_path = Path(tmp_path) / "sample.krn"
    original_transcription = "\n".join(
        [
            "*clefG2",
            "*k[]",
            "*M4/4",
            "=1",
            "4c",
            "=2",
            "4d",
            "==",
            "*-",
        ]
    )
    kern_path.write_text(original_transcription, encoding="utf-8")

    dense_img = np.full((1485, 1050, 3), 255, dtype=np.uint8)
    dense_img[20:80, 20:500] = 0

    monkeypatch.setattr(worker, "_image_width", 1050)
    monkeypatch.setattr(worker, "_image_height", 1485)
    monkeypatch.setattr(worker, "_augment_seed", None)
    monkeypatch.setattr(worker, "_file_renderer", object())
    monkeypatch.setattr(worker, "_render_pedals_enabled", False)
    monkeypatch.setattr(worker, "_render_pedals_measures_probability", 0.0)
    monkeypatch.setattr(worker, "_render_instrument_piano_enabled", False)
    monkeypatch.setattr(worker, "_render_instrument_piano_probability", 0.0)
    monkeypatch.setattr(worker, "_render_sforzando_enabled", True)
    monkeypatch.setattr(worker, "_render_sforzando_probability", 1.0)
    monkeypatch.setattr(worker, "_render_sforzando_per_note_probability", 1.0)
    monkeypatch.setattr(worker, "_render_accent_enabled", False)
    monkeypatch.setattr(worker, "_render_accent_probability", 0.0)
    monkeypatch.setattr(worker, "_render_accent_per_note_probability", 0.0)
    monkeypatch.setattr(worker, "_render_tempo_enabled", False)
    monkeypatch.setattr(worker, "_render_tempo_probability", 0.0)
    monkeypatch.setattr(worker, "_render_tempo_include_mm_probability", 0.0)
    monkeypatch.setattr(worker, "_render_hairpins_enabled", False)
    monkeypatch.setattr(worker, "_render_hairpins_probability", 0.0)
    monkeypatch.setattr(worker, "_render_hairpins_max_spans", 1)
    monkeypatch.setattr(worker, "_render_dynamic_marks_enabled", False)
    monkeypatch.setattr(worker, "_render_dynamic_marks_probability", 0.0)
    monkeypatch.setattr(worker, "_render_dynamic_marks_min_count", 1)
    monkeypatch.setattr(worker, "_render_dynamic_marks_max_count", 2)
    monkeypatch.setattr(worker, "_courtesy_naturals_probability", 0.0)
    monkeypatch.setattr(worker, "_capture_stderr_fd", lambda: contextlib.nullcontext([]))

    captured = {}

    def fake_generate_score_with_diagnostics(kern_for_verovio, renderer, config):
        captured["render_input"] = kern_for_verovio
        return (
            GeneratedScore(
                image=dense_img,
                transcription=kern_for_verovio,
                actual_system_count=4,
                metadata_prefix="",
            ),
            None,
        )

    monkeypatch.setattr(worker, "generate_score_with_diagnostics", fake_generate_score_with_diagnostics)

    offline_mod = types.SimpleNamespace(
        offline_augment=lambda image, **kwargs: image,
    )
    monkeypatch.setitem(
        sys.modules,
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment",
        offline_mod,
    )

    result = worker.generate_sample_from_path(kern_path, variant_idx=0)

    assert result[0] is not None
    assert isinstance(result[0], bytes)
    assert result[1] == original_transcription
    assert "z" not in result[1]
    assert "z" in captured["render_input"]


def test_generate_sample_from_path_injects_render_only_accents(monkeypatch, tmp_path):
    kern_path = Path(tmp_path) / "sample.krn"
    original_transcription = "\n".join(
        [
            "*clefG2",
            "*k[]",
            "*M4/4",
            "=1",
            "4c",
            "=2",
            "4d",
            "==",
            "*-",
        ]
    )
    kern_path.write_text(original_transcription, encoding="utf-8")

    dense_img = np.full((1485, 1050, 3), 255, dtype=np.uint8)
    dense_img[20:80, 20:500] = 0

    monkeypatch.setattr(worker, "_image_width", 1050)
    monkeypatch.setattr(worker, "_image_height", 1485)
    monkeypatch.setattr(worker, "_augment_seed", None)
    monkeypatch.setattr(worker, "_file_renderer", object())
    monkeypatch.setattr(worker, "_render_pedals_enabled", False)
    monkeypatch.setattr(worker, "_render_pedals_measures_probability", 0.0)
    monkeypatch.setattr(worker, "_render_instrument_piano_enabled", False)
    monkeypatch.setattr(worker, "_render_instrument_piano_probability", 0.0)
    monkeypatch.setattr(worker, "_render_sforzando_enabled", False)
    monkeypatch.setattr(worker, "_render_sforzando_probability", 0.0)
    monkeypatch.setattr(worker, "_render_sforzando_per_note_probability", 0.0)
    monkeypatch.setattr(worker, "_render_accent_enabled", True)
    monkeypatch.setattr(worker, "_render_accent_probability", 1.0)
    monkeypatch.setattr(worker, "_render_accent_per_note_probability", 1.0)
    monkeypatch.setattr(worker, "_render_tempo_enabled", False)
    monkeypatch.setattr(worker, "_render_tempo_probability", 0.0)
    monkeypatch.setattr(worker, "_render_tempo_include_mm_probability", 0.0)
    monkeypatch.setattr(worker, "_render_hairpins_enabled", False)
    monkeypatch.setattr(worker, "_render_hairpins_probability", 0.0)
    monkeypatch.setattr(worker, "_render_hairpins_max_spans", 1)
    monkeypatch.setattr(worker, "_render_dynamic_marks_enabled", False)
    monkeypatch.setattr(worker, "_render_dynamic_marks_probability", 0.0)
    monkeypatch.setattr(worker, "_render_dynamic_marks_min_count", 1)
    monkeypatch.setattr(worker, "_render_dynamic_marks_max_count", 2)
    monkeypatch.setattr(worker, "_courtesy_naturals_probability", 0.0)
    monkeypatch.setattr(worker, "_capture_stderr_fd", lambda: contextlib.nullcontext([]))

    captured = {}

    def fake_generate_score_with_diagnostics(kern_for_verovio, renderer, config):
        captured["render_input"] = kern_for_verovio
        return (
            GeneratedScore(
                image=dense_img,
                transcription=kern_for_verovio,
                actual_system_count=4,
                metadata_prefix="",
            ),
            None,
        )

    monkeypatch.setattr(worker, "generate_score_with_diagnostics", fake_generate_score_with_diagnostics)

    offline_mod = types.SimpleNamespace(
        offline_augment=lambda image, **kwargs: image,
    )
    monkeypatch.setitem(
        sys.modules,
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment",
        offline_mod,
    )

    result = worker.generate_sample_from_path(kern_path, variant_idx=0)

    assert result[0] is not None
    assert isinstance(result[0], bytes)
    assert result[1] == original_transcription
    assert "^" not in result[1]
    assert "^" in captured["render_input"]


def test_generate_sample_from_path_injects_render_only_tempo_markings(monkeypatch, tmp_path):
    kern_path = Path(tmp_path) / "sample.krn"
    original_transcription = "\n".join(
        [
            "*clefG2\t*clefF4",
            "*k[]\t*k[]",
            "*M4/4\t*M4/4",
            "=1\t=1",
            "4c\t4e",
            "==\t==",
            "*-\t*-",
        ]
    )
    kern_path.write_text(original_transcription, encoding="utf-8")

    dense_img = np.full((1485, 1050, 3), 255, dtype=np.uint8)
    dense_img[20:80, 20:500] = 0

    monkeypatch.setattr(worker, "_image_width", 1050)
    monkeypatch.setattr(worker, "_image_height", 1485)
    monkeypatch.setattr(worker, "_augment_seed", None)
    monkeypatch.setattr(worker, "_file_renderer", object())
    monkeypatch.setattr(worker, "_render_pedals_enabled", False)
    monkeypatch.setattr(worker, "_render_pedals_measures_probability", 0.0)
    monkeypatch.setattr(worker, "_render_instrument_piano_enabled", False)
    monkeypatch.setattr(worker, "_render_instrument_piano_probability", 0.0)
    monkeypatch.setattr(worker, "_render_sforzando_enabled", False)
    monkeypatch.setattr(worker, "_render_sforzando_probability", 0.0)
    monkeypatch.setattr(worker, "_render_sforzando_per_note_probability", 0.0)
    monkeypatch.setattr(worker, "_render_accent_enabled", False)
    monkeypatch.setattr(worker, "_render_accent_probability", 0.0)
    monkeypatch.setattr(worker, "_render_accent_per_note_probability", 0.0)
    monkeypatch.setattr(worker, "_render_tempo_enabled", True)
    monkeypatch.setattr(worker, "_render_tempo_probability", 1.0)
    monkeypatch.setattr(worker, "_render_tempo_include_mm_probability", 1.0)
    monkeypatch.setattr(worker, "_render_hairpins_enabled", False)
    monkeypatch.setattr(worker, "_render_hairpins_probability", 0.0)
    monkeypatch.setattr(worker, "_render_hairpins_max_spans", 1)
    monkeypatch.setattr(worker, "_render_dynamic_marks_enabled", False)
    monkeypatch.setattr(worker, "_render_dynamic_marks_probability", 0.0)
    monkeypatch.setattr(worker, "_render_dynamic_marks_min_count", 1)
    monkeypatch.setattr(worker, "_render_dynamic_marks_max_count", 2)
    monkeypatch.setattr(worker, "_courtesy_naturals_probability", 0.0)
    monkeypatch.setattr(worker, "_capture_stderr_fd", lambda: contextlib.nullcontext([]))

    captured = {}

    def fake_generate_score_with_diagnostics(kern_for_verovio, renderer, config):
        captured["render_input"] = kern_for_verovio
        return (
            GeneratedScore(
                image=dense_img,
                transcription=kern_for_verovio,
                actual_system_count=4,
                metadata_prefix="",
            ),
            None,
        )

    monkeypatch.setattr(worker, "generate_score_with_diagnostics", fake_generate_score_with_diagnostics)

    offline_mod = types.SimpleNamespace(
        offline_augment=lambda image, **kwargs: image,
    )
    monkeypatch.setitem(
        sys.modules,
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment",
        offline_mod,
    )

    result = worker.generate_sample_from_path(kern_path, variant_idx=0)

    assert result[0] is not None
    assert isinstance(result[0], bytes)
    assert result[1] == original_transcription
    assert "!!!OMD:" not in result[1]
    assert "*MM" not in result[1]
    assert captured["render_input"].startswith("**kern\t**kern\n")
    assert "!!!OMD:" in captured["render_input"]
    assert "*MM" in captured["render_input"]


def test_generate_sample_from_path_injects_render_only_hairpins(monkeypatch, tmp_path):
    kern_path = Path(tmp_path) / "sample.krn"
    original_transcription = "\n".join(
        [
            "*clefF4\t*clefG2",
            "*k[]\t*k[]",
            "*M4/4\t*M4/4",
            "*^\t*",
            "4c\t4e\t4g",
            "4d\t.\t.",
            "*v\t*v\t*",
            "=1\t=1",
            "4e\t4f",
            "==\t==",
            "*-\t*-",
        ]
    )
    kern_path.write_text(original_transcription, encoding="utf-8")

    dense_img = np.full((1485, 1050, 3), 255, dtype=np.uint8)
    dense_img[20:80, 20:500] = 0

    monkeypatch.setattr(worker, "_image_width", 1050)
    monkeypatch.setattr(worker, "_image_height", 1485)
    monkeypatch.setattr(worker, "_augment_seed", None)
    monkeypatch.setattr(worker, "_file_renderer", object())
    monkeypatch.setattr(worker, "_render_pedals_enabled", False)
    monkeypatch.setattr(worker, "_render_pedals_measures_probability", 0.0)
    monkeypatch.setattr(worker, "_render_instrument_piano_enabled", False)
    monkeypatch.setattr(worker, "_render_instrument_piano_probability", 0.0)
    monkeypatch.setattr(worker, "_render_sforzando_enabled", False)
    monkeypatch.setattr(worker, "_render_sforzando_probability", 0.0)
    monkeypatch.setattr(worker, "_render_sforzando_per_note_probability", 0.0)
    monkeypatch.setattr(worker, "_render_accent_enabled", False)
    monkeypatch.setattr(worker, "_render_accent_probability", 0.0)
    monkeypatch.setattr(worker, "_render_accent_per_note_probability", 0.0)
    monkeypatch.setattr(worker, "_render_tempo_enabled", False)
    monkeypatch.setattr(worker, "_render_tempo_probability", 0.0)
    monkeypatch.setattr(worker, "_render_tempo_include_mm_probability", 0.0)
    monkeypatch.setattr(worker, "_render_hairpins_enabled", True)
    monkeypatch.setattr(worker, "_render_hairpins_probability", 1.0)
    monkeypatch.setattr(worker, "_render_hairpins_max_spans", 1)
    monkeypatch.setattr(worker, "_render_dynamic_marks_enabled", False)
    monkeypatch.setattr(worker, "_render_dynamic_marks_probability", 0.0)
    monkeypatch.setattr(worker, "_render_dynamic_marks_min_count", 1)
    monkeypatch.setattr(worker, "_render_dynamic_marks_max_count", 2)
    monkeypatch.setattr(worker, "_capture_stderr_fd", lambda: contextlib.nullcontext([]))

    captured = {}

    def fake_generate_score_with_diagnostics(kern_for_verovio, renderer, config):
        captured["render_input"] = kern_for_verovio
        return (
            GeneratedScore(
                image=dense_img,
                transcription=kern_for_verovio,
                actual_system_count=4,
                metadata_prefix="",
            ),
            None,
        )

    monkeypatch.setattr(worker, "generate_score_with_diagnostics", fake_generate_score_with_diagnostics)

    offline_mod = types.SimpleNamespace(
        offline_augment=lambda image, **kwargs: image,
    )
    monkeypatch.setitem(
        sys.modules,
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment",
        offline_mod,
    )

    result = worker.generate_sample_from_path(kern_path, variant_idx=0)

    assert result[0] is not None
    assert isinstance(result[0], bytes)
    assert result[1] == original_transcription
    assert "**dynam" not in result[1]
    assert "<" not in result[1]
    assert ">" not in result[1]
    assert captured["render_input"].startswith("**kern\t**kern\t**dynam\n")
    assert "*^\t*\t*" in captured["render_input"]
    assert "*v\t*v\t*" in captured["render_input"]
    assert any(token in captured["render_input"] for token in ("<", ">", "[", "]", "(", ")"))


def test_generate_sample_from_path_injects_render_only_dynamic_marks(monkeypatch, tmp_path):
    kern_path = Path(tmp_path) / "sample.krn"
    original_transcription = "\n".join(
        [
            "*clefF4\t*clefG2",
            "*k[]\t*k[]",
            "*M4/4\t*M4/4",
            "=1\t=1",
            "4c\t4e",
            "4d\t4f",
            "4e\t4g",
            "4f\t4a",
            "==\t==",
            "*-\t*-",
        ]
    )
    kern_path.write_text(original_transcription, encoding="utf-8")

    dense_img = np.full((1485, 1050, 3), 255, dtype=np.uint8)
    dense_img[20:80, 20:500] = 0

    monkeypatch.setattr(worker, "_image_width", 1050)
    monkeypatch.setattr(worker, "_image_height", 1485)
    monkeypatch.setattr(worker, "_augment_seed", None)
    monkeypatch.setattr(worker, "_file_renderer", object())
    monkeypatch.setattr(worker, "_render_pedals_enabled", False)
    monkeypatch.setattr(worker, "_render_pedals_measures_probability", 0.0)
    monkeypatch.setattr(worker, "_render_instrument_piano_enabled", False)
    monkeypatch.setattr(worker, "_render_instrument_piano_probability", 0.0)
    monkeypatch.setattr(worker, "_render_sforzando_enabled", False)
    monkeypatch.setattr(worker, "_render_sforzando_probability", 0.0)
    monkeypatch.setattr(worker, "_render_sforzando_per_note_probability", 0.0)
    monkeypatch.setattr(worker, "_render_accent_enabled", False)
    monkeypatch.setattr(worker, "_render_accent_probability", 0.0)
    monkeypatch.setattr(worker, "_render_accent_per_note_probability", 0.0)
    monkeypatch.setattr(worker, "_render_tempo_enabled", False)
    monkeypatch.setattr(worker, "_render_tempo_probability", 0.0)
    monkeypatch.setattr(worker, "_render_tempo_include_mm_probability", 0.0)
    monkeypatch.setattr(worker, "_render_hairpins_enabled", False)
    monkeypatch.setattr(worker, "_render_hairpins_probability", 0.0)
    monkeypatch.setattr(worker, "_render_hairpins_max_spans", 1)
    monkeypatch.setattr(worker, "_render_dynamic_marks_enabled", True)
    monkeypatch.setattr(worker, "_render_dynamic_marks_probability", 1.0)
    monkeypatch.setattr(worker, "_render_dynamic_marks_min_count", 2)
    monkeypatch.setattr(worker, "_render_dynamic_marks_max_count", 2)
    monkeypatch.setattr(worker, "_courtesy_naturals_probability", 0.0)
    monkeypatch.setattr(worker, "_capture_stderr_fd", lambda: contextlib.nullcontext([]))

    captured = {}

    def fake_generate_score_with_diagnostics(kern_for_verovio, renderer, config):
        captured["render_input"] = kern_for_verovio
        return (
            GeneratedScore(
                image=dense_img,
                transcription=kern_for_verovio,
                actual_system_count=4,
                metadata_prefix="",
            ),
            None,
        )

    monkeypatch.setattr(worker, "generate_score_with_diagnostics", fake_generate_score_with_diagnostics)

    offline_mod = types.SimpleNamespace(
        offline_augment=lambda image, **kwargs: image,
    )
    monkeypatch.setitem(
        sys.modules,
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment",
        offline_mod,
    )

    result = worker.generate_sample_from_path(kern_path, variant_idx=0)

    assert result[0] is not None
    assert isinstance(result[0], bytes)
    assert result[1] == original_transcription
    assert "**dynam" not in result[1]
    assert captured["render_input"].startswith("**kern\t**kern\t**dynam\n")

    lines = captured["render_input"].splitlines()
    dynam_idx = len(lines[0].split("\t")) - 1
    dynam_tokens = [
        line.split("\t")[dynam_idx]
        for line in lines
        if line and not line.startswith(("*", "=", "!"))
    ]
    assert any(token in {"pp", "p", "mp", "mf", "f", "ff", "sf", "z"} for token in dynam_tokens)


def test_generate_sample_from_path_combines_hairpins_and_dynamic_marks(monkeypatch, tmp_path):
    kern_path = Path(tmp_path) / "sample.krn"
    original_transcription = "\n".join(
        [
            "*clefF4\t*clefG2",
            "*k[]\t*k[]",
            "*M4/4\t*M4/4",
            "=1\t=1",
            "4c\t4e",
            "4d\t4f",
            "4e\t4g",
            "4f\t4a",
            "4g\t4b",
            "4a\t4cc",
            "4b\t4dd",
            "4cc\t4ee",
            "4dd\t4ff",
            "4ee\t4gg",
            "==\t==",
            "*-\t*-",
        ]
    )
    kern_path.write_text(original_transcription, encoding="utf-8")

    dense_img = np.full((1485, 1050, 3), 255, dtype=np.uint8)
    dense_img[20:80, 20:500] = 0

    monkeypatch.setattr(worker, "_image_width", 1050)
    monkeypatch.setattr(worker, "_image_height", 1485)
    monkeypatch.setattr(worker, "_augment_seed", None)
    monkeypatch.setattr(worker, "_file_renderer", object())
    monkeypatch.setattr(worker, "_render_pedals_enabled", False)
    monkeypatch.setattr(worker, "_render_pedals_measures_probability", 0.0)
    monkeypatch.setattr(worker, "_render_instrument_piano_enabled", False)
    monkeypatch.setattr(worker, "_render_instrument_piano_probability", 0.0)
    monkeypatch.setattr(worker, "_render_sforzando_enabled", False)
    monkeypatch.setattr(worker, "_render_sforzando_probability", 0.0)
    monkeypatch.setattr(worker, "_render_sforzando_per_note_probability", 0.0)
    monkeypatch.setattr(worker, "_render_accent_enabled", False)
    monkeypatch.setattr(worker, "_render_accent_probability", 0.0)
    monkeypatch.setattr(worker, "_render_accent_per_note_probability", 0.0)
    monkeypatch.setattr(worker, "_render_tempo_enabled", False)
    monkeypatch.setattr(worker, "_render_tempo_probability", 0.0)
    monkeypatch.setattr(worker, "_render_tempo_include_mm_probability", 0.0)
    monkeypatch.setattr(worker, "_render_hairpins_enabled", True)
    monkeypatch.setattr(worker, "_render_hairpins_probability", 1.0)
    monkeypatch.setattr(worker, "_render_hairpins_max_spans", 1)
    monkeypatch.setattr(worker, "_render_dynamic_marks_enabled", True)
    monkeypatch.setattr(worker, "_render_dynamic_marks_probability", 1.0)
    monkeypatch.setattr(worker, "_render_dynamic_marks_min_count", 2)
    monkeypatch.setattr(worker, "_render_dynamic_marks_max_count", 2)
    monkeypatch.setattr(worker, "_capture_stderr_fd", lambda: contextlib.nullcontext([]))

    captured = {}

    def fake_generate_score_with_diagnostics(kern_for_verovio, renderer, config):
        captured["render_input"] = kern_for_verovio
        return (
            GeneratedScore(
                image=dense_img,
                transcription=kern_for_verovio,
                actual_system_count=4,
                metadata_prefix="",
            ),
            None,
        )

    monkeypatch.setattr(worker, "generate_score_with_diagnostics", fake_generate_score_with_diagnostics)

    offline_mod = types.SimpleNamespace(
        offline_augment=lambda image, **kwargs: image,
    )
    monkeypatch.setitem(
        sys.modules,
        "scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment",
        offline_mod,
    )

    result = worker.generate_sample_from_path(kern_path, variant_idx=0)

    assert result[0] is not None
    assert isinstance(result[0], bytes)
    assert result[1] == original_transcription
    assert captured["render_input"].startswith("**kern\t**kern\t**dynam\n")

    lines = captured["render_input"].splitlines()
    header_tokens = lines[0].split("\t")
    assert header_tokens.count("**dynam") == 1
    dynam_idx = len(header_tokens) - 1
    dynam_tokens = [
        line.split("\t")[dynam_idx]
        for line in lines
        if line and not line.startswith(("*", "=", "!"))
    ]
    assert any(token in {"<", ">", "(", ")", "[", "]"} for token in dynam_tokens)
    assert any(token in {"pp", "p", "mp", "mf", "f", "ff", "sf", "z"} for token in dynam_tokens)


@pytest.mark.parametrize("invalid_probability", [-0.1, 1.1])
def test_init_file_worker_rejects_invalid_render_pedal_probability(invalid_probability):
    with pytest.raises(ValueError, match="render_pedals_measures_probability must be in \\[0.0, 1.0\\]"):
        worker.init_file_worker(
            image_width=1050,
            image_height=1485,
            render_pedals_measures_probability=invalid_probability,
        )


@pytest.mark.parametrize("invalid_probability", [-0.1, 1.1])
def test_init_file_worker_rejects_invalid_render_pedal_sample_probability(invalid_probability):
    with pytest.raises(ValueError, match="render_pedals_probability must be in \\[0.0, 1.0\\]"):
        worker.init_file_worker(
            image_width=1050,
            image_height=1485,
            render_pedals_probability=invalid_probability,
        )


@pytest.mark.parametrize("invalid_probability", [-0.1, 1.1])
def test_init_file_worker_rejects_invalid_render_piano_probability(invalid_probability):
    with pytest.raises(
        ValueError, match="render_instrument_piano_probability must be in \\[0.0, 1.0\\]"
    ):
        worker.init_file_worker(
            image_width=1050,
            image_height=1485,
            render_instrument_piano_probability=invalid_probability,
        )


@pytest.mark.parametrize("invalid_probability", [-0.1, 1.1])
def test_init_file_worker_rejects_invalid_render_sforzando_probability(invalid_probability):
    with pytest.raises(
        ValueError, match="render_sforzando_probability must be in \\[0.0, 1.0\\]"
    ):
        worker.init_file_worker(
            image_width=1050,
            image_height=1485,
            render_sforzando_probability=invalid_probability,
        )


@pytest.mark.parametrize("invalid_probability", [-0.1, 1.1])
def test_init_file_worker_rejects_invalid_render_sforzando_per_note_probability(invalid_probability):
    with pytest.raises(
        ValueError, match="render_sforzando_per_note_probability must be in \\[0.0, 1.0\\]"
    ):
        worker.init_file_worker(
            image_width=1050,
            image_height=1485,
            render_sforzando_per_note_probability=invalid_probability,
        )


@pytest.mark.parametrize("invalid_probability", [-0.1, 1.1])
def test_init_file_worker_rejects_invalid_render_accent_probability(invalid_probability):
    with pytest.raises(
        ValueError, match="render_accent_probability must be in \\[0.0, 1.0\\]"
    ):
        worker.init_file_worker(
            image_width=1050,
            image_height=1485,
            render_accent_probability=invalid_probability,
        )


@pytest.mark.parametrize("invalid_probability", [-0.1, 1.1])
def test_init_file_worker_rejects_invalid_render_accent_per_note_probability(invalid_probability):
    with pytest.raises(
        ValueError, match="render_accent_per_note_probability must be in \\[0.0, 1.0\\]"
    ):
        worker.init_file_worker(
            image_width=1050,
            image_height=1485,
            render_accent_per_note_probability=invalid_probability,
        )


@pytest.mark.parametrize("invalid_probability", [-0.1, 1.1])
def test_init_file_worker_rejects_invalid_render_tempo_probability(invalid_probability):
    with pytest.raises(
        ValueError, match="render_tempo_probability must be in \\[0.0, 1.0\\]"
    ):
        worker.init_file_worker(
            image_width=1050,
            image_height=1485,
            render_tempo_probability=invalid_probability,
        )


@pytest.mark.parametrize("invalid_probability", [-0.1, 1.1])
def test_init_file_worker_rejects_invalid_render_tempo_include_mm_probability(invalid_probability):
    with pytest.raises(
        ValueError, match="render_tempo_include_mm_probability must be in \\[0.0, 1.0\\]"
    ):
        worker.init_file_worker(
            image_width=1050,
            image_height=1485,
            render_tempo_include_mm_probability=invalid_probability,
        )


@pytest.mark.parametrize("invalid_probability", [-0.1, 1.1])
def test_init_file_worker_rejects_invalid_render_hairpins_probability(invalid_probability):
    with pytest.raises(
        ValueError, match="render_hairpins_probability must be in \\[0.0, 1.0\\]"
    ):
        worker.init_file_worker(
            image_width=1050,
            image_height=1485,
            render_hairpins_probability=invalid_probability,
        )


@pytest.mark.parametrize("invalid_max_spans", [0, -1])
def test_init_file_worker_rejects_invalid_render_hairpins_max_spans(invalid_max_spans):
    with pytest.raises(ValueError, match="render_hairpins_max_spans must be >= 1"):
        worker.init_file_worker(
            image_width=1050,
            image_height=1485,
            render_hairpins_max_spans=invalid_max_spans,
        )


@pytest.mark.parametrize("invalid_probability", [-0.1, 1.1])
def test_init_file_worker_rejects_invalid_render_dynamic_marks_probability(invalid_probability):
    with pytest.raises(
        ValueError, match="render_dynamic_marks_probability must be in \\[0.0, 1.0\\]"
    ):
        worker.init_file_worker(
            image_width=1050,
            image_height=1485,
            render_dynamic_marks_probability=invalid_probability,
        )


@pytest.mark.parametrize("invalid_min_count", [0, -1])
def test_init_file_worker_rejects_invalid_render_dynamic_marks_min_count(invalid_min_count):
    with pytest.raises(ValueError, match="render_dynamic_marks_min_count must be >= 1"):
        worker.init_file_worker(
            image_width=1050,
            image_height=1485,
            render_dynamic_marks_min_count=invalid_min_count,
        )


def test_init_file_worker_rejects_invalid_render_dynamic_marks_count_range():
    with pytest.raises(
        ValueError,
        match="render_dynamic_marks_max_count must be >= render_dynamic_marks_min_count",
    ):
        worker.init_file_worker(
            image_width=1050,
            image_height=1485,
            render_dynamic_marks_min_count=2,
            render_dynamic_marks_max_count=1,
        )
