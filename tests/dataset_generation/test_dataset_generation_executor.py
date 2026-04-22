import json
from collections import Counter, defaultdict
from concurrent.futures import Future, TimeoutError
from pathlib import Path

import numpy as np
import pytest
from datasets import load_from_disk
from pebble import ProcessExpired
from tokenizers import Tokenizer, models, pre_tokenizers

import scripts.dataset_generation.dataset_generation.executor as executor_module
from scripts.dataset_generation.dataset_generation.composer import compose_label_transcription
from scripts.dataset_generation.dataset_generation.executor import run_dataset_generation
from scripts.dataset_generation.dataset_generation.io import encode_jpeg_image
from scripts.dataset_generation.dataset_generation.recipe import (
    ProductionRecipe,
    RenderOnlyAugmentationPolicy,
)
from scripts.dataset_generation.dataset_generation.system_balance import (
    DEFAULT_TOKENIZER_DIR,
    CandidatePlanScore,
    compute_recipe_fingerprint,
    compute_tokenizer_fingerprint,
    load_system_balance_spec,
)
from scripts.dataset_generation.dataset_generation.types import (
    AcceptedSample,
    AugmentationPreviewArtifacts,
    AugmentationTraceEvent,
    BoundsGateTrace,
    FailureRenderAttempt,
    GeometryTrace,
    MarginTrace,
    OuterGateTrace,
    QualityGateTrace,
    RenderResult,
    SamplePlan,
    SourceSegment,
    SvgLayoutDiagnostics,
    VerovioDiagnosticEvent,
    WorkerFailure,
    WorkerSuccess,
)


class FakePebblePool:
    def __init__(self, *, outcomes_by_sample_idx, **kwargs):
        del kwargs
        self._outcomes_by_sample_idx = {
            int(sample_idx): list(outcomes)
            for sample_idx, outcomes in outcomes_by_sample_idx.items()
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def schedule(self, function, args=(), kwargs=None, timeout=None):
        del function, kwargs, timeout
        plan = args[0]
        sample_idx = int(plan.sample_id.split("_")[-1])
        outcome_factory = self._outcomes_by_sample_idx[sample_idx].pop(0)
        outcome = outcome_factory(plan) if callable(outcome_factory) else outcome_factory
        future = Future()
        if isinstance(outcome, BaseException):
            future.set_exception(outcome)
        else:
            future.set_result(outcome)
        return future


def _make_worker_success(plan: SamplePlan) -> WorkerSuccess:
    image = np.full((1485, 1050, 3), 255, dtype=np.uint8)
    image[10:70, 10:600] = 0
    sample = AcceptedSample(
        sample_id=plan.sample_id,
        label_transcription=plan.label_transcription,
        image_bytes=encode_jpeg_image(image),
        initial_kern_spine_count=plan.label_transcription.splitlines()[0].count("\t") + 1,
        segment_count=plan.segment_count,
        source_ids=tuple(segment.source_id for segment in plan.segments),
        source_measure_count=plan.source_measure_count,
        source_non_empty_line_count=plan.source_non_empty_line_count,
        system_count=4,
        truncation_applied=False,
        truncation_reason=None,
        truncation_ratio=None,
        bottom_whitespace_ratio=0.10,
        vertical_fill_ratio=0.72,
        bottom_whitespace_px=149,
        top_whitespace_px=33,
        content_height_px=1069,
    )
    return WorkerSuccess(
        sample=sample,
        truncation_attempted=False,
        truncation_rescued=False,
        full_render_system_count=4,
        full_render_content_height_px=1069,
        full_render_vertical_fill_ratio=0.72,
        full_render_rejection_reason=None,
        accepted_render_system_count=4,
    )


def _make_worker_success_with_truncation(plan: SamplePlan) -> WorkerSuccess:
    image = np.full((1485, 1050, 3), 255, dtype=np.uint8)
    image[10:70, 10:600] = 0
    sample = AcceptedSample(
        sample_id=plan.sample_id,
        label_transcription=plan.label_transcription,
        image_bytes=encode_jpeg_image(image),
        initial_kern_spine_count=plan.label_transcription.splitlines()[0].count("\t") + 1,
        segment_count=plan.segment_count,
        source_ids=tuple(segment.source_id for segment in plan.segments),
        source_measure_count=plan.source_measure_count,
        source_non_empty_line_count=plan.source_non_empty_line_count,
        system_count=5,
        truncation_applied=True,
        truncation_reason="system_count_policy",
        truncation_ratio=0.5,
        bottom_whitespace_ratio=0.22,
        vertical_fill_ratio=0.68,
        bottom_whitespace_px=326,
        top_whitespace_px=28,
        content_height_px=1010,
    )
    return WorkerSuccess(
        sample=sample,
        truncation_attempted=True,
        truncation_rescued=True,
        full_render_system_count=7,
        full_render_content_height_px=1185,
        full_render_vertical_fill_ratio=0.81,
        full_render_rejection_reason="multi_page",
        accepted_render_system_count=5,
        preferred_5_6_rescue_attempted=True,
        preferred_5_6_rescue_succeeded=False,
        preferred_5_6_status="preferred_5_6_truncated",
    )


def _make_bounds_gate_trace(*, passed: bool, failure_reason: str | None) -> BoundsGateTrace:
    return BoundsGateTrace(
        passed=passed,
        failure_reason=failure_reason,
        margins_px=MarginTrace(top_px=20, bottom_px=1200, left_px=20, right_px=500),
        border_touch_count=0,
        dx_frac=0.01,
        dy_frac=0.02,
        area_retention=0.95,
    )


def _make_geometry_trace(*, sampled: bool, conservative: bool) -> GeometryTrace:
    return GeometryTrace(
        sampled=sampled,
        conservative=conservative,
        angle_deg=0.8 if sampled else None,
        scale=1.02 if sampled else None,
        tx_px=2.0 if sampled else None,
        ty_px=3.0 if sampled else None,
        x_scale=0.95 if sampled else None,
        y_scale=1.0 if sampled else None,
        perspective_applied=False,
    )


def _make_augmented_worker_success(
    plan: SamplePlan,
    *,
    outer_gate_passed: bool,
    final_geometry_applied: bool,
    initial_oob_failure_reason: str | None = None,
    retry_oob_failure_reason: str | None = None,
    outer_gate_failure_reason: str | None = None,
) -> WorkerSuccess:
    outcome = _make_worker_success(plan)
    image = np.full((1485, 1050, 3), 255, dtype=np.uint8)
    image[10:70, 10:600] = 0
    trace = AugmentationTraceEvent(
        event="augmentation_trace",
        sample_id=plan.sample_id,
        sample_idx=int(plan.sample_id.split("_")[-1]),
        seed=plan.seed,
        render_height_px=1200,
        bottom_padding_px=285,
        top_whitespace_px=20,
        bottom_whitespace_px=600,
        content_height_px=580,
        band="balanced",
        branch="geometric",
        initial_geometry=_make_geometry_trace(sampled=True, conservative=False),
        retry_geometry=_make_geometry_trace(sampled=True, conservative=True),
        selected_geometry=_make_geometry_trace(
            sampled=final_geometry_applied,
            conservative=not final_geometry_applied,
        ),
        final_geometry_applied=final_geometry_applied,
        initial_oob_gate=_make_bounds_gate_trace(
            passed=initial_oob_failure_reason is None,
            failure_reason=initial_oob_failure_reason,
        ),
        retry_oob_gate=_make_bounds_gate_trace(
            passed=retry_oob_failure_reason is None,
            failure_reason=retry_oob_failure_reason,
        ),
        augraphy_outcome="applied",
        augraphy_normalize_accepted=True,
        augraphy_fallback_attempted=False,
        augraphy_fallback_outcome=None,
        augraphy_fallback_normalize_accepted=None,
        outer_gate=OuterGateTrace(
            passed=outer_gate_passed,
            failure_reason=outer_gate_failure_reason,
            quality_gate=QualityGateTrace(
                passed=outer_gate_passed,
                failure_reason=None if outer_gate_passed else "min_margin",
                mean_luma=200.0,
                black_ratio=0.1,
                margins_px=MarginTrace(top_px=20, bottom_px=1200, left_px=20, right_px=500),
                border_touch_count=0,
            ),
            transform_consistency=_make_bounds_gate_trace(
                passed=outer_gate_passed,
                failure_reason=None if outer_gate_passed else "min_margin",
            ),
        ),
        final_outcome="fully_augmented" if outer_gate_passed else "clean_gate_rejected",
        offline_geom_ms=1.0,
        offline_gates_ms=0.5,
        offline_augraphy_ms=2.0,
        offline_texture_ms=0.25,
    )
    preview = AugmentationPreviewArtifacts(
        base_image_jpeg=encode_jpeg_image(image),
        pre_augraphy_image_jpeg=encode_jpeg_image(image),
        final_image_jpeg=encode_jpeg_image(image),
    )
    return WorkerSuccess(
        sample=outcome.sample,
        truncation_attempted=outcome.truncation_attempted,
        truncation_rescued=outcome.truncation_rescued,
        augmentation_trace=trace,
        augmentation_preview=preview,
    )


def _make_worker_success_with_verovio_diagnostic(plan: SamplePlan) -> WorkerSuccess:
    outcome = _make_worker_success(plan)
    return WorkerSuccess(
        sample=outcome.sample,
        truncation_attempted=outcome.truncation_attempted,
        truncation_rescued=outcome.truncation_rescued,
        full_render_system_count=outcome.full_render_system_count,
        full_render_content_height_px=outcome.full_render_content_height_px,
        full_render_vertical_fill_ratio=outcome.full_render_vertical_fill_ratio,
        full_render_rejection_reason=outcome.full_render_rejection_reason,
        accepted_render_system_count=outcome.accepted_render_system_count,
        preferred_5_6_rescue_attempted=outcome.preferred_5_6_rescue_attempted,
        preferred_5_6_rescue_succeeded=outcome.preferred_5_6_rescue_succeeded,
        preferred_5_6_status=outcome.preferred_5_6_status,
        verovio_diagnostics=(
            VerovioDiagnosticEvent(
                event="verovio_diagnostic",
                sample_id=plan.sample_id,
                sample_idx=int(plan.sample_id.split("_")[-1]),
                source_paths=tuple(str(segment.path.resolve()) for segment in plan.segments),
                stage="full",
                seed=plan.seed,
                render_attempt_idx=1,
                diagnostic_kind="inconsistent_rhythm_analysis",
                raw_message="Error: Inconsistent rhythm analysis occurring near line 12",
                near_line=12,
                expected_duration_from_start="64",
                found_duration_from_start="62",
                line_text="4G\t.\t.\t4c 4e",
            ),
        ),
    )


def _make_worker_failure_with_verovio_diagnostic(plan: SamplePlan) -> WorkerFailure:
    return WorkerFailure(
        sample_id=plan.sample_id,
        failure_reason="truncation_exhausted",
        truncation_attempted=True,
        verovio_diagnostics=(
            VerovioDiagnosticEvent(
                event="verovio_diagnostic",
                sample_id=plan.sample_id,
                sample_idx=int(plan.sample_id.split("_")[-1]),
                source_paths=tuple(str(segment.path.resolve()) for segment in plan.segments),
                stage="truncation_candidate",
                seed=plan.seed + 17,
                render_attempt_idx=1,
                diagnostic_kind="verovio_error",
                raw_message="Error: Generic Verovio error",
                truncation_chunk_count=1,
                truncation_total_chunks=2,
                truncation_ratio=0.5,
            ),
        ),
    )


def _make_worker_failure_with_attempt_ledger(plan: SamplePlan) -> WorkerFailure:
    return WorkerFailure(
        sample_id=plan.sample_id,
        failure_reason="truncation_exhausted",
        truncation_attempted=True,
        truncation_mode="required",
        preferred_5_6_rescue_attempted=True,
        preferred_5_6_status="preferred_5_6_failed",
        failure_attempts=(
            FailureRenderAttempt(
                stage="full",
                seed=plan.seed,
                system_count=6,
                page_count=2,
                content_height_px=1185,
                vertical_fill_ratio=0.81,
                render_rejection_reason=None,
                decision_reason="truncation_required",
                accepted=False,
                verovio_diagnostic_count=1,
            ),
            FailureRenderAttempt(
                stage="full_layout_rescue",
                seed=((plan.seed & 0xFFFFFFFF) ^ 0x5F3759DF) & 0xFFFFFFFF,
                system_count=6,
                page_count=1,
                content_height_px=1120,
                vertical_fill_ratio=0.75,
                render_rejection_reason="right_clearance",
                decision_reason="right_clearance",
                accepted=False,
                verovio_diagnostic_count=0,
            ),
            FailureRenderAttempt(
                stage="truncation_candidate",
                seed=plan.seed + 17,
                chunk_count=1,
                total_chunks=2,
                ratio=0.5,
                system_count=5,
                page_count=1,
                content_height_px=1010,
                vertical_fill_ratio=0.68,
                render_rejection_reason="multi_page",
                decision_reason="multi_page",
                accepted=False,
                verovio_diagnostic_count=0,
            ),
            FailureRenderAttempt(
                stage="truncation_candidate_layout_rescue",
                seed=((plan.seed + 17) & 0xFFFFFFFF) ^ 0x5F3759DF,
                chunk_count=1,
                total_chunks=2,
                ratio=0.5,
                system_count=5,
                page_count=1,
                content_height_px=1010,
                vertical_fill_ratio=0.68,
                render_rejection_reason="right_clearance",
                decision_reason="right_clearance",
                accepted=False,
                verovio_diagnostic_count=2,
            ),
            FailureRenderAttempt(
                stage="truncation_candidate",
                seed=plan.seed + 34,
                chunk_count=2,
                total_chunks=2,
                ratio=1.0,
                system_count=8,
                page_count=2,
                content_height_px=1200,
                vertical_fill_ratio=0.84,
                render_rejection_reason=None,
                decision_reason="post_truncation_required",
                accepted=False,
                verovio_diagnostic_count=0,
            ),
        ),
    )


def _make_fake_plan_sample(entry_groups_by_sample_idx):
    def fake_plan_sample(source_index, recipe, *, sample_idx, base_seed=0, excluded_paths=None):
        del recipe, base_seed
        if sample_idx not in entry_groups_by_sample_idx:
            raise ValueError("missing plan group")
        entries = [source_index.entries[idx] for idx in entry_groups_by_sample_idx[sample_idx]]
        if excluded_paths is not None:
            entries = [entry for entry in entries if entry.path not in excluded_paths]
        if not entries:
            raise ValueError("Cannot compose from an empty source index")
        segments = tuple(
            SourceSegment(source_id=entry.source_id, path=entry.path, order=idx)
            for idx, entry in enumerate(entries)
        )
        return SamplePlan(
            sample_id=f"sample_{sample_idx:08d}",
            seed=sample_idx,
            segments=segments,
            label_transcription=compose_label_transcription(entries),
            source_measure_count=sum(entry.measure_count for entry in entries),
            source_non_empty_line_count=sum(entry.non_empty_line_count for entry in entries),
            source_max_initial_spine_count=max(entry.initial_spine_count for entry in entries),
            segment_count=len(entries),
        )

    return fake_plan_sample


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class _DummyRenderer:
    pass


def _install_fake_pool(monkeypatch, *, outcomes_by_sample_idx, serial_wait=False):
    monkeypatch.setattr(executor_module, "ProcessPool", lambda **kwargs: FakePebblePool(
        outcomes_by_sample_idx=outcomes_by_sample_idx,
        **kwargs,
    ))
    monkeypatch.setattr(executor_module, "VerovioRenderer", _DummyRenderer)
    monkeypatch.setattr(executor_module.mp, "get_context", lambda method: None)
    if serial_wait:
        def wait_one(futures, timeout=None, return_when=None):
            del timeout, return_when
            future_list = list(futures)
            if not future_list:
                return set(), set()
            return {future_list[0]}, set(future_list[1:])

        monkeypatch.setattr(executor_module, "wait", wait_one)


def _make_simple_input_dir(tmp_path: Path, names: tuple[str, ...]) -> Path:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in names:
        (input_dir / f"{name}.krn").write_text("*clefG2\n=1\n4c\n*-\n", encoding="utf-8")
    return input_dir


def _make_tokenizer_dir(tmp_path: Path) -> Path:
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer = Tokenizer(
        models.WordLevel(
            {
                "[UNK]": 0,
                "*clefG2": 1,
                "=1": 2,
                "4c": 3,
                "*-": 4,
                "alpha": 5,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    return tokenizer_dir


def _make_balance_spec(tmp_path: Path, recipe: ProductionRecipe, tokenizer_dir: Path) -> Path:
    del tokenizer_dir
    spec_path = tmp_path / "balance.json"
    spec_path.write_text(
        json.dumps(
            {
                "mode": "spine_aware_line_proxy",
                "tokenizer": {
                    "path": str(DEFAULT_TOKENIZER_DIR),
                    "fingerprint": compute_tokenizer_fingerprint(DEFAULT_TOKENIZER_DIR),
                },
                "recipe_fingerprint": compute_recipe_fingerprint(recipe),
                "candidate_plan_count": 8,
                "recommended_line_count_ranges": {
                    "all": {
                        str(bucket): {"min": bucket, "max": bucket + 1, "center": float(bucket) + 0.5}
                        for bucket in range(1, 7)
                    }
                },
                "vertical_fit_model": {
                    "all": {
                        str(bucket): {
                            "safe_max_line_count": bucket + 1,
                            "median_content_height_px": 1000.0 + bucket,
                            "safe_sample_count": 2,
                        }
                        for bucket in range(1, 7)
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return spec_path


def _install_bundled_spec(monkeypatch, spec_path: Path):
    monkeypatch.setattr(
        executor_module,
        "load_bundled_system_balance_spec",
        lambda: load_system_balance_spec(spec_path),
    )


def _install_fake_balanced_planner(monkeypatch, entry_groups_by_sample_idx):
    fake_plan_sample = _make_fake_plan_sample(entry_groups_by_sample_idx)

    def fake_choose_balanced_plan(
        *,
        source_index,
        recipe,
        sample_idx,
        base_seed,
        excluded_paths,
        spec,
        accepted_system_histogram,
        candidate_plan_count=None,
    ):
        del spec, accepted_system_histogram, candidate_plan_count
        plan = fake_plan_sample(
            source_index,
            recipe,
            sample_idx=sample_idx,
            base_seed=base_seed,
            excluded_paths=excluded_paths,
        )
        return CandidatePlanScore(
            candidate_idx=0,
            plan=plan,
            line_count=plan.source_non_empty_line_count,
            source_max_initial_spine_count=plan.source_max_initial_spine_count,
            spine_class="1",
            target_bucket=1,
            target_center_line_count=1.0,
            in_target_range=True,
            distance_to_bucket=0.0,
            vertical_fit_penalty=0.0,
        )

    monkeypatch.setattr(executor_module, "choose_balanced_plan", fake_choose_balanced_plan)
    monkeypatch.setattr(
        executor_module,
        "_resolve_system_balance_runtime",
        lambda *, recipe, quiet: executor_module.SystemBalanceRuntime(
            mode="spine_aware_line_proxy",
            spec=object(),
        ),
    )


def test_plan_with_quarantine_preserves_planning_exhausted_error(tmp_path, monkeypatch):
    input_dir = _make_simple_input_dir(tmp_path, ("one",))
    source_index = executor_module.build_source_index(input_dir)

    monkeypatch.setattr(
        executor_module,
        "choose_balanced_plan",
        lambda **kwargs: (_ for _ in ()).throw(
            ValueError("All candidate plans were invalid or exhausted: boundary mismatch")
        ),
    )

    with pytest.raises(RuntimeError, match="Failed to build a valid balanced plan"):
        executor_module._plan_with_quarantine(
            source_index=source_index,
            recipe=ProductionRecipe(),
            sample_idx=0,
            base_seed=0,
            quarantined_sources=set(),
            system_balance_runtime=executor_module.SystemBalanceRuntime(
                mode="spine_aware_line_proxy",
                spec=object(),
            ),
            accepted_system_histogram=defaultdict(Counter),
        )


def test_plan_with_quarantine_keeps_no_schedulable_message_for_true_empty_case(
    tmp_path, monkeypatch
):
    input_dir = _make_simple_input_dir(tmp_path, ("one",))
    source_index = executor_module.build_source_index(input_dir)
    quarantined = {entry.path for entry in source_index.entries}

    monkeypatch.setattr(
        executor_module,
        "choose_balanced_plan",
        lambda **kwargs: (_ for _ in ()).throw(
            ValueError("All candidate plans were invalid or exhausted: Cannot compose from an empty source index")
        ),
    )

    with pytest.raises(RuntimeError, match="No schedulable sources remain after applying quarantine"):
        executor_module._plan_with_quarantine(
            source_index=source_index,
            recipe=ProductionRecipe(),
            sample_idx=0,
            base_seed=0,
            quarantined_sources=quarantined,
            system_balance_runtime=executor_module.SystemBalanceRuntime(
                mode="spine_aware_line_proxy",
                spec=object(),
            ),
            accepted_system_histogram=defaultdict(Counter),
        )


def test_executor_smoke_run_writes_truncated_hf_dataset(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "piece.krn").write_text(
        "*clefG2\n*M4/4\n=1\n4c\n=2\n4d\n=3\n4e\n=4\n4f\n*-\n",
        encoding="utf-8",
    )

    def fake_render(render_text, recipe, *, seed, renderer):
        image = np.full((1485, 1050, 3), 255, dtype=np.uint8)
        image[10:70, 10:600] = 0
        system_count = 8 if "=4" in render_text else 4
        return RenderResult(
            image=image,
            render_layers=None,
            svg_diagnostics=SvgLayoutDiagnostics(system_count=system_count, page_count=1),
            bottom_whitespace_ratio=0.10,
            vertical_fill_ratio=0.72,
            bottom_whitespace_px=149,
            top_whitespace_px=33,
            content_height_px=1069,
        )

    recipe = ProductionRecipe(
        render_only_aug=RenderOnlyAugmentationPolicy(
            include_title_probability=0.0,
            include_author_probability=0.0,
            render_pedals_probability=0.0,
            render_pedals_measures_probability=0.0,
            render_instrument_piano_probability=0.0,
            render_sforzando_probability=0.0,
            render_sforzando_per_note_probability=0.0,
            render_accent_probability=0.0,
            render_accent_per_note_probability=0.0,
            render_tempo_probability=0.0,
            render_tempo_include_mm_probability=0.0,
            render_hairpins_probability=0.0,
            render_hairpins_max_spans=1,
            render_dynamic_marks_probability=0.0,
            render_dynamic_marks_min_count=1,
            render_dynamic_marks_max_count=1,
            max_render_attempts=1,
        ),
    )
    tokenizer_dir = _make_tokenizer_dir(tmp_path)
    spec_path = _make_balance_spec(tmp_path, recipe, tokenizer_dir)
    _install_bundled_spec(monkeypatch, spec_path)

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=1,
        max_attempts=1,
        recipe=recipe,
        renderer=object(),
        render_fn=fake_render,
        augment_fn=lambda plan, render_result, recipe: render_result.image,
    )

    assert summary.accepted_samples == 1
    ds = load_from_disk(str(output_dir))
    info = _read_json(summary.run_artifacts_dir / "info.json")
    assert len(ds) == 1
    assert ds[0]["truncation_applied"] is True
    assert ds[0]["initial_kern_spine_count"] == 1
    assert ds[0]["svg_system_count"] == 4
    assert ds[0]["source_ids"] == ["input/piece"]
    assert "initial_kern_spine_count" in ds.features
    assert info["system_balance"]["mandatory"] is True
    assert info["system_balance"]["mode"] == "spine_aware_line_proxy"
    assert info["system_balance"]["spec_path"].endswith("balance.json")


def test_executor_writes_verovio_events_jsonl(tmp_path, monkeypatch):
    input_dir = _make_simple_input_dir(tmp_path, ("a",))
    output_dir = tmp_path / "output"
    _install_fake_pool(
        monkeypatch,
        outcomes_by_sample_idx={0: [_make_worker_success_with_verovio_diagnostic]},
    )
    _install_fake_balanced_planner(monkeypatch, {0: [0]})

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=1,
        num_workers=2,
        max_attempts=1,
        quiet=True,
    )

    info = _read_json(summary.run_artifacts_dir / "info.json")
    events_path = summary.run_artifacts_dir / "verovio_events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]

    assert info["capture_verovio_diagnostics"] is True
    assert info["verovio_events_path"] == str(events_path)
    assert len(events) == 1
    assert events[0]["event"] == "verovio_diagnostic"
    assert events[0]["sample_id"] == "sample_00000000"
    assert events[0]["stage"] == "full"
    assert events[0]["diagnostic_kind"] == "inconsistent_rhythm_analysis"
    assert events[0]["near_line"] == 12
    assert events[0]["source_paths"] == [str((input_dir / "a.krn").resolve())]


def test_executor_writes_verovio_events_for_rejected_outcomes(tmp_path, monkeypatch):
    input_dir = _make_simple_input_dir(tmp_path, ("a", "b"))
    output_dir = tmp_path / "output"
    _install_fake_pool(
        monkeypatch,
        outcomes_by_sample_idx={
            0: [_make_worker_failure_with_verovio_diagnostic],
            1: [_make_worker_success],
        },
    )
    _install_fake_balanced_planner(monkeypatch, {0: [0], 1: [1]})

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=1,
        num_workers=2,
        max_attempts=2,
        quiet=True,
    )

    events_path = summary.run_artifacts_dir / "verovio_events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]

    assert summary.rejected_samples == 1
    assert len(events) == 1
    assert events[0]["sample_id"] == "sample_00000000"
    assert events[0]["stage"] == "truncation_candidate"
    assert events[0]["diagnostic_kind"] == "verovio_error"
    assert events[0]["truncation_chunk_count"] == 1
    assert events[0]["truncation_total_chunks"] == 2
    assert events[0]["truncation_ratio"] == 0.5


def test_executor_writes_failure_events_jsonl_for_rejected_outcomes(tmp_path, monkeypatch):
    input_dir = _make_simple_input_dir(tmp_path, ("a", "b"))
    output_dir = tmp_path / "output"
    _install_fake_pool(
        monkeypatch,
        outcomes_by_sample_idx={
            0: [_make_worker_failure_with_attempt_ledger],
            1: [_make_worker_success],
        },
    )
    _install_fake_balanced_planner(monkeypatch, {0: [0], 1: [1]})

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=1,
        num_workers=2,
        max_attempts=2,
        quiet=True,
    )

    info = _read_json(summary.run_artifacts_dir / "info.json")
    events_path = summary.run_artifacts_dir / "failure_events.jsonl"
    events = _read_jsonl(events_path)

    assert summary.rejected_samples == 1
    assert info["failure_events_path"] == str(events_path)
    assert len(events) == 1
    assert events[0]["event"] == "failure_trace"
    assert events[0]["sample_id"] == "sample_00000000"
    assert events[0]["source_paths"] == [str((input_dir / "a.krn").resolve())]
    assert events[0]["target_bucket"] == 1
    assert events[0]["planned_line_count"] == 4
    assert events[0]["candidate_in_target_range"] is True


def test_executor_failure_events_preserve_truncation_attempt_ledger(tmp_path, monkeypatch):
    input_dir = _make_simple_input_dir(tmp_path, ("a", "b"))
    output_dir = tmp_path / "output"
    _install_fake_pool(
        monkeypatch,
        outcomes_by_sample_idx={
            0: [_make_worker_failure_with_attempt_ledger],
            1: [_make_worker_success],
        },
    )
    _install_fake_balanced_planner(monkeypatch, {0: [0], 1: [1]})

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=1,
        num_workers=2,
        max_attempts=2,
        quiet=True,
    )

    events = _read_jsonl(summary.run_artifacts_dir / "failure_events.jsonl")

    assert len(events) == 1
    assert events[0]["failure_reason"] == "truncation_exhausted"
    assert events[0]["truncation_mode"] == "required"
    assert events[0]["preferred_5_6_rescue_attempted"] is True
    assert events[0]["preferred_5_6_status"] == "preferred_5_6_failed"
    assert [attempt["stage"] for attempt in events[0]["attempts"]] == [
        "full",
        "full_layout_rescue",
        "truncation_candidate",
        "truncation_candidate_layout_rescue",
        "truncation_candidate",
    ]
    assert events[0]["attempts"][0]["decision_reason"] == "truncation_required"
    assert events[0]["attempts"][2]["chunk_count"] == 1
    assert events[0]["attempts"][2]["total_chunks"] == 2
    assert events[0]["attempts"][2]["ratio"] == 0.5
    assert events[0]["attempts"][2]["decision_reason"] == "multi_page"
    assert events[0]["attempts"][3]["decision_reason"] == "right_clearance"
    assert events[0]["attempts"][4]["decision_reason"] == "post_truncation_required"


def test_executor_writes_success_events_jsonl_for_committed_successes(tmp_path, monkeypatch):
    input_dir = _make_simple_input_dir(tmp_path, ("a",))
    output_dir = tmp_path / "output"
    _install_fake_pool(
        monkeypatch,
        outcomes_by_sample_idx={0: [_make_worker_success]},
    )
    _install_fake_balanced_planner(monkeypatch, {0: [0]})

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=1,
        num_workers=2,
        max_attempts=1,
        quiet=True,
    )

    ds = load_from_disk(str(output_dir))
    info = _read_json(summary.run_artifacts_dir / "info.json")
    events_path = summary.run_artifacts_dir / "success_events.jsonl"
    events = _read_jsonl(events_path)

    assert summary.accepted_samples == 1
    assert info["success_events_path"] == str(events_path)
    assert len(events) == 1
    assert events[0]["event"] == "success_trace"
    assert events[0]["sample_id"] == "sample_00000000"
    assert events[0]["source_paths"] == [str((input_dir / "a.krn").resolve())]
    assert events[0]["target_bucket"] == 1
    assert events[0]["planned_line_count"] == 4
    assert events[0]["candidate_in_target_range"] is True
    assert events[0]["committed_to_dataset"] is True
    assert events[0]["full_render_system_count"] == 4
    assert events[0]["accepted_render_system_count"] == 4
    assert events[0]["truncation_applied"] is False
    assert "full_render_system_count" not in ds.features
    assert "target_bucket" not in ds.features


def test_executor_success_events_preserve_truncation_metadata(tmp_path, monkeypatch):
    input_dir = _make_simple_input_dir(tmp_path, ("a",))
    output_dir = tmp_path / "output"
    _install_fake_pool(
        monkeypatch,
        outcomes_by_sample_idx={0: [_make_worker_success_with_truncation]},
    )
    _install_fake_balanced_planner(monkeypatch, {0: [0]})

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=1,
        num_workers=2,
        max_attempts=1,
        quiet=True,
    )

    events = _read_jsonl(summary.run_artifacts_dir / "success_events.jsonl")

    assert len(events) == 1
    assert events[0]["target_bucket"] == 1
    assert events[0]["full_render_system_count"] == 7
    assert events[0]["accepted_render_system_count"] == 5
    assert events[0]["full_render_rejection_reason"] == "multi_page"
    assert events[0]["preferred_5_6_status"] == "preferred_5_6_truncated"
    assert events[0]["truncation_attempted"] is True
    assert events[0]["truncation_rescued"] is True
    assert events[0]["truncation_applied"] is True
    assert events[0]["truncation_reason"] == "system_count_policy"
    assert events[0]["truncation_ratio"] == 0.5


def test_executor_writes_redesigned_augmentation_events_and_previews(tmp_path, monkeypatch):
    input_dir = _make_simple_input_dir(tmp_path, ("a", "b"))
    output_dir = tmp_path / "output"
    _install_fake_pool(
        monkeypatch,
        outcomes_by_sample_idx={
            0: [
                lambda plan: _make_augmented_worker_success(
                    plan,
                    outer_gate_passed=False,
                    final_geometry_applied=False,
                    initial_oob_failure_reason="centroid_shift_x",
                    retry_oob_failure_reason="min_margin",
                    outer_gate_failure_reason="quality:min_margin",
                )
            ],
            1: [
                lambda plan: _make_augmented_worker_success(
                    plan,
                    outer_gate_passed=True,
                    final_geometry_applied=False,
                    initial_oob_failure_reason="centroid_shift_x",
                    retry_oob_failure_reason="min_margin",
                    outer_gate_failure_reason=None,
                )
            ],
        },
    )
    _install_fake_balanced_planner(monkeypatch, {0: [0], 1: [1]})

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=2,
        num_workers=2,
        max_attempts=2,
        quiet=True,
    )

    info = _read_json(summary.run_artifacts_dir / "info.json")
    events_path = summary.run_artifacts_dir / "augmentation_events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    previews_dir = summary.run_artifacts_dir / "augmentation_previews"

    assert info["augmentation_events_path"] == str(events_path)
    assert info["augmentation_previews_dir"] == str(previews_dir)
    assert len(events) == 2
    assert events[0]["initial_geometry"]["sampled"] is True
    assert events[0]["retry_geometry"]["conservative"] is True
    assert events[0]["selected_geometry"]["sampled"] is False
    assert events[0]["initial_oob_gate"]["failure_reason"] == "centroid_shift_x"
    assert events[0]["retry_oob_gate"]["failure_reason"] == "min_margin"
    assert events[0]["outer_gate"]["failure_reason"] == "quality:min_margin"
    assert (previews_dir / "sample_00000000" / "base.jpg").exists()
    assert (previews_dir / "sample_00000000" / "pre_augraphy.jpg").exists()
    assert (previews_dir / "sample_00000000" / "final.jpg").exists()
    assert (previews_dir / "sample_00000000" / "trace.json").exists()
    assert (previews_dir / "sample_00000001" / "trace.json").exists()


def test_executor_tracks_new_augmentation_summary_counters(tmp_path, monkeypatch):
    input_dir = _make_simple_input_dir(tmp_path, ("a", "b", "c"))
    output_dir = tmp_path / "output"
    _install_fake_pool(
        monkeypatch,
        outcomes_by_sample_idx={
            0: [
                lambda plan: _make_augmented_worker_success(
                    plan,
                    outer_gate_passed=False,
                    final_geometry_applied=False,
                    initial_oob_failure_reason="centroid_shift_x",
                    retry_oob_failure_reason="min_margin",
                    outer_gate_failure_reason="quality:min_margin",
                )
            ],
            1: [
                lambda plan: _make_augmented_worker_success(
                    plan,
                    outer_gate_passed=True,
                    final_geometry_applied=False,
                    initial_oob_failure_reason="centroid_shift_x",
                    retry_oob_failure_reason="min_margin",
                )
            ],
            2: [
                lambda plan: _make_augmented_worker_success(
                    plan,
                    outer_gate_passed=True,
                    final_geometry_applied=True,
                )
            ],
        },
    )
    _install_fake_balanced_planner(monkeypatch, {0: [0], 1: [1], 2: [2]})

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=3,
        num_workers=2,
        max_attempts=3,
        quiet=True,
    )

    info = _read_json(summary.run_artifacts_dir / "info.json")
    progress = _read_json(summary.run_artifacts_dir / "progress.json")

    assert progress["final_geometry_counts"] == {
        "base_image_returned": 1,
        "geometry_discarded": 1,
        "geometry_survived": 1,
    }
    assert progress["oob_failure_reason_counts"] == {
        "centroid_shift_x": 2,
        "min_margin": 2,
    }
    assert progress["outer_gate_failure_reason_counts"] == {"quality:min_margin": 1}
    assert info["snapshot"]["final_geometry_counts"]["geometry_survived"] == 1
    assert info["finalization"]["snapshot"]["final_geometry_counts"]["geometry_discarded"] == 1


def test_executor_skips_invalid_sources_and_records_auto_quarantine(tmp_path, monkeypatch):
    input_dir = _make_simple_input_dir(tmp_path, ("valid",))
    output_dir = tmp_path / "output"
    (input_dir / "invalid.krn").write_text(
        "*clefF4\t*clefG2\n"
        "*^\t*\n"
        "*\t*^\t*\n"
        "4c\t4e\t4g\t4b\n"
        "*v\t*v\t*v\t*\n"
        "=\t=\t=\n",
        encoding="utf-8",
    )
    tokenizer_dir = _make_tokenizer_dir(tmp_path)
    spec_path = _make_balance_spec(tmp_path, ProductionRecipe(), tokenizer_dir)
    _install_bundled_spec(monkeypatch, spec_path)

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=1,
        max_attempts=1,
        renderer=object(),
        render_fn=lambda render_text, recipe, *, seed, renderer: RenderResult(
            image=np.pad(
                np.zeros((60, 590, 3), dtype=np.uint8),
                ((10, 1415), (10, 450), (0, 0)),
                constant_values=255,
            ),
            render_layers=None,
            svg_diagnostics=SvgLayoutDiagnostics(system_count=4, page_count=1),
            bottom_whitespace_ratio=0.10,
            vertical_fill_ratio=0.72,
            bottom_whitespace_px=149,
            top_whitespace_px=33,
            content_height_px=1069,
        ),
        augment_fn=lambda plan, render_result, recipe: render_result.image,
        quiet=True,
    )

    info = _read_json(summary.run_artifacts_dir / "info.json")
    quarantined = _read_json(summary.run_artifacts_dir / "quarantined_sources.json")
    ds = load_from_disk(str(output_dir))

    assert summary.accepted_samples == 1
    assert len(ds) == 1
    assert info["auto_quarantined_source_count"] == 1
    assert len(info["invalid_source_examples"]) == 1
    assert info["invalid_source_examples"][0]["reason_code"] == "width_mismatch"
    assert quarantined["quarantined_sources"] == [str((input_dir / "invalid.krn").resolve())]


def test_executor_fails_cleanly_when_all_sources_are_invalid(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "invalid.krn").write_text(
        "*clefF4\t*clefG2\n"
        "*^\t*\n"
        "*\t*^\t*\n"
        "4c\t4e\t4g\t4b\n"
        "*v\t*v\t*v\t*\n"
        "=\t=\t=\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="No schedulable sources remain after applying quarantine"):
        run_dataset_generation(
            input_dirs=(input_dir,),
            output_dir=output_dir,
            target_samples=1,
            max_attempts=1,
            quiet=True,
        )

    run_dirs = sorted((output_dir.parent / "_runs" / output_dir.name).iterdir())
    assert len(run_dirs) == 1
    run_artifacts_dir = run_dirs[0]
    info = _read_json(run_artifacts_dir / "info.json")
    quarantined = _read_json(run_artifacts_dir / "quarantined_sources.json")

    assert info["auto_quarantined_source_count"] == 1
    assert quarantined["quarantined_sources"] == [str((input_dir / "invalid.krn").resolve())]


def test_executor_resume_recovers_after_failure_without_duplicates(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    for idx in range(3):
        (input_dir / f"{idx:03d}.krn").write_text("*clefG2\n=1\n4c\n*-\n", encoding="utf-8")

    seen_calls: list[int] = []

    def flaky_render(render_text, recipe, *, seed, renderer):
        sample_idx = len(seen_calls)
        seen_calls.append(sample_idx)
        if sample_idx == 1:
            raise RuntimeError("simulated interruption")
        image = np.full((1485, 1050, 3), 255, dtype=np.uint8)
        image[10:70, 10:600] = 0
        return RenderResult(
            image=image,
            render_layers=None,
            svg_diagnostics=SvgLayoutDiagnostics(system_count=4, page_count=1),
            bottom_whitespace_ratio=0.10,
            vertical_fill_ratio=0.72,
            bottom_whitespace_px=149,
            top_whitespace_px=33,
            content_height_px=1069,
        )

    recipe = ProductionRecipe(
        render_only_aug=RenderOnlyAugmentationPolicy(
            include_title_probability=0.0,
            include_author_probability=0.0,
            render_pedals_probability=0.0,
            render_pedals_measures_probability=0.0,
            render_instrument_piano_probability=0.0,
            render_sforzando_probability=0.0,
            render_sforzando_per_note_probability=0.0,
            render_accent_probability=0.0,
            render_accent_per_note_probability=0.0,
            render_tempo_probability=0.0,
            render_tempo_include_mm_probability=0.0,
            render_hairpins_probability=0.0,
            render_hairpins_max_spans=1,
            render_dynamic_marks_probability=0.0,
            render_dynamic_marks_min_count=1,
            render_dynamic_marks_max_count=1,
            max_render_attempts=1,
        ),
    )
    tokenizer_dir = _make_tokenizer_dir(tmp_path)
    spec_path = _make_balance_spec(tmp_path, recipe, tokenizer_dir)
    _install_bundled_spec(monkeypatch, spec_path)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_dataset_generation(
            input_dirs=(input_dir,),
            output_dir=output_dir,
            target_samples=2,
            max_attempts=4,
            recipe=recipe,
            renderer=object(),
            render_fn=flaky_render,
            augment_fn=lambda plan, render_result, recipe: render_result.image,
        )

    def stable_render(render_text, recipe, *, seed, renderer):
        image = np.full((1485, 1050, 3), 255, dtype=np.uint8)
        image[10:70, 10:600] = 0
        return RenderResult(
            image=image,
            render_layers=None,
            svg_diagnostics=SvgLayoutDiagnostics(system_count=4, page_count=1),
            bottom_whitespace_ratio=0.10,
            vertical_fill_ratio=0.72,
        )

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=2,
        max_attempts=4,
        recipe=recipe,
        resume_mode="auto",
        renderer=object(),
        render_fn=stable_render,
        augment_fn=lambda plan, render_result, recipe: render_result.image,
    )

    ds = load_from_disk(str(output_dir))
    assert summary.accepted_samples == 2
    assert len(ds) == 2
    assert len(set(ds["sample_id"])) == 2


def test_executor_resume_mode_require_fails_without_state(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "piece.krn").write_text("*clefG2\n=1\n4c\n*-\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="resume_mode=require"):
        run_dataset_generation(
            input_dirs=(input_dir,),
            output_dir=output_dir,
            target_samples=1,
            resume_mode="require",
        )


def test_executor_resume_detects_config_mismatch(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "piece.krn").write_text("*clefG2\n=1\n4c\n*-\n", encoding="utf-8")

    def crash_render(render_text, recipe, *, seed, renderer):
        raise RuntimeError("crash")

    with pytest.raises(RuntimeError, match="crash"):
        run_dataset_generation(
            input_dirs=(input_dir,),
            output_dir=output_dir,
            target_samples=1,
            base_seed=1,
            renderer=object(),
            render_fn=crash_render,
        )

    with pytest.raises(RuntimeError, match="fingerprint changed"):
        run_dataset_generation(
            input_dirs=(input_dir,),
            output_dir=output_dir,
            target_samples=1,
            base_seed=2,
            resume_mode="auto",
        )


def test_executor_timeout_retry_succeeds_without_aborting(tmp_path, monkeypatch):
    input_dir = _make_simple_input_dir(tmp_path, ("piece",))
    output_dir = tmp_path / "output"
    _install_fake_pool(
        monkeypatch,
        outcomes_by_sample_idx={
            0: [TimeoutError(), _make_worker_success],
        },
    )

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=1,
        num_workers=2,
        max_attempts=1,
        quiet=True,
    )

    ds = load_from_disk(str(output_dir))
    info = _read_json(summary.run_artifacts_dir / "info.json")
    timeout_events = (summary.run_artifacts_dir / "timeout_events.jsonl").read_text(
        encoding="utf-8"
    ).strip().splitlines()

    assert summary.accepted_samples == 1
    assert len(ds) == 1
    assert info["snapshot"]["retry_counts"]["timeout"] == 1
    assert len(timeout_events) == 1
    assert json.loads(timeout_events[0])["will_retry"] is True
    assert json.loads(timeout_events[0])["crash_artifact_json_path"] is None


def test_executor_terminal_timeout_writes_crash_artifact(tmp_path, monkeypatch):
    input_dir = _make_simple_input_dir(tmp_path, ("a", "b", "c"))
    output_dir = tmp_path / "output"
    _install_fake_pool(
        monkeypatch,
        outcomes_by_sample_idx={
            0: [TimeoutError()],
            1: [_make_worker_success],
        },
        serial_wait=True,
    )
    _install_fake_balanced_planner(monkeypatch, {0: [0, 1], 1: [2]})

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=1,
        num_workers=2,
        max_attempts=2,
        failure_policy="throughput",
        quiet=True,
    )

    timeout_events = (summary.run_artifacts_dir / "timeout_events.jsonl").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    timeout_event = json.loads(timeout_events[0])
    crash_artifact_path = Path(timeout_event["crash_artifact_json_path"])
    crash_artifact = _read_json(crash_artifact_path)
    info = _read_json(summary.run_artifacts_dir / "info.json")

    assert crash_artifact_path.exists()
    assert crash_artifact["event_type"] == "timeout"
    assert crash_artifact["stage_unknown_due_to_timeout"] is True
    assert crash_artifact["source_paths"] == [
        str((input_dir / "a.krn").resolve()),
        str((input_dir / "b.krn").resolve()),
    ]
    assert crash_artifact["repro_entries"][0]["stage"] == "full"
    assert crash_artifact["repro_entries"][0]["render_transcription_path"].endswith("_full.krn")
    full_render_path = Path(crash_artifact["repro_entries"][0]["render_transcription_path"])
    assert full_render_path.exists()
    assert full_render_path.read_text(encoding="utf-8").rstrip().endswith("*-")
    assert any(entry["stage"] == "truncation_candidate" for entry in crash_artifact["repro_entries"])
    assert timeout_event["crash_repro_stage_count"] == len(crash_artifact["repro_entries"])
    assert info["snapshot"]["terminal_timeout_crash_artifacts"] == 1


def test_executor_process_expired_does_not_poison_remaining_tasks(tmp_path, monkeypatch):
    input_dir = _make_simple_input_dir(tmp_path, ("a", "b"))
    output_dir = tmp_path / "output"
    _install_fake_pool(
        monkeypatch,
        outcomes_by_sample_idx={
            0: [ProcessExpired("boom", code=11, pid=4321)],
            1: [_make_worker_success],
        },
    )
    _install_fake_balanced_planner(monkeypatch, {0: [0], 1: [1]})

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=1,
        num_workers=2,
        max_attempts=2,
        quiet=True,
    )

    ds = load_from_disk(str(output_dir))
    info = _read_json(summary.run_artifacts_dir / "info.json")
    process_expired_events = (
        summary.run_artifacts_dir / "process_expired_events.jsonl"
    ).read_text(encoding="utf-8").strip().splitlines()

    assert summary.accepted_samples == 1
    assert summary.rejected_samples == 1
    assert len(ds) == 1
    assert info["snapshot"]["failure_reason_counts"]["process_expired"] == 1
    assert len(process_expired_events) == 1
    process_expired_event = json.loads(process_expired_events[0])
    crash_artifact_path = Path(process_expired_event["crash_artifact_json_path"])
    crash_artifact = _read_json(crash_artifact_path)
    assert process_expired_event["pid"] == 4321
    assert process_expired_event["crash_repro_stage_count"] == len(crash_artifact["repro_entries"])
    assert crash_artifact_path.exists()
    assert crash_artifact["event_type"] == "process_expired"
    assert crash_artifact["exception"]["pid"] == 4321
    assert crash_artifact["repro_entries"][0]["stage"] == "full"
    assert info["snapshot"]["terminal_process_expired_crash_artifacts"] == 1


def test_executor_terminal_timeout_quarantines_all_sources_in_plan(tmp_path, monkeypatch):
    input_dir = _make_simple_input_dir(tmp_path, ("a", "b", "c"))
    output_dir = tmp_path / "output"
    _install_fake_pool(
        monkeypatch,
        outcomes_by_sample_idx={
            0: [TimeoutError()],
            1: [_make_worker_success],
            2: [_make_worker_success],
        },
        serial_wait=True,
    )
    _install_fake_balanced_planner(monkeypatch, {0: [0, 1], 1: [1], 2: [2]})

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=1,
        num_workers=2,
        max_attempts=3,
        failure_policy="throughput",
        quiet=True,
    )

    ds = load_from_disk(str(output_dir))
    info = _read_json(summary.run_artifacts_dir / "info.json")
    quarantined = _read_json(summary.run_artifacts_dir / "quarantined_sources.json")
    timeout_events = (summary.run_artifacts_dir / "timeout_events.jsonl").read_text(
        encoding="utf-8"
    ).strip().splitlines()

    expected_paths = {
        str((input_dir / "a.krn").resolve()),
        str((input_dir / "b.krn").resolve()),
    }
    assert summary.accepted_samples == 1
    assert summary.rejected_samples == 2
    assert len(ds) == 1
    assert ds[0]["source_ids"] == ["input/c"]
    assert info["snapshot"]["failure_reason_counts"]["timeout"] == 1
    assert info["snapshot"]["failure_reason_counts"]["quarantined"] == 1
    assert set(quarantined["quarantined_sources"]) == expected_paths
    assert json.loads(timeout_events[0])["dropped_pending_tasks"] == 1


def test_executor_skips_done_batch_future_removed_by_terminal_timeout_quarantine(
    tmp_path, monkeypatch
):
    input_dir = _make_simple_input_dir(tmp_path, ("a", "b", "c"))
    output_dir = tmp_path / "output"
    _install_fake_pool(
        monkeypatch,
        outcomes_by_sample_idx={
            0: [TimeoutError()],
            1: [_make_worker_success],
            2: [_make_worker_success],
        },
    )
    _install_fake_balanced_planner(monkeypatch, {0: [0, 1], 1: [1], 2: [2]})

    def wait_done_batch(futures, timeout=None, return_when=None):
        del timeout, return_when
        future_list = list(futures)
        return future_list, []

    monkeypatch.setattr(executor_module, "wait", wait_done_batch)

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=1,
        num_workers=3,
        max_attempts=3,
        failure_policy="throughput",
        quiet=True,
    )

    ds = load_from_disk(str(output_dir))
    info = _read_json(summary.run_artifacts_dir / "info.json")
    quarantined = _read_json(summary.run_artifacts_dir / "quarantined_sources.json")
    timeout_events = (summary.run_artifacts_dir / "timeout_events.jsonl").read_text(
        encoding="utf-8"
    ).strip().splitlines()

    assert summary.accepted_samples == 1
    assert summary.rejected_samples == 2
    assert len(ds) == 1
    assert ds[0]["source_ids"] == ["input/c"]
    assert info["snapshot"]["failure_reason_counts"]["timeout"] == 1
    assert info["snapshot"]["failure_reason_counts"]["quarantined"] == 1
    assert set(quarantined["quarantined_sources"]) == {
        str((input_dir / "a.krn").resolve()),
        str((input_dir / "b.krn").resolve()),
    }
    assert json.loads(timeout_events[0])["dropped_pending_tasks"] == 1


def test_executor_semantic_rejection_does_not_quarantine_sources(tmp_path, monkeypatch):
    input_dir = _make_simple_input_dir(tmp_path, ("a", "b"))
    output_dir = tmp_path / "output"
    _install_fake_pool(
        monkeypatch,
        outcomes_by_sample_idx={
            0: [
                WorkerFailure(
                    sample_id="sample_00000000",
                    failure_reason="truncation_exhausted",
                    truncation_attempted=True,
                )
            ],
            1: [_make_worker_success],
        },
        serial_wait=True,
    )
    _install_fake_balanced_planner(monkeypatch, {0: [0], 1: [0]})

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=1,
        num_workers=2,
        max_attempts=2,
        quiet=True,
    )

    ds = load_from_disk(str(output_dir))
    info = _read_json(summary.run_artifacts_dir / "info.json")
    quarantined = _read_json(summary.run_artifacts_dir / "quarantined_sources.json")

    assert summary.accepted_samples == 1
    assert summary.rejected_samples == 1
    assert len(ds) == 1
    assert info["snapshot"]["failure_reason_counts"]["truncation_exhausted"] == 1
    assert quarantined["quarantined_sources"] == []
    assert not (summary.run_artifacts_dir / "crash_samples").exists()


def test_executor_does_not_write_failure_events_for_discarded_after_target_successes(
    tmp_path, monkeypatch
):
    input_dir = _make_simple_input_dir(tmp_path, ("a", "b"))
    output_dir = tmp_path / "output"
    _install_fake_pool(
        monkeypatch,
        outcomes_by_sample_idx={
            0: [_make_worker_success],
            1: [_make_worker_success],
        },
        serial_wait=True,
    )
    _install_fake_balanced_planner(monkeypatch, {0: [0], 1: [1]})

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=1,
        num_workers=2,
        max_attempts=2,
        quiet=True,
    )

    info = _read_json(summary.run_artifacts_dir / "info.json")
    failure_events_path = summary.run_artifacts_dir / "failure_events.jsonl"
    success_events_path = summary.run_artifacts_dir / "success_events.jsonl"

    assert summary.accepted_samples == 1
    assert summary.rejected_samples == 0
    assert info["snapshot"]["failure_reason_counts"]["discarded_after_target"] == 1
    assert info["failure_events_path"] == str(failure_events_path)
    assert info["success_events_path"] == str(success_events_path)
    assert _read_jsonl(failure_events_path) == []
    success_events = _read_jsonl(success_events_path)
    assert len(success_events) == 2
    assert success_events[0]["committed_to_dataset"] is True
    assert success_events[1]["committed_to_dataset"] is False


def test_executor_tracks_requested_target_buckets_and_candidate_hits(tmp_path, monkeypatch):
    input_dir = _make_simple_input_dir(tmp_path, ("piece",))
    output_dir = tmp_path / "output"
    tokenizer_dir = _make_tokenizer_dir(tmp_path)
    recipe = ProductionRecipe(
        render_only_aug=RenderOnlyAugmentationPolicy(
            include_title_probability=0.0,
            include_author_probability=0.0,
            render_pedals_probability=0.0,
            render_pedals_measures_probability=0.0,
            render_instrument_piano_probability=0.0,
            render_sforzando_probability=0.0,
            render_sforzando_per_note_probability=0.0,
            render_accent_probability=0.0,
            render_accent_per_note_probability=0.0,
            render_tempo_probability=0.0,
            render_tempo_include_mm_probability=0.0,
            render_hairpins_probability=0.0,
            render_hairpins_max_spans=1,
            render_dynamic_marks_probability=0.0,
            render_dynamic_marks_min_count=1,
            render_dynamic_marks_max_count=1,
            max_render_attempts=1,
        ),
    )
    spec_path = _make_balance_spec(tmp_path, recipe, tokenizer_dir)
    _install_bundled_spec(monkeypatch, spec_path)

    def fake_balanced_plan(**kwargs):
        plan = _make_fake_plan_sample({0: [0]})(kwargs["source_index"], kwargs["recipe"], sample_idx=0)
        return CandidatePlanScore(
            candidate_idx=2,
            plan=plan,
            line_count=plan.source_non_empty_line_count,
            source_max_initial_spine_count=plan.source_max_initial_spine_count,
            spine_class="1",
            target_bucket=3,
            target_center_line_count=5.5,
            in_target_range=True,
            distance_to_bucket=0.0,
            vertical_fit_penalty=0.0,
        )

    monkeypatch.setattr(executor_module, "choose_balanced_plan", fake_balanced_plan)

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=1,
        max_attempts=1,
        recipe=recipe,
        renderer=object(),
        render_fn=lambda render_text, recipe, *, seed, renderer: RenderResult(
            image=np.pad(
                np.zeros((60, 590, 3), dtype=np.uint8),
                ((10, 1415), (10, 450), (0, 0)),
                constant_values=255,
            ),
            render_layers=None,
            svg_diagnostics=SvgLayoutDiagnostics(system_count=4, page_count=1),
            bottom_whitespace_ratio=0.10,
            vertical_fill_ratio=0.72,
            bottom_whitespace_px=149,
            top_whitespace_px=33,
            content_height_px=1069,
        ),
        augment_fn=lambda plan, render_result, recipe: render_result.image,
        quiet=True,
    )

    info = _read_json(summary.run_artifacts_dir / "info.json")
    progress = _read_json(summary.run_artifacts_dir / "progress.json")

    assert info["snapshot"]["requested_target_bucket_histogram"]["3"] == 1
    assert info["snapshot"]["candidate_hit_counts"]["inside_target_bucket"] == 1
    assert info["snapshot"]["full_render_system_histogram"]["4"] == 1
    assert progress["requested_target_bucket_histogram"]["3"] == 1
    assert progress["candidate_hit_counts"]["inside_target_bucket"] == 1
    assert progress["full_render_system_histogram"]["4"] == 1


def test_executor_fails_when_bundled_spec_is_missing(tmp_path, monkeypatch):
    input_dir = _make_simple_input_dir(tmp_path, ("piece",))
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        executor_module,
        "load_bundled_system_balance_spec",
        lambda: (_ for _ in ()).throw(FileNotFoundError("missing spec")),
    )

    with pytest.raises(RuntimeError, match="Bundled system balance spec could not be loaded"):
        run_dataset_generation(
            input_dirs=(input_dir,),
            output_dir=output_dir,
            target_samples=1,
            max_attempts=1,
            quiet=True,
        )


def test_executor_fails_when_bundled_spec_fingerprint_mismatches(tmp_path, monkeypatch):
    input_dir = _make_simple_input_dir(tmp_path, ("piece",))
    output_dir = tmp_path / "output"
    tokenizer_dir = _make_tokenizer_dir(tmp_path)
    spec_path = tmp_path / "bad_balance.json"
    spec_path.write_text(
        json.dumps(
            {
                "mode": "spine_aware_line_proxy",
                "tokenizer": {"path": str(tokenizer_dir.resolve()), "fingerprint": "bad"},
                "recipe_fingerprint": "bad",
                "candidate_plan_count": 8,
                "recommended_line_count_ranges": {
                    "all": {
                        str(bucket): {"min": bucket, "max": bucket + 1, "center": float(bucket) + 0.5}
                        for bucket in range(1, 7)
                    }
                },
                "vertical_fit_model": {},
            }
        ),
        encoding="utf-8",
    )
    _install_bundled_spec(monkeypatch, spec_path)

    with pytest.raises(RuntimeError, match="recipe fingerprint|tokenizer fingerprint"):
        run_dataset_generation(
            input_dirs=(input_dir,),
            output_dir=output_dir,
            target_samples=1,
            max_attempts=1,
            quiet=True,
        )


def test_executor_writes_pixel_metrics_and_layout_summary_stats(tmp_path):
    input_dir = _make_simple_input_dir(tmp_path, ("piece",))
    output_dir = tmp_path / "output"

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=1,
        max_attempts=1,
        renderer=object(),
        render_fn=lambda render_text, recipe, *, seed, renderer: RenderResult(
            image=np.pad(
                np.zeros((120, 600, 3), dtype=np.uint8),
                ((33, 1332), (10, 440), (0, 0)),
                constant_values=255,
            ),
            render_layers=None,
            svg_diagnostics=SvgLayoutDiagnostics(system_count=4, page_count=1),
            bottom_whitespace_ratio=149 / 1485,
            vertical_fill_ratio=1069 / 1485,
            bottom_whitespace_px=149,
            top_whitespace_px=33,
            content_height_px=1069,
        ),
        augment_fn=lambda plan, render_result, recipe: np.full((1485, 1050, 3), 127, dtype=np.uint8),
        quiet=True,
    )

    ds = load_from_disk(str(output_dir))
    info = _read_json(summary.run_artifacts_dir / "info.json")
    progress = _read_json(summary.run_artifacts_dir / "progress.json")

    assert ds[0]["bottom_whitespace_px"] == 149
    assert ds[0]["top_whitespace_px"] == 33
    assert ds[0]["content_height_px"] == 1069
    assert progress["bottom_whitespace_px_stats"]["count"] == 1
    assert progress["bottom_whitespace_px_stats"]["mean"] == pytest.approx(149.0)
    assert progress["top_whitespace_px_stats"]["p50"] == pytest.approx(33.0)
    assert progress["content_height_px_stats"]["p95"] == pytest.approx(1069.0)
    assert progress["bottom_whitespace_ratio_stats"]["mean"] == pytest.approx(149 / 1485)
    assert progress["vertical_fill_ratio_stats"]["mean"] == pytest.approx(1069 / 1485)
    assert info["layout_summary"]["bottom_whitespace_px_stats"]["count"] == 1
    assert info["layout_summary"]["top_whitespace_px_stats"]["mean"] == pytest.approx(33.0)
    assert info["layout_summary"]["content_height_px_stats"]["max"] == pytest.approx(1069.0)


def test_executor_resume_preserves_retry_counters_after_interrupted_finalize(
    tmp_path,
    monkeypatch,
):
    input_dir = _make_simple_input_dir(tmp_path, ("piece",))
    output_dir = tmp_path / "output"
    _install_fake_pool(
        monkeypatch,
        outcomes_by_sample_idx={
            0: [TimeoutError(), _make_worker_success],
        },
    )

    original_finalize = executor_module.ResumableShardStore.finalize

    def fail_finalize(self):
        raise RuntimeError("interrupt after commit")

    monkeypatch.setattr(executor_module.ResumableShardStore, "finalize", fail_finalize)
    with pytest.raises(RuntimeError, match="interrupt after commit"):
        run_dataset_generation(
            input_dirs=(input_dir,),
            output_dir=output_dir,
            target_samples=1,
            num_workers=2,
            max_attempts=1,
            quiet=True,
        )

    monkeypatch.setattr(executor_module.ResumableShardStore, "finalize", original_finalize)
    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=1,
        num_workers=2,
        max_attempts=1,
        resume_mode="auto",
        quiet=True,
    )

    ds = load_from_disk(str(output_dir))
    info = _read_json(summary.run_artifacts_dir / "info.json")

    assert summary.accepted_samples == 1
    assert len(ds) == 1
    assert info["snapshot"]["retry_counts"]["timeout"] == 1


def test_executor_final_snapshot_preserves_counters_without_pending_rows(tmp_path, monkeypatch):
    input_dir = _make_simple_input_dir(tmp_path, ("piece",))
    output_dir = tmp_path / "output"
    _install_fake_pool(
        monkeypatch,
        outcomes_by_sample_idx={
            0: [_make_worker_success],
            1: [_make_worker_success],
        },
        serial_wait=True,
    )
    _install_fake_balanced_planner(monkeypatch, {0: [0], 1: [0]})

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=1,
        num_workers=2,
        max_attempts=2,
        quiet=True,
    )

    info = _read_json(summary.run_artifacts_dir / "info.json")
    progress = _read_json(summary.run_artifacts_dir / "progress.json")

    assert summary.accepted_samples == 1
    assert summary.attempted_samples == 2
    assert progress["failure_reason_counts"]["discarded_after_target"] == 1
    assert info["snapshot"]["next_sample_idx"] == 2
    assert info["snapshot"]["failure_reason_counts"]["discarded_after_target"] == 1
    assert info["finalization"]["snapshot"]["failure_reason_counts"]["discarded_after_target"] == 1


def test_executor_end_to_end_save_and_load_with_real_renderer(tmp_path):
    pytest.importorskip("verovio")

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    fixture_path = Path("binaries/accumulating_rhythm/tests/fixtures/in_1.krn")
    (input_dir / "piece.krn").write_text(fixture_path.read_text(encoding="utf-8"), encoding="utf-8")

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=1,
        num_workers=1,
        max_attempts=8,
    )

    ds = load_from_disk(str(output_dir))
    assert summary.accepted_samples == 1
    assert len(ds) == 1
    assert ds[0]["image"].size == (1050, 1485)
    grouped = Counter((row["initial_kern_spine_count"], row["svg_system_count"]) for row in ds)
    assert grouped == Counter({(int(ds[0]["initial_kern_spine_count"]), int(ds[0]["svg_system_count"])): 1})


def test_executor_multiprocessing_smoke_with_real_renderer(tmp_path):
    pytest.importorskip("verovio")

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    fixture_path = Path("binaries/accumulating_rhythm/tests/fixtures/in_1.krn")
    for idx in range(2):
        (input_dir / f"piece_{idx}.krn").write_text(
            fixture_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    summary = run_dataset_generation(
        input_dirs=(input_dir,),
        output_dir=output_dir,
        target_samples=1,
        num_workers=2,
        max_attempts=6,
    )

    ds = load_from_disk(str(output_dir))
    assert summary.accepted_samples == 1
    assert len(ds) == 1
