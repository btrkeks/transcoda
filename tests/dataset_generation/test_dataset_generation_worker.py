from dataclasses import replace
from pathlib import Path

import numpy as np

from scripts.dataset_generation.dataset_generation.image_generation.types import RenderedPage
from scripts.dataset_generation.dataset_generation.recipe import (
    ProductionRecipe,
    RenderOnlyAugmentationPolicy,
)
from scripts.dataset_generation.dataset_generation.renderer import (
    prepare_render_attempt,
    render_sample,
    render_sample_with_layout_rescue,
)
from scripts.dataset_generation.dataset_generation.truncation import PrefixTruncationCandidate
from scripts.dataset_generation.dataset_generation.types import (
    RenderResult,
    SamplePlan,
    SourceSegment,
    SvgLayoutDiagnostics,
    VerovioDiagnostic,
    WorkerFailure,
    WorkerSuccess,
)
from scripts.dataset_generation.dataset_generation.worker import (
    compute_initial_kern_spine_count,
    evaluate_sample_plan,
)


def test_compute_initial_kern_spine_count_with_header():
    assert compute_initial_kern_spine_count("**kern\t**kern\n=1\t=1\n4c\t4e\n*-\t*-\n") == 2


def test_compute_initial_kern_spine_count_without_header():
    assert compute_initial_kern_spine_count("=1\t=1\n4c\t4e\n*-\t*-\n") == 2


def test_compute_initial_kern_spine_count_skips_leading_blank_lines_and_comments():
    transcription = "\n\n!! generated sample\n!! another comment\n**kern\t**kern\n=1\t=1\n"
    assert compute_initial_kern_spine_count(transcription) == 2


def test_compute_initial_kern_spine_count_uses_truncated_candidate_shape():
    transcription = "**kern\n=1\n4c\n*-\n"
    assert compute_initial_kern_spine_count(transcription) == 1


def _make_plan() -> SamplePlan:
    return SamplePlan(
        sample_id="sample_00000000",
        seed=123,
        segments=(SourceSegment(source_id="input/piece", path="/tmp/piece.krn", order=0),),
        label_transcription="**kern\n=1\n4c\n*-\n",
        source_measure_count=1,
        source_non_empty_line_count=4,
        source_max_initial_spine_count=1,
        segment_count=1,
    )


def _make_recipe() -> ProductionRecipe:
    return ProductionRecipe(
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
            max_render_attempts=2,
        ),
    )


def test_render_options_use_requested_staff_and_system_spacing_ranges():
    recipe = _make_recipe()

    for seed in range(100):
        attempt = prepare_render_attempt(
            "**kern\n=1\n4c\n*-\n",
            recipe,
            seed=seed,
        )
        options = attempt["render_options"]

        assert 4 <= options["spacingStaff"] <= 20
        assert 3 <= options["spacingSystem"] <= 10


def _good_render(*, system_count: int) -> RenderResult:
    image = np.full((1485, 1050, 3), 255, dtype=np.uint8)
    image[10:70, 10:600] = 0
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


def _bad_render(*, system_count: int, reason: str = "left_clearance") -> RenderResult:
    return RenderResult(
        image=None,
        render_layers=None,
        svg_diagnostics=SvgLayoutDiagnostics(system_count=system_count, page_count=1),
        bottom_whitespace_ratio=0.02,
        vertical_fill_ratio=0.90,
        bottom_whitespace_px=30,
        top_whitespace_px=14,
        content_height_px=1337,
        rejection_reason=reason,
    )


def _with_verovio_diagnostic(render_result: RenderResult) -> RenderResult:
    return replace(
        render_result,
        verovio_diagnostics=(
            VerovioDiagnostic(
                diagnostic_kind="inconsistent_rhythm_analysis",
                raw_message="Error: Inconsistent rhythm analysis occurring near line 12",
                render_attempt_idx=1,
                near_line=12,
                expected_duration_from_start="64",
                found_duration_from_start="62",
                line_text="4G\t.\t.\t4c 4e",
            ),
        ),
    )


def test_full_5_6_render_that_passes_quality_is_accepted_without_truncation():
    plan = _make_plan()
    recipe = _make_recipe()

    outcome = evaluate_sample_plan(
        plan,
        recipe=recipe,
        renderer=object(),
        render_fn=lambda render_text, recipe, *, seed, renderer: _good_render(system_count=5),
        augment_fn=lambda plan, render_result, recipe: render_result.image,
    )

    assert isinstance(outcome, WorkerSuccess)
    assert outcome.sample.truncation_applied is False
    assert outcome.sample.initial_kern_spine_count == 1
    assert outcome.preferred_5_6_status == "preferred_5_6_accepted_full"
    assert outcome.full_render_system_count == 5
    assert outcome.accepted_render_system_count == 5
    assert outcome.sample.bottom_whitespace_px == 149
    assert outcome.sample.top_whitespace_px == 33
    assert outcome.sample.content_height_px == 1069


def test_full_7_render_that_passes_quality_is_accepted_without_truncation():
    plan = _make_plan()
    recipe = _make_recipe()

    outcome = evaluate_sample_plan(
        plan,
        recipe=recipe,
        renderer=object(),
        render_fn=lambda render_text, recipe, *, seed, renderer: _good_render(system_count=7),
        augment_fn=lambda plan, render_result, recipe: render_result.image,
    )

    assert isinstance(outcome, WorkerSuccess)
    assert outcome.sample.truncation_applied is False
    assert outcome.preferred_5_6_status == "preferred_5_6_accepted_full"
    assert outcome.full_render_system_count == 7
    assert outcome.accepted_render_system_count == 7


def test_full_render_verovio_diagnostics_are_stage_attributed():
    plan = _make_plan()
    recipe = _make_recipe()

    outcome = evaluate_sample_plan(
        plan,
        recipe=recipe,
        renderer=object(),
        render_fn=lambda render_text, recipe, *, seed, renderer: _with_verovio_diagnostic(
            _good_render(system_count=4)
        ),
        augment_fn=lambda plan, render_result, recipe: render_result.image,
    )

    assert isinstance(outcome, WorkerSuccess)
    assert len(outcome.verovio_diagnostics) == 1
    event = outcome.verovio_diagnostics[0]
    assert event.event == "verovio_diagnostic"
    assert event.sample_id == plan.sample_id
    assert event.sample_idx == 0
    assert event.source_paths == (str(Path("/tmp/piece.krn").resolve()),)
    assert event.stage == "full"
    assert event.seed == plan.seed
    assert event.render_attempt_idx == 1
    assert event.diagnostic_kind == "inconsistent_rhythm_analysis"
    assert event.near_line == 12


def test_failed_5_6_render_attempts_rescue_before_truncation():
    plan = _make_plan()
    recipe = _make_recipe()
    calls = []

    def fake_render(render_text, recipe, *, seed, renderer):
        calls.append(seed)
        return _bad_render(system_count=6)

    outcome = evaluate_sample_plan(
        plan,
        recipe=recipe,
        renderer=object(),
        render_fn=fake_render,
        rescue_render_fn=lambda render_text, recipe, *, seed, renderer: _bad_render(
            system_count=6,
            reason="right_clearance",
        ),
        augment_fn=lambda plan, render_result, recipe: render_result.image,
    )

    assert isinstance(outcome, WorkerFailure)
    assert outcome.preferred_5_6_rescue_attempted is True
    assert outcome.preferred_5_6_status == "preferred_5_6_failed"


def test_rescued_5_6_render_is_accepted_as_full_not_truncated():
    plan = _make_plan()
    recipe = _make_recipe()

    outcome = evaluate_sample_plan(
        plan,
        recipe=recipe,
        renderer=object(),
        render_fn=lambda render_text, recipe, *, seed, renderer: _bad_render(system_count=5),
        rescue_render_fn=lambda render_text, recipe, *, seed, renderer: _good_render(system_count=5),
        augment_fn=lambda plan, render_result, recipe: render_result.image,
    )

    assert isinstance(outcome, WorkerSuccess)
    assert outcome.sample.truncation_applied is False
    assert outcome.preferred_5_6_rescue_attempted is True
    assert outcome.preferred_5_6_rescue_succeeded is True
    assert outcome.preferred_5_6_status == "preferred_5_6_rescued"
    assert outcome.accepted_render_system_count == 5


def test_rescue_render_verovio_diagnostics_are_stage_attributed():
    plan = _make_plan()
    recipe = _make_recipe()

    outcome = evaluate_sample_plan(
        plan,
        recipe=recipe,
        renderer=object(),
        render_fn=lambda render_text, recipe, *, seed, renderer: _with_verovio_diagnostic(
            _bad_render(system_count=5)
        ),
        rescue_render_fn=lambda render_text, recipe, *, seed, renderer: _with_verovio_diagnostic(
            _good_render(system_count=5)
        ),
        augment_fn=lambda plan, render_result, recipe: render_result.image,
    )

    assert isinstance(outcome, WorkerSuccess)
    assert [event.stage for event in outcome.verovio_diagnostics] == [
        "full",
        "preferred_5_6_rescue",
    ]
    assert outcome.verovio_diagnostics[1].seed == ((plan.seed & 0xFFFFFFFF) ^ 0x5F3759DF)


def test_accepted_sample_keeps_clean_render_layout_metrics_before_augmentation():
    plan = _make_plan()
    recipe = _make_recipe()
    augmented = np.full((1485, 1050, 3), 200, dtype=np.uint8)

    outcome = evaluate_sample_plan(
        plan,
        recipe=recipe,
        renderer=object(),
        render_fn=lambda render_text, recipe, *, seed, renderer: _good_render(system_count=4),
        augment_fn=lambda plan, render_result, recipe: augmented,
    )

    assert isinstance(outcome, WorkerSuccess)
    assert outcome.sample.bottom_whitespace_px == 149
    assert outcome.sample.top_whitespace_px == 33
    assert outcome.sample.content_height_px == 1069


def test_failed_5_6_render_can_still_truncate_successfully(monkeypatch):
    plan = _make_plan()
    recipe = _make_recipe()
    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.worker.build_prefix_candidates",
        lambda transcription, recipe: [
            PrefixTruncationCandidate(
                transcription="**kern\n=1\n4c\n*-\n",
                chunk_count=1,
                total_chunks=2,
                ratio=0.5,
            )
        ],
    )

    def fake_render(render_text, recipe, *, seed, renderer):
        if seed == plan.seed + 17:
            return _good_render(system_count=4)
        return _bad_render(system_count=6)

    outcome = evaluate_sample_plan(
        plan,
        recipe=recipe,
        renderer=object(),
        render_fn=fake_render,
        rescue_render_fn=lambda render_text, recipe, *, seed, renderer: _bad_render(system_count=6),
        augment_fn=lambda plan, render_result, recipe: render_result.image,
    )

    assert isinstance(outcome, WorkerSuccess)
    assert outcome.sample.truncation_applied is True
    assert outcome.sample.initial_kern_spine_count == 1
    assert outcome.preferred_5_6_status == "preferred_5_6_truncated"
    assert outcome.accepted_render_system_count == 4


def test_truncation_candidate_verovio_diagnostics_are_stage_attributed(monkeypatch):
    plan = _make_plan()
    recipe = _make_recipe()
    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.worker.build_prefix_candidates",
        lambda transcription, recipe: [
            PrefixTruncationCandidate(
                transcription="**kern\n=1\n4c\n*-\n",
                chunk_count=3,
                total_chunks=5,
                ratio=0.6,
            )
        ],
    )

    def fake_render(render_text, recipe, *, seed, renderer):
        if seed == plan.seed + 3 * 17:
            return _with_verovio_diagnostic(_good_render(system_count=4))
        return _with_verovio_diagnostic(_bad_render(system_count=8))

    outcome = evaluate_sample_plan(
        plan,
        recipe=recipe,
        renderer=object(),
        render_fn=fake_render,
        augment_fn=lambda plan, render_result, recipe: render_result.image,
    )

    assert isinstance(outcome, WorkerSuccess)
    assert [event.stage for event in outcome.verovio_diagnostics] == [
        "full",
        "truncation_candidate",
    ]
    candidate_event = outcome.verovio_diagnostics[1]
    assert candidate_event.seed == plan.seed + 3 * 17
    assert candidate_event.truncation_chunk_count == 3
    assert candidate_event.truncation_total_chunks == 5
    assert candidate_event.truncation_ratio == 0.6


def test_required_over_7_systems_still_go_straight_to_truncation():
    plan = _make_plan()
    recipe = _make_recipe()
    rescue_calls = {"count": 0}

    def rescue_render(*args, **kwargs):
        rescue_calls["count"] += 1
        return _good_render(system_count=6)

    def fake_render(render_text, recipe, *, seed, renderer):
        if seed == plan.seed + 17:
            return _good_render(system_count=4)
        return _good_render(system_count=8)

    outcome = evaluate_sample_plan(
        plan,
        recipe=recipe,
        renderer=object(),
        render_fn=fake_render,
        rescue_render_fn=rescue_render,
        augment_fn=lambda plan, render_result, recipe: render_result.image,
    )

    assert isinstance(outcome, WorkerSuccess)
    assert outcome.sample.truncation_applied is True
    assert rescue_calls["count"] == 0


def test_layout_rescue_tightens_render_options_deterministically():
    class FakeRenderer:
        def __init__(self):
            self.calls = []

        def render_with_counts(self, renderable, options):
            self.calls.append(dict(options))
            image = np.full((1200, 900, 3), 255, dtype=np.uint8)
            image[40:200, 60:700] = 0
            rendered = RenderedPage(image=image, foreground=image, alpha=np.full((1200, 900), 255, dtype=np.uint8))
            return rendered, 5, 1

    recipe = _make_recipe()
    baseline_renderer = FakeRenderer()
    render_sample(
        "**kern\n=1\n4c\n*-\n",
        recipe,
        seed=11,
        renderer=baseline_renderer,
    )
    rescue_renderer = FakeRenderer()
    result = render_sample_with_layout_rescue(
        "**kern\n=1\n4c\n*-\n",
        recipe,
        seed=11,
        renderer=rescue_renderer,
    )

    assert result.rejection_reason is None
    assert len(baseline_renderer.calls) == 1
    assert len(rescue_renderer.calls) == 1
    baseline = baseline_renderer.calls[0]
    rescue = rescue_renderer.calls[0]
    assert rescue["pageWidth"] > baseline["pageWidth"]
    assert rescue["scale"] < baseline["scale"]
    assert rescue["spacingSystem"] <= baseline["spacingSystem"]
    assert rescue["measureMinWidth"] <= baseline["measureMinWidth"]
