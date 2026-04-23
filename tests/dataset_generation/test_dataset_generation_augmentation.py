import numpy as np

from scripts.dataset_generation.dataset_generation.augmentation import augment_accepted_render
from scripts.dataset_generation.dataset_generation.image_augmentation.offline_augment import (
    OfflineAugmentTrace,
)
from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.types_events import AugmentationTraceEvent
from tests.dataset_generation.factories import (
    make_bounds_gate_trace,
    make_geometry_trace,
    make_margin_trace,
    make_outer_gate_trace,
    make_quality_gate_trace,
    make_render_result,
    make_sample_plan,
)


def _make_plan_and_render(*, system_count=4, fill=0.62, bottom_ws=0.15, segment_count=1):
    base = np.full((1485, 1050, 3), 255, dtype=np.uint8)
    base[20:80, 20:500] = 0
    plan = make_sample_plan(
        seed=7,
        label_transcription="=1\n4c\n*-\n",
        source_measure_count=8,
        source_non_empty_line_count=16,
        segment_count=segment_count,
    )
    render_result = make_render_result(
        image=base,
        system_count=system_count,
        bottom_whitespace_ratio=bottom_ws,
        vertical_fill_ratio=fill,
        bottom_whitespace_px=None,
        top_whitespace_px=None,
        content_height_px=None,
    )
    return plan, render_result, base


def _make_offline_trace(**overrides):
    defaults = dict(
        branch="geometric",
        initial_geometry=make_geometry_trace(
            angle_deg=0.8, scale=1.01, tx_px=1.0, ty_px=2.0, x_scale=0.96, y_scale=1.0
        ),
        retry_geometry=None,
        selected_geometry=make_geometry_trace(
            angle_deg=0.8, scale=1.01, tx_px=1.0, ty_px=2.0, x_scale=0.96, y_scale=1.0
        ),
        final_geometry_applied=True,
        initial_oob_gate=make_bounds_gate_trace(
            dx_frac=0.01, dy_frac=0.01, area_retention=0.98,
            margins_px=None,  # defaults to make_margin_trace()
        ),
        retry_oob_gate=None,
        outer_gate=make_outer_gate_trace(),
        augraphy_outcome="applied",
        augraphy_normalize_accepted=True,
        augraphy_fallback_attempted=False,
        augraphy_fallback_outcome=None,
        augraphy_fallback_normalize_accepted=None,
        offline_geom_ms=1.0,
        offline_gates_ms=0.5,
        offline_augraphy_ms=2.0,
        offline_texture_ms=0.3,
    )
    defaults.update(overrides)
    return OfflineAugmentTrace(**defaults)


def test_augment_accepted_render_falls_back_to_base_image_when_candidate_is_invalid(monkeypatch):
    plan, render_result, base = _make_plan_and_render()
    invalid = np.zeros_like(base)

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.augmentation.offline_augment",
        lambda *args, **kwargs: (
            invalid,
            invalid,
            _make_offline_trace(
                outer_gate=make_outer_gate_trace(
                    passed=False,
                    failure_reason="quality:min_margin",
                    quality_gate=make_quality_gate_trace(
                        passed=False,
                        failure_reason="min_margin",
                        mean_luma=200.0,
                        content_ratio=0.1,
                        margins_px=make_margin_trace(left_px=0),
                    ),
                ),
            ),
        ),
    )

    augmented = augment_accepted_render(plan, render_result, ProductionRecipe())
    trace = augmented.trace

    assert np.array_equal(augmented.final_image, base)
    assert isinstance(trace, AugmentationTraceEvent)
    assert not trace.outer_gate.passed
    assert trace.final_outcome == "clean_gate_rejected"


def test_augment_accepted_render_returns_augmented_image_when_gates_pass(monkeypatch):
    plan, render_result, base = _make_plan_and_render()
    augmented_img = base.copy()
    augmented_img[30:70, 30:490] = 40

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.augmentation.offline_augment",
        lambda *args, **kwargs: (augmented_img, augmented_img, _make_offline_trace()),
    )

    result = augment_accepted_render(plan, render_result, ProductionRecipe())
    trace = result.trace

    assert np.array_equal(result.final_image, augmented_img)
    assert trace.outer_gate.passed
    assert trace.final_outcome == "fully_augmented"


def test_augment_accepted_render_records_band_correctly(monkeypatch):
    plan, render_result, base = _make_plan_and_render(
        system_count=2, fill=0.40, bottom_ws=0.20, segment_count=1
    )

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.augmentation.offline_augment",
        lambda *args, **kwargs: (base, base, _make_offline_trace()),
    )

    trace = augment_accepted_render(plan, render_result, ProductionRecipe()).trace
    assert trace.band == "roomy"

    plan_tight, render_tight, base_tight = _make_plan_and_render(system_count=8)
    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.augmentation.offline_augment",
        lambda *args, **kwargs: (base_tight, base_tight, _make_offline_trace()),
    )
    trace_tight = augment_accepted_render(plan_tight, render_tight, ProductionRecipe()).trace
    assert trace_tight.band == "tight"


def test_augment_accepted_render_trace_event_fields(monkeypatch):
    plan, render_result, base = _make_plan_and_render()

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.augmentation.offline_augment",
        lambda *args, **kwargs: (base, base, _make_offline_trace(branch="foreground")),
    )

    trace = augment_accepted_render(plan, render_result, ProductionRecipe()).trace

    assert trace.event == "augmentation_trace"
    assert trace.sample_id == "sample_00000000"
    assert trace.sample_idx == 0
    assert trace.seed == 7
    assert trace.branch == "foreground"
    assert trace.final_geometry_applied
    assert trace.initial_geometry.angle_deg == 0.8
    assert trace.offline_geom_ms == 1.0


def test_augment_accepted_render_augraphy_on_base_outcome(monkeypatch):
    plan, render_result, base = _make_plan_and_render()

    trace_data = _make_offline_trace(
        augraphy_normalize_accepted=False,
        augraphy_fallback_attempted=True,
        augraphy_fallback_outcome="applied",
        augraphy_fallback_normalize_accepted=True,
    )

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.augmentation.offline_augment",
        lambda *args, **kwargs: (base, base, trace_data),
    )

    trace = augment_accepted_render(plan, render_result, ProductionRecipe()).trace

    assert trace.outer_gate.passed
    assert trace.final_outcome == "augraphy_on_base"
