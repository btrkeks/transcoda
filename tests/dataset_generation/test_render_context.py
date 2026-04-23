from __future__ import annotations

import json
import random
from dataclasses import replace

from scripts.dataset_generation.augmentation.articulations import ACCENT
from scripts.dataset_generation.dataset_generation.crash_repro import write_crash_artifact
from scripts.dataset_generation.dataset_generation.recipe import (
    ProductionRecipe,
    RenderOnlyAugmentationPolicy,
)
from scripts.dataset_generation.dataset_generation.render_transcription import (
    DynamicMarkPlan,
    HairpinPlan,
    NoteSuffixPlan,
    PedalSpanPlan,
    RenderAugmentationPlan,
    materialize_render_transcription,
    sample_render_augmentation_plan,
)
from scripts.dataset_generation.dataset_generation.renderer import (
    prepare_render_attempt,
    prepare_sample_render_context,
)
from scripts.dataset_generation.dataset_generation.run_context import build_run_context
from scripts.dataset_generation.dataset_generation.truncation import (
    TruncationSearchResult,
    build_prefix_candidates,
    build_prefix_truncation_space,
)
from scripts.dataset_generation.dataset_generation.worker import evaluate_sample_plan
from tests.dataset_generation.factories import make_render_result, make_sample_plan


def _rich_label_transcription() -> str:
    return "\n".join(
        [
            "*clefG2",
            "*M4/4",
            "=1",
            "4c",
            "=2",
            "4d",
            "=3",
            "4e",
            "=4",
            "4f",
        ]
    )


def _rich_recipe() -> ProductionRecipe:
    return ProductionRecipe(
        render_only_aug=RenderOnlyAugmentationPolicy(
            include_title_probability=0.0,
            include_author_probability=0.0,
            render_pedals_probability=1.0,
            render_pedals_measures_probability=1.0,
            render_instrument_piano_probability=1.0,
            render_sforzando_probability=1.0,
            render_sforzando_per_note_probability=0.5,
            render_accent_probability=1.0,
            render_accent_per_note_probability=0.5,
            render_tempo_probability=1.0,
            render_tempo_include_mm_probability=1.0,
            render_hairpins_probability=1.0,
            render_hairpins_max_spans=1,
            render_dynamic_marks_probability=1.0,
            render_dynamic_marks_min_count=1,
            render_dynamic_marks_max_count=1,
            max_render_attempts=2,
        )
    )


def test_sample_render_augmentation_plan_is_deterministic_for_same_seed():
    transcription = _rich_label_transcription()
    recipe = _rich_recipe()

    first = sample_render_augmentation_plan(
        transcription,
        recipe,
        rng=random.Random(123),
    )
    second = sample_render_augmentation_plan(
        transcription,
        recipe,
        rng=random.Random(123),
    )

    assert first == second


def test_materialize_render_transcription_drops_spans_beyond_prefix():
    transcription = _rich_label_transcription()
    recipe = _rich_recipe()
    candidate = build_prefix_truncation_space(transcription).candidate_for_chunk_count(2)
    assert candidate is not None

    augmentation_plan = RenderAugmentationPlan(
        note_suffixes=(NoteSuffixPlan(origin_line_idx=3, col_idx=0, suffix=ACCENT),),
        pedal_spans=(PedalSpanPlan(start_barline_line_idx=2, end_barline_line_idx=6),),
        hairpins=(HairpinPlan(origin_line_indices=(3, 5, 7), is_crescendo=True),),
        dynamic_marks=(DynamicMarkPlan(origin_line_idx=5, token="ff"),),
    )

    rendered_prefix = materialize_render_transcription(
        candidate.transcription,
        recipe,
        augmentation_plan=augmentation_plan,
        source_line_indices=candidate.origin_line_indices,
    )

    assert "^" in rendered_prefix
    assert "*ped" not in rendered_prefix
    assert "*Xped" not in rendered_prefix
    assert "<" not in rendered_prefix
    assert "(" not in rendered_prefix
    assert "[" not in rendered_prefix


def test_prepare_render_attempt_reuses_context_across_seeds():
    transcription = _rich_label_transcription()
    recipe = _rich_recipe()
    context = prepare_sample_render_context(transcription, recipe, seed=123)

    default_one = prepare_render_attempt(
        "**kern\n=1\n4c\n*-\n",
        recipe,
        seed=11,
        context=context,
    )
    default_two = prepare_render_attempt(
        "**kern\n=1\n4c\n*-\n",
        recipe,
        seed=999,
        context=context,
    )
    rescue = prepare_render_attempt(
        "**kern\n=1\n4c\n*-\n",
        recipe,
        seed=77,
        mode="layout_rescue",
        context=context,
    )

    assert default_one["metadata_prefix"] == default_two["metadata_prefix"]
    assert default_one["render_options"] == default_two["render_options"]
    assert rescue["metadata_prefix"] == default_one["metadata_prefix"]
    assert rescue["render_options"] != default_one["render_options"]


def test_evaluate_sample_plan_reuses_same_context_for_truncation_probes(monkeypatch):
    transcription = _rich_label_transcription()
    plan = make_sample_plan(
        seed=123,
        label_transcription=transcription,
        source_measure_count=4,
        source_non_empty_line_count=len(transcription.splitlines()),
    )
    recipe = _rich_recipe()
    candidate = build_prefix_candidates(transcription, recipe)[0]
    observed_context_ids: list[int] = []
    observed_options: list[dict[str, object]] = []

    def fake_find_best_truncation_candidate(
        kern_text,
        *,
        max_trials,
        probe_candidate,
        local_refinement_radius=2,
    ):
        del kern_text, max_trials, local_refinement_radius
        probe = probe_candidate(candidate)
        return TruncationSearchResult(
            selected_candidate=candidate if probe.accepted else None,
            selected_probe=probe if probe.accepted else None,
            probes=(probe,),
            exhausted_budget=False,
        )

    def fake_render(render_text, recipe, *, seed, renderer, context=None):
        assert context is not None
        observed_context_ids.append(id(context))
        observed_options.append(dict(context.base_render_options))
        if seed == plan.seed:
            return make_render_result(system_count=8, page_count=2, rejection_reason="multi_page")
        return make_render_result(system_count=4, page_count=1)

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.worker.find_best_truncation_candidate",
        fake_find_best_truncation_candidate,
    )

    outcome = evaluate_sample_plan(
        plan,
        recipe=recipe,
        renderer=object(),
        render_fn=fake_render,
        augment_fn=lambda plan, render_result, recipe: render_result.image,
    )

    assert outcome.sample.truncation_applied is True
    assert len(set(observed_context_ids)) == 1
    assert all(options == observed_options[0] for options in observed_options)


def test_write_crash_artifact_reuses_frozen_render_context(tmp_path):
    transcription = _rich_label_transcription()
    plan = make_sample_plan(
        seed=123,
        label_transcription=transcription,
        source_measure_count=4,
        source_non_empty_line_count=len(transcription.splitlines()),
    )
    recipe = _rich_recipe()
    run_context = build_run_context(output_dir=tmp_path / "dataset")

    artifact_path, repro_count = write_crash_artifact(
        run_context=run_context,
        recipe=recipe,
        plan=plan,
        sample_idx=1,
        event_type="timeout",
        retry_count=1,
        will_retry=False,
        queue_wait_ms=0.0,
        dropped_pending_tasks=0,
        target_bucket=None,
        planned_line_count=None,
        candidate_in_target_range=None,
        exception_payload=None,
        last_worker_stage_event=None,
    )

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert repro_count == len(artifact["repro_entries"])

    full_text = (
        run_context.crash_samples_dir / plan.sample_id / "00000001_full.krn"
    ).read_text(encoding="utf-8")
    candidate_entries = [entry for entry in artifact["repro_entries"] if entry["stage"] == "truncation_candidate"]
    assert candidate_entries
    first_candidate_text = run_context.crash_samples_dir.joinpath(
        plan.sample_id,
        "00000001_truncation_01.krn",
    ).read_text(encoding="utf-8")

    full_omd = next(line for line in full_text.splitlines() if line.startswith("!!!OMD:"))
    assert full_omd in first_candidate_text
    assert '*I"Piano' in full_text
    assert '*I"Piano' in first_candidate_text
