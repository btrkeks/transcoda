import json
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from scripts.dataset_generation.dataset_generation.calibrate_system_balance import (
    run_calibration,
)
from scripts.dataset_generation.dataset_generation.recipe import (
    CompositionPolicy,
    ProductionRecipe,
    RenderOnlyAugmentationPolicy,
)
from scripts.dataset_generation.dataset_generation.system_balance import (
    build_calibration_artifact,
    choose_balanced_plan,
    choose_target_bucket,
    load_system_balance_spec,
    load_tokenizer,
)
from scripts.dataset_generation.dataset_generation.types import SamplePlan, SourceSegment
from scripts.dataset_generation.dataset_generation.types import RenderResult, SvgLayoutDiagnostics


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
                "4d": 4,
                "*-": 5,
                "alpha": 6,
                "beta": 7,
                "gamma": 8,
                "delta": 9,
                "epsilon": 10,
                "zeta": 11,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    return tokenizer_dir


def test_build_calibration_artifact_and_load_spec(tmp_path: Path):
    tokenizer_dir = _make_tokenizer_dir(tmp_path)
    recipe = ProductionRecipe()
    records = [
        {
            "sample_idx": 0,
            "sample_id": "sample_00000000",
            "token_length": 10,
            "full_render_system_count": 5,
            "accepted_render_system_count": 1,
            "accepted": True,
            "failure_reason": None,
            "truncation_applied": True,
            "preferred_5_6_status": "preferred_5_6_truncated",
            "segment_count": 1,
            "source_root_labels": ["a"],
        },
        {
            "sample_idx": 1,
            "sample_id": "sample_00000001",
            "token_length": 12,
            "full_render_system_count": 1,
            "accepted_render_system_count": 1,
            "accepted": True,
            "failure_reason": None,
            "truncation_applied": False,
            "preferred_5_6_status": None,
            "segment_count": 1,
            "source_root_labels": ["a"],
        },
        {
            "sample_idx": 2,
            "sample_id": "sample_00000002",
            "token_length": 20,
            "full_render_system_count": 2,
            "accepted_render_system_count": 2,
            "accepted": True,
            "failure_reason": None,
            "truncation_applied": False,
            "preferred_5_6_status": None,
            "segment_count": 2,
            "source_root_labels": ["b"],
        },
    ]

    artifact = build_calibration_artifact(
        records=records,
        tokenizer_dir=tokenizer_dir,
        recipe=recipe,
        input_dirs=(tmp_path / "in",),
        sample_budget=3,
    )

    output_path = tmp_path / "balance.json"
    output_path.write_text(json.dumps(artifact), encoding="utf-8")
    spec = load_system_balance_spec(output_path)

    assert artifact["tokenizer"]["path"] == str(tokenizer_dir)
    assert artifact["calibration_target_metric"] == "full_render_system_count_preferred"
    assert artifact["systems"]["1"]["accepted_count"] == 2
    assert artifact["systems"]["1"]["calibration_count"] == 1
    assert artifact["systems"]["1"]["full_render_count"] == 1
    assert artifact["systems"]["5"]["calibration_count"] == 1
    assert artifact["systems"]["5"]["accepted_count"] == 0
    assert artifact["systems"]["5"]["full_render_count"] == 1
    assert artifact["coverage"]["full_render_system_histogram"]["5"] == 1
    assert artifact["coverage"]["truncated_accepted_output_system_histogram"]["1"] == 1
    assert artifact["recommended_token_length_ranges"]["5"]["center"] == 10.0
    assert "1" in artifact["recommended_token_length_ranges"]
    assert spec.tokenizer_path == tokenizer_dir.resolve()
    assert 1 in spec.token_length_ranges


def test_choose_target_bucket_prefers_most_underfilled_bucket():
    assert choose_target_bucket({1: 3, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1}) == 2


def test_choose_balanced_plan_prefers_candidate_closest_to_target_bucket(
    tmp_path: Path, monkeypatch
):
    tokenizer_dir = _make_tokenizer_dir(tmp_path)
    tokenizer_dir_resolved = tokenizer_dir.resolve()
    _, tokenizer = load_tokenizer(tokenizer_dir_resolved)
    spec_path = tmp_path / "balance.json"
    spec_path.write_text(
        json.dumps(
            {
                "mode": "length_proxy",
                "tokenizer": {"path": str(tokenizer_dir_resolved), "fingerprint": "x"},
                "recipe_fingerprint": "recipe",
                "candidate_plan_count": 3,
                    "recommended_token_length_ranges": {
                        "1": {"min": 2, "max": 3, "center": 2.5},
                        "2": {"min": 4, "max": 5, "center": 4.5},
                        "3": {"min": 6, "max": 7, "center": 6.5},
                        "4": {"min": 8, "max": 9, "center": 8.5},
                        "5": {"min": 6, "max": 6, "center": 6.0},
                        "6": {"min": 12, "max": 13, "center": 12.5},
                    },
                }
        ),
        encoding="utf-8",
    )
    spec = load_system_balance_spec(spec_path)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "piece.krn").write_text("*clefG2\n=1\n4c\n*-\n", encoding="utf-8")
    from scripts.dataset_generation.dataset_generation.source_index import build_source_index

    source_index = build_source_index(input_dir)
    candidate_texts = [
        "alpha beta\n",
        "alpha beta gamma delta epsilon zeta\n",
        "alpha beta gamma\n",
    ]
    calls = {"value": 0}

    def fake_plan_sample(source_index, recipe, *, sample_idx, base_seed, excluded_paths=None):
        del source_index, recipe, sample_idx, base_seed, excluded_paths
        idx = calls["value"]
        calls["value"] += 1
        return SamplePlan(
            sample_id="sample_00000000",
            seed=idx,
            segments=(
                SourceSegment(source_id="input/piece", path=input_dir / "piece.krn", order=0),
            ),
            label_transcription=candidate_texts[idx],
            source_measure_count=1,
            source_non_empty_line_count=1,
            segment_count=1,
        )

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.system_balance.plan_sample",
        fake_plan_sample,
    )

    selected = choose_balanced_plan(
        source_index=source_index,
        recipe=ProductionRecipe(),
        sample_idx=0,
        base_seed=0,
        excluded_paths=None,
        tokenizer=tokenizer,
        spec=spec,
        accepted_system_histogram={1: 1, 2: 1, 3: 1, 4: 1, 5: 0, 6: 1},
        candidate_plan_count=3,
    )

    assert selected.target_bucket == 5
    assert selected.in_target_range is True
    assert selected.plan.label_transcription == candidate_texts[1]


def test_choose_balanced_plan_skips_invalid_candidates(tmp_path: Path, monkeypatch):
    tokenizer_dir = _make_tokenizer_dir(tmp_path)
    tokenizer_dir_resolved = tokenizer_dir.resolve()
    _, tokenizer = load_tokenizer(tokenizer_dir_resolved)
    spec_path = tmp_path / "balance.json"
    spec_path.write_text(
        json.dumps(
            {
                "mode": "length_proxy",
                "tokenizer": {"path": str(tokenizer_dir_resolved), "fingerprint": "x"},
                "recipe_fingerprint": "recipe",
                "candidate_plan_count": 3,
                "recommended_token_length_ranges": {
                    str(bucket): {"min": bucket, "max": bucket + 1, "center": float(bucket) + 0.5}
                    for bucket in range(1, 7)
                },
            }
        ),
        encoding="utf-8",
    )
    spec = load_system_balance_spec(spec_path)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "piece.krn").write_text("*clefG2\n=1\n4c\n*-\n", encoding="utf-8")
    from scripts.dataset_generation.dataset_generation.source_index import build_source_index

    source_index = build_source_index(input_dir)
    calls = {"value": 0}

    def fake_plan_sample(source_index, recipe, *, sample_idx, base_seed, excluded_paths=None):
        del source_index, recipe, sample_idx, base_seed, excluded_paths
        idx = calls["value"]
        calls["value"] += 1
        if idx == 0:
            raise ValueError("boundary mismatch")
        return SamplePlan(
            sample_id="sample_00000000",
            seed=idx,
            segments=(SourceSegment(source_id="input/piece", path=input_dir / "piece.krn", order=0),),
            label_transcription="alpha beta gamma\n",
            source_measure_count=1,
            source_non_empty_line_count=1,
            segment_count=1,
        )

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.system_balance.plan_sample",
        fake_plan_sample,
    )

    selected = choose_balanced_plan(
        source_index=source_index,
        recipe=ProductionRecipe(),
        sample_idx=0,
        base_seed=0,
        excluded_paths=None,
        tokenizer=tokenizer,
        spec=spec,
        accepted_system_histogram={bucket: 0 for bucket in range(1, 7)},
        candidate_plan_count=3,
    )

    assert selected.plan.label_transcription == "alpha beta gamma\n"
    assert calls["value"] == 3


def test_choose_balanced_plan_raises_after_all_candidates_fail(tmp_path: Path, monkeypatch):
    tokenizer_dir = _make_tokenizer_dir(tmp_path)
    tokenizer_dir_resolved = tokenizer_dir.resolve()
    _, tokenizer = load_tokenizer(tokenizer_dir_resolved)
    spec_path = tmp_path / "balance.json"
    spec_path.write_text(
        json.dumps(
            {
                "mode": "length_proxy",
                "tokenizer": {"path": str(tokenizer_dir_resolved), "fingerprint": "x"},
                "recipe_fingerprint": "recipe",
                "candidate_plan_count": 2,
                "recommended_token_length_ranges": {
                    str(bucket): {"min": bucket, "max": bucket + 1, "center": float(bucket) + 0.5}
                    for bucket in range(1, 7)
                },
            }
        ),
        encoding="utf-8",
    )
    spec = load_system_balance_spec(spec_path)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "piece.krn").write_text("*clefG2\n=1\n4c\n*-\n", encoding="utf-8")
    from scripts.dataset_generation.dataset_generation.source_index import build_source_index

    source_index = build_source_index(input_dir)

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.system_balance.plan_sample",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("boundary mismatch")),
    )

    with pytest.raises(ValueError, match="All candidate plans were invalid or exhausted"):
        choose_balanced_plan(
            source_index=source_index,
            recipe=ProductionRecipe(),
            sample_idx=0,
            base_seed=0,
            excluded_paths=None,
            tokenizer=tokenizer,
            spec=spec,
            accepted_system_histogram={bucket: 0 for bucket in range(1, 7)},
            candidate_plan_count=2,
        )


def test_run_calibration_writes_expected_json(tmp_path: Path):
    tokenizer_dir = _make_tokenizer_dir(tmp_path)
    input_dir = tmp_path / "input"
    output_json = tmp_path / "calibration.json"
    input_dir.mkdir()
    for idx in range(2):
        (input_dir / f"{idx}.krn").write_text("*clefG2\n=1\n4c\n*-\n", encoding="utf-8")

    def fake_render(render_text, recipe, *, seed, renderer):
        del render_text, recipe, seed, renderer
        image = np.full((1485, 1050, 3), 255, dtype=np.uint8)
        image[10:70, 10:600] = 0
        return RenderResult(
            image=image,
            render_layers=None,
            svg_diagnostics=SvgLayoutDiagnostics(system_count=2, page_count=1),
            bottom_whitespace_ratio=0.10,
            vertical_fill_ratio=0.72,
        )

    recipe = ProductionRecipe(
        composition=CompositionPolicy(
            segment_count_weights=((1, 1.0),),
            min_total_measures=1,
            max_total_measures=8,
            max_selection_attempts=1,
        ),
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

    summary = run_calibration(
        input_dirs=(input_dir,),
        tokenizer_path=tokenizer_dir,
        sample_budget=3,
        output_json=output_json,
        recipe=recipe,
        renderer=object(),
        render_fn=fake_render,
        augment_fn=lambda plan, render_result, recipe: render_result.image,
        quiet=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert summary["attempted_samples"] == 3
    assert payload["attempted_samples"] == 3
    assert payload["accepted_samples"] == 3
    assert payload["tokenizer"]["path"] == str(tokenizer_dir.resolve())
    assert payload["records"][0]["segment_count"] == 1
    assert payload["records"][0]["full_render_system_count"] == 2
    assert payload["records"][0]["accepted_render_system_count"] == 2
