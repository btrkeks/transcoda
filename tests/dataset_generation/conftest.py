"""Shared fixtures and test helpers for dataset-generation tests.

Plain helper functions live here alongside pytest fixtures. Tests import
them directly (``from tests.dataset_generation.conftest import ...``) when
they need the same setup across multiple test cases.
"""

from __future__ import annotations

import json
from concurrent.futures import Future
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

import scripts.dataset_generation.dataset_generation.executor as executor_module
from scripts.dataset_generation.dataset_generation.composer import compose_label_transcription
from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.system_balance import (
    DEFAULT_TOKENIZER_DIR,
    CandidatePlanScore,
    compute_recipe_fingerprint,
    compute_tokenizer_fingerprint,
    load_system_balance_spec,
)
from scripts.dataset_generation.dataset_generation.types_domain import SamplePlan, SourceSegment


class FakePebblePool:
    """In-process stand-in for ``pebble.ProcessPool`` used in executor tests."""

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


class DummyRenderer:
    pass


def make_simple_input_dir(tmp_path: Path, names: tuple[str, ...]) -> Path:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in names:
        (input_dir / f"{name}.krn").write_text("*clefG2\n=1\n4c\n*-\n", encoding="utf-8")
    return input_dir


def make_tokenizer_dir(tmp_path: Path) -> Path:
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


def make_balance_spec(tmp_path: Path, recipe: ProductionRecipe, tokenizer_dir: Path) -> Path:
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
                        str(bucket): {
                            "min": bucket,
                            "max": bucket + 1,
                            "center": float(bucket) + 0.5,
                        }
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


def install_bundled_spec(monkeypatch, spec_path: Path) -> None:
    monkeypatch.setattr(
        executor_module,
        "load_bundled_system_balance_spec",
        lambda: load_system_balance_spec(spec_path),
    )


def install_fake_pool(monkeypatch, *, outcomes_by_sample_idx, serial_wait: bool = False) -> None:
    monkeypatch.setattr(
        executor_module,
        "ProcessPool",
        lambda **kwargs: FakePebblePool(
            outcomes_by_sample_idx=outcomes_by_sample_idx,
            **kwargs,
        ),
    )
    monkeypatch.setattr(executor_module, "VerovioRenderer", DummyRenderer)
    monkeypatch.setattr(executor_module.mp, "get_context", lambda method: None)
    if serial_wait:
        def wait_one(futures, timeout=None, return_when=None):
            del timeout, return_when
            future_list = list(futures)
            if not future_list:
                return set(), set()
            return {future_list[0]}, set(future_list[1:])

        monkeypatch.setattr(executor_module, "wait", wait_one)


def make_fake_plan_sample(entry_groups_by_sample_idx):
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


def install_fake_balanced_planner(monkeypatch, entry_groups_by_sample_idx) -> None:
    fake_plan_sample = make_fake_plan_sample(entry_groups_by_sample_idx)

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


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture
def default_recipe() -> ProductionRecipe:
    return ProductionRecipe()
