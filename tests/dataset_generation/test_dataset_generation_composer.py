from pathlib import Path

import pytest

import scripts.dataset_generation.dataset_generation.composer as composer_module
from scripts.dataset_generation.dataset_generation.composer import (
    _choose_entries,
    compose_label_transcription,
    plan_sample,
)
from scripts.dataset_generation.dataset_generation.recipe import CompositionPolicy, ProductionRecipe
from scripts.dataset_generation.dataset_generation.renderer import count_systems_in_svg
from scripts.dataset_generation.dataset_generation.source_index import build_source_index
from scripts.dataset_generation.dataset_generation.types import SourceEntry
from src.core.kern_concatenation import (
    diagnose_spine_topology,
    restore_terminal_spine_count_before_final_barline,
    summarize_spine_topology,
)


class _FixedRng:
    def __init__(self, randrange_values: list[int], random_values: list[float] | None = None):
        self._randrange_values = list(randrange_values)
        self._random_values = list(random_values or [])

    def randrange(self, stop: int) -> int:
        value = self._randrange_values.pop(0) if self._randrange_values else 0
        return value % stop

    def random(self) -> float:
        return self._random_values.pop(0) if self._random_values else 0.0

    def randint(self, a: int, b: int) -> int:
        del b
        return a


def _make_source_entry(path: Path, source_id: str, text: str) -> SourceEntry:
    topology = summarize_spine_topology(text)
    restored_topology = summarize_spine_topology(
        restore_terminal_spine_count_before_final_barline(text)
    )
    assert topology.initial_spine_count is not None
    assert topology.terminal_spine_count is not None
    assert restored_topology.terminal_spine_count is not None
    return SourceEntry(
        path=path,
        source_id=source_id,
        root_dir=path.parent,
        root_label=path.parent.name,
        measure_count=1,
        non_empty_line_count=4,
        has_header=True,
        initial_spine_count=topology.initial_spine_count,
        terminal_spine_count=topology.terminal_spine_count,
        restored_terminal_spine_count=restored_topology.terminal_spine_count,
    )


def _compose_entries(tmp_path: Path, *texts: str) -> str:
    entries: list[SourceEntry] = []
    for idx, text in enumerate(texts):
        path = tmp_path / f"snippet_{idx}.krn"
        path.write_text(text, encoding="utf-8")
        entries.append(_make_source_entry(path, f"snippet_{idx}", text))
    return compose_label_transcription(entries)


def test_compose_label_transcription_omits_duplicate_boundary_headers(tmp_path: Path):
    first = tmp_path / "first.krn"
    second = tmp_path / "second.krn"
    first.write_text("**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4c\n*-\n", encoding="utf-8")
    second.write_text("**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4d\n*-\n", encoding="utf-8")

    composed = compose_label_transcription(
        [
            _make_source_entry(first, "first", first.read_text(encoding="utf-8")),
            _make_source_entry(second, "second", second.read_text(encoding="utf-8")),
        ]
    )

    assert composed.count("**kern") == 1
    assert composed.count("*-") == 1
    assert composed.count("*clefG2") == 1
    assert composed.count("*k[]") == 1
    assert composed.count("*M4/4") == 1
    assert composed.endswith("*-\n")
    assert "4c" in composed
    assert "4d" in composed


def test_compose_label_transcription_inserts_only_changed_meter(tmp_path: Path):
    composed = _compose_entries(
        tmp_path,
        "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4c\n*-\n",
        "**kern\n*clefG2\n*k[]\n*M3/4\n=1\n4d\n*-\n",
    )

    assert composed.count("*clefG2") == 1
    assert composed.count("*k[]") == 1
    assert composed.count("*M4/4") == 1
    assert composed.count("*M3/4") == 1


def test_compose_label_transcription_inserts_only_changed_key(tmp_path: Path):
    composed = _compose_entries(
        tmp_path,
        "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4c\n*-\n",
        "**kern\n*clefG2\n*k[f#]\n*M4/4\n=1\n4d\n*-\n",
    )

    assert composed.count("*clefG2") == 1
    assert composed.count("*M4/4") == 1
    assert composed.count("*k[]") == 1
    assert composed.count("*k[f#]") == 1


def test_compose_label_transcription_inserts_only_changed_clef(tmp_path: Path):
    composed = _compose_entries(
        tmp_path,
        "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4c\n*-\n",
        "**kern\n*clefF4\n*k[]\n*M4/4\n=1\n4d\n*-\n",
    )

    assert composed.count("*clefG2") == 1
    assert composed.count("*clefF4") == 1
    assert composed.count("*k[]") == 1
    assert composed.count("*M4/4") == 1


def test_compose_label_transcription_emits_per_spine_transition_lines(tmp_path: Path):
    composed = _compose_entries(
        tmp_path,
        "**kern\t**kern\n*clefF4\t*clefG2\n*k[]\t*k[]\n*M4/4\t*M4/4\n=1\t=1\n4c\t4e\n*-\t*-\n",
        "**kern\t**kern\n*clefF4\t*clefGv2\n*k[]\t*k[]\n*M4/4\t*M4/4\n=1\t=1\n4d\t4f\n*-\t*-\n",
    )

    assert "*\t*clefGv2" in composed
    assert "*clefF4\t*clefGv2" not in composed


def test_compose_label_transcription_uses_terminal_meter_state_at_boundary(tmp_path: Path):
    composed = _compose_entries(
        tmp_path,
        "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4c\n*M3/4\n=2\n4d\n*-\n",
        "**kern\n*clefG2\n*k[]\n*M3/4\n=1\n4e\n*-\n",
    )

    assert composed.count("*M3/4") == 1


def test_compose_label_transcription_uses_terminal_key_and_clef_state_at_boundary(
    tmp_path: Path,
):
    composed = _compose_entries(
        tmp_path,
        "**kern\t**kern\n*clefF4\t*clefG2\n*k[]\t*k[]\n=1\t=1\n4c\t4e\n*clefF4\t*clefGv2\n*k[]\t*k[f#]\n=2\t=2\n4d\t4f\n*-\t*-\n",
        "**kern\t**kern\n*clefF4\t*clefGv2\n*k[]\t*k[f#]\n=1\t=1\n4g\t4b\n*-\t*-\n",
    )

    assert composed.count("*clefF4\t*clefGv2") == 1
    assert composed.count("*k[]\t*k[f#]") == 1


def test_compose_label_transcription_keeps_single_final_terminator_across_many_segments(
    tmp_path: Path,
):
    composed = _compose_entries(
        tmp_path,
        "**kern\n*clefG2\n=1\n4c\n*-\n",
        "**kern\n*clefG2\n=1\n4d\n*-\n",
        "**kern\n*clefG2\n=1\n4e\n*-\n",
    )

    assert composed.count("*-") == 1
    assert composed.endswith("*-\n")


def test_compose_label_transcription_drops_non_tracked_leading_lines_from_followup(
    tmp_path: Path,
):
    composed = _compose_entries(
        tmp_path,
        "**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4c\n*-\n",
        "!! source comment\n**kern\n*MM120\n*clefG2\n*k[]\n*M4/4\n=1\n4d\n*-\n",
    )

    assert "!! source comment" not in composed
    assert "*MM120" not in composed


def test_compose_label_transcription_restores_nonfinal_segment_before_followup(tmp_path: Path):
    composed = _compose_entries(
        tmp_path,
        "**kern\n*clefG2\n=1\n4c\n*^\n*\t*\n=2\t=2\n4d\t4f\n==\t==\n*-\t*-\n",
        "**kern\n*clefG2\n=1\n4e\n*-\n",
    )

    assert composed.count("**kern") == 1
    assert "*v\t*v" in composed
    assert composed.endswith("*-\n")
    assert "4c" in composed
    assert "4d\t4f" in composed
    assert "4e" in composed


def test_compose_label_transcription_rejects_unrestorable_boundary_spine_count_mismatch(
    tmp_path: Path,
):
    first = (
        "**kern\t**kern\n*clefF4\t*clefG2\n=1\t=1\n4c\t4e\n==\t==\n*-\t*-\n"
    )
    second = "**kern\n*clefG2\n=1\n4d\n*-\n"

    with pytest.raises(ValueError, match="Boundary spine count mismatch"):
        _compose_entries(tmp_path, first, second)


def test_compose_label_transcription_routes_output_through_terminal_spine_restorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[str] = []

    def fake_restore(text: str) -> str:
        calls.append(text)
        return text

    monkeypatch.setattr(
        composer_module,
        "restore_terminal_spine_count_before_final_barline",
        fake_restore,
    )

    composed = _compose_entries(
        tmp_path,
        "**kern\n*clefG2\n=1\n4c\n*-\n",
        "**kern\n*clefG2\n=1\n4d\n*-\n",
    )

    assert len(calls) == 2
    assert calls[0].endswith("*-\n")
    assert calls[1].endswith("*-")
    assert composed.endswith("*-\n")


def test_plan_sample_is_deterministic_for_seed(tmp_path: Path):
    for idx in range(3):
        (tmp_path / f"{idx:03d}.krn").write_text(
            "*clefG2\n*M4/4\n=1\n4c\n=2\n4d\n*-\n",
            encoding="utf-8",
        )

    source_index = build_source_index(tmp_path)
    recipe = ProductionRecipe()

    first = plan_sample(source_index, recipe, sample_idx=7, base_seed=123)
    second = plan_sample(source_index, recipe, sample_idx=7, base_seed=123)

    assert first == second


def test_plan_sample_respects_single_segment_policy_even_below_min_measures(tmp_path: Path):
    for idx in range(3):
        (tmp_path / f"{idx:03d}.krn").write_text(
            "*clefG2\n=1\n4c\n*-\n",
            encoding="utf-8",
        )

    source_index = build_source_index(tmp_path)
    recipe = ProductionRecipe(
        composition=CompositionPolicy(
            segment_count_weights=((1, 1.0),),
            min_total_measures=10,
            max_total_measures=32,
            max_selection_attempts=48,
        )
    )

    plan = plan_sample(source_index, recipe, sample_idx=3, base_seed=99)

    assert plan.segment_count == 1
    assert len(plan.segments) == 1


def test_plan_sample_tracks_max_initial_spine_count_across_segments(tmp_path: Path, monkeypatch):
    first_double_path = tmp_path / "double_a.krn"
    second_double_path = tmp_path / "double_b.krn"
    first_double_text = "**kern\t**kern\n=1\t=1\n4c\t4e\n*-\t*-\n"
    second_double_text = "**kern\t**kern\n=1\t=1\n4d\t4f\n*-\t*-\n"
    first_double_path.write_text(first_double_text, encoding="utf-8")
    second_double_path.write_text(second_double_text, encoding="utf-8")

    source_index = build_source_index(tmp_path)
    recipe = ProductionRecipe(
        composition=CompositionPolicy(
            segment_count_weights=((2, 1.0),),
            min_total_measures=1,
            max_total_measures=8,
            max_selection_attempts=4,
        )
    )

    monkeypatch.setattr(
        "scripts.dataset_generation.dataset_generation.composer._choose_entries",
        lambda available_entries, recipe, rng, excluded_paths=None: (
            next(entry for entry in available_entries if entry.path == first_double_path),
            next(entry for entry in available_entries if entry.path == second_double_path),
        ),
    )

    plan = plan_sample(source_index, recipe, sample_idx=0, base_seed=0)

    assert plan.segment_count == 2
    assert plan.source_max_initial_spine_count == 2


def test_choose_entries_uses_restored_terminal_spine_count_for_compatibility(tmp_path: Path):
    anchor_text = "**kern\n*clefG2\n=1\n4c\n*^\n*\t*\n==\t==\n*-\t*-\n"
    compatible_text = "**kern\n*clefG2\n=1\n4e\n*-\n"
    incompatible_text = "**kern\t**kern\n*clefG2\t*clefG2\n=1\t=1\n4d\t4f\n*-\t*-\n"
    anchor_path = tmp_path / "anchor.krn"
    compatible_path = tmp_path / "compatible.krn"
    incompatible_path = tmp_path / "incompatible.krn"
    anchor_path.write_text(anchor_text, encoding="utf-8")
    compatible_path.write_text(compatible_text, encoding="utf-8")
    incompatible_path.write_text(incompatible_text, encoding="utf-8")
    anchor = _make_source_entry(anchor_path, "anchor", anchor_text)
    compatible = _make_source_entry(compatible_path, "compatible", compatible_text)
    incompatible = _make_source_entry(incompatible_path, "incompatible", incompatible_text)

    chosen = _choose_entries(
        [anchor, compatible, incompatible],
        ProductionRecipe(
            composition=CompositionPolicy(
                segment_count_weights=((2, 1.0),),
                min_total_measures=1,
                max_total_measures=8,
                max_selection_attempts=2,
            )
        ),
        _FixedRng([0, 0], [0.0, 0.0]),
    )

    assert tuple(entry.source_id for entry in chosen) == ("anchor", "compatible")


def test_choose_entries_stops_when_no_boundary_compatible_followup_exists(
    tmp_path: Path,
):
    anchor_text = "**kern\t**kern\n*clefF4\t*clefG2\n=1\t=1\n4c\t4e\n==\t==\n*-\t*-\n"
    incompatible_text = "**kern\n*clefG2\n=1\n4e\n*-\n"
    anchor_path = tmp_path / "anchor.krn"
    incompatible_path = tmp_path / "incompatible.krn"
    anchor_path.write_text(anchor_text, encoding="utf-8")
    incompatible_path.write_text(incompatible_text, encoding="utf-8")
    anchor = _make_source_entry(anchor_path, "anchor", anchor_text)
    incompatible = _make_source_entry(incompatible_path, "incompatible", incompatible_text)

    chosen = _choose_entries(
        [anchor, incompatible],
        ProductionRecipe(
            composition=CompositionPolicy(
                segment_count_weights=((2, 1.0),),
                min_total_measures=1,
                max_total_measures=8,
                max_selection_attempts=2,
            )
        ),
        _FixedRng([0], [0.0]),
    )

    assert len(chosen) == 1
    assert tuple(entry.source_id for entry in chosen) == ("anchor",)


def test_build_source_index_combines_multiple_roots(tmp_path: Path):
    first_root = tmp_path / "a"
    second_root = tmp_path / "b"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "one.krn").write_text("*clefG2\n=1\n4c\n*-\n", encoding="utf-8")
    (second_root / "two.krn").write_text("*clefG2\n=1\n4d\n*-\n", encoding="utf-8")

    source_index = build_source_index(first_root, second_root)

    assert len(source_index.entries) == 2
    assert source_index.invalid_sources == ()
    assert {entry.source_id for entry in source_index.entries} == {"a/one", "b/two"}
    assert {entry.initial_spine_count for entry in source_index.entries} == {1}
    assert {entry.terminal_spine_count for entry in source_index.entries} == {1}


def test_build_source_index_disambiguates_duplicate_root_names(tmp_path: Path):
    first_root = tmp_path / "train" / "grandstaff" / "3_normalized"
    second_root = tmp_path / "train" / "musetrainer" / "3_normalized"
    first_root.mkdir(parents=True)
    second_root.mkdir(parents=True)
    (first_root / "one.krn").write_text("*clefG2\n=1\n4c\n*-\n", encoding="utf-8")
    (second_root / "two.krn").write_text("*clefG2\n=1\n4d\n*-\n", encoding="utf-8")

    source_index = build_source_index(first_root, second_root)

    assert len(source_index.entries) == 2
    assert {entry.source_id for entry in source_index.entries} == {
        "grandstaff/3_normalized/one",
        "musetrainer/3_normalized/two",
    }


def test_build_source_index_tracks_terminal_spine_count_after_split(tmp_path: Path):
    root = tmp_path / "input"
    root.mkdir()
    (root / "piece.krn").write_text("*clefG2\n=1\n4c\n*^\n4d\t4f\n*-\t*-\n", encoding="utf-8")

    source_index = build_source_index(root)

    assert len(source_index.entries) == 1
    entry = source_index.entries[0]
    assert entry.initial_spine_count == 1
    assert entry.terminal_spine_count == 2
    assert entry.restored_terminal_spine_count == 2


def test_build_source_index_tracks_restored_terminal_spine_count_when_repairable(tmp_path: Path):
    root = tmp_path / "input"
    root.mkdir()
    (root / "piece.krn").write_text(
        "**kern\n=1\n4c\n*^\n*\t*\n==\t==\n*-\t*-\n",
        encoding="utf-8",
    )

    source_index = build_source_index(root)

    entry = source_index.entries[0]
    assert entry.initial_spine_count == 1
    assert entry.terminal_spine_count == 2
    assert entry.restored_terminal_spine_count == 1


def test_diagnose_spine_topology_reports_width_mismatch_details():
    diagnostic = diagnose_spine_topology(
        "\n".join(
            [
                "*clefF4\t*clefG2",
                "*k[b-e-a-d-g-]\t*k[b-e-a-d-g-]",
                "*M2/4\t*M2/4",
                "*^\t*",
                "*\t*^\t*",
                "8d-L\t2d-\t2BB-\t16.fLL",
                ".\t.\t.\t32eJJk",
                "8E\t.\t.\t4.g",
                "8G\t.\t.\t.",
                "8EJ\t.\t.\t.",
                "=\t=\t=\t=",
                "8d-L\t2d-\t2BB-\t16.fLL",
                ".\t.\t.\t32eJJk",
                "8E\t.\t.\t4g",
                "8G\t.\t.\t.",
                "8EJ\t.\t.\t16.gLL",
                ".\t.\t.\t32dd-JJk",
                "*v\t*v\t*v\t*",
                "=\t=\t=",
            ]
        )
    )

    assert diagnostic is not None
    assert diagnostic.reason_code == "width_mismatch"
    assert diagnostic.line_number == 19
    assert diagnostic.expected_spine_count == 2
    assert diagnostic.actual_spine_count == 3
    assert diagnostic.line_text == "=\t=\t="


def test_build_source_index_skips_invalid_sources_and_records_diagnostics(tmp_path: Path):
    root = tmp_path / "input"
    root.mkdir()
    (root / "valid.krn").write_text("*clefG2\n=1\n4c\n*-\n", encoding="utf-8")
    (root / "invalid.krn").write_text(
        "*clefF4\t*clefG2\n"
        "*^\t*\n"
        "*\t*^\t*\n"
        "4c\t4e\t4g\t4b\n"
        "*v\t*v\t*v\t*\n"
        "=\t=\t=\n",
        encoding="utf-8",
    )

    source_index = build_source_index(root)

    assert [entry.source_id for entry in source_index.entries] == ["input/valid"]
    assert len(source_index.invalid_sources) == 1
    diagnostic = source_index.invalid_sources[0]
    assert diagnostic.path == (root / "invalid.krn").resolve()
    assert diagnostic.root_dir == root.resolve()
    assert diagnostic.root_label == "input"
    assert diagnostic.reason_code == "width_mismatch"
    assert "expected 2 spine field(s), got 3" in diagnostic.message


def test_count_systems_in_svg_uses_svg_structure():
    svg = """
    <svg>
      <g class="system"></g>
      <g class="foo system bar"></g>
    </svg>
    """

    assert count_systems_in_svg(svg) == 2
