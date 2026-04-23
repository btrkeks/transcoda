from __future__ import annotations

from notebooks.dataset_viewer import DatasetViewer


def _build_dataset() -> list[dict[str, object]]:
    return [
        {"initial_kern_spine_count": 1, "svg_system_count": 4},
        {"initial_kern_spine_count": 2, "svg_system_count": 7},
        {"initial_kern_spine_count": 2, "svg_system_count": 5},
        {"initial_kern_spine_count": 3, "svg_system_count": 7},
    ]


def test_compute_filtered_indices_returns_all_items_without_filters() -> None:
    viewer = DatasetViewer(_build_dataset(), enable_filters=True)

    assert viewer._compute_filtered_indices() == [0, 1, 2, 3]
    assert viewer.filtered_indices == [0, 1, 2, 3]


def test_compute_filtered_indices_filters_by_spine_class() -> None:
    viewer = DatasetViewer(_build_dataset(), enable_filters=True)

    assert viewer._compute_filtered_indices(spine_class="2") == [1, 2]


def test_compute_filtered_indices_filters_by_spine_class_and_system_count() -> None:
    viewer = DatasetViewer(_build_dataset(), enable_filters=True)

    assert viewer._compute_filtered_indices(spine_class="2", svg_system_count=7) == [1]


def test_derive_spine_class_maps_three_or_more_to_three_plus() -> None:
    assert DatasetViewer._derive_spine_class(3) == "3_plus"
    assert DatasetViewer._derive_spine_class(8) == "3_plus"


def test_filter_change_resets_current_position_to_first_match() -> None:
    viewer = DatasetViewer(_build_dataset(), enable_filters=True)
    viewer.display_current = lambda: None

    viewer.slider.value = 3
    assert viewer.current_idx == 3

    viewer.spine_class_dropdown.value = "2"

    assert viewer.filtered_indices == [1, 2]
    assert viewer.current_idx == 0
    assert viewer.slider.value == 0
    assert viewer.slider.max == 1
    assert viewer.jump_input.value == 0


def test_empty_filter_result_disables_navigation_and_shows_no_match_state(monkeypatch) -> None:
    viewer = DatasetViewer(_build_dataset(), enable_filters=True)
    captured = []
    original_display_current = viewer.display_current

    monkeypatch.setattr("notebooks.dataset_viewer.clear_output", lambda wait=True: None)
    monkeypatch.setattr("notebooks.dataset_viewer.display", lambda obj: captured.append(obj))

    viewer.display_current = lambda: None
    viewer.spine_class_dropdown.value = "1"
    viewer.display_current = original_display_current
    viewer.svg_system_count_dropdown.value = 7

    assert viewer.filtered_indices == []
    assert viewer.slider.disabled is True
    assert viewer.jump_input.disabled is True
    assert viewer.prev_btn.disabled is True
    assert viewer.next_btn.disabled is True
    assert "0 / 4 samples" in viewer.match_summary.value
    assert captured
    assert "No samples match current filters." in captured[0].value
