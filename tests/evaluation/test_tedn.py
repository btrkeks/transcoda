from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.tedn import compute_tedn_from_musicxml

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def test_compute_tedn_from_musicxml_matches_legato_example() -> None:
    prediction = (EXAMPLES_DIR / "duetto_I_bmv_802_page_pred.xml").read_text(encoding="utf-8")
    ground_truth = (EXAMPLES_DIR / "duetto_I_bmv_802.xml").read_text(encoding="utf-8")

    score = compute_tedn_from_musicxml(prediction, ground_truth)

    assert score.tedn_percent == pytest.approx(1.4989293361884368)
    assert score.normalized_edit_cost == pytest.approx(0.014989293361884369)
    assert score.edit_cost == 7
    assert score.gold_cost == 467


def test_compute_tedn_from_musicxml_returns_zero_for_identical_xml() -> None:
    ground_truth = (EXAMPLES_DIR / "duetto_I_bmv_802.xml").read_text(encoding="utf-8")

    score = compute_tedn_from_musicxml(ground_truth, ground_truth)

    assert score.tedn_percent == pytest.approx(0.0)
    assert score.normalized_edit_cost == pytest.approx(0.0)
    assert score.edit_cost == 0
    assert score.gold_cost > 0
