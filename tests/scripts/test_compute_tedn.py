from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.benchmark import compute_tedn

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def test_compute_tedn_cli_prints_human_output() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[2] / "scripts/benchmark/compute_tedn.py"),
            "--prediction",
            str(EXAMPLES_DIR / "duetto_I_bmv_802_page_pred.xml"),
            "--ground-truth",
            str(EXAMPLES_DIR / "duetto_I_bmv_802.xml"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output = result.stdout
    assert "TEDn: 1.4989293361884368%" in output
    assert "edit_cost: 7" in output
    assert "gold_cost: 467" in output


def test_compute_tedn_cli_prints_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = compute_tedn.main(
        [
            "--prediction",
            str(EXAMPLES_DIR / "duetto_I_bmv_802_page_pred.xml"),
            "--ground-truth",
            str(EXAMPLES_DIR / "duetto_I_bmv_802.xml"),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["tedn_percent"] == pytest.approx(1.4989293361884368)
    assert payload["normalized_edit_cost"] == pytest.approx(0.014989293361884369)
    assert payload["edit_cost"] == 7
    assert payload["gold_cost"] == 467
