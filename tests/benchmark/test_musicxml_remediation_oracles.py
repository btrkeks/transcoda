from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.benchmark.conversion import ensure_humdrum_document
from src.grammar.rhythm_rule import RhythmRule
from src.grammar.semantic_sequence_finalizer import finalize_kern_sequence_text
from src.grammar.spine_structure_rule import SpineStructureRule

_FIXTURE_DIR = Path("tests/fixtures/musicxml_conversion_remediation")


def _run_rhythm_checker(path: Path) -> dict:
    result = subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "--manifest-path",
            "binaries/rhythm_checker/Cargo.toml",
            "--",
            "--format",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("hum2xml") is None, reason="hum2xml is not installed")
@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
@pytest.mark.parametrize(
    "fixture_name",
    [
        "test_000004_excerpt.krn",
        "test_000008_excerpt.krn",
        "test_000009_excerpt.krn",
        "test_000013_excerpt.krn",
    ],
)
def test_finalizer_repairs_curated_musicxml_regressions(
    tmp_path: Path,
    fixture_name: str,
) -> None:
    body = (_FIXTURE_DIR / fixture_name).read_text(encoding="utf-8")

    finalized = finalize_kern_sequence_text(
        text=body,
        saw_eos=False,
        hit_max_length=True,
        rule_factories=(SpineStructureRule, RhythmRule),
    )

    if not finalized.text:
        assert finalized.trimmed_incomplete_tail is True
        assert finalized.truncated is True
        return

    document = ensure_humdrum_document(finalized.text)
    sample_path = tmp_path / fixture_name
    sample_path.write_text(document, encoding="utf-8")

    hum2xml = subprocess.run(
        ["hum2xml", str(sample_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert hum2xml.returncode == 0, hum2xml.stderr or hum2xml.stdout

    rhythm = _run_rhythm_checker(sample_path)
    assert rhythm["total_errors"] == 0
