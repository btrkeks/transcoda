from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.dataset_generation.dataset_generation.truncation import (
    validate_truncation_candidate_terminal_state,
)

_MALFORMED_TRUNCATION_CANDIDATE = (
    "**kern\t**kern\t**kern\n"
    "4c\t4e\t4g\n"
    "*\t*v\t*v\n"
    "*-\t*-\t*-\n"
)


def test_malformed_merge_terminator_candidate_is_flagged_as_invalid_terminal_state():
    assert (
        validate_truncation_candidate_terminal_state(_MALFORMED_TRUNCATION_CANDIDATE)
        == "invalid_terminal_spine_state"
    )


def test_repaired_merge_terminator_candidate_renders_without_subprocess_abort():
    repo_root = Path(__file__).resolve().parents[2]
    script = f"""
from scripts.dataset_generation.dataset_generation.image_generation.rendering.verovio_backend import VerovioRenderer
from scripts.dataset_generation.dataset_generation.recipe import ProductionRecipe
from scripts.dataset_generation.dataset_generation.renderer import render_sample
from src.core.kern_postprocess import append_terminator_if_missing, strip_terminal_terminator_lines

text = { _MALFORMED_TRUNCATION_CANDIDATE!r }
fixed = append_terminator_if_missing(strip_terminal_terminator_lines(text))
render_sample(fixed, ProductionRecipe(), seed=2303307946, renderer=VerovioRenderer())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
