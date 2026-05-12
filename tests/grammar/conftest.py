"""Pytest fixtures for grammar module tests."""

from __future__ import annotations

from pathlib import Path

import pytest

# Path to the project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Path to the kern grammar file
KERN_GRAMMAR_PATH = PROJECT_ROOT / "grammars" / "kern.gbnf"


@pytest.fixture
def kern_grammar_path() -> Path:
    """Return path to the kern.gbnf grammar file."""
    return KERN_GRAMMAR_PATH
