"""Main CLI entrypoint for synthetic dataset generation."""

from __future__ import annotations

import sys
from pathlib import Path

import fire

# Keep direct execution behavior consistent with previous implementation.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.dataset_generation.dataset_generation.cli import main


if __name__ == "__main__":
    fire.Fire(main)

