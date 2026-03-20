"""Dataset generation package for synthetic OMR data generation.

This package provides a generator for creating synthetic sheet music images
and their corresponding **kern transcriptions from kern files.

Usage:
    from scripts.dataset_generation.dataset_generation import FileDataGenerator

    generator = FileDataGenerator(
        kern_dir="data/interim/3_normalized",
        num_workers=8
    )

    for sample in generator.generate(num_samples=1000):
        # sample contains 'image' (bytes), 'transcription' (str), 'source' (str)
        process_sample(sample)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts.dataset_generation.dataset_generation.file_generator import FileDataGenerator

__all__ = ["FileDataGenerator"]


def __getattr__(name: str) -> Any:
    """Lazily export FileDataGenerator to keep package import side-effect free."""
    if name == "FileDataGenerator":
        from scripts.dataset_generation.dataset_generation.file_generator import FileDataGenerator

        return FileDataGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
