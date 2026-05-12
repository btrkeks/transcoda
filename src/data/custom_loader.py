"""
Utility functions for loading custom image/transcription pairs from raw data directory.

This module provides reusable functions to:
1. Discover matching image/transcription pairs by basename
2. Load raw **kern transcription files (no preprocessing)
3. Create HuggingFace Dataset objects compatible with existing datasets

Note: Preprocessing and normalization should be applied separately via
the normalization pipeline after loading.
"""

from pathlib import Path

from datasets import Dataset, Features, Value
from datasets import Image as HFImage
from PIL import Image


def find_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    """
    Find matching image/transcription pairs in a directory.

    Searches for .krn files and matches them with corresponding image files
    (.png or .jpg) with the same basename.

    Args:
        raw_dir: Path to directory containing raw data files

    Returns:
        List of (image_path, krn_path) tuples for matched pairs

    Example:
        >>> pairs = find_pairs(Path("data/raw"))
        >>> for img_path, krn_path in pairs:
        ...     print(f"{img_path.name} <-> {krn_path.name}")
    """
    pairs = []
    krn_files = list(raw_dir.glob("*.krn"))

    for krn_file in krn_files:
        basename = krn_file.stem

        # Check for .png first, then .jpg
        img_file = raw_dir / f"{basename}.png"
        if not img_file.exists():
            img_file = raw_dir / f"{basename}.jpg"

        if img_file.exists():
            pairs.append((img_file, krn_file))
        else:
            print(f"Warning: No image found for {krn_file.name}")

    return pairs


def load_custom_pairs(raw_dir: Path) -> Dataset:
    """
    Load custom image/transcription pairs into a HuggingFace Dataset.

    Discovers all matching pairs in the directory, loads raw transcriptions
    (no preprocessing), loads images as PIL objects, and creates a Dataset
    with the standard schema.

    Note: Transcriptions are loaded as-is. Apply preprocessing and normalization
    separately via the normalization pipeline after loading.

    Args:
        raw_dir: Path to directory containing raw data files

    Returns:
        HuggingFace Dataset with 'image' and 'transcription' columns

    Example:
        >>> from pathlib import Path
        >>> dataset = load_custom_pairs(Path("data/raw"))
        >>> print(f"Loaded {len(dataset)} custom samples")
        >>> print(dataset.features)
        {'image': Image(mode='RGB', decode=True), 'transcription': Value('string')}
    """
    pairs = find_pairs(raw_dir)

    if not pairs:
        print(f"Warning: No image/transcription pairs found in {raw_dir}")
        # Return empty dataset with correct schema
        return Dataset.from_dict(
            {"image": [], "transcription": []},
            features=Features({"image": HFImage(mode="RGB"), "transcription": Value("string")}),
        )

    images = []
    transcriptions = []

    for img_path, krn_path in pairs:
        # Load and convert image to RGB
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)

        # Load raw transcription
        with open(krn_path) as f:
            transcription = f.read().strip()
        transcriptions.append(transcription)

    # Create dataset with proper schema
    features = Features({"image": HFImage(mode="RGB"), "transcription": Value("string")})

    dataset = Dataset.from_dict(
        {"image": images, "transcription": transcriptions}, features=features
    )

    print(f"✓ Loaded {len(dataset)} custom sample(s) from {raw_dir}")
    return dataset
