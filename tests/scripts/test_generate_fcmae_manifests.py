from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.generate_fcmae_manifests import generate_manifests


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), 255, dtype=np.uint8)).save(path)


def test_generate_fcmae_manifests_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    image_root = tmp_path / "fcmae_images"
    for index in range(12):
        _write_image(image_root / f"source-{index % 3}" / f"page-{index:02d}.png")

    first_output = tmp_path / "first" / "manifests"
    second_output = tmp_path / "second" / "manifests"

    first = generate_manifests(
        image_root=image_root,
        output_dir=first_output,
        validation_size=4,
        preview_size=2,
        seed=123,
    )
    second = generate_manifests(
        image_root=image_root,
        output_dir=second_output,
        validation_size=4,
        preview_size=2,
        seed=123,
    )

    train_lines = set(first["train"].read_text(encoding="utf-8").splitlines())
    validation_lines = set(first["validation"].read_text(encoding="utf-8").splitlines())
    preview_lines = set(first["preview"].read_text(encoding="utf-8").splitlines())
    second_validation = set(second["validation"].read_text(encoding="utf-8").splitlines())
    metadata = json.loads(first["metadata"].read_text(encoding="utf-8"))

    assert len(validation_lines) == 4
    assert len(preview_lines) == 2
    assert train_lines.isdisjoint(validation_lines)
    assert preview_lines <= validation_lines
    assert validation_lines == second_validation
    assert metadata["validation_count"] == 4
    assert metadata["preview_count"] == 2


def test_generate_fcmae_manifests_rejects_validation_that_consumes_all_images(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "fcmae_images"
    _write_image(image_root / "a.png")

    try:
        generate_manifests(
            image_root=image_root,
            output_dir=tmp_path / "manifests",
            validation_size=1,
            preview_size=1,
            seed=0,
        )
    except ValueError as exc:
        assert "leaves no training images" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
