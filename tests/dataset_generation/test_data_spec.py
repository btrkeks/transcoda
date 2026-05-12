import json

import pytest

from scripts.dataset_generation.data_spec import (
    DataSpec,
    load_data_spec,
    resolve_image_size,
    resolve_image_size_from_spec,
)


def test_load_data_spec_success(tmp_path):
    spec_path = tmp_path / "data_spec.json"
    spec_path.write_text(
        json.dumps({"image_width": 1200, "image_height": 1800}),
        encoding="utf-8",
    )

    spec = load_data_spec(spec_path)

    assert spec == DataSpec(image_width=1200, image_height=1800)


def test_load_data_spec_requires_keys(tmp_path):
    spec_path = tmp_path / "data_spec.json"
    spec_path.write_text(json.dumps({"image_width": 1200}), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required key 'image_height'"):
        load_data_spec(spec_path)


def test_resolve_image_size_uses_spec_defaults():
    width, height = resolve_image_size(
        image_width=None,
        image_height=None,
        data_spec=DataSpec(image_width=1050, image_height=1485),
        strict_data_spec=True,
    )

    assert (width, height) == (1050, 1485)


def test_resolve_image_size_rejects_strict_mismatch():
    with pytest.raises(ValueError, match="CLI image-size overrides do not match data spec"):
        resolve_image_size(
            image_width=999,
            image_height=1485,
            data_spec=DataSpec(image_width=1050, image_height=1485),
            strict_data_spec=True,
        )


def test_resolve_image_size_allows_non_strict_override():
    width, height = resolve_image_size(
        image_width=999,
        image_height=1001,
        data_spec=DataSpec(image_width=1050, image_height=1485),
        strict_data_spec=False,
    )

    assert (width, height) == (999, 1001)


def test_resolve_image_size_from_spec(tmp_path):
    spec_path = tmp_path / "data_spec.json"
    spec_path.write_text(
        json.dumps({"image_width": 900, "image_height": 1300}),
        encoding="utf-8",
    )

    width, height = resolve_image_size_from_spec(
        image_width=None,
        image_height=None,
        data_spec_path=spec_path,
        strict_data_spec=True,
    )

    assert (width, height) == (900, 1300)
