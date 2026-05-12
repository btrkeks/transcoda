"""Shared dataset image-size specification utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_SPEC_PATH = Path(__file__).resolve().parents[2] / "config" / "data_spec.json"


@dataclass(frozen=True)
class DataSpec:
    image_width: int
    image_height: int


def load_data_spec(path: str | Path = DEFAULT_DATA_SPEC_PATH) -> DataSpec:
    """Load and validate the dataset image-size spec."""
    spec_path = Path(path)
    if not spec_path.exists():
        raise FileNotFoundError(f"Data spec file not found: {spec_path}")

    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in data spec file: {spec_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Data spec must be a JSON object: {spec_path}")

    width = _coerce_positive_int(payload.get("image_width"), key="image_width", spec_path=spec_path)
    height = _coerce_positive_int(payload.get("image_height"), key="image_height", spec_path=spec_path)
    return DataSpec(image_width=width, image_height=height)


def resolve_image_size(
    *,
    image_width: int | None,
    image_height: int | None,
    data_spec: DataSpec,
    strict_data_spec: bool = True,
) -> tuple[int, int]:
    """Resolve final image size against data spec and optional CLI overrides."""
    final_width = data_spec.image_width if image_width is None else int(image_width)
    final_height = data_spec.image_height if image_height is None else int(image_height)

    if final_width <= 0 or final_height <= 0:
        raise ValueError("Resolved image dimensions must be positive")

    if strict_data_spec:
        mismatches: list[str] = []
        if image_width is not None and final_width != data_spec.image_width:
            mismatches.append(f"image_width={final_width} (spec={data_spec.image_width})")
        if image_height is not None and final_height != data_spec.image_height:
            mismatches.append(f"image_height={final_height} (spec={data_spec.image_height})")
        if mismatches:
            mismatch_text = ", ".join(mismatches)
            raise ValueError(
                f"CLI image-size overrides do not match data spec: {mismatch_text}. "
                "Edit config/data_spec.json or disable strict_data_spec."
            )

    return final_width, final_height


def resolve_image_size_from_spec(
    *,
    image_width: int | None,
    image_height: int | None,
    data_spec_path: str | Path = DEFAULT_DATA_SPEC_PATH,
    strict_data_spec: bool = True,
) -> tuple[int, int]:
    """Load data spec and resolve final image size."""
    data_spec = load_data_spec(path=data_spec_path)
    return resolve_image_size(
        image_width=image_width,
        image_height=image_height,
        data_spec=data_spec,
        strict_data_spec=strict_data_spec,
    )


def _coerce_positive_int(value: object, *, key: str, spec_path: Path) -> int:
    if value is None:
        raise ValueError(f"Missing required key '{key}' in data spec: {spec_path}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Key '{key}' must be an integer in data spec: {spec_path}") from exc
    if parsed <= 0:
        raise ValueError(f"Key '{key}' must be > 0 in data spec: {spec_path}")
    return parsed
