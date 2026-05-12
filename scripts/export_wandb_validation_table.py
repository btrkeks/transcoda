#!/usr/bin/env python3
"""Export a W&B validation table into per-sample review folders."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

DEFAULT_BUNDLE_DIR = Path("wandb/run_ztpp6k36_bundle")
REQUIRED_COLUMNS = ("ID", "Category", "Ground Truth Image", "Ground Truth", "Prediction")
OPTIONAL_METADATA_COLUMNS = ("SER", "CER", "CER_no_ties_beams")
SET_COLUMNS = (
    "Validation Set",
    "Val Set",
    "Set",
    "Dataset",
    "val_set_name",
    "validation_set",
)


def _safe_path_part(value: Any, fallback: str) -> str:
    raw = str(value or "").strip()
    if raw:
        raw = Path(raw).name
        raw = re.sub(r"\.[A-Za-z0-9]{1,8}$", "", raw)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    return cleaned or fallback


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_validation_set(row_id: Any, explicit_value: Any = None) -> str:
    if explicit_value is not None and str(explicit_value).strip():
        return _safe_path_part(explicit_value, "unknown").lower()

    row_id_text = str(row_id or "").strip().lower()
    if "polish" in row_id_text:
        return "polish"
    if "synth" in row_id_text:
        return "synth"
    if row_id_text == "unknown":
        return "synth"
    if re.match(r"^(train|test|val)_\d+", row_id_text):
        return "polish"
    return "unknown_set"


def _load_table(table_path: Path) -> tuple[list[str], list[list[Any]]]:
    with table_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    columns = payload.get("columns")
    data = payload.get("data")
    if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
        raise ValueError(f"{table_path} does not look like a W&B table: missing columns")
    if not isinstance(data, list):
        raise ValueError(f"{table_path} does not look like a W&B table: missing data rows")

    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"{table_path} is missing required columns: {', '.join(missing)}")

    return columns, data


def _resolve_table_path(bundle_or_table: Path) -> Path:
    if bundle_or_table.is_file():
        return bundle_or_table

    candidates = sorted(bundle_or_table.glob("qualitative/*.table.json"))
    if not candidates:
        candidates = sorted(bundle_or_table.glob("**/*.table.json"))
    if not candidates:
        raise FileNotFoundError(f"could not find a *.table.json under {bundle_or_table}")
    if len(candidates) > 1:
        names = "\n  ".join(str(p) for p in candidates)
        raise ValueError(f"found multiple table files; pass one explicitly:\n  {names}")
    return candidates[0]


def _media_path(table_path: Path, media_value: Any) -> Path | None:
    if not isinstance(media_value, dict):
        return None
    raw_path = media_value.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return table_path.parent / path


def _copy_optional_media(
    table_path: Path,
    row_dir: Path,
    media_value: Any,
    output_name: str,
    *,
    required: bool = False,
) -> str | None:
    src = _media_path(table_path, media_value)
    if src is None:
        return None
    if not src.exists():
        if not required:
            return None
        raise FileNotFoundError(f"media file referenced by table row does not exist: {src}")
    dst = row_dir / f"{output_name}{src.suffix}"
    shutil.copy2(src, dst)
    return dst.name


def _write_text(path: Path, value: Any) -> None:
    text = "" if value is None else str(value)
    path.write_text(text.rstrip("\n") + "\n", encoding="utf-8")


def export_validation_table(table_path: Path, output_dir: Path, overwrite: bool = False) -> int:
    table_path = table_path.resolve()
    columns, rows = _load_table(table_path)
    column_index = {name: i for i, name in enumerate(columns)}
    set_column = next((name for name in SET_COLUMNS if name in column_index), None)

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)
    counts: dict[str, int] = {}
    set_counts: dict[str, int] = {}

    for row_index, row in enumerate(rows):
        if not isinstance(row, list):
            raise ValueError(f"row {row_index} is not a list")

        def cell(column: str, current_row: list[Any] = row) -> Any:
            index = column_index[column]
            return current_row[index] if index < len(current_row) else None

        category = _safe_path_part(cell("Category"), "uncategorized").lower()
        validation_set = _infer_validation_set(
            cell("ID"),
            cell(set_column) if set_column is not None else None,
        )
        sample_id = _safe_path_part(cell("ID"), "unknown")
        row_dir = output_dir / category / validation_set / f"{row_index:03d}_{sample_id}"
        row_dir.mkdir(parents=True)

        image_name = _copy_optional_media(
            table_path,
            row_dir,
            cell("Ground Truth Image"),
            "image",
            required=True,
        )
        diff_name = None
        if "Diff" in column_index:
            diff_name = _copy_optional_media(table_path, row_dir, cell("Diff"), "diff")

        _write_text(row_dir / "ground_truth.krn", cell("Ground Truth"))
        _write_text(row_dir / "prediction.krn", cell("Prediction"))

        metadata = {
            "row_index": row_index,
            "id": cell("ID"),
            "category": cell("Category"),
            "validation_set": validation_set,
            "image": image_name,
            "diff": diff_name,
            "metrics": {
                name: _parse_float(cell(name))
                for name in OPTIONAL_METADATA_COLUMNS
                if name in column_index
            },
        }
        (row_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        counts[category] = counts.get(category, 0) + 1
        set_counts[validation_set] = set_counts.get(validation_set, 0) + 1

    (output_dir / "index.json").write_text(
        json.dumps(
            {
                "source_table": str(table_path),
                "row_count": len(rows),
                "categories": dict(sorted(counts.items())),
                "validation_sets": dict(sorted(set_counts.items())),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Turn a W&B Validation Table.table.json plus media/ directory into "
            "table/{best,worst}/{polish,synth}/sample folders for manual review."
        )
    )
    parser.add_argument(
        "bundle_or_table",
        nargs="?",
        type=Path,
        default=DEFAULT_BUNDLE_DIR,
        help=(
            "W&B bundle directory or a specific *.table.json file "
            f"(default: {DEFAULT_BUNDLE_DIR})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <bundle>/table, or sibling table/ for a table path).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before exporting.",
    )
    args = parser.parse_args()

    table_path = _resolve_table_path(args.bundle_or_table)
    if args.output_dir is None:
        output_dir = (
            table_path.parents[1] / "table"
            if table_path.parent.name == "qualitative"
            else table_path.parent / "table"
        )
    else:
        output_dir = args.output_dir

    row_count = export_validation_table(table_path, output_dir, overwrite=args.overwrite)
    print(f"Exported {row_count} rows to {output_dir}")


if __name__ == "__main__":
    main()
