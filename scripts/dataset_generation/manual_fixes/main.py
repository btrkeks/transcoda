#!/usr/bin/env python3
"""Manage manual transcription fixes for small curated datasets.

Workflow:
1. `prepare`: materialize/update working copy (`4_manual_fixes`) from base
   (`3_normalized`) and optional tracked overrides.
2. Manually edit files directly in `4_manual_fixes`.
3. `publish`: export diffs (changed files only) into tracked overrides.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fire


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_bool(value: bool | str | int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _collect_krn_files(root: Path, *, allow_empty: bool = False) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    files = sorted(root.glob("*.krn"))
    if not files and not allow_empty:
        raise ValueError(f"No .krn files found in {root}")
    return {path.name: path for path in files}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line_diff_stats(base_text: str, candidate_text: str) -> tuple[int, int]:
    added = 0
    removed = 0
    for line in difflib.ndiff(base_text.splitlines(), candidate_text.splitlines()):
        if line.startswith("+ "):
            added += 1
        elif line.startswith("- "):
            removed += 1
    return added, removed


def _ensure_file_set_parity(
    *, base_files: dict[str, Path], candidate_files: dict[str, Path], candidate_name: str
) -> None:
    base_names = set(base_files.keys())
    candidate_names = set(candidate_files.keys())
    missing = sorted(base_names - candidate_names)
    extra = sorted(candidate_names - base_names)
    if missing or extra:
        raise ValueError(
            f"{candidate_name} file set mismatch. "
            f"missing={missing[:10]} extra={extra[:10]}"
        )


def _changed_entries(base_files: dict[str, Path], candidate_files: dict[str, Path]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for name in sorted(base_files):
        base_path = base_files[name]
        candidate_path = candidate_files[name]
        base_text = base_path.read_text(encoding="utf-8")
        candidate_text = candidate_path.read_text(encoding="utf-8")
        if base_text == candidate_text:
            continue
        added_lines, removed_lines = _line_diff_stats(base_text, candidate_text)
        changes.append(
            {
                "file": name,
                "base_sha256": _sha256(base_path),
                "working_sha256": _sha256(candidate_path),
                "added_lines": added_lines,
                "removed_lines": removed_lines,
            }
        )
    return changes


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def prepare_manual_fixes(
    *,
    base_dir: str = "data/interim/val/polish-scores/3_normalized",
    working_dir: str = "data/interim/val/polish-scores/4_manual_fixes",
    overrides_dir: str = "curation/polish-scores/overrides",
    manifest_out: str = "reports/dataset_generation/manual_fixes/polish-scores/latest.json",
    refresh: bool | str | int = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """Prepare/update a manually editable working directory.

    Args:
        base_dir: Canonical machine-generated labels (`3_normalized`).
        working_dir: Editable directory for manual fixes (`4_manual_fixes`).
        overrides_dir: Tracked changed-file mirror (changed files only).
        manifest_out: Output JSON manifest path.
        refresh: If true, rebuild `working_dir` from scratch before overlaying overrides.
        quiet: Suppress summary output.
    """
    base_path = Path(base_dir)
    working_path = Path(working_dir)
    overrides_path = Path(overrides_dir)
    manifest_path = Path(manifest_out)
    refresh_bool = _parse_bool(refresh)

    base_files = _collect_krn_files(base_path)

    bootstrapped = False
    if refresh_bool and working_path.exists():
        shutil.rmtree(working_path)
    if not working_path.exists():
        working_path.mkdir(parents=True, exist_ok=True)
        for name, src in base_files.items():
            shutil.copy2(src, working_path / name)
        bootstrapped = True
    else:
        # Preserve user edits while backfilling any newly introduced base files.
        for name, src in base_files.items():
            dst = working_path / name
            if not dst.exists():
                shutil.copy2(src, dst)

    override_files = (
        _collect_krn_files(overrides_path, allow_empty=True) if overrides_path.exists() else {}
    )
    unknown_override_files = sorted(set(override_files.keys()) - set(base_files.keys()))
    if unknown_override_files:
        raise ValueError(
            f"Override files not present in base_dir: {unknown_override_files[:10]}"
        )
    for name, src in override_files.items():
        shutil.copy2(src, working_path / name)

    working_files = _collect_krn_files(working_path)
    _ensure_file_set_parity(
        base_files=base_files,
        candidate_files=working_files,
        candidate_name="working_dir",
    )

    changed_files = _changed_entries(base_files, working_files)
    manifest = {
        "schema_version": "1.0",
        "command": "prepare",
        "generated_at_utc": _utc_now_iso(),
        "base_dir": str(base_path),
        "working_dir": str(working_path),
        "overrides_dir": str(overrides_path),
        "refresh": refresh_bool,
        "bootstrapped_working_dir": bootstrapped,
        "total_files": len(base_files),
        "applied_override_count": len(override_files),
        "applied_override_files": sorted(override_files.keys()),
        "changed_count": len(changed_files),
        "unchanged_count": len(base_files) - len(changed_files),
        "changed_files": changed_files,
    }
    _write_json(manifest_path, manifest)

    if not quiet:
        print(
            f"Prepared manual fixes: total={manifest['total_files']} "
            f"changed={manifest['changed_count']} "
            f"overrides_applied={manifest['applied_override_count']} "
            f"manifest={manifest_path}"
        )

    return manifest


def publish_manual_fixes(
    *,
    base_dir: str = "data/interim/val/polish-scores/3_normalized",
    working_dir: str = "data/interim/val/polish-scores/4_manual_fixes",
    overrides_dir: str = "curation/polish-scores/overrides",
    canonical_manifest_out: str = "curation/polish-scores/overrides_manifest.json",
    quiet: bool = False,
) -> dict[str, Any]:
    """Publish changed files from `working_dir` into tracked override mirror."""
    base_path = Path(base_dir)
    working_path = Path(working_dir)
    overrides_path = Path(overrides_dir)
    canonical_manifest_path = Path(canonical_manifest_out)

    base_files = _collect_krn_files(base_path)
    working_files = _collect_krn_files(working_path)
    _ensure_file_set_parity(
        base_files=base_files,
        candidate_files=working_files,
        candidate_name="working_dir",
    )

    changed_files = _changed_entries(base_files, working_files)

    overrides_path.mkdir(parents=True, exist_ok=True)
    for existing in overrides_path.glob("*.krn"):
        existing.unlink()
    for entry in changed_files:
        filename = entry["file"]
        shutil.copy2(working_files[filename], overrides_path / filename)

    manifest = {
        "schema_version": "1.0",
        "command": "publish",
        "generated_at_utc": _utc_now_iso(),
        "base_dir": str(base_path),
        "working_dir": str(working_path),
        "overrides_dir": str(overrides_path),
        "total_files": len(base_files),
        "changed_count": len(changed_files),
        "unchanged_count": len(base_files) - len(changed_files),
        "override_files": [
            {
                "file": item["file"],
                "sha256": item["working_sha256"],
                "added_lines": item["added_lines"],
                "removed_lines": item["removed_lines"],
            }
            for item in changed_files
        ],
    }
    _write_json(canonical_manifest_path, manifest)

    if not quiet:
        print(
            f"Published overrides: total={manifest['total_files']} "
            f"changed={manifest['changed_count']} "
            f"overrides_dir={overrides_path} "
            f"manifest={canonical_manifest_path}"
        )

    return manifest


class ManualFixesCLI:
    """Fire CLI wrapper."""

    def prepare(
        self,
        base_dir: str = "data/interim/val/polish-scores/3_normalized",
        working_dir: str = "data/interim/val/polish-scores/4_manual_fixes",
        overrides_dir: str = "curation/polish-scores/overrides",
        manifest_out: str = "reports/dataset_generation/manual_fixes/polish-scores/latest.json",
        refresh: bool | str | int = False,
        quiet: bool = False,
    ) -> dict[str, Any]:
        return prepare_manual_fixes(
            base_dir=base_dir,
            working_dir=working_dir,
            overrides_dir=overrides_dir,
            manifest_out=manifest_out,
            refresh=refresh,
            quiet=quiet,
        )

    def publish(
        self,
        base_dir: str = "data/interim/val/polish-scores/3_normalized",
        working_dir: str = "data/interim/val/polish-scores/4_manual_fixes",
        overrides_dir: str = "curation/polish-scores/overrides",
        canonical_manifest_out: str = "curation/polish-scores/overrides_manifest.json",
        quiet: bool = False,
    ) -> dict[str, Any]:
        return publish_manual_fixes(
            base_dir=base_dir,
            working_dir=working_dir,
            overrides_dir=overrides_dir,
            canonical_manifest_out=canonical_manifest_out,
            quiet=quiet,
        )


def main() -> None:
    fire.Fire(ManualFixesCLI)


if __name__ == "__main__":
    main()
