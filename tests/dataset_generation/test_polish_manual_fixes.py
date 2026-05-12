from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.dataset_generation.manual_fixes.main import (
    prepare_manual_fixes,
    publish_manual_fixes,
)


def _write_krn(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_prepare_bootstraps_working_dir_and_writes_manifest(tmp_path):
    base_dir = tmp_path / "interim" / "3_normalized"
    working_dir = tmp_path / "interim" / "4_manual_fixes"
    overrides_dir = tmp_path / "curation" / "overrides"
    manifest_out = tmp_path / "reports" / "latest.json"

    _write_krn(base_dir / "a.krn", "**kern\n4c\n*-\n")
    _write_krn(base_dir / "b.krn", "**kern\n4d\n*-\n")

    payload = prepare_manual_fixes(
        base_dir=str(base_dir),
        working_dir=str(working_dir),
        overrides_dir=str(overrides_dir),
        manifest_out=str(manifest_out),
        refresh=False,
        quiet=True,
    )

    assert payload["total_files"] == 2
    assert payload["changed_count"] == 0
    assert payload["bootstrapped_working_dir"] is True
    assert _read(working_dir / "a.krn") == _read(base_dir / "a.krn")
    assert _read(working_dir / "b.krn") == _read(base_dir / "b.krn")
    assert json.loads(manifest_out.read_text(encoding="utf-8"))["changed_count"] == 0


def test_prepare_preserves_manual_edits_when_refresh_is_false(tmp_path):
    base_dir = tmp_path / "interim" / "3_normalized"
    working_dir = tmp_path / "interim" / "4_manual_fixes"
    manifest_out = tmp_path / "reports" / "latest.json"

    _write_krn(base_dir / "a.krn", "**kern\n4c\n*-\n")
    _write_krn(base_dir / "b.krn", "**kern\n4d\n*-\n")

    prepare_manual_fixes(
        base_dir=str(base_dir),
        working_dir=str(working_dir),
        manifest_out=str(manifest_out),
        quiet=True,
    )
    _write_krn(working_dir / "a.krn", "**kern\n4e\n*-\n")

    payload = prepare_manual_fixes(
        base_dir=str(base_dir),
        working_dir=str(working_dir),
        manifest_out=str(manifest_out),
        refresh=False,
        quiet=True,
    )

    assert _read(working_dir / "a.krn") == "**kern\n4e\n*-\n"
    assert payload["changed_count"] == 1


def test_prepare_refresh_rebuilds_working_dir(tmp_path):
    base_dir = tmp_path / "interim" / "3_normalized"
    working_dir = tmp_path / "interim" / "4_manual_fixes"
    manifest_out = tmp_path / "reports" / "latest.json"

    _write_krn(base_dir / "a.krn", "**kern\n4c\n*-\n")

    prepare_manual_fixes(
        base_dir=str(base_dir),
        working_dir=str(working_dir),
        manifest_out=str(manifest_out),
        quiet=True,
    )
    _write_krn(working_dir / "a.krn", "**kern\n4f\n*-\n")

    payload = prepare_manual_fixes(
        base_dir=str(base_dir),
        working_dir=str(working_dir),
        manifest_out=str(manifest_out),
        refresh=True,
        quiet=True,
    )

    assert payload["refresh"] is True
    assert payload["changed_count"] == 0
    assert _read(working_dir / "a.krn") == _read(base_dir / "a.krn")


def test_prepare_applies_overrides_and_tracks_diff_stats(tmp_path):
    base_dir = tmp_path / "interim" / "3_normalized"
    working_dir = tmp_path / "interim" / "4_manual_fixes"
    overrides_dir = tmp_path / "curation" / "overrides"
    manifest_out = tmp_path / "reports" / "latest.json"

    _write_krn(base_dir / "a.krn", "**kern\n4c\n*-\n")
    _write_krn(base_dir / "b.krn", "**kern\n4d\n*-\n")
    _write_krn(overrides_dir / "b.krn", "**kern\n4g\n4a\n*-\n")

    payload = prepare_manual_fixes(
        base_dir=str(base_dir),
        working_dir=str(working_dir),
        overrides_dir=str(overrides_dir),
        manifest_out=str(manifest_out),
        quiet=True,
    )

    assert payload["applied_override_count"] == 1
    assert payload["changed_count"] == 1
    assert _read(working_dir / "b.krn") == "**kern\n4g\n4a\n*-\n"
    changed = payload["changed_files"][0]
    assert changed["file"] == "b.krn"
    assert changed["added_lines"] > 0
    assert changed["removed_lines"] > 0


def test_prepare_rejects_unknown_override_file(tmp_path):
    base_dir = tmp_path / "interim" / "3_normalized"
    overrides_dir = tmp_path / "curation" / "overrides"

    _write_krn(base_dir / "a.krn", "**kern\n4c\n*-\n")
    _write_krn(overrides_dir / "x.krn", "**kern\n4x\n*-\n")

    with pytest.raises(ValueError, match="Override files not present in base_dir"):
        prepare_manual_fixes(
            base_dir=str(base_dir),
            overrides_dir=str(overrides_dir),
            quiet=True,
        )


def test_prepare_rejects_working_dir_file_set_mismatch(tmp_path):
    base_dir = tmp_path / "interim" / "3_normalized"
    working_dir = tmp_path / "interim" / "4_manual_fixes"

    _write_krn(base_dir / "a.krn", "**kern\n4c\n*-\n")
    _write_krn(working_dir / "a.krn", "**kern\n4c\n*-\n")
    _write_krn(working_dir / "extra.krn", "**kern\n4z\n*-\n")

    with pytest.raises(ValueError, match="working_dir file set mismatch"):
        prepare_manual_fixes(
            base_dir=str(base_dir),
            working_dir=str(working_dir),
            refresh=False,
            quiet=True,
        )


def test_publish_writes_only_changed_files_and_prunes_stale_overrides(tmp_path):
    base_dir = tmp_path / "interim" / "3_normalized"
    working_dir = tmp_path / "interim" / "4_manual_fixes"
    overrides_dir = tmp_path / "curation" / "overrides"
    canonical_manifest_out = tmp_path / "curation" / "overrides_manifest.json"

    _write_krn(base_dir / "a.krn", "**kern\n4c\n*-\n")
    _write_krn(base_dir / "b.krn", "**kern\n4d\n*-\n")
    _write_krn(working_dir / "a.krn", "**kern\n4e\n*-\n")
    _write_krn(working_dir / "b.krn", "**kern\n4d\n*-\n")
    _write_krn(overrides_dir / "stale.krn", "**kern\n4z\n*-\n")

    payload = publish_manual_fixes(
        base_dir=str(base_dir),
        working_dir=str(working_dir),
        overrides_dir=str(overrides_dir),
        canonical_manifest_out=str(canonical_manifest_out),
        quiet=True,
    )

    assert payload["changed_count"] == 1
    assert sorted(p.name for p in overrides_dir.glob("*.krn")) == ["a.krn"]
    assert _read(overrides_dir / "a.krn") == "**kern\n4e\n*-\n"

    _write_krn(working_dir / "a.krn", "**kern\n4c\n*-\n")
    payload2 = publish_manual_fixes(
        base_dir=str(base_dir),
        working_dir=str(working_dir),
        overrides_dir=str(overrides_dir),
        canonical_manifest_out=str(canonical_manifest_out),
        quiet=True,
    )

    assert payload2["changed_count"] == 0
    assert list(overrides_dir.glob("*.krn")) == []

