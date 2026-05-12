from __future__ import annotations

from pathlib import Path

import pytest

from notebooks.polish_manual_fixes import (
    ManualFixPaths,
    build_sample_refs,
    check_manual_stage_readiness,
    parse_sample_filename,
    render_transcription_isolated,
    save_transcription,
)


def test_parse_sample_filename_valid() -> None:
    split, idx = parse_sample_filename("test_000123.krn")
    assert split == "test"
    assert idx == 123


def test_parse_sample_filename_invalid() -> None:
    with pytest.raises(ValueError):
        parse_sample_filename("bad-name.krn")


def test_build_sample_refs_sorted(tmp_path: Path) -> None:
    (tmp_path / "val_000002.krn").write_text("x", encoding="utf-8")
    (tmp_path / "train_000001.krn").write_text("x", encoding="utf-8")
    (tmp_path / "README.txt").write_text("ignore", encoding="utf-8")

    refs = build_sample_refs(tmp_path)

    assert [ref.filename for ref in refs] == ["train_000001.krn", "val_000002.krn"]
    assert refs[0].split == "train"
    assert refs[1].index == 2


def test_readiness_missing_manual_dir(tmp_path: Path) -> None:
    normalized_dir = tmp_path / "3_normalized"
    normalized_dir.mkdir()
    (normalized_dir / "test_000000.krn").write_text("x", encoding="utf-8")

    readiness = check_manual_stage_readiness(
        ManualFixPaths(normalized_dir=normalized_dir, manual_dir=tmp_path / "4_manual_fixes")
    )

    assert readiness.ready is False
    assert "missing" in readiness.message.lower()
    assert "rsync -a" in readiness.message


def test_readiness_mismatch_files(tmp_path: Path) -> None:
    normalized_dir = tmp_path / "3_normalized"
    manual_dir = tmp_path / "4_manual_fixes"
    normalized_dir.mkdir()
    manual_dir.mkdir()

    (normalized_dir / "test_000000.krn").write_text("x", encoding="utf-8")
    (normalized_dir / "test_000001.krn").write_text("x", encoding="utf-8")
    (manual_dir / "test_000000.krn").write_text("x", encoding="utf-8")
    (manual_dir / "extra_000999.krn").write_text("x", encoding="utf-8")

    readiness = check_manual_stage_readiness(
        ManualFixPaths(normalized_dir=normalized_dir, manual_dir=manual_dir)
    )

    assert readiness.ready is True
    assert readiness.missing_in_manual == ("test_000001.krn",)
    assert readiness.extra_in_manual == ("extra_000999.krn",)
    assert "proceed" in readiness.message.lower()


def test_save_transcription_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "sample.krn"
    path.write_text("old", encoding="utf-8")

    save_transcription(path, "new value")

    assert path.read_text(encoding="utf-8") == "new value"


def test_render_transcription_isolated_success() -> None:
    pytest.importorskip("verovio")
    pytest.importorskip("cairosvg")

    transcription = "*clefG2\n*M4/4\n=1\n4c\n=2\n4d\n==\n*-\n"
    result = render_transcription_isolated(transcription, timeout_s=10.0)

    assert result.ok is True
    assert result.page_count >= 1
    assert len(result.page_png_bytes) == result.page_count
    assert all(isinstance(page, bytes) and len(page) > 0 for page in result.page_png_bytes)
