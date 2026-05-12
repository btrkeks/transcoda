from __future__ import annotations

from pathlib import Path

from src.benchmark.conversion import (
    build_sample_key,
    ensure_humdrum_document,
    resolve_abc2xml_command,
)


def test_build_sample_key_uses_source() -> None:
    assert build_sample_key(12, "test_000012.krn") == "test_000012.krn"


def test_build_sample_key_falls_back_to_index() -> None:
    assert build_sample_key(12, None) == "000012"


def test_ensure_humdrum_document_adds_header_and_terminator() -> None:
    result = ensure_humdrum_document("4c\t4e\n4d\t4f")
    assert result.startswith("**kern\t**kern\n")
    assert result.endswith("*-\t*-")


def test_ensure_humdrum_document_preserves_existing_header() -> None:
    source = "**kern\n4c\n*-"
    assert ensure_humdrum_document(source) == source


def test_resolve_abc2xml_command_prefers_system(monkeypatch) -> None:
    monkeypatch.setattr("src.benchmark.conversion.shutil.which", lambda _name: "/usr/bin/abc2xml")
    assert resolve_abc2xml_command("abc2xml") == ["/usr/bin/abc2xml"]


def test_resolve_abc2xml_command_falls_back_to_local(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    local_tool = tmp_path / "tools" / "abc2xml" / "abc2xml.py"
    local_tool.parent.mkdir(parents=True)
    local_tool.write_text("#!/usr/bin/env python\n")
    monkeypatch.setattr("src.benchmark.conversion.shutil.which", lambda _name: None)
    assert resolve_abc2xml_command("abc2xml") == ["python", str(Path("tools/abc2xml/abc2xml.py"))]
