#!/usr/bin/env python3
"""Extract the score MusicXML member from a compressed MXL archive."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

CONTAINER_PATH = "META-INF/container.xml"


def _is_meta_inf_xml(member: str) -> bool:
    return member.lower().startswith("meta-inf/") and member.lower().endswith(".xml")


def _first_non_meta_inf_xml(archive: zipfile.ZipFile) -> str | None:
    for member in archive.namelist():
        if member.lower().endswith(".xml") and not _is_meta_inf_xml(member):
            return member
    return None


def _container_rootfile(archive: zipfile.ZipFile) -> str | None:
    try:
        container_xml = archive.read(CONTAINER_PATH)
    except KeyError:
        return None

    try:
        root = ElementTree.fromstring(container_xml)
    except ElementTree.ParseError:
        return None

    names = set(archive.namelist())
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "rootfile":
            continue
        full_path = element.attrib.get("full-path")
        if full_path in names:
            return full_path
    return None


def score_xml_member(mxl_path: Path) -> str | None:
    """Return the score XML member, preferring META-INF/container.xml rootfile."""
    with zipfile.ZipFile(mxl_path, "r") as archive:
        return _container_rootfile(archive) or _first_non_meta_inf_xml(archive)


def extract_score_xml(mxl_path: Path, output_path: Path) -> None:
    with zipfile.ZipFile(mxl_path, "r") as archive:
        member = _container_rootfile(archive) or _first_non_meta_inf_xml(archive)
        if member is None:
            raise ValueError(f"no score XML member found in {mxl_path}")
        output_path.write_bytes(archive.read(member))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mxl_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args(argv)

    try:
        extract_score_xml(args.mxl_path, args.output_path)
    except Exception as exc:
        print(f"{args.mxl_path}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
