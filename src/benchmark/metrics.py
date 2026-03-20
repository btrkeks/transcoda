"""XML-based metric helpers for benchmark evaluation."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from src.evaluation.omr_ned import OMRNEDResult, compute_omr_ned_from_musicxml

_XML_NODE_LIMIT = 6000


def _truncate_xml_tree(xml_text: str, *, limit: int = _XML_NODE_LIMIT) -> str:
    """Trim oversized MusicXML trees in the same style as LEGATO's evaluator."""
    root = ET.fromstring(xml_text)
    num_nodes = sum(1 for _ in root.iter())

    while num_nodes > limit:
        parts = root.findall("part")
        if not parts:
            break
        changed = False
        for part in parts:
            if len(part) > 0:
                part.remove(part[-1])
                changed = True
        if not changed:
            break
        num_nodes = sum(1 for _ in root.iter())

    return ET.tostring(root, encoding="unicode")


def compute_tedn_from_musicxml(predicted_xml: str, gold_xml: str) -> float:
    """Compute TEDn percentage on a pair of MusicXML documents."""
    try:
        from src.benchmark.vendor.TEDn import TEDn_xml_xml
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment.
        raise RuntimeError(
            "TEDn dependencies are missing. Install benchmark extras with `uv sync` so `zss` is available."
        ) from exc

    pred_cut = _truncate_xml_tree(predicted_xml)
    gold_cut = _truncate_xml_tree(gold_xml)
    result = TEDn_xml_xml(pred_cut, gold_cut, flavor="lmx")
    if result.gold_cost == 0:
        return 0.0
    return float(result.edit_cost) / float(result.gold_cost) * 100.0


def compute_omr_ned_xml(predicted_xml: str, gold_xml: str) -> OMRNEDResult:
    """Compute OMR-NED for MusicXML pairs."""
    return compute_omr_ned_from_musicxml(predicted_xml, gold_xml)
