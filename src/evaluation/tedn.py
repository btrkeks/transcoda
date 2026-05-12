"""TEDn metric helpers for MusicXML comparisons."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Literal

_XML_NODE_LIMIT = 6000


@dataclass(frozen=True)
class TEDnScore:
    """TEDn result for one MusicXML prediction/ground-truth pair."""

    tedn_percent: float
    normalized_edit_cost: float
    edit_cost: int
    gold_cost: int
    evaluation_time_seconds: float


def truncate_xml_tree(xml_text: str, *, limit: int = _XML_NODE_LIMIT) -> str:
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


def compute_tedn_from_musicxml(
    predicted_xml: str,
    gold_xml: str,
    *,
    flavor: Literal["lmx", "full"] = "lmx",
) -> TEDnScore:
    """Compute TEDn on a pair of MusicXML documents.

    The default ``lmx`` flavor matches LEGATO's XML-vs-XML evaluator and
    prunes MusicXML features that LEGATO's LMX representation does not cover.
    """
    try:
        from src.benchmark.vendor.TEDn import TEDn_xml_xml
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment.
        raise RuntimeError(
            "TEDn dependencies are missing. Install the project environment with `uv sync` "
            "so `zss` and `rapidfuzz` are available."
        ) from exc

    pred_cut = truncate_xml_tree(predicted_xml)
    gold_cut = truncate_xml_tree(gold_xml)
    result = TEDn_xml_xml(pred_cut, gold_cut, flavor=flavor)
    normalized = 0.0 if result.gold_cost == 0 else result.edit_cost / result.gold_cost
    return TEDnScore(
        tedn_percent=normalized * 100.0,
        normalized_edit_cost=normalized,
        edit_cost=result.edit_cost,
        gold_cost=result.gold_cost,
        evaluation_time_seconds=result.evaluation_time_seconds,
    )
