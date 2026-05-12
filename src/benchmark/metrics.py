"""XML-based metric helpers for benchmark evaluation."""

from __future__ import annotations

from src.evaluation.omr_ned import OMRNEDResult, compute_omr_ned_from_musicxml
from src.evaluation.tedn import compute_tedn_from_musicxml as compute_tedn_score_from_musicxml


def compute_tedn_from_musicxml(predicted_xml: str, gold_xml: str) -> float:
    """Compute TEDn percentage on a pair of MusicXML documents."""
    return compute_tedn_score_from_musicxml(predicted_xml, gold_xml).tedn_percent


def compute_omr_ned_xml(predicted_xml: str, gold_xml: str) -> OMRNEDResult:
    """Compute OMR-NED for MusicXML pairs."""
    return compute_omr_ned_from_musicxml(predicted_xml, gold_xml)
