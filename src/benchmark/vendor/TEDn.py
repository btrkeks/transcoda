# Vendored from https://github.com/ufal/olimpic-icdar24 under the MIT License.
# Copyright (c) 2024 Jiří Mayer
# SPDX-License-Identifier: MIT
#
# Local adaptations:
# - Use xml.etree throughout.
# - Fall back to RapidFuzz when python-Levenshtein is unavailable.
# - Include the XML-vs-XML wrapper helpers used by this benchmark runner.

from __future__ import annotations

import copy
import time
import xml.etree.ElementTree as ET
from fractions import Fraction
from typing import List, Literal, Optional, TextIO, Tuple

import zss

try:
    import Levenshtein
except ImportError:  # pragma: no cover - exercised only when dependency is absent.
    from rapidfuzz.distance import Levenshtein as _RapidFuzzLevenshtein

    class _LevenshteinShim:
        @staticmethod
        def distance(left: str, right: str) -> int:
            return int(_RapidFuzzLevenshtein.distance(left, right))

    Levenshtein = _LevenshteinShim()


def TEDn(predicted_element: ET.Element, gold_element: ET.Element) -> "TEDnResult":
    """Compute TEDn on two MusicXML trees."""
    assert gold_element.tag in ["part", "score-partwise"], "Unsupported input element type"
    assert gold_element.tag == predicted_element.tag, "Both arguments must be of the same element type"

    start_time = time.time()
    metric_class = Xml4ZSS_Levenshtein

    if metric_class is Xml4ZSS_Levenshtein:
        coder = NoteContentCoder()
        gold_element = encode_notes(copy.deepcopy(gold_element), coder)
        predicted_element = encode_notes(copy.deepcopy(predicted_element), coder)

    edit_cost = zss.distance(
        predicted_element,
        gold_element,
        get_children=metric_class.get_children,
        update_cost=metric_class.update,
        insert_cost=metric_class.insert,
        remove_cost=metric_class.remove,
    )

    gold_cost = zss.distance(
        ET.Element(predicted_element.tag),
        gold_element,
        get_children=metric_class.get_children,
        update_cost=metric_class.update,
        insert_cost=metric_class.insert,
        remove_cost=metric_class.remove,
    )

    end_time = time.time()
    return TEDnResult(
        gold_cost=gold_cost,
        edit_cost=edit_cost,
        evaluation_time_seconds=(end_time - start_time),
    )


class TEDnResult:
    def __init__(
        self,
        gold_cost: int,
        edit_cost: int,
        evaluation_time_seconds: float,
    ) -> None:
        self.gold_cost = int(gold_cost)
        self.edit_cost = int(edit_cost)
        self.evaluation_time_seconds = evaluation_time_seconds

    @property
    def normalized_edit_cost(self) -> float:
        if self.gold_cost == 0:
            return 0.0
        return float(self.edit_cost) / float(self.gold_cost)

    def __repr__(self) -> str:
        return (
            "TEDnResult("
            f"gold_cost={self.gold_cost}, "
            f"edit_cost={self.edit_cost}, "
            f"evaluation_time_seconds={round(self.evaluation_time_seconds, 2)})"
        )


class ZSSMetricClass:
    @staticmethod
    def get_children(e):
        raise NotImplementedError()

    @staticmethod
    def update(e, f):
        raise NotImplementedError()

    @staticmethod
    def insert(e):
        raise NotImplementedError()

    @staticmethod
    def remove(e):
        raise NotImplementedError()


class Xml4ZSS(ZSSMetricClass):
    @staticmethod
    def get_children(e: ET.Element) -> List[ET.Element]:
        return list(e)

    @staticmethod
    def update(e: ET.Element, f: ET.Element) -> int:
        tag_equal = e.tag == f.tag

        text_equal = False
        if e.text is None:
            text_equal = f.text is None
        elif f.text is not None and e.text.strip() == f.text.strip():
            text_equal = True

        return 0 if (tag_equal and text_equal) else 1

    @staticmethod
    def insert(e: ET.Element) -> int:
        return 1

    @staticmethod
    def remove(e: ET.Element) -> int:
        return 1


class Xml4ZSS_Filtered(Xml4ZSS):
    filtered_out = {
        "*": {
            "footnote",
            "level",
        },
        "score-partwise": {
            "work",
            "movement-number",
            "movement-title",
            "identification",
            "defaults",
            "credit",
            "part-list",
        },
        "measure": {
            "print",
            "sound",
            "listening",
        },
        "note": {
            "duration",
            "listen",
            "play",
            "tie",
        },
    }

    @staticmethod
    def get_children(e: ET.Element) -> List[ET.Element]:
        children = []
        for child in e:
            if child.tag in Xml4ZSS_Filtered.filtered_out["*"]:
                continue
            if e.tag in Xml4ZSS_Filtered.filtered_out:
                if child.tag in Xml4ZSS_Filtered.filtered_out[e.tag]:
                    continue
            children.append(child)
        return children


class Xml4ZSS_Levenshtein(Xml4ZSS_Filtered):
    @staticmethod
    def update(e: ET.Element, f: ET.Element) -> int:
        tag_change_cost = 0 if e.tag == f.tag else 1

        text_edit_cost = 0
        if (e.tag == "note") or (f.tag == "note"):
            if e.text is None:
                text_edit_cost = len(f.text or "")
            elif f.text is None:
                text_edit_cost = len(e.text)
            else:
                text_edit_cost = int(Levenshtein.distance(e.text, f.text))
        elif (e.text or "").strip() != (f.text or "").strip():
            text_edit_cost += 1

        return tag_change_cost + text_edit_cost

    @staticmethod
    def insert(e: ET.Element) -> int:
        if e.tag == "note":
            return 1 + len(e.text or "")
        return 1

    @staticmethod
    def remove(e: ET.Element) -> int:
        return 1


class PitchCoder:
    code_letters = (
        "0123456789"
        + "abcdefghijklmnopqrstuvwxyz"
        + "ABCDEFGHIJKLMNOPQ"
        + "STUVWXYZ"
        + ".,!?:;/\\|-_=+><[]{}()*&^%$#@~`"
        + "áčďéěíňóřšťúůýžäëïöüÿ"
        + "ÁČĎÉĚÍŇÓŘŠŤÚŮÝŽÄËÏÖÜŸ"
        + "бгґдєйжилпфцчшщьюя"
        + "БГҐДЄЙЖИЛПФЦЧШЩЬЮЯ"
    )

    def __init__(self) -> None:
        self.n_pitches = 0
        self.codes: dict[tuple[str, str, str], str] = {}

    def pitch2pitch_index(self, pitch_element: ET.Element) -> Tuple[str, str, str]:
        pitch_values = {"step": "C", "alter": "0", "octave": "0"}
        for e in pitch_element:
            pitch_values[e.tag] = e.text or pitch_values[e.tag]
        return tuple(pitch_values.values())

    def encode(self, pitch_element: ET.Element) -> str:
        pitch_index = self.pitch2pitch_index(pitch_element)
        if pitch_index not in self.codes:
            if self.n_pitches == len(PitchCoder.code_letters):
                raise ValueError(
                    f"Too many distinct entities to encode, only {len(PitchCoder.code_letters)} codes available."
                )
            self.codes[pitch_index] = PitchCoder.code_letters[self.n_pitches]
            self.n_pitches += 1
        return self.codes[pitch_index]


class NoteContentCoder:
    note_type_table = {
        "1024th": "0",
        "512th": "0",
        "256th": "0",
        "128th": "0",
        "64th": "1",
        "32nd": "2",
        "16th": "3",
        "eighth": "4",
        "quarter": "5",
        "half": "6",
        "whole": "7",
        "breve": "8",
        "long": "9",
        "maxima": "9",
    }

    stem_type_table = {
        "up": "U",
        "down": "D",
        "none": "N",
        None: "-",
    }

    REST_CODE = "R"
    MISSING_PITCH_CODE = "~"
    ENCODES_TAGS = ["pitch", "voice", "type", "stem"]

    def __init__(self) -> None:
        self.pitch_coder = PitchCoder()

    def encode(self, note: ET.Element) -> str:
        is_rest = note.find("rest") is not None
        pitch_element = note.find("pitch")
        voice_element = note.find("voice")
        type_element = note.find("type")
        stem_element = note.find("stem")

        if is_rest:
            pitch_code = NoteContentCoder.REST_CODE
        elif pitch_element is not None:
            pitch_code = self.pitch_coder.encode(pitch_element)
        else:
            pitch_code = NoteContentCoder.MISSING_PITCH_CODE

        voice_code = voice_element.text if voice_element is not None and voice_element.text else "1"
        if type_element is not None and type_element.text is not None:
            type_code = NoteContentCoder.note_type_table[type_element.text]
        else:
            type_code = NoteContentCoder.note_type_table["whole"]
        stem_key = stem_element.text if stem_element is not None else None
        stem_code = NoteContentCoder.stem_type_table[stem_key]
        return f"{pitch_code}{voice_code}{type_code}{stem_code}"


def encode_notes(root: ET.Element, coder: NoteContentCoder) -> ET.Element:
    for note in root.iter("note"):
        code = coder.encode(note)
        note.text = code
        to_remove = [e for e in note if e.tag in NoteContentCoder.ENCODES_TAGS]
        for element in to_remove:
            note.remove(element)
    return root


def actual_durations_to_fractional(part: ET.Element) -> None:
    assert part.tag == "part"
    if len(part) == 0:
        return

    current_divisions: Optional[int] = None

    def visit_duration(duration_element: ET.Element | None) -> None:
        nonlocal current_divisions
        if duration_element is None:
            return
        assert current_divisions is not None
        duration = Fraction(duration_element.text)
        duration = duration / current_divisions
        duration_element.text = str(duration)

    def visit_notelike(notelike_element: ET.Element) -> None:
        visit_duration(notelike_element.find("duration"))

    def visit_attributes(attributes_element: ET.Element) -> None:
        nonlocal current_divisions
        divisions_element = attributes_element.find("divisions")
        if divisions_element is None:
            return
        current_divisions = int(divisions_element.text)
        attributes_element.remove(divisions_element)

    for measure in part:
        for element in measure:
            if element.tag in {"note", "forward", "backup"}:
                visit_notelike(element)
            elif element.tag == "attributes":
                visit_attributes(element)


class Pruner:
    def __init__(
        self,
        prune_durations: bool = False,
        prune_measure_attributes: bool = False,
        prune_prints: bool = True,
        prune_directions: bool = True,
        prune_barlines: bool = True,
        prune_harmony: bool = True,
        prune_slur_numbering: bool = True,
    ) -> None:
        self.prune_durations = prune_durations
        self.prune_measure_attributes = prune_measure_attributes
        self.prune_slur_numbering = prune_slur_numbering

        self.measure_prune_tags = {
            "sound",
            "listening",
            "figured-bass",
            "bookmark",
            "grouping",
        }
        if prune_prints:
            self.measure_prune_tags.add("print")
        if prune_directions:
            self.measure_prune_tags.add("direction")
        if prune_barlines:
            self.measure_prune_tags.add("barline")
        if prune_harmony:
            self.measure_prune_tags.add("harmony")

        self.note_prune_tags = {
            "cue",
            "lyric",
            "instrument",
            "play",
            "listen",
            "notehead-text",
            "notehead",
        }
        if prune_durations:
            self.note_prune_tags.add("duration")

    def process_part(self, part: ET.Element) -> None:
        assert part.tag == "part"
        for measure in part:
            self.process_measure(measure)

    def process_measure(self, measure: ET.Element) -> None:
        assert measure.tag == "measure"

        if self.prune_measure_attributes:
            measure.attrib.pop("number", None)
            measure.attrib.pop("implicit", None)
            measure.attrib.pop("width", None)

        prune_children(measure, self.measure_prune_tags)

        for element in measure:
            if element.tag == "note":
                self.process_note(element)
            elif element.tag == "backup":
                self.process_backup(element)
            elif element.tag == "forward":
                self.process_forward(element)
            elif element.tag == "attributes":
                self.process_attributes(element)

    def process_forward(self, forward: ET.Element) -> None:
        if self.prune_durations:
            prune_children(forward, {"duration"})

    def process_backup(self, backup: ET.Element) -> None:
        if self.prune_durations:
            prune_children(backup, {"duration"})

    def process_attributes(self, attributes: ET.Element) -> None:
        prune_children(
            attributes,
            {
                "instruments",
                "staff-details",
                "measure-style",
                "directive",
                "part-symbol",
            },
        )

        if self.prune_durations:
            divisions_element = attributes.find("divisions")
            if divisions_element is not None:
                attributes.remove(divisions_element)

        time_element = attributes.find("time")
        if time_element is not None:
            time_element.attrib.pop("symbol", None)

        for clef_element in attributes.findall("clef"):
            clef_element.attrib.pop("after-barline", None)

    def process_note(self, note: ET.Element) -> None:
        assert note.tag == "note"
        note.attrib.pop("default-x", None)
        note.attrib.pop("default-y", None)
        note.attrib.pop("dynamics", None)
        prune_children(note, self.note_prune_tags)

        rest_element = note.find("rest")
        if rest_element is not None and len(rest_element) > 0:
            rest_element[:] = []

        type_element = note.find("type")
        if type_element is not None:
            type_element.attrib.clear()

        accidental_element = note.find("accidental")
        if accidental_element is not None:
            accidental_element.attrib.clear()

        time_modification_element = note.find("time-modification")
        if time_modification_element is not None:
            allow_children(time_modification_element, {"actual-notes", "normal-notes"})

        notations_element = note.find("notations")
        if notations_element is not None:
            self.process_notations(notations_element)
            if len(notations_element) == 0:
                note.remove(notations_element)

    def process_notations(self, notations: ET.Element) -> None:
        allow_children(
            notations,
            {
                "tied",
                "slur",
                "tuplet",
                "ornaments",
                "articulations",
                "fermata",
                "arpeggiate",
            },
        )
        for element in list(notations):
            if element.tag == "slur":
                element.attrib.pop("placement", None)
                element.attrib.pop("line-type", None)
                if self.prune_slur_numbering:
                    element.attrib.pop("number", None)
            elif element.tag == "tuplet":
                element.attrib.pop("bracket", None)
                element.attrib.pop("show-number", None)
                element[:] = []
            elif element.tag in {"fermata", "arpeggiate"}:
                element.attrib.clear()
            elif element.tag == "articulations":
                self.process_articulations(element)
                if len(element) == 0:
                    notations.remove(element)
            elif element.tag == "ornaments":
                self.process_ornaments(element)
                if len(element) == 0:
                    notations.remove(element)

    def process_articulations(self, articulations: ET.Element) -> None:
        allow_children(articulations, {"staccato", "accent", "strong-accent", "tenuto"})
        for element in articulations:
            element.clear()

    def process_ornaments(self, ornaments: ET.Element) -> None:
        allow_children(ornaments, {"tremolo", "trill-mark"})
        for element in list(ornaments):
            if element.tag == "trill-mark":
                element.clear()


def prune_children(element: ET.Element, tags: set[str]) -> None:
    children_to_remove = [child for child in element if child.tag in tags]
    for child in children_to_remove:
        element.remove(child)


def allow_children(element: ET.Element, tags: set[str]) -> None:
    children_to_remove = [child for child in element if child.tag not in tags]
    for child in children_to_remove:
        element.remove(child)


def TEDn_xml_xml(
    predicted_xml: str,
    gold_musicxml: str,
    flavor: Literal["full", "lmx"] = "lmx",
    canonicalize: bool = True,
    debug: bool = False,
    errout: Optional[TextIO] = None,
) -> TEDnResult:
    """String-based XML-vs-XML wrapper around TEDn."""
    del debug, errout
    assert flavor in {"full", "lmx"}

    if canonicalize:
        gold_musicxml = ET.canonicalize(gold_musicxml, strip_text=True)
        predicted_xml = ET.canonicalize(predicted_xml, strip_text=True)

    gold_score = ET.fromstring(gold_musicxml)
    pred_score = ET.fromstring(predicted_xml)
    assert gold_score.tag == "score-partwise"
    assert pred_score.tag == "score-partwise"

    gold_parts = gold_score.findall("part")
    pred_parts = pred_score.findall("part")[:2]

    for gold_child in list(gold_score):
        if gold_child.tag != "part":
            gold_score.remove(gold_child)

    count = 0
    for pred_child in list(pred_score):
        if pred_child.tag != "part" or count >= 2:
            pred_score.remove(pred_child)
        else:
            count += 1

    if len(gold_parts) < len(pred_parts):
        pred_parts = pred_parts[: len(gold_parts)]

    for gold_part in gold_parts:
        actual_durations_to_fractional(gold_part)
    for pred_part in pred_parts:
        actual_durations_to_fractional(pred_part)

    if flavor == "lmx":
        pruner = Pruner(
            prune_durations=False,
            prune_measure_attributes=False,
            prune_prints=True,
            prune_slur_numbering=True,
            prune_directions=True,
            prune_barlines=True,
            prune_harmony=True,
        )
        for predicted_part in pred_parts:
            pruner.process_part(predicted_part)
        for gold_part in gold_parts:
            pruner.process_part(gold_part)

    if len(pred_parts) == len(gold_parts):
        scores = [TEDn(pred_part, gold_part) for pred_part, gold_part in zip(pred_parts, gold_parts)]
        return TEDnResult(
            sum(score.gold_cost for score in scores),
            sum(score.edit_cost for score in scores),
            sum(score.evaluation_time_seconds for score in scores),
        )

    return TEDn(pred_score, gold_score)
