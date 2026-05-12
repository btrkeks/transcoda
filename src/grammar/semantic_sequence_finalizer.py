"""Finalize decoded **kern text conservatively after constrained generation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import cast

from .stateful_kern_logits_processor import KernSemanticRule, SemanticRuleFactory


@dataclass(frozen=True)
class FinalizedKernSequence:
    """Final decoded **kern body text plus truncation diagnostics."""

    text: str
    trimmed_incomplete_tail: bool
    appended_terminator: bool
    hit_max_length: bool
    saw_eos: bool
    truncated: bool


@dataclass
class _ReplaySnapshot:
    text: str
    rules: list[KernSemanticRule]
    last_field_count: int | None


def finalize_generated_kern_sequence(
    *,
    token_ids: list[int],
    i2w: dict[int, str],
    bos_token_id: int | None,
    eos_token_id: int | None,
    pad_token_id: int | None,
    max_length: int | None,
    rule_factories: tuple[SemanticRuleFactory, ...],
) -> FinalizedKernSequence:
    """Decode token ids and conservatively repair incomplete trailing structure."""
    decoded_tokens: list[str] = []
    generated_length = 0
    saw_eos = False

    for token_id in token_ids:
        if pad_token_id is not None and token_id == pad_token_id:
            break
        generated_length += 1
        if bos_token_id is not None and token_id == bos_token_id:
            continue
        if eos_token_id is not None and token_id == eos_token_id:
            saw_eos = True
            break
        decoded_tokens.append(i2w.get(token_id, ""))

    hit_max_length = bool(max_length is not None and not saw_eos and generated_length >= int(max_length))
    return finalize_kern_sequence_text(
        text="".join(decoded_tokens),
        saw_eos=saw_eos,
        hit_max_length=hit_max_length,
        rule_factories=rule_factories,
    )


def finalize_kern_sequence_text(
    *,
    text: str,
    saw_eos: bool,
    hit_max_length: bool,
    rule_factories: tuple[SemanticRuleFactory, ...],
) -> FinalizedKernSequence:
    """Finalize decoded **kern body text using the configured semantic rule stack."""
    if not text:
        return FinalizedKernSequence(
            text=text,
            trimmed_incomplete_tail=False,
            appended_terminator=False,
            hit_max_length=hit_max_length,
            saw_eos=saw_eos,
            truncated=bool(hit_max_length and not saw_eos),
        )

    rules = [factory() for factory in rule_factories]
    initial_fields = _parse_fields(text.split("\n", 1)[0]) if text else None
    if initial_fields is not None and not all(field.startswith("**") for field in initial_fields):
        synthetic_header = tuple(["**kern"] * len(initial_fields))
        if _can_close_line(rules, synthetic_header):
            _apply_closed_line(rules, synthetic_header)
    last_valid = _ReplaySnapshot(text="", rules=deepcopy(rules), last_field_count=None)
    offset = 0
    invalid_tail = False

    for line in text.splitlines(keepends=True):
        if not line.endswith("\n"):
            break
        raw_line = line[:-1]
        fields = _parse_fields(raw_line)
        if fields is None or not _can_close_line(rules, fields):
            invalid_tail = True
            break
        _apply_closed_line(rules, fields)
        offset += len(line)
        last_valid = _ReplaySnapshot(
            text=text[:offset],
            rules=deepcopy(rules),
            last_field_count=len(fields),
        )

    tail = text[offset:]
    trimmed_incomplete_tail = invalid_tail
    appended_terminator = False

    if not invalid_tail and tail:
        fields = _parse_fields(tail)
        if fields is not None:
            if _can_end_sequence(rules, fields):
                return FinalizedKernSequence(
                    text=text,
                    trimmed_incomplete_tail=False,
                    appended_terminator=False,
                    hit_max_length=hit_max_length,
                    saw_eos=saw_eos,
                    truncated=bool(hit_max_length and not saw_eos),
                )

            if (not saw_eos) and _can_close_line(rules, fields):
                rules_after_tail = deepcopy(rules)
                _apply_closed_line(rules_after_tail, fields)
                if _can_append_terminator(rules_after_tail, len(fields)):
                    terminator = _build_terminator_line(rules_after_tail, len(fields))
                    return FinalizedKernSequence(
                        text=f"{text}\n{terminator}",
                        trimmed_incomplete_tail=False,
                        appended_terminator=True,
                        hit_max_length=hit_max_length,
                        saw_eos=saw_eos,
                        truncated=bool(hit_max_length and not saw_eos),
                    )
        trimmed_incomplete_tail = True

    if not tail and _is_terminated_state(rules):
        return FinalizedKernSequence(
            text=text,
            trimmed_incomplete_tail=False,
            appended_terminator=False,
            hit_max_length=hit_max_length,
            saw_eos=saw_eos,
            truncated=bool(hit_max_length and not saw_eos),
        )

    finalized_text = last_valid.text
    if finalized_text and _can_append_terminator(last_valid.rules, last_valid.last_field_count):
        terminator = _build_terminator_line(last_valid.rules, last_valid.last_field_count)
        if finalized_text.endswith("\n"):
            finalized_text = f"{finalized_text}{terminator}"
        else:
            finalized_text = f"{finalized_text}\n{terminator}"
        appended_terminator = True

    return FinalizedKernSequence(
        text=finalized_text,
        trimmed_incomplete_tail=trimmed_incomplete_tail,
        appended_terminator=appended_terminator,
        hit_max_length=hit_max_length,
        saw_eos=saw_eos,
        truncated=bool(trimmed_incomplete_tail or (hit_max_length and not saw_eos)),
    )


def _parse_fields(line: str) -> tuple[str, ...] | None:
    if not line:
        return None
    fields = tuple(line.split("\t"))
    if any(field == "" for field in fields):
        return None
    return fields


def _can_close_line(rules: list[KernSemanticRule], fields: tuple[str, ...]) -> bool:
    try:
        return all(rule.can_close_line(fields) for rule in rules)
    except ValueError:
        return False


def _can_end_sequence(rules: list[KernSemanticRule], fields: tuple[str, ...]) -> bool:
    try:
        return all(rule.can_end_sequence(fields) for rule in rules)
    except ValueError:
        return False


def _apply_closed_line(rules: list[KernSemanticRule], fields: tuple[str, ...]) -> None:
    for rule in rules:
        rule.on_line_closed(fields)


def _is_terminated_state(rules: list[KernSemanticRule]) -> bool:
    return any(rule.terminated for rule in rules)


def _build_terminator_line(rules: list[KernSemanticRule], fallback_width: int | None) -> str:
    active_spines = _resolve_active_spines(rules, fallback_width)
    if active_spines is None or active_spines <= 0:
        raise ValueError("Cannot infer active spine count for terminator synthesis")
    return "\t".join(["*-"] * active_spines)


def _can_append_terminator(rules: list[KernSemanticRule], fallback_width: int | None) -> bool:
    active_spines = _resolve_active_spines(rules, fallback_width)
    if active_spines is None or active_spines <= 0:
        return False

    terminator_fields = tuple(["*-"] * active_spines)
    return _can_close_line(rules, terminator_fields)


def _resolve_active_spines(
    rules: list[KernSemanticRule],
    fallback_width: int | None,
) -> int | None:
    for rule in rules:
        active_spines = getattr(rule, "active_spines", None)
        if isinstance(active_spines, int):
            return active_spines
    return cast(int | None, fallback_width)
