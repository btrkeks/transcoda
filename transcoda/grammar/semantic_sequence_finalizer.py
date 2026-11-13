"""Finalize decoded model-dialect **kern text after generation."""

from __future__ import annotations

from dataclasses import dataclass

from .stateful_kern_logits_processor import SemanticRuleFactory
from .timeline_rule import TimelineRuleFactory, spell_timeline


@dataclass(frozen=True)
class FinalizedKernSequence:
    """Final decoded model text plus truncation diagnostics."""

    text: str
    trimmed_incomplete_tail: bool
    hit_max_length: bool
    saw_eos: bool
    truncated: bool
    stopped_on_padding: bool = False


def finalize_generated_kern_sequence(
    *,
    token_ids: list[int],
    i2w: dict[int, str],
    bos_token_id: int | None,
    eos_token_id: int | None,
    pad_token_id: int | None,
    max_length: int | None,
    rule_factories: tuple[SemanticRuleFactory, ...] = (),
) -> FinalizedKernSequence:
    """Decode token ids and trim untrusted max-length tails.

    Finalization never synthesizes Humdrum terminators. Of ``rule_factories`` it
    only reads whether the timeline rule decoded the text.
    """
    decoded_tokens: list[str] = []
    generated_length = 0
    saw_eos = False
    stopped_on_padding = False

    for token_id in token_ids:
        if pad_token_id is not None and token_id == pad_token_id:
            stopped_on_padding = True
            break
        generated_length += 1
        if bos_token_id is not None and token_id == bos_token_id:
            continue
        if eos_token_id is not None and token_id == eos_token_id:
            saw_eos = True
            break
        decoded_tokens.append(i2w.get(token_id, ""))

    hit_max_length = bool(
        max_length is not None and not saw_eos and generated_length >= int(max_length)
    )
    return finalize_kern_sequence_text(
        text="".join(decoded_tokens),
        saw_eos=saw_eos,
        hit_max_length=hit_max_length,
        stopped_on_padding=stopped_on_padding,
        rule_factories=rule_factories,
    )


def finalize_kern_sequence_text(
    *,
    text: str,
    saw_eos: bool,
    hit_max_length: bool,
    stopped_on_padding: bool = False,
    rule_factories: tuple[SemanticRuleFactory, ...] = (),
) -> FinalizedKernSequence:
    """Finalize decoded model-dialect text without adding terminators.

    A cut-off sequence loses its incomplete last line. Under the timeline rule the
    text then gets the invisible rests its timeline implies, so every spine meets
    Humdrum's ``*-`` terminators at one time.
    """
    should_trim = bool((hit_max_length and not saw_eos) or stopped_on_padding)
    finalized_text, trimmed_incomplete_tail = _trim_incomplete_tail(text, should_trim=should_trim)
    if any(isinstance(f, TimelineRuleFactory) for f in rule_factories):
        finalized_text = spell_timeline(finalized_text)
    return FinalizedKernSequence(
        text=finalized_text,
        trimmed_incomplete_tail=trimmed_incomplete_tail,
        hit_max_length=hit_max_length,
        saw_eos=saw_eos,
        truncated=bool(
            trimmed_incomplete_tail or (hit_max_length and not saw_eos) or stopped_on_padding
        ),
        stopped_on_padding=stopped_on_padding,
    )


def _trim_incomplete_tail(text: str, *, should_trim: bool) -> tuple[str, bool]:
    if not text or not should_trim:
        return text, False

    lines = text.splitlines(keepends=True)
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].rstrip("\r\n") == "":
            continue
        retained_end = index
        for prior_index in range(index - 1, -1, -1):
            prior_line = lines[prior_index].rstrip("\r\n")
            if prior_line == "":
                continue
            if prior_line == "!!LO:LB" or prior_line.startswith("!!LO:LB:g="):
                retained_end = prior_index
            break
        return "".join(lines[:retained_end]), True

    return "", True
