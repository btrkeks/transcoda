from __future__ import annotations

import pytest

from src.grammar.spine_decoder_state import SpineDecoderState, SpineDecoderStateError, SpineLineKind


def _accept_line(state: SpineDecoderState, fields: list[str]) -> None:
    for index, field in enumerate(fields):
        state.accept_token_text(field)
        if index != len(fields) - 1:
            state.accept_token_text("\t")
    state.accept_token_text("\n")


def test_first_line_establishes_initial_spine_count():
    state = SpineDecoderState()

    _accept_line(state, ["4c", "4e", "4g"])

    assert state.active_spines == 3
    assert state.current_line_kind == SpineLineKind.UNKNOWN


def test_width_is_preserved_across_ordinary_lines():
    state = SpineDecoderState()

    _accept_line(state, ["4c", "4e"])
    _accept_line(state, ["4d", "4f"])

    assert state.active_spines == 2
    assert not state.terminated


def test_split_line_increases_width():
    state = SpineDecoderState()

    _accept_line(state, ["4c", "4e"])
    _accept_line(state, ["*^", "*"])

    assert state.active_spines == 3


def test_adjacent_merge_group_reduces_width():
    state = SpineDecoderState()

    _accept_line(state, ["4c", "4e", "4g"])
    _accept_line(state, ["*v", "*v", "*"])

    assert state.active_spines == 2


def test_orphan_merge_is_rejected():
    state = SpineDecoderState()
    _accept_line(state, ["4c", "4e"])

    state.accept_token_text("*")
    state.accept_token_text("\t")
    state.accept_token_text("*v")

    with pytest.raises(SpineDecoderStateError, match="Invalid merge operation"):
        state.accept_token_text("\n")


def test_terminating_line_sets_terminated_even_when_star_dash_is_split():
    state = SpineDecoderState()
    _accept_line(state, ["4c", "4e"])

    state.accept_token_text("*")
    state.accept_token_text("-")
    state.accept_token_text("\t")
    state.accept_token_text("*")
    state.accept_token_text("-")
    state.accept_token_text("\n")

    assert state.active_spines == 0
    assert state.terminated
