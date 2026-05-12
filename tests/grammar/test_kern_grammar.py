"""Tests for kern.gbnf grammar acceptance/rejection of **kern strings."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import xgrammar as xgr
    from xgrammar.testing import _is_grammar_accept_string

    _HAS_XGRAMMAR = True
except ImportError:
    _HAS_XGRAMMAR = False

pytestmark = pytest.mark.skipif(
    not _HAS_XGRAMMAR,
    reason="xgrammar not installed (install with: uv sync --group grammar)",
)


@pytest.fixture(scope="module")
def grammar() -> "xgr.Grammar":
    """Load and parse the kern GBNF grammar."""
    grammar_path = Path(__file__).parent.parent.parent / "grammars" / "kern.gbnf"
    grammar_text = grammar_path.read_text()
    return xgr.Grammar.from_ebnf(grammar_text)


def _accepts(grammar: "xgr.Grammar", text: str) -> bool:
    return _is_grammar_accept_string(grammar, text)


class TestSpineOpLinesShouldNotMixWithInterpretations:
    """Spine operations (*^, *v) must not share a line with tandem interpretations."""

    def test_rejects_split_mixed_with_clef(self, grammar: "xgr.Grammar"):
        """A line with *^ and a clef change should be rejected."""
        text = (
            "*clefG2\t*clefF4\n"
            "*k[]\t*k[]\n"
            "*M4/4\t*M4/4\n"
            "*^\t*clefG2\n"  # invalid: split mixed with clef
            "==\t==\n"
            "*-\t*-\n"
        )
        assert not _accepts(grammar, text)

    def test_rejects_join_mixed_with_keysig(self, grammar: "xgr.Grammar"):
        """A line with *v and a key signature change should be rejected."""
        text = (
            "*clefG2\t*clefF4\n"
            "*k[]\t*k[]\n"
            "*M4/4\t*M4/4\n"
            "*v\t*k[f#]\n"  # invalid: join mixed with keysig
            "==\t==\n"
            "*-\t*-\n"
        )
        assert not _accepts(grammar, text)

    def test_accepts_split_with_null(self, grammar: "xgr.Grammar"):
        """A split line with *^ and * (null) should be accepted."""
        text = (
            "*clefG2\t*clefF4\n"
            "*k[]\t*k[]\n"
            "*M4/4\t*M4/4\n"
            "*^\t*\n"  # valid: split with null
            "4c\t4e\t4G\n"
            "==\t==\t==\n"
            "*-\t*-\t*-\n"
        )
        assert _accepts(grammar, text)

    def test_accepts_join_with_null(self, grammar: "xgr.Grammar"):
        """A join line with *v and * (null) should be accepted."""
        text = (
            "*clefG2\t*clefF4\n"
            "*k[]\t*k[]\n"
            "*M4/4\t*M4/4\n"
            "*^\t*\n"
            "4c\t4e\t4G\n"
            "*v\t*v\t*\n"  # valid: joins with null
            "4c\t4G\n"
            "==\t==\n"
            "*-\t*-\n"
        )
        assert _accepts(grammar, text)

    def test_accepts_clef_change_with_null(self, grammar: "xgr.Grammar"):
        """A tandem interpretation line with clef and * should be accepted."""
        text = (
            "*clefG2\t*clefF4\n"
            "*k[]\t*k[]\n"
            "*M4/4\t*M4/4\n"
            "*clefC3\t*\n"  # valid: clef change with null
            "==\t==\n"
            "*-\t*-\n"
        )
        assert _accepts(grammar, text)


class TestOptionalTermination:
    """Termination is optional and no longer coupled to final barlines."""

    def test_accepts_note_as_terminal_line(self, grammar: "xgr.Grammar"):
        """A transcription may end directly on a note line."""
        text = (
            "*clefG2\t*clefF4\n"
            "*k[]\t*k[]\n"
            "*M4/4\t*M4/4\n"
            "4c\t4G\n"
        )
        assert _accepts(grammar, text)

    def test_accepts_interp_as_terminal_line(self, grammar: "xgr.Grammar"):
        """A transcription may end directly on a tandem interpretation."""
        text = (
            "*clefG2\t*clefF4\n"
            "*k[]\t*k[]\n"
            "*M4/4\t*M4/4\n"
            "4c\t4G\n"
            "*clefC3\t*\n"
        )
        assert _accepts(grammar, text)

    def test_accepts_termination_after_note(self, grammar: "xgr.Grammar"):
        """Optional terminator after note line should be accepted."""
        text = (
            "*clefG2\t*clefF4\n"
            "*k[]\t*k[]\n"
            "*M4/4\t*M4/4\n"
            "4c\t4G\n"
            "*-\t*-\n"
        )
        assert _accepts(grammar, text)

    def test_accepts_termination_after_numbered_barline(self, grammar: "xgr.Grammar"):
        """Termination no longer requires a final-style barline."""
        text = (
            "*clefG2\t*clefF4\n"
            "*k[]\t*k[]\n"
            "*M4/4\t*M4/4\n"
            "4c\t4G\n"
            "=2\t=2\n"
            "*-\t*-\n"
        )
        assert _accepts(grammar, text)

    def test_accepts_final_barline_without_termination(self, grammar: "xgr.Grammar"):
        """Final barline is sufficient; trailing *- line is optional."""
        text = (
            "*clefG2\t*clefF4\n"
            "*k[]\t*k[]\n"
            "*M4/4\t*M4/4\n"
            "4c\t4G\n"
            "==\t==\n"
        )
        assert _accepts(grammar, text)
