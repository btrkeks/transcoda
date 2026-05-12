from src.core.kern_utils import (
    strip_tie_beam_markers_from_kern_text,
    strip_tie_beam_markers_from_note_token,
)


def test_strip_tie_beam_markers_from_note_token():
    assert strip_tie_beam_markers_from_note_token("4a]L") == "4a"
    assert strip_tie_beam_markers_from_note_token("[8cc_Jk") == "8cc"
    assert strip_tie_beam_markers_from_note_token("") == ""


def test_strip_tie_beam_markers_from_kern_text_note_aware():
    source = (
        "*k[f#]\t*k[b-e-]\n"
        "4a]L\t4b[JJ\n"
        "8cc_L 16dK\t.\n"
        "=1\t=1"
    )
    expected = (
        "*k[f#]\t*k[b-e-]\n"
        "4a\t4b\n"
        "8cc 16d\t.\n"
        "=1\t=1"
    )
    assert strip_tie_beam_markers_from_kern_text(source) == expected
