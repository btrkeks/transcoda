from src.core.kern_postprocess import append_terminator_if_missing, strip_terminal_terminator_lines


def test_strip_terminal_terminator_lines():
    text = "**kern\t**kern\n4c\t4e\n==\t==\n*-\t*-"
    assert strip_terminal_terminator_lines(text) == "**kern\t**kern\n4c\t4e\n==\t=="


def test_strip_terminal_terminator_lines_noop():
    text = "**kern\t**kern\n4c\t4e\n=\t="
    assert strip_terminal_terminator_lines(text) == text


def test_append_terminator_if_missing():
    text = "**kern\t**kern\n4c\t4e\n=\t="
    assert append_terminator_if_missing(text) == "**kern\t**kern\n4c\t4e\n=\t=\n*-\t*-"


def test_append_terminator_if_missing_noop():
    text = "**kern\t**kern\n4c\t4e\n=\t=\n*-\t*-"
    assert append_terminator_if_missing(text) == text


def test_append_terminator_if_missing_uses_post_split_spine_count():
    text = "**kern\t**kern\n4c\t4e\n*\t*^"
    assert append_terminator_if_missing(text).splitlines()[-1] == "*-\t*-\t*-"


def test_append_terminator_if_missing_uses_post_merge_spine_count():
    text = "**kern\t**kern\t**kern\n4c\t4e\t4g\n*\t*v\t*v"
    assert append_terminator_if_missing(text).splitlines()[-1] == "*-\t*-"
