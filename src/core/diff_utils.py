# diff_utils.py

"""Utilities for generating diff visualizations between ground truth and predictions."""

import difflib


def generate_html_diff(
    ground_truth: str,
    prediction: str,
    *,
    show_full: bool = True,
    fromdesc: str = "Ground truth",
    todesc: str = "Prediction",
    wrapcolumn: int | None = None,  # set to an int (e.g., 160) if you want intraline wrapping
    disable_char_junk: bool = True,  # improves intraline accuracy
    num_context_lines: int = 3,  # used only when show_full=False
) -> str:
    """
    Generate an HTML side-by-side diff between ground truth and prediction.

    Args:
        ground_truth: Reference text
        prediction: Hypothesis text
        show_full: If True, include unchanged lines (no collapsing)
        fromdesc / todesc: Column headers
        wrapcolumn: Optional hard wrap for intraline highlighting
        disable_char_junk: If True, don't treat punctuation/whitespace as 'junk'
        num_context_lines: When show_full is False, how many context lines to keep

    Returns:
        HTML string
    """
    hd = difflib.HtmlDiff(
        tabsize=8,
        wrapcolumn=wrapcolumn,
        linejunk=None,
        charjunk=None if disable_char_junk else difflib.IS_CHARACTER_JUNK,
    )
    html = hd.make_file(
        ground_truth.splitlines(),
        prediction.splitlines(),
        fromdesc=fromdesc,
        todesc=todesc,
        context=not show_full,
        numlines=num_context_lines,
    )
    return html
