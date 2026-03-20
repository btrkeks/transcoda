"""OMR-NED (Normalized Edit Distance) metric using musicdiff.

This metric provides semantically-aware comparison of **kern notation,
considering musical structure rather than just string similarity.

The musicdiff library is an optional dependency. If not installed,
compute_omr_ned() will return a result with omr_ned=None and an error message.

References:
    - musicdiff: https://github.com/gregchapman-dev/musicdiff
    - Foscarin et al., "A diff procedure for music score files" (2019)
"""

from __future__ import annotations

import copy
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

# Optional dependency - musicdiff requires music21, converter21


def _copy_musicdiff_memoized_result(
    value: Any,
    *,
    fallback: Callable[[Any, dict[int, Any] | None], Any],
    memo: dict[int, Any] | None = None,
) -> Any:
    """Copy memoized musicdiff results without recursively cloning op payloads."""
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], list):
        return (list(value[0]), value[1])
    if isinstance(value, list):
        return list(value)
    return fallback(value, memo)


def _install_musicdiff_fast_memo_copy(module: Any) -> None:
    """Patch musicdiff's local deepcopy usage for memoizer cache hits."""
    current_copy_module = module.copy
    current_deepcopy = current_copy_module.deepcopy
    if getattr(current_deepcopy, "_smt_fast_memo_copy", False):
        return

    original_deepcopy = current_deepcopy

    def _patched_deepcopy(value: Any, memo: dict[int, Any] | None = None) -> Any:
        return _copy_musicdiff_memoized_result(
            value,
            fallback=original_deepcopy,
            memo=memo,
        )

    _patched_deepcopy._smt_fast_memo_copy = True  # type: ignore[attr-defined]
    module.copy = SimpleNamespace(
        copy=current_copy_module.copy,
        deepcopy=_patched_deepcopy,
    )


try:
    import converter21
    import music21 as m21
    import musicdiff.comparison as musicdiff_comparison

    # Register converter21 for Humdrum support
    converter21.register()

    from musicdiff.annotation import AnnScore
    from musicdiff.comparison import Comparison

    _install_musicdiff_fast_memo_copy(musicdiff_comparison)

    MUSICDIFF_AVAILABLE = True
except ImportError:
    MUSICDIFF_AVAILABLE = False
    m21 = None  # type: ignore[assignment]
    AnnScore = None  # type: ignore[assignment,misc]
    Comparison = None  # type: ignore[assignment,misc]


@dataclass
class OMRNEDResult:
    """Result of OMR-NED computation for a single sample.

    Attributes:
        omr_ned: Normalized Edit Distance as percentage (0-100), or None if failed
        edit_distance: Raw edit distance between scores
        pred_notation_size: Number of notation symbols in prediction
        gt_notation_size: Number of notation symbols in ground truth
        parse_error: Error message if parsing failed, None otherwise
        syntax_errors_fixed: Number of syntax errors auto-fixed in prediction
    """

    omr_ned: float | None
    edit_distance: int | None
    pred_notation_size: int | None
    gt_notation_size: int | None
    parse_error: str | None
    syntax_errors_fixed: int


def is_musicdiff_available() -> bool:
    """Check if musicdiff and its dependencies are available."""
    return MUSICDIFF_AVAILABLE


def compute_omr_ned(
    pred: str,
    target: str,
    fix_prediction_syntax: bool = True,
) -> OMRNEDResult:
    """Compute OMR-NED between prediction and ground truth **kern strings.

    OMR-NED (Normalized Edit Distance) measures the structural similarity
    of music notation, accounting for musical semantics like note durations,
    pitches, ties, and beaming.

    Args:
        pred: Predicted **kern string
        target: Ground truth **kern string
        fix_prediction_syntax: If True, attempt to fix syntax errors in
            prediction (errors count toward edit distance)

    Returns:
        OMRNEDResult with computed metrics or parse error information.
        Returns omr_ned=None if musicdiff is not installed or parsing fails.
    """
    if not MUSICDIFF_AVAILABLE:
        return OMRNEDResult(
            omr_ned=None,
            edit_distance=None,
            pred_notation_size=None,
            gt_notation_size=None,
            parse_error="musicdiff not installed. Install with: uv sync --group omr-ned",
            syntax_errors_fixed=0,
        )

    if not pred.strip() or not target.strip():
        return OMRNEDResult(
            omr_ned=None,
            edit_distance=None,
            pred_notation_size=None,
            gt_notation_size=None,
            parse_error="Empty prediction or target",
            syntax_errors_fixed=0,
        )

    # musicdiff requires file paths - create temp files
    pred_path: Path | None = None
    target_path: Path | None = None

    try:
        # Ensure **kern header is present for parsing
        pred_kern = _ensure_kern_header(pred)
        target_kern = _ensure_kern_header(target)

        # Create temporary files
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".krn", delete=False, encoding="utf-8"
        ) as pred_file:
            pred_file.write(pred_kern)
            pred_path = Path(pred_file.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".krn", delete=False, encoding="utf-8"
        ) as target_file:
            target_file.write(target_kern)
            target_path = Path(target_file.name)

        # Parse scores with music21
        try:
            pred_score = m21.converter.parse(
                str(pred_path),
                forceSource=True,
                acceptSyntaxErrors=fix_prediction_syntax,
            )
            target_score = m21.converter.parse(
                str(target_path),
                forceSource=True,
                acceptSyntaxErrors=False,
            )
        except Exception as e:
            return OMRNEDResult(
                omr_ned=None,
                edit_distance=None,
                pred_notation_size=None,
                gt_notation_size=None,
                parse_error=f"Parse error: {str(e)[:200]}",
                syntax_errors_fixed=0,
            )

        # Create annotated scores for comparison
        try:
            pred_ann = AnnScore(pred_score)
            target_ann = AnnScore(target_score)
        except Exception as e:
            return OMRNEDResult(
                omr_ned=None,
                edit_distance=None,
                pred_notation_size=None,
                gt_notation_size=None,
                parse_error=f"Annotation error: {str(e)[:200]}",
                syntax_errors_fixed=0,
            )

        # Compute comparison
        try:
            _ops, edit_distance = Comparison.annotated_scores_diff(pred_ann, target_ann)
        except Exception as e:
            return OMRNEDResult(
                omr_ned=None,
                edit_distance=None,
                pred_notation_size=pred_ann.notation_size(),
                gt_notation_size=target_ann.notation_size(),
                parse_error=f"Comparison error: {str(e)[:200]}",
                syntax_errors_fixed=pred_ann.num_syntax_errors_fixed,
            )

        # Compute NED as percentage
        pred_size = pred_ann.notation_size()
        target_size = target_ann.notation_size()
        syntax_fixes = pred_ann.num_syntax_errors_fixed

        total_size = pred_size + target_size
        if total_size == 0:
            # Both empty - consider as perfect match
            omr_ned = 0.0
        else:
            # NED formula from musicdiff, converted to percentage
            omr_ned = ((edit_distance + syntax_fixes) / total_size) * 100.0
            # Clamp to [0, 100] - can exceed in extreme cases
            omr_ned = min(omr_ned, 100.0)

        return OMRNEDResult(
            omr_ned=omr_ned,
            edit_distance=edit_distance,
            pred_notation_size=pred_size,
            gt_notation_size=target_size,
            parse_error=None,
            syntax_errors_fixed=syntax_fixes,
        )

    finally:
        # Clean up temp files
        if pred_path and pred_path.exists():
            pred_path.unlink()
        if target_path and target_path.exists():
            target_path.unlink()


def compute_omr_ned_from_musicxml(
    pred_xml: str,
    target_xml: str,
) -> OMRNEDResult:
    """Compute OMR-NED between prediction and ground truth MusicXML strings."""
    if not MUSICDIFF_AVAILABLE:
        return OMRNEDResult(
            omr_ned=None,
            edit_distance=None,
            pred_notation_size=None,
            gt_notation_size=None,
            parse_error="musicdiff not installed. Install with: uv sync --group omr-ned",
            syntax_errors_fixed=0,
        )

    if not pred_xml.strip() or not target_xml.strip():
        return OMRNEDResult(
            omr_ned=None,
            edit_distance=None,
            pred_notation_size=None,
            gt_notation_size=None,
            parse_error="Empty prediction or target",
            syntax_errors_fixed=0,
        )

    pred_path: Path | None = None
    target_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".musicxml", delete=False, encoding="utf-8"
        ) as pred_file:
            pred_file.write(pred_xml)
            pred_path = Path(pred_file.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".musicxml", delete=False, encoding="utf-8"
        ) as target_file:
            target_file.write(target_xml)
            target_path = Path(target_file.name)

        try:
            pred_score = m21.converter.parse(str(pred_path), forceSource=True)
            target_score = m21.converter.parse(str(target_path), forceSource=True)
        except Exception as exc:
            return OMRNEDResult(
                omr_ned=None,
                edit_distance=None,
                pred_notation_size=None,
                gt_notation_size=None,
                parse_error=f"Parse error: {str(exc)[:200]}",
                syntax_errors_fixed=0,
            )

        try:
            pred_ann = AnnScore(pred_score)
            target_ann = AnnScore(target_score)
        except Exception as exc:
            return OMRNEDResult(
                omr_ned=None,
                edit_distance=None,
                pred_notation_size=None,
                gt_notation_size=None,
                parse_error=f"Annotation error: {str(exc)[:200]}",
                syntax_errors_fixed=0,
            )

        try:
            _ops, edit_distance = Comparison.annotated_scores_diff(pred_ann, target_ann)
        except Exception as exc:
            return OMRNEDResult(
                omr_ned=None,
                edit_distance=None,
                pred_notation_size=pred_ann.notation_size(),
                gt_notation_size=target_ann.notation_size(),
                parse_error=f"Comparison error: {str(exc)[:200]}",
                syntax_errors_fixed=pred_ann.num_syntax_errors_fixed,
            )

        pred_size = pred_ann.notation_size()
        target_size = target_ann.notation_size()
        syntax_fixes = pred_ann.num_syntax_errors_fixed
        total_size = pred_size + target_size
        if total_size == 0:
            omr_ned = 0.0
        else:
            omr_ned = ((edit_distance + syntax_fixes) / total_size) * 100.0
            omr_ned = min(omr_ned, 100.0)

        return OMRNEDResult(
            omr_ned=omr_ned,
            edit_distance=edit_distance,
            pred_notation_size=pred_size,
            gt_notation_size=target_size,
            parse_error=None,
            syntax_errors_fixed=syntax_fixes,
        )
    finally:
        if pred_path and pred_path.exists():
            pred_path.unlink()
        if target_path and target_path.exists():
            target_path.unlink()


def _ensure_kern_header(kern_str: str) -> str:
    """Ensure **kern string has proper Humdrum header for parsing.

    If the string doesn't start with a spine indicator (**kern),
    prepends the appropriate header based on the number of spines.

    Args:
        kern_str: Raw **kern string, possibly without header

    Returns:
        **kern string with proper header
    """
    stripped = kern_str.strip()
    if not stripped:
        return kern_str

    lines = stripped.split("\n")

    # Check if first line is already a spine indicator
    first_line = lines[0].strip()
    if first_line.startswith("**"):
        return kern_str

    # Count spines from first data line (tab-separated)
    first_data = lines[0]
    num_spines = first_data.count("\t") + 1

    # Add **kern headers and terminator
    header = "\t".join(["**kern"] * num_spines)
    terminator = "\t".join(["*-"] * num_spines)

    # Check if terminator already exists
    if lines[-1].strip().startswith("*-"):
        return header + "\n" + kern_str
    else:
        return header + "\n" + kern_str + "\n" + terminator
