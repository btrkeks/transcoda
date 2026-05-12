from __future__ import annotations

import io
import re

from PIL import Image

try:  # Optional dependencies used for rendering
    import verovio
except Exception:  # pragma: no cover - verovio may be absent in some environments
    verovio = None  # type: ignore[assignment]

try:
    from cairosvg import svg2png
except Exception:  # pragma: no cover - cairosvg may be absent in some environments
    svg2png = None  # type: ignore[assignment]


_PITCH_CORE_RE = re.compile(r"([A-Ga-g]+(?:[#x\-]+|n+)?)")


def _is_comment_line(text: str) -> bool:
    return text.startswith("!")


def _is_interpretation_token(token: str) -> bool:
    return token.startswith("*")


def _is_barline_token(token: str) -> bool:
    return token.startswith("=")


def _is_null_token(token: str) -> bool:
    return token == "."


def _normalize_pitch_core(core: str) -> str:
    match = re.match(r"([A-Ga-g]+)(.*)$", core)
    if not match:
        return core
    letters, acc = match.groups()
    acc = acc.replace("x", "##").replace("n", "")
    return letters + acc


def _extract_pitch_core_span(subtoken: str) -> tuple[str, tuple[int, int]] | None:
    match = _PITCH_CORE_RE.search(subtoken)
    if not match:
        return None
    core = match.group(1)
    norm = _normalize_pitch_core(core)
    return norm, match.span(1)


def _split_chord_preserve_ws(token: str) -> list[str]:
    parts = re.split(r"(\s+)", token)
    return [part for part in parts if part]


def _token_is_pitch_bearing(token: str) -> bool:
    if _is_interpretation_token(token) or _is_barline_token(token) or _is_null_token(token):
        return False
    if token in ("", "r"):
        return False
    return _PITCH_CORE_RE.search(token) is not None


def _token_has_pitch_error(gt_token: str, pred_token: str) -> bool:
    if (
        _is_interpretation_token(pred_token)
        or _is_barline_token(pred_token)
        or _is_null_token(pred_token)
    ):
        return False
    if not _token_is_pitch_bearing(pred_token):
        return False

    gt_parts = _split_chord_preserve_ws(gt_token)
    pred_parts = _split_chord_preserve_ws(pred_token)

    gt_norm_pitches: list[str] = []
    for idx, piece in enumerate(gt_parts):
        if idx % 2 == 1:
            continue
        info = _extract_pitch_core_span(piece)
        if info is None:
            continue
        gt_norm_pitches.append(info[0])

    remaining = {pitch: gt_norm_pitches.count(pitch) for pitch in set(gt_norm_pitches)}

    for idx, piece in enumerate(pred_parts):
        if idx % 2 == 1:
            continue
        info = _extract_pitch_core_span(piece)
        if info is None:
            continue
        norm = info[0]
        if remaining.get(norm, 0) > 0:
            remaining[norm] -= 1
        else:
            return True

    return False


def _build_color_line(error_flags: list[bool], color: str) -> str:
    token = f"*color:{color}"
    return "\t".join(token if flag else "*" for flag in error_flags)


def _process_data_line(gt_line: str, pred_line: str) -> list[str]:
    gt_spines = gt_line.split("\t") if gt_line else []
    pred_spines = pred_line.split("\t") if pred_line else []

    if len(gt_spines) < len(pred_spines):
        gt_spines.extend([""] * (len(pred_spines) - len(gt_spines)))
    elif len(pred_spines) < len(gt_spines):
        pred_spines.extend([""] * (len(gt_spines) - len(pred_spines)))

    error_flags: list[bool] = []
    for gt_token, pred_token in zip(gt_spines, pred_spines, strict=False):
        error_flags.append(_token_has_pitch_error(gt_token, pred_token))

    if not any(error_flags):
        return [pred_line]

    start_line = _build_color_line(error_flags, "red")
    end_line = _build_color_line(error_flags, "black")
    return [start_line, pred_line, end_line]


def create_diff_humdrum(gt_humdrum: str, pred_humdrum: str) -> str:
    """Return the prediction Humdrum with red highlights for incorrect notes.

    Args:
        gt_humdrum: Ground-truth Humdrum string without the **kern header.
        pred_humdrum: Prediction Humdrum string without the **kern header.

    Returns:
        The prediction Humdrum string augmented with `*color:red` and
        `*color:black` commands that highlight incorrect notes in red.
    """
    gt_lines = gt_humdrum.splitlines()
    pred_lines = pred_humdrum.splitlines()

    if len(gt_lines) < len(pred_lines):
        gt_lines.extend([""] * (len(pred_lines) - len(gt_lines)))
    elif len(pred_lines) < len(gt_lines):
        pred_lines.extend([""] * (len(gt_lines) - len(pred_lines)))

    out_lines: list[str] = []

    for gt_line, pred_line in zip(gt_lines, pred_lines, strict=False):
        if not pred_line:
            out_lines.append(pred_line)
            continue
        if _is_comment_line(pred_line):
            out_lines.append(pred_line)
            continue

        tokens = pred_line.split("\t")
        if tokens and all(_is_interpretation_token(tok) for tok in tokens):
            out_lines.append(pred_line)
            continue

        out_lines.extend(_process_data_line(gt_line, pred_line))

    return "\n".join(out_lines)


def _placeholder_image() -> Image.Image:
    return Image.new("RGB", (200, 50), color="white")


# Default page width for Verovio visualization rendering.
# This is independent of training image_width - it's for display purposes only.
DEFAULT_PAGE_WIDTH = 2100

if verovio is not None:
    _VEROVIO_TOOLKIT = verovio.toolkit()
    try:
        _VEROVIO_TOOLKIT.enableLog(verovio.LOG_OFF)
    except AttributeError:  # pragma: no cover - constant name differs on older builds
        pass
    _VEROVIO_OPTIONS = {
        "pageWidth": DEFAULT_PAGE_WIDTH,
        "footer": "none",
        "barLineWidth": 0.5,
        "staffLineWidth": 0.2,
        "adjustPageHeight": 1,
    }
else:  # pragma: no cover - verovio unavailable
    _VEROVIO_TOOLKIT = None
    _VEROVIO_OPTIONS: dict[str, int | float | str] = {}


def render_humdrum_to_image(humdrum_string: str) -> Image.Image:
    """Render a Humdrum string to a PIL image using Verovio.

    Args:
        humdrum_string: A complete Humdrum string including headers.

    Returns:
        A PIL image containing the rendered score. If rendering fails, a blank
        placeholder image is returned instead.
    """
    if not humdrum_string.strip():
        return _placeholder_image()
    if _VEROVIO_TOOLKIT is None or svg2png is None:
        return _placeholder_image()

    try:
        _VEROVIO_TOOLKIT.setOptions(_VEROVIO_OPTIONS)
        _VEROVIO_TOOLKIT.loadData(humdrum_string)
        svg = _VEROVIO_TOOLKIT.renderToSVG()
        png_bytes = svg2png(bytestring=svg.encode("utf-8"), background_color="white")
        image = Image.open(io.BytesIO(png_bytes))
        return image.convert("RGB")
    except Exception:  # pragma: no cover - rendering may fail at runtime
        return _placeholder_image()
