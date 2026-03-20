"""Interactive notebook app for manual Polish transcription fixes.

Workflow:
- Read transcriptions from 4_manual_fixes.
- Show upstream PRAIG scan and Verovio render (all pages).
- Edit transcription and submit to save in-place.
- Render in a subprocess so Verovio crashes do not kill the notebook kernel.
"""

from __future__ import annotations

import html
import multiprocessing as mp
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from IPython.display import HTML, clear_output, display
from IPython.display import Image as IPyImage
from PIL import Image

try:
    import ipywidgets as widgets
except Exception:  # pragma: no cover - optional runtime dependency
    widgets = None  # type: ignore[assignment]

from datasets import load_dataset

_SAMPLE_RE = re.compile(r"^([a-z]+)_(\d+)\.krn$")
_RATIONAL_DURATION_PATTERN = re.compile(r"\d%-?\d")
_DEFAULT_RENDER_OPTIONS = {
    "pageWidth": 2100,
    "adjustPageHeight": 1,
    "footer": "none",
    "barLineWidth": 0.5,
    "staffLineWidth": 0.2,
}


@dataclass(frozen=True)
class ManualFixPaths:
    normalized_dir: Path | str
    manual_dir: Path | str


@dataclass(frozen=True)
class SampleRef:
    filename: str
    split: str
    index: int
    path: Path


@dataclass(frozen=True)
class RenderResult:
    ok: bool
    page_png_bytes: list[bytes]
    page_count: int
    warning: str | None
    error: str | None
    exit_code: int | None


@dataclass(frozen=True)
class ManualStageReadiness:
    ready: bool
    message: str
    missing_in_manual: tuple[str, ...]
    extra_in_manual: tuple[str, ...]


def is_valid_kern(content: str) -> tuple[bool, str | None]:
    """Validate kern content for known malformation patterns."""
    if _RATIONAL_DURATION_PATTERN.search(content):
        return False, "rational duration (corrupted source)"
    return True, None


def ensure_kern_header(content: str) -> str:
    """Prepend **kern header if no exclusive interpretation is present."""
    first_data_like_line: str | None = None
    for line in content.splitlines():
        if not line.strip():
            continue
        if line.startswith("!!"):
            continue
        first_data_like_line = line
        break

    if first_data_like_line is None:
        return content
    if first_data_like_line.startswith("**"):
        return content

    num_spines = first_data_like_line.count("\t") + 1
    header = "\t".join(["**kern"] * num_spines)
    return header + "\n" + content


def parse_sample_filename(filename: str) -> tuple[str, int]:
    match = _SAMPLE_RE.match(filename)
    if not match:
        raise ValueError(f"Unexpected filename format: {filename}")
    return match.group(1), int(match.group(2))


def build_sample_refs(manual_dir: str | Path) -> list[SampleRef]:
    manual_path = Path(manual_dir)
    refs: list[SampleRef] = []
    for path in sorted(manual_path.glob("*.krn")):
        try:
            split, idx = parse_sample_filename(path.name)
        except ValueError:
            continue
        refs.append(
            SampleRef(
                filename=path.name,
                split=split,
                index=idx,
                path=path,
            )
        )
    return refs


def _manual_init_command(paths: ManualFixPaths) -> str:
    normalized_dir = Path(paths.normalized_dir)
    manual_dir = Path(paths.manual_dir)
    return f"rsync -a {normalized_dir.as_posix()}/ {manual_dir.as_posix()}/"


def check_manual_stage_readiness(paths: ManualFixPaths) -> ManualStageReadiness:
    normalized_dir = Path(paths.normalized_dir)
    manual_dir = Path(paths.manual_dir)

    if not normalized_dir.exists():
        return ManualStageReadiness(
            ready=False,
            message=f"Normalized directory is missing: {normalized_dir}",
            missing_in_manual=(),
            extra_in_manual=(),
        )

    if not manual_dir.exists():
        return ManualStageReadiness(
            ready=False,
            message=(
                f"Manual-fix directory is missing: {manual_dir}. "
                f"Initialize it first with: `{_manual_init_command(paths)}`"
            ),
            missing_in_manual=(),
            extra_in_manual=(),
        )

    normalized_names = {path.name for path in normalized_dir.glob("*.krn")}
    manual_names = {path.name for path in manual_dir.glob("*.krn")}

    missing = tuple(sorted(normalized_names - manual_names))
    extra = tuple(sorted(manual_names - normalized_names))
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(f"missing in manual_dir: {len(missing)}")
        if extra:
            parts.append(f"extra in manual_dir: {len(extra)}")
        details = ", ".join(parts)
        return ManualStageReadiness(
            ready=True,
            message=(
                f"Manual-fix directory is not aligned with normalized files ({details}). "
                f"Proceeding with manual fixes anyway. Suggested sync: "
                f"`{_manual_init_command(paths)}`"
            ),
            missing_in_manual=missing,
            extra_in_manual=extra,
        )

    return ManualStageReadiness(
        ready=True,
        message="Manual-fix directory is ready.",
        missing_in_manual=(),
        extra_in_manual=(),
    )


def save_transcription(file_path: str | Path, transcription: str) -> None:
    Path(file_path).write_text(transcription, encoding="utf-8")


def _render_worker(
    transcription: str,
    render_options: dict[str, Any],
    send_conn: Any,
) -> None:
    try:
        import cairosvg
        import verovio

        toolkit = verovio.toolkit()
        try:
            toolkit.enableLog(verovio.LOG_OFF)
        except Exception:
            pass

        toolkit.setOptions(render_options)
        toolkit.loadData(ensure_kern_header(transcription))

        page_count = int(toolkit.getPageCount())
        if page_count < 1:
            page_count = 1

        pages: list[bytes] = []
        for page in range(1, page_count + 1):
            svg = toolkit.renderToSVG(page)
            png = cairosvg.svg2png(
                bytestring=svg.encode("utf-8"),
                background_color="white",
            )
            pages.append(png)

        send_conn.send(
            {
                "ok": True,
                "page_count": page_count,
                "pages": pages,
                "warning": None,
                "error": None,
            }
        )
    except Exception as exc:  # noqa: BLE001
        send_conn.send(
            {
                "ok": False,
                "page_count": 0,
                "pages": [],
                "warning": None,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            }
        )
    finally:
        try:
            send_conn.close()
        except Exception:
            pass


def render_transcription_isolated(
    transcription: str,
    timeout_s: float = 15.0,
    render_options: dict[str, Any] | None = None,
) -> RenderResult:
    opts = dict(_DEFAULT_RENDER_OPTIONS)
    if render_options:
        opts.update(render_options)

    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_render_worker, args=(transcription, opts, send_conn))

    process.start()
    send_conn.close()

    payload: dict[str, Any] | None = None
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if recv_conn.poll(0.1):
                try:
                    payload = recv_conn.recv()
                except EOFError:
                    payload = None
                break
            if not process.is_alive():
                break
    finally:
        recv_conn.close()

    if payload is None:
        if process.is_alive():
            process.terminate()
            process.join(5)
            return RenderResult(
                ok=False,
                page_png_bytes=[],
                page_count=0,
                warning=None,
                error=f"Render timed out after {timeout_s:.1f}s",
                exit_code=process.exitcode,
            )

        process.join(1)
        if process.exitcode and process.exitcode != 0:
            return RenderResult(
                ok=False,
                page_png_bytes=[],
                page_count=0,
                warning=None,
                error=f"Render worker exited unexpectedly (exit_code={process.exitcode})",
                exit_code=process.exitcode,
            )
        return RenderResult(
            ok=False,
            page_png_bytes=[],
            page_count=0,
            warning=None,
            error="Render worker returned no payload.",
                exit_code=process.exitcode,
            )

    process.join(1)
    return RenderResult(
        ok=bool(payload.get("ok", False)),
        page_png_bytes=list(payload.get("pages", [])),
        page_count=int(payload.get("page_count", 0)),
        warning=payload.get("warning"),
        error=payload.get("error"),
        exit_code=process.exitcode,
    )


class PolishManualFixApp:
    """Jupyter UI for manual Polish score transcription fixes."""

    def __init__(
        self,
        normalized_dir: str = "data/interim/val/polish-scores/3_normalized",
        manual_dir: str = "data/interim/val/polish-scores/4_manual_fixes",
        hf_dataset_name: str = "PRAIG/polish-scores",
        hf_dataset: Any | None = None,
        hf_cache_dir: str | None = None,
        render_timeout_s: float = 15.0,
        preview_panel_max_height_px: int = 760,
        editor_height_px: int = 300,
    ) -> None:
        if widgets is None:
            raise RuntimeError(
                "ipywidgets is required for PolishManualFixApp. "
                "Install dev dependencies and run in Jupyter."
            )

        self.paths = ManualFixPaths(
            normalized_dir=Path(normalized_dir),
            manual_dir=Path(manual_dir),
        )
        self.render_timeout_s = render_timeout_s

        self.readiness = check_manual_stage_readiness(self.paths)
        self.samples = build_sample_refs(self.paths.manual_dir) if self.paths.manual_dir.exists() else []

        self.hf_dataset_name = hf_dataset_name
        self.hf_dataset = hf_dataset
        self.hf_cache_dir = hf_cache_dir
        self.hf_load_error: str | None = None
        self.preview_panel_max_height_px = preview_panel_max_height_px
        self.editor_height_px = editor_height_px

        if self.hf_dataset is None:
            self._load_hf_dataset()

        self._init_widgets()
        self._recompute_filtered_samples()
        self._set_edit_mode(self.readiness.ready)

    def _load_hf_dataset(self) -> None:
        try:
            kwargs = {}
            if self.hf_cache_dir:
                kwargs["cache_dir"] = self.hf_cache_dir
            self.hf_dataset = load_dataset(self.hf_dataset_name, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self.hf_load_error = f"Failed to load dataset '{self.hf_dataset_name}': {exc}"
            self.hf_dataset = None

    def _init_widgets(self) -> None:
        self.split_dropdown = widgets.Dropdown(
            options=[("all", "all"), ("train", "train"), ("val", "val"), ("test", "test")],
            value="all",
            description="Split:",
            layout=widgets.Layout(width="180px"),
        )

        self.prev_btn = widgets.Button(description="◄ Previous", button_style="info")
        self.next_btn = widgets.Button(description="Next ►", button_style="info")
        self.refresh_btn = widgets.Button(description="Refresh Render", button_style="")
        self.submit_btn = widgets.Button(description="Submit", button_style="success")

        self.index_slider = widgets.IntSlider(
            min=0,
            max=0,
            value=0,
            step=1,
            description="Index:",
            continuous_update=False,
            layout=widgets.Layout(width="420px"),
        )

        self.sample_label = widgets.HTML(value="<b>No samples loaded</b>")
        self.status_html = widgets.HTML(value="")

        self.editor = widgets.Textarea(
            value="",
            description="",
            layout=widgets.Layout(width="100%", height=f"{self.editor_height_px}px"),
        )

        self.scan_output = widgets.Output(
            layout=widgets.Layout(
                width="48%",
                max_height=f"{self.preview_panel_max_height_px}px",
                overflow="auto",
                border="1px solid #ddd",
            )
        )
        self.render_output = widgets.Output(
            layout=widgets.Layout(
                width="48%",
                max_height=f"{self.preview_panel_max_height_px}px",
                overflow="auto",
                border="1px solid #ddd",
            )
        )

        self.split_dropdown.observe(self._on_split_change, names="value")
        self.index_slider.observe(self._on_index_change, names="value")
        self.prev_btn.on_click(self._on_prev)
        self.next_btn.on_click(self._on_next)
        self.refresh_btn.on_click(self._on_refresh)
        self.submit_btn.on_click(self._on_submit)

    def _set_edit_mode(self, editable: bool) -> None:
        self.editor.disabled = not editable
        self.submit_btn.disabled = not editable

    @property
    def filtered_samples(self) -> list[SampleRef]:
        return self._filtered_samples

    def _recompute_filtered_samples(self) -> None:
        selected_split = self.split_dropdown.value
        if selected_split == "all":
            filtered = self.samples
        else:
            filtered = [sample for sample in self.samples if sample.split == selected_split]

        self._filtered_samples = filtered

        if not filtered:
            self.index_slider.max = 0
            self.index_slider.value = 0
            self.index_slider.disabled = True
            self.prev_btn.disabled = True
            self.next_btn.disabled = True
            self.refresh_btn.disabled = True
            return

        self.index_slider.disabled = False
        self.refresh_btn.disabled = False
        self.index_slider.max = len(filtered) - 1
        self.index_slider.value = min(self.index_slider.value, len(filtered) - 1)
        self._sync_nav_button_state()

    def _sync_nav_button_state(self) -> None:
        if not self.filtered_samples:
            self.prev_btn.disabled = True
            self.next_btn.disabled = True
            return
        self.prev_btn.disabled = self.index_slider.value <= 0
        self.next_btn.disabled = self.index_slider.value >= len(self.filtered_samples) - 1

    def _current_sample(self) -> SampleRef | None:
        if not self.filtered_samples:
            return None
        idx = self.index_slider.value
        if idx < 0 or idx >= len(self.filtered_samples):
            return None
        return self.filtered_samples[idx]

    def _set_status(self, text: str, level: str = "info") -> None:
        colors = {
            "info": "#1f2937",
            "success": "#166534",
            "warning": "#92400e",
            "error": "#991b1b",
        }
        color = colors.get(level, "#1f2937")
        self.status_html.value = (
            "<div style='margin-top:8px; padding:8px; border:1px solid #ddd; "
            f"color:{color}; background:#f9fafb'>{html.escape(text)}</div>"
        )

    def _set_warning_from_readiness(self) -> None:
        if self.readiness.ready and not (
            self.readiness.missing_in_manual or self.readiness.extra_in_manual
        ):
            return
        self._set_status(self.readiness.message, level="warning")

    def _fetch_scan_image(self, sample: SampleRef) -> tuple[Image.Image | None, str | None]:
        if self.hf_load_error:
            return None, self.hf_load_error
        if self.hf_dataset is None:
            return None, "Dataset is not loaded."
        if sample.split not in self.hf_dataset:
            return None, f"Split '{sample.split}' not found in dataset."

        split_ds = self.hf_dataset[sample.split]
        if sample.index >= len(split_ds):
            return None, (
                f"Index {sample.index} out of range for split '{sample.split}' "
                f"(size={len(split_ds)})."
            )

        image = split_ds[sample.index]["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        return image.convert("RGB"), None

    def _render_scan_panel(self, sample: SampleRef) -> None:
        with self.scan_output:
            clear_output(wait=True)
            image, error = self._fetch_scan_image(sample)
            if error:
                display(HTML(f"<div style='color:#991b1b'>{html.escape(error)}</div>"))
                return
            assert image is not None
            display(image)

    def _render_transcription_panel(self, transcription: str) -> RenderResult:
        with self.render_output:
            clear_output(wait=True)
            result = render_transcription_isolated(
                transcription,
                timeout_s=self.render_timeout_s,
            )
            if not result.ok:
                err_text = result.error or "unknown render error"
                display(HTML(f"<div style='color:#991b1b'>{html.escape(err_text)}</div>"))
                return result

            for png in result.page_png_bytes:
                display(IPyImage(data=png, format="png"))
            return result

    def _load_current_sample(self) -> None:
        sample = self._current_sample()
        if sample is None:
            self.sample_label.value = "<b>No samples for selected split.</b>"
            with self.scan_output:
                clear_output(wait=True)
            with self.render_output:
                clear_output(wait=True)
            self.editor.value = ""
            return

        self.sample_label.value = (
            f"<b>{html.escape(sample.filename)}</b> "
            f"({self.index_slider.value + 1}/{len(self.filtered_samples)})"
        )

        transcription = sample.path.read_text(encoding="utf-8")
        self.editor.value = transcription

        self._render_scan_panel(sample)
        render_result = self._render_transcription_panel(transcription)

        if not self.readiness.ready or (
            self.readiness.missing_in_manual or self.readiness.extra_in_manual
        ):
            self._set_warning_from_readiness()
        elif render_result.ok:
            self._set_status("Loaded sample.", level="info")
        else:
            self._set_status(render_result.error or "Render failed.", level="warning")

    def _on_split_change(self, _change: Any) -> None:
        self._recompute_filtered_samples()
        self._sync_nav_button_state()
        self._load_current_sample()

    def _on_index_change(self, _change: Any) -> None:
        self._sync_nav_button_state()
        self._load_current_sample()

    def _on_prev(self, _btn: Any) -> None:
        if self.index_slider.value > 0:
            self.index_slider.value -= 1

    def _on_next(self, _btn: Any) -> None:
        if self.index_slider.value < self.index_slider.max:
            self.index_slider.value += 1

    def _on_refresh(self, _btn: Any) -> None:
        if not self.filtered_samples:
            return
        result = self._render_transcription_panel(self.editor.value)
        if result.ok:
            self._set_status("Render refreshed.", level="info")
        else:
            self._set_status(result.error or "Render failed.", level="warning")

    def _on_submit(self, _btn: Any) -> None:
        sample = self._current_sample()
        if sample is None:
            self._set_status("No sample selected.", level="warning")
            return

        transcription = self.editor.value
        warnings: list[str] = []

        is_valid, error_msg = is_valid_kern(transcription)
        if not is_valid and error_msg:
            warnings.append(f"Validation warning: {error_msg}")

        save_transcription(sample.path, transcription)
        render_result = self._render_transcription_panel(transcription)
        if not render_result.ok:
            warnings.append(f"Render warning: {render_result.error}")

        if warnings:
            self._set_status(
                "Saved. " + " | ".join(warnings),
                level="warning",
            )
        else:
            self._set_status("Saved and re-rendered.", level="success")

    def show(self) -> None:
        controls_row = widgets.HBox(
            [
                self.split_dropdown,
                self.index_slider,
                self.prev_btn,
                self.next_btn,
                self.refresh_btn,
                self.submit_btn,
            ]
        )

        top = widgets.VBox([self.sample_label, controls_row, self.status_html])
        panels = widgets.HBox([self.scan_output, self.render_output])
        root = widgets.VBox([top, panels, self.editor])

        display(root)
        self._load_current_sample()
