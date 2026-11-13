"""Isolated Verovio rendering probe. No metric mask is applied to predictions."""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from pipelines.baselines.contracts import canonical_json_bytes
from pipelines.baselines.io import sha256_file
from pipelines.baselines.normalization import SubprocessCommandRunner

DEFINITION = (
    "After the recorded notation conversion with no repairs, Verovio "
    "must load the full prediction, produce at least one page, and render every "
    "page to a nonempty SVG even with error diagnostics. Secondary renderability also requires no error diagnostic. Warnings are allowed. "
    "Kern must contain a **kern exclusive interpretation. "
    "ABC (including Legato) is converted with the recorded abc2xml script and "
    "Python executable, then the resulting MusicXML is tested. Conversion errors, "
    "native crashes, timeouts and inference failures are not renderable."
)


def verovio_provenance() -> dict[str, Any]:
    import verovio

    root = Path(verovio.__file__).parent
    return {
        "version": verovio.toolkit().getVersion(),
        "native_modules": {path.name: sha256_file(path) for path in sorted(root.glob("*.so"))},
        "definition": DEFINITION,
        "options": {"inputFrom": "humdrum|musicxml|mei (per normalized format)"},
    }


def probe_renderability(text: str, format: str, *, timeout_s: float) -> dict[str, Any]:
    try:
        result = SubprocessCommandRunner().run(
            [sys.executable, "-m", "pipelines.baselines.renderability"],
            input_bytes=canonical_json_bytes({"text": text, "format": format}),
            timeout_s=timeout_s,
        )
        full_diagnostic = result.stderr.decode("utf-8", errors="replace")
        diagnostic = full_diagnostic[-4000:]
        if result.returncode != 0:
            return {
                "renderable": False,
                "renders_svg": False,
                "renders_svg_without_error": False,
                "has_error_diagnostic": bool(
                    re.search(r"\[Error\]|\bError:", full_diagnostic, re.I)
                ),
                "error": f"render process exit {result.returncode}: {diagnostic}",
            }
        payload = json.loads(result.stdout)
        if re.search(r"\[Error\]|\bError:", full_diagnostic, re.IGNORECASE):
            payload.update(
                renders_svg_without_error=False, has_error_diagnostic=True, error=diagnostic
            )
        payload["stderr"] = diagnostic
        return payload
    except (OSError, RuntimeError, ValueError) as exc:
        return {
            "renderable": False,
            "renders_svg": False,
            "renders_svg_without_error": False,
            "has_error_diagnostic": False,
            "error": str(exc)[:4000],
        }


def main() -> None:
    import verovio

    request = json.load(sys.stdin)
    toolkit = verovio.toolkit()
    toolkit.setInputFrom(
        {"kern": "humdrum", "musicxml": "musicxml", "mei": "mei"}[request["format"]]
    )
    logs = []
    try:
        if request["format"] == "kern" and not any(
            "**kern" in line.split("\t") for line in request["text"].splitlines()
        ):
            raise ValueError("Kern prediction has no **kern exclusive interpretation")
        loaded = toolkit.loadData(request["text"])
        logs.append(toolkit.getLog())
        page_count = toolkit.getPageCount() if loaded else 0
        if not loaded or page_count < 1:
            raise ValueError("Verovio did not load a score with at least one page")
        for page in range(1, page_count + 1):
            svg = toolkit.renderToSVG(page)
            logs.append(toolkit.getLog())
            if not svg or ET.fromstring(svg).tag.rsplit("}", 1)[-1] != "svg":
                raise ValueError(f"Verovio did not render page {page} to SVG")
        log = "\n".join(logs)
        errors = re.search(r"\[Error\]|\bError:", log, re.IGNORECASE)
        payload = {
            "renderable": True,
            "renders_svg": True,
            "renders_svg_without_error": not bool(errors),
            "has_error_diagnostic": bool(errors),
            "page_count": page_count,
            "error": log[-4000:] if errors else None,
            "log": log[-4000:],
        }
    except (RuntimeError, ValueError, ET.ParseError) as exc:
        payload = {
            "renderable": False,
            "renders_svg": False,
            "renders_svg_without_error": False,
            "has_error_diagnostic": bool(re.search(r"\[Error\]|\bError:", "\n".join(logs), re.I)),
            "error": str(exc),
            "log": "\n".join(logs)[-4000:],
        }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
