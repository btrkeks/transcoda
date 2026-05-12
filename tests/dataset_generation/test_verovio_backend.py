import sys
import types

import pytest

from scripts.dataset_generation.dataset_generation.image_generation.rendering.verovio_backend import (
    VerovioRenderer,
    count_nr_of_systems_in_svg,
)
from scripts.dataset_generation.dataset_generation.verovio_diagnostics import (
    parse_verovio_diagnostics,
)


def test_count_nr_of_systems_in_svg_prefers_modern_system_class() -> None:
    svg = """
    <svg>
      <g id="s1" class="system"></g>
      <g id="s2" class="foo system bar"></g>
      <g id="m1" class="section systemMilestone"></g>
      <g id="legacy" class="grpSym"></g>
    </svg>
    """

    assert count_nr_of_systems_in_svg(svg) == 2


def test_count_nr_of_systems_in_svg_falls_back_to_legacy_grp_sym() -> None:
    svg = """
    <svg>
      <g class="grpSym"></g>
      <g class="grpSym"></g>
    </svg>
    """

    assert count_nr_of_systems_in_svg(svg) == 2


def test_parse_verovio_inconsistent_rhythm_diagnostic() -> None:
    stderr_text = """Error: Inconsistent rhythm analysis occurring near line 353
Expected durationFromStart to be: 230 but found it to be 229
Line: *-\t*-\t*-\t*-\t*-
"""

    diagnostics = parse_verovio_diagnostics(stderr_text, render_attempt_idx=2)

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.diagnostic_kind == "inconsistent_rhythm_analysis"
    assert diagnostic.render_attempt_idx == 2
    assert diagnostic.near_line == 353
    assert diagnostic.expected_duration_from_start == "230"
    assert diagnostic.found_duration_from_start == "229"
    assert diagnostic.line_text == "*-\t*-\t*-\t*-\t*-"
    assert "Inconsistent rhythm analysis" in diagnostic.raw_message


def test_parse_verovio_generic_error_diagnostic() -> None:
    stderr_text = """Error: Something else happened
Detail: native parser did not like this
"""

    diagnostics = parse_verovio_diagnostics(stderr_text, render_attempt_idx=1)

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.diagnostic_kind == "verovio_error"
    assert diagnostic.render_attempt_idx == 1
    assert diagnostic.raw_message == "Error: Something else happened\nDetail: native parser did not like this"


def test_render_to_svg_applies_options_before_loading_data_on_reused_toolkit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeToolkit:
        def __init__(self) -> None:
            self.options = None
            self.page_count = 0
            self.system_count = 0

        def setOptions(self, options) -> None:
            self.options = dict(options)

        def loadData(self, data: str) -> None:
            if data == "primer":
                self.page_count = 1
                self.system_count = 1
                return
            if data != "target":
                raise AssertionError(f"Unexpected data payload: {data!r}")
            # The target layout only computes correctly if the target options are
            # already active when the score is loaded.
            if self.options == {"profile": "target"}:
                self.page_count = 2
                self.system_count = 3
            else:
                self.page_count = 1
                self.system_count = 5

        def getPageCount(self) -> int:
            return self.page_count

        def renderToSVG(self, pageNo: int = 1) -> str:
            assert pageNo == 1
            systems = "".join(f'<g id="s{i}" class="system"></g>' for i in range(self.system_count))
            return f"<svg>{systems}</svg>"

    fake_verovio = types.SimpleNamespace(
        LOG_ERROR=1,
        enableLog=lambda _level: None,
        toolkit=FakeToolkit,
    )
    monkeypatch.setitem(sys.modules, "verovio", fake_verovio)

    renderer = VerovioRenderer()
    primer_svg, primer_pages = renderer.render_to_svg("primer", {"profile": "primer"})
    target_svg, target_pages = renderer.render_to_svg("target", {"profile": "target"})

    assert primer_pages == 1
    assert count_nr_of_systems_in_svg(primer_svg) == 1
    assert target_pages == 2
    assert count_nr_of_systems_in_svg(target_svg) == 3
