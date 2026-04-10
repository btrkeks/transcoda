from scripts.dataset_generation.dataset_generation.image_generation.rendering.verovio_backend import (
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
