from scripts.dataset_generation.dataset_generation.image_generation.rendering.verovio_backend import (
    count_nr_of_systems_in_svg,
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
