from src.core.kern_concatenation import (
    restore_terminal_spine_count_before_final_barline,
    summarize_spine_topology,
)


def test_summarize_spine_topology_tracks_terminal_split_width():
    text = "\n".join(
        [
            "**kern",
            "*clefG2",
            "=1",
            "4c",
            "*^",
            "*\t*",
            "=2\t=2",
            "4d\t4f",
            "*-\t*-",
        ]
    )

    topology = summarize_spine_topology(text)

    assert topology.initial_spine_count == 1
    assert topology.terminal_spine_count == 2


def test_restore_terminal_spine_count_is_currently_a_no_op_for_mismatched_widths():
    text = "\n".join(
        [
            "**kern",
            "*clefG2",
            "=1",
            "4c",
            "*^",
            "*\t*",
            "=2\t=2",
            "4d\t4f",
            "*-\t*-",
        ]
    )

    assert restore_terminal_spine_count_before_final_barline(text) == text
