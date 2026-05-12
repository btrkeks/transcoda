from __future__ import annotations

import torch

from src.grammar.rhythm_rule import RhythmRule
from src.grammar.spine_structure_rule import SpineStructureRule
from src.grammar.stateful_kern_logits_processor import StatefulKernLogitsProcessor


def _scores_for_prefix(
    *,
    prefix_ids: list[int],
    rule_factories: list[type],
) -> torch.Tensor:
    processor = StatefulKernLogitsProcessor(
        i2w={
            0: "<pad>",
            1: "<bos>",
            2: "<eos>",
            3: "\t",
            4: "\n",
            5: "**kern",
            6: "*M4/4",
            7: "4c",
            8: "4g",
            9: "2g",
        },
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        rule_factories=rule_factories,
    )
    scores = None
    for end in range(1, len(prefix_ids) + 1):
        input_ids = torch.tensor([prefix_ids[:end]], dtype=torch.long)
        scores = processor(input_ids, torch.zeros((1, 10), dtype=torch.float32))
    assert scores is not None
    return scores[0]


def test_rhythm_rule_masks_overfull_token_that_spine_only_allows() -> None:
    prefix_ids = [
        1,
        5,
        4,
        6,
        4,
        7,
        4,
        7,
        4,
        7,
        4,
    ]

    spine_only_scores = _scores_for_prefix(
        prefix_ids=prefix_ids,
        rule_factories=[SpineStructureRule],
    )
    combined_scores = _scores_for_prefix(
        prefix_ids=prefix_ids,
        rule_factories=[SpineStructureRule, RhythmRule],
    )

    assert spine_only_scores[9] != float("-inf")
    assert combined_scores[9] == float("-inf")
    assert combined_scores[8] != float("-inf")


def test_combined_processor_handles_body_only_spine_ops_without_crashing() -> None:
    for prefix_ids in ([1, 5], [1, 5, 3], [1, 5, 3, 6]):
        processor = StatefulKernLogitsProcessor(
            i2w={
                0: "<pad>",
                1: "<bos>",
                2: "<eos>",
                3: "\t",
                4: "\n",
                5: "*",
                6: "*^",
                7: "4c",
            },
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            rule_factories=[SpineStructureRule, RhythmRule],
        )
        input_ids = torch.tensor([prefix_ids], dtype=torch.long)
        scores = processor(input_ids, torch.zeros((1, 8), dtype=torch.float32))
        assert scores.shape == (1, 8)


def test_partial_time_signature_preview_masks_newline_and_eos_without_crashing() -> None:
    for partial_token_id in (6, 7):
        processor = StatefulKernLogitsProcessor(
            i2w={
                0: "<pad>",
                1: "<bos>",
                2: "<eos>",
                3: "\t",
                4: "\n",
                5: "*M9/16",
                6: "*M9",
                7: "*M9/",
            },
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            rule_factories=[RhythmRule],
        )
        input_ids = torch.tensor([[1, 5, 3, partial_token_id]], dtype=torch.long)

        scores = processor(input_ids, torch.zeros((1, 8), dtype=torch.float32))

        assert scores.shape == (1, 8)
        assert scores[0, 4] == float("-inf")
        assert scores[0, 2] == float("-inf")


def test_grace_alignment_line_keeps_newline_unmasked() -> None:
    processor = StatefulKernLogitsProcessor(
        i2w={
            0: "<pad>",
            1: "<bos>",
            2: "<eos>",
            3: "\t",
            4: "\n",
            5: "**kern",
            6: "*M4/4",
            7: ".",
            8: "bq",
        },
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        rule_factories=[RhythmRule],
    )
    scores = None
    prefix_ids = [1, 5, 3, 5, 4, 6, 3, 6, 4, 7, 3, 8]
    for end in range(1, len(prefix_ids) + 1):
        input_ids = torch.tensor([prefix_ids[:end]], dtype=torch.long)
        scores = processor(input_ids, torch.zeros((1, 9), dtype=torch.float32))

    assert scores is not None
    assert scores.shape == (1, 9)
    assert scores[0, 4] != float("-inf")


def test_late_timed_line_after_grace_only_sequence_keeps_newline_unmasked() -> None:
    processor = StatefulKernLogitsProcessor(
        i2w={
            0: "<pad>",
            1: "<bos>",
            2: "<eos>",
            3: "\t",
            4: "\n",
            5: "**kern",
            6: "*M4/4",
            7: "8d",
            8: "16ff#L",
            9: ".",
            10: "32ddL",
            11: "32bJJJ",
            12: "8e",
            13: "8cc#L",
            14: "bq",
            15: "8EJ",
            16: "8bJ",
            17: "aqLLL",
            18: "bqJJJ",
            19: "4AA",
            20: "8a",
            21: "ccc#qLLL",
            22: "dddqJJJ",
            23: "8eee",
        },
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        rule_factories=[RhythmRule],
    )
    scores = None
    prefix_ids = [
        1,
        5,
        3,
        5,
        4,
        6,
        3,
        6,
        4,
        7,
        3,
        8,
        4,
        9,
        3,
        10,
        4,
        9,
        3,
        11,
        4,
        12,
        3,
        13,
        4,
        9,
        3,
        14,
        4,
        15,
        3,
        16,
        4,
        9,
        3,
        17,
        4,
        9,
        3,
        18,
        4,
        19,
        3,
        20,
        4,
        9,
        3,
        21,
        4,
        9,
        3,
        22,
        4,
        9,
        3,
        23,
    ]
    for end in range(1, len(prefix_ids) + 1):
        input_ids = torch.tensor([prefix_ids[:end]], dtype=torch.long)
        scores = processor(input_ids, torch.zeros((1, 24), dtype=torch.float32))

    assert scores is not None
    assert scores.shape == (1, 24)
    assert scores[0, 4] != float("-inf")


def test_stateful_processor_exposes_stats() -> None:
    processor = StatefulKernLogitsProcessor(
        i2w={
            0: "<pad>",
            1: "<bos>",
            2: "<eos>",
            3: "\t",
            4: "\n",
            5: "**kern",
            6: "4c",
        },
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        rule_factories=[SpineStructureRule],
        collect_stats=True,
    )

    for prefix_ids in ([1], [1, 5], [1, 5, 4], [1, 5, 4, 6]):
        input_ids = torch.tensor([prefix_ids], dtype=torch.long)
        processor(input_ids, torch.zeros((1, 7), dtype=torch.float32))

    stats = processor.stats()

    assert stats["calls"] == 4
    assert stats["rows_processed"] == 4
    assert stats["total_ms"] >= 0.0
