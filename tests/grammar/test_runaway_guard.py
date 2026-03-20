from types import SimpleNamespace

import torch

from src.grammar.runaway_guard import (
    RunawayBreakerLogitsProcessor,
    RunawayGuardConfig,
    resolve_runaway_guard_config,
)


def _make_config(**kwargs) -> RunawayGuardConfig:
    base = {
        "max_same_control_token": 8,
        "max_control_lines_streak": 6,
        "max_spine_splits": 64,
        "max_spine_merges": 96,
        "max_ottava_markers": 16,
        "max_tuplet_markers": 12,
        "max_tremolo_markers": 12,
    }
    base.update(kwargs)
    return RunawayGuardConfig(**base)


def _make_vocab() -> dict[int, str]:
    return {
        0: "<pad>",
        1: "<bos>",
        2: "<eos>",
        3: "*X8va",
        4: "\n",
        5: "*^",
        6: "*v",
        7: "*",
        8: "*tremolo",
        9: "*Xtremolo",
        10: "*Xtuplet",
        11: "16c#",
        12: "\t",
    }


def test_resolve_defaults_by_strictness():
    for strictness, expected_same in (("lenient", 12), ("moderate", 8), ("strict", 5)):
        cfg = resolve_runaway_guard_config(
            SimpleNamespace(
                runaway_guard_strictness=strictness,
                runaway_guard_max_same_control_token=None,
                runaway_guard_max_control_lines_streak=None,
                runaway_guard_max_spine_splits=None,
                runaway_guard_max_spine_merges=None,
                runaway_guard_max_ottava_markers=None,
                runaway_guard_max_tuplet_markers=None,
                runaway_guard_max_tremolo_markers=None,
            )
        )
        assert cfg.max_same_control_token == expected_same


def test_overrides_take_precedence():
    cfg = resolve_runaway_guard_config(
        SimpleNamespace(
            runaway_guard_strictness="moderate",
            runaway_guard_max_same_control_token=3,
            runaway_guard_max_control_lines_streak=2,
            runaway_guard_max_spine_splits=7,
            runaway_guard_max_spine_merges=11,
            runaway_guard_max_ottava_markers=13,
            runaway_guard_max_tuplet_markers=17,
            runaway_guard_max_tremolo_markers=19,
        )
    )
    assert cfg.max_same_control_token == 3
    assert cfg.max_control_lines_streak == 2
    assert cfg.max_spine_splits == 7
    assert cfg.max_spine_merges == 11
    assert cfg.max_ottava_markers == 13
    assert cfg.max_tuplet_markers == 17
    assert cfg.max_tremolo_markers == 19


def test_blocks_repeated_same_control_token():
    processor = RunawayBreakerLogitsProcessor(
        tokenizer_i2w=_make_vocab(),
        bos_token_id=1,
        eos_token_id=2,
        config=_make_config(max_same_control_token=2),
    )
    input_ids = torch.tensor([[1, 3, 3]], dtype=torch.long)
    scores = torch.zeros((1, 13), dtype=torch.float32)

    out = processor(input_ids, scores)
    assert torch.isneginf(out[0, 3])


def test_blocks_class_when_cap_reached():
    processor = RunawayBreakerLogitsProcessor(
        tokenizer_i2w=_make_vocab(),
        bos_token_id=1,
        eos_token_id=2,
        config=_make_config(max_spine_splits=1),
    )
    # "*^\n" in history completes one split line.
    input_ids = torch.tensor([[1, 5, 4]], dtype=torch.long)
    scores = torch.zeros((1, 13), dtype=torch.float32)

    out = processor(input_ids, scores)
    assert torch.isneginf(out[0, 5])
    assert not torch.isneginf(out[0, 11])


def test_blocks_controls_after_control_line_streak():
    processor = RunawayBreakerLogitsProcessor(
        tokenizer_i2w=_make_vocab(),
        bos_token_id=1,
        eos_token_id=2,
        config=_make_config(max_control_lines_streak=2),
    )
    # Two control-only lines: "*^\n*^\n"
    input_ids = torch.tensor([[1, 5, 4, 5, 4]], dtype=torch.long)
    scores = torch.zeros((1, 13), dtype=torch.float32)

    out = processor(input_ids, scores)
    assert torch.isneginf(out[0, 5])


def test_no_crash_on_unexpected_tokenization():
    vocab = _make_vocab()
    vocab[13] = "<<<weird-piece>>>"
    processor = RunawayBreakerLogitsProcessor(
        tokenizer_i2w=vocab,
        bos_token_id=1,
        eos_token_id=2,
        config=_make_config(),
    )

    input_ids = torch.tensor([[1, 13, 11]], dtype=torch.long)
    scores = torch.zeros((1, 14), dtype=torch.float32)
    out = processor(input_ids, scores)

    assert out.shape == (1, 14)


def test_never_masks_eos():
    processor = RunawayBreakerLogitsProcessor(
        tokenizer_i2w=_make_vocab(),
        bos_token_id=1,
        eos_token_id=2,
        config=_make_config(max_same_control_token=1),
    )
    input_ids = torch.tensor([[1, 3]], dtype=torch.long)
    scores = torch.zeros((1, 13), dtype=torch.float32)

    out = processor(input_ids, scores)
    assert not torch.isneginf(out[0, 2])

