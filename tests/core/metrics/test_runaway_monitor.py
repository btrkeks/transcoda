from types import SimpleNamespace

import torch

from src.core.metrics.runaway_monitor import (
    CatastrophicLoopConfig,
    RunawayMonitorConfig,
    RunawayMonitorTracker,
    RunawayTextProbe,
    analyze_catastrophic_repetition,
    resolve_runaway_monitor_config,
)


def _tracker(config: RunawayMonitorConfig) -> RunawayMonitorTracker:
    return RunawayMonitorTracker(
        pad_id=0,
        bos_id=1,
        eos_id=2,
        i2w={
            0: "<pad>",
            1: "<bos>",
            2: "<eos>",
            3: "A\n",
            4: "B\n",
            5: "C\n",
            6: "D\n",
        },
        config=config,
    )


def test_length_blowup_flags_runaway():
    tracker = _tracker(
        RunawayMonitorConfig(
            max_len_ratio=1.5,
            repeat_ngram_size=4,
            repeat_ngram_max_occurrences=10,
            max_identical_line_run=20,
            flag_no_eos_at_max_length=True,
        )
    )

    diag = tracker.analyze_sample(
        pred_ids=[1, 3, 4, 5, 6, 2, 0],
        target_ids=[1, 3, 2, 0],
        max_length_cap=16,
    )
    assert diag.length_blowup is True
    assert diag.is_runaway is True


def test_repeat_loop_flags_runaway_by_ngram():
    tracker = _tracker(
        RunawayMonitorConfig(
            max_len_ratio=10.0,
            repeat_ngram_size=2,
            repeat_ngram_max_occurrences=3,
            max_identical_line_run=20,
            flag_no_eos_at_max_length=True,
        )
    )

    diag = tracker.analyze_sample(
        pred_ids=[1, 3, 4, 3, 4, 3, 4, 2, 0],
        target_ids=[1, 3, 4, 2, 0],
        max_length_cap=16,
    )
    assert diag.repeat_loop is True
    assert diag.is_runaway is True


def test_repeat_loop_flags_runaway_by_identical_line_run():
    tracker = _tracker(
        RunawayMonitorConfig(
            max_len_ratio=10.0,
            repeat_ngram_size=4,
            repeat_ngram_max_occurrences=10,
            max_identical_line_run=4,
            flag_no_eos_at_max_length=True,
        )
    )

    diag = tracker.analyze_sample(
        pred_ids=[1, 3, 3, 3, 3, 2, 0],
        target_ids=[1, 3, 2, 0],
        max_length_cap=16,
    )
    assert diag.repeat_loop is True
    assert diag.is_runaway is True


def test_no_eos_at_max_length_flags_runaway():
    tracker = _tracker(
        RunawayMonitorConfig(
            max_len_ratio=10.0,
            repeat_ngram_size=4,
            repeat_ngram_max_occurrences=10,
            max_identical_line_run=20,
            flag_no_eos_at_max_length=True,
        )
    )

    diag = tracker.analyze_sample(
        pred_ids=[1, 3, 4, 0],
        target_ids=[1, 3, 2, 0],
        max_length_cap=2,
    )
    assert diag.max_length_hit is True
    assert diag.no_eos_at_max_length is True
    assert diag.is_runaway is True


def test_no_eos_at_max_length_uses_generated_length_including_bos():
    tracker = _tracker(
        RunawayMonitorConfig(
            max_len_ratio=10.0,
            repeat_ngram_size=4,
            repeat_ngram_max_occurrences=10,
            max_identical_line_run=20,
            flag_no_eos_at_max_length=True,
        )
    )

    # Typical generate() output at max_length includes BOS and no PAD/EOS.
    diag = tracker.analyze_sample(
        pred_ids=[1, 3, 4, 5, 6],
        target_ids=[1, 3, 2, 0, 0],
        max_length_cap=5,
    )
    assert diag.max_length_hit is True
    assert diag.no_eos_at_max_length is True
    assert diag.is_runaway is True


def test_non_runaway_sample_not_flagged():
    tracker = _tracker(
        RunawayMonitorConfig(
            max_len_ratio=2.0,
            repeat_ngram_size=4,
            repeat_ngram_max_occurrences=10,
            max_identical_line_run=20,
            flag_no_eos_at_max_length=True,
        )
    )

    diag = tracker.analyze_sample(
        pred_ids=[1, 3, 4, 2, 0],
        target_ids=[1, 3, 4, 2, 0],
        max_length_cap=16,
    )
    assert diag.is_runaway is False


def test_presets_resolve_expected_thresholds():
    cfg = resolve_runaway_monitor_config(
        SimpleNamespace(
            runaway_monitor_strictness="moderate",
            runaway_monitor_max_len_ratio=None,
            runaway_monitor_repeat_ngram_size=None,
            runaway_monitor_repeat_ngram_max_occurrences=None,
            runaway_monitor_max_identical_line_run=None,
            runaway_monitor_flag_no_eos_at_max_length=True,
        )
    )
    assert cfg.max_len_ratio == 1.8
    assert cfg.repeat_ngram_size == 4
    assert cfg.repeat_ngram_max_occurrences == 5
    assert cfg.max_identical_line_run == 6
    assert cfg.flag_no_eos_at_max_length is True


def test_overrides_replace_preset_values():
    cfg = resolve_runaway_monitor_config(
        SimpleNamespace(
            runaway_monitor_strictness="moderate",
            runaway_monitor_max_len_ratio=2.2,
            runaway_monitor_repeat_ngram_size=5,
            runaway_monitor_repeat_ngram_max_occurrences=9,
            runaway_monitor_max_identical_line_run=7,
            runaway_monitor_flag_no_eos_at_max_length=False,
        )
    )
    assert cfg.max_len_ratio == 2.2
    assert cfg.repeat_ngram_size == 5
    assert cfg.repeat_ngram_max_occurrences == 9
    assert cfg.max_identical_line_run == 7
    assert cfg.flag_no_eos_at_max_length is False


def test_compute_returns_expected_rates_and_p95():
    tracker = _tracker(
        RunawayMonitorConfig(
            max_len_ratio=1.6,
            repeat_ngram_size=2,
            repeat_ngram_max_occurrences=3,
            max_identical_line_run=5,
            flag_no_eos_at_max_length=True,
        )
    )

    preds = torch.tensor(
        [
            [1, 3, 4, 5, 2, 0],  # length blowup vs short target
            [1, 3, 4, 2, 0, 0],  # normal
            [1, 3, 4, 3, 4, 0],  # no eos + max_length hit + repetition ngram
        ]
    )
    targets = torch.tensor(
        [
            [1, 3, 2, 0, 0, 0],
            [1, 3, 4, 2, 0, 0],
            [1, 3, 4, 2, 0, 0],
        ]
    )

    tracker.update_batch(preds, targets, max_length_cap=4)
    computed = tracker.compute()

    assert computed["runaway_samples"] == 3
    assert computed["runaway_count"] == 2
    assert computed["runaway_rate"] > 60.0
    assert computed["runaway_len_ratio_p95"] >= 1.5


def test_runaway_text_probe_exposes_repeat_loop_reason() -> None:
    probe = RunawayTextProbe(
        repeat_ngram_size=2,
        repeat_ngram_max_occurrences=3,
        max_identical_line_run=20,
    )

    diag = probe.analyze_text("A\nB\nA\nB\nA\nB\n")

    assert diag.repeat_loop is True
    assert diag.repeat_loop_reason == "repeated_ngram"
    assert diag.max_repeated_ngram_occurrences == 3


def test_catastrophic_repetition_flags_identical_line_run_at_threshold() -> None:
    diag = analyze_catastrophic_repetition(".\t20gg\n" * 8)

    assert diag.repeat_loop is True
    assert diag.repeat_loop_reason == "identical_line_run"
    assert diag.max_identical_line_run == 8


def test_catastrophic_repetition_flags_high_coverage_repeated_ngram() -> None:
    text = ("=\t=\n1r\t1r\n=\t=\n1r\t1r\n") * 12

    diag = analyze_catastrophic_repetition(text)

    assert diag.repeat_loop is True
    assert diag.repeat_loop_reason == "repeated_ngram"
    assert diag.max_repeated_ngram_occurrences >= 12
    assert diag.repeated_ngram_line_coverage >= 0.33


def test_catastrophic_repetition_does_not_flag_gold_like_low_coverage_alternation() -> None:
    lines = []
    motif = ["=\t=", "1r\t1r", "=\t=", "1r\t1r"]
    for idx in range(12):
        lines.extend(motif)
        lines.extend([f"fill-{idx}-a\tA", f"fill-{idx}-b\tA", f"fill-{idx}-c\tA", f"fill-{idx}-d\tA"])
    text = "\n".join(lines) + "\n"

    diag = analyze_catastrophic_repetition(
        text,
        config=CatastrophicLoopConfig(
            max_identical_line_run=8,
            repeated_ngram_size=4,
            min_repeated_ngram_occurrences=12,
            min_repeated_ngram_line_coverage=0.33,
        ),
    )

    assert diag.repeat_loop is False
    assert diag.max_repeated_ngram_occurrences >= 12
    assert diag.repeated_ngram_line_coverage < 0.33
