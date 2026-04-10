"""Helpers for computing and plotting token sequence length distributions."""

from __future__ import annotations

import statistics

import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizerFast


def compute_seq_lengths(dataset, tokenizer: PreTrainedTokenizerFast) -> list[int]:
    """Tokenize all transcriptions and return a list of token counts.

    Uses ``select_columns`` to avoid decoding images.
    """
    transcriptions = dataset.select_columns(["transcription"])
    return [
        len(tokenizer(row["transcription"], add_special_tokens=True)["input_ids"])
        for row in transcriptions
    ]


def _percentile(sorted_lengths: list[int], p: float) -> int:
    idx = min(int(len(sorted_lengths) * p / 100), len(sorted_lengths) - 1)
    return sorted_lengths[idx]


def plot_seq_length_distribution(
    seq_lengths: list[int],
    *,
    dataset_name: str = "",
    bins: int = 50,
) -> None:
    """Plot a histogram of sequence lengths with percentile annotations."""
    sorted_lengths = sorted(seq_lengths)

    title = "Sequence length distribution"
    if dataset_name:
        title += f" ({dataset_name}, {len(seq_lengths)} samples)"

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(seq_lengths, bins=bins, edgecolor="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Count")

    for pct, color, style in [(50, "green", "--"), (95, "orange", "--"), (99, "red", "--")]:
        val = _percentile(sorted_lengths, pct)
        ax.axvline(val, color=color, linestyle=style, linewidth=1.5, label=f"p{pct}: {val}")

    ax.legend()
    fig.tight_layout()
    plt.show()

    print(
        f"Min: {sorted_lengths[0]}  p50: {_percentile(sorted_lengths, 50)}  "
        f"p95: {_percentile(sorted_lengths, 95)}  "
        f"p99: {_percentile(sorted_lengths, 99)}  "
        f"Max: {sorted_lengths[-1]}  Mean: {statistics.mean(seq_lengths):.0f}"
    )
