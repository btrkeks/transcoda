"""Helpers for loading and plotting generated-run histograms in notebooks."""

from __future__ import annotations

from collections import Counter
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

HISTOGRAM_BUCKET_WIDTHS = {
    "bottom_whitespace_px_histogram": 50,
    "content_height_px_histogram": 25,
}


def _extract_snapshot(info: dict) -> dict:
    snapshot = info.get("snapshot")
    if not isinstance(snapshot, dict):
        snapshot = info.get("finalization", {}).get("snapshot", {})
    return snapshot if isinstance(snapshot, dict) else {}


def _run_dir_sort_key(path: Path) -> tuple[tuple[int, ...], str]:
    numeric_parts = tuple(int(part) for part in re.findall(r"\d+", path.name))
    return numeric_parts, path.name


def load_latest_run_info(dataset_name: str) -> tuple[dict, Path]:
    runs_root = PROJECT_ROOT / "data" / "datasets" / "generated" / "_runs" / dataset_name
    if not runs_root.exists():
        raise FileNotFoundError(f"Run artifacts directory not found: {runs_root}")

    latest_run_path = runs_root / "latest_run.json"
    if latest_run_path.exists():
        latest_run = json.loads(latest_run_path.read_text())
        info_path = Path(latest_run["info_path"])
        if info_path.exists():
            return json.loads(info_path.read_text()), info_path

    run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {runs_root}")

    latest_run_dir = max(run_dirs, key=_run_dir_sort_key)
    info_path = latest_run_dir / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Run info file not found: {info_path}")

    return json.loads(info_path.read_text()), info_path


def collect_histograms(info: dict) -> dict[str, dict[int, int]]:
    snapshot = _extract_snapshot(info)

    histograms: dict[str, dict[int, int]] = {}
    for key, value in snapshot.items():
        if not key.endswith("_histogram") or not isinstance(value, dict) or not value:
            continue
        # Nested histograms (e.g. accepted_system_histogram is keyed by
        # spine class → system count → count) can't be parsed as flat
        # {int: int}; store an empty placeholder so the key is still visible
        # to plot_histograms (which handles it via the heatmap branch).
        if any(isinstance(v, dict) for v in value.values()):
            histograms[key] = {}
            continue
        histograms[key] = {int(hist_key): int(count) for hist_key, count in value.items()}
    return histograms


def _bucket_histogram(histogram: dict[int, int], bucket_width: int) -> dict[int, int]:
    bucketed: dict[int, int] = {}
    for value, count in histogram.items():
        bucket_start = (value // bucket_width) * bucket_width
        bucketed[bucket_start] = bucketed.get(bucket_start, 0) + count
    return dict(sorted(bucketed.items()))


def _plot_spine_system_heatmap(ax, dataset) -> None:
    pair_counts = Counter(
        (int(row["initial_kern_spine_count"]), int(row["svg_system_count"]))
        for row in dataset
    )

    spine_values = list(range(1, 5))
    system_values = list(range(1, 8))
    grid = np.array(
        [
            [pair_counts.get((spine, system), 0) for spine in spine_values]
            for system in system_values
        ]
    )

    image = ax.imshow(grid, origin="lower", cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(spine_values)), labels=spine_values)
    ax.set_yticks(range(len(system_values)), labels=system_values)
    ax.set_xlabel("initial_kern_spine_count")
    ax.set_ylabel("svg_system_count")
    ax.set_title("Initial kern spine count vs SVG system count")

    max_count = int(grid.max()) if grid.size else 0
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            value = int(grid[y, x])
            text_color = "white" if max_count and value > 0.6 * max_count else "black"
            ax.text(x, y, str(value), ha="center", va="center", color=text_color)

    plt.colorbar(image, ax=ax, label="Count")


def _collect_rejection_summary(info: dict) -> dict | None:
    snapshot = _extract_snapshot(info)
    accepted_samples = int(snapshot.get("accepted_samples", 0) or 0)
    rejected_samples = int(snapshot.get("rejected_samples", 0) or 0)

    raw_failure_reason_counts = snapshot.get("failure_reason_counts", {})
    failure_reason_counts = (
        {str(name): int(count) for name, count in raw_failure_reason_counts.items()}
        if isinstance(raw_failure_reason_counts, dict)
        else {}
    )
    attempted_samples = accepted_samples + sum(failure_reason_counts.values())
    resolved_samples = accepted_samples + rejected_samples

    raw_augmentation_outcome_counts = snapshot.get("augmentation_outcome_counts", {})
    augmentation_outcome_counts = (
        {str(name): int(count) for name, count in raw_augmentation_outcome_counts.items()}
        if isinstance(raw_augmentation_outcome_counts, dict)
        else {}
    )
    clean_gate_rejected = int(augmentation_outcome_counts.get("clean_gate_rejected", 0))

    rates: list[dict[str, float | int | str]] = []
    if attempted_samples > 0 and rejected_samples > 0:
        rates.append(
            {
                "label": "Hard rejects / attempts",
                "numerator": rejected_samples,
                "denominator": attempted_samples,
                "rate": rejected_samples / attempted_samples,
            }
        )
    if resolved_samples > 0 and rejected_samples > 0:
        rates.append(
            {
                "label": "Hard rejects / resolved",
                "numerator": rejected_samples,
                "denominator": resolved_samples,
                "rate": rejected_samples / resolved_samples,
            }
        )
    if accepted_samples > 0 and clean_gate_rejected > 0:
        rates.append(
            {
                "label": "Clean-gate rejects / accepted",
                "numerator": clean_gate_rejected,
                "denominator": accepted_samples,
                "rate": clean_gate_rejected / accepted_samples,
            }
        )

    if not rates and not failure_reason_counts:
        return None

    return {
        "accepted_samples": accepted_samples,
        "attempted_samples": attempted_samples,
        "rejected_samples": rejected_samples,
        "failure_reason_counts": failure_reason_counts,
        "rates": rates,
    }


def _plot_rejection_summary(ax, summary: dict) -> None:
    rates = summary.get("rates", [])
    labels = [str(item["label"]) for item in rates]
    percentages = [float(item["rate"]) * 100.0 for item in rates]
    positions = np.arange(len(rates))

    colors = ["#c2410c", "#ea580c", "#2563eb"]
    ax.barh(positions, percentages, color=colors[: len(rates)])
    ax.set_yticks(positions, labels=labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Rate (%)")
    ax.set_title("Rejection rate summary")
    ax.grid(axis="x", alpha=0.25)

    for pos, pct, item in zip(positions, percentages, rates):
        numerator = int(item["numerator"])
        denominator = int(item["denominator"])
        label = f"{numerator}/{denominator} ({pct:.1f}%)"
        text_x = min(pct + 1.0, 99.0)
        ax.text(text_x, pos, label, va="center", ha="left", fontsize=9)

    failure_reason_counts = summary.get("failure_reason_counts", {})
    if failure_reason_counts:
        reason_lines = []
        for reason, count in sorted(
            failure_reason_counts.items(), key=lambda item: (-int(item[1]), str(item[0]))
        )[:4]:
            suffix = " (not a rejection)" if reason == "discarded_after_target" else ""
            reason_lines.append(f"{reason}: {int(count)}{suffix}")
        ax.text(
            1.0,
            0.02,
            "\n".join(reason_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
        )


def plot_histograms(
    histograms: dict[str, dict[int, int]],
    *,
    dataset_name: str,
    info_path: Path,
    dataset=None,
) -> None:
    info = json.loads(info_path.read_text(encoding="utf-8"))
    rejection_summary = _collect_rejection_summary(info)

    panels: list[tuple[str, str, dict]] = [
        ("histogram", name, histogram) for name, histogram in sorted(histograms.items())
    ]
    if rejection_summary is not None:
        panels.append(("summary", "rejection_rate_summary", rejection_summary))

    if not panels:
        print(f"No histograms found in {info_path}")
        return

    columns = 2
    rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16, 4 * rows))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, (panel_type, name, payload) in zip(axes, panels):
        if panel_type == "summary":
            _plot_rejection_summary(ax, payload)
            continue

        histogram = payload
        if name == "accepted_system_histogram" and dataset is not None:
            _plot_spine_system_heatmap(ax, dataset)
            continue
        bucket_width = HISTOGRAM_BUCKET_WIDTHS.get(name)
        plot_histogram = _bucket_histogram(histogram, bucket_width) if bucket_width else histogram
        items = sorted(plot_histogram.items())
        x_values = [value for value, _ in items]
        y_values = [count for _, count in items]
        if bucket_width:
            ax.bar(x_values, y_values, width=bucket_width * 0.9, align="edge")
            ax.set_title(f"{name.replace('_', ' ')} ({bucket_width}px buckets)")
            ax.set_xlabel("Value bucket start (px)")
        else:
            ax.bar(x_values, y_values, width=0.9)
            ax.set_title(name.replace("_", " "))
            ax.set_xlabel("Value")
        if name == "bottom_whitespace_px_histogram":
            ax.axvline(
                80,
                color="crimson",
                linestyle="--",
                linewidth=2,
                label="polish dataset median",
            )
            ax.legend()
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(
        f"Latest generated-run histograms for {dataset_name} ({info_path.parent.name})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()
