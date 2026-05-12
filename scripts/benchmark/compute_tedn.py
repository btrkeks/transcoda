#!/usr/bin/env python
"""Compute TEDn for one predicted MusicXML file against ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, type=Path, help="Predicted MusicXML path.")
    parser.add_argument("--ground-truth", required=True, type=Path, help="Ground-truth MusicXML path.")
    parser.add_argument(
        "--flavor",
        choices=("lmx", "full"),
        default="lmx",
        help="TEDn comparison flavor. Defaults to LEGATO-compatible lmx.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from src.evaluation.tedn import compute_tedn_from_musicxml

    args = parse_args(argv)
    prediction_xml = args.prediction.read_text(encoding="utf-8")
    ground_truth_xml = args.ground_truth.read_text(encoding="utf-8")
    score = compute_tedn_from_musicxml(
        prediction_xml,
        ground_truth_xml,
        flavor=args.flavor,
    )

    if args.json:
        print(json.dumps(asdict(score), sort_keys=True))
    else:
        print(f"TEDn: {score.tedn_percent}%")
        print(f"normalized_edit_cost: {score.normalized_edit_cost}")
        print(f"edit_cost: {score.edit_cost}")
        print(f"gold_cost: {score.gold_cost}")
        print(f"evaluation_time_seconds: {score.evaluation_time_seconds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
