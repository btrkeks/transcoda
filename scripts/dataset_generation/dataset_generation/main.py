"""CLI entrypoint for the rewritten dataset generation pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .executor import run_dataset_generation

DEFAULT_GENERATED_DATASETS_ROOT = Path("data/datasets/generated")


def _validate_dataset_name(name: str) -> str:
    normalized = str(name).strip()
    candidate = Path(normalized)
    if (
        not normalized
        or candidate.is_absolute()
        or len(candidate.parts) != 1
        or candidate.parts[0] in {".", ".."}
    ):
        raise ValueError(
            "dataset name must be a single relative path segment without separators"
        )
    return normalized


def _resolve_output_layout(
    *,
    output_dir: str | None,
    name: str | None,
    artifacts_out_dir: str | None,
) -> tuple[str, str | None]:
    if output_dir is not None and name is not None:
        raise ValueError("Provide either output_dir or name, not both")
    if name is not None:
        dataset_name = _validate_dataset_name(name)
        generated_root = DEFAULT_GENERATED_DATASETS_ROOT
        return str(generated_root / dataset_name), str(generated_root / "_runs")
    if output_dir is None:
        raise ValueError("Either output_dir or name is required")
    return output_dir, artifacts_out_dir


def main(
    *input_dirs: str,
    output_dir: str | None = None,
    name: str | None = None,
    target_samples: int,
    num_workers: int = 1,
    artifacts_out_dir: str | None = None,
    resume_mode: str = "auto",
    base_seed: int = 0,
    max_attempts: int | None = None,
    failure_policy: str = "balanced",
    quarantine_in: str | None = None,
    quarantine_out: str | None = None,
    quiet: bool = False,
) -> dict[str, object]:
    resolved_output_dir, resolved_artifacts_out_dir = _resolve_output_layout(
        output_dir=output_dir,
        name=name,
        artifacts_out_dir=artifacts_out_dir,
    )
    summary = run_dataset_generation(
        input_dirs=input_dirs,
        output_dir=resolved_output_dir,
        target_samples=target_samples,
        num_workers=num_workers,
        artifacts_out_dir=resolved_artifacts_out_dir,
        resume_mode=resume_mode,
        base_seed=base_seed,
        max_attempts=max_attempts,
        failure_policy=failure_policy,
        quarantine_in=quarantine_in,
        quarantine_out=quarantine_out,
        quiet=quiet,
    )
    return asdict(summary)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the production rewrite synthetic training dataset.",
    )
    parser.add_argument(
        "input_dirs",
        nargs="+",
        help="One or more normalized input directories containing .krn files.",
    )
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        "--output_dir",
        help="Destination directory for the Hugging Face dataset.",
    )
    output_group.add_argument(
        "--name",
        help=(
            "Convenience dataset name. Stores output in "
            "data/datasets/generated/<name> and run artifacts in "
            "data/datasets/generated/_runs/<name>."
        ),
    )
    parser.add_argument(
        "--target_samples",
        required=True,
        type=int,
        help="Number of accepted samples to generate.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes to use.",
    )
    parser.add_argument(
        "--artifacts_out_dir",
        default=None,
        help=(
            "Root directory for resumable run artifacts. Defaults to a sibling _runs "
            "directory next to --output_dir, or to data/datasets/generated/_runs when "
            "--name is used."
        ),
    )
    parser.add_argument(
        "--resume_mode",
        choices=("auto", "off", "require"),
        default="auto",
        help="Resume policy for existing incomplete runs.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=0,
        help="Base RNG seed used to derive deterministic sample seeds.",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=None,
        help="Optional cap on attempted plans before stopping.",
    )
    parser.add_argument(
        "--failure_policy",
        choices=("throughput", "balanced", "coverage"),
        default="balanced",
        help="Timeout/retry policy for worker tasks.",
    )
    parser.add_argument(
        "--quarantine_in",
        default=None,
        help="Optional JSON file listing source files to pre-quarantine.",
    )
    parser.add_argument(
        "--quarantine_out",
        default=None,
        help="Optional path to write the accumulated quarantined source list.",
    )
    parser.add_argument(
        "--quiet",
        type=_parse_bool,
        default=False,
        help="Whether to suppress executor progress logging (true/false).",
    )

    args = parser.parse_args(argv)
    summary = main(
        *args.input_dirs,
        output_dir=args.output_dir,
        name=args.name,
        target_samples=args.target_samples,
        num_workers=args.num_workers,
        artifacts_out_dir=args.artifacts_out_dir,
        resume_mode=args.resume_mode,
        base_seed=args.base_seed,
        max_attempts=args.max_attempts,
        failure_policy=args.failure_policy,
        quarantine_in=args.quarantine_in,
        quarantine_out=args.quarantine_out,
        quiet=args.quiet,
    )
    print(json.dumps(summary, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
