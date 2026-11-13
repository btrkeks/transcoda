"""CLI entrypoint for dataset materialization."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from pipelines.datasets.data_spec import DEFAULT_DATA_SPEC_PATH
from pipelines.datasets.materialization.assemble import (
    DEFAULT_WRITER_BATCH_SIZE,
    materialize_dataset,
)


def main(
    *,
    registry: str = "config/final_dataset_sources.json",
    output_dir: str,
    data_spec_path: str = str(DEFAULT_DATA_SPEC_PATH),
    strict_data_spec: bool = True,
    preview_dir: str | None = None,
    max_synthetic_rows: int | None = None,
    num_workers: int = 1,
    writer_batch_size: int = DEFAULT_WRITER_BATCH_SIZE,
    hf_cache_dir: str | None = None,
    synthetic_augraphy: bool = True,
    synthetic_background: str = "current",
    quiet: bool = False,
) -> dict[str, object]:
    summary = materialize_dataset(
        registry=registry,
        output_dir=output_dir,
        data_spec_path=data_spec_path,
        strict_data_spec=strict_data_spec,
        preview_dir=preview_dir,
        max_synthetic_rows=max_synthetic_rows,
        num_workers=num_workers,
        writer_batch_size=writer_batch_size,
        hf_cache_dir=hf_cache_dir,
        synthetic_augraphy=synthetic_augraphy,
        synthetic_background=synthetic_background,
        quiet=quiet,
    )
    payload = asdict(summary)
    payload["output_dir"] = str(payload["output_dir"])
    return payload


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize a synthetic Hugging Face dataset."
    )
    parser.add_argument("--registry", default="config/final_dataset_sources.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_spec_path", default=str(DEFAULT_DATA_SPEC_PATH))
    parser.add_argument("--strict_data_spec", type=_parse_bool, default=True)
    parser.add_argument("--preview_dir", default=None)
    parser.add_argument("--max_synthetic_rows", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--writer_batch_size", type=int, default=DEFAULT_WRITER_BATCH_SIZE)
    parser.add_argument(
        "--hf_cache_dir",
        default=None,
        help="Parent directory for the temporary Hugging Face generator cache.",
    )
    parser.add_argument("--synthetic_augraphy", type=_parse_bool, default=True)
    parser.add_argument(
        "--synthetic_background",
        choices=("current", "simple", "white"),
        default="current",
        help="Synthetic SVG background mode.",
    )
    parser.add_argument("--quiet", type=_parse_bool, default=False)
    args = parser.parse_args(argv)

    summary = main(
        registry=args.registry,
        output_dir=args.output_dir,
        data_spec_path=args.data_spec_path,
        strict_data_spec=args.strict_data_spec,
        preview_dir=args.preview_dir,
        max_synthetic_rows=args.max_synthetic_rows,
        num_workers=args.num_workers,
        writer_batch_size=args.writer_batch_size,
        hf_cache_dir=args.hf_cache_dir,
        synthetic_augraphy=args.synthetic_augraphy,
        synthetic_background=args.synthetic_background,
        quiet=args.quiet,
    )
    print(json.dumps(summary, sort_keys=True, default=str))
    return 0


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


if __name__ == "__main__":
    raise SystemExit(cli())
