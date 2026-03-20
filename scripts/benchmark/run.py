"""CLI entrypoint for running the MusicXML benchmark pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.runner import BenchmarkRunner, BenchmarkRunnerConfig


def parse_auto_int(value: str) -> int | str:
    lowered = value.strip().lower()
    if lowered == "auto":
        return "auto"
    try:
        return int(lowered)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected integer or 'auto', got {value!r}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark evaluation across OMR models.")
    parser.add_argument("--dataset-root", default="data/datasets/benchmark")
    parser.add_argument("--datasets", default="", help="Comma-separated dataset subset")
    parser.add_argument("--models", default="ours,smtpp,legato", help="Comma-separated model subset")
    parser.add_argument("--metrics", default="omr_ned,tedn,cer", help="Comma-separated metric subset")
    parser.add_argument("--output-root", default="outputs/benchmark")
    parser.add_argument("--ours-checkpoint", default="weights/GrandStaff/smt-model.ckpt")
    parser.add_argument("--smtpp-model-id", default="PRAIG/smt-fp-grandstaff")
    parser.add_argument("--smtpp-max-length", type=int, default=None)
    parser.add_argument("--legato-model-id", default="guangyangmusic/legato")
    parser.add_argument("--legato-encoder-path", default=None)
    parser.add_argument("--legato-max-length", type=int, default=2048)
    parser.add_argument("--legato-num-beams", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=parse_auto_int, default="auto")
    parser.add_argument("--metric-workers", type=parse_auto_int, default="auto")
    parser.add_argument("--hum2xml-path", default="hum2xml")
    parser.add_argument("--abc2xml-path", default="abc2xml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-invalid-gold", action="store_true")
    parser.add_argument("--ours-normalize-layout", action="store_true")
    parser.add_argument("--ours-strategy", choices=("greedy", "beam"), default=None)
    parser.add_argument("--ours-num-beams", type=int, default=None)
    parser.add_argument("--ours-repetition-penalty", type=float, default=None)
    parser.add_argument("--ours-loop-recovery", action="store_true")
    parser.add_argument("--ours-loop-recovery-repetition-penalty", type=float, default=1.35)
    parser.add_argument("--disable-constraints", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-warmup-batches", type=int, default=2)
    parser.add_argument("--profile-max-batches", type=int, default=None)
    parser.add_argument("--profile-trace", action="store_true")
    return parser.parse_args()


def split_csv_arg(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    args = parse_args()
    config = BenchmarkRunnerConfig(
        dataset_root=args.dataset_root,
        datasets=split_csv_arg(args.datasets),
        models=split_csv_arg(args.models) or ("ours", "smtpp", "legato"),
        metrics=split_csv_arg(args.metrics) or ("omr_ned", "tedn", "cer"),
        output_root=args.output_root,
        ours_checkpoint=args.ours_checkpoint,
        smtpp_model_id=args.smtpp_model_id,
        smtpp_max_length=args.smtpp_max_length,
        legato_model_id=args.legato_model_id,
        legato_encoder_path=args.legato_encoder_path,
        legato_max_length=args.legato_max_length,
        legato_num_beams=args.legato_num_beams,
        device=args.device,
        limit=args.limit,
        batch_size=args.batch_size,
        metric_workers=args.metric_workers,
        hum2xml_path=args.hum2xml_path,
        abc2xml_path=args.abc2xml_path,
        resume=args.resume,
        skip_inference=args.skip_inference,
        skip_invalid_gold=args.skip_invalid_gold,
        ours_normalize_layout=args.ours_normalize_layout,
        ours_strategy=args.ours_strategy,
        ours_num_beams=args.ours_num_beams,
        ours_repetition_penalty=args.ours_repetition_penalty,
        ours_loop_recovery=args.ours_loop_recovery,
        ours_loop_recovery_repetition_penalty=args.ours_loop_recovery_repetition_penalty,
        disable_constraints=args.disable_constraints,
        profile=args.profile,
        profile_warmup_batches=args.profile_warmup_batches,
        profile_max_batches=args.profile_max_batches,
        profile_trace=args.profile_trace,
    )
    runner = BenchmarkRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
