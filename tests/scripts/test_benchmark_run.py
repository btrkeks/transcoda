from __future__ import annotations

import sys

from scripts.benchmark import run as benchmark_run


def test_parse_args_defaults_include_cer(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["benchmark-run"])

    args = benchmark_run.parse_args()

    assert args.metrics == "omr_ned,tedn,cer"
    assert benchmark_run.split_csv_arg(args.metrics) == ("omr_ned", "tedn", "cer")


def test_parse_args_accepts_explicit_cer_metric(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["benchmark-run", "--metrics", "cer"])

    args = benchmark_run.parse_args()

    assert benchmark_run.split_csv_arg(args.metrics) == ("cer",)


def test_parse_args_accepts_legato_encoder_path(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark-run",
            "--legato-model-id",
            "/workspace/hf/legato",
            "--legato-encoder-path",
            "/workspace/hf/llama32-11b-vision",
        ],
    )

    args = benchmark_run.parse_args()

    assert args.legato_model_id == "/workspace/hf/legato"
    assert args.legato_encoder_path == "/workspace/hf/llama32-11b-vision"
