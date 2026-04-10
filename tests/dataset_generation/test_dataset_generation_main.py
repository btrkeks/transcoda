import subprocess
import sys
from pathlib import Path

import pytest

from scripts.dataset_generation.dataset_generation import main as main_module


def test_main_uses_named_generated_dataset_layout(monkeypatch):
    captured = {}

    def fake_run_dataset_generation(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(main_module, "run_dataset_generation", fake_run_dataset_generation)
    monkeypatch.setattr(main_module, "asdict", lambda summary: {"ok": True})

    result = main_module.main(
        "data/interim/train/pdmx/3_normalized",
        name="smoke_v1",
        target_samples=4,
        quiet=True,
    )

    assert result == {"ok": True}
    assert captured["output_dir"] == "data/datasets/generated/smoke_v1"
    assert captured["artifacts_out_dir"] == "data/datasets/generated/_runs"


def test_main_keeps_explicit_output_dir_and_custom_artifacts(monkeypatch):
    captured = {}

    def fake_run_dataset_generation(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(main_module, "run_dataset_generation", fake_run_dataset_generation)
    monkeypatch.setattr(main_module, "asdict", lambda summary: {"ok": True})

    result = main_module.main(
        "data/interim/train/pdmx/3_normalized",
        output_dir="/tmp/custom-dataset",
        artifacts_out_dir="/tmp/custom-runs",
        target_samples=4,
        quiet=True,
    )

    assert result == {"ok": True}
    assert captured["output_dir"] == "/tmp/custom-dataset"
    assert captured["artifacts_out_dir"] == "/tmp/custom-runs"


def test_main_defaults_explicit_output_dir_to_sibling_runs(monkeypatch):
    captured = {}

    def fake_run_dataset_generation(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(main_module, "run_dataset_generation", fake_run_dataset_generation)
    monkeypatch.setattr(main_module, "asdict", lambda summary: {"ok": True})

    result = main_module.main(
        "data/interim/train/pdmx/3_normalized",
        output_dir="/tmp/custom-dataset",
        target_samples=4,
        quiet=True,
    )

    assert result == {"ok": True}
    assert captured["output_dir"] == "/tmp/custom-dataset"
    assert captured["artifacts_out_dir"] is None


def test_main_passes_capture_verovio_diagnostics_flag(monkeypatch):
    captured = {}

    def fake_run_dataset_generation(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(main_module, "run_dataset_generation", fake_run_dataset_generation)
    monkeypatch.setattr(main_module, "asdict", lambda summary: {"ok": True})

    result = main_module.main(
        "data/interim/train/pdmx/3_normalized",
        name="smoke_v1",
        target_samples=4,
        quiet=True,
        capture_verovio_diagnostics=False,
    )

    assert result == {"ok": True}
    assert captured["capture_verovio_diagnostics"] is False


def test_main_rejects_name_and_output_dir_together():
    with pytest.raises(ValueError, match="either output_dir or name"):
        main_module.main(
            "data/interim/train/pdmx/3_normalized",
            name="smoke_v1",
            output_dir="/tmp/custom-dataset",
            target_samples=4,
            quiet=True,
        )


def test_main_rejects_invalid_dataset_name():
    with pytest.raises(ValueError, match="single relative path segment"):
        main_module.main(
            "data/interim/train/pdmx/3_normalized",
            name="nested/smoke_v1",
            target_samples=4,
            quiet=True,
        )


def test_cli_rejects_name_and_output_dir_together():
    with pytest.raises(SystemExit, match="2"):
        main_module.cli(
            [
                "data/interim/train/pdmx/3_normalized",
                "--name",
                "smoke_v1",
                "--output_dir",
                "/tmp/custom-dataset",
                "--target_samples",
                "4",
                "--quiet",
                "true",
            ]
        )


@pytest.mark.parametrize(
    "flag_args",
    [
        ["--system_balance_mode", "length_proxy"],
        ["--system_balance_spec", "/tmp/spec.json"],
        ["--tokenizer_path", "vocab/bpe3k-splitspaces"],
    ],
)
def test_cli_rejects_removed_balance_flags(flag_args):
    with pytest.raises(SystemExit, match="2"):
        main_module.cli(
            [
                "data/interim/train/pdmx/3_normalized",
                "--name",
                "smoke_v1",
                "--target_samples",
                "4",
                *flag_args,
                "--quiet",
                "true",
            ]
        )


def test_cli_parses_capture_verovio_diagnostics(monkeypatch):
    captured = {}

    def fake_main(*input_dirs, **kwargs):
        captured["input_dirs"] = input_dirs
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(main_module, "main", fake_main)

    result = main_module.cli(
        [
            "data/interim/train/pdmx/3_normalized",
            "--name",
            "smoke_v1",
            "--target_samples",
            "4",
            "--capture_verovio_diagnostics",
            "false",
            "--quiet",
            "true",
        ]
    )

    assert result == 0
    assert captured["capture_verovio_diagnostics"] is False


def test_python_module_help_smoke():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.dataset_generation.dataset_generation.main",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--target_samples" in result.stdout
