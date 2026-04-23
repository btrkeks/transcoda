import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SH = REPO_ROOT / "train.sh"


def _write_stub_command(bin_dir: Path, name: str) -> None:
    path = bin_dir / name
    path.write_text("#!/usr/bin/env bash\nexit 0\n")
    path.chmod(0o755)


def _write_config(config_path: Path, checkpoint_dir: Path) -> None:
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "train_path": "./data/datasets/train_full",
                    "validation_paths": {"synth": "./data/datasets/validation/synth"},
                    "vocab_name": "bpe3k-splitspaces",
                    "image_width": 1050,
                    "fixed_image_height": 1485,
                    "fixed_image_width": 1050,
                },
                "model": {
                    "d_model": 256,
                    "dim_ff": 256,
                    "num_hidden_layers": 2,
                    "num_attn_heads": 2,
                    "encoder_model_name_or_path": "facebook/convnextv2-tiny-22k-224",
                    "encoder_provider": "transformers",
                    "freeze_encoder_stages": 0,
                    "max_image_height": 1485,
                    "max_image_width": 1050,
                    "max_seq_len": 512,
                    "positional_encoding": "rope",
                },
                "optimizer": {
                    "learning_rate": 0.001,
                    "encoder_lr_factor": 0.3,
                    "weight_decay": 0.01,
                    "warmup_steps": 10,
                    "cosine_eta_min_factor": 0.05,
                    "lr_scheduler": "cosine",
                    "cosine_restart_period": 100,
                    "cosine_restart_mult": 2,
                },
                "checkpoint": {
                    "dirpath": str(checkpoint_dir),
                    "filename": "smt-model",
                    "run_name": "test-run",
                    "monitor": "val/synth/CER",
                    "mode": "min",
                    "save_last": True,
                    "auto_resume": False,
                },
                "generation": {"strategy": "greedy"},
                "training": {
                    "batch_size": 1,
                    "val_batch_size": 1,
                    "num_workers": 0,
                    "max_epochs": 1,
                    "min_steps": 1,
                    "val_check_interval": 1,
                    "limit_val_batches": 1.0,
                    "accumulate_grad_batches": 1,
                    "progress_train_interval_seconds": 30.0,
                    "progress_train_every_n_steps": 1,
                    "progress_val_percent_interval": 50,
                    "progress_enable_ascii_bar": False,
                    "gradient_clip_val": 1.0,
                    "gradient_clip_algorithm": "norm",
                    "log_example_images": False,
                    "compile_model": False,
                    "compile_mode": "default",
                    "compute_omr_ned": False,
                    "use_grammar_constraints": False,
                    "use_spine_structure_constraints": False,
                    "use_rhythm_constraints": False,
                    "grammar_path": "grammars/kern.gbnf",
                    "runaway_guard_enabled": False,
                    "runaway_guard_strictness": "moderate",
                    "label_smoothing": 0.0,
                    "tiered_validation_enabled": False,
                    "full_validation_every_n_steps": 1,
                    "frequent_validation_set_names": [],
                    "frequent_validation_subset_sizes": {},
                    "early_stopping_enabled": False,
                    "early_stopping_patience": 1,
                    "early_stopping_min_delta": 0.0,
                },
            }
        )
    )


def _run_validate_dry_run(tmp_path: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stub_command(bin_dir, "sbatch")

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    return subprocess.run(
        [
            "bash",
            str(TRAIN_SH),
            "validate",
            "--no-doctor",
            "--no-sync",
            "--dry-run",
            *extra_args,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def test_validate_auto_resolves_last_ckpt(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "last.ckpt").write_text("last")
    (checkpoint_dir / "epoch=002.ckpt").write_text("older")

    config_path = tmp_path / "config.json"
    _write_config(config_path, checkpoint_dir)

    result = _run_validate_dry_run(tmp_path, "--config", str(config_path))

    assert "Auto-selected validation checkpoint (auto_validate_last_ckpt):" in result.stdout
    assert f"--checkpoint_path={checkpoint_dir / 'last.ckpt'}" in result.stdout


def test_validate_auto_resolves_newest_ckpt_from_forwarded_dir_override(tmp_path: Path) -> None:
    config_dir = tmp_path / "config-checkpoints"
    config_dir.mkdir()
    _write_config(tmp_path / "config.json", config_dir)

    override_dir = tmp_path / "override-checkpoints"
    override_dir.mkdir()
    older = override_dir / "epoch=001.ckpt"
    newer = override_dir / "epoch=002.ckpt"
    older.write_text("older")
    newer.write_text("newer")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    result = _run_validate_dry_run(
        tmp_path,
        "--config",
        str(tmp_path / "config.json"),
        "--",
        f"--checkpoint.dirpath={override_dir}",
    )

    assert "Auto-selected validation checkpoint (auto_validate_newest_ckpt):" in result.stdout
    assert f"--checkpoint_path={newer}" in result.stdout
