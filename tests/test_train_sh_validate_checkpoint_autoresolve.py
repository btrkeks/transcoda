import json
import os
import re
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TRAIN_SH = REPO_ROOT / "train.sh"


def _prepare_workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    train_sh = workspace / "train.sh"
    shutil.copy2(SOURCE_TRAIN_SH, train_sh)
    train_sh.chmod(0o755)
    return workspace, train_sh


def _write_stub_command(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text(f"#!/usr/bin/env bash\n{body}\n")
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


def _run_train_sh(
    workspace: Path,
    train_sh: Path,
    *args: str,
    bin_setup: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = workspace / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in (bin_setup or {}).items():
        _write_stub_command(bin_dir, name, body)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    return subprocess.run(
        ["bash", str(train_sh), *args],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def _read_state_file(workspace: Path) -> str:
    return (workspace / "logs" / "train.sh-last-run.env").read_text()


def test_submit_auto_assigns_run_id_and_records_last_run_state(tmp_path: Path) -> None:
    workspace, train_sh = _prepare_workspace(tmp_path)
    config_path = workspace / "config.json"
    _write_config(config_path, workspace / "weights" / "GrandStaff" / "baseline")

    result = _run_train_sh(
        workspace,
        train_sh,
        "submit",
        "--no-doctor",
        "--no-sync",
        "--config",
        str(config_path),
        bin_setup={"sbatch": 'echo "Submitted batch job 4321"'},
    )

    assert "Submitted batch job 4321" in result.stdout
    assert re.search(r"Assigned run id: test-run-\d{8}-\d{6}", result.stdout)

    state_text = _read_state_file(workspace)
    assert "LAST_RUN_JOB_ID=4321" in state_text
    assert re.search(r"LAST_RUN_ID=test-run-\d{8}-\d{6}", state_text)
    assert "LAST_RUN_SAVE_LAST=true" in state_text

    checkpoint_dir_match = re.search(r"LAST_RUN_CHECKPOINT_DIR=(.+)", state_text)
    assert checkpoint_dir_match is not None
    checkpoint_dir = Path(checkpoint_dir_match.group(1).strip())
    assert checkpoint_dir.is_dir()
    assert checkpoint_dir.parent == workspace / "weights" / "GrandStaff"


def test_validate_prefers_last_submitted_run_and_adds_dependency_when_checkpoint_pending(
    tmp_path: Path,
) -> None:
    workspace, train_sh = _prepare_workspace(tmp_path)
    config_path = workspace / "config.json"
    _write_config(config_path, workspace / "weights" / "GrandStaff" / "baseline")

    _run_train_sh(
        workspace,
        train_sh,
        "submit",
        "--no-doctor",
        "--no-sync",
        "--config",
        str(config_path),
        bin_setup={"sbatch": 'echo "Submitted batch job 4321"'},
    )

    state_text = _read_state_file(workspace)
    checkpoint_dir = Path(re.search(r"LAST_RUN_CHECKPOINT_DIR=(.+)", state_text).group(1).strip())
    run_id = re.search(r"LAST_RUN_ID=(.+)", state_text).group(1).strip()

    result = _run_train_sh(
        workspace,
        train_sh,
        "validate",
        "--no-doctor",
        "--no-sync",
        "--dry-run",
        "--config",
        str(config_path),
        bin_setup={"sbatch": "exit 0"},
    )

    assert f"Selected validation run: {run_id} ({checkpoint_dir})" in result.stdout
    assert "Validation will wait for training job 4321 before starting." in result.stdout
    assert "--dependency=afterok:4321" in result.stdout
    assert f"--checkpoint_path={checkpoint_dir / 'last.ckpt'}" in result.stdout
    assert f"--checkpoint.run_name={run_id}-validate" in result.stdout


def test_validate_auto_resolves_newest_ckpt_from_forwarded_dir_override(tmp_path: Path) -> None:
    workspace, train_sh = _prepare_workspace(tmp_path)
    config_path = workspace / "config.json"
    _write_config(config_path, workspace / "weights" / "GrandStaff" / "baseline")

    override_dir = workspace / "override-checkpoints"
    override_dir.mkdir(parents=True)
    older = override_dir / "epoch=001.ckpt"
    newer = override_dir / "epoch=002.ckpt"
    older.write_text("older")
    newer.write_text("newer")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    result = _run_train_sh(
        workspace,
        train_sh,
        "validate",
        "--no-doctor",
        "--no-sync",
        "--dry-run",
        "--config",
        str(config_path),
        "--",
        f"--checkpoint.dirpath={override_dir}",
        bin_setup={"sbatch": "exit 0"},
    )

    assert f"Selected validation run: {override_dir.name} ({override_dir})" in result.stdout
    assert "Auto-selected validation checkpoint (auto_validate_newest_ckpt):" in result.stdout
    assert f"--checkpoint_path={newer}" in result.stdout
    assert f"--checkpoint.run_name={override_dir.name}-validate" in result.stdout
