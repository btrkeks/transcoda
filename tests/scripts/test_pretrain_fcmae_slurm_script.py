import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "pretrain_fcmae.sh"


RUN_ID_RE = r"fcmae-real-scan-\d{8}-\d{6}"


def test_pretrain_fcmae_help_lists_required_commands() -> None:
    result = subprocess.run(
        [str(SCRIPT), "help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "./pretrain_fcmae.sh submit" in result.stdout
    assert "./pretrain_fcmae.sh local" in result.stdout
    assert "./pretrain_fcmae.sh export CHECKPOINT_PATH OUTPUT_DIR" in result.stdout


def test_pretrain_fcmae_submit_dry_run_renders_bash_lc_and_bare_overrides() -> None:
    result = subprocess.run(
        [
            str(SCRIPT),
            "submit",
            "--dry-run",
            "--no-sync",
            "--job-name",
            "fcmae-test",
            "--resume",
            "weights/fcmae/last.ckpt",
            "--",
            "--training.max_steps=10",
            "logging.wandb_enabled=true",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--job-name=fcmae-test" in result.stdout
    assert "bash -lc" in result.stdout
    assert "source .venv/bin/activate" in result.stdout
    assert "python -m scripts.pretrain_fcmae" in result.stdout
    assert "training.max_steps=10" in result.stdout
    assert "logging.wandb_enabled=true" in result.stdout
    assert "training.resume_from_checkpoint=weights/fcmae/last.ckpt" in result.stdout
    assert re.search(
        rf"checkpoint\.dirpath={re.escape(str(ROOT))}/weights/{RUN_ID_RE}",
        result.stdout,
    )


def test_pretrain_fcmae_submit_dry_run_uses_real_scan_defaults() -> None:
    config = json.loads((ROOT / "config/pretrain_fcmae_base.json").read_text())

    assert config["data"]["image_dir"] == "data/fcmae_images"
    assert config["data"]["manifest_path"] is None
    assert config["checkpoint"]["dirpath"] == "weights/fcmae-real-scans"
    assert config["logging"]["wandb_enabled"] is True
    assert config["logging"]["run_name"] == "fcmae-real-scan"

    result = subprocess.run(
        [
            str(SCRIPT),
            "submit",
            "--dry-run",
            "--no-sync",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--job-name=fcmae-pretrain" in result.stdout
    assert "config/pretrain_fcmae_base.json" in result.stdout
    match = re.search(
        rf"checkpoint\.dirpath={re.escape(str(ROOT))}/weights/({RUN_ID_RE})",
        result.stdout,
    )
    assert match is not None
    assert f"logging.run_name={match.group(1)}" in result.stdout
    assert "checkpoint.dirpath=weights/fcmae-real-scans" not in result.stdout
    assert "logging.wandb_enabled=true" not in result.stdout


def test_pretrain_fcmae_submit_dry_run_preserves_explicit_checkpoint_dir() -> None:
    result = subprocess.run(
        [
            str(SCRIPT),
            "submit",
            "--dry-run",
            "--no-sync",
            "--",
            "checkpoint.dirpath=weights/fcmae-smoke",
            "logging.wandb_enabled=false",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "checkpoint.dirpath=weights/fcmae-smoke" in result.stdout
    assert not re.search(RUN_ID_RE, result.stdout)


def test_pretrain_fcmae_submit_dry_run_preserves_explicit_run_name() -> None:
    result = subprocess.run(
        [
            str(SCRIPT),
            "submit",
            "--dry-run",
            "--no-sync",
            "--",
            "logging.run_name=custom-fcmae",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "logging.run_name=custom-fcmae" in result.stdout
    assert "logging.run_name=custom-fcmae-" not in result.stdout
    assert re.search(
        rf"checkpoint\.dirpath={re.escape(str(ROOT))}/weights/custom-fcmae-\d{{8}}-\d{{6}}",
        result.stdout,
    )


def test_pretrain_fcmae_local_dry_run_strips_double_dash_override_prefix() -> None:
    result = subprocess.run(
        [
            str(SCRIPT),
            "local",
            "--dry-run",
            "--",
            "--training.max_steps=1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "training.max_steps=1" in result.stdout
    assert "--training.max_steps=1" not in result.stdout
    assert not re.search(RUN_ID_RE, result.stdout)


def test_pretrain_fcmae_submit_dry_run_defaults_export_output_dir() -> None:
    result = subprocess.run(
        [
            str(SCRIPT),
            "submit",
            "--dry-run",
            "--no-sync",
            "--",
            "export.export_on_train_end=true",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    match = re.search(
        rf"checkpoint\.dirpath=({re.escape(str(ROOT))}/weights/{RUN_ID_RE})",
        result.stdout,
    )
    assert match is not None
    assert f"export.output_dir={match.group(1)}/exported_encoder" in result.stdout
    assert "export.export_on_train_end=true" in result.stdout


def test_pretrain_fcmae_submit_dry_run_preserves_explicit_export_output_dir() -> None:
    result = subprocess.run(
        [
            str(SCRIPT),
            "submit",
            "--dry-run",
            "--no-sync",
            "--",
            "export.export_on_train_end=true",
            "export.output_dir=weights/custom-export",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "export.output_dir=weights/custom-export" in result.stdout
    assert "/exported_encoder" not in result.stdout


def test_pretrain_fcmae_submit_dry_run_auto_forwards_ddp_for_multi_gpu() -> None:
    result = subprocess.run(
        [
            str(SCRIPT),
            "submit",
            "--dry-run",
            "--no-sync",
            "--gpus",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--gres=gpu:2" in result.stdout
    assert "--cpus-per-task=16" in result.stdout
    assert "training.devices=2" in result.stdout
    assert "training.strategy=ddp" in result.stdout
    assert "auto-scaled --cpus-per-task to 16" in result.stderr


def test_pretrain_fcmae_submit_dry_run_preserves_explicit_ddp_overrides() -> None:
    result = subprocess.run(
        [
            str(SCRIPT),
            "submit",
            "--dry-run",
            "--no-sync",
            "--gpus",
            "2",
            "--",
            "training.devices=1",
            "training.strategy=auto",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "training.devices=1" in result.stdout
    assert "training.strategy=auto" in result.stdout
    assert "training.devices=2" not in result.stdout
    assert "training.strategy=ddp" not in result.stdout
