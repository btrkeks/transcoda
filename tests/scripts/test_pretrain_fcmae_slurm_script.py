import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "pretrain_fcmae.sh"


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
    assert "python scripts/pretrain_fcmae.py" in result.stdout
    assert "training.max_steps=10" in result.stdout
    assert "logging.wandb_enabled=true" in result.stdout
    assert "training.resume_from_checkpoint=weights/fcmae/last.ckpt" in result.stdout


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
