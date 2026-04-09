from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _setup_fake_cloud_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    script_source = Path("scripts/cloud/generate_synth_data.sh")
    script_target = repo_root / "scripts" / "cloud" / "generate_synth_data.sh"
    script_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(script_source, script_target)

    (repo_root / "config").mkdir(parents=True, exist_ok=True)
    (repo_root / "config" / "data_spec.json").write_text("{}", encoding="utf-8")

    for relative_dir in (
        "data/interim/train/grandstaff/3_normalized",
        "data/interim/train/pdmx/3_normalized",
        "data/interim/train/musetrainer/3_normalized",
        "data/interim/train/openscore-lieder/3_normalized",
        "data/interim/train/openscore-stringquartets/3_normalized",
    ):
        (repo_root / relative_dir).mkdir(parents=True, exist_ok=True)

    venv_bin = repo_root / ".venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    activate_path = venv_bin / "activate"
    activate_path.write_text(
        "export PATH=\"$(cd -- \"$(dirname -- \"${BASH_SOURCE[0]}\")\" && pwd):$PATH\"\n",
        encoding="utf-8",
    )
    python_path = venv_bin / "python"
    python_path.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"${CAPTURE_ARGS_PATH:?CAPTURE_ARGS_PATH is required}\"\n",
        encoding="utf-8",
    )
    python_path.chmod(0o755)
    script_target.chmod(0o755)
    return repo_root, script_target


def test_generate_synth_data_uses_rewrite_contract_defaults(tmp_path):
    repo_root, script_path = _setup_fake_cloud_repo(tmp_path)
    capture_path = repo_root / "captured-args.txt"

    env = os.environ.copy()
    env.update(
        {
            "SKIP_PY_SETUP": "true",
            "SKIP_RUNTIME_DEPENDENCY_PROBE": "true",
            "CAPTURE_ARGS_PATH": str(capture_path),
            "OUTPUT_DIR": "data/datasets/test_default",
            "NUM_WORKERS": "7",
            "TARGET_SAMPLES": "55",
        }
    )

    subprocess.run(
        ["bash", str(script_path)],
        cwd=repo_root,
        env=env,
        check=True,
    )

    args = capture_path.read_text(encoding="utf-8").splitlines()
    assert "data/interim/train/grandstaff/3_normalized" in args
    assert "data/interim/train/pdmx/3_normalized" in args
    assert "--num_workers" in args
    assert "7" in args
    assert "--target_samples" in args
    assert "55" in args


def test_generate_synth_data_forwards_supported_overrides(tmp_path):
    repo_root, script_path = _setup_fake_cloud_repo(tmp_path)
    capture_path = repo_root / "captured-args.txt"

    env = os.environ.copy()
    env.update(
        {
            "SKIP_PY_SETUP": "true",
            "SKIP_RUNTIME_DEPENDENCY_PROBE": "true",
            "CAPTURE_ARGS_PATH": str(capture_path),
            "OUTPUT_DIR": "data/datasets/test_overrides",
            "TARGET_SAMPLES": "9",
            "CLOUD_FAILURE_POLICY": "coverage",
            "BASE_SEED": "17",
            "MAX_ATTEMPTS": "21",
            "QUIET": "true",
        }
    )

    subprocess.run(
        ["bash", str(script_path)],
        cwd=repo_root,
        env=env,
        check=True,
    )

    args = capture_path.read_text(encoding="utf-8").splitlines()
    assert "--target_samples" in args
    assert "9" in args
    assert "--failure_policy" in args
    assert "coverage" in args
    assert "--base_seed" in args
    assert "17" in args
    assert "--max_attempts" in args
    assert "21" in args
    assert "--quiet" in args
    assert "true" in args


def test_generate_synth_data_rejects_legacy_cloud_overrides(tmp_path):
    repo_root, script_path = _setup_fake_cloud_repo(tmp_path)
    capture_path = repo_root / "captured-args.txt"

    env = os.environ.copy()
    env.update(
        {
            "SKIP_PY_SETUP": "true",
            "SKIP_RUNTIME_DEPENDENCY_PROBE": "true",
            "CAPTURE_ARGS_PATH": str(capture_path),
            "OUTPUT_DIR": "data/datasets/test_legacy_override",
            "CLOUD_DATASET_PRESET": "legacy_cloud_default",
        }
    )

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Unsupported legacy setting" in result.stderr
