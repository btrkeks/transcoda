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


def test_generate_synth_data_defaults_to_legacy_cloud_preset(tmp_path):
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
        }
    )

    subprocess.run(
        ["bash", str(script_path)],
        cwd=repo_root,
        env=env,
        check=True,
    )

    args = capture_path.read_text(encoding="utf-8").splitlines()
    assert "--dataset_preset" in args
    assert "legacy_cloud_default" in args
    assert "--num_workers" in args
    assert "7" in args


def test_generate_synth_data_forwards_cloud_overrides(tmp_path):
    repo_root, script_path = _setup_fake_cloud_repo(tmp_path)
    capture_path = repo_root / "captured-args.txt"

    env = os.environ.copy()
    env.update(
        {
            "SKIP_PY_SETUP": "true",
            "SKIP_RUNTIME_DEPENDENCY_PROBE": "true",
            "CAPTURE_ARGS_PATH": str(capture_path),
            "OUTPUT_DIR": "data/datasets/test_overrides",
            "CLOUD_DATASET_PRESET": "ablation_no_render_or_gt_aug",
            "CLOUD_DISABLE_OFFLINE_IMAGE_AUGMENTATIONS": "true",
            "CLOUD_COURTESY_NATURALS_PROBABILITY": "0.0",
            "CLOUD_RENDER_PEDALS_ENABLED": "true",
            "CLOUD_VARIANTS_PER_FILE": "9",
        }
    )

    subprocess.run(
        ["bash", str(script_path)],
        cwd=repo_root,
        env=env,
        check=True,
    )

    args = capture_path.read_text(encoding="utf-8").splitlines()
    assert "--dataset_preset" in args
    assert "ablation_no_render_or_gt_aug" in args
    assert "--disable_offline_image_augmentations" in args
    assert "true" in args
    assert "--courtesy_naturals_probability" in args
    assert "0.0" in args
    assert "--render_pedals_enabled" in args
    assert "--variants_per_file" in args
    assert "9" in args
