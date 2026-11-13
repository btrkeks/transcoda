"""Artifact locations are configured once in config/artifacts.json."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def artifact_config() -> dict:
    return json.loads((ROOT / "config/artifacts.json").read_text())

def resolve_checkpoint(path: Path | None = None) -> Path:
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    config = artifact_config()
    local = ROOT / "weights" / config["checkpoint_filename"]
    if not local.is_file():
        from huggingface_hub import hf_hub_download
        local = Path(hf_hub_download(repo_id=config["model_id"], filename=config["checkpoint_filename"]))
    expected = json.loads((ROOT / "release/model-artifact.json").read_text())["exported_sha256"]
    with local.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError("Checkpoint checksum differs from the released artifact")
    return local
