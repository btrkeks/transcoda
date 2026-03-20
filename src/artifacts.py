# src/artifacts.py
"""Run artifact schema and utilities for reproducible experiments."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch


def _sha256_bytes(b: bytes) -> str:
    """Compute SHA256 hash of bytes."""
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _hash_vocab_dict(d: Mapping[int | str, int | str]) -> str:
    """Compute stable hash of vocabulary dictionary.

    Sorts by key and uses stable JSON encoding to ensure consistent hashing
    regardless of dict iteration order.
    """
    # Stable hash: sort by key, JSON-encode with stable separators
    payload = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _hash_tokenizer_file(path: str) -> str:
    """Compute SHA256 hash of tokenizer JSON file.

    Args:
        path: Path to the tokenizer JSON file

    Returns:
        Hex digest of SHA256 hash
    """
    from pathlib import Path

    with Path(path).open("rb") as f:
        return _sha256_bytes(f.read())


@dataclass(frozen=True)
class PreprocessingSpec:
    """Preprocessing configuration for images and sequences."""

    image_width: int  # Target width in pixels
    fixed_size: tuple[int, int] | None  # (H, W) or None


@dataclass(frozen=True)
class DecodingSpec:
    """Decoding strategy and parameters for inference."""

    strategy: str = "beam"  # "greedy" | "beam"
    max_len: int = 0
    eos_token: str = "</eos>"
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    num_beams: int | None = None
    length_penalty: float | None = 1.0
    repetition_penalty: float = 1.1
    early_stopping: bool | str = True
    num_return_sequences: int = 1
    do_sample: bool = False
    use_cache: bool = True


@dataclass(frozen=True)
class VocabSpec:
    """Vocabulary specification with hashes for validation."""

    w2i_hash: str
    i2w_hash: str
    pad_token: int
    bos_token: str
    eos_token: str


@dataclass(frozen=True)
class TokenizerSpec:
    """Tokenizer configuration for reproducibility and inference validation."""

    vocab_size: int | None = None


@dataclass(frozen=True)
class EnvSpec:
    """Environment snapshot: versions of key dependencies."""

    torch: str
    lightning: str
    transformers: str
    cuda: str | None
    cudnn: str | None
    python: str
    platform: str


@dataclass(frozen=True)
class SeedSpec:
    """Random seed configuration."""

    global_seed: int
    deterministic: bool


@dataclass(frozen=True)
class SlurmSpec:
    """Slurm runtime metadata for reproducibility/debugging."""

    job_id: str | None = None
    job_name: str | None = None
    partition: str | None = None
    nodelist: str | None = None
    cpus_per_task: int | None = None
    gpus_on_node: str | None = None
    gpu_binding: str | None = None
    submit_host: str | None = None
    cluster_name: str | None = None
    array_job_id: str | None = None
    array_task_id: str | None = None


@dataclass(frozen=True)
class RunArtifact:
    """Complete run artifact capturing all configuration and environment state."""

    experiment_config: dict  # the full JSON loaded in train.py
    preprocessing: PreprocessingSpec
    decoding: DecodingSpec
    vocab: VocabSpec
    tokenizer: TokenizerSpec
    env: EnvSpec
    seed: SeedSpec
    slurm: SlurmSpec | None = None

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    @staticmethod
    def from_json(s: str) -> RunArtifact:
        """Deserialize from JSON string."""
        obj = json.loads(s)
        decoding_payload = dict(obj["decoding"])
        decoding_payload.setdefault("repetition_penalty", 1.1)

        return RunArtifact(
            experiment_config=obj["experiment_config"],
            preprocessing=PreprocessingSpec(**obj["preprocessing"]),
            decoding=DecodingSpec(**decoding_payload),
            vocab=VocabSpec(**obj["vocab"]),
            tokenizer=TokenizerSpec(**obj["tokenizer"]),
            env=EnvSpec(**obj["env"]),
            seed=SeedSpec(**obj["seed"]),
            slurm=SlurmSpec(**obj["slurm"]) if obj.get("slurm") is not None else None,
        )


def _int_or_none(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def collect_slurm() -> SlurmSpec | None:
    """Collect Slurm environment metadata if running under Slurm."""
    spec = SlurmSpec(
        job_id=os.getenv("SLURM_JOB_ID"),
        job_name=os.getenv("SLURM_JOB_NAME"),
        partition=os.getenv("SLURM_JOB_PARTITION"),
        nodelist=os.getenv("SLURM_NODELIST"),
        cpus_per_task=_int_or_none(os.getenv("SLURM_CPUS_PER_TASK")),
        gpus_on_node=os.getenv("SLURM_GPUS_ON_NODE"),
        gpu_binding=os.getenv("SLURM_GPU_BIND"),
        submit_host=os.getenv("SLURM_SUBMIT_HOST"),
        cluster_name=os.getenv("SLURM_CLUSTER_NAME"),
        array_job_id=os.getenv("SLURM_ARRAY_JOB_ID"),
        array_task_id=os.getenv("SLURM_ARRAY_TASK_ID"),
    )

    if all(value is None for value in asdict(spec).values()):
        return None
    return spec


def collect_env() -> EnvSpec:
    """Collect current environment versions."""
    import lightning
    import transformers

    return EnvSpec(
        torch=torch.__version__,
        lightning=lightning.__version__,
        transformers=transformers.__version__,
        cuda=torch.version.cuda if torch.cuda.is_available() else None,
        cudnn=str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
        python=platform.python_version(),
        platform=f"{platform.system()}-{platform.machine()}",
    )
