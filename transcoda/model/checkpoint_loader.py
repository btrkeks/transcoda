"""Utilities for loading Transcoda checkpoints outside the training module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

from transcoda.artifacts import RunArtifact, _hash_vocab_dict
from transcoda.model import TranscodaConfig, TranscodaModelForCausalLM
from transcoda.tokenizer_compat import assert_vocab_hashes_match


@dataclass
class LoadedCheckpoint:
    """Resolved checkpoint bundle used by inference and evaluation."""

    model: TranscodaModelForCausalLM
    tokenizer: PreTrainedTokenizerFast
    tokenizer_load_path: str
    i2w: dict[int, str]
    pad_token_id: int
    artifact: RunArtifact
    vocab_dir: str
    image_width: int
    fixed_size: tuple[int, int] | None


@dataclass(frozen=True)
class _LoadedTokenizer:
    tokenizer: PreTrainedTokenizerFast
    tokenizer_load_path: str
    w2i: dict[str, int]
    i2w: dict[int, str]


def _vocab_from_tokenizer(tok: PreTrainedTokenizerFast) -> tuple[dict[str, int], dict[int, str]]:
    """Extract stable vocabulary mappings from a tokenizer."""
    w2i = tok.get_vocab()
    i2w = {int(i): t for t, i in w2i.items()}
    return w2i, i2w


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_vocab_dir(artifact: RunArtifact) -> str:
    if artifact.tokenizer.versioned_vocab_dir:
        return artifact.tokenizer.versioned_vocab_dir

    data_config = artifact.experiment_config["data"]
    if "vocab_dir" in data_config:
        return data_config["vocab_dir"]
    if "vocab_name" in data_config:
        return f"./vocab/{data_config['vocab_name']}"
    raise RuntimeError("Could not find vocab_dir or vocab_name in checkpoint.")


def _resolve_tokenizer_load_path(vocab_dir: str, *, repo_root: Path | None = None) -> str:
    vocab_path = Path(vocab_dir).expanduser()
    if not _looks_like_local_vocab_path(vocab_dir):
        return vocab_dir

    root = repo_root or Path.cwd()
    resolved_path = vocab_path if vocab_path.is_absolute() else root / vocab_path

    legacy_aliases = {
        Path("vocab/bpe3k-splitspaces"): Path("vocab/bpe3k-transcoda-v1"),
    }
    normalized = Path(*vocab_path.parts[1:]) if vocab_path.parts[:1] == (".",) else vocab_path
    alias = legacy_aliases.get(normalized)
    if alias is not None and not resolved_path.exists():
        alias_path = root / alias
        if alias_path.exists():
            return str(alias_path.resolve())

    return str(resolved_path.resolve(strict=False))


def _load_tokenizer_matching_artifact(
    vocab_dir: str,
    artifact: RunArtifact,
    checkpoint_path: str,
    *,
    repo_root: Path | None = None,
) -> _LoadedTokenizer:
    """Load the tokenizer matching checkpoint vocab hashes.

    Older checkpoints may record a stable-looking path such as
    ``./vocab/bpe3k-transcoda-v2`` even though that local directory was later
    regenerated. Prefer the recorded path when it still matches, otherwise scan
    local versioned tokenizer directories and select the one with matching
    artifact hashes.
    """
    root = repo_root or _repo_root()
    tokenizer_load_path = _resolve_tokenizer_load_path(vocab_dir, repo_root=root)
    tried = []

    for candidate_path in _tokenizer_candidate_paths(
        vocab_dir,
        tokenizer_load_path,
        repo_root=root,
    ):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(candidate_path)
        w2i, i2w = _vocab_from_tokenizer(tokenizer)
        w2i_hash = _hash_vocab_dict(w2i)
        i2w_hash = _hash_vocab_dict(i2w)
        if w2i_hash == artifact.vocab.w2i_hash and i2w_hash == artifact.vocab.i2w_hash:
            return _LoadedTokenizer(
                tokenizer=tokenizer,
                tokenizer_load_path=candidate_path,
                w2i=w2i,
                i2w=i2w,
            )
        tried.append(
            (
                candidate_path,
                w2i,
                i2w,
                w2i_hash,
                i2w_hash,
            )
        )

    if tried:
        _, first_w2i, first_i2w, _, _ = tried[0]
        try:
            assert_vocab_hashes_match(
                expected_w2i_hash=artifact.vocab.w2i_hash,
                expected_i2w_hash=artifact.vocab.i2w_hash,
                w2i=first_w2i,
                i2w=first_i2w,
                context_label=f"Checkpoint {checkpoint_path}",
            )
        except ValueError as exc:
            searched = "\n".join(
                f"  {path}: w2i_hash={w_hash} i2w_hash={i_hash}"
                for path, _w2i, _i2w, w_hash, i_hash in tried
            )
            raise ValueError(f"{exc}\nSearched tokenizer candidates:\n{searched}") from exc

    raise ValueError(
        f"Checkpoint {checkpoint_path} tokenizer mismatch detected. "
        f"No tokenizer candidates were found for vocab_dir={vocab_dir!r}."
    )


def _tokenizer_candidate_paths(
    vocab_dir: str,
    tokenizer_load_path: str,
    *,
    repo_root: Path,
) -> tuple[str, ...]:
    candidates = [str(Path(tokenizer_load_path).expanduser().resolve(strict=False))]
    if _looks_like_local_vocab_path(vocab_dir):
        vocab_root = repo_root / "vocab"
        if vocab_root.is_dir():
            candidates.extend(
                str(path.parent.resolve(strict=False))
                for path in sorted(vocab_root.glob("*/tokenizer.json"))
            )

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return tuple(deduped)


def _looks_like_local_vocab_path(vocab_dir: str) -> bool:
    if vocab_dir.startswith((".", "/", "~")):
        return True
    return Path(vocab_dir).parts[:1] == ("vocab",)


def _resolve_fixed_size(artifact: RunArtifact) -> tuple[int, int] | None:
    fixed_size = artifact.preprocessing.fixed_size
    if fixed_size is not None:
        return fixed_size

    data_cfg = artifact.experiment_config.get("data", {})
    fixed_h = data_cfg.get("fixed_image_height")
    fixed_w = data_cfg.get("fixed_image_width")
    if fixed_h is not None and fixed_w is not None:
        return (int(fixed_h), int(fixed_w))
    return None


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> LoadedCheckpoint:
    """Load a model checkpoint plus all metadata needed for inference."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]

    if "run_artifact_json" not in hparams:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' is missing 'run_artifact_json'. "
            "Cannot load model configuration."
        )

    artifact = RunArtifact.from_json(hparams["run_artifact_json"])
    vocab_dir = _resolve_vocab_dir(artifact)
    loaded_tokenizer = _load_tokenizer_matching_artifact(
        vocab_dir,
        artifact,
        checkpoint_path,
    )
    tokenizer = loaded_tokenizer.tokenizer
    tokenizer_load_path = loaded_tokenizer.tokenizer_load_path
    w2i = loaded_tokenizer.w2i
    i2w = loaded_tokenizer.i2w

    model_cfg = artifact.experiment_config["model"]
    num_hidden_layers = (
        model_cfg["num_hidden_layers"]
        if "num_hidden_layers" in model_cfg
        else model_cfg["num_dec_layers"]
    )
    model_config = TranscodaConfig(
        out_categories=hparams["out_categories"],
        pad_token_id=hparams["pad_token_id"],
        bos_token_id=hparams["bos_token_id"],
        eos_token_id=hparams["eos_token_id"],
        w2i=w2i,
        i2w=i2w,
        d_model=model_cfg["d_model"],
        dim_ff=model_cfg["dim_ff"],
        num_attn_heads=model_cfg["num_attn_heads"],
        num_hidden_layers=num_hidden_layers,
        encoder_model_name_or_path=model_cfg["encoder_model_name_or_path"],
        encoder_provider=model_cfg.get("encoder_provider", "transformers"),
        freeze_encoder_stages=model_cfg.get("freeze_encoder_stages", 0),
        vision_frontend=model_cfg.get("vision_frontend", "conv"),
        projector_hidden_mult=model_cfg.get("projector_hidden_mult", 4.0),
        multiscale_hidden_state_index=model_cfg.get("multiscale_hidden_state_index", -2),
        multiscale_fusion_init_logit=model_cfg.get("multiscale_fusion_init_logit", -4.0),
        positional_encoding=model_cfg.get("positional_encoding", "absolute"),
        rope_theta=model_cfg.get("rope_theta", 10000.0),
    )

    model = TranscodaModelForCausalLM(model_config)
    compile_prefix = "_compiled_forward_model._orig_mod."
    state_dict = {
        k.replace("model.", "", 1).replace(compile_prefix, ""): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return LoadedCheckpoint(
        model=model,
        tokenizer=tokenizer,
        tokenizer_load_path=tokenizer_load_path,
        i2w=i2w,
        pad_token_id=tokenizer.pad_token_id,
        artifact=artifact,
        vocab_dir=tokenizer_load_path,
        image_width=int(artifact.preprocessing.image_width),
        fixed_size=_resolve_fixed_size(artifact),
    )
