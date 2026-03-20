"""Utilities for loading SMT checkpoints outside the training module."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerFast

from src.artifacts import RunArtifact
from src.model import SMTConfig, SMTModelForCausalLM


@dataclass
class LoadedCheckpoint:
    """Resolved checkpoint bundle used by inference and benchmarking."""

    model: SMTModelForCausalLM
    i2w: dict[int, str]
    pad_token_id: int
    artifact: RunArtifact
    vocab_dir: str
    image_width: int
    fixed_size: tuple[int, int] | None


def _vocab_from_tokenizer(tok: PreTrainedTokenizerFast) -> tuple[dict[str, int], dict[int, str]]:
    """Extract stable vocabulary mappings from a tokenizer."""
    w2i = tok.get_vocab()
    i2w = {int(i): t for t, i in w2i.items()}
    return w2i, i2w


def _resolve_vocab_dir(artifact: RunArtifact) -> str:
    data_config = artifact.experiment_config["data"]
    if "vocab_dir" in data_config:
        return data_config["vocab_dir"]
    if "vocab_name" in data_config:
        return f"./vocab/{data_config['vocab_name']}"
    raise RuntimeError("Could not find vocab_dir or vocab_name in checkpoint.")


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

    tokenizer = PreTrainedTokenizerFast.from_pretrained(vocab_dir)
    w2i, i2w = _vocab_from_tokenizer(tokenizer)

    model_cfg = artifact.experiment_config["model"]
    num_hidden_layers = (
        model_cfg["num_hidden_layers"]
        if "num_hidden_layers" in model_cfg
        else model_cfg["num_dec_layers"]
    )
    model_config = SMTConfig(
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
        positional_encoding=model_cfg.get("positional_encoding", "absolute"),
        rope_theta=model_cfg.get("rope_theta", 10000.0),
    )

    model = SMTModelForCausalLM(model_config)
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
        i2w=i2w,
        pad_token_id=tokenizer.pad_token_id,
        artifact=artifact,
        vocab_dir=vocab_dir,
        image_width=int(artifact.preprocessing.image_width),
        fixed_size=_resolve_fixed_size(artifact),
    )
