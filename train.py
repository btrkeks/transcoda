import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

# Bucketed decoder padding (training.decoder_length_buckets) varies activation
# sizes between batches; without expandable segments the allocator fragments
# and OOMs on the ~1 GiB it can no longer coalesce. Must be set before the
# first CUDA allocation; respects an explicit override from the environment.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import fire
import torch
from rich.console import Console
from transformers import PreTrainedTokenizerFast

from transcoda.artifacts import (
    DecodingSpec,
    PreprocessingSpec,
    RunArtifact,
    SeedSpec,
    TokenizerSpec,
    VocabSpec,
    _hash_vocab_dict,
    collect_env,
    collect_slurm,
)
from transcoda.callbacks.gated_checkpoint import GatedModelCheckpoint
from transcoda.callbacks.validation_prediction_writer import predictions_output_dir
from transcoda.config import (
    Checkpoint,
    Data,
    ExperimentConfig,
    Generation,
    ModelConfig,
    OptimizerConfig,
    TokenizerConfig,
    Training,
    ValidationConfig,
    ValidationDecoding,
    ValidationMetrics,
    ValidationSchedule,
    ValidationSelection,
    ValidationVeroSim,
    experiment_config_from_dict,
)
from transcoda.data.datamodules import PregeneratedSyntheticGrandStaffDM
from transcoda.evaluation.verosim_omr_ned import (
    log_verosim_omr_ned_summary,
    run_verosim_omr_ned,
)
from transcoda.metrics_schema import FINAL_VAL_PREFIX, VAL_PREFIX
from transcoda.tokenizer_compat import (
    assert_vocab_hashes_match,
    read_tokenizer_manifest,
    vocab_from_tokenizer,
)
from transcoda.training.lightning_module import TranscodaTrainer
from transcoda.training.setup import setup_callbacks, setup_logger, setup_trainer
from transcoda.utils.repro import seed_everything

# PyTorch 2.6+ defaults to weights_only=True for security. Allowlist our config
# classes so Lightning can restore checkpoints containing them. The tuple
# entries preserve checkpoints serialized before the package rename.
_CHECKPOINT_CONFIG_CLASSES = [
    Checkpoint,
    Data,
    ExperimentConfig,
    Generation,
    ModelConfig,
    OptimizerConfig,
    TokenizerConfig,
    Training,
    ValidationConfig,
    ValidationDecoding,
    ValidationMetrics,
    ValidationSchedule,
    ValidationSelection,
    ValidationVeroSim,
]
torch.serialization.add_safe_globals(
    [
        *_CHECKPOINT_CONFIG_CLASSES,
        *[
            (config_class, f"src.config.{config_class.__qualname__}")
            for config_class in _CHECKPOINT_CONFIG_CLASSES
        ],
    ]
)

console = Console()
torch.set_float32_matmul_precision("high")


# -------- helpers --------
def _apply_overrides(config_dict: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """
    Apply CLI overrides to config dict using dot-notation keys.

    Example:
        overrides = {"model.d_model": 512, "training.max_epochs": 10}
        Becomes: config_dict["model"]["d_model"] = 512, etc.
    """
    for key, value in overrides.items():
        parts = key.split(".")
        target = config_dict
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return config_dict


def _verify_checkpoint_tokenizer_compatibility(
    checkpoint_path: str,
    w2i: dict[str, int],
    i2w: dict[int, str],
) -> None:
    """Fail fast if checkpoint run artifact expects a different tokenizer mapping."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        # Let Lightning raise its own checkpoint-path error later.
        return

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    run_artifact_json = hyper_parameters.get("run_artifact_json")
    if not isinstance(run_artifact_json, str):
        return

    artifact = json.loads(run_artifact_json)
    vocab_spec = artifact.get("vocab", {})
    expected_w2i_hash = vocab_spec.get("w2i_hash")
    expected_i2w_hash = vocab_spec.get("i2w_hash")
    if not expected_w2i_hash or not expected_i2w_hash:
        return

    assert_vocab_hashes_match(
        expected_w2i_hash=expected_w2i_hash,
        expected_i2w_hash=expected_i2w_hash,
        w2i=w2i,
        i2w=i2w,
        context_label=f"Checkpoint {checkpoint_path}",
    )


def _manifest_string(manifest: dict[str, Any] | None, key: str) -> str | None:
    if manifest is None:
        return None
    value = manifest.get(key)
    return value if isinstance(value, str) else None


def _load_checkpoint_experiment_config(checkpoint_path: str) -> dict[str, Any] | None:
    """Return the experiment config embedded in a checkpoint artifact, if present."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        return None

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    run_artifact_json = hyper_parameters.get("run_artifact_json")
    if not isinstance(run_artifact_json, str):
        return None

    artifact = json.loads(run_artifact_json)
    experiment_config = artifact.get("experiment_config")
    if not isinstance(experiment_config, dict):
        return None

    return experiment_config


def _apply_checkpoint_model_defaults(
    config_dict: dict[str, Any],
    checkpoint_path: str,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """
    Prefer checkpoint-time model settings unless explicitly overridden.

    This keeps the instantiated model architecture aligned with the checkpoint even
    if the local repo config has since changed.
    """
    checkpoint_config = _load_checkpoint_experiment_config(checkpoint_path)
    if checkpoint_config is None:
        return config_dict

    checkpoint_model_cfg = checkpoint_config.get("model", {})
    if not isinstance(checkpoint_model_cfg, dict):
        return config_dict

    config_dict.setdefault("model", {})
    applied: list[str] = []
    model_fields = set(ModelConfig.model_fields)
    for checkpoint_key, checkpoint_value in checkpoint_model_cfg.items():
        config_key = checkpoint_key
        if checkpoint_key == "num_dec_layers":
            config_key = "num_hidden_layers"
        if config_key not in model_fields or f"model.{config_key}" in overrides:
            continue

        current_value = config_dict["model"].get(config_key)
        if current_value != checkpoint_value:
            applied.append(config_key)
        config_dict["model"][config_key] = checkpoint_value

    if applied:
        console.print(
            "[yellow]checkpoint resume: using checkpoint artifact "
            f"model config for {', '.join(sorted(applied))}[/yellow]"
        )
    return config_dict


def _infer_total_steps(
    dm, max_epochs: int, accumulate_grad_batches: int | None, explicit_max_steps: int | None
):
    steps_per_epoch = len(dm.train_dataloader())
    agb = accumulate_grad_batches or 1
    if explicit_max_steps:
        return explicit_max_steps
    return (steps_per_epoch // agb) * max_epochs


def _collect_git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _collect_launch_command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def _resolve_checkpoint_path_for_fit(
    config: ExperimentConfig,
    explicit_checkpoint_path: str | None,
    *,
    fresh_run: bool,
) -> tuple[str | None, str]:
    """Resolve checkpoint path for fit runs and return (path, reason)."""
    if explicit_checkpoint_path:
        return explicit_checkpoint_path, "explicit_checkpoint"

    if fresh_run:
        return None, "fresh_run_no_resume"

    if not config.checkpoint.auto_resume:
        return None, "auto_resume_disabled"

    last_ckpt = Path(config.checkpoint.dirpath) / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt), "auto_resume_last_ckpt"

    return None, "no_last_ckpt_start_scratch"


def _validate_final_validation_checkpoint_policy(config: ExperimentConfig) -> None:
    if config.checkpoint.save_top_k == 0 and not config.checkpoint.save_last:
        raise ValueError(
            "Automatic final validation requires at least one persisted checkpoint. "
            "Set checkpoint.save_top_k > 0 or checkpoint.save_last=true."
        )


def _existing_path(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)
    return None


def _resolve_checkpoint_path_for_final_validation(
    config: ExperimentConfig,
    primary_checkpointer: Any | None,
    omr_ned_checkpointer: Any | None = None,
) -> tuple[str, str]:
    if config.validation.selection.enabled:
        omr_ned_best_path = _existing_path(getattr(omr_ned_checkpointer, "best_model_path", None))
        if omr_ned_best_path is not None:
            return omr_ned_best_path, "omr_ned_best_model_path"

    best_model_path = _existing_path(getattr(primary_checkpointer, "best_model_path", None))
    if best_model_path is not None:
        return best_model_path, "best_model_path"

    last_model_path = _existing_path(getattr(primary_checkpointer, "last_model_path", None))
    if last_model_path is not None:
        return last_model_path, "last_model_path"

    disk_last_ckpt = _existing_path(str(Path(config.checkpoint.dirpath) / "last.ckpt"))
    if disk_last_ckpt is not None:
        return disk_last_ckpt, "dirpath_last_ckpt"

    raise RuntimeError(
        "Unable to resolve a checkpoint for automatic final validation. "
        "Checked primary best_model_path, primary last_model_path, and "
        f"{Path(config.checkpoint.dirpath) / 'last.ckpt'}."
    )


def _checkpoint_score(checkpointer: Any | None) -> float | None:
    value = getattr(checkpointer, "best_model_score", None)
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_selected_checkpoint_manifest(
    config: ExperimentConfig,
    checkpoint_path: str,
    reason: str,
    primary_checkpointer: Any | None,
    omr_ned_checkpointer: Any | None,
) -> Path:
    path = Path(checkpoint_path).resolve()
    checkpoint_dir = Path(config.checkpoint.dirpath).resolve()
    if path.parent != checkpoint_dir:
        raise RuntimeError(
            f"selected checkpoint must be directly under {checkpoint_dir}, got {path}"
        )

    if reason == "omr_ned_best_model_path":
        selection_reason = "omr_ned_best"
        monitor = config.validation.selection.primary
        metric_value = _checkpoint_score(omr_ned_checkpointer)
    elif reason == "best_model_path":
        selection_reason = "cer_best"
        monitor = config.checkpoint.monitor
        metric_value = _checkpoint_score(primary_checkpointer)
    elif reason in {"last_model_path", "dirpath_last_ckpt"}:
        selection_reason = "last"
        monitor = None
        metric_value = None
    else:
        raise ValueError(f"unknown checkpoint selection reason: {reason}")

    payload = {
        "schema_version": 1,
        "checkpoint_filename": path.name,
        "checkpoint_sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "selection_reason": selection_reason,
        "monitor": monitor,
        "metric_value": metric_value,
        "omr_ned_checkpoint_available": bool(
            _existing_path(getattr(omr_ned_checkpointer, "best_model_path", None))
        ),
    }
    manifest_path = checkpoint_dir / "selected_checkpoint.json"
    tmp_path = manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(manifest_path)
    return manifest_path


def _run_final_validation(
    trainer,
    model_wrapper,
    datamodule,
    config: ExperimentConfig,
    primary_checkpointer: Any | None,
    omr_ned_checkpointer: Any | None = None,
    selected_checkpoint: tuple[str, str] | None = None,
) -> None:
    """Run the automatic post-fit validation pass with prediction recording enabled."""
    checkpoint_path, reason = selected_checkpoint or _resolve_checkpoint_path_for_final_validation(
        config, primary_checkpointer, omr_ned_checkpointer
    )
    console.print(
        f"[bold cyan]Running full final validation from {reason}: {checkpoint_path}[/bold cyan]"
    )
    datamodule.setup(stage="validate")
    model_wrapper.disable_compiled_forward_model()
    model_wrapper.set_validation_metric_prefix(FINAL_VAL_PREFIX)
    model_wrapper.set_validation_example_logging_override(False)
    model_wrapper.set_validation_prediction_recording(True)
    try:
        trainer.validate(model_wrapper, datamodule=datamodule, ckpt_path=checkpoint_path)
    finally:
        model_wrapper.set_validation_metric_prefix(VAL_PREFIX)
        model_wrapper.set_validation_example_logging_override(None)
        model_wrapper.set_validation_prediction_recording(False)


def _find_omr_ned_checkpointer(callbacks: list[Any]) -> GatedModelCheckpoint | None:
    for callback in callbacks:
        if isinstance(callback, GatedModelCheckpoint):
            return callback
    return None


def _log_predictions_artifact(wandb_logger, predictions_dir: Path, label: str) -> None:
    """Best-effort upload of the prediction dump as a W&B artifact.

    The dump on disk is the primary artifact; a failed upload at the end of a
    long training run must not fail the job.
    """
    if not predictions_dir.exists():
        return
    try:
        import wandb

        experiment = wandb_logger.experiment
        log_artifact = getattr(experiment, "log_artifact", None)
        if not callable(log_artifact):
            return
        run_id = getattr(experiment, "id", None) or "run"
        artifact = wandb.Artifact(
            name=f"{label}-predictions-{run_id}", type="validation-predictions"
        )
        artifact.add_dir(str(predictions_dir))
        log_artifact(artifact)
        console.print(f"[cyan]Uploaded {label} predictions to W&B ({predictions_dir})")
    except Exception as exc:
        console.print(f"[yellow]W&B predictions artifact upload failed: {exc}[/yellow]")


# -------- main --------
def _resolve_training_seed(config: ExperimentConfig, seed: int | None) -> int:
    """Require direct CLI compatibility assertions to match the semantic config."""

    if seed is not None and (type(seed) is not int or seed != config.seed):
        raise ValueError(
            "Direct --seed must match the resolved config seed; change seed in the "
            "experiment config so it receives a distinct run identity"
        )
    return config.seed


def main(
    config_path: str,
    checkpoint_path: str | None = None,
    validate_only: bool = False,
    seed: int | None = None,
    fresh_run: bool = False,
    **overrides: Any,
):
    """
    Main training entry point.

    Args:
        config_path: Path to the JSON configuration file
        checkpoint_path: Optional path to checkpoint to resume training from
        validate_only: Run validation-only pass on a checkpoint (no training)
        seed: Optional compatibility assertion; must match the resolved config seed.
        fresh_run: Disable auto-resume and force training from scratch
        **overrides: Config overrides using dot notation (e.g., model.d_model=512)
    """
    # ----- config & seeding -----
    with open(config_path) as f:
        config_dict = json.load(f)

    # Apply CLI overrides
    if overrides:
        console.print(f"[yellow]Applying {len(overrides)} config override(s):")
        for key, value in overrides.items():
            console.print(f"  {key} = {value}")
        config_dict = _apply_overrides(config_dict, overrides)

    # Validation-only runs always log validation example images.
    # This is enforced at entrypoint level (not just shell wrapper flags)
    # to guarantee consistent behavior in Slurm and direct python invocation.
    if validate_only:
        config_dict.setdefault("training", {})
        if config_dict["training"].get("log_example_images") is not True:
            console.print(
                "[yellow]validate_only=True: forcing training.log_example_images=true[/yellow]"
            )
        config_dict["training"]["log_example_images"] = True

    if validate_only and checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided when validate_only=True")

    config = experiment_config_from_dict(config_dict)
    seed = _resolve_training_seed(config, seed)
    config_dict["seed"] = seed
    config = experiment_config_from_dict(config_dict)
    if not validate_only and not config.training.skip_final_validation:
        _validate_final_validation_checkpoint_policy(config)
    resolved_checkpoint_path = checkpoint_path
    checkpoint_resolution_reason = "validate_only_explicit_checkpoint" if validate_only else "n/a"

    if not validate_only:
        resolved_checkpoint_path, checkpoint_resolution_reason = _resolve_checkpoint_path_for_fit(
            config=config,
            explicit_checkpoint_path=checkpoint_path,
            fresh_run=fresh_run,
        )
        if checkpoint_resolution_reason == "explicit_checkpoint":
            console.print(
                f"[bold yellow]Resuming from explicitly provided checkpoint: "
                f"{resolved_checkpoint_path}[/bold yellow]"
            )
        elif checkpoint_resolution_reason == "auto_resume_last_ckpt":
            console.print(
                f"[bold yellow]Auto-resuming from last checkpoint: "
                f"{resolved_checkpoint_path}[/bold yellow]"
            )
        elif checkpoint_resolution_reason == "fresh_run_no_resume":
            console.print(
                "[yellow]fresh_run=true: starting from scratch (auto-resume disabled)[/yellow]"
            )
        elif checkpoint_resolution_reason == "auto_resume_disabled":
            console.print("[yellow]checkpoint.auto_resume=false: starting from scratch[/yellow]")
        else:
            console.print("[yellow]No last.ckpt found: starting from scratch[/yellow]")

    if resolved_checkpoint_path is not None:
        config_dict = _apply_checkpoint_model_defaults(
            config_dict,
            resolved_checkpoint_path,
            overrides,
        )
        config = experiment_config_from_dict(config_dict)

    # Disable trackers that aren't DDP-aware. RunawayMonitorTracker accumulates
    # Python-side state and doesn't all-reduce across ranks, so under DDP each
    # rank would compute partial aggregates that silently combine into a
    # misleading global. Auto-disable rather than report wrong numbers.
    ddp_world_size = int(config.training.devices) * int(config.training.num_nodes)
    if ddp_world_size > 1:
        config_dict.setdefault("training", {})
        if config.training.runaway_monitor_enabled:
            console.print(
                f"[yellow]DDP run detected (world_size={ddp_world_size}): "
                "disabling training.runaway_monitor_enabled (tracker is not DDP-safe).[/yellow]"
            )
            config.training.runaway_monitor_enabled = False
            config_dict["training"]["runaway_monitor_enabled"] = False

    deterministic = False
    seed_everything(seed, deterministic=deterministic)

    # ----- tokenizer -----
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.data.vocab_dir)
    assert tokenizer.pad_token_id is not None
    assert tokenizer.bos_token_id is not None
    assert tokenizer.eos_token_id is not None
    w2i, i2w = vocab_from_tokenizer(tokenizer)
    tokenizer_manifest = read_tokenizer_manifest(config.data.vocab_dir)

    if resolved_checkpoint_path:
        _verify_checkpoint_tokenizer_compatibility(
            checkpoint_path=resolved_checkpoint_path,
            w2i=w2i,
            i2w=i2w,
        )

    # ----- datamodule -----
    console.print("[yellow]Loading dataset...")
    datamodule = PregeneratedSyntheticGrandStaffDM(
        tokenizer=tokenizer,
        data_config=config.data,
        training_config=config.training,
        max_decoder_len=int(config.model.max_seq_len),
        training_seed=seed,
    )
    setup_stage = "validate" if validate_only else "fit"
    datamodule.setup(stage=setup_stage)

    # Dataset confirmation logging
    train_sample_count = 0
    if datamodule.train_set is not None:
        train_sample_count = len(datamodule.train_set)
        train_label = config.data.train_path or ", ".join(
            f"{source.release}@{source.weight}" for source in config.data.train_sources or ()
        )
        console.print(f"[cyan]Train: {train_label} ({train_sample_count:,} samples)")
    else:
        console.print("[cyan]Train: skipped in validate-only mode[/cyan]")
    total_val_samples = 0
    for name, ds in datamodule.val_sets.items():
        console.print(f"[cyan]Val/{name}: {len(ds):,} samples")
        total_val_samples += len(ds)
    console.print("[blue]Datasets loaded")

    max_height = config.model.max_image_height
    max_width = config.model.max_image_width
    max_len = config.model.max_seq_len

    # ----- training schedule -----
    if validate_only:
        total_steps = int(config.training.max_steps or 0)
        console.print("[cyan]Validation-only run: skipped scheduler total-step inference.")
    else:
        total_steps = _infer_total_steps(
            datamodule,
            max_epochs=int(config.training.max_epochs),
            accumulate_grad_batches=config.training.accumulate_grad_batches or 1,
            explicit_max_steps=config.training.max_steps,
        )
        console.print(f"[cyan]Scheduler configured for {total_steps:,} total optimizer steps.")

    # ----- run artifact -----
    console.print("[yellow]Assembling run artifact...")

    preproc = PreprocessingSpec(
        image_width=config.data.image_width,
        fixed_size=(config.data.fixed_image_height, config.data.fixed_image_width),
    )

    # Use tokenizer specials
    eos_token = tokenizer.eos_token or "</eos>"

    decoding = DecodingSpec(
        strategy=config.generation.strategy,
        max_len=int(config.generation.max_length or max_len),
        eos_token=eos_token,
        temperature=None,
        top_k=None,
        top_p=None,
        num_beams=config.generation.num_beams,
        length_penalty=config.generation.length_penalty,
        repetition_penalty=config.generation.repetition_penalty,
        early_stopping=config.generation.early_stopping,
        num_return_sequences=config.generation.num_return_sequences,
        do_sample=config.generation.do_sample,
        use_cache=config.generation.use_cache,
        spine_structure_barline_width=config.validation.decoding.spine_structure_barline_width,
        spine_structure_empty_field_mode=(
            config.validation.decoding.spine_structure_empty_field_mode
        ),
    )

    vocab = VocabSpec(
        w2i_hash=_hash_vocab_dict(w2i),
        i2w_hash=_hash_vocab_dict(i2w),
        pad_token=int(tokenizer.pad_token_id),
        bos_token=tokenizer.bos_token or "<bos>",
        eos_token=eos_token,
    )

    tokenizer_spec = TokenizerSpec(
        vocab_size=len(tokenizer),
        tokenizer_id=_manifest_string(tokenizer_manifest, "tokenizer_id"),
        tokenizer_json_sha256=_manifest_string(tokenizer_manifest, "tokenizer_json_sha256"),
        vocab_dir=config.data.vocab_dir,
        versioned_vocab_dir=_manifest_string(tokenizer_manifest, "versioned_vocab_dir"),
    )
    slurm_spec = collect_slurm()

    artifact = RunArtifact(
        experiment_config=config_dict,
        preprocessing=preproc,
        decoding=decoding,
        vocab=vocab,
        tokenizer=tokenizer_spec,
        env=collect_env(),
        seed=SeedSpec(global_seed=seed, deterministic=deterministic),
        slurm=slurm_spec,
    )

    console.print("[green]Run artifact assembled")

    # ----- model -----
    model_wrapper = TranscodaTrainer(
        model_config=config.model,
        optimizer_config=config.optimizer,
        training=config.training,
        generation=config.generation,
        maxh=int(max_height),
        maxw=int(max_width),
        maxlen=int(max_len),
        out_categories=int(len(tokenizer)),
        w2i=w2i,
        i2w=i2w,
        vocab_dir=config.data.vocab_dir,
        val_set_names=datamodule.val_loader_names,
        base_val_set_names=datamodule.val_set_names,
        run_artifact_json=artifact.to_json(),
        bos_token_id=int(tokenizer.bos_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        pad_token_id=int(tokenizer.pad_token_id),
    )

    # ----- logging / trainer -----
    callbacks, primary_checkpointer = setup_callbacks(
        config=config,
        pad_token_id=int(tokenizer.pad_token_id),
        bos_token_id=int(tokenizer.bos_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        i2w=i2w,
    )
    omr_ned_checkpointer = _find_omr_ned_checkpointer(callbacks)
    wandb_logger = setup_logger(config, config_dict)

    # Under DDP, WandbLogger.experiment on non-zero ranks returns a DummyExperiment
    # whose attribute access returns no-op functions: `summary` is not dict-like,
    # `config.update(...)` raises AttributeError because `config` is a function.
    # Gate every pre-Trainer W&B mutation and local file write on rank zero.
    # Lightning's subprocess launcher sets LOCAL_RANK (not RANK); torchrun sets
    # both. Check LOCAL_RANK first so this works under both launchers.
    is_rank_zero = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0

    run_artifact_payload = artifact.to_json()
    if is_rank_zero:
        experiment = wandb_logger.experiment
        summary = getattr(experiment, "summary", None)
        if hasattr(summary, "__setitem__"):
            summary["run_artifact"] = run_artifact_payload
            console.print("[cyan]Run artifact saved to W&B summary")
        else:
            # Test-double path: no real summary; log via metric-history instead.
            experiment.log({"run_artifact": run_artifact_payload})
            console.print("[cyan]Run artifact logged to W&B")

        run_dir = Path(config.checkpoint.dirpath)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config_resolved.json").write_text(json.dumps(config_dict, indent=2))
        (run_dir / "run_artifact.json").write_text(artifact.to_json())
        console.print(f"[cyan]Local config/artifact saved to {run_dir}")

    # Log scale metrics for easy filtering/comparison in W&B
    num_params = sum(p.numel() for p in model_wrapper.parameters())
    num_trainable = sum(p.numel() for p in model_wrapper.parameters() if p.requires_grad)
    slurm_config = {}
    if slurm_spec is not None:
        slurm_config = {
            "slurm/job_id": slurm_spec.job_id,
            "slurm/job_name": slurm_spec.job_name,
            "slurm/partition": slurm_spec.partition,
            "slurm/nodelist": slurm_spec.nodelist,
            "slurm/cpus_per_task": slurm_spec.cpus_per_task,
            "slurm/gpus_on_node": slurm_spec.gpus_on_node,
            "slurm/gpu_binding": slurm_spec.gpu_binding,
            "slurm/submit_host": slurm_spec.submit_host,
            "slurm/cluster_name": slurm_spec.cluster_name,
            "slurm/array_job_id": slurm_spec.array_job_id,
            "slurm/array_task_id": slurm_spec.array_task_id,
        }
        slurm_config = {k: v for k, v in slurm_config.items() if v is not None}

    git_sha = _collect_git_sha()
    launch_command = _collect_launch_command()

    run_metadata = {
        "launch/command": launch_command,
        "resume/reason": checkpoint_resolution_reason,
        "resume/used_checkpoint": resolved_checkpoint_path is not None,
    }
    if git_sha is not None:
        run_metadata["git/sha"] = git_sha
    if resolved_checkpoint_path is not None:
        run_metadata["resume/checkpoint_path"] = resolved_checkpoint_path

    if is_rank_zero:
        wandb_logger.experiment.config.update(
            {
                "scale/num_params": num_params,
                "scale/num_trainable_params": num_trainable,
                "scale/train_samples": train_sample_count,
                "scale/val_samples": total_val_samples,
                "scale/total_steps": total_steps,
                "scale/effective_batch_size": config.training.batch_size
                * (config.training.accumulate_grad_batches or 1),
                **slurm_config,
                **run_metadata,
            },
            allow_val_change=True,
        )
    console.print(f"[cyan]Model: {num_trainable:,} trainable params ({num_params:,} total)")

    num_train_batches = None
    if not validate_only:
        num_train_batches = len(datamodule.train_dataloader())
    trainer = setup_trainer(config, callbacks, wandb_logger, num_train_batches=num_train_batches)

    if validate_only:
        console.print(
            f"[bold cyan]Running validation-only pass from checkpoint: "
            f"{resolved_checkpoint_path}[/bold cyan]"
        )
        model_wrapper.set_validation_metric_prefix(VAL_PREFIX)
        model_wrapper.set_validation_example_logging_override(None)
        model_wrapper.set_validation_prediction_recording(True)
        trainer.validate(model_wrapper, datamodule=datamodule, ckpt_path=resolved_checkpoint_path)
        if is_rank_zero:
            validation_predictions_dir = predictions_output_dir(
                config.checkpoint.dirpath, VAL_PREFIX
            )
            verosim_cfg = config.validation.metrics.verosim
            if verosim_cfg.enabled:
                omr_ned_result = run_verosim_omr_ned(
                    predictions_dir=validation_predictions_dir,
                    kern_parser_binary=verosim_cfg.kern_parser_binary,
                    verosim_binary=verosim_cfg.verosim_binary,
                    mode=verosim_cfg.mode,
                    jobs=verosim_cfg.jobs,
                    failure_score=verosim_cfg.failure_score,
                    target_text=verosim_cfg.target_text,
                    validation_paths=config.data.validation_paths,
                    exclude_val_sets=verosim_cfg.exclude_val_sets or (),
                )
                if omr_ned_result.skipped:
                    console.print(
                        f"[yellow]VeroSim OMR-NED skipped: {omr_ned_result.reason}[/yellow]"
                    )
                elif omr_ned_result.summary is not None:
                    log_verosim_omr_ned_summary(
                        wandb_logger,
                        omr_ned_result.summary,
                        diagnostics_summary=getattr(omr_ned_result, "diagnostics_summary", None),
                    )
                    console.print(
                        "[cyan]VeroSim OMR-NED summary written to "
                        f"{omr_ned_result.summary_json}[/cyan]"
                    )
            _log_predictions_artifact(
                wandb_logger,
                validation_predictions_dir,
                label=VAL_PREFIX,
            )
    else:
        if resolved_checkpoint_path:
            console.print(
                f"[bold yellow]Resuming training from checkpoint: "
                f"{resolved_checkpoint_path}[/bold yellow]"
            )
        else:
            console.print("[bold green]Starting training from scratch...[/bold green]")

        if is_rank_zero:
            (Path(config.checkpoint.dirpath) / "selected_checkpoint.json").unlink(missing_ok=True)
        trainer.fit(model_wrapper, datamodule=datamodule, ckpt_path=resolved_checkpoint_path)
        selected_checkpoint = None
        try:
            selected_checkpoint = _resolve_checkpoint_path_for_final_validation(
                config,
                primary_checkpointer,
                omr_ned_checkpointer,
            )
        except RuntimeError:
            if not config.training.skip_final_validation:
                raise
        if selected_checkpoint is not None and is_rank_zero:
            manifest_path = _write_selected_checkpoint_manifest(
                config,
                selected_checkpoint[0],
                selected_checkpoint[1],
                primary_checkpointer,
                omr_ned_checkpointer,
            )
            console.print(f"[cyan]Selected checkpoint manifest written to {manifest_path}[/cyan]")
        if config.training.skip_final_validation:
            console.print(
                "[yellow]training.skip_final_validation=true: skipped final validation.[/yellow]"
            )
        else:
            _run_final_validation(
                trainer,
                model_wrapper,
                datamodule,
                config,
                primary_checkpointer,
                omr_ned_checkpointer,
                selected_checkpoint,
            )
            if is_rank_zero:
                _log_predictions_artifact(
                    wandb_logger,
                    predictions_output_dir(config.checkpoint.dirpath, FINAL_VAL_PREFIX),
                    label=FINAL_VAL_PREFIX,
                )


def launch(
    config_path: str,
    checkpoint_path: str | None = None,
    validate_only: bool = False,
    seed: int | None = None,
    fresh_run: bool = False,
    **overrides: Any,
):
    """
    CLI entry point.

    Args:
        config_path: Path to the JSON configuration file
        checkpoint_path: Optional path to checkpoint to resume training from
        validate_only: Run validation-only pass on a checkpoint (no training)
        seed: Optional compatibility assertion; must match the resolved config seed.
        fresh_run: Disable auto-resume and force training from scratch
        **overrides: Config overrides using dot notation

    Examples:
        # Basic training
        python train.py config/pretraining.json

        # With CLI overrides
        python train.py config/pretraining.json --model.d_model=256 --training.max_epochs=5

        # Quick debug run
        python train.py config/pretraining.json --training.limit_train_batches=10 --training.max_epochs=1

        # Assert the resolved config already carries the expected seed
        python train.py config/pretraining.json --seed=1337
    """
    main(
        config_path,
        checkpoint_path,
        validate_only,
        seed,
        fresh_run=fresh_run,
        **overrides,
    )


if __name__ == "__main__":
    fire.Fire(launch)
