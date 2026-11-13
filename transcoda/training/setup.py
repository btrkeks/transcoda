import os
from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler
from rich.console import Console

from transcoda.callbacks.durable_checkpoint import DurableCheckpoint
from transcoda.callbacks.epoch_timing import EpochTimingCallback
from transcoda.callbacks.fit_validation_warmup import FitValidationWarmupCallback
from transcoda.callbacks.gated_checkpoint import GatedModelCheckpoint
from transcoda.callbacks.learning_rate import CompactLearningRateMonitor
from transcoda.callbacks.log_progress import LogProgressCallback
from transcoda.callbacks.validation_prediction_writer import ValidationPredictionWriter
from transcoda.callbacks.verosim_validation_metric import VeroSimValidationMetricCallback
from transcoda.callbacks.wandb_visualizer import WandbVisualizerCallback
from transcoda.config import ExperimentConfig, Training
from transcoda.validation.metrics.stored_targets import target_lookup_for
from transcoda.validation.metrics.verosim import VeroSimBinaryResolver, VeroSimOMRNEDBackend
from transcoda.validation.schedule import FullValidationEveryNStepsSchedule
from transcoda.validation.selection import OMRNEDCheckpointSelectionPolicy

console = Console()
REPO_ROOT = Path(__file__).resolve().parents[2]


def _derive_wandb_group(dirpath: str) -> str:
    """Derive a stable non-empty W&B group name from checkpoint dirpath."""
    group = Path(dirpath).name.strip()
    if not group or group in {".", "/"}:
        return "default"
    return group


def wandb_run_identity(dirpath: str, env) -> dict[str, str]:
    """A lab job names its W&B run after the lab run id so retries resume it."""
    run_id = str(env.get("LAB_RUN_ID") or "").strip()
    if run_id:
        return {"id": run_id, "resume": "allow", "group": run_id}
    return {"group": _derive_wandb_group(dirpath)}


def setup_callbacks(
    config: ExperimentConfig,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    i2w: dict[int, str],
):
    """Set up training callbacks including checkpointing and early stopping."""
    training_cfg = config.training if config.training is not None else Training()
    lr_monitor = CompactLearningRateMonitor()

    # Only add early stopping if validation is enabled
    validation_enabled = (
        config.training is None
        or config.training.limit_val_batches is None
        or config.training.limit_val_batches > 0
    )
    early_stopping = None
    if validation_enabled and training_cfg.early_stopping_enabled:
        early_stopping = EarlyStopping(
            monitor=config.checkpoint.monitor,
            min_delta=training_cfg.early_stopping_min_delta,
            patience=training_cfg.early_stopping_patience,
            mode=config.checkpoint.mode,
            verbose=True,
        )

    # Under a lab step budget the durable checkpoints are the only ones read: no
    # selection, no final validation and no auto-resume from last.ckpt.
    durable_only = (
        config.checkpoint.durable_every_steps is not None
        and not config.validation.selection.enabled
        and training_cfg.skip_final_validation
        and not config.checkpoint.auto_resume
    )
    checkpointers: list[ModelCheckpoint] = []
    checkpointer = None
    if not durable_only:
        checkpointer = ModelCheckpoint(
            dirpath=config.checkpoint.dirpath,
            filename=config.checkpoint.filename,
            monitor=config.checkpoint.monitor,
            mode=config.checkpoint.mode,
            save_top_k=config.checkpoint.save_top_k,
            save_last=config.checkpoint.save_last,
            verbose=config.checkpoint.verbose,
        )
        # Periodic checkpoint every N epochs (independent of validation)
        periodic_checkpointer = ModelCheckpoint(
            dirpath=config.checkpoint.dirpath,
            filename=f"{config.checkpoint.filename}_epoch_{{epoch:03d}}",
            every_n_epochs=5,  # Save every 5 epochs
            save_top_k=-1,  # Keep all periodic checkpoints
            verbose=True,
        )
        checkpointers = [checkpointer, periodic_checkpointer]

    # Validation visualization callback for W&B logging
    wandb_visualizer = WandbVisualizerCallback(
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        i2w=i2w,
        n_best=5,
        n_worst=20,
    )

    # Per-sample prediction dump; only active when the module enables
    # validation prediction recording (final validation / validate-only runs).
    prediction_writer = ValidationPredictionWriter(
        output_root=config.checkpoint.dirpath,
        i2w=i2w,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        prediction_set_label=config.checkpoint.run_name,
    )

    verosim_metric_callback = None
    verosim_cfg = config.validation.metrics.verosim
    schedule_cfg = config.validation.schedule
    if validation_enabled and verosim_cfg.enabled:
        verosim_metric_callback = VeroSimValidationMetricCallback(
            output_root=config.checkpoint.dirpath,
            backend=VeroSimOMRNEDBackend(
                binary_resolver=VeroSimBinaryResolver(repo_root=REPO_ROOT),
                kern_parser_binary=verosim_cfg.kern_parser_binary,
                verosim_binary=verosim_cfg.verosim_binary,
                mode=verosim_cfg.mode,
                jobs=verosim_cfg.jobs,
                failure_score=verosim_cfg.failure_score,
                target_lookup=target_lookup_for(
                    verosim_cfg.target_text, config.data.validation_paths
                ),
                exclude_val_sets=verosim_cfg.exclude_val_sets or (),
            ),
            schedule=FullValidationEveryNStepsSchedule(
                every_n_steps=schedule_cfg.full_validation_every_n_steps,
            ),
        )

    omr_ned_checkpointer = None
    selection_cfg = config.validation.selection
    if (
        validation_enabled
        and verosim_cfg.enabled
        and selection_cfg.enabled
    ):
        omr_ned_checkpointer = GatedModelCheckpoint(
            selection_policy=OMRNEDCheckpointSelectionPolicy(
                primary_metric=selection_cfg.primary,
                min_step=selection_cfg.omr_ned_min_step,
                min_samples=selection_cfg.min_samples,
                max_verosim_failure_rate=selection_cfg.max_verosim_failure_rate,
                max_prediction_truncation_rate=selection_cfg.max_prediction_truncation_rate,
                max_runaway_rate=selection_cfg.max_runaway_rate,
            ),
            dirpath=config.checkpoint.dirpath,
            filename=f"{config.checkpoint.filename}_omr_ned",
            monitor=selection_cfg.primary,
            mode=selection_cfg.mode,
            save_top_k=1,
            save_last=False,
            verbose=config.checkpoint.verbose,
        )

    callbacks = [
        *checkpointers,
        lr_monitor,
        LogProgressCallback(
            train_every_n_steps=training_cfg.progress_train_every_n_steps,
            train_interval_seconds=training_cfg.progress_train_interval_seconds,
            val_percent_interval=training_cfg.progress_val_percent_interval,
            enable_ascii_bar=training_cfg.progress_enable_ascii_bar,
        ),
        EpochTimingCallback(),
        wandb_visualizer,
        prediction_writer,
    ]
    if validation_enabled and schedule_cfg.min_fit_validation_step > 0:
        callbacks.append(
            FitValidationWarmupCallback(
                min_validation_step=schedule_cfg.min_fit_validation_step,
            )
        )
    if verosim_metric_callback is not None:
        callbacks.append(verosim_metric_callback)
    if omr_ned_checkpointer is not None:
        callbacks.append(omr_ned_checkpointer)
    if early_stopping is not None:
        callbacks.append(early_stopping)
    if config.checkpoint.durable_every_steps is not None:
        callbacks.append(
            DurableCheckpoint(
                dirpath=config.checkpoint.dirpath,
                every_steps=config.checkpoint.durable_every_steps,
                target_steps=training_cfg.max_steps,
            )
        )

    return callbacks, checkpointer


def setup_logger(config: ExperimentConfig, config_dict: dict):
    """
    Set up Weights & Biases logger and log hyperparameters.

    Args:
        config: The parsed ExperimentConfig object.
        config_dict: The raw configuration dictionary from the JSON file.
    """
    # log_model=False to avoid slow checkpoint uploads after each validation
    # Checkpoints are still saved locally by ModelCheckpoint callback
    wandb_logger = WandbLogger(
        project=config.checkpoint.project,
        name=config.checkpoint.run_name,
        tags=config.checkpoint.tags or None,
        log_model=False,
        **wandb_run_identity(config.checkpoint.dirpath, os.environ),
    )

    # Log the entire configuration dictionary
    wandb_logger.log_hyperparams(params=config_dict)

    return wandb_logger


def setup_trainer(config: ExperimentConfig, callbacks, logger, num_train_batches: int | None = None):
    """Set up PyTorch Lightning Trainer with all configuration."""
    training_config = config.training if config.training is not None else Training()

    # Cap val_check_interval to the number of training batches to avoid Lightning error
    val_check_interval = training_config.val_check_interval
    if num_train_batches is not None and val_check_interval > num_train_batches:
        console.print(
            f"[yellow]val_check_interval ({val_check_interval}) > num_train_batches ({num_train_batches}), "
            f"capping to {num_train_batches}[/yellow]"
        )
        val_check_interval = num_train_batches

    trainer_kwargs = {
        "max_epochs": training_config.max_epochs,
        "val_check_interval": val_check_interval,
        "num_sanity_val_steps": 0,
        "logger": logger,
        "callbacks": callbacks,
        "precision": "bf16-true",
        "strategy": training_config.strategy,
        "devices": training_config.devices,
        "num_nodes": training_config.num_nodes,
        "log_every_n_steps": 10,
        # Without a ModelCheckpoint (durable-only budgets) Lightning must not add its own.
        "enable_checkpointing": any(isinstance(cb, ModelCheckpoint) for cb in callbacks),
    }

    if (
        training_config.accumulate_grad_batches is not None
        and training_config.accumulate_grad_batches > 1
    ):
        trainer_kwargs["accumulate_grad_batches"] = training_config.accumulate_grad_batches
        console.print(
            f"[green]Using gradient accumulation with {training_config.accumulate_grad_batches} batches."
        )
    if training_config.max_steps is not None:
        trainer_kwargs["max_steps"] = training_config.max_steps
    if training_config.min_steps is not None:
        trainer_kwargs["min_steps"] = training_config.min_steps
    if training_config.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = training_config.limit_train_batches
    if training_config.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = training_config.limit_val_batches
    if training_config.gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = training_config.gradient_clip_val
        trainer_kwargs["gradient_clip_algorithm"] = training_config.gradient_clip_algorithm

    # Add profiler if enabled
    if training_config.enable_profiling:
        dirpath = training_config.profiler_dirpath or "./profiler_logs"
        profiler = PyTorchProfiler(
            dirpath=dirpath,
            filename="perf_logs",
            schedule=torch.profiler.schedule(
                wait=training_config.profiler_wait_steps,
                warmup=training_config.profiler_warmup_steps,
                active=training_config.profiler_active_steps,
                repeat=training_config.profiler_repeat,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(dirpath),
            record_shapes=True,
            # profile_memory/with_stack add enough device+host overhead to OOM
            # the baseline model, which runs at ~30.4/31.4 GiB on RTX 5090.
            profile_memory=False,
            with_stack=False,
        )
        trainer_kwargs["profiler"] = profiler
        console.print(f"[green]PyTorch profiler enabled. Output: {dirpath}[/green]")

    # Log trainer configuration for debugging
    console.print("[bold blue]Trainer Configuration:[/bold blue]")
    for key, value in trainer_kwargs.items():
        if key == "callbacks":
            console.print(f"  {key}:")
            for i, callback in enumerate(value):
                callback_name = callback.__class__.__name__
                if hasattr(callback, "every_n_epochs"):
                    console.print(
                        f"    {i + 1}. {callback_name} (every_n_epochs={callback.every_n_epochs})"
                    )
                elif hasattr(callback, "monitor"):
                    console.print(f"    {i + 1}. {callback_name} (monitor={callback.monitor})")
                else:
                    console.print(f"    {i + 1}. {callback_name}")
        else:
            console.print(f"  {key}: {value}")

    return Trainer(**trainer_kwargs)
