from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler
from rich.console import Console

from src.callbacks.epoch_timing import EpochTimingCallback
from src.callbacks.log_progress import LogProgressCallback
from src.callbacks.wandb_visualizer import WandbVisualizerCallback
from src.config import ExperimentConfig, Training

console = Console()


def _derive_wandb_group(dirpath: str) -> str:
    """Derive a stable non-empty W&B group name from checkpoint dirpath."""
    group = Path(dirpath).name.strip()
    if not group or group in {".", "/"}:
        return "default"
    return group


def setup_callbacks(
    config: ExperimentConfig,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    i2w: dict[int, str],
):
    """Set up training callbacks including checkpointing and early stopping."""
    training_cfg = config.training if config.training is not None else Training()
    lr_monitor = LearningRateMonitor(logging_interval="step")

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

    epoch_timing = EpochTimingCallback()

    # Validation visualization callback for W&B logging
    wandb_visualizer = WandbVisualizerCallback(
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        i2w=i2w,
        n_best=5,
        n_worst=20,
    )

    callbacks = [
        checkpointer,
        periodic_checkpointer,
        lr_monitor,
        epoch_timing,
        LogProgressCallback(
            train_every_n_steps=training_cfg.progress_train_every_n_steps,
            train_interval_seconds=training_cfg.progress_train_interval_seconds,
            val_percent_interval=training_cfg.progress_val_percent_interval,
            enable_ascii_bar=training_cfg.progress_enable_ascii_bar,
        ),
        wandb_visualizer,
    ]
    if early_stopping is not None:
        callbacks.append(early_stopping)

    return callbacks, checkpointer


def setup_logger(config: ExperimentConfig, config_dict: dict):
    """
    Set up Weights & Biases logger and log hyperparameters.

    Args:
        config: The parsed ExperimentConfig object.
        config_dict: The raw configuration dictionary from the JSON file.
    """
    group = _derive_wandb_group(config.checkpoint.dirpath)

    # log_model=False to avoid slow checkpoint uploads after each validation
    # Checkpoints are still saved locally by ModelCheckpoint callback
    wandb_logger = WandbLogger(
        project=config.checkpoint.project,
        group=group,
        name=config.checkpoint.run_name,
        tags=config.checkpoint.tags or None,
        log_model=False,
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
        "strategy": "auto",
        "devices": 1,
        "log_every_n_steps": 10,
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
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(dirpath),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,  # Captures Python call stacks as separate trace events
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
