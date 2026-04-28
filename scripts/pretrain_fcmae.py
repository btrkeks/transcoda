from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from scripts.export_fcmae_encoder import export_fcmae_encoder
from src.artifacts import collect_slurm
from src.pretraining.fcmae.config import FCMAEConfig
from src.pretraining.fcmae.data import FCMAEDataModule
from src.pretraining.fcmae.lightning_module import FCMAEPretrainer
from src.pretraining.fcmae.logging import FCMAEReconstructionLogger


def _parse_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _apply_overrides(config_dict: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must use key=value form, got {override!r}")
        key, raw_value = override.split("=", 1)
        target = config_dict
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = _parse_value(raw_value)
    return config_dict


def _derive_wandb_group(dirpath: str) -> str:
    group = Path(dirpath).name.strip()
    if not group or group in {".", "/"}:
        return "default"
    return group


def _slurm_config() -> dict[str, Any]:
    slurm = collect_slurm()
    if slurm is None:
        return {}
    values = {
        "slurm/job_id": slurm.job_id,
        "slurm/job_name": slurm.job_name,
        "slurm/partition": slurm.partition,
        "slurm/nodelist": slurm.nodelist,
        "slurm/cpus_per_task": slurm.cpus_per_task,
        "slurm/gpus_on_node": slurm.gpus_on_node,
        "slurm/gpu_binding": slurm.gpu_binding,
        "slurm/submit_host": slurm.submit_host,
        "slurm/cluster_name": slurm.cluster_name,
        "slurm/array_job_id": slurm.array_job_id,
        "slurm/array_task_id": slurm.array_task_id,
    }
    return {key: value for key, value in values.items() if value is not None}


def _is_rank_zero() -> bool:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0


def _setup_logger(config: FCMAEConfig, config_dict: dict[str, Any]) -> WandbLogger | bool:
    if not config.logging.wandb_enabled:
        return False

    logger = WandbLogger(
        project=config.logging.project,
        group=config.logging.group or _derive_wandb_group(config.checkpoint.dirpath),
        name=config.logging.run_name,
        tags=config.logging.tags or None,
        log_model=config.logging.log_model,
    )
    logger.log_hyperparams(config_dict)
    world_size = config.training.devices * config.training.num_nodes
    effective_batch_size = (
        config.training.batch_size * config.training.accumulate_grad_batches * world_size
    )
    resolved_lr = config.training.base_learning_rate * effective_batch_size / 256
    if _is_rank_zero():
        logger.experiment.config.update(
            {
                "scale/effective_batch_size": effective_batch_size,
                "scale/world_size": world_size,
                "optimization/base_learning_rate": config.training.base_learning_rate,
                "optimization/resolved_initial_learning_rate": resolved_lr,
                "data/pixels_per_sample": config.data.image_height * config.data.image_width * 3,
                **_slurm_config(),
            },
            allow_val_change=True,
        )
    return logger


def _build_checkpoint_callback(config: FCMAEConfig) -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath=config.checkpoint.dirpath,
        filename=config.checkpoint.filename,
        save_last=config.checkpoint.save_last,
        save_top_k=config.checkpoint.save_top_k,
        every_n_train_steps=config.checkpoint.every_n_train_steps,
        every_n_epochs=None,
        save_on_train_epoch_end=False,
    )


def _log_dataset_size(datamodule: FCMAEDataModule, logger: WandbLogger | bool) -> int:
    datamodule.setup("fit")
    if datamodule.train_set is None:
        raise RuntimeError("FCMAE datamodule did not initialize a train dataset")
    image_count = len(datamodule.train_set)
    print(f"FCMAE pretraining images used: {image_count}", flush=True)
    if isinstance(logger, WandbLogger) and _is_rank_zero():
        logger.experiment.config.update({"data/num_images_used": image_count}, allow_val_change=True)
    return image_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense FCMAE-style ConvNeXtV2 pretraining")
    parser.add_argument("config_path")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    config_dict = _apply_overrides(json.loads(Path(args.config_path).read_text()), args.overrides)
    config = FCMAEConfig.model_validate(config_dict)

    L.seed_everything(config.training.seed, workers=True)
    datamodule = FCMAEDataModule(
        config.data,
        patch_size=config.model.patch_size,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )
    module = FCMAEPretrainer(config.model, config.training, full_config=config.model_dump())

    callbacks: list[Any] = []
    callbacks.append(_build_checkpoint_callback(config))
    if config.logging.wandb_enabled and config.logging.log_reconstructions:
        callbacks.append(
            FCMAEReconstructionLogger(
                every_n_steps=config.logging.log_reconstruction_every_n_steps,
                max_batches=config.logging.log_reconstruction_max_batches,
                max_images=config.logging.log_reconstruction_max_images,
            )
        )

    logger = _setup_logger(config, config_dict)
    _log_dataset_size(datamodule, logger)
    trainer = L.Trainer(
        max_steps=config.training.max_steps,
        accelerator="auto",
        devices=config.training.devices,
        strategy=config.training.strategy,
        num_nodes=config.training.num_nodes,
        precision=config.training.precision,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        default_root_dir=config.checkpoint.dirpath,
    )
    trainer.fit(
        module,
        datamodule=datamodule,
        ckpt_path=config.training.resume_from_checkpoint,
    )

    if config.export.export_on_train_end and trainer.is_global_zero:
        checkpoint_path = Path(config.checkpoint.dirpath) / "last.ckpt"
        export_fcmae_encoder(
            checkpoint_path=checkpoint_path,
            output_dir=Path(config.export.output_dir or ""),
            overwrite=config.export.overwrite,
            validate=config.export.validate_export,
        )


if __name__ == "__main__":
    main()
