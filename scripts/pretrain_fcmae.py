from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from src.pretraining.fcmae.config import FCMAEConfig
from src.pretraining.fcmae.data import FCMAEDataModule
from src.pretraining.fcmae.lightning_module import FCMAEPretrainer


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense FCMAE-style ConvNeXtV2 pretraining")
    parser.add_argument("config_path")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    config_dict = json.loads(Path(args.config_path).read_text())
    config = FCMAEConfig.model_validate(_apply_overrides(config_dict, args.overrides))

    L.seed_everything(config.training.seed, workers=True)
    datamodule = FCMAEDataModule(
        config.data,
        patch_size=config.model.patch_size,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )
    module = FCMAEPretrainer(config.model, config.training)

    checkpoint = ModelCheckpoint(
        dirpath=config.checkpoint.dirpath,
        filename=config.checkpoint.filename,
        save_last=config.checkpoint.save_last,
        save_top_k=config.checkpoint.save_top_k,
    )
    trainer = L.Trainer(
        max_steps=config.training.max_steps,
        accelerator="auto",
        devices="auto",
        precision=config.training.precision,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        callbacks=[checkpoint],
        log_every_n_steps=1,
        default_root_dir=config.checkpoint.dirpath,
    )
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()

