from __future__ import annotations

import math
from typing import Any

import lightning.pytorch as L
import torch
from torch.optim.lr_scheduler import LambdaLR

from src.pretraining.fcmae.config import FCMAEModelConfig, FCMAETrainingConfig
from src.pretraining.fcmae.model import DenseMaskedImageModelingConvNeXtV2


class FCMAEPretrainer(L.LightningModule):
    def __init__(
        self,
        model_config: FCMAEModelConfig | dict[str, Any],
        training_config: FCMAETrainingConfig | dict[str, Any],
        *,
        model: DenseMaskedImageModelingConvNeXtV2 | None = None,
        full_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        model_config = FCMAEModelConfig.model_validate(model_config)
        training_config = FCMAETrainingConfig.model_validate(training_config)
        self.model_config = model_config
        self.training_config = training_config
        self.model = model or DenseMaskedImageModelingConvNeXtV2(model_config)
        self._latest_preview: dict[str, Any] | None = None
        self.full_config = full_config
        self.save_hyperparameters(
            {
                "model_config": model_config.model_dump(),
                "training_config": training_config.model_dump(),
                "full_config": full_config,
            },
            ignore=["model"],
        )

    def _log_if_attached(self, name: str, value: Any, *, prog_bar: bool = False) -> None:
        if self._trainer is None:
            return
        self.log(
            name,
            value,
            on_step=True,
            prog_bar=prog_bar,
            batch_size=self.training_config.batch_size,
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        output = self.model(
            batch["pixel_values"],
            valid_patch_mask=batch.get("valid_patch_mask"),
        )
        self._log_if_attached("train/loss", output.loss, prog_bar=True)
        self._log_if_attached("train/mask_ratio", self.model_config.mask_ratio)
        self._log_if_attached(
            "train/masked_foreground_ratio",
            output.masked_foreground_ratio,
        )
        self._log_if_attached(
            "train/samples_skipped_no_valid_patches",
            output.samples_skipped_no_valid_patches.float(),
        )
        valid_patch_mask = batch.get("valid_patch_mask")
        if valid_patch_mask is None:
            valid_patch_ratio = torch.ones((), device=output.loss.device)
        else:
            valid_patch_ratio = valid_patch_mask.to(device=output.loss.device).float().mean()
        self._log_if_attached("train/valid_patch_ratio", valid_patch_ratio)
        self._latest_preview = {
            "pixel_values": batch["pixel_values"].detach(),
            "mask": output.mask.detach(),
            "valid_patch_mask": (
                None if output.valid_patch_mask is None else output.valid_patch_mask.detach()
            ),
        }
        optimizer = None
        if self._trainer is not None:
            optimizer = self.optimizers(use_pl_optimizer=False)
        if optimizer is not None:
            self._log_if_attached("train/lr", optimizer.param_groups[0]["lr"])
        return output.loss

    def _scaled_learning_rate(self) -> float:
        world_size = int(getattr(self.trainer, "world_size", 1) or 1)
        effective_batch = (
            self.training_config.batch_size
            * self.training_config.accumulate_grad_batches
            * world_size
        )
        return self.training_config.base_learning_rate * effective_batch / 256

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self._scaled_learning_rate(),
            weight_decay=self.training_config.weight_decay,
            betas=(0.9, 0.95),
        )

        max_steps = max(1, int(self.training_config.max_steps))
        warmup_steps = min(int(self.training_config.warmup_steps), max_steps)

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            if max_steps <= warmup_steps:
                return 1.0
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
