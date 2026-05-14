from __future__ import annotations

import math
from typing import Any

import lightning.pytorch as L
import torch
from torch.optim.lr_scheduler import LambdaLR

from src.pretraining.fcmae.config import FCMAEModelConfig, FCMAETrainingConfig
from src.pretraining.fcmae.masking import random_patch_mask
from src.pretraining.fcmae.model import DenseMaskedImageModelingConvNeXtV2

_ALLOWED_COMPILE_MODES = {
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}


def compute_patch_ink_density_batched(pixel_values: torch.Tensor, patch_size: int) -> torch.Tensor:
    if pixel_values.ndim != 4:
        raise ValueError("pixel_values must have shape (B, C, H, W)")
    height, width = pixel_values.shape[-2:]
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("pixel_values dimensions must be divisible by patch_size")
    intensity = (pixel_values.float().mean(dim=1) + 1.0) / 2.0
    ink = (1.0 - intensity).clamp_(0.0, 1.0)
    grid_h = height // patch_size
    grid_w = width // patch_size
    return ink.reshape(-1, grid_h, patch_size, grid_w, patch_size).mean(dim=(2, 4))


def _resolve_preview_cadence(full_config: dict[str, Any] | None) -> int:
    if not full_config:
        return 0
    logging_cfg = full_config.get("logging") if isinstance(full_config, dict) else None
    if not isinstance(logging_cfg, dict):
        return 0
    if not logging_cfg.get("wandb_enabled", False):
        return 0
    if not logging_cfg.get("log_reconstructions", False):
        return 0
    cadence = logging_cfg.get("log_reconstruction_every_n_steps", 0)
    try:
        return max(0, int(cadence))
    except (TypeError, ValueError):
        return 0


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
        self._compiled_forward_model: torch.nn.Module | None = None
        self._compile_initialized = False
        self._latest_preview: dict[str, Any] | None = None
        self._latest_val_preview: dict[str, Any] | None = None
        self.full_config = full_config
        self._preview_cadence = _resolve_preview_cadence(full_config)
        self.save_hyperparameters(
            {
                "model_config": model_config.model_dump(),
                "training_config": training_config.model_dump(),
                "full_config": full_config,
            },
            ignore=["model"],
        )

    def _forward_model(self) -> torch.nn.Module:
        return self._compiled_forward_model or self.model

    def _status_print(self, message: str) -> None:
        if self._trainer is None:
            return
        self.print(message)

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

    def _log_validation_metric(self, name: str, value: Any, *, prog_bar: bool = False) -> None:
        if self._trainer is None:
            return
        self.log(
            name,
            value,
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar,
            batch_size=self.training_config.batch_size,
            sync_dist=True,
        )

    def _compute_ink_density(self, batch: dict[str, Any]) -> torch.Tensor:
        pixel_values = batch["pixel_values"]
        ink_density = batch.get("ink_density")
        if ink_density is None:
            ink_density = compute_patch_ink_density_batched(
                pixel_values, self.model_config.patch_size
            )
        return ink_density

    def _validation_mask(
        self,
        pixel_values: torch.Tensor,
        valid_patch_mask: torch.Tensor | None,
        ink_density: torch.Tensor | None,
        *,
        batch_idx: int,
    ) -> torch.Tensor:
        grid_h = pixel_values.shape[-2] // self.model_config.patch_size
        grid_w = pixel_values.shape[-1] // self.model_config.patch_size
        seed = int(self.training_config.validation_seed) + int(batch_idx)
        devices = []
        if pixel_values.device.type == "cuda" and pixel_values.device.index is not None:
            devices = [pixel_values.device.index]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            return random_patch_mask(
                pixel_values.shape[0],
                grid_h,
                grid_w,
                self.model_config.mask_ratio,
                pixel_values.device,
                valid_patch_mask=valid_patch_mask,
                patch_weights=ink_density,
                bias_strength=self.model_config.ink_bias_strength,
            )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        pixel_values = batch["pixel_values"]
        ink_density = self._compute_ink_density(batch)
        grid_h = pixel_values.shape[-2] // self.model_config.patch_size
        grid_w = pixel_values.shape[-1] // self.model_config.patch_size
        mask = random_patch_mask(
            pixel_values.shape[0],
            grid_h,
            grid_w,
            self.model_config.mask_ratio,
            pixel_values.device,
            valid_patch_mask=batch.get("valid_patch_mask"),
            patch_weights=ink_density,
            bias_strength=self.model_config.ink_bias_strength,
        )
        output = self._forward_model()(
            pixel_values,
            valid_patch_mask=batch.get("valid_patch_mask"),
            ink_density=ink_density,
            mask=mask,
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
        if output.masked_ink_density is not None:
            self._log_if_attached("train/masked_ink_density", output.masked_ink_density)
        if output.masked_background_loss is not None:
            self._log_if_attached(
                "train/loss_background_heavy_masked_patches",
                output.masked_background_loss,
            )
        if output.masked_foreground_loss is not None:
            self._log_if_attached(
                "train/loss_foreground_heavy_masked_patches",
                output.masked_foreground_loss,
            )
        if output.masked_background_patch_ratio is not None:
            self._log_if_attached(
                "train/masked_background_heavy_patch_ratio",
                output.masked_background_patch_ratio,
            )
        if output.masked_foreground_patch_ratio is not None:
            self._log_if_attached(
                "train/masked_foreground_heavy_patch_ratio",
                output.masked_foreground_patch_ratio,
            )
        valid_patch_mask = batch.get("valid_patch_mask")
        if valid_patch_mask is None:
            valid_patch_ratio = torch.ones((), device=output.loss.device)
        else:
            valid_patch_ratio = valid_patch_mask.to(device=output.loss.device).float().mean()
        self._log_if_attached("train/valid_patch_ratio", valid_patch_ratio)
        if self._should_snapshot_preview():
            self._latest_preview = {
                "pixel_values": pixel_values.detach(),
                "pred_patches": output.pred_patches.detach(),
                "mask": output.mask.detach(),
                "valid_patch_mask": (
                    None if output.valid_patch_mask is None else output.valid_patch_mask.detach()
                ),
                "norm_pix_loss": self.model_config.norm_pix_loss,
            }
        optimizer = None
        if self._trainer is not None:
            optimizer = self.optimizers(use_pl_optimizer=False)
        if optimizer is not None:
            self._log_if_attached("train/lr", optimizer.param_groups[0]["lr"])
        return output.loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        pixel_values = batch["pixel_values"]
        valid_patch_mask = batch.get("valid_patch_mask")
        ink_density = self._compute_ink_density(batch)
        mask = self._validation_mask(
            pixel_values,
            valid_patch_mask,
            ink_density,
            batch_idx=batch_idx,
        )
        output = self._forward_model()(
            pixel_values,
            valid_patch_mask=valid_patch_mask,
            ink_density=ink_density,
            mask=mask,
        )
        self._log_validation_metric("val/loss", output.loss, prog_bar=True)
        self._log_validation_metric("val/mask_ratio", self.model_config.mask_ratio)
        self._log_validation_metric(
            "val/masked_foreground_ratio",
            output.masked_foreground_ratio,
        )
        self._log_validation_metric(
            "val/samples_skipped_no_valid_patches",
            output.samples_skipped_no_valid_patches.float(),
        )
        if output.masked_ink_density is not None:
            self._log_validation_metric("val/masked_ink_density", output.masked_ink_density)
        if output.masked_background_loss is not None:
            self._log_validation_metric(
                "val/loss_background_heavy_masked_patches",
                output.masked_background_loss,
            )
        if output.masked_foreground_loss is not None:
            self._log_validation_metric(
                "val/loss_foreground_heavy_masked_patches",
                output.masked_foreground_loss,
            )
        if output.masked_background_patch_ratio is not None:
            self._log_validation_metric(
                "val/masked_background_heavy_patch_ratio",
                output.masked_background_patch_ratio,
            )
        if output.masked_foreground_patch_ratio is not None:
            self._log_validation_metric(
                "val/masked_foreground_heavy_patch_ratio",
                output.masked_foreground_patch_ratio,
            )
        if valid_patch_mask is None:
            valid_patch_ratio = torch.ones((), device=output.loss.device)
        else:
            valid_patch_ratio = valid_patch_mask.to(device=output.loss.device).float().mean()
        self._log_validation_metric("val/valid_patch_ratio", valid_patch_ratio)
        if batch_idx == 0:
            self._latest_val_preview = {
                "pixel_values": pixel_values.detach(),
                "pred_patches": output.pred_patches.detach(),
                "mask": output.mask.detach(),
                "valid_patch_mask": (
                    None if output.valid_patch_mask is None else output.valid_patch_mask.detach()
                ),
                "norm_pix_loss": self.model_config.norm_pix_loss,
            }
        return output.loss

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        compile_prefix = "_compiled_forward_model._orig_mod."
        state_dict = checkpoint.get("state_dict", {})
        if any(compile_prefix in key for key in state_dict):
            checkpoint["state_dict"] = {
                key: value for key, value in state_dict.items() if compile_prefix not in key
            }
            checkpoint["optimizer_states"] = []
            checkpoint["lr_schedulers"] = []
            self._status_print(
                "Stripped torch.compile artifacts from checkpoint. "
                "Optimizer and scheduler state reset (model weights preserved)."
            )

    def on_fit_start(self) -> None:
        if self._compile_initialized:
            return
        self._compile_initialized = True

        if not bool(self.training_config.compile_model):
            return

        compile_mode = self.training_config.compile_mode
        if compile_mode not in _ALLOWED_COMPILE_MODES:
            raise ValueError(
                f"Invalid training.compile_mode='{compile_mode}'. "
                f"Expected one of: {sorted(_ALLOWED_COMPILE_MODES)}."
            )

        disable_gc = getattr(self.model.encoder, "gradient_checkpointing_disable", None)
        if callable(disable_gc):
            disable_gc()
            self._status_print(
                "Disabled FCMAE encoder gradient checkpointing for torch.compile stability."
            )

        self._status_print(
            f"Compiling FCMAE forward path with torch.compile(mode='{compile_mode}')..."
        )
        try:
            self._compiled_forward_model = torch.compile(self.model, mode=compile_mode)
        except Exception as exc:
            raise RuntimeError(
                "torch.compile initialization failed with "
                f"training.compile_mode='{compile_mode}'. "
                "Set training.compile_model=false to continue in eager mode."
            ) from exc

        self._status_print("torch.compile enabled for FCMAE training/validation forward path.")

    def _should_snapshot_preview(self) -> bool:
        if self._preview_cadence <= 0:
            return False
        if self._trainer is None:
            return False
        return ((int(self.trainer.global_step) + 1) % self._preview_cadence) == 0

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
