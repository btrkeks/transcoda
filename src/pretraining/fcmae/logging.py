from __future__ import annotations

import logging
from typing import Any

import lightning.pytorch as L
import torch
from lightning.pytorch.loggers import WandbLogger

from src.pretraining.fcmae.masking import patchify, unpatchify, upsample_mask

log = logging.getLogger(__name__)


class FCMAEReconstructionLogger(L.Callback):
    """Log lightweight FCMAE reconstruction previews to W&B."""

    def __init__(
        self,
        *,
        every_n_steps: int,
        max_batches: int | None,
        max_images: int,
    ) -> None:
        super().__init__()
        if every_n_steps < 1:
            raise ValueError("every_n_steps must be >= 1")
        if max_batches is not None and max_batches < 1:
            raise ValueError("max_batches must be None or >= 1")
        if max_images < 1:
            raise ValueError("max_images must be >= 1")
        self.every_n_steps = every_n_steps
        self.max_batches = max_batches
        self.max_images = max_images
        self._logged_batches = 0
        self._last_logged_step: int | None = None

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not getattr(trainer, "is_global_zero", True):
            return
        logger = trainer.logger
        if not isinstance(logger, WandbLogger):
            return
        if self.max_batches is not None and self._logged_batches >= self.max_batches:
            return
        global_step = int(trainer.global_step)
        if global_step <= 0 or global_step % self.every_n_steps != 0:
            return
        if self._last_logged_step == global_step:
            return

        payload = getattr(pl_module, "_latest_preview", None)
        if not payload:
            return
        pl_module._latest_preview = None

        try:
            self._log_payload(logger, payload, global_step=global_step)
            self._logged_batches += 1
            self._last_logged_step = global_step
        except Exception as exc:  # pragma: no cover - preview logging is best effort.
            log.warning("FCMAE reconstruction preview logging failed: %s", exc)

    @torch.no_grad()
    def _log_payload(
        self,
        logger: WandbLogger,
        payload: dict[str, Any],
        *,
        global_step: int,
    ) -> None:
        import wandb

        pixel_values = payload["pixel_values"].detach().cpu()
        pred_patches = payload["pred_patches"].detach().cpu()
        mask = payload["mask"].detach().cpu()
        limit = min(self.max_images, pixel_values.shape[0])
        patch_size = pixel_values.shape[-2] // mask.shape[-2]
        grid_h, grid_w = mask.shape[-2:]
        if payload.get("norm_pix_loss", False):
            target_patches = patchify(pixel_values, patch_size)
            mean = target_patches.mean(dim=-1, keepdim=True)
            var = target_patches.var(dim=-1, keepdim=True)
            pred_patches = pred_patches * (var + 1.0e-6).sqrt() + mean
        reconstruction = unpatchify(pred_patches, grid_h, grid_w, patch_size)

        images = []
        for idx in range(limit):
            original = (pixel_values[idx] + 1.0).div(2.0).clamp(0.0, 1.0)
            reconstructed = (reconstruction[idx] + 1.0).div(2.0).clamp(0.0, 1.0)
            pixel_mask = upsample_mask(mask[idx : idx + 1], patch_size)[0].unsqueeze(0)
            masked = torch.where(pixel_mask, torch.ones_like(original), original)
            filled = torch.where(pixel_mask, reconstructed, original)
            overlay = original.clone()
            overlay[0] = torch.where(pixel_mask[0], torch.ones_like(overlay[0]), overlay[0])
            overlay[1] = torch.where(pixel_mask[0], overlay[1] * 0.35, overlay[1])
            overlay[2] = torch.where(pixel_mask[0], overlay[2] * 0.35, overlay[2])

            images.append(wandb.Image(original.permute(1, 2, 0).numpy(), caption=f"original/{idx}"))
            images.append(wandb.Image(masked.permute(1, 2, 0).numpy(), caption=f"masked/{idx}"))
            images.append(
                wandb.Image(filled.permute(1, 2, 0).numpy(), caption=f"reconstruction/{idx}")
            )
            images.append(
                wandb.Image(overlay.permute(1, 2, 0).numpy(), caption=f"mask_overlay/{idx}")
            )

        if images:
            logger.experiment.log(
                {"train/reconstruction_preview": images},
                step=global_step,
            )
