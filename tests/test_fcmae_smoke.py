from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from src.pretraining.fcmae.config import (
    FCMAEDataConfig,
    FCMAEModelConfig,
    FCMAETrainingConfig,
)
from src.pretraining.fcmae.data import FCMAEImageDataset
from src.pretraining.fcmae.lightning_module import FCMAEPretrainer
from src.pretraining.fcmae.logging import FCMAEReconstructionLogger
from src.pretraining.fcmae.masking import patchify, random_patch_mask, unpatchify, upsample_mask
from src.pretraining.fcmae.model import DenseMaskedImageModelingConvNeXtV2


class TinyEncoder(torch.nn.Module):
    def __init__(self, out_channels: int = 8) -> None:
        super().__init__()
        self.net = torch.nn.Conv2d(3, out_channels, kernel_size=32, stride=32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _write_image(path: Path, size: tuple[int, int]) -> None:
    width, height = size
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    image[height // 4 : height // 2, width // 5 : width // 2] = 0
    Image.fromarray(image).save(path)


def test_patchify_unpatchify_round_trip_rectangular() -> None:
    x = torch.randn(2, 3, 64, 96)
    patches = patchify(x, patch_size=16)
    restored = unpatchify(patches, grid_h=4, grid_w=6, patch_size=16)
    assert torch.allclose(restored, x)


def test_random_patch_mask_counts_without_valid_mask() -> None:
    mask = random_patch_mask(2, 4, 5, 0.6, torch.device("cpu"))
    assert mask.dtype == torch.bool
    assert mask.sum(dim=(1, 2)).tolist() == [12, 12]


def test_random_patch_mask_respects_valid_patch_mask() -> None:
    valid = torch.zeros(2, 4, 5, dtype=torch.bool)
    valid[:, :2, :2] = True
    mask = random_patch_mask(2, 4, 5, 0.6, torch.device("cpu"), valid_patch_mask=valid)
    assert (mask & ~valid).sum() == 0
    assert mask.sum(dim=(1, 2)).tolist() == [2, 2]


def test_random_patch_mask_masks_single_valid_patch_by_design() -> None:
    valid = torch.zeros(1, 4, 5, dtype=torch.bool)
    valid[0, 2, 3] = True
    mask = random_patch_mask(1, 4, 5, 0.6, torch.device("cpu"), valid_patch_mask=valid)
    assert mask.sum().item() == 1
    assert mask[0, 2, 3]


def test_upsample_mask_rectangular_shape() -> None:
    mask = torch.zeros(2, 3, 5, dtype=torch.bool)
    upsampled = upsample_mask(mask, scale=4)
    assert upsampled.shape == (2, 12, 20)


def test_fcmae_dataset_and_forward_backward(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_image(image_dir / "a.png", (80, 120))
    _write_image(image_dir / "b.png", (150, 60))

    data_config = FCMAEDataConfig(
        image_dir=str(image_dir),
        image_height=128,
        image_width=96,
    )
    dataset = FCMAEImageDataset(data_config, patch_size=32)
    batch = [dataset[0], dataset[1]]
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    valid_patch_mask = torch.stack([item["valid_patch_mask"] for item in batch])

    model_config = FCMAEModelConfig(
        patch_size=32,
        mask_ratio=0.6,
        decoder_dim=16,
        decoder_depth=1,
        norm_pix_loss=True,
    )
    model = DenseMaskedImageModelingConvNeXtV2(
        model_config,
        encoder=TinyEncoder(out_channels=8),
        encoder_output_dim=8,
        encoder_stride=32,
    )
    output = model(pixel_values, valid_patch_mask=valid_patch_mask)

    assert torch.isfinite(output.loss)
    assert output.pred_patches.shape[-1] == 32 * 32 * 3
    assert output.target_patches.shape == output.pred_patches.shape
    assert output.mask.dtype == torch.bool
    assert output.valid_patch_mask is not None

    module = FCMAEPretrainer(
        model_config,
        FCMAETrainingConfig(batch_size=2, num_workers=0, max_steps=1, warmup_steps=0),
        model=model,
    )
    loss = module.training_step(
        {"pixel_values": pixel_values, "valid_patch_mask": valid_patch_mask},
        0,
    )
    loss.backward()
    grads = [param.grad for param in module.parameters() if param.grad is not None]
    assert grads
    assert any(torch.isfinite(grad).all() for grad in grads)
    assert module._latest_preview is not None
    assert "pred_patches" in module._latest_preview
    assert module._latest_preview["norm_pix_loss"] is True


def test_reconstruction_logger_logs_filled_reconstruction(monkeypatch) -> None:
    logged_images = []

    class FakeImage:
        def __init__(self, data: np.ndarray, *, caption: str) -> None:
            self.data = data
            self.caption = caption
            logged_images.append(self)

    class FakeExperiment:
        def __init__(self) -> None:
            self.logged = []

        def log(self, payload: dict[str, object], *, step: int) -> None:
            self.logged.append((payload, step))

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(Image=FakeImage),
    )

    pixel_values = torch.zeros(1, 3, 4, 4)
    pixel_values[:, :, :2, :2] = -1.0
    pred_patches = patchify(pixel_values, patch_size=2)
    mask = torch.zeros(1, 2, 2, dtype=torch.bool)
    mask[:, 0, 0] = True

    logger = SimpleNamespace(experiment=FakeExperiment())
    callback = FCMAEReconstructionLogger(every_n_steps=1, max_batches=None, max_images=1)
    callback._log_payload(
        logger,
        {
            "pixel_values": pixel_values,
            "pred_patches": pred_patches,
            "mask": mask,
            "norm_pix_loss": False,
        },
        global_step=500,
    )

    assert [image.caption for image in logged_images] == [
        "original/0",
        "masked/0",
        "reconstruction/0",
        "mask_overlay/0",
    ]
    assert logger.experiment.logged[0][1] == 500
    reconstruction = logged_images[2].data
    assert reconstruction.shape == (4, 4, 3)
    assert np.allclose(reconstruction[:2, :2], 0.0)


def test_pretrainer_accepts_checkpoint_hparam_dicts() -> None:
    model_config = FCMAEModelConfig(patch_size=32, decoder_dim=16, decoder_depth=1)
    training_config = FCMAETrainingConfig(batch_size=2, num_workers=0, max_steps=1)
    model = DenseMaskedImageModelingConvNeXtV2(
        model_config,
        encoder=TinyEncoder(out_channels=8),
        encoder_output_dim=8,
        encoder_stride=32,
    )
    module = FCMAEPretrainer(
        model_config.model_dump(),
        training_config.model_dump(),
        model=model,
    )
    assert isinstance(module.model_config, FCMAEModelConfig)
    assert isinstance(module.training_config, FCMAETrainingConfig)
