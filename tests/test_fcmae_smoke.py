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
    def __init__(self, embed_channels: int = 4, out_channels: int = 8) -> None:
        super().__init__()
        self.embeddings = torch.nn.Conv2d(3, embed_channels, kernel_size=4, stride=4)
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(embed_channels, out_channels, kernel_size=8, stride=8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embeddings(x))


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


def test_random_patch_mask_ignores_weights_when_bias_strength_zero() -> None:
    weights = torch.zeros(1, 4, 5)
    weights[0, 0, 0] = 1.0
    torch.manual_seed(0)
    biased = random_patch_mask(
        1,
        4,
        5,
        0.5,
        torch.device("cpu"),
        patch_weights=weights,
        bias_strength=0.0,
    )
    torch.manual_seed(0)
    uniform = random_patch_mask(1, 4, 5, 0.5, torch.device("cpu"))
    assert torch.equal(biased, uniform)


def test_random_patch_mask_prefers_high_weight_patches() -> None:
    grid_h, grid_w = 8, 8
    weights = torch.zeros(1, grid_h, grid_w)
    high_weight = torch.zeros(grid_h, grid_w, dtype=torch.bool)
    high_weight[:4, :4] = True
    weights[0][high_weight] = 1.0

    hits = 0
    trials = 200
    torch.manual_seed(0)
    for _ in range(trials):
        mask = random_patch_mask(
            1,
            grid_h,
            grid_w,
            0.25,
            torch.device("cpu"),
            patch_weights=weights,
            bias_strength=4.0,
        )
        hits += int((mask[0] & high_weight).sum().item())
    total_masked = trials * int(round(0.25 * grid_h * grid_w))
    biased_share = hits / total_masked
    uniform_share = high_weight.sum().item() / (grid_h * grid_w)
    assert biased_share > uniform_share + 0.4


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
        image_height=130,
        image_width=98,
    )
    dataset = FCMAEImageDataset(data_config, patch_size=32)
    batch = [dataset[0], dataset[1]]
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    valid_patch_mask = torch.stack([item["valid_patch_mask"] for item in batch])
    ink_density = torch.stack([item["ink_density"] for item in batch])
    assert pixel_values.shape == (2, 3, 160, 128)
    assert torch.all(pixel_values[:, :, 130:, :] == 1.0)
    assert torch.all(pixel_values[:, :, :, 98:] == 1.0)
    assert valid_patch_mask.shape == (2, 5, 4)
    assert not valid_patch_mask[:, -1, :].any()
    assert not valid_patch_mask[:, :, -1].any()
    assert ink_density.shape == (2, 5, 4)
    assert ((ink_density >= 0) & (ink_density <= 1)).all()

    model_config = FCMAEModelConfig(
        patch_size=32,
        mask_ratio=0.6,
        decoder_dim=16,
        decoder_depth=1,
        norm_pix_loss=True,
        ink_bias_strength=0.3,
    )
    model = DenseMaskedImageModelingConvNeXtV2(
        model_config,
        encoder=TinyEncoder(out_channels=8),
        encoder_output_dim=8,
        encoder_stride=32,
    )
    output = model(
        pixel_values,
        valid_patch_mask=valid_patch_mask,
        ink_density=ink_density,
    )

    assert torch.isfinite(output.loss)
    assert output.pred_patches.shape[-1] == 32 * 32 * 3
    assert output.target_patches.shape == output.pred_patches.shape
    assert output.mask.dtype == torch.bool
    assert output.valid_patch_mask is not None
    assert output.masked_ink_density is not None
    assert torch.isfinite(output.masked_ink_density)
    assert model.mask_token.shape == (1, 4, 1, 1)

    module = FCMAEPretrainer(
        model_config,
        FCMAETrainingConfig(batch_size=2, num_workers=0, max_steps=1, warmup_steps=0),
        model=model,
    )
    loss = module.training_step(
        {
            "pixel_values": pixel_values,
            "valid_patch_mask": valid_patch_mask,
            "ink_density": ink_density,
        },
        0,
    )
    loss.backward()
    assert model.mask_token.grad is not None
    assert torch.isfinite(model.mask_token.grad).all()
    grads = [param.grad for param in module.parameters() if param.grad is not None]
    assert grads
    assert any(torch.isfinite(grad).all() for grad in grads)
    assert module._latest_preview is not None
    assert "pred_patches" in module._latest_preview
    assert module._latest_preview["norm_pix_loss"] is True


def test_fcmae_dataset_skips_empty_image_files(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_image(image_dir / "valid.png", (80, 120))
    (image_dir / "empty.webp").write_bytes(b"")

    dataset = FCMAEImageDataset(
        FCMAEDataConfig(
            image_dir=str(image_dir),
            image_height=128,
            image_width=96,
        ),
        patch_size=32,
    )

    assert [path.name for path in dataset.paths] == ["valid.png"]


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


def test_reconstruction_logger_logs_once_per_global_step(monkeypatch) -> None:
    class FakeExperiment:
        def __init__(self) -> None:
            self.logs = []

        def log(self, payload: dict[str, object], *, step: int) -> None:
            self.logs.append((payload, step))

    class FakeWandbLogger:
        def __init__(self) -> None:
            self.experiment = FakeExperiment()

    monkeypatch.setattr("src.pretraining.fcmae.logging.WandbLogger", FakeWandbLogger)
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(Image=lambda data, *, caption: SimpleNamespace(data=data, caption=caption)),
    )

    pixel_values = torch.zeros(1, 3, 4, 4)
    pred_patches = patchify(pixel_values, patch_size=2)
    payload = {
        "pixel_values": pixel_values,
        "pred_patches": pred_patches,
        "mask": torch.zeros(1, 2, 2, dtype=torch.bool),
        "norm_pix_loss": False,
    }
    logger = FakeWandbLogger()
    trainer = SimpleNamespace(logger=logger, global_step=500)
    pl_module = SimpleNamespace(_latest_preview=payload)
    callback = FCMAEReconstructionLogger(every_n_steps=500, max_batches=2, max_images=1)

    for _ in range(4):
        pl_module._latest_preview = payload
        callback.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

    trainer.global_step = 1000
    pl_module._latest_preview = payload
    callback.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

    assert [step for _payload, step in logger.experiment.logs] == [500, 1000]
    assert callback._logged_batches == 2


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
