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
from src.pretraining.fcmae.data import FCMAEImageDataset, compute_patch_ink_density
from src.pretraining.fcmae.lightning_module import (
    FCMAEPretrainer,
    compute_patch_ink_density_batched,
)
from src.pretraining.fcmae.logging import FCMAEReconstructionLogger
from src.pretraining.fcmae.masking import patchify, random_patch_mask, unpatchify, upsample_mask
from src.pretraining.fcmae.model import (
    BACKGROUND_HEAVY_INK_DENSITY_THRESHOLD,
    FOREGROUND_HEAVY_INK_DENSITY_THRESHOLD,
    DenseMaskedImageModelingConvNeXtV2,
)


class TinyEncoder(torch.nn.Module):
    def __init__(self, embed_channels: int = 4, out_channels: int = 8) -> None:
        super().__init__()
        self.embeddings = torch.nn.Conv2d(3, embed_channels, kernel_size=4, stride=4)
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(embed_channels, out_channels, kernel_size=8, stride=8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embeddings(x))


class TinyEncoderWithUnusedHead(TinyEncoder):
    def __init__(self, embed_channels: int = 4, out_channels: int = 8) -> None:
        super().__init__(embed_channels=embed_channels, out_channels=out_channels)
        self.layernorm = torch.nn.LayerNorm(out_channels)


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


def test_random_patch_mask_uses_foreground_medium_background_quotas() -> None:
    grid_h, grid_w = 1, 20
    weights = torch.full((1, grid_h, grid_w), 0.02)
    weights[0, 0, :8] = 0.04
    weights[0, 0, 14:] = 0.0

    torch.manual_seed(0)
    mask = random_patch_mask(
        1,
        grid_h,
        grid_w,
        0.5,
        torch.device("cpu"),
        patch_weights=weights,
        foreground_mask_ratio=0.6,
        medium_mask_ratio=0.25,
        background_mask_ratio=0.15,
    )

    assert mask.sum().item() == 10
    selected = mask[0, 0]
    assert selected[:8].sum().item() == 6
    assert selected[8:14].sum().item() == 2
    assert selected[14:].sum().item() == 2


def test_random_patch_mask_quota_backfills_and_respects_valid_patch_mask() -> None:
    grid_h, grid_w = 1, 10
    weights = torch.zeros(1, grid_h, grid_w)
    weights[0, 0, 0] = FOREGROUND_HEAVY_INK_DENSITY_THRESHOLD
    valid = torch.ones(1, grid_h, grid_w, dtype=torch.bool)
    valid[0, 0, -1] = False

    torch.manual_seed(0)
    mask = random_patch_mask(
        1,
        grid_h,
        grid_w,
        0.6,
        torch.device("cpu"),
        valid_patch_mask=valid,
        patch_weights=weights,
        foreground_mask_ratio=0.6,
        medium_mask_ratio=0.25,
        background_mask_ratio=0.15,
    )

    assert mask.sum().item() == 5
    assert (mask & ~valid).sum().item() == 0
    assert mask[0, 0, 0]


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
    ink_density = compute_patch_ink_density_batched(pixel_values, patch_size=32)
    assert pixel_values.shape == (2, 3, 160, 128)
    assert torch.all(pixel_values[:, :, 130:, :] == 1.0)
    assert torch.all(pixel_values[:, :, :, 98:] == 1.0)
    assert valid_patch_mask.shape == (2, 5, 4)
    assert not valid_patch_mask[:, -1, :].any()
    assert not valid_patch_mask[:, :, -1].any()
    assert ink_density.shape == (2, 5, 4)
    assert ((ink_density >= 0) & (ink_density <= 1)).all()
    per_sample = torch.stack(
        [compute_patch_ink_density(item["pixel_values"], patch_size=32) for item in batch]
    )
    assert torch.allclose(ink_density, per_sample)

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
    assert output.masked_background_loss is not None
    assert output.masked_foreground_loss is not None
    assert output.masked_background_patch_ratio is not None
    assert output.masked_foreground_patch_ratio is not None
    assert output.loss_weight_mean_masked is not None
    assert torch.isfinite(output.loss_weight_mean_masked)
    assert model.mask_token.shape == (1, 4, 1, 1)
    assert model.pred.weight.stride() == torch.empty_like(
        model.pred.weight,
        memory_format=torch.channels_last,
    ).stride()

    module = FCMAEPretrainer(
        model_config,
        FCMAETrainingConfig(batch_size=2, num_workers=0, max_steps=1, warmup_steps=0),
        model=model,
    )
    module._should_snapshot_preview = lambda: True
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
    assert model.pred.weight.grad is not None
    assert model.pred.weight.grad.stride() == model.pred.weight.stride()
    grads = [param.grad for param in module.parameters() if param.grad is not None]
    assert grads
    assert any(torch.isfinite(grad).all() for grad in grads)
    assert module._latest_preview is not None
    assert "pred_patches" in module._latest_preview
    assert module._latest_preview["norm_pix_loss"] is True


def test_fcmae_freezes_unused_encoder_head_for_ddp() -> None:
    model_config = FCMAEModelConfig(
        patch_size=32,
        mask_ratio=0.6,
        decoder_dim=16,
        decoder_depth=1,
    )
    encoder = TinyEncoderWithUnusedHead(out_channels=8)
    encoder.eval()
    model = DenseMaskedImageModelingConvNeXtV2(
        model_config,
        encoder=encoder,
        encoder_output_dim=8,
        encoder_stride=32,
    )

    assert model.encoder.training
    assert not any(param.requires_grad for param in model.encoder.layernorm.parameters())

    pixel_values = torch.randn(2, 3, 64, 96)
    output = model(pixel_values)
    output.loss.backward()

    missing_grads = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert missing_grads == []


def test_fcmae_masked_ink_bin_diagnostics_use_masked_valid_patches_only() -> None:
    model_config = FCMAEModelConfig(
        patch_size=32,
        mask_ratio=0.5,
        decoder_dim=16,
        decoder_depth=1,
    )
    model = DenseMaskedImageModelingConvNeXtV2(
        model_config,
        encoder=TinyEncoder(out_channels=8),
        encoder_output_dim=8,
        encoder_stride=32,
    )
    per_patch_loss = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
    ink_density = torch.tensor(
        [
            [
                [
                    BACKGROUND_HEAVY_INK_DENSITY_THRESHOLD / 2.0,
                    FOREGROUND_HEAVY_INK_DENSITY_THRESHOLD,
                    FOREGROUND_HEAVY_INK_DENSITY_THRESHOLD,
                    (BACKGROUND_HEAVY_INK_DENSITY_THRESHOLD + FOREGROUND_HEAVY_INK_DENSITY_THRESHOLD)
                    / 2.0,
                ]
            ]
        ]
    )

    background_loss, foreground_loss, background_ratio, foreground_ratio = (
        model._masked_ink_bin_diagnostics(per_patch_loss, loss_mask, ink_density)
    )

    assert background_loss is not None
    assert foreground_loss is not None
    assert background_ratio is not None
    assert foreground_ratio is not None
    assert torch.allclose(background_loss, torch.tensor(1.0))
    assert torch.allclose(foreground_loss, torch.tensor(3.0))
    assert torch.allclose(background_ratio, torch.tensor(1.0 / 3.0))
    assert torch.allclose(foreground_ratio, torch.tensor(1.0 / 3.0))


def test_fcmae_reconstruction_loss_uses_ink_weighted_denominator() -> None:
    model_config = FCMAEModelConfig(
        patch_size=1,
        mask_ratio=0.5,
        decoder_dim=16,
        decoder_depth=1,
        norm_pix_loss=False,
        ink_loss_weight_alpha=3.0,
        ink_loss_weight_target_density=0.05,
    )
    model = DenseMaskedImageModelingConvNeXtV2(
        model_config,
        encoder=TinyEncoder(out_channels=8),
        encoder_output_dim=8,
        encoder_stride=1,
    )
    imgs = torch.zeros(1, 3, 1, 3)
    pred = torch.tensor([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]])
    mask = torch.ones(1, 1, 3, dtype=torch.bool)
    ink_density = torch.tensor([[[0.0, 0.025, 0.05]]])

    output = model._reconstruction_loss(
        imgs,
        pred,
        mask,
        valid_patch_mask=None,
        ink_density=ink_density,
    )

    per_patch = torch.tensor([1.0, 4.0, 9.0])
    weights = torch.tensor([1.0, 2.5, 4.0])
    expected = (per_patch * weights).sum() / weights.sum()
    assert torch.allclose(output.loss, expected)
    assert output.loss_weight_mean_masked is not None
    assert torch.allclose(output.loss_weight_mean_masked, weights.mean())


def test_fcmae_masked_ink_bin_diagnostics_report_nan_for_empty_bins() -> None:
    model_config = FCMAEModelConfig(
        patch_size=32,
        mask_ratio=0.5,
        decoder_dim=16,
        decoder_depth=1,
    )
    model = DenseMaskedImageModelingConvNeXtV2(
        model_config,
        encoder=TinyEncoder(out_channels=8),
        encoder_output_dim=8,
        encoder_stride=32,
    )
    per_patch_loss = torch.tensor([[1.0, 3.0]])
    loss_mask = torch.tensor([[1.0, 1.0]])
    ink_density = torch.full((1, 1, 2), FOREGROUND_HEAVY_INK_DENSITY_THRESHOLD)

    background_loss, foreground_loss, background_ratio, foreground_ratio = (
        model._masked_ink_bin_diagnostics(per_patch_loss, loss_mask, ink_density)
    )

    assert background_loss is not None
    assert foreground_loss is not None
    assert background_ratio is not None
    assert foreground_ratio is not None
    assert torch.isnan(background_loss)
    assert torch.allclose(background_ratio, torch.tensor(0.0))
    assert torch.allclose(foreground_loss, torch.tensor(2.0))
    assert torch.allclose(foreground_ratio, torch.tensor(1.0))


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


def test_reconstruction_logger_skips_non_global_zero(monkeypatch) -> None:
    class FakeExperiment:
        def __init__(self) -> None:
            self.logs = []

        def log(self, payload: dict[str, object], *, step: int) -> None:
            self.logs.append((payload, step))

    class FakeWandbLogger:
        def __init__(self) -> None:
            self.experiment = FakeExperiment()

    monkeypatch.setattr("src.pretraining.fcmae.logging.WandbLogger", FakeWandbLogger)

    logger = FakeWandbLogger()
    trainer = SimpleNamespace(logger=logger, global_step=500, is_global_zero=False)
    pl_module = SimpleNamespace(_latest_preview={"payload": object()})
    callback = FCMAEReconstructionLogger(every_n_steps=500, max_batches=2, max_images=1)

    callback.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

    assert logger.experiment.logs == []
    assert callback._logged_batches == 0


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
