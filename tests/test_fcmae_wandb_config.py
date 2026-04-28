from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.pretrain_fcmae import (
    _apply_overrides,
    _build_checkpoint_callback,
    _derive_wandb_group,
    _log_dataset_size,
    _setup_logger,
)
from src.pretraining.fcmae.config import FCMAEConfig


def _base_config() -> dict:
    return {
        "data": {
            "image_dir": "data/fcmae_images",
            "manifest_path": None,
            "image_height": 1485,
            "image_width": 1050,
        }
    }


def test_fcmae_logging_defaults_keep_wandb_off() -> None:
    config = FCMAEConfig.model_validate(_base_config())
    assert config.data.image_height == 1485
    assert config.data.image_width == 1050
    assert config.model.ink_bias_strength == 0.3
    assert config.model.use_foreground_quota_masking is False
    assert config.model.foreground_mask_ratio == 0.60
    assert config.model.medium_mask_ratio == 0.25
    assert config.model.background_mask_ratio == 0.15
    assert config.model.ink_loss_weight_alpha == 0.0
    assert config.model.ink_loss_weight_target_density == 0.05
    assert config.model.ink_aux_loss_weight == 1.0
    assert config.model.ink_aux_bce_weight == 0.5
    assert config.model.ink_aux_dice_weight == 0.5
    assert config.model.ink_aux_target_threshold == 0.05
    assert config.logging.wandb_enabled is False
    assert config.logging.project == "SMT-FCMAE"
    assert config.logging.log_model is False
    assert config.logging.log_reconstructions is True
    assert config.training.devices == 1
    assert config.training.strategy == "auto"
    assert config.training.num_nodes == 1


def test_fcmae_training_config_auto_selects_ddp_for_multi_device() -> None:
    payload = _base_config()
    payload["training"] = {"devices": 2}
    config = FCMAEConfig.model_validate(payload)
    assert config.training.devices == 2
    assert config.training.strategy == "ddp"


@pytest.mark.parametrize(
    ("training", "match"),
    [
        ({"devices": 0}, "training.devices"),
        ({"num_nodes": 0}, "training.num_nodes"),
        ({"strategy": "ddp_spawn"}, "ddp_spawn"),
    ],
)
def test_fcmae_training_config_rejects_bad_ddp_settings(
    training: dict[str, object],
    match: str,
) -> None:
    payload = _base_config()
    payload["training"] = training
    with pytest.raises(ValueError, match=match):
        FCMAEConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("model", "match"),
    [
        (
            {
                "foreground_mask_ratio": 0.5,
                "medium_mask_ratio": 0.25,
                "background_mask_ratio": 0.15,
            },
            "sum to 1.0",
        ),
        (
            {
                "foreground_mask_ratio": -0.1,
                "medium_mask_ratio": 0.5,
                "background_mask_ratio": 0.6,
            },
            "foreground_mask_ratio",
        ),
        ({"ink_loss_weight_alpha": -1.0}, "ink_loss_weight_alpha"),
        ({"ink_loss_weight_target_density": 0.0}, "ink_loss_weight_target_density"),
        ({"ink_aux_loss_weight": -1.0}, "ink_aux_loss_weight"),
        ({"ink_aux_bce_weight": -1.0}, "ink_aux_bce_weight"),
        ({"ink_aux_dice_weight": -1.0}, "ink_aux_dice_weight"),
        ({"ink_aux_target_threshold": -0.1}, "ink_aux_target_threshold"),
    ],
)
def test_fcmae_model_config_rejects_bad_foreground_objective_settings(
    model: dict[str, object],
    match: str,
) -> None:
    payload = _base_config()
    payload["model"] = model
    with pytest.raises(ValueError, match=match):
        FCMAEConfig.model_validate(payload)


def test_fcmae_config_allows_downstream_non_divisible_canvas() -> None:
    config = FCMAEConfig.model_validate(_base_config())
    assert config.data.image_height % config.model.patch_size != 0
    assert config.data.image_width % config.model.patch_size != 0


def test_fcmae_logging_validation_rejects_bad_preview_cadence() -> None:
    payload = _base_config()
    payload["logging"] = {"log_reconstruction_every_n_steps": 0}
    with pytest.raises(ValueError, match="log_reconstruction_every_n_steps"):
        FCMAEConfig.model_validate(payload)


def test_fcmae_checkpoint_defaults_save_periodic_step_checkpoints() -> None:
    config = FCMAEConfig.model_validate(_base_config())
    assert config.checkpoint.save_last is True
    assert config.checkpoint.save_top_k == -1
    assert config.checkpoint.every_n_train_steps == 5000

    callback = _build_checkpoint_callback(config)
    assert callback.save_last is True
    assert callback.save_top_k == -1
    assert callback._every_n_train_steps == 5000
    assert callback._every_n_epochs == 0
    assert callback._save_on_train_epoch_end is False


def test_fcmae_checkpoint_validation_rejects_bad_step_interval() -> None:
    payload = _base_config()
    payload["checkpoint"] = {"every_n_train_steps": 0}
    with pytest.raises(ValueError, match="checkpoint.every_n_train_steps"):
        FCMAEConfig.model_validate(payload)


def test_log_dataset_size_prints_count_and_updates_wandb_config(capsys, monkeypatch) -> None:
    class FakeWandbLogger:
        def __init__(self) -> None:
            self.experiment = SimpleNamespace(updates=[])

        def config_update(self, payload: dict[str, object], *, allow_val_change: bool) -> None:
            self.experiment.updates.append((payload, allow_val_change))

    logger = FakeWandbLogger()
    logger.experiment.config = SimpleNamespace(update=logger.config_update)
    monkeypatch.setattr("scripts.pretrain_fcmae.WandbLogger", FakeWandbLogger)

    datamodule = SimpleNamespace(
        train_set=[object(), object(), object()],
        setup=lambda stage: None,
    )

    assert _log_dataset_size(datamodule, logger) == 3
    assert "FCMAE pretraining images used: 3" in capsys.readouterr().out
    assert logger.experiment.updates == [({"data/num_images_used": 3}, True)]


def test_log_dataset_size_skips_wandb_config_update_off_rank_zero(capsys, monkeypatch) -> None:
    class FakeWandbLogger:
        def __init__(self) -> None:
            self.experiment = SimpleNamespace(updates=[])

        def config_update(self, payload: dict[str, object], *, allow_val_change: bool) -> None:
            self.experiment.updates.append((payload, allow_val_change))

    logger = FakeWandbLogger()
    logger.experiment.config = SimpleNamespace(update=logger.config_update)
    monkeypatch.setattr("scripts.pretrain_fcmae.WandbLogger", FakeWandbLogger)
    monkeypatch.setenv("LOCAL_RANK", "1")

    datamodule = SimpleNamespace(
        train_set=[object(), object(), object()],
        setup=lambda stage: None,
    )

    assert _log_dataset_size(datamodule, logger) == 3
    assert "FCMAE pretraining images used: 3" in capsys.readouterr().out
    assert logger.experiment.updates == []


def test_setup_logger_uses_world_size_effective_batch_and_skips_update_off_rank_zero(
    monkeypatch,
) -> None:
    class FakeConfig:
        def __init__(self) -> None:
            self.updates = []

        def update(self, payload: dict[str, object], *, allow_val_change: bool) -> None:
            self.updates.append((payload, allow_val_change))

    class FakeWandbLogger:
        last_instance: "FakeWandbLogger | None" = None

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.hyperparams = None
            self.experiment = SimpleNamespace(config=FakeConfig())
            FakeWandbLogger.last_instance = self

        def log_hyperparams(self, params: dict[str, object]) -> None:
            self.hyperparams = params

    monkeypatch.setattr("scripts.pretrain_fcmae.WandbLogger", FakeWandbLogger)
    monkeypatch.setenv("LOCAL_RANK", "1")

    payload = _base_config()
    payload["logging"] = {"wandb_enabled": True}
    payload["training"] = {"batch_size": 4, "accumulate_grad_batches": 8, "devices": 2}
    config = FCMAEConfig.model_validate(payload)

    logger = _setup_logger(config, payload)

    assert isinstance(logger, FakeWandbLogger)
    assert logger.experiment.config.updates == []


def test_setup_logger_logs_world_size_effective_batch_on_rank_zero(monkeypatch) -> None:
    class FakeConfig:
        def __init__(self) -> None:
            self.updates = []

        def update(self, payload: dict[str, object], *, allow_val_change: bool) -> None:
            self.updates.append((payload, allow_val_change))

    class FakeWandbLogger:
        last_instance: "FakeWandbLogger | None" = None

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.hyperparams = None
            self.experiment = SimpleNamespace(config=FakeConfig())
            FakeWandbLogger.last_instance = self

        def log_hyperparams(self, params: dict[str, object]) -> None:
            self.hyperparams = params

    monkeypatch.setattr("scripts.pretrain_fcmae.WandbLogger", FakeWandbLogger)
    monkeypatch.setattr(
        "scripts.pretrain_fcmae.collect_slurm",
        lambda: None,
    )

    payload = _base_config()
    payload["logging"] = {"wandb_enabled": True}
    payload["training"] = {"batch_size": 4, "accumulate_grad_batches": 8, "devices": 2}
    config = FCMAEConfig.model_validate(payload)

    logger = _setup_logger(config, payload)

    assert isinstance(logger, FakeWandbLogger)
    assert logger.experiment.config.updates
    update, allow_val_change = logger.experiment.config.updates[0]
    assert allow_val_change is True
    assert update["scale/effective_batch_size"] == 64
    assert update["scale/world_size"] == 2
    assert update["optimization/resolved_initial_learning_rate"] == pytest.approx(
        config.training.base_learning_rate * 64 / 256
    )


def test_fcmae_export_on_train_end_requires_output_dir() -> None:
    payload = _base_config()
    payload["export"] = {"export_on_train_end": True}
    with pytest.raises(ValueError, match="export.output_dir"):
        FCMAEConfig.model_validate(payload)


def test_pretrain_overrides_parse_bare_dotlist_values() -> None:
    payload = _apply_overrides(
        _base_config(),
        [
            "logging.wandb_enabled=true",
            "logging.tags=[\"fcmae\",\"smoke\"]",
            "training.resume_from_checkpoint=weights/fcmae/last.ckpt",
        ],
    )
    config = FCMAEConfig.model_validate(payload)
    assert config.logging.wandb_enabled is True
    assert config.logging.tags == ["fcmae", "smoke"]
    assert config.training.resume_from_checkpoint == "weights/fcmae/last.ckpt"


def test_wandb_group_derived_from_checkpoint_dir() -> None:
    assert _derive_wandb_group("weights/fcmae-real-scans") == "fcmae-real-scans"
    assert _derive_wandb_group("/") == "default"
