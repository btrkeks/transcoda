from __future__ import annotations

import pytest

from scripts.pretrain_fcmae import _apply_overrides, _build_checkpoint_callback, _derive_wandb_group
from src.pretraining.fcmae.config import FCMAEConfig


def _base_config() -> dict:
    return {
        "data": {
            "image_dir": "data/fcmae_images",
            "manifest_path": None,
            "image_height": 768,
            "image_width": 544,
        }
    }


def test_fcmae_logging_defaults_keep_wandb_off() -> None:
    config = FCMAEConfig.model_validate(_base_config())
    assert config.logging.wandb_enabled is False
    assert config.logging.project == "SMT-FCMAE"
    assert config.logging.log_model is False
    assert config.logging.log_reconstructions is True


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
