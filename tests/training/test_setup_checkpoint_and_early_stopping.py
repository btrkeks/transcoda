from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from src.config import experiment_config_from_dict
from src.training.setup import setup_callbacks


def _build_config(early_stopping_enabled: bool = True):
    return experiment_config_from_dict(
        {
            "data": {
                "train_path": "./data/datasets/train_full",
                "validation_paths": {"synth": "./data/datasets/validation/synth"},
                "vocab_name": "bpe3k-splitspaces",
            },
            "checkpoint": {
                "monitor": "val/polish/CER",
                "mode": "min",
                "save_last": True,
                "filename": "smt-model",
            },
            "training": {
                "early_stopping_enabled": early_stopping_enabled,
                "early_stopping_patience": 7,
                "early_stopping_min_delta": 0.02,
            },
        }
    )


def test_setup_callbacks_uses_checkpoint_monitor_for_early_stopping():
    config = _build_config(early_stopping_enabled=True)
    callbacks, _ = setup_callbacks(
        config=config,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        i2w={0: "<pad>", 1: "<bos>", 2: "<eos>"},
    )

    early_stopping = next(cb for cb in callbacks if isinstance(cb, EarlyStopping))
    assert early_stopping.monitor == "val/polish/CER"
    assert early_stopping.mode == "min"
    assert early_stopping.patience == 7
    assert abs(early_stopping.min_delta) == 0.02


def test_setup_callbacks_enables_save_last_on_primary_checkpointer():
    config = _build_config(early_stopping_enabled=True)
    callbacks, _ = setup_callbacks(
        config=config,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        i2w={0: "<pad>", 1: "<bos>", 2: "<eos>"},
    )

    primary = next(
        cb
        for cb in callbacks
        if isinstance(cb, ModelCheckpoint) and cb.filename == config.checkpoint.filename
    )
    assert primary.monitor == "val/polish/CER"
    assert primary.mode == "min"
    assert primary.save_last is True


def test_setup_callbacks_can_disable_early_stopping():
    config = _build_config(early_stopping_enabled=False)
    callbacks, _ = setup_callbacks(
        config=config,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        i2w={0: "<pad>", 1: "<bos>", 2: "<eos>"},
    )

    assert all(not isinstance(cb, EarlyStopping) for cb in callbacks)
