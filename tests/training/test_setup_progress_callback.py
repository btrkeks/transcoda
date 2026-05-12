from src.callbacks.log_progress import LogProgressCallback
from src.config import experiment_config_from_dict
from src.training.setup import setup_callbacks


def test_setup_callbacks_wires_log_progress_from_training_config():
    config = experiment_config_from_dict(
        {
            "data": {
                "train_path": "./data/datasets/train_full",
                "validation_paths": {"synth": "./data/datasets/validation/synth"},
                "vocab_name": "bpe3k-splitspaces",
            },
            "checkpoint": {},
            "training": {
                "progress_train_interval_seconds": 12.5,
                "progress_train_every_n_steps": 25,
                "progress_val_percent_interval": 20,
                "progress_enable_ascii_bar": True,
            },
        }
    )

    callbacks, _ = setup_callbacks(
        config=config,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        i2w={0: "<pad>", 1: "<bos>", 2: "<eos>"},
    )

    progress_callback = next(cb for cb in callbacks if isinstance(cb, LogProgressCallback))
    assert progress_callback.train_interval_seconds == 12.5
    assert progress_callback.train_every_n_steps == 25
    assert progress_callback.val_percent_interval == 20
    assert progress_callback.enable_ascii_bar is True
