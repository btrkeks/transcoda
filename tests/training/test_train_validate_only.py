import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import train as train_entry
from src.artifacts import SlurmSpec


def _write_min_config(tmp_path: Path) -> Path:
    config = {
        "data": {
            "train_path": "./data/datasets/train_full",
            "validation_paths": {
                "synth": "./data/datasets/validation/synth",
                "polish": "./data/datasets/validation/polish",
            },
            "vocab_dir": "./vocab/bpe4k",
        },
        "checkpoint": {
            "dirpath": str(tmp_path / "weights"),
            "filename": "unit-test-model",
            "project": "unit-tests",
            "run_name": "validate-only-test",
        },
        "training": {
            "max_epochs": 1,
            "accumulate_grad_batches": 1,
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


class _DummyTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    bos_token = "<bos>"
    eos_token = "</eos>"
    vocab_size = 5

    def get_vocab(self):
        return {"<pad>": 0, "<bos>": 1, "</eos>": 2, "a": 3, "b": 4}

    def __len__(self):
        return self.vocab_size


class _DummyPretrainedTokenizerFast:
    @staticmethod
    def from_pretrained(_vocab_dir: str):
        return _DummyTokenizer()


class _DummyDataModule:
    last_instance = None
    train_dataloader_calls = 0

    def __init__(self, **_kwargs):
        _DummyDataModule.last_instance = self
        self.train_set = [0, 1, 2, 3]
        self.val_sets = {"synth": [0, 1], "polish": [2]}

    def setup(self, stage: str | None = None):
        self.stage = stage

    @property
    def val_set_names(self) -> list[str]:
        return list(self.val_sets.keys())

    @property
    def val_loader_names(self) -> list[str]:
        return self.val_set_names

    def train_dataloader(self):
        _DummyDataModule.train_dataloader_calls += 1
        return [0] * 8


class _DummyModelWrapper:
    last_kwargs = None
    last_validation_metric_prefix = None
    validation_example_logging_overrides = []
    disable_compiled_forward_model_calls = 0

    def __init__(self, **kwargs):
        _DummyModelWrapper.last_kwargs = kwargs
        self.compiled_forward_model_disabled = False

    def parameters(self):
        return [
            torch.nn.Parameter(torch.ones(2), requires_grad=True),
            torch.nn.Parameter(torch.ones(3), requires_grad=False),
        ]

    def set_validation_metric_prefix(self, prefix):
        _DummyModelWrapper.last_validation_metric_prefix = prefix

    def set_validation_example_logging_override(self, enabled):
        _DummyModelWrapper.validation_example_logging_overrides.append(enabled)

    def disable_compiled_forward_model(self):
        _DummyModelWrapper.disable_compiled_forward_model_calls += 1
        self.compiled_forward_model_disabled = True
        return True


class _DummyTrainer:
    def __init__(self):
        self.fit_calls = []
        self.validate_calls = []
        self.validate_compiled_disabled_states = []
        self.should_stop = False

    def fit(self, *args, **kwargs):
        self.fit_calls.append((args, kwargs))

    def validate(self, *args, **kwargs):
        self.validate_calls.append((args, kwargs))
        self.validate_compiled_disabled_states.append(
            getattr(args[0], "compiled_forward_model_disabled", None)
        )


class _DummyExperimentConfig:
    def __init__(self):
        self.updates = []

    def update(self, *_args, **_kwargs):
        self.updates.append((_args, _kwargs))
        return None


class _DummyExperiment:
    def __init__(self):
        self.config = _DummyExperimentConfig()
        self.logged = []

    def log(self, payload):
        self.logged.append(payload)


class _DummyLogger:
    last_instance = None

    def __init__(self):
        _DummyLogger.last_instance = self
        self.experiment = _DummyExperiment()


def _patch_entrypoint(monkeypatch: pytest.MonkeyPatch, trainer: _DummyTrainer) -> None:
    _DummyDataModule.last_instance = None
    _DummyDataModule.train_dataloader_calls = 0
    _DummyLogger.last_instance = None
    _DummyModelWrapper.last_kwargs = None
    _DummyModelWrapper.last_validation_metric_prefix = None
    _DummyModelWrapper.validation_example_logging_overrides = []
    _DummyModelWrapper.disable_compiled_forward_model_calls = 0
    monkeypatch.setattr(train_entry, "PreTrainedTokenizerFast", _DummyPretrainedTokenizerFast)
    monkeypatch.setattr(train_entry, "PregeneratedSyntheticGrandStaffDM", _DummyDataModule)
    monkeypatch.setattr(train_entry, "SMTTrainer", _DummyModelWrapper)
    monkeypatch.setattr(
        train_entry,
        "setup_callbacks",
        lambda **_kwargs: (
            [],
            SimpleNamespace(best_model_path="", last_model_path=""),
        ),
    )
    monkeypatch.setattr(train_entry, "setup_logger", lambda *_args, **_kwargs: _DummyLogger())
    monkeypatch.setattr(train_entry, "setup_trainer", lambda *_args, **_kwargs: trainer)
    monkeypatch.setattr(train_entry, "seed_everything", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        train_entry, "_verify_checkpoint_tokenizer_compatibility", lambda **_kwargs: None
    )


def test_main_validate_only_runs_validate_not_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = _write_min_config(tmp_path)
    trainer = _DummyTrainer()
    _patch_entrypoint(monkeypatch, trainer)
    checkpoint_path = str(tmp_path / "best.ckpt")

    train_entry.main(
        config_path=str(config_path),
        checkpoint_path=checkpoint_path,
        validate_only=True,
    )

    assert len(trainer.validate_calls) == 1
    assert len(trainer.fit_calls) == 0
    _, kwargs = trainer.validate_calls[0]
    assert kwargs["ckpt_path"] == checkpoint_path
    assert _DummyDataModule.last_instance is not None
    assert _DummyDataModule.last_instance.stage == "validate"
    assert _DummyDataModule.train_dataloader_calls == 0


def test_main_default_path_runs_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = _write_min_config(tmp_path)
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = weights_dir / "last.ckpt"
    last_ckpt.write_bytes(b"placeholder")
    trainer = _DummyTrainer()
    _patch_entrypoint(monkeypatch, trainer)

    train_entry.main(config_path=str(config_path))

    assert len(trainer.fit_calls) == 1
    assert len(trainer.validate_calls) == 1
    _, fit_kwargs = trainer.fit_calls[0]
    assert fit_kwargs["ckpt_path"] == str(last_ckpt)
    _, validate_kwargs = trainer.validate_calls[0]
    assert validate_kwargs["ckpt_path"] == str(last_ckpt)
    assert _DummyDataModule.last_instance is not None
    assert _DummyDataModule.last_instance.stage == "validate"
    assert _DummyDataModule.train_dataloader_calls == 2


def test_validate_only_requires_checkpoint(tmp_path: Path):
    config_path = _write_min_config(tmp_path)

    with pytest.raises(
        ValueError,
        match="checkpoint_path must be provided when validate_only=True",
    ):
        train_entry.main(config_path=str(config_path), validate_only=True)


def test_main_logs_slurm_and_launch_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = _write_min_config(tmp_path)
    trainer = _DummyTrainer()
    _patch_entrypoint(monkeypatch, trainer)
    checkpoint_path = str(tmp_path / "best.ckpt")

    monkeypatch.setattr(
        train_entry,
        "collect_slurm",
        lambda: SlurmSpec(
            job_id="30",
            job_name="smt-validate",
            partition="gpu",
            nodelist="sarmatia",
            cpus_per_task=8,
            gpus_on_node="1",
            submit_host="antemurale",
            cluster_name="ude",
        ),
    )
    monkeypatch.setattr(train_entry, "_collect_git_sha", lambda: "abc123")
    monkeypatch.setattr(train_entry, "_collect_launch_command", lambda: "python train.py ...")

    train_entry.main(
        config_path=str(config_path),
        checkpoint_path=checkpoint_path,
        validate_only=True,
    )

    assert _DummyLogger.last_instance is not None
    updates = _DummyLogger.last_instance.experiment.config.updates
    assert updates, "Expected at least one config.update call"
    payload = updates[-1][0][0]

    assert payload["slurm/job_id"] == "30"
    assert payload["slurm/job_name"] == "smt-validate"
    assert payload["slurm/partition"] == "gpu"
    assert payload["slurm/cpus_per_task"] == 8
    assert payload["git/sha"] == "abc123"
    assert payload["launch/command"] == "python train.py ..."


def test_validate_only_forces_log_example_images_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = _write_min_config(tmp_path)
    cfg = json.loads(config_path.read_text())
    cfg.setdefault("training", {})
    cfg["training"]["log_example_images"] = False
    config_path.write_text(json.dumps(cfg))

    trainer = _DummyTrainer()
    _patch_entrypoint(monkeypatch, trainer)
    checkpoint_path = str(tmp_path / "best.ckpt")

    train_entry.main(
        config_path=str(config_path),
        checkpoint_path=checkpoint_path,
        validate_only=True,
    )

    assert _DummyModelWrapper.last_kwargs is not None
    training_cfg = _DummyModelWrapper.last_kwargs["training"]
    assert training_cfg.log_example_images is True


def test_main_auto_resume_uses_last_ckpt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = _write_min_config(tmp_path)
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = weights_dir / "last.ckpt"
    last_ckpt.write_bytes(b"placeholder")

    trainer = _DummyTrainer()
    _patch_entrypoint(monkeypatch, trainer)

    train_entry.main(config_path=str(config_path))

    assert len(trainer.fit_calls) == 1
    assert len(trainer.validate_calls) == 1
    _, kwargs = trainer.fit_calls[0]
    assert kwargs["ckpt_path"] == str(last_ckpt)


def test_main_fresh_run_disables_auto_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = _write_min_config(tmp_path)
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "last.ckpt").write_bytes(b"placeholder")

    trainer = _DummyTrainer()
    _patch_entrypoint(monkeypatch, trainer)

    train_entry.main(config_path=str(config_path), fresh_run=True)

    assert len(trainer.fit_calls) == 1
    _, kwargs = trainer.fit_calls[0]
    assert kwargs["ckpt_path"] is None


def test_main_explicit_checkpoint_overrides_auto_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = _write_min_config(tmp_path)
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "last.ckpt").write_bytes(b"placeholder")
    explicit_ckpt = str(tmp_path / "explicit.ckpt")

    trainer = _DummyTrainer()
    _patch_entrypoint(monkeypatch, trainer)

    train_entry.main(config_path=str(config_path), checkpoint_path=explicit_ckpt)

    assert len(trainer.fit_calls) == 1
    assert len(trainer.validate_calls) == 1
    _, kwargs = trainer.fit_calls[0]
    assert kwargs["ckpt_path"] == explicit_ckpt


def test_main_training_runs_final_validation_from_best_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = _write_min_config(tmp_path)
    best_ckpt = tmp_path / "best.ckpt"
    best_ckpt.write_bytes(b"placeholder")
    trainer = _DummyTrainer()
    _patch_entrypoint(monkeypatch, trainer)
    monkeypatch.setattr(
        train_entry,
        "setup_callbacks",
        lambda **_kwargs: (
            [],
            SimpleNamespace(best_model_path=str(best_ckpt), last_model_path=""),
        ),
    )

    train_entry.main(config_path=str(config_path))

    assert len(trainer.validate_calls) == 1
    _, kwargs = trainer.validate_calls[0]
    assert kwargs["ckpt_path"] == str(best_ckpt)
    assert _DummyModelWrapper.disable_compiled_forward_model_calls == 1
    assert trainer.validate_compiled_disabled_states == [True]
    assert _DummyModelWrapper.last_validation_metric_prefix == train_entry.VAL_PREFIX
    assert _DummyModelWrapper.validation_example_logging_overrides == [False, None]


def test_main_training_final_validation_falls_back_to_checkpoint_last_model_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = _write_min_config(tmp_path)
    last_model_ckpt = tmp_path / "callback-last.ckpt"
    last_model_ckpt.write_bytes(b"placeholder")
    trainer = _DummyTrainer()
    _patch_entrypoint(monkeypatch, trainer)
    monkeypatch.setattr(
        train_entry,
        "setup_callbacks",
        lambda **_kwargs: (
            [],
            SimpleNamespace(best_model_path="", last_model_path=str(last_model_ckpt)),
        ),
    )

    train_entry.main(config_path=str(config_path))

    assert len(trainer.validate_calls) == 1
    _, kwargs = trainer.validate_calls[0]
    assert kwargs["ckpt_path"] == str(last_model_ckpt)


def test_main_training_final_validation_still_runs_when_fit_stops_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = _write_min_config(tmp_path)
    best_ckpt = tmp_path / "best.ckpt"
    best_ckpt.write_bytes(b"placeholder")
    trainer = _DummyTrainer()
    trainer.should_stop = True
    _patch_entrypoint(monkeypatch, trainer)
    monkeypatch.setattr(
        train_entry,
        "setup_callbacks",
        lambda **_kwargs: (
            [],
            SimpleNamespace(best_model_path=str(best_ckpt), last_model_path=""),
        ),
    )

    train_entry.main(config_path=str(config_path))

    assert len(trainer.fit_calls) == 1
    assert len(trainer.validate_calls) == 1


def test_main_training_requires_persisted_checkpoint_for_final_validation(tmp_path: Path):
    config_path = _write_min_config(tmp_path)
    cfg = json.loads(config_path.read_text())
    cfg["checkpoint"]["save_last"] = False
    cfg["checkpoint"]["save_top_k"] = 0
    config_path.write_text(json.dumps(cfg))

    with pytest.raises(
        ValueError,
        match="Automatic final validation requires at least one persisted checkpoint",
    ):
        train_entry.main(config_path=str(config_path))


def test_main_training_raises_when_final_validation_checkpoint_cannot_be_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = _write_min_config(tmp_path)
    trainer = _DummyTrainer()
    _patch_entrypoint(monkeypatch, trainer)

    with pytest.raises(RuntimeError, match="Unable to resolve a checkpoint for automatic final validation"):
        train_entry.main(config_path=str(config_path))
