from types import SimpleNamespace

import torch

from src.config import Data, Training
from src.data.datamodules import PregeneratedSyntheticGrandStaffDM


def _fake_dataset():
    return [{"pixel_values": torch.zeros(3, 8, 8), "labels": [1, 2, 3]}]


def _fake_dataset_with_size(size: int):
    return [
        {"pixel_values": torch.zeros(3, 8, 8), "labels": [1, 2, 3], "sample_id": idx}
        for idx in range(size)
    ]


def _patch_dataset_loading(monkeypatch):
    monkeypatch.setattr(
        "src.data.datamodules.load_dataset_direct",
        lambda dataset_path, tokenizer: _fake_dataset(),
    )


def _patch_dataset_loading_with_calls(monkeypatch):
    calls = []

    def fake_loader(dataset_path, tokenizer):
        calls.append(dataset_path)
        return _fake_dataset()

    monkeypatch.setattr("src.data.datamodules.load_dataset_direct", fake_loader)
    return calls


def _capture_dataloader_calls(monkeypatch):
    calls = []

    def fake_dataloader(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return kwargs

    monkeypatch.setattr("src.data.datamodules.torch.utils.data.DataLoader", fake_dataloader)
    return calls


def _build_dm(training_cfg: Training):
    data_cfg = Data(train_path="train", validation_paths={"synth": "val_synth", "polish": "val_polish"})
    tokenizer = SimpleNamespace(pad_token_id=0)
    return PregeneratedSyntheticGrandStaffDM(
        tokenizer=tokenizer,
        data_config=data_cfg,
        training_config=training_cfg,
        max_decoder_len=2000,
    )


def test_training_config_loader_fields_are_backward_compatible():
    cfg = Training(batch_size=4, num_workers=6)
    assert cfg.train_prefetch_factor == 8
    assert cfg.train_persistent_workers is True
    assert cfg.train_pin_memory is True
    assert cfg.val_num_workers is None
    assert cfg.val_prefetch_factor == 2
    assert cfg.val_persistent_workers is False
    assert cfg.val_pin_memory is False


def test_datamodule_uses_split_train_val_loader_policies(monkeypatch):
    _patch_dataset_loading(monkeypatch)
    calls = _capture_dataloader_calls(monkeypatch)

    training_cfg = Training(batch_size=6, val_batch_size=30, num_workers=12)
    dm = _build_dm(training_cfg)
    dm.setup(stage="fit")
    dm.train_dataloader()
    dm.val_dataloader()

    assert len(calls) == 3  # 1 train + 2 val loaders

    train_kwargs = calls[0]["kwargs"]
    assert train_kwargs["batch_size"] == 6
    assert train_kwargs["num_workers"] == 12
    assert train_kwargs["pin_memory"] is True
    assert train_kwargs["persistent_workers"] is True
    assert train_kwargs["prefetch_factor"] == 8
    assert train_kwargs["worker_init_fn"] is not None

    val_kwargs = calls[1]["kwargs"]
    assert val_kwargs["batch_size"] == 30
    assert val_kwargs["num_workers"] == 2  # derived from min(2, num_workers)
    assert val_kwargs["pin_memory"] is False
    assert val_kwargs["persistent_workers"] is False
    assert val_kwargs["prefetch_factor"] == 2
    assert val_kwargs["worker_init_fn"] is not None


def test_datamodule_omits_prefetch_and_persistent_with_zero_workers(monkeypatch):
    _patch_dataset_loading(monkeypatch)
    calls = _capture_dataloader_calls(monkeypatch)

    training_cfg = Training(batch_size=4, val_batch_size=8, num_workers=0)
    dm = _build_dm(training_cfg)
    dm.setup(stage="fit")
    dm.train_dataloader()
    dm.val_dataloader()

    train_kwargs = calls[0]["kwargs"]
    assert train_kwargs["num_workers"] == 0
    assert "prefetch_factor" not in train_kwargs
    assert "persistent_workers" not in train_kwargs

    val_kwargs = calls[1]["kwargs"]
    assert val_kwargs["num_workers"] == 0
    assert "prefetch_factor" not in val_kwargs
    assert "persistent_workers" not in val_kwargs


def test_datamodule_respects_explicit_validation_loader_overrides(monkeypatch):
    _patch_dataset_loading(monkeypatch)
    calls = _capture_dataloader_calls(monkeypatch)

    training_cfg = Training(
        batch_size=2,
        val_batch_size=5,
        num_workers=8,
        val_num_workers=4,
        val_prefetch_factor=1,
        val_persistent_workers=True,
        val_pin_memory=True,
    )
    dm = _build_dm(training_cfg)
    dm.setup(stage="fit")
    dm.val_dataloader()

    val_kwargs = calls[0]["kwargs"]
    assert val_kwargs["num_workers"] == 4
    assert val_kwargs["prefetch_factor"] == 1
    assert val_kwargs["persistent_workers"] is True
    assert val_kwargs["pin_memory"] is True


def test_datamodule_uses_fixed_size_collator_from_data_config():
    data_cfg = Data(
        train_path="train",
        validation_paths={"synth": "val_synth"},
        fixed_image_height=1485,
        fixed_image_width=1050,
    )
    tokenizer = SimpleNamespace(pad_token_id=0)
    dm = PregeneratedSyntheticGrandStaffDM(
        tokenizer=tokenizer,
        data_config=data_cfg,
        training_config=Training(),
        max_decoder_len=2000,
    )

    assert dm.collator.enforce_fixed_size is True
    assert dm.collator.fixed_height == 1485
    assert dm.collator.fixed_width == 1050
    assert dm.collator.fixed_decoder_length == 2000


def test_datamodule_validate_stage_skips_train_loading(monkeypatch):
    calls = _patch_dataset_loading_with_calls(monkeypatch)
    dm = _build_dm(Training())

    dm.setup(stage="validate")

    assert dm.train_set is None
    assert sorted(dm.val_sets.keys()) == ["polish", "synth"]
    assert calls == ["val_synth", "val_polish"]


def test_datamodule_setup_is_idempotent_across_stage_reentry(monkeypatch):
    calls = _patch_dataset_loading_with_calls(monkeypatch)
    dm = _build_dm(Training())

    dm.setup(stage="fit")
    dm.setup(stage=SimpleNamespace(value="fit"))  # Lightning enum-like stage
    dm.setup(stage="validate")

    # Train + 2 val datasets should be loaded once total.
    assert calls == ["train", "val_synth", "val_polish"]


def test_datamodule_tiered_validation_runs_only_frequent_sets_between_full_passes(monkeypatch):
    _patch_dataset_loading(monkeypatch)
    calls = _capture_dataloader_calls(monkeypatch)

    training_cfg = Training(
        tiered_validation_enabled=True,
        full_validation_every_n_steps=5000,
        frequent_validation_set_names=["polish"],
    )
    dm = _build_dm(training_cfg)
    dm.setup(stage="fit")
    dm.trainer = SimpleNamespace(global_step=1000)
    dm.val_dataloader()

    assert len(calls) == 2
    synth_dataset = calls[0]["args"][0]
    polish_dataset = calls[1]["args"][0]
    assert len(synth_dataset) == 0
    assert len(polish_dataset) == 1


def test_datamodule_validation_subset_indices_are_deterministic(monkeypatch):
    monkeypatch.setattr(
        "src.data.datamodules.load_dataset_direct",
        lambda dataset_path, tokenizer: _fake_dataset_with_size(8),
    )

    training_cfg = Training(
        frequent_validation_subset_sizes={"synth": 3},
        frequent_validation_subset_seed=11,
    )
    dm = _build_dm(training_cfg)
    dm.setup(stage="fit")
    first = list(dm._val_subset_indices_by_name["synth"])

    dm.setup(stage="validate")
    second = list(dm._val_subset_indices_by_name["synth"])

    assert first == second
    assert len(first) == 3
    assert first == sorted(first)


def test_datamodule_tiered_validation_keeps_subset_loader_active_between_full_passes(monkeypatch):
    _patch_dataset_loading(monkeypatch)
    calls = _capture_dataloader_calls(monkeypatch)

    training_cfg = Training(
        tiered_validation_enabled=True,
        full_validation_every_n_steps=5000,
        frequent_validation_set_names=["polish"],
        frequent_validation_subset_sizes={"synth": 1},
    )
    dm = _build_dm(training_cfg)
    dm.setup(stage="fit")
    dm.trainer = SimpleNamespace(global_step=1000)
    dm.val_dataloader()

    assert len(calls) == 3
    synth_dataset = calls[0]["args"][0]
    polish_dataset = calls[1]["args"][0]
    synth_subset_dataset = calls[2]["args"][0]
    assert len(synth_dataset) == 0
    assert len(polish_dataset) == 1
    assert len(synth_subset_dataset) == 1


def test_datamodule_tiered_validation_runs_full_pass_on_full_interval(monkeypatch):
    _patch_dataset_loading(monkeypatch)
    calls = _capture_dataloader_calls(monkeypatch)

    training_cfg = Training(
        tiered_validation_enabled=True,
        full_validation_every_n_steps=5000,
        frequent_validation_set_names=["polish"],
        frequent_validation_subset_sizes={"synth": 1},
    )
    dm = _build_dm(training_cfg)
    dm.setup(stage="fit")
    dm.trainer = SimpleNamespace(global_step=5000)
    dm.val_dataloader()

    assert len(calls) == 3
    synth_dataset = calls[0]["args"][0]
    polish_dataset = calls[1]["args"][0]
    synth_subset_dataset = calls[2]["args"][0]
    assert len(synth_dataset) == 1
    assert len(polish_dataset) == 1
    assert len(synth_subset_dataset) == 1


def test_datamodule_validate_stage_always_uses_full_validation_sets(monkeypatch):
    _patch_dataset_loading(monkeypatch)
    calls = _capture_dataloader_calls(monkeypatch)

    training_cfg = Training(
        tiered_validation_enabled=True,
        full_validation_every_n_steps=5000,
        frequent_validation_set_names=["polish"],
        frequent_validation_subset_sizes={"synth": 1},
    )
    dm = _build_dm(training_cfg)
    dm.setup(stage="validate")
    dm.trainer = SimpleNamespace(global_step=1000)
    dm.val_dataloader()

    assert len(calls) == 2
    synth_dataset = calls[0]["args"][0]
    polish_dataset = calls[1]["args"][0]
    assert len(synth_dataset) == 1
    assert len(polish_dataset) == 1
    assert all(not isinstance(call["args"][0], torch.utils.data.Subset) for call in calls)


def test_validate_stage_applies_memory_safety_loader_clamp(monkeypatch):
    _patch_dataset_loading(monkeypatch)
    calls = _capture_dataloader_calls(monkeypatch)

    training_cfg = Training(
        batch_size=12,
        val_batch_size=64,
        num_workers=12,
        val_num_workers=8,
        val_prefetch_factor=4,
        val_persistent_workers=True,
        val_pin_memory=True,
    )
    dm = _build_dm(training_cfg)
    dm.setup(stage="validate")
    monkeypatch.setattr(dm, "_available_host_memory_bytes", lambda: int(64 * (1024**3)))
    dm.val_dataloader()

    assert len(calls) == 2
    val_kwargs = calls[0]["kwargs"]
    assert val_kwargs["num_workers"] == 7
    assert val_kwargs["prefetch_factor"] == 1
    assert val_kwargs["pin_memory"] is False
    assert val_kwargs["persistent_workers"] is True


def test_fit_stage_does_not_apply_validation_memory_safety_clamp(monkeypatch):
    _patch_dataset_loading(monkeypatch)
    calls = _capture_dataloader_calls(monkeypatch)

    training_cfg = Training(
        batch_size=12,
        val_batch_size=64,
        num_workers=12,
        val_num_workers=8,
        val_prefetch_factor=4,
        val_persistent_workers=True,
        val_pin_memory=True,
    )
    dm = _build_dm(training_cfg)
    dm.setup(stage="fit")
    dm.trainer = SimpleNamespace(global_step=1000)
    monkeypatch.setattr(dm, "_available_host_memory_bytes", lambda: int(64 * (1024**3)))
    dm.val_dataloader()

    assert len(calls) == 2
    val_kwargs = calls[0]["kwargs"]
    assert val_kwargs["num_workers"] == 8
    assert val_kwargs["prefetch_factor"] == 4
    assert val_kwargs["pin_memory"] is True
    assert val_kwargs["persistent_workers"] is True
