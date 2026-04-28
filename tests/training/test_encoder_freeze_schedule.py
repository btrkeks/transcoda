from types import SimpleNamespace

import torch

from src.training.lightning_module import SMTTrainer


class _FreezeScheduleStub:
    def __init__(self, freeze_encoder_steps: int):
        self.encoder_param = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.permanently_frozen_param = torch.nn.Parameter(
            torch.ones(1), requires_grad=False
        )
        self._scheduled_encoder_trainable_params = [self.encoder_param]
        self._encoder_freeze_schedule_active = False
        self.hparams = SimpleNamespace(
            training=SimpleNamespace(
                freeze_encoder_steps=freeze_encoder_steps,
                compile_model=False,
                compile_mode="default",
            )
        )
        self.global_step = 0
        self._compile_initialized = False
        self._compiled_forward_model = None
        self.messages = []
        self.logged = []

    def print(self, msg):
        self.messages.append(msg)

    def log(self, *args, **kwargs):
        self.logged.append((args, kwargs))


def test_encoder_freeze_steps_defaults_to_noop():
    stub = _FreezeScheduleStub(freeze_encoder_steps=0)

    SMTTrainer.on_fit_start(stub)

    assert stub.encoder_param.requires_grad is True
    assert stub.permanently_frozen_param.requires_grad is False
    assert stub._encoder_freeze_schedule_active is False


def test_encoder_freeze_steps_freezes_at_fit_start_and_logs_batch_state():
    stub = _FreezeScheduleStub(freeze_encoder_steps=2)

    SMTTrainer.on_fit_start(stub)
    SMTTrainer.on_train_batch_start(stub, batch={}, batch_idx=0)

    assert stub.encoder_param.requires_grad is False
    assert stub.permanently_frozen_param.requires_grad is False
    assert stub._encoder_freeze_schedule_active is True
    assert stub.logged[-1][0][:2] == ("train/encoder_frozen", 1.0)


def test_encoder_freeze_steps_unfreezes_at_threshold_only_for_original_trainable_params():
    stub = _FreezeScheduleStub(freeze_encoder_steps=2)

    SMTTrainer.on_fit_start(stub)
    stub.global_step = 2
    SMTTrainer.on_train_batch_start(stub, batch={}, batch_idx=0)

    assert stub.encoder_param.requires_grad is True
    assert stub.permanently_frozen_param.requires_grad is False
    assert stub._encoder_freeze_schedule_active is False
    assert stub.logged[-1][0][:2] == ("train/encoder_frozen", 0.0)


def test_encoder_freeze_steps_does_not_freeze_when_resuming_past_threshold():
    stub = _FreezeScheduleStub(freeze_encoder_steps=2)
    stub.global_step = 3

    SMTTrainer.on_fit_start(stub)

    assert stub.encoder_param.requires_grad is True
    assert stub.permanently_frozen_param.requires_grad is False
    assert stub._encoder_freeze_schedule_active is False
