from types import SimpleNamespace

import pytest
import torch

import src.training.lightning_module as lightning_module
from src.grammar.constraint_factory import ConstraintBundle
from src.model import VisionFrontendOutput
from src.model.generation_policy import GenerationSettings
from src.training.lightning_module import SMTTrainer


class _ForwardModelStub:
    def __init__(self, loss_value: float = 0.5):
        self.calls = []
        self.loss_value = loss_value

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(loss=torch.tensor(self.loss_value))


class _GenerateModelStub:
    def __init__(self):
        self.calls = []
        self.encoder_calls = []
        self.config = SimpleNamespace(pad_token_id=0, bos_token_id=1, eos_token_id=2)

    def forward_encoder(self, pixel_values, image_sizes=None):
        self.encoder_calls.append({"pixel_values": pixel_values, "image_sizes": image_sizes})
        batch = pixel_values.shape[0]
        return VisionFrontendOutput(
            encoder_tokens_raw=torch.randn(batch, 4, 8),
            encoder_tokens_pos=torch.randn(batch, 4, 8),
            encoder_attention_mask=torch.ones(batch, 4, dtype=torch.bool),
        )

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([[1, 2, 3]])


class _CompileInitStub:
    def __init__(self, compile_model: bool, compile_mode: str, model=None):
        self.hparams = SimpleNamespace(
            training=SimpleNamespace(compile_model=compile_model, compile_mode=compile_mode)
        )
        self.model = object() if model is None else model
        self._compiled_forward_model = None
        self._compile_initialized = False
        self.messages = []

    def print(self, msg):
        self.messages.append(msg)


def test_on_fit_start_compiles_when_enabled(monkeypatch):
    stub = _CompileInitStub(compile_model=True, compile_mode="default")
    compiled_sentinel = object()
    compile_calls = []

    def fake_compile(model, mode):
        compile_calls.append((model, mode))
        return compiled_sentinel

    monkeypatch.setattr(torch, "compile", fake_compile)

    SMTTrainer.on_fit_start(stub)

    assert stub._compile_initialized is True
    assert stub._compiled_forward_model is compiled_sentinel
    assert compile_calls == [(stub.model, "default")]


def test_on_fit_start_accepts_no_cudagraph_mode(monkeypatch):
    stub = _CompileInitStub(compile_model=True, compile_mode="max-autotune-no-cudagraphs")
    compiled_sentinel = object()
    compile_calls = []

    def fake_compile(model, mode):
        compile_calls.append((model, mode))
        return compiled_sentinel

    monkeypatch.setattr(torch, "compile", fake_compile)

    SMTTrainer.on_fit_start(stub)

    assert stub._compiled_forward_model is compiled_sentinel
    assert compile_calls == [(stub.model, "max-autotune-no-cudagraphs")]


def test_on_fit_start_raises_when_compile_fails(monkeypatch):
    stub = _CompileInitStub(compile_model=True, compile_mode="default")

    def fake_compile(_model, _mode):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(torch, "compile", fake_compile)

    with pytest.raises(RuntimeError, match="torch.compile initialization failed"):
        SMTTrainer.on_fit_start(stub)


def test_on_fit_start_disables_encoder_gradient_checkpointing_when_available(monkeypatch):
    class _Encoder:
        def __init__(self):
            self.disabled = False

        def gradient_checkpointing_disable(self):
            self.disabled = True

    encoder = _Encoder()
    model = SimpleNamespace(frontend=SimpleNamespace(encoder=encoder))
    stub = _CompileInitStub(compile_model=True, compile_mode="default", model=model)

    monkeypatch.setattr(torch, "compile", lambda _model, mode=None: object())

    SMTTrainer.on_fit_start(stub)

    assert encoder.disabled is True


def test_disable_compiled_forward_model_allows_strict_restore_of_stripped_checkpoint():
    class _CompiledMirror(torch.nn.Module):
        def __init__(self, orig_mod: torch.nn.Module):
            super().__init__()
            self._orig_mod = orig_mod

    class _RestoreStub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(4, 4)
            self._compiled_forward_model = _CompiledMirror(self.model)
            self._compile_initialized = True
            self.messages = []

        def print(self, msg):
            self.messages.append(msg)

    stub = _RestoreStub()
    checkpoint = {
        "state_dict": {
            k: v.clone()
            for k, v in stub.state_dict().items()
            if "_compiled_forward_model._orig_mod." not in k
        }
    }

    with pytest.raises(RuntimeError, match="_compiled_forward_model\\._orig_mod\\.weight"):
        stub.load_state_dict(checkpoint["state_dict"], strict=True)

    disabled = SMTTrainer.disable_compiled_forward_model(stub)

    stub.load_state_dict(checkpoint["state_dict"], strict=True)

    assert disabled is True
    assert stub._compiled_forward_model is None
    assert stub._compile_initialized is False
    assert "_compiled_forward_model._orig_mod.weight" not in stub.state_dict()
    assert stub.messages == [
        "Disabled torch.compile forward wrapper; subsequent checkpoint restores run eagerly."
    ]


def test_training_step_uses_compiled_forward_path():
    eager_model = _ForwardModelStub()
    compiled_model = _ForwardModelStub()

    class _TrainerStepStub:
        def __init__(self):
            self.model = eager_model
            self._compiled_forward_model = compiled_model
            self._batch_ready_time = None
            self.global_step = 0
            self.stage_calculator = lambda _step: 1
            self.mark_calls = 0

        def _forward_model(self):
            return self._compiled_forward_model or self.model

        def _mark_compiled_step_begin(self):
            self.mark_calls += 1

        def log(self, *args, **kwargs):
            return None

    stub = _TrainerStepStub()
    batch = {
        "pixel_values": torch.randn(2, 3, 8, 8),
        "labels": torch.randint(0, 10, (2, 4)),
    }

    loss = SMTTrainer.training_step(stub, batch, _batch_idx=0)

    assert loss.item() == pytest.approx(0.5)
    assert len(compiled_model.calls) == 1
    assert len(eager_model.calls) == 0
    assert stub.mark_calls == 1


def test_generate_path_stays_on_eager_model(monkeypatch):
    eager_model = _GenerateModelStub()

    class _NeverUsedCompiled:
        def generate(self, **kwargs):
            raise AssertionError("compiled generate path should not be used")

    class _TrainerGenerateStub:
        def __init__(self):
            self.model = eager_model
            self._compiled_forward_model = _NeverUsedCompiled()
            self._generation_max_length = None
            self._constraint_factory = SimpleNamespace(
                build=lambda settings: ConstraintBundle(
                    logits_processors=None,
                    stopping_criteria=None,
                    generation_settings=settings,
                    semantic_rule_factories=(),
                )
            )
            self._generation_settings = SimpleNamespace(num_beams=1, num_return_sequences=1)

    monkeypatch.setattr(
        lightning_module,
        "build_generate_kwargs",
        lambda **kwargs: {"foo": "bar", "max_length": kwargs["max_length"]},
    )

    stub = _TrainerGenerateStub()
    preds = SMTTrainer._generate_with_grammar(
        stub,
        pixel_values=torch.randn(1, 3, 8, 8),
        image_sizes=None,
        max_length=32,
    )

    assert preds.shape == (1, 3)
    assert len(eager_model.calls) == 1
    assert len(eager_model.encoder_calls) == 1
    assert eager_model.calls[0]["max_length"] == 32
    assert eager_model.calls[0]["input_ids"].tolist() == [[1]]
    assert "encoder_outputs" in eager_model.calls[0]
    assert "encoder_attention_mask" not in eager_model.calls[0]
    assert "pixel_values" not in eager_model.calls[0]


def test_generate_path_expands_encoder_state_for_beam(monkeypatch):
    eager_model = _GenerateModelStub()

    class _TrainerGenerateStub:
        def __init__(self):
            self.model = eager_model
            self._compiled_forward_model = None
            self._generation_max_length = None
            self._constraint_factory = SimpleNamespace(
                build=lambda settings: ConstraintBundle(
                    logits_processors=None,
                    stopping_criteria=None,
                    generation_settings=settings,
                    semantic_rule_factories=(),
                )
            )
            self._generation_settings = SimpleNamespace(num_beams=4, num_return_sequences=1)

    monkeypatch.setattr(
        lightning_module,
        "build_generate_kwargs",
        lambda **kwargs: {"foo": "bar", "max_length": kwargs["max_length"]},
    )

    stub = _TrainerGenerateStub()
    pixel_values = torch.randn(2, 3, 8, 8)
    _ = SMTTrainer._generate_with_grammar(
        stub,
        pixel_values=pixel_values,
        image_sizes=None,
        max_length=32,
    )

    assert len(eager_model.calls) == 1
    call = eager_model.calls[0]
    assert call["input_ids"].shape == (2, 1)
    assert call["input_ids"].tolist() == [[1], [1]]
    assert call["encoder_outputs"].encoder_tokens_raw.shape[0] == 8
    assert call["encoder_outputs"].encoder_tokens_pos.shape[0] == 8
    assert call["encoder_outputs"].encoder_attention_mask.shape[0] == 8
    assert "encoder_attention_mask" not in call


def test_generate_path_forces_greedy_when_constraints_enabled(monkeypatch):
    eager_model = _GenerateModelStub()

    class _TrainerGenerateStub:
        def __init__(self):
            self.model = eager_model
            self._compiled_forward_model = None
            self._generation_max_length = None
            self._constraint_factory = SimpleNamespace(
                build=lambda _settings: ConstraintBundle(
                    logits_processors=[object(), object()],
                    stopping_criteria=None,
                    generation_settings=GenerationSettings(
                        strategy="greedy",
                        num_beams=1,
                        length_penalty=1.0,
                        repetition_penalty=1.1,
                        early_stopping=True,
                        num_return_sequences=1,
                        use_cache=True,
                        do_sample=False,
                    ),
                    semantic_rule_factories=(),
                )
            )
            self._generation_settings = GenerationSettings(
                strategy="beam",
                num_beams=4,
                length_penalty=1.0,
                repetition_penalty=1.1,
                early_stopping=True,
                num_return_sequences=1,
                use_cache=True,
                do_sample=False,
            )

    monkeypatch.setattr(
        lightning_module,
        "build_generate_kwargs",
        lambda **kwargs: {"max_length": kwargs["max_length"], "num_beams": kwargs["settings"].num_beams},
    )

    stub = _TrainerGenerateStub()
    pixel_values = torch.randn(2, 3, 8, 8)
    _ = SMTTrainer._generate_with_grammar(
        stub,
        pixel_values=pixel_values,
        image_sizes=None,
        max_length=32,
    )

    assert len(eager_model.calls) == 1
    call = eager_model.calls[0]
    assert call["num_beams"] == 1
    assert call["encoder_outputs"].encoder_tokens_raw.shape[0] == 2
