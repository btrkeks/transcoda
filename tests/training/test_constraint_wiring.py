from types import SimpleNamespace

import torch

import src.training.lightning_module as lightning_module
from src.config import Generation, ModelConfig, OptimizerConfig, Training
from src.training.lightning_module import SMTTrainer


class _MetricCollectionStub(torch.nn.Module):
    def __init__(self, *_args, **_kwargs):
        super().__init__()

    def clone(self, prefix: str = ""):
        return _MetricCollectionStub()


class _ModelStub(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = SimpleNamespace(
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
        )


def test_trainer_disables_rhythm_constraint_wiring(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(lightning_module, "SMTModelForCausalLM", _ModelStub)
    monkeypatch.setattr(lightning_module, "MetricCollection", _MetricCollectionStub)
    monkeypatch.setattr(lightning_module, "SymbolErrorRate", lambda *args, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(lightning_module, "CharacterErrorRate", lambda *args, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(lightning_module, "LineErrorRate", lambda *args, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(lightning_module, "GrammarProvider", lambda **kwargs: object())

    def fake_factory(**kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr(lightning_module, "ConstrainedDecodingFactory", fake_factory)

    _ = SMTTrainer(
        model_config=ModelConfig(),
        optimizer_config=OptimizerConfig(),
        training=Training(
            use_grammar_constraints=True,
            use_spine_structure_constraints=True,
            use_rhythm_constraints=True,
            runaway_guard_enabled=False,
            runaway_monitor_enabled=False,
        ),
        generation=Generation(strategy="greedy"),
        maxh=16,
        maxw=16,
        maxlen=32,
        out_categories=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        w2i={"<pad>": 0, "<bos>": 1, "<eos>": 2},
        i2w={0: "<pad>", 1: "<bos>", 2: "<eos>"},
        vocab_dir="./vocab/test",
    )

    assert captured["kwargs"]["use_spine_structure_constraints"] is True
    assert captured["kwargs"]["use_interpretation_transition_constraints"] is True
    assert captured["kwargs"]["use_rhythm_constraints"] is False
    assert captured["kwargs"]["interpretation_transition_config"] is not None
