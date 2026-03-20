from types import SimpleNamespace

import torch

from src.grammar.runaway_guard import RunawayBreakerLogitsProcessor, RunawayGuardConfig
from src.grammar.constraint_factory import ConstraintBundle
from src.model import VisionFrontendOutput
from src.model.generation_policy import GenerationSettings
from src.training.lightning_module import SMTTrainer


class _GenerateModelStub:
    def __init__(self):
        self.calls = []
        self.config = SimpleNamespace(pad_token_id=0, bos_token_id=1, eos_token_id=2)

    def forward_encoder(self, pixel_values, image_sizes=None):
        batch = pixel_values.shape[0]
        return VisionFrontendOutput(
            encoder_tokens_raw=torch.randn(batch, 4, 8),
            encoder_tokens_pos=torch.randn(batch, 4, 8),
            encoder_attention_mask=torch.ones(batch, 4, dtype=torch.bool),
        )

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([[1, 2, 3]])


class _ConstraintFactoryStub:
    def __init__(self, bundle: ConstraintBundle):
        self.bundle = bundle

    def build(self, settings):
        return ConstraintBundle(
            logits_processors=self.bundle.logits_processors,
            stopping_criteria=self.bundle.stopping_criteria,
            generation_settings=settings if self.bundle.generation_settings is None else self.bundle.generation_settings,
            semantic_rule_factories=self.bundle.semantic_rule_factories,
        )


def _guard_config() -> RunawayGuardConfig:
    return RunawayGuardConfig(
        max_same_control_token=8,
        max_control_lines_streak=6,
        max_spine_splits=64,
        max_spine_merges=96,
        max_ottava_markers=16,
        max_tuplet_markers=12,
        max_tremolo_markers=12,
    )


def test_guardrail_not_attached_without_grammar_provider():
    model = _GenerateModelStub()
    stub = SimpleNamespace(
        model=model,
        _compiled_forward_model=None,
        _generation_max_length=None,
        _constraint_factory=_ConstraintFactoryStub(
            ConstraintBundle(
                logits_processors=None,
                stopping_criteria=None,
                generation_settings=None,
                semantic_rule_factories=(),
            )
        ),
        _generation_settings=GenerationSettings(
            strategy="greedy",
            num_beams=1,
            length_penalty=1.0,
            repetition_penalty=1.1,
            early_stopping=True,
            num_return_sequences=1,
            use_cache=True,
            do_sample=False,
        ),
        config=SimpleNamespace(i2w={1: "<bos>", 2: "<eos>", 3: "*^"}),
    )

    _ = SMTTrainer._generate_with_grammar(
        stub,
        pixel_values=torch.randn(1, 3, 8, 8),
        image_sizes=None,
        max_length=32,
    )

    assert len(model.calls) == 1
    assert "logits_processor" not in model.calls[0]
    assert "stopping_criteria" not in model.calls[0]


def test_guardrail_attached_with_grammar_provider():
    model = _GenerateModelStub()
    grammar_lp = object()
    spine_lp = object()
    stub = SimpleNamespace(
        model=model,
        _compiled_forward_model=None,
        _generation_max_length=None,
        _constraint_factory=_ConstraintFactoryStub(
            ConstraintBundle(
                logits_processors=[
                    grammar_lp,
                    spine_lp,
                    RunawayBreakerLogitsProcessor(
                        tokenizer_i2w={1: "<bos>", 2: "<eos>", 3: "*^", 4: "*v"},
                        bos_token_id=1,
                        eos_token_id=2,
                        config=_guard_config(),
                    ),
                ],
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
        ),
        _generation_settings=GenerationSettings(
            strategy="beam",
            num_beams=4,
            length_penalty=1.0,
            repetition_penalty=1.1,
            early_stopping=True,
            num_return_sequences=1,
            use_cache=True,
            do_sample=False,
        ),
        config=SimpleNamespace(i2w={1: "<bos>", 2: "<eos>", 3: "*^", 4: "*v"}),
    )

    _ = SMTTrainer._generate_with_grammar(
        stub,
        pixel_values=torch.randn(1, 3, 8, 8),
        image_sizes=None,
        max_length=32,
    )

    assert len(model.calls) == 1
    processors = model.calls[0]["logits_processor"]
    assert len(processors) == 3
    assert processors[0] is grammar_lp
    assert processors[1] is spine_lp
    assert isinstance(processors[2], RunawayBreakerLogitsProcessor)
    assert "stopping_criteria" not in model.calls[0]
