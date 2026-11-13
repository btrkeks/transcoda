from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import transcoda.inference.runtime as runtime_module
from transcoda.artifacts import DecodingSpec
from transcoda.grammar.constraint_factory import ConstraintBundle
from transcoda.grammar.semantic_sequence_finalizer import FinalizedKernSequence
from transcoda.inference.runtime import GenerationOptions, InferenceRuntime, beam_count_override
from transcoda.model.generation_policy import GenerationSettings


def test_beam_strategy_defaults_to_valid_beam_count_from_greedy_checkpoint() -> None:
    settings = GenerationSettings(
        strategy="greedy",
        num_beams=1,
        length_penalty=1.0,
        repetition_penalty=1.0,
        early_stopping=True,
        num_return_sequences=1,
        use_cache=True,
        do_sample=False,
    )

    assert beam_count_override(settings, GenerationOptions(strategy="beam")) == 4


def test_explicit_beam_count_is_preserved() -> None:
    settings = GenerationSettings(
        strategy="greedy",
        num_beams=1,
        length_penalty=1.0,
        repetition_penalty=1.0,
        early_stopping=True,
        num_return_sequences=1,
        use_cache=True,
        do_sample=False,
    )

    assert beam_count_override(settings, GenerationOptions(strategy="beam", num_beams=2)) == 2


class _FakePreprocessor:
    def __call__(self, image):
        return SimpleNamespace(
            pixel_values=torch.zeros((3, 4, 4)),
            model_input_size=(4, 4),
            original_size=(image.height, image.width),
            debug_metadata={},
        )


def _patch_preprocessor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime_module.ImagePreprocessor,
        "from_checkpoint_metadata",
        lambda **_kwargs: _FakePreprocessor(),
    )


def _finalized(text: str = "4c") -> FinalizedKernSequence:
    return FinalizedKernSequence(
        text=text,
        trimmed_incomplete_tail=False,
        hit_max_length=False,
        saw_eos=True,
        truncated=False,
    )


def _runtime(model, artifact=None, grammar_provider=None) -> InferenceRuntime:
    return InferenceRuntime(
        model=model,
        i2w={0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "4c", 4: "4d"},
        pad_token_id=0,
        image_width=1050,
        fixed_size=(1485, 1050),
        artifact=artifact or SimpleNamespace(decoding=DecodingSpec(strategy="greedy", max_len=16)),
        device=torch.device("cpu"),
        weights_path=Path("unused.ckpt"),
        grammar_provider=grammar_provider,
    )


def test_runtime_load_requires_grammar_path_before_checkpoint_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_load(*_args, **_kwargs):
        raise AssertionError("checkpoint loader should not be called")

    monkeypatch.setattr(runtime_module, "load_model_from_checkpoint", fail_load)

    with pytest.raises(ValueError, match="grammar_path"):
        InferenceRuntime.load("unused.ckpt", device=torch.device("cpu"), use_grammar=True)


def test_runtime_load_forwards_explicit_grammar_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    loaded = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(out_categories=17)),
        i2w={0: "<pad>"},
        pad_token_id=0,
        image_width=1050,
        fixed_size=(1485, 1050),
        artifact=SimpleNamespace(decoding=DecodingSpec(strategy="greedy", max_len=16)),
        vocab_dir=tmp_path / "vocab",
        tokenizer_load_path=tmp_path / "resolved-vocab",
    )

    def fake_load(path, device):
        captured["load"] = (path, device)
        return loaded

    def fake_provider(**kwargs):
        captured["provider"] = kwargs
        return object()

    grammar_path = tmp_path / "grammar.gbnf"
    monkeypatch.setattr(runtime_module, "load_model_from_checkpoint", fake_load)
    monkeypatch.setattr(runtime_module, "GrammarProvider", fake_provider)

    runtime = InferenceRuntime.load(
        tmp_path / "model.ckpt",
        device=torch.device("cpu"),
        use_grammar=True,
        grammar_path=grammar_path,
    )

    assert runtime.grammar_provider is not None
    assert captured["provider"] == {
        "grammar_path": str(grammar_path),
        "vocab_dir": loaded.tokenizer_load_path,
        "vocab_size": 17,
    }
    assert runtime.vocab_dir == str(loaded.tokenizer_load_path)


def test_runtime_load_unconstrained_does_not_require_grammar_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(out_categories=17)),
        i2w={0: "<pad>"},
        pad_token_id=0,
        image_width=1050,
        fixed_size=(1485, 1050),
        artifact=SimpleNamespace(decoding=DecodingSpec(strategy="greedy", max_len=16)),
        vocab_dir=tmp_path / "vocab",
    )

    monkeypatch.setattr(runtime_module, "load_model_from_checkpoint", lambda *_args: loaded)
    monkeypatch.setattr(
        runtime_module,
        "GrammarProvider",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("not called")),
    )

    runtime = InferenceRuntime.load(
        tmp_path / "model.ckpt",
        device=torch.device("cpu"),
        use_grammar=False,
    )

    assert runtime.grammar_provider is None


def test_runtime_uses_shared_constraint_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        config = SimpleNamespace(maxlen=32, bos_token_id=1, eos_token_id=2)

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return torch.tensor([[1, 3, 2]], dtype=torch.long)

    _patch_preprocessor(monkeypatch)
    monkeypatch.setattr(
        runtime_module,
        "ConstrainedDecodingFactory",
        lambda **kwargs: (
            captured.__setitem__("factory_kwargs", kwargs)
            or SimpleNamespace(
                build=lambda settings: ConstraintBundle(
                    logits_processors=["lp"],
                    stopping_criteria=["stop"],
                    generation_settings=settings,
                    semantic_rule_factories=(),
                )
            )
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "finalize_generated_kern_sequence",
        lambda **_kwargs: _finalized("4c"),
    )

    runtime = _runtime(
        FakeModel(),
        artifact=SimpleNamespace(
            decoding=DecodingSpec(strategy="beam", num_beams=4, max_len=16)
        ),
        grammar_provider=object(),
    )
    result = runtime.run(Image.new("RGB", (8, 8), "white"), use_grammar=True)

    assert captured["factory_kwargs"]["use_spine_structure_constraints"] is True
    assert captured["factory_kwargs"]["spine_structure_barline_width"] == "per_spine"
    assert captured["factory_kwargs"]["spine_structure_empty_field_mode"] == "forbid"
    assert "use_rhythm_constraints" not in captured["factory_kwargs"]
    assert captured["generate_kwargs"]["num_beams"] == 1
    assert captured["generate_kwargs"]["logits_processor"] == ["lp"]
    assert captured["generate_kwargs"]["stopping_criteria"] == ["stop"]
    assert captured["generate_kwargs"]["repetition_penalty"] == 1.3
    assert result.kern_text == "4c"


def test_runtime_preserves_unconstrained_beam_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        config = SimpleNamespace(maxlen=32, bos_token_id=1, eos_token_id=2)

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return torch.tensor([[1, 3, 2]], dtype=torch.long)

    _patch_preprocessor(monkeypatch)
    monkeypatch.setattr(
        runtime_module,
        "finalize_generated_kern_sequence",
        lambda **_kwargs: _finalized("4c"),
    )

    runtime = _runtime(
        FakeModel(),
        artifact=SimpleNamespace(decoding=DecodingSpec(strategy="beam", num_beams=4, max_len=16)),
    )
    runtime.run(Image.new("RGB", (8, 8), "white"))

    assert captured["generate_kwargs"]["num_beams"] == 4


def test_runtime_wires_spine_only_constraints_and_audits_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        config = SimpleNamespace(maxlen=32, bos_token_id=1, eos_token_id=2)

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return torch.tensor([[1, 3, 2]], dtype=torch.long)

    _patch_preprocessor(monkeypatch)
    monkeypatch.setattr(
        runtime_module,
        "ConstrainedDecodingFactory",
        lambda **kwargs: (
            captured.__setitem__("factory_kwargs", kwargs)
            or SimpleNamespace(
                build=lambda settings: ConstraintBundle(
                    logits_processors=["spine-lp"],
                    stopping_criteria=None,
                    generation_settings=settings,
                    semantic_rule_factories=(),
                    active_constraints=("spine_structure",),
                )
            )
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "finalize_generated_kern_sequence",
        lambda **_kwargs: _finalized("4c"),
    )

    runtime = _runtime(
        FakeModel(),
        artifact=SimpleNamespace(decoding=DecodingSpec(strategy="beam", num_beams=4, max_len=16)),
    )
    result = runtime.run(
        Image.new("RGB", (8, 8), "white"),
        use_grammar=False,
        use_spine_structure_constraints=True,
        spine_structure_barline_width="per_spine",
        spine_structure_empty_field_mode="forbid",
    )

    assert captured["factory_kwargs"]["grammar_provider"] is None
    assert captured["factory_kwargs"]["use_spine_structure_constraints"] is True
    assert captured["generate_kwargs"]["num_beams"] == 1
    assert captured["generate_kwargs"]["logits_processor"] == ["spine-lp"]
    assert result.generation_metadata["active_constraints"] == ["spine_structure"]
    assert result.generation_metadata["use_grammar"] is False
    assert result.generation_metadata["use_spine_structure_constraints"] is True
    assert result.generation_metadata["spine_structure_barline_width"] == "per_spine"
    assert result.generation_metadata["spine_structure_empty_field_mode"] == "forbid"
    assert result.generation_metadata["saw_eos"] is True
    assert result.generation_metadata["hit_max_length"] is False


def test_runtime_allows_repetition_penalty_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        config = SimpleNamespace(maxlen=32, bos_token_id=1, eos_token_id=2)

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return torch.tensor([[1, 3, 2]], dtype=torch.long)

    _patch_preprocessor(monkeypatch)
    monkeypatch.setattr(
        runtime_module,
        "finalize_generated_kern_sequence",
        lambda **_kwargs: _finalized("4c"),
    )

    runtime = _runtime(FakeModel())
    runtime.run(
        Image.new("RGB", (8, 8), "white"),
        generation=GenerationOptions(repetition_penalty=1.35),
    )

    assert captured["generate_kwargs"]["repetition_penalty"] == 1.35


def test_runtime_returns_batch_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        config = SimpleNamespace(maxlen=32, bos_token_id=1, eos_token_id=2)

        def generate(self, **kwargs):
            scores = (
                torch.tensor(
                    [
                        [0.0, -10.0, -10.0, 2.0, -1.0],
                        [0.0, -10.0, -10.0, 1.0, -1.0],
                    ],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [
                        [0.0, -10.0, 3.0, -1.0, -1.0],
                        [0.0, -10.0, 2.0, -1.0, -1.0],
                    ],
                    dtype=torch.float32,
                ),
            )
            return SimpleNamespace(
                sequences=torch.tensor([[1, 3, 2], [1, 3, 2]], dtype=torch.long),
                scores=scores,
                beam_indices=None,
            )

        def compute_transition_scores(self, *args, **kwargs):
            raise RuntimeError("force greedy fallback")

    _patch_preprocessor(monkeypatch)
    monkeypatch.setattr(
        runtime_module,
        "finalize_generated_kern_sequence",
        lambda **_kwargs: _finalized("4c"),
    )

    runtime = _runtime(FakeModel())
    results = runtime.run_batch(
        [Image.new("RGB", (8, 8), "white"), Image.new("RGB", (8, 8), "white")],
        generation=GenerationOptions(include_confidence=True),
    )

    assert [result.kern_text for result in results] == ["4c", "4c"]
    assert results[0].confidence["num_scored_tokens"] == 2
    assert results[1].confidence["num_scored_tokens"] == 2
    assert results[0].confidence["tokens"] == ["4c", ""]
