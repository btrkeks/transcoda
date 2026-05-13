from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import inference as inference_script
from src.artifacts import DecodingSpec
from src.grammar.constraint_factory import ConstraintBundle
from src.grammar.semantic_sequence_finalizer import FinalizedKernSequence


def test_inference_uses_shared_constraint_factory(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(maxlen=32, bos_token_id=1, eos_token_id=2)

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return torch.tensor([[1, 3, 2]], dtype=torch.long)

    fake_artifact = SimpleNamespace(decoding=DecodingSpec(strategy="beam", num_beams=4, max_len=16))
    fake_model = FakeModel()

    monkeypatch.setattr(
        inference_script,
        "load_model_and_grammar",
        lambda *_args: (
            fake_model,
            {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "4c"},
            0,
            object(),
            1050,
            (1485, 1050),
            fake_artifact,
        ),
    )
    monkeypatch.setattr(
        inference_script,
        "preprocess_image",
        lambda *_args, **_kwargs: (torch.zeros((1, 3, 4, 4)), (4, 4), (4, 4)),
    )
    monkeypatch.setattr(
        inference_script,
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
        inference_script,
        "finalize_generated_kern_sequence",
        lambda **_kwargs: FinalizedKernSequence(
            text="4c",
            trimmed_incomplete_tail=False,
            appended_terminator=False,
            hit_max_length=False,
            saw_eos=True,
            truncated=False,
        ),
    )

    output_path = tmp_path / "out.krn"
    inference_script.inference(
        weights="unused.ckpt",
        image="unused.png",
        output=str(output_path),
        use_grammar=True,
    )

    assert captured["factory_kwargs"]["use_rhythm_constraints"] is False
    assert captured["factory_kwargs"]["use_interpretation_transition_constraints"] is True
    assert captured["factory_kwargs"]["use_spine_structure_constraints"] is True
    assert captured["generate_kwargs"]["num_beams"] == 1
    assert captured["generate_kwargs"]["logits_processor"] == ["lp"]
    assert captured["generate_kwargs"]["stopping_criteria"] == ["stop"]
    assert captured["generate_kwargs"]["repetition_penalty"] == 1.3
    assert output_path.read_text() == "**kern\t**kern\n4c\n*-"


def test_inference_allows_repetition_penalty_override(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(maxlen=32, bos_token_id=1, eos_token_id=2)

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return torch.tensor([[1, 3, 2]], dtype=torch.long)

    fake_artifact = SimpleNamespace(decoding=DecodingSpec(strategy="greedy", num_beams=1, max_len=16))

    monkeypatch.setattr(
        inference_script,
        "load_model_and_grammar",
        lambda *_args: (
            FakeModel(),
            {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "4c"},
            0,
            object(),
            1050,
            (1485, 1050),
            fake_artifact,
        ),
    )
    monkeypatch.setattr(
        inference_script,
        "preprocess_image",
        lambda *_args, **_kwargs: (torch.zeros((1, 3, 4, 4)), (4, 4), (4, 4)),
    )
    monkeypatch.setattr(
        inference_script,
        "ConstrainedDecodingFactory",
        lambda **_kwargs: SimpleNamespace(
            build=lambda settings: ConstraintBundle(
                logits_processors=None,
                stopping_criteria=None,
                generation_settings=settings,
                semantic_rule_factories=(),
            )
        ),
    )
    monkeypatch.setattr(
        inference_script,
        "finalize_generated_kern_sequence",
        lambda **_kwargs: FinalizedKernSequence(
            text="4c",
            trimmed_incomplete_tail=False,
            appended_terminator=False,
            hit_max_length=False,
            saw_eos=True,
            truncated=False,
        ),
    )

    output_path = tmp_path / "out.krn"
    inference_script.inference(
        weights="unused.ckpt",
        image="unused.png",
        output=str(output_path),
        repetition_penalty=1.35,
    )

    assert captured["generate_kwargs"]["repetition_penalty"] == 1.35


def test_inference_writes_confidence_sidecar(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(maxlen=32, bos_token_id=1, eos_token_id=2)

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            scores = (
                torch.tensor([[0.0, -10.0, -10.0, 2.0, -1.0]], dtype=torch.float32),
                torch.tensor([[0.0, -10.0, -10.0, -1.0, 1.0]], dtype=torch.float32),
                torch.tensor([[0.0, -10.0, 3.0, -1.0, -1.0]], dtype=torch.float32),
            )
            return SimpleNamespace(
                sequences=torch.tensor([[1, 3, 4, 2]], dtype=torch.long),
                scores=scores,
                beam_indices=None,
            )

    fake_artifact = SimpleNamespace(decoding=DecodingSpec(strategy="greedy", num_beams=1, max_len=16))

    monkeypatch.setattr(
        inference_script,
        "load_model_and_grammar",
        lambda *_args: (
            FakeModel(),
            {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "4c", 4: "4d"},
            0,
            None,
            1050,
            (1485, 1050),
            fake_artifact,
        ),
    )
    monkeypatch.setattr(
        inference_script,
        "preprocess_image",
        lambda *_args, **_kwargs: (torch.zeros((1, 3, 4, 4)), (4, 4), (4, 4)),
    )
    monkeypatch.setattr(
        inference_script,
        "finalize_generated_kern_sequence",
        lambda **_kwargs: FinalizedKernSequence(
            text="4c4d",
            trimmed_incomplete_tail=False,
            appended_terminator=False,
            hit_max_length=False,
            saw_eos=True,
            truncated=False,
        ),
    )

    output_path = tmp_path / "out.krn"
    confidence_path = tmp_path / "confidence.json"
    inference_script.inference(
        weights="unused.ckpt",
        image="unused.png",
        output=str(output_path),
        confidence_output=str(confidence_path),
    )

    assert output_path.read_text() == "**kern\t**kern\n4c4d\n*-"
    assert captured["generate_kwargs"]["return_dict_in_generate"] is True
    assert captured["generate_kwargs"]["output_scores"] is True

    confidence = json.loads(confidence_path.read_text())
    assert confidence["confidence_kind"] == "post_processor_generation_scores"
    assert confidence["num_scored_tokens"] == 3
    assert confidence["saw_eos"] is True
    assert confidence["hit_max_length"] is False

    manual_logprobs = torch.tensor(
        [
            torch.log_softmax(
                torch.tensor([0.0, -10.0, -10.0, 2.0, -1.0]), dim=-1
            )[3],
            torch.log_softmax(
                torch.tensor([0.0, -10.0, -10.0, -1.0, 1.0]), dim=-1
            )[4],
            torch.log_softmax(
                torch.tensor([0.0, -10.0, 3.0, -1.0, -1.0]), dim=-1
            )[2],
        ]
    )
    manual_probs = manual_logprobs.exp()
    assert confidence["mean_logprob"] == pytest.approx(float(manual_logprobs.mean()))
    assert confidence["mean_prob"] == pytest.approx(float(manual_probs.mean()))
    assert confidence["min_prob"] == pytest.approx(float(manual_probs.min()))
    assert confidence["p05_prob"] == pytest.approx(float(torch.quantile(manual_probs, 0.05)))
    assert confidence["p10_prob"] == pytest.approx(float(torch.quantile(manual_probs, 0.10)))


def test_inference_print_confidence_uses_default_sidecar_path(monkeypatch, tmp_path: Path) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(maxlen=32, bos_token_id=1, eos_token_id=2)

        def generate(self, **_kwargs):
            return SimpleNamespace(
                sequences=torch.tensor([[1, 3, 2]], dtype=torch.long),
                scores=(torch.tensor([[0.0, -10.0, -10.0, 1.0]], dtype=torch.float32),),
                beam_indices=None,
            )

    fake_artifact = SimpleNamespace(decoding=DecodingSpec(strategy="greedy", num_beams=1, max_len=16))

    monkeypatch.setattr(
        inference_script,
        "load_model_and_grammar",
        lambda *_args: (
            FakeModel(),
            {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "4c"},
            0,
            None,
            1050,
            (1485, 1050),
            fake_artifact,
        ),
    )
    monkeypatch.setattr(
        inference_script,
        "preprocess_image",
        lambda *_args, **_kwargs: (torch.zeros((1, 3, 4, 4)), (4, 4), (4, 4)),
    )
    monkeypatch.setattr(
        inference_script,
        "finalize_generated_kern_sequence",
        lambda **_kwargs: FinalizedKernSequence(
            text="4c",
            trimmed_incomplete_tail=False,
            appended_terminator=False,
            hit_max_length=False,
            saw_eos=True,
            truncated=False,
        ),
    )

    output_path = tmp_path / "out.krn"
    inference_script.inference(
        weights="unused.ckpt",
        image="unused.png",
        output=str(output_path),
        print_confidence=True,
    )

    default_sidecar = tmp_path / "out.krn.confidence.json"
    assert default_sidecar.exists()
    assert json.loads(default_sidecar.read_text())["num_scored_tokens"] == 1


def test_inference_preserves_unconstrained_beam_settings(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(maxlen=32, bos_token_id=1, eos_token_id=2)

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return torch.tensor([[1, 3, 2]], dtype=torch.long)

    fake_artifact = SimpleNamespace(decoding=DecodingSpec(strategy="beam", num_beams=4, max_len=16))

    monkeypatch.setattr(
        inference_script,
        "load_model_and_grammar",
        lambda *_args: (
            FakeModel(),
            {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "4c"},
            0,
            None,
            1050,
            (1485, 1050),
            fake_artifact,
        ),
    )
    monkeypatch.setattr(
        inference_script,
        "preprocess_image",
        lambda *_args, **_kwargs: (torch.zeros((1, 3, 4, 4)), (4, 4), (4, 4)),
    )
    monkeypatch.setattr(
        inference_script,
        "finalize_generated_kern_sequence",
        lambda **_kwargs: FinalizedKernSequence(
            text="4c",
            trimmed_incomplete_tail=False,
            appended_terminator=False,
            hit_max_length=False,
            saw_eos=True,
            truncated=False,
        ),
    )

    inference_script.inference(weights="unused.ckpt", image="unused.png", output=str(tmp_path / "out.krn"))

    assert captured["generate_kwargs"]["num_beams"] == 4
