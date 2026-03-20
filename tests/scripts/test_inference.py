from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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
    inference_script.inference(weights="unused.ckpt", image="unused.png", output=str(output_path))

    assert captured["factory_kwargs"]["use_rhythm_constraints"] is False
    assert captured["generate_kwargs"]["logits_processor"] == ["lp"]
    assert captured["generate_kwargs"]["stopping_criteria"] == ["stop"]
    assert captured["generate_kwargs"]["repetition_penalty"] == 1.1
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
