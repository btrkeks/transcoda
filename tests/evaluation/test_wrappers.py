from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import torch

from src.evaluation.wrappers import PRAIGModelWrapper


def test_praig_wrapper_batch_predict_decodes_batched_output(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakePRAIGModel:
        @classmethod
        def from_pretrained(cls, _model_id):
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

        def predict(self, pixel_values, image_sizes=None, convert_to_str=False):
            captured["pixel_values_shape"] = tuple(pixel_values.shape)
            captured["image_sizes"] = image_sizes.detach().cpu()
            captured["convert_to_str"] = convert_to_str
            return [["A", "<b>", "B"], ["C", "<t>", "D"]], None

    fake_module = ModuleType("smt_model")
    fake_module.SMTModelForCausalLM = FakePRAIGModel
    monkeypatch.setitem(sys.modules, "smt_model", fake_module)

    wrapper = PRAIGModelWrapper(
        "unused-model",
        torch.device("cpu"),
        praig_module_path=str(Path("models/external/praig-smt").resolve()),
    )

    predictions = wrapper.batch_predict(
        pixel_values=torch.zeros((2, 1, 4, 5), dtype=torch.float32),
        image_sizes=torch.tensor([[4, 5], [3, 2]], dtype=torch.long),
    )

    assert predictions == ["A\nB", "C\tD"]
    assert captured["pixel_values_shape"] == (2, 1, 4, 5)
    assert torch.equal(
        captured["image_sizes"],
        torch.tensor([[4, 5], [3, 2]], dtype=torch.long),
    )
    assert captured["convert_to_str"] is True


def test_praig_wrapper_batch_predict_normalizes_single_item_batch(monkeypatch) -> None:
    class FakePRAIGModel:
        @classmethod
        def from_pretrained(cls, _model_id):
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

        def predict(self, pixel_values, image_sizes=None, convert_to_str=False):
            del pixel_values, image_sizes, convert_to_str
            return ["A", "<b>", "B"], None

    fake_module = ModuleType("smt_model")
    fake_module.SMTModelForCausalLM = FakePRAIGModel
    monkeypatch.setitem(sys.modules, "smt_model", fake_module)

    wrapper = PRAIGModelWrapper(
        "unused-model",
        torch.device("cpu"),
        praig_module_path=str(Path("models/external/praig-smt").resolve()),
    )

    predictions = wrapper.batch_predict(
        pixel_values=torch.zeros((1, 1, 4, 5), dtype=torch.float32),
        image_sizes=torch.tensor([[4, 5]], dtype=torch.long),
    )

    assert predictions == ["A\nB"]
