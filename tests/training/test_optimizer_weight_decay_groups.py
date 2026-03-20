import torch

from src.training.optim.layerwise import split_named_params_for_weight_decay


class _TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3, bias=True)
        self.norm = torch.nn.LayerNorm(3)
        self.grn_scale = torch.nn.Parameter(torch.ones(3))
        self.frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)


def _collect_group_params(group):
    return {id(p) for p in group["params"]}


def test_split_named_params_for_weight_decay_routes_norm_and_bias_to_no_decay():
    module = _TinyModule()
    groups = split_named_params_for_weight_decay(
        module.named_parameters(),
        lr=1e-3,
        weight_decay=0.1,
        name_prefix="decoder",
    )

    assert len(groups) == 2
    decay_group = next(g for g in groups if g["name"] == "decoder.decay")
    no_decay_group = next(g for g in groups if g["name"] == "decoder.no_decay")
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0

    decay_params = _collect_group_params(decay_group)
    no_decay_params = _collect_group_params(no_decay_group)

    assert id(module.linear.weight) in decay_params
    assert id(module.linear.bias) in no_decay_params
    assert id(module.norm.weight) in no_decay_params
    assert id(module.norm.bias) in no_decay_params
    assert id(module.grn_scale) in no_decay_params
    assert id(module.frozen) not in decay_params
    assert id(module.frozen) not in no_decay_params


def test_split_named_params_for_weight_decay_all_no_decay_is_supported():
    module = _TinyModule()
    module.linear.weight.requires_grad = False
    groups = split_named_params_for_weight_decay(
        module.named_parameters(),
        lr=1e-3,
        weight_decay=0.1,
        name_prefix="bridge",
    )

    assert all(g["name"] != "bridge.decay" for g in groups)
    assert any(g["name"] == "bridge.no_decay" for g in groups)
