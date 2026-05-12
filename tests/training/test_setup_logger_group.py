from src.training.setup import _derive_wandb_group


def test_derive_wandb_group_handles_trailing_slash():
    assert _derive_wandb_group("weights/GrandStaff/") == "GrandStaff"
    assert _derive_wandb_group("weights/GrandStaff") == "GrandStaff"


def test_derive_wandb_group_falls_back_for_empty_like_paths():
    assert _derive_wandb_group("") == "default"
    assert _derive_wandb_group(".") == "default"
    assert _derive_wandb_group("/") == "default"
