import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import transcoda.model.checkpoint_loader as checkpoint_loader
from transcoda.artifacts import _hash_vocab_dict
from transcoda.model.checkpoint_loader import (
    _load_tokenizer_matching_artifact,
    _resolve_tokenizer_load_path,
    _resolve_vocab_dir,
)


def test_resolve_tokenizer_load_path_uses_absolute_local_paths(tmp_path: Path) -> None:
    vocab_dir = tmp_path / "vocab" / "example"
    vocab_dir.mkdir(parents=True)

    resolved = _resolve_tokenizer_load_path("./vocab/example", repo_root=tmp_path)

    assert resolved == str(vocab_dir.resolve())


def test_resolve_tokenizer_load_path_maps_legacy_splitspaces_alias(tmp_path: Path) -> None:
    alias_dir = tmp_path / "vocab" / "bpe3k-transcoda-v1"
    alias_dir.mkdir(parents=True)

    resolved = _resolve_tokenizer_load_path("./vocab/bpe3k-splitspaces", repo_root=tmp_path)

    assert resolved == str(alias_dir.resolve())


def test_resolve_tokenizer_load_path_leaves_remote_ids_unchanged() -> None:
    assert _resolve_tokenizer_load_path("org/tokenizer") == "org/tokenizer"


def test_resolve_vocab_dir_prefers_tokenizer_versioned_vocab_dir() -> None:
    artifact = SimpleNamespace(
        tokenizer=SimpleNamespace(versioned_vocab_dir="./vocab/bpe3k-transcoda-v2-20260622"),
        experiment_config={"data": {"vocab_dir": "./vocab/bpe3k-transcoda-v2"}},
    )

    assert _resolve_vocab_dir(artifact) == "./vocab/bpe3k-transcoda-v2-20260622"


def test_resolve_vocab_dir_falls_back_to_legacy_vocab_name() -> None:
    artifact = SimpleNamespace(
        tokenizer=SimpleNamespace(versioned_vocab_dir=None),
        experiment_config={"data": {"vocab_name": "legacy-bpe"}},
    )

    assert _resolve_vocab_dir(artifact) == "./vocab/legacy-bpe"


def test_load_model_from_checkpoint_rejects_tokenizer_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "4C": 3}
    expected_vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "4D": 3}
    expected_i2w = {idx: token for token, idx in expected_vocab.items()}

    artifact = {
        "experiment_config": {
            "data": {"vocab_dir": "./vocab/example"},
            "model": {},
        },
        "preprocessing": {"image_width": 1050, "fixed_size": [1485, 1050]},
        "decoding": {},
        "vocab": {
            "w2i_hash": _hash_vocab_dict(expected_vocab),
            "i2w_hash": _hash_vocab_dict(expected_i2w),
            "pad_token": 0,
            "bos_token": "<bos>",
            "eos_token": "<eos>",
        },
        "tokenizer": {"vocab_size": 4},
        "env": {
            "torch": "test",
            "lightning": "test",
            "transformers": "test",
            "cuda": None,
            "cudnn": None,
            "python": "test",
            "platform": "test",
        },
        "seed": {"global_seed": 42, "deterministic": False},
        "slurm": None,
    }
    checkpoint = {
        "hyper_parameters": {
            "run_artifact_json": json.dumps(artifact),
        },
    }

    monkeypatch.setattr(
        checkpoint_loader.torch,
        "load",
        lambda *_args, **_kwargs: checkpoint,
    )
    monkeypatch.setattr(
        checkpoint_loader.PreTrainedTokenizerFast,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(get_vocab=lambda: actual_vocab),
    )

    with pytest.raises(ValueError, match="tokenizer mismatch detected"):
        checkpoint_loader.load_model_from_checkpoint(
            "fake.ckpt", checkpoint_loader.torch.device("cpu")
        )


def test_load_tokenizer_matching_artifact_falls_back_to_matching_vocab_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    current_vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "4C": 3}
    historical_vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "4D": 3}
    historical_i2w = {idx: token for token, idx in historical_vocab.items()}
    current_dir = tmp_path / "vocab" / "bpe3k-transcoda-v2"
    historical_dir = tmp_path / "vocab" / "bpe3k-transcoda-v2-20260615"
    current_dir.mkdir(parents=True)
    historical_dir.mkdir()
    (current_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (historical_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    artifact = SimpleNamespace(
        vocab=SimpleNamespace(
            w2i_hash=_hash_vocab_dict(historical_vocab),
            i2w_hash=_hash_vocab_dict(historical_i2w),
        )
    )

    def fake_from_pretrained(path: str):
        vocab = historical_vocab if str(path) == str(historical_dir) else current_vocab
        return SimpleNamespace(get_vocab=lambda: vocab)

    monkeypatch.setattr(
        checkpoint_loader.PreTrainedTokenizerFast,
        "from_pretrained",
        fake_from_pretrained,
    )

    loaded = _load_tokenizer_matching_artifact(
        "./vocab/bpe3k-transcoda-v2",
        artifact,
        "fake.ckpt",
        repo_root=tmp_path,
    )

    assert loaded.tokenizer_load_path == str(historical_dir)
    assert loaded.w2i == historical_vocab


def test_load_tokenizer_matching_artifact_reports_searched_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    actual_vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "4C": 3}
    expected_vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "4D": 3}
    expected_i2w = {idx: token for token, idx in expected_vocab.items()}
    current_dir = tmp_path / "vocab" / "bpe3k-transcoda-v2"
    current_dir.mkdir(parents=True)
    (current_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    artifact = SimpleNamespace(
        vocab=SimpleNamespace(
            w2i_hash=_hash_vocab_dict(expected_vocab),
            i2w_hash=_hash_vocab_dict(expected_i2w),
        )
    )
    monkeypatch.setattr(
        checkpoint_loader.PreTrainedTokenizerFast,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(get_vocab=lambda: actual_vocab),
    )

    with pytest.raises(ValueError, match="Searched tokenizer candidates"):
        _load_tokenizer_matching_artifact(
            "./vocab/bpe3k-transcoda-v2",
            artifact,
            "fake.ckpt",
            repo_root=tmp_path,
        )
