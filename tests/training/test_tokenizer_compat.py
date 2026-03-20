import pytest

from src.artifacts import _hash_vocab_dict
from src.tokenizer_compat import assert_vocab_hashes_match, resolve_vocab_dir, vocab_from_tokenizer


class _DummyTokenizer:
    def get_vocab(self):
        return {"<bos>": 1, "b": 3, "<pad>": 0, "</eos>": 2, "a": 4}


def test_vocab_from_tokenizer_builds_inverse_map():
    w2i, i2w = vocab_from_tokenizer(_DummyTokenizer())

    assert w2i["<pad>"] == 0
    assert i2w[3] == "b"
    assert i2w[4] == "a"


def test_resolve_vocab_dir_requires_explicit_field():
    assert resolve_vocab_dir({"vocab_dir": "./vocab/bpe4k"}) == "./vocab/bpe4k"

    with pytest.raises(ValueError, match="vocab_dir"):
        resolve_vocab_dir({"vocab_name": "legacy"})


def test_assert_vocab_hashes_match_accepts_matching_hashes():
    w2i, i2w = vocab_from_tokenizer(_DummyTokenizer())
    assert_vocab_hashes_match(
        expected_w2i_hash=_hash_vocab_dict(w2i),
        expected_i2w_hash=_hash_vocab_dict(i2w),
        w2i=w2i,
        i2w=i2w,
        context_label="UnitTest",
    )


def test_assert_vocab_hashes_match_raises_on_mismatch():
    w2i, i2w = vocab_from_tokenizer(_DummyTokenizer())

    with pytest.raises(ValueError, match="UnitTest tokenizer mismatch detected"):
        assert_vocab_hashes_match(
            expected_w2i_hash="bad",
            expected_i2w_hash="hash",
            w2i=w2i,
            i2w=i2w,
            context_label="UnitTest",
        )
