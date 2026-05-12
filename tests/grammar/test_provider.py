"""Tests for GrammarProvider."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.grammar import GrammarProvider


def _xgrammar_installed() -> bool:
    """Check if xgrammar is available."""
    try:
        import xgrammar  # noqa: F401

        return True
    except ImportError:
        return False


# Skip all tests in this module if xgrammar is not installed
pytestmark = pytest.mark.skipif(
    not _xgrammar_installed(),
    reason="xgrammar not installed (install with: uv sync --group grammar)",
)


class TestGrammarProvider:
    """Tests for GrammarProvider class."""

    def test_init_fails_without_grammar_file(self, tmp_path: Path):
        """GrammarProvider should fail if grammar file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Grammar file not found"):
            GrammarProvider(
                grammar_path=str(tmp_path / "nonexistent.gbnf"),
                vocab_dir=str(tmp_path),
                vocab_size=100,
            )

    def test_compiled_grammar_is_cached(self, kern_grammar_path: Path):
        """GrammarProvider should cache the compiled grammar."""
        # Skip if the default vocab doesn't exist
        default_vocab = Path("./vocab/bpe4k")
        if not default_vocab.exists():
            pytest.skip("Default vocab directory not found")

        provider = GrammarProvider(
            grammar_path=str(kern_grammar_path),
            vocab_dir=str(default_vocab),
            vocab_size=4004,
        )

        # Access compiled grammar twice
        grammar1 = provider.compiled_grammar
        grammar2 = provider.compiled_grammar

        # Should be the same object (cached)
        assert grammar1 is grammar2

    def test_create_logits_processor_returns_new_instance(self, kern_grammar_path: Path):
        """Each call to create_logits_processor should return a new instance."""
        # Skip if the default vocab doesn't exist
        default_vocab = Path("./vocab/bpe4k")
        if not default_vocab.exists():
            pytest.skip("Default vocab directory not found")

        provider = GrammarProvider(
            grammar_path=str(kern_grammar_path),
            vocab_dir=str(default_vocab),
            vocab_size=4004,
        )

        # Create two processors
        processor1 = provider.create_logits_processor()
        processor2 = provider.create_logits_processor()

        # Should be different instances
        assert processor1 is not processor2

class TestGrammarProviderIntegration:
    """Integration tests that require xgrammar and a real tokenizer."""

    def test_kern_grammar_compiles(self, kern_grammar_path: Path):
        """The kern.gbnf grammar should compile without errors."""
        # Skip if the default vocab doesn't exist
        default_vocab = Path("./vocab/bpe4k")
        if not default_vocab.exists():
            pytest.skip("Default vocab directory not found")

        if not kern_grammar_path.exists():
            pytest.skip(f"Grammar file not found: {kern_grammar_path}")

        # Should not raise an exception
        provider = GrammarProvider(
            grammar_path=str(kern_grammar_path),
            vocab_dir=str(default_vocab),
            vocab_size=4004,
        )

        assert provider.compiled_grammar is not None
