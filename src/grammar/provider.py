"""Grammar provider for constrained decoding."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .logits_processor import GrammarConstrainedLogitsProcessor
from .tokenizer_adapter import create_tokenizer_info

if TYPE_CHECKING:
    import xgrammar as xgr
    from transformers import LogitsProcessor

# Default number of threads for grammar compilation
_DEFAULT_MAX_THREADS = 8


class GrammarProvider:
    """Factory for grammar-constrained generation components.

    Compiles the GBNF grammar eagerly at initialization to fail fast on
    grammar errors. The compiled grammar is cached and reused across all
    generation calls.

    Requires xgrammar to be installed: uv sync --group grammar

    Example:
        provider = GrammarProvider("grammars/kern.gbnf", vocab_dir, vocab_size)
        # In validation_step:
        logits_processor = provider.create_logits_processor(pad_token_id=pad_token_id)
        preds = model.generate(..., logits_processor=[logits_processor])
    """

    def __init__(
        self,
        grammar_path: str,
        vocab_dir: str,
        vocab_size: int,
        max_threads: int = _DEFAULT_MAX_THREADS,
    ) -> None:
        """Initialize grammar provider with eager compilation.

        Args:
            grammar_path: Path to GBNF grammar file.
            vocab_dir: Path to directory containing tokenizer.json.
            vocab_size: Model's vocabulary size for tokenizer info.
            max_threads: Number of threads for grammar compilation.

        Raises:
            FileNotFoundError: If grammar file does not exist.
            ImportError: If xgrammar is not installed.
            Exception: If grammar compilation fails (syntax errors, etc.)
        """
        self._grammar_path = grammar_path
        self._vocab_dir = vocab_dir
        self._vocab_size = vocab_size
        self._max_threads = max_threads

        # Eager compilation - fail fast on grammar errors
        self._compiled_grammar = self._compile()

    def _compile(self) -> "xgr.CompiledGrammar":
        """Compile the GBNF grammar with xgrammar.

        Returns:
            Compiled grammar ready for creating matchers.

        Raises:
            FileNotFoundError: If grammar file does not exist.
            ImportError: If xgrammar is not installed.
            Exception: If grammar compilation fails.
        """
        # Check grammar file exists before expensive tokenizer/compiler setup
        grammar_path = Path(self._grammar_path)
        if not grammar_path.exists():
            raise FileNotFoundError(f"Grammar file not found: {grammar_path}")

        import xgrammar as xgr

        # Create tokenizer info from saved HF tokenizer
        tokenizer_info = create_tokenizer_info(self._vocab_dir, self._vocab_size)

        # Create compiler
        compiler = xgr.GrammarCompiler(
            tokenizer_info,
            max_threads=self._max_threads,
        )

        grammar_text = grammar_path.read_text()
        return compiler.compile_grammar(grammar_text)

    def create_logits_processor(
        self,
        *,
        pad_token_id: int | None = None,
        collect_stats: bool = False,
    ) -> "LogitsProcessor":
        """Create a fresh LogitsProcessor for a generation batch.

        A new processor is created for each batch to ensure clean matcher
        state. The underlying compiled grammar is reused.

        Returns:
            HuggingFace-compatible LogitsProcessor that enforces the grammar.
        """
        return GrammarConstrainedLogitsProcessor(
            self._compiled_grammar,
            pad_token_id=pad_token_id,
            collect_stats=collect_stats,
        )

    @property
    def compiled_grammar(self) -> "xgr.CompiledGrammar":
        """Access the compiled grammar (for testing/debugging)."""
        return self._compiled_grammar
