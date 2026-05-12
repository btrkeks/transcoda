from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import BpeTrainer

SPECIALS_DEFAULT = ["<pad>", "<bos>", "<eos>"]


class DebugStats:
    """Helper class to collect and persist debug statistics during BPE training."""

    def __init__(self):
        self.meta: dict = {}
        self.start_time = time.time()

    def log(self, **kv):
        """Add key-value pairs to the debug metadata."""
        self.meta.update(kv)

    def compute_hash(self, text: str) -> str:
        """Compute a SHA256 hash of the text (for checksum verification)."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def percentile(self, values: list, p: int) -> float:
        """Compute the p-th percentile of a list of values."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def dump(self, path: str | Path) -> None:
        """Save debug metadata to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)
        print(f"\n[DEBUG] Stats saved to: {path}")


@dataclass(frozen=True)
class TokenizerConfig:
    type: str = "word"  # "word" | "bpe"
    vocab_size: int | None = None
    min_freq: int = 2
    specials: list[str] | None = None


class BaseTokenizer:
    def encode_tokens(self, tokens: Sequence[str]) -> list[str]:
        raise NotImplementedError

    def save(self, dirpath: str | Path, name: str) -> None:
        raise NotImplementedError

    @staticmethod
    def load(dirpath: str | Path, name: str) -> BaseTokenizer:
        raise NotImplementedError

    @property
    def w2i(self) -> dict[str, int]:
        raise NotImplementedError

    @property
    def i2w(self) -> dict[int, str]:
        raise NotImplementedError


class BPETokenizer(BaseTokenizer):
    """Byte-Pair Encoding (BPE) tokenizer for subword segmentation using HuggingFace tokenizers.

    This tokenizer learns to split word-level tokens into frequently-occurring
    character sequences (subwords) to reduce vocabulary size while maintaining
    the ability to represent rare tokens.

    The algorithm operates WITHIN tokens, not across token boundaries:
        Input:  ['4c', '16ccc', '=1']  (word-level tokens)
        Output: ['4c', '16', 'ccc', '=1']  (subword tokens)

    Attributes:
        vocab_size: Target vocabulary size
        min_freq: Minimum frequency threshold for merges
        specials: Special tokens (e.g., <pad>, <bos>, <eos>)
        hf_tokenizer: Underlying HuggingFace Tokenizer instance
    """

    def __init__(self, vocab_size: int, min_freq: int = 2, specials: list[str] | None = None):
        self.vocab_size = int(vocab_size)
        self.min_freq = int(min_freq)
        self.specials = specials or SPECIALS_DEFAULT

        # Initialize HuggingFace BPE tokenizer
        self.hf_tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        # Use WhitespaceSplit to respect token boundaries
        self.hf_tokenizer.pre_tokenizer = WhitespaceSplit()

        self._w2i: dict[str, int] = {}
        self._i2w: dict[int, str] = {}

    def train(
        self,
        sequences: Iterable[Sequence[str]],
        verbose: bool = True,
        debug: bool = True,
        debug_output_dir: str | Path | None = None,
        debug_sample_size: int = 1000,
    ) -> None:
        """Train BPE on word-level token sequences using HuggingFace tokenizers.

        This method learns subword merge rules by analyzing character-level patterns
        within individual tokens using the industry-standard HuggingFace tokenizers library.

        Args:
            sequences: Iterable of token sequences. Each sequence should be a list/sequence
                of word-level string tokens. For example:
                    [['4c', '8d', '=1'], ['2e', '4f'], ...]

                NOT: ['4c 8d =1', '2e 4f', ...]  (raw strings - wrong!)

            verbose: If True, print detailed logging about training progress
            debug: If True, collect comprehensive debug statistics
            debug_output_dir: Directory to save debug artifacts (if None, uses current dir)
            debug_sample_size: Number of sequences to sample for char/pair diagnostics

        Note:
            After training, the learned merge rules are stored in the HuggingFace tokenizer
            and can be applied to new tokens via encode_tokens().
        """
        # Initialize debug stats
        dbg = DebugStats() if debug else None

        if debug:
            # Log platform and environment info
            dbg.log(
                python_version=sys.version,
                platform=platform.platform(),
                tokenizers_version=getattr(__import__("tokenizers"), "__version__", "unknown"),
            )

        if verbose:
            print("\n" + "=" * 70)
            print("BPE TRAINING - HuggingFace Tokenizers (DEBUG MODE)")
            print("=" * 70)

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 1: Iterator Audit (verify shape, types, counts)
        # ═══════════════════════════════════════════════════════════════════════
        if verbose:
            print("\n[SECTION 1] Iterator Audit")

        # Materialize sequences to a list for analysis
        sequences_list = list(sequences)
        N = len(sequences_list)

        if debug:
            dbg.log(num_sequences=N)

        # Assert non-empty
        assert N > 0, "ERROR: sequences_list is empty!"

        # Compute sequence lengths and total tokens
        lengths = [len(seq) for seq in sequences_list]
        total_tokens = sum(lengths)

        if debug:
            dbg.log(
                total_tokens=total_tokens,
                avg_seq_len=total_tokens / N if N > 0 else 0,
                min_seq_len=min(lengths) if lengths else 0,
                max_seq_len=max(lengths) if lengths else 0,
                p50_seq_len=dbg.percentile(lengths, 50),
                p90_seq_len=dbg.percentile(lengths, 90),
                p99_seq_len=dbg.percentile(lengths, 99),
            )

        if verbose:
            print(f"  Sequences: {N}")
            print(f"  Total tokens: {total_tokens}")
            print(f"  Avg seq len: {total_tokens / N:.1f}")
            if debug and dbg:
                print(
                    f"  Seq len (min/p50/p90/p99/max): {min(lengths)}/{dbg.percentile(lengths, 50):.0f}/{dbg.percentile(lengths, 90):.0f}/{dbg.percentile(lengths, 99):.0f}/{max(lengths)}"
                )
            else:
                # Compute percentiles inline when debug is off
                sorted_lengths = sorted(lengths)
                p50 = sorted_lengths[len(sorted_lengths) // 2]
                p90 = sorted_lengths[int(len(sorted_lengths) * 0.9)]
                p99 = sorted_lengths[int(len(sorted_lengths) * 0.99)]
                print(
                    f"  Seq len (min/p50/p90/p99/max): {min(lengths)}/{p50}/{p90}/{p99}/{max(lengths)}"
                )

        # Show first sequence structure
        first_seq = list(sequences_list[0])
        if verbose:
            print(f"\n  First sequence (length {len(first_seq)}):")
            print(f"    Type: {type(first_seq).__name__}")
            print(f"    First 10 items: {first_seq[:10]}")
            if first_seq:
                print(f"    First item type: {type(first_seq[0]).__name__}")

        # Type assertion: all tokens must be strings
        bad_types = []
        for i, seq in enumerate(sequences_list):
            if i >= 100:  # Check first 100 sequences thoroughly
                break
            for j, token in enumerate(seq):
                if not isinstance(token, str):
                    bad_types.append((i, j, type(token).__name__, repr(token)))
                    if len(bad_types) >= 10:  # Limit error samples
                        break

        if bad_types:
            raise TypeError(
                f"ERROR: Found {len(bad_types)} non-string tokens in sequences! "
                f"Examples: {bad_types[:3]}"
            )

        if verbose:
            print("  ✓ Type check passed: all tokens are strings")

        # Assert total tokens > 0
        assert total_tokens > 0, "ERROR: total_tokens is 0!"

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 2: Char/Pair Diagnostics (sample-based)
        # ═══════════════════════════════════════════════════════════════════════
        if verbose:
            print("\n[SECTION 2] Character/Pair Frequency Analysis")

        char_counter = Counter()
        pair_counter = Counter()

        # Sample sequences for diagnostics
        sample_size = min(debug_sample_size, N)
        sample_indices = (
            np.random.choice(N, size=sample_size, replace=False) if N > sample_size else range(N)
        )

        for idx in sample_indices:
            seq = sequences_list[idx]
            for token in seq:
                # Count characters
                for char in token:
                    char_counter[char] += 1

                # Count within-token character pairs
                for i in range(len(token) - 1):
                    pair = token[i : i + 2]
                    pair_counter[pair] += 1

        top_chars = char_counter.most_common(30)
        top_pairs = pair_counter.most_common(30)

        if verbose:
            print(f"  Sampled {sample_size} sequences")
            print(f"  Top 15 characters: {top_chars[:15]}")
            print(f"  Top 15 pairs: {top_pairs[:15]}")

        if debug:
            dbg.log(
                sample_size=sample_size,
                top_30_chars=top_chars,
                top_30_pairs=top_pairs,
                unique_chars=len(char_counter),
                unique_pairs=len(pair_counter),
            )

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 3: Training Iterator with Counting
        # ═══════════════════════════════════════════════════════════════════════
        if verbose:
            print("\n[SECTION 3] Building Training Iterator")

        yield_count = 0
        all_yielded_text = []  # For checksum (optional, memory intensive)

        def sequence_iterator():
            nonlocal yield_count
            for seq in sequences_list:
                line = " ".join(seq)
                yield_count += 1

                if verbose and yield_count <= 2:
                    print(f"  Line {yield_count} (first 120 chars): {line[:120]}")

                # Optional: collect for checksum (disable if memory is a concern)
                if debug and yield_count <= 100:  # Only first 100 for checksum
                    all_yielded_text.append(line)

                yield line

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 4: Trainer Configuration
        # ═══════════════════════════════════════════════════════════════════════
        if verbose:
            print("\n[SECTION 4] Trainer Configuration")

        all_specials = list(self.specials) + ["<unk>"]
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_freq,
            special_tokens=all_specials,
            show_progress=verbose,
        )

        if verbose:
            print(f"  Vocab size: {self.vocab_size}")
            print(f"  Min frequency: {self.min_freq}")
            print(f"  Special tokens: {all_specials}")
            print(f"  Pre-tokenizer: {type(self.hf_tokenizer.pre_tokenizer).__name__}")

        if debug:
            dbg.log(
                vocab_size=self.vocab_size,
                min_frequency=self.min_freq,
                special_tokens=all_specials,
                pre_tokenizer=type(self.hf_tokenizer.pre_tokenizer).__name__,
            )

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 5: Training
        # ═══════════════════════════════════════════════════════════════════════
        if verbose:
            print("\n[SECTION 5] Training BPE...")

        train_start = time.time()

        # Train with length hint
        self.hf_tokenizer.train_from_iterator(sequence_iterator(), trainer=trainer, length=N)

        train_elapsed = time.time() - train_start

        # Assert that we yielded the expected number of lines
        if yield_count != N:
            print(f"  WARNING: Expected to yield {N} lines, but yielded {yield_count}!")

        if debug:
            dbg.log(
                yielded_lines=yield_count,
                expected_lines=N,
                train_elapsed_sec=train_elapsed,
            )

            # Compute checksum of first 100 yielded lines
            if all_yielded_text:
                checksum_text = "\n".join(all_yielded_text)
                checksum = dbg.compute_hash(checksum_text)
                dbg.log(checksum_first_100_lines=checksum)

        if verbose:
            print(f"  Training completed in {train_elapsed:.2f}s")
            print(f"  Yielded lines: {yield_count} (expected: {N})")

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 6: Post-Training Analysis
        # ═══════════════════════════════════════════════════════════════════════
        if verbose:
            print("\n[SECTION 6] Post-Training Analysis")

        # Get vocab
        vocab = self.hf_tokenizer.get_vocab()

        # Try to get merges (API may vary)
        try:
            # Try the standard way
            if hasattr(self.hf_tokenizer.model, "get_merges"):
                merges = self.hf_tokenizer.model.get_merges()
            else:
                # Alternative: access via save/load
                merges = []
                if hasattr(self.hf_tokenizer.model, "merges"):
                    merges = self.hf_tokenizer.model.merges
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not retrieve merges: {e}")
            merges = []

        if verbose:
            print(f"  Merges learned: {len(merges)}")
            print(f"  Vocab size: {len(vocab)}")

        if debug:
            dbg.log(
                num_merges=len(merges),
                final_vocab_size=len(vocab),
            )

        # Show top tokens by ID
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        top_20_tokens = sorted_vocab[:20]

        if verbose:
            print(f"  Top 20 tokens by ID: {top_20_tokens}")

        if debug:
            dbg.log(top_20_tokens_by_id=top_20_tokens)

        # Check special token IDs
        pad_id = vocab.get("<pad>", None)
        unk_id = vocab.get("<unk>", None)

        if verbose:
            print(f"  <pad> ID: {pad_id}")
            print(f"  <unk> ID: {unk_id}")

        if debug:
            dbg.log(pad_token_id=pad_id, unk_token_id=unk_id)

        # Ensure <pad> is always at index 0
        if "<pad>" in vocab:
            pad_id = vocab["<pad>"]
            if pad_id != 0:
                if verbose:
                    print("  Rebuilding vocab to place <pad> at index 0...")
                # Rebuild vocab with <pad> at 0
                sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
                new_vocab = {"<pad>": 0}
                idx = 1
                for token, _ in sorted_vocab:
                    if token != "<pad>":
                        new_vocab[token] = idx
                        idx += 1
                vocab = new_vocab

        self._w2i = vocab
        self._i2w = {i: w for w, i in vocab.items()}

        # Assertion: vocab should be reasonably sized
        min_expected_vocab = len(all_specials) + 100
        if len(vocab) < min_expected_vocab:
            print(
                f"\n  WARNING: Vocabulary size ({len(vocab)}) is suspiciously small! "
                f"Expected at least {min_expected_vocab}."
            )

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 7: Save Debug Artifacts
        # ═══════════════════════════════════════════════════════════════════════
        if debug:
            if verbose:
                print("\n[SECTION 7] Saving Debug Artifacts")

            output_dir = Path(debug_output_dir) if debug_output_dir else Path(".")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save debug stats
            debug_stats_path = output_dir / "debug_stats.json"
            dbg.dump(debug_stats_path)

            # Save char/pair counts
            char_pair_path = output_dir / "char_pair_top.json"
            with open(char_pair_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "top_30_chars": top_chars,
                        "top_30_pairs": top_pairs,
                        "unique_chars": len(char_counter),
                        "unique_pairs": len(pair_counter),
                    },
                    f,
                    indent=2,
                )
            if verbose:
                print(f"  Saved: {char_pair_path}")

            # Save trainer config
            trainer_config_path = output_dir / "trainer_config.json"
            with open(trainer_config_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "vocab_size": self.vocab_size,
                        "min_frequency": self.min_freq,
                        "special_tokens": all_specials,
                        "pre_tokenizer": type(self.hf_tokenizer.pre_tokenizer).__name__,
                    },
                    f,
                    indent=2,
                )
            if verbose:
                print(f"  Saved: {trainer_config_path}")

        if verbose:
            print("=" * 70 + "\n")

    def encode_tokens(self, tokens: Sequence[str]) -> list[str]:
        """Encode word-level tokens into BPE subword tokens.

        Applies learned BPE merge rules to split tokens into subwords.
        Special tokens are passed through unchanged.

        Args:
            tokens: Sequence of word-level tokens, e.g., ['4c', '16ccc', '<pad>']

        Returns:
            List of subword tokens after BPE encoding.
            Example: ['4c', '16', 'ccc', '<pad>']

        Note:
            - Special tokens (in self.specials) are never split
            - Each token is processed independently (no cross-token merging)
            - Rare tokens may be split into character-level pieces if no merges apply
        """
        # Use is_pretokenized=True to respect token boundaries
        encoding = self.hf_tokenizer.encode(list(tokens), is_pretokenized=True)
        return encoding.tokens

    @property
    def w2i(self) -> dict[str, int]:
        return self._w2i

    @property
    def i2w(self) -> dict[int, str]:
        return self._i2w

    def save(self, dirpath: str | Path, name: str) -> None:
        """Save the tokenizer to disk.

        Saves both the HuggingFace tokenizer JSON and metadata for compatibility.

        Args:
            dirpath: Directory path to save files
            name: Name prefix for saved files
        """
        p = Path(dirpath)
        p.mkdir(parents=True, exist_ok=True)

        # Save HuggingFace tokenizer
        self.hf_tokenizer.save(str(p / f"{name}tokenizer.json"))

        # Save metadata for compatibility
        meta = {
            "type": "bpe",
            "vocab_size": self.vocab_size,
            "min_freq": self.min_freq,
            "specials": self.specials,
            "backend": "huggingface",
        }
        (p / f"{name}metadata.json").write_text(json.dumps(meta, indent=2))

    @staticmethod
    def load(dirpath: str | Path, name: str) -> BPETokenizer:
        """Load a tokenizer from disk.

        Args:
            dirpath: Directory path containing saved files
            name: Name prefix of saved files

        Returns:
            Loaded BPETokenizer instance
        """
        p = Path(dirpath)

        # Load metadata
        meta = json.loads((p / f"{name}metadata.json").read_text())

        # Create instance
        tokenizer = BPETokenizer(
            vocab_size=meta.get("vocab_size", 8000),
            min_freq=meta.get("min_freq", 2),
            specials=meta.get("specials", SPECIALS_DEFAULT),
        )

        # Load HuggingFace tokenizer
        tokenizer.hf_tokenizer = Tokenizer.from_file(str(p / f"{name}tokenizer.json"))

        # Build w2i and i2w from loaded vocab
        vocab = tokenizer.hf_tokenizer.get_vocab()
        tokenizer._w2i = vocab
        tokenizer._i2w = {i: w for w, i in vocab.items()}

        return tokenizer


class BPEVocabulary:
    """BPE-based vocabulary for subword tokenization using HuggingFace tokenizers.

    This class wraps a BPETokenizer and provides vocabulary building/loading functionality.
    The BPE algorithm learns to merge frequent character sequences within tokens to create
    a compact subword vocabulary.

    Uses the industry-standard HuggingFace tokenizers library for fast, reliable BPE training.

    Attributes:
        tokenizer: The underlying BPETokenizer instance (wrapping HF tokenizer)
        w2i: Word-to-index mapping (dict[str, int])
        i2w: Index-to-word mapping (dict[int, str])
    """

    def __init__(self, tokenizer: BPETokenizer):
        self.tokenizer = tokenizer
        self.w2i = tokenizer.w2i
        self.i2w = tokenizer.i2w

    @classmethod
    def build(
        cls,
        sequences_list: list[list[str]],
        save_path: str,
        name: str,
        vocab_size: int,
        min_freq: int = 2,
        specials: list[str] | None = None,
        verbose: bool = True,
        debug: bool = True,
        debug_sample_size: int = 1000,
    ) -> BPEVocabulary:
        """Build a BPE vocabulary from word-level token sequences.

        This method trains a BPE tokenizer on word-level tokens and creates a vocabulary
        with subword units learned through byte-pair encoding using HuggingFace tokenizers.

        Args:
            sequences_list: List of sequences, where each sequence is a list of word-level
                tokens. For example:
                    [['4c', '8d', '=1'], ['2e', '4f'], ['4c', '4e', '4g']]

                IMPORTANT: Input must be PRE-TOKENIZED at the word level, NOT raw strings.
                Each token will be split into characters for BPE learning by the HF tokenizer.

            save_path: Directory path where vocabulary files will be saved
            name: Name prefix for saved files (e.g., 'synthetic-bpe8k')
            vocab_size: Target vocabulary size (approximate, due to merge constraints)
            min_freq: Minimum frequency for a token/pair to be considered for merging
            specials: List of special tokens (e.g., ['<pad>', '<bos>', '<eos>'])
            verbose: If True, print detailed training progress
            debug: If True, collect comprehensive debug statistics
            debug_sample_size: Number of sequences to sample for char/pair diagnostics

        Returns:
            BPEVocabulary instance with trained tokenizer and word/index mappings

        Example:
            >>> sequences = dataset.get_gt()  # Returns list[list[str]]
            >>> vocab = BPEVocabulary.build(
            ...     sequences_list=sequences,
            ...     save_path='vocab/',
            ...     name='my-bpe',
            ...     vocab_size=8000,
            ... )
            >>> vocab.tokenizer.encode_tokens(['4c', '8d'])  # ['4c', '8d'] or subwords
        """

        print("\n[BPEVocabulary.build] Input data analysis:")
        print(f"  sequences_list type: {type(sequences_list).__name__}")
        print(f"  sequences_list length: {len(sequences_list)}")
        if sequences_list:
            print(f"  First item type: {type(sequences_list[0]).__name__}")
            print(
                f"  First item length: {len(sequences_list[0]) if hasattr(sequences_list[0], '__len__') else 'N/A'}"
            )
            if hasattr(sequences_list[0], "__iter__") and not isinstance(sequences_list[0], str):
                first_seq = list(sequences_list[0])[:5]
                print(f"  First sequence preview: {first_seq}")
            else:
                print(f"  First item: {sequences_list[0]}")

        # Create and train tokenizer
        tok = BPETokenizer(vocab_size=vocab_size, min_freq=min_freq, specials=specials)

        print("\n[BPEVocabulary.build] Training with HuggingFace tokenizers")
        print("  This will flatten sequences_list and train BPE")

        # Prepare debug output directory
        p = Path(save_path)
        p.mkdir(parents=True, exist_ok=True)
        debug_dir = p / "debug_artifacts"

        # Train with debug instrumentation
        tok.train(
            sequences_list,
            verbose=verbose,
            debug=debug,
            debug_output_dir=debug_dir,
            debug_sample_size=debug_sample_size,
        )

        # Save tokenizer
        tok.save(save_path, name)

        # Also save legacy numpy files for backward compatibility
        np.save(str(p / f"{name}w2i.npy"), tok.w2i)
        np.save(str(p / f"{name}i2w.npy"), tok.i2w)

        return cls(tok)

    @classmethod
    def from_files(cls, path: str, name: str) -> BPEVocabulary:
        """Load a BPE vocabulary from saved files.

        Args:
            path: Directory path containing saved files
            name: Name prefix of saved files

        Returns:
            Loaded BPEVocabulary instance
        """
        p = Path(path)

        # Load tokenizer
        tok = BPETokenizer.load(path, name)

        # Try to load numpy files if they exist (backward compatibility)
        w2i_path = p / f"{name}w2i.npy"
        i2w_path = p / f"{name}i2w.npy"

        if w2i_path.exists() and i2w_path.exists():
            w2i = np.load(str(w2i_path), allow_pickle=True).item()
            i2w = np.load(str(i2w_path), allow_pickle=True).item()
            tok._w2i = w2i
            tok._i2w = i2w

        return cls(tok)
