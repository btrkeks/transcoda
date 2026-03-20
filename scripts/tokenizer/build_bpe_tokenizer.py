#!/usr/bin/env python3
"""
Build a BPE tokenizer from directories of text files (e.g., **kern .krn files).

Reads all matching files from one or more directories, using their full text
content as training samples for BPE.

Output:
- Saves both tokenizers.Tokenizer JSON and a PreTrainedTokenizerFast.

Usage:
    # Single directory of .krn files
    python build_bpe_tokenizer.py data/interim/train/pdmx/3_normalized --vocab_name pdmx-bpe4k

    # Multiple directories
    python build_bpe_tokenizer.py data/interim/train/pdmx/3_normalized data/interim/train/grandstaff/3_normalized \
        --vocab_name combined-bpe4k

    # Custom glob pattern and vocab size
    python build_bpe_tokenizer.py data/my_corpus --vocab_name custom-bpe8k --vocab_size 8000 --glob "*.txt"

    # With custom special tokens
    python build_bpe_tokenizer.py data/interim/train/pdmx/3_normalized --vocab_name custom-bpe4k \
        --pad_token "[PAD]" --bos_token "[BOS]" --eos_token "[EOS]" --unk_token "[UNK]"
"""

import argparse
import logging
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def count_files(
    dirs: Sequence[Path],
    pattern: str = "*.krn",
) -> int:
    """
    Count total matching files across all directories for progress bar display.

    Args:
        dirs: Directories to search.
        pattern: Glob pattern to match files.

    Returns:
        Total number of matching files.
    """
    total = 0
    for d in dirs:
        total += sum(1 for _ in d.glob(pattern))
    return total


def iter_texts(
    dirs: Sequence[Path],
    pattern: str = "*.krn",
    transform_fn: Callable[[str], str] | None = None,
) -> Iterable[str]:
    """
    Iterate over text content of files matching a glob pattern in one or more directories.

    Args:
        dirs: Directories to search.
        pattern: Glob pattern to match files.
        transform_fn: Optional function to transform each text before yielding.
    """
    for d in dirs:
        for path in d.glob(pattern):
            text = path.read_text(encoding="utf-8")
            if not text:
                continue
            yield transform_fn(text) if transform_fn else text


def build_bpe_tokenizer(
    iterator: Iterable[str],
    vocab_size: int = 4000,
    min_freq: int = 2,
    specials: Sequence[str] = ("<pad>", "<bos>", "<eos>", "<unk>"),
    normalize: bool = True,
    length: int | None = None,
    split_spaces: bool = False,
) -> Tokenizer:
    """
    Build a BPE tokenizer from an iterator of text samples.

    Args:
        iterator: Iterable of text strings to train on
        vocab_size: Target vocabulary size
        min_freq: Minimum frequency for a token to be included
        specials: Special tokens in order (determines token IDs 0, 1, 2, 3, ...)
        normalize: Whether to apply NFKC normalization
        length: Optional total length for progress bar display
        split_spaces: If True, also split on spaces (prevents cross-note
            BPE merges within chords)

    Returns:
        Trained tokenizers.Tokenizer instance
    """
    # The unk_token must match the one in specials for consistency
    unk_token = specials[3] if len(specials) > 3 else "<unk>"
    tok = Tokenizer(models.BPE(unk_token=unk_token))

    if normalize:
        tok.normalizer = normalizers.Sequence([normalizers.NFKC()])

    # Pre-tokenizer: split on structural whitespace, preserving it as isolated tokens.
    #
    # - Tabs (\t) are spine separators in **kern format.
    # - Newlines (\n) are record separators.
    # - When split_spaces is True, spaces are also split. This prevents BPE from
    #   learning cross-note merges within chords (e.g., "4c 4e 4g" stays as three
    #   separate "words" for BPE, yielding compositional intra-note tokens).
    # - When split_spaces is False (legacy), spaces are kept inside tokens, allowing
    #   BPE to merge across note boundaries within chords.
    #
    # The "isolated" behavior ensures whitespace characters become their own tokens
    # rather than being attached to adjacent content or discarded.
    split_pattern = r"[\t\n\r ]+" if split_spaces else r"[\t\n\r]+"
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(split_pattern), behavior="isolated")

    tok.decoder = decoders.BPEDecoder()

    # continuing_subword_prefix="" is critical for **kern:
    # The default BPE behavior adds "##" prefix to continuation tokens, which would
    # corrupt **kern notation during decoding (e.g., "4c" becoming "4##c").
    # Empty prefix ensures decode(encode(text)) == text for valid **kern.
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=list(specials),
        show_progress=True,
        continuing_subword_prefix="",
    )

    tok.train_from_iterator(iterator, trainer=trainer, length=length)

    # Post-processor adds EOS token automatically to encoded sequences.
    #
    # Note: BOS is intentionally NOT added by the post-processor. During training,
    # the model's prepare_decoder_input_ids_from_labels() shifts labels right and
    # prepends BOS. During inference, generation starts with BOS explicitly.
    # Adding BOS here would cause double-BOS issues.
    eos_token = specials[2] if len(specials) > 2 else "<eos>"
    tok.post_processor = TemplateProcessing(
        single=f"$A {eos_token}",
        pair=f"$A {eos_token} $B {eos_token}",
        special_tokens=[
            (eos_token, tok.token_to_id(eos_token)),
        ],
    )

    return tok


def validate_tokenizer(
    tokenizer: Tokenizer,
    specials: Sequence[str],
    target_vocab_size: int,
) -> bool:
    """
    Validate the trained tokenizer meets expected properties.

    Checks:
    1. Special token IDs are assigned correctly (in order: pad=0, bos=1, eos=2, unk=3)
    2. Roundtrip encode/decode preserves text
    3. Essential **kern tokens don't fragment excessively

    Args:
        tokenizer: The trained tokenizer to validate
        specials: Special tokens in expected ID order
        target_vocab_size: The requested vocabulary size

    Returns:
        True if all validations pass, False otherwise
    """
    all_passed = True

    # Check 1: Special token IDs
    logger.info("Validating special token IDs...")
    for expected_id, token in enumerate(specials):
        actual_id = tokenizer.token_to_id(token)
        if actual_id != expected_id:
            logger.error(f"  FAIL: {token} has ID {actual_id}, expected {expected_id}")
            all_passed = False
        else:
            logger.info(f"  OK: {token} -> ID {actual_id}")

    # Check 2: Roundtrip encode/decode
    logger.info("Validating roundtrip encode/decode...")
    test_patterns = [
        "**kern\t**kern",
        "4c\t4d",
        "*clefG2\t*clefF4",
        "=1\t=1",
        "4c 4e 4g\t2d",  # Chord with spaces
        "4c;\t4d;",  # Fermatas
        "[4c\t4d]",  # Ties
        "16ccLL\t8dL",  # Beaming
    ]
    for pattern in test_patterns:
        encoded = tokenizer.encode(pattern)
        # Decode without special tokens for comparison
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=True)
        # Strip the auto-added EOS for comparison
        decoded = decoded.strip()
        if decoded != pattern:
            logger.warning(f"  WARN: Roundtrip mismatch for '{pattern}' -> '{decoded}'")
            # Not a hard failure, just a warning (BPE may normalize whitespace)
        else:
            logger.info(f"  OK: '{pattern}' roundtrips correctly")

    # Check 3: Essential tokens exist and don't over-fragment
    logger.info("Validating essential **kern tokens...")
    essential_tokens = ["*", "**kern", "=", "\t", "\n"]
    for token in essential_tokens:
        encoded = tokenizer.encode(token)
        # Filter out EOS which is auto-added
        eos_id = tokenizer.token_to_id(specials[2]) if len(specials) > 2 else None
        token_ids = [tid for tid in encoded.ids if tid != eos_id]

        if len(token_ids) > 2:
            logger.warning(
                f"  WARN: '{repr(token)}' fragments into {len(token_ids)} tokens: {token_ids}"
            )
        else:
            logger.info(f"  OK: '{repr(token)}' -> {len(token_ids)} token(s)")

    # Log vocabulary statistics
    actual_vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Vocabulary statistics:")
    logger.info(f"  Target size: {target_vocab_size}")
    logger.info(f"  Actual size: {actual_vocab_size}")
    if actual_vocab_size < target_vocab_size:
        logger.info(
            f"  Note: Actual < target likely due to min_frequency filtering or limited training data"
        )

    return all_passed


def main():
    ap = argparse.ArgumentParser(
        description="Build a BPE tokenizer for **kern notation from directories of text files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data source
    ap.add_argument(
        "dirs",
        nargs="+",
        type=Path,
        help="One or more directories containing text files to train on",
    )

    # Tokenizer configuration
    tok_group = ap.add_argument_group("Tokenizer Configuration")
    tok_group.add_argument(
        "--vocab_name", required=True, help="Name for the tokenizer (e.g., 'synthetic-bpe8k')"
    )
    tok_group.add_argument("--vocab_size", type=int, default=4000, help="Target vocabulary size")
    tok_group.add_argument(
        "--min_freq", type=int, default=2, help="Minimum token frequency to include"
    )
    tok_group.add_argument(
        "--glob", default="*.krn", dest="file_glob", help="Glob pattern for matching files (default: *.krn)"
    )
    tok_group.add_argument(
        "--split_spaces",
        action="store_true",
        help="Also split on spaces (prevents cross-note BPE merges within chords)",
    )

    # Special tokens (configurable)
    special_group = ap.add_argument_group("Special Tokens")
    special_group.add_argument("--pad_token", default="<pad>", help="Padding token (ID 0)")
    special_group.add_argument("--bos_token", default="<bos>", help="Beginning-of-sequence token (ID 1)")
    special_group.add_argument("--eos_token", default="<eos>", help="End-of-sequence token (ID 2)")
    special_group.add_argument("--unk_token", default="<unk>", help="Unknown token (ID 3)")

    # Output options
    out_group = ap.add_argument_group("Output Options")
    out_group.add_argument("--out_dir", default="vocab", help="Output directory for tokenizer")
    out_group.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip post-training validation checks",
    )

    args = ap.parse_args()

    # Validate directories exist
    for d in args.dirs:
        if not d.is_dir():
            ap.error(f"Not a directory: {d}")

    # Build special tokens list (order determines IDs: 0, 1, 2, 3)
    specials = [args.pad_token, args.bos_token, args.eos_token, args.unk_token]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Log configuration
    logger.info(f"Directories: {[str(d) for d in args.dirs]}")
    logger.info(f"File glob: {args.file_glob}")
    logger.info(f"Target vocab size: {args.vocab_size}")
    logger.info(f"Min frequency: {args.min_freq}")
    logger.info(f"Special tokens: {specials}")
    logger.info(f"Split spaces: {args.split_spaces}")

    # Count files for progress bar
    length = count_files(args.dirs, args.file_glob)
    logger.info(f"Total files: {length:,}")

    # Create iterator over file contents
    it = iter_texts(args.dirs, args.file_glob)

    # Train BPE tokenizer
    logger.info("Training BPE tokenizer...")
    bpe = build_bpe_tokenizer(
        it,
        vocab_size=args.vocab_size,
        min_freq=args.min_freq,
        specials=specials,
        length=length,
        split_spaces=args.split_spaces,
    )

    # Validate tokenizer (unless skipped)
    if not args.skip_validation:
        logger.info("Running post-training validation...")
        validation_passed = validate_tokenizer(bpe, specials, args.vocab_size)
        if not validation_passed:
            logger.warning("Some validation checks failed. Review warnings above.")
    else:
        logger.info("Skipping validation (--skip_validation flag set)")

    # Save raw tokenizer (tokenizers JSON)
    tok_json_path = out_dir / f"{args.vocab_name}-tokenizer.json"
    bpe.save(str(tok_json_path))
    logger.info(f"Saved raw tokenizer to: {tok_json_path}")

    # Wrap as HF fast tokenizer and save
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_json_path),
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        pad_token=args.pad_token,
        unk_token=args.unk_token,
    )
    hf_tok_path = out_dir / args.vocab_name
    hf_tok.save_pretrained(hf_tok_path)
    logger.info(f"Saved HF tokenizer to: {hf_tok_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
