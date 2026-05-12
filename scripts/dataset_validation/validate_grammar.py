#!/usr/bin/env python3
"""Validate **kern files against the GBNF grammar using xgrammar.

Finds files that violate the grammar and reports errors.
"""

import argparse
import sys
from pathlib import Path

import xgrammar as xgr
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast


def load_tokenizer(vocab_path: Path) -> PreTrainedTokenizerFast:
    """Load the tokenizer from the vocab directory."""
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(vocab_path / "tokenizer.json"))
    return tokenizer


def compile_grammar(grammar_path: Path, tokenizer: PreTrainedTokenizerFast) -> xgr.CompiledGrammar:
    """Compile the GBNF grammar with xgrammar."""
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=tokenizer.vocab_size
    )
    compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)
    grammar_text = grammar_path.read_text()
    return compiler.compile_grammar(grammar_text)


def validate_transcription(
    compiled_grammar: xgr.CompiledGrammar,
    tokenizer: PreTrainedTokenizerFast,
    transcription: str,
    stop_token_ids: list[int],
) -> tuple[bool, str | None]:
    """Validate a single transcription against the grammar.

    Returns:
        (True, None) if valid
        (False, error_message) if invalid
    """
    matcher = xgr.GrammarMatcher(compiled_grammar)

    # Tokenize the transcription
    token_ids = tokenizer.encode(transcription, add_special_tokens=False)

    # Feed each token to the matcher
    for i, token_id in enumerate(token_ids):
        if not matcher.accept_token(token_id):
            # Find context around the failure
            accepted_tokens = token_ids[:i]
            rejected_token = token_ids[i]
            remaining_tokens = token_ids[i + 1 :]

            # Decode for human-readable output
            accepted_text = tokenizer.decode(accepted_tokens) if accepted_tokens else ""
            rejected_text = tokenizer.decode([rejected_token])
            remaining_text = tokenizer.decode(remaining_tokens) if remaining_tokens else ""

            # Find line and column of failure in original text
            line_num = accepted_text.count("\n") + 1
            last_newline = accepted_text.rfind("\n")
            col_num = len(accepted_text) - last_newline if last_newline >= 0 else len(accepted_text) + 1

            error_msg = (
                f"Grammar violation at token {i} (line {line_num}, col ~{col_num}):\n"
                f"  Rejected token: {repr(rejected_text)} (id={rejected_token})\n"
                f"  Accepted so far: {repr(accepted_text[-100:]) if len(accepted_text) > 100 else repr(accepted_text)}\n"
                f"  Next tokens: {repr(remaining_text[:50]) if len(remaining_text) > 50 else repr(remaining_text)}"
            )
            return False, error_msg

    # Try to accept a stop token to signal end-of-input
    for stop_id in stop_token_ids:
        if matcher.accept_token(stop_id):
            break
    else:
        # No stop token was accepted - grammar expects more input
        return False, "Transcription ended but grammar expects more input (no stop token accepted)"

    # Check if we're at a valid terminal state
    if not matcher.is_terminated():
        return False, "Transcription ended but grammar is not at a terminal state"

    return True, None


def main():
    parser = argparse.ArgumentParser(
        description="Find **kern files that violate the grammar"
    )
    parser.add_argument(
        "--directory",
        "-d",
        default="data/interim/pdmx/3_normalized",
        help="Path to the directory containing .krn files (default: data/interim/pdmx/3_normalized)",
    )
    parser.add_argument(
        "--grammar",
        "-g",
        default="grammars/kern.gbnf",
        help="Path to the GBNF grammar file (default: grammars/kern.gbnf)",
    )
    parser.add_argument(
        "--vocab",
        "-V",
        default="vocab/bpe4k",
        help="Path to vocabulary directory (default: vocab/bpe4k)",
    )
    parser.add_argument(
        "--skip",
        "-s",
        type=int,
        default=0,
        help="Number of files to skip from the start (default: 0)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Maximum number of files to check (default: all)",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Find all violations instead of stopping at first",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Write list of invalid file paths to this file (one per line)",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    directory_path = project_root / args.directory
    grammar_path = project_root / args.grammar
    vocab_path = project_root / args.vocab

    if not directory_path.exists():
        print(f"Error: Directory not found at {directory_path}", file=sys.stderr)
        sys.exit(1)

    if not grammar_path.exists():
        print(f"Error: Grammar not found at {grammar_path}", file=sys.stderr)
        sys.exit(1)

    if not vocab_path.exists():
        print(f"Error: Vocab not found at {vocab_path}", file=sys.stderr)
        sys.exit(1)

    # Load tokenizer
    print(f"Loading tokenizer from {vocab_path}...", file=sys.stderr)
    tokenizer = load_tokenizer(vocab_path)

    # Load and compile grammar
    print(f"Loading grammar from {grammar_path}...", file=sys.stderr)
    try:
        compiled_grammar = compile_grammar(grammar_path, tokenizer)
    except Exception as e:
        print(f"Error compiling grammar: {e}", file=sys.stderr)
        sys.exit(1)

    # Get stop token IDs from a temporary matcher
    temp_matcher = xgr.GrammarMatcher(compiled_grammar)
    stop_token_ids = list(temp_matcher.stop_token_ids)
    print(f"Stop token IDs: {stop_token_ids}", file=sys.stderr)

    # Find all .krn files
    print(f"Scanning directory {directory_path}...", file=sys.stderr)
    krn_files = sorted(directory_path.glob("*.krn"))
    total = len(krn_files)
    print(f"Found {total} .krn files", file=sys.stderr)

    if args.skip > 0:
        krn_files = krn_files[args.skip:]
        print(f"Skipping first {args.skip} files", file=sys.stderr)

    if args.limit is not None:
        krn_files = krn_files[:args.limit]
        print(f"Limiting to {args.limit} files", file=sys.stderr)

    violations = []

    print(f"Validating {len(krn_files)} files...", file=sys.stderr)

    for krn_file in tqdm(krn_files, desc="Validating", miniters=50, initial=args.skip, total=total):
        try:
            transcription = krn_file.read_text()
        except Exception as e:
            violation = {
                "file": krn_file,
                "error": f"Failed to read file: {e}",
                "transcription": None,
            }
            violations.append(violation)
            if not args.all:
                print(f"\nError reading file: {krn_file}")
                print(f"Error: {e}")
                sys.exit(1)
            continue

        is_valid, error = validate_transcription(compiled_grammar, tokenizer, transcription, stop_token_ids)

        if not is_valid:
            violation = {
                "file": krn_file,
                "error": error,
                "transcription": transcription,
            }

            if args.all:
                violations.append(violation)
                print(f"\n{'=' * 60}", file=sys.stderr)
                print(f"Violation in file: {krn_file.name}", file=sys.stderr)
                print(f"{'=' * 60}", file=sys.stderr)
                print(error, file=sys.stderr)
            else:
                # Print and exit on first violation
                print(f"\nFirst grammar violation found in: {krn_file}")
                print(f"\nError:\n{error}")
                print(f"\nFull transcription:\n{repr(transcription)}")
                sys.exit(1)

    if args.all:
        if violations:
            print(f"\n{'=' * 60}")
            print(f"Summary: Found {len(violations)} violations out of {len(krn_files)} files")
            print(f"Invalid files:")
            for v in violations[:20]:
                print(f"  - {v['file'].name}")
            if len(violations) > 20:
                print(f"  ... and {len(violations) - 20} more")

            if args.output:
                output_path = Path(args.output)
                with output_path.open("w") as f:
                    for v in violations:
                        f.write(f"{v['file']}\n")
                print(f"\nWrote list of invalid files to: {args.output}")

            sys.exit(1)
        else:
            print(f"\nAll {len(krn_files)} files are valid!")
            sys.exit(0)
    else:
        print(f"\nAll {len(krn_files)} files are valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()
