# Rhythm Checker

A CLI tool for validating rhythm correctness in Humdrum **kern files.

## Overview

This tool parses **kern files and validates that each measure contains the correct total duration based on the time signature. It uses exact rational arithmetic (via `num-rational`) to avoid floating-point errors.

## Installation

```bash
cd binaries/rhythm_checker
cargo build --release
```

The binary will be at `target/release/rhythm_checker`.

## Usage

```bash
# Validate a single file
rhythm_checker song.krn

# Validate all .krn files in a directory (recursive)
rhythm_checker data/interim/pdmx/train/

# JSON output for CI integration
rhythm_checker --format json files/ > results.json

# Parallel processing with specific thread count
rhythm_checker -j 8 /large/corpus/

# Quiet mode (only show errors)
rhythm_checker -q files/
```

## Options

| Flag | Description |
|------|-------------|
| `-f, --format <FORMAT>` | Output format: `human` (default) or `json` |
| `-j, --jobs <N>` | Number of parallel threads (0 = auto-detect) |
| `-q, --quiet` | Only show errors, suppress progress |
| `-v, --verbose` | Show per-measure validation details |
| `--allow-anacrusis` | Permit incomplete first measure (default: true) |
| `--allow-incomplete-final` | Permit incomplete final measure (default: true) |
| `--allow-repeat-pairing` | Permit pickup/repeat short-measure pairing (default: true) |
| `--fail-fast` | Stop after the first file that has an error |

## Output

### Human Format

```
song.krn:42:1: error: measure 5 duration mismatch
  expected: 1 (4/4)
  actual:   7/8
  diff:     -1/8

OK 100 files, 1250 measures
```

### JSON Format

```json
{
  "files_processed": 1,
  "files_with_errors": 1,
  "total_errors": 1,
  "total_measures": 50,
  "file_results": [
    {
      "file": "song.krn",
      "measures_checked": 50,
      "errors": [
        {
          "file": "song.krn",
          "line": 42,
          "spine": 0,
          "measure": 5,
          "expected": "1",
          "actual": "7/8",
          "time_signature": { "beats": 4, "beat_unit": 4 },
          "is_first_measure": false,
          "is_final_measure": false
        }
      ]
    }
  ]
}
```

## What It Validates

- **Duration sums**: Each measure's total duration matches the time signature
- **Time signature changes**: Tracks `*M4/4`, `*M6/8`, etc.
- **Spine operations**: Handles splits (`*^`), merges (`*v`), and exchanges (`*x`)
- **Multiple spines**: Validates each **kern spine independently

## What It Handles

| Feature | Handling |
|---------|----------|
| Dotted notes | `4.c` = 3/8, `4..c` = 7/16 |
| Triplets | `12c` = 1/12 (triplet eighth) |
| Grace notes | Skipped (zero duration) |
| Chords | `4c 4e 4g` counted once |
| Rests | `4r` counted same as notes |
| Non-kern spines | Ignored (`**dynam`, `**text`, etc.) |
| Anacrusis | Incomplete first measure allowed by default |

## Exit Codes

- `0`: All files validated successfully
- `1`: One or more files have rhythm errors

## Running Tests

```bash
cargo test
```
