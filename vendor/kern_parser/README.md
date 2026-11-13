# kern_parser for Transcoda

This crate is the subset of an internal Humdrum **kern parser that the
Transcoda paper release needs. It parses scores, prepares synthetic data,
and converts model predictions for scoring.

## Build

Install Rust and a C++ toolchain. The engraving dependency builds Verovio.

```sh
cargo build --release --locked -j 4
cargo test --locked -j 4 -- --test-threads=4
```

Run `target/release/kern_parser <command> --help` for options.

## Commands

- `normalize`: normalize converted **kern source files for data generation.
- `quality-filter-dir`: copy files that pass the data quality checks.
- `concatenate-dir`: build page snippets and record source provenance.
- `render-settings-search`: choose engraving settings and write SVG pages.
- `generate-layout-targets`: derive V2 training targets from rendered pages.
- `serialize-jsonl`: convert transcription fields in JSONL records.
- `serialize-dir`: convert a directory of scores with an explicit preset.

Render search starts its own hidden worker command. The library also exposes
score resolution, bar splitting, normalization, concatenation, and manifest
updates for the release's `continuation_prepare` and `meter_concat` helpers.

## License

This code uses the PolyForm Noncommercial License 1.0.0. See LICENSE.
Commercial use needs a separate license from the author, Daniel Dratschuk.

Required Notice: Copyright 2026 Daniel Dratschuk

The dependencies keep their own licenses. The engraving wrapper verovioxide
uses Apache-2.0. It is pinned to a Git revision here and in the Transcoda
workspace. Verovio uses LGPL-3.0. See THIRD_PARTY_NOTICES.md.
