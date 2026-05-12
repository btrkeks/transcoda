Fixture corpus for `restore_terminal_spine_count_before_final_barline()`.

Layout:
- One directory per case
- `input.krn`: original snippet before terminal-width restoration
- `expected.krn`: desired output after inserting terminal spine merges before the final barline

These case directory names currently preserve the source fixture IDs from the working set
they were curated from. If we later understand the musical patterns well enough, we can
rename them to semantic labels without changing the test loader contract.
