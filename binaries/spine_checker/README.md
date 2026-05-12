# Spine Checker

A standalone Rust crate for checking whether a Humdrum `**kern` snippet obeys strict canonical spine rules.

## Status

This crate is intentionally scaffold-only right now:

- public API is in place
- CLI entrypoint is in place
- fixture-driven TDD tests are in place
- validation logic is still unimplemented

## Layout

- `src/lib.rs`: public library surface
- `src/checker.rs`: checker API and future implementation entrypoint
- `src/error.rs`: placeholder error and diagnostic types
- `src/main.rs`: thin CLI wrapper
- `tests/snippet_fixtures.rs`: fixture-driven acceptance and rejection tests
- `tests/fixtures/accepted/`: snippets that should be accepted
- `tests/fixtures/rejected/`: snippets that should be rejected

## Test-First Workflow

1. Add a `.krn` snippet under `tests/fixtures/accepted/` when it should pass.
2. Add a `.krn` snippet under `tests/fixtures/rejected/` when it should fail.
3. Run `cargo test`.
4. Implement only enough logic to make the next failing test pass.

Rejected fixtures currently pass only if the checker returns a real canonical-rule violation. The placeholder `NotYetImplemented` error is treated as a failure, so both acceptance and rejection tests drive implementation forward.

## Running

```bash
cd binaries/spine_checker
cargo test
```

To exercise the CLI later:

```bash
cargo run -- path/to/snippet.krn
cat snippet.krn | cargo run -- --stdin
```
