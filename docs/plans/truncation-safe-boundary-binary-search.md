## Truncation Refactor Plan: Safe-Boundary Binary Search

### Summary

- Refactor truncation search in `scripts/dataset_generation/dataset_generation/` from linear prefix rendering to bounded binary search over safe, chunk-aligned prefix candidates.
- Preserve current truncation policy and output semantics:
  - `> 7` systems or multi-page remains `required`
  - `5..7` systems remains `preferred`
  - `<= 4` systems on one page remains the only acceptable post-truncation outcome
- Keep the search space chunk-based so existing `chunk_count`, `total_chunks`, and `truncation_ratio` semantics stay valid.
- Treat the known spine-transition crash as required scope: malformed terminal spine state must be rejected before Verovio can abort the worker process.
- Keep dataset row schema and current truncation metadata fields unchanged.

### Implementation Changes

- In `truncation.py`:
  - keep `PrefixTruncationCandidate` and `PrefixTruncationSpace`
  - add `build_canonical_prefix_candidates(...)` to build unique candidates ordered from shortest to longest while preserving the highest original `chunk_count` for duplicate final transcriptions
  - add `TruncationProbeResult` and `TruncationSearchResult`
  - add `find_best_truncation_candidate(...)` that performs bounded binary search plus a local refinement scan and returns the longest accepted candidate
  - add `validate_truncation_candidate_terminal_state(...)` to reject malformed terminal spine state before rendering
- In `worker.py`:
  - replace the linear truncation loop with one call to `find_best_truncation_candidate(...)`
  - keep the surrounding flow unchanged: full render first, preferred-band rescue unchanged, truncation only when policy allows it
  - probe each candidate using the current production render path, including truncation-candidate layout rescue when applicable
  - keep failure and Verovio diagnostic ledgers compatible with the existing artifact schema
- In `src/core/kern_postprocess.py`:
  - expose `resolve_terminal_active_spine_count(...)`
  - use active spine count when synthesizing a missing terminator
  - preserve the current fallback behavior only when active spine count cannot be resolved
- In `attempts.py`:
  - add a helper that finalizes an attempt from a precomputed `RenderResult` so guarded structural rejections flow through the same ledger path as real renders

### Interfaces / Types

- Keep these existing interfaces unchanged in meaning:
  - `classify_truncation_mode(...)`
  - `decide_acceptance(...)`
  - `PrefixTruncationCandidate`
  - `truncate_by_chunk_count(...)`
- Add internal-only helpers:
  - `build_canonical_prefix_candidates(kern_text: str) -> list[PrefixTruncationCandidate]`
  - `find_best_truncation_candidate(...) -> TruncationSearchResult`
  - `resolve_terminal_active_spine_count(text: str) -> int | None`
  - `validate_truncation_candidate_terminal_state(text: str) -> str | None`

### Test Plan

- Add unit tests for:
  - canonical candidate construction and deduplication
  - bounded search behavior on monotone and slightly non-monotone acceptance profiles
  - probe-budget exhaustion
  - terminal spine count inference and terminator synthesis after `*^` and contiguous `*v`
  - invalid terminal terminator width rejection
- Add worker-level tests for:
  - successful truncation through the new search path
  - truncation exhaustion preserving existing failure semantics
  - truncation-candidate and truncation-candidate-layout-rescue stage attribution
  - invalid truncation candidates being rejected before any render call
- Add a crash regression that loads a captured crash artifact from `data/datasets/generated/_runs/.../crash_samples/`, validates that it is structurally invalid, strips and rebuilds its terminator, and verifies in a subprocess that the repaired text no longer aborts the interpreter when rendered

### Assumptions / Defaults

- This is an internal refactor plus robustness fix, not a policy redesign.
- Current rescue behavior, dataset schema, and run-artifact field names remain unchanged.
- The search space remains chunk-based; raw line/barline search is intentionally out of scope.
- The search objective remains implicit in the current acceptance policy: choose the longest prefix that becomes acceptable after truncation.
- Use a local refinement radius of `2` candidates to absorb mild layout non-monotonicity without falling back to exhaustive rendering.
