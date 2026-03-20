# Kern Normalization System

A composable pipeline for normalizing Humdrum \*\*kern transcriptions.

## Overview

Uses a **pass pipeline** pattern: small, single-purpose passes composed into an ordered pipeline. Each pass follows a three-phase lifecycle:

1. **prepare** — Parse/analyze the input
2. **transform** — Modify the kern string
3. **validate** — Check the output is valid

## Architecture

### Files

- `base.py` — `NormalizationPass` protocol and `NormalizationContext` (shared dict for cross-pass data)
- `pipeline.py` — `Pipeline` class: composes passes, runs lifecycle, supports JSON config loading/saving
- `presets.py` — Pre-built pipelines (e.g. `normalize_kern_transcription()` used by the dataset generation pipeline)
- `main.py` — CLI entry point for batch-normalizing a directory of kern files
- `passes/` — All normalization passes, registered in `passes/__init__.py` via `PASS_REGISTRY`

### Active Pipeline

The full pipeline is defined in `presets.py`. Current pass order:

1. `RemoveNonKernSpines` — must run before `CleanupKern` (needs `**kern` header)
2. `CleanupKern`
3. `RemoveGraceRests` — must run before `CanonicalizeNoteOrder`
4. `RemoveRedundantStria`
5. `RemoveLeadingBarlines`
6. `RemoveNullLines`
7. `CanonicalizeBarlines`
8. `MergeSplitNormalizer`
9. `StripTerminalTerminator`
10. `CapSlurs`
11. `RemoveRedundantTimeSignatures`
12. `CanonicalizeHeaderOrder`
13. `MergeHeaderClefLines` — after reordering brings clef lines together
14. `NormalizeNullKeysig`
15. `CanonicalizeNoteOrder`
16. `OrderNotes(ascending=True)`
17. `SymbolsBeforeSplit`
18. `RemoveContradictoryAccidentals` — must run after `CanonicalizeNoteOrder`
19. `RemoveConflictingBowings`
20. `RemoveNullTies` — canonicalizes repeated ties (`[[`, `]]`, `__`) and removes null ties/slurs; must run after `CanonicalizeNoteOrder`
21. `ValidateSpineOperations` — final semantic check for spine width consistency and valid `*v` merge grouping

## Implementing a New Pass

1. Create a new file in `passes/`:

```python
from ..base import NormalizationContext


class MyPass:
    name = "my_pass"

    def prepare(self, text: str, ctx: NormalizationContext) -> None:
        pass  # optional: parse/analyze input, store in ctx

    def transform(self, text: str, ctx: NormalizationContext) -> str:
        # Your normalization logic here
        return modified_text

    def validate(self, text: str, ctx: NormalizationContext) -> None:
        pass  # optional: raise ValueError if output is invalid
```

2. Register it in `passes/__init__.py`: add the import and an entry in `PASS_REGISTRY`.

3. Add it to the pipeline in `presets.py` at the appropriate position (mind ordering constraints).
