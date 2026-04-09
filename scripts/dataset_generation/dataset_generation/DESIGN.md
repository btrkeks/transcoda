# Dataset Generation Rewrite Design

## Purpose

This document defines the target design for a rewrite of the synthetic dataset generation pipeline in `scripts/dataset_generation/dataset_generation_new/`.

The rewrite should optimize for:

- correctness of generated `(image, transcription)` pairs
- clear, typed pipeline stages
- first-class support for concatenating multiple normalized `.krn` files into one sample
- a single hard-coded dataset recipe in code
- easy future tuning in code without a large public config surface

The rewrite does not need to preserve the current implementation shape or current output schema.

## Product Decisions

These decisions are fixed unless explicitly revisited.

- There is exactly one dataset recipe.
- Public config knobs should be removed from the generation core.
- Composition only needs to support concatenating whole normalized `.krn` files.
- Combined labels should use literal concatenation.
- To keep the result valid `.krn`, headers must be removed from every file after the first before concatenation.
- System-count detection should be based on SVG structure, not inferred later from raster output.
- Samples with more than 6 systems should always be truncated.
- Samples with 5 or 6 systems should prefer truncation.
- Samples with 4 or fewer systems should never be truncated.
- V1 should prioritize generation quality and correctness over orchestration features such as resume/progress/quarantine.

## Design Principles

### 1. Plan samples explicitly

The current pipeline effectively schedules `file_path + variant_idx` and then derives everything else implicitly inside the worker. The rewrite should instead create an explicit `SamplePlan` before rendering.

That plan is the core unit of the system.

### 2. Separate data planning from execution

The rewrite should keep these concerns separate:

- source indexing and source selection
- sample composition
- render transcription construction
- rendering and SVG diagnostics
- truncation policy
- offline image augmentation
- dataset record serialization

### 3. Use policy code, not generic config

The new system should still be tunable, but tuning should happen by editing clearly named code modules, not by passing many booleans and probabilities through constructors and CLIs.

### 4. Make sample quality decisions from real signals

The pipeline should prefer decisions based on:

- number of input snippets
- total measure count
- SVG-derived system count
- rendered layout diagnostics

and not only raw token length.

## Target Module Layout

Suggested layout:

```text
scripts/dataset_generation/dataset_generation_new/
  __init__.py
  main.py
  recipe.py
  types.py
  source_index.py
  composer.py
  render_transcription.py
  renderer.py
  truncation.py
  augmentation.py
  acceptance.py
  records.py
  executor.py
  io.py
  README.md                  # optional later
```

Suggested responsibilities:

- `recipe.py`
  - Defines the single hard-coded dataset recipe.
  - Owns probabilities, composition limits, augmentation strength bands, and acceptance policy constants.

- `types.py`
  - Defines the immutable dataclasses used across the pipeline.

- `source_index.py`
  - Scans normalized `.krn` inputs.
  - Computes cached source facts such as measure count and non-empty line count.

- `composer.py`
  - Selects one or more whole `.krn` files.
  - Concatenates them into one valid transcription.
  - Produces a `SamplePlan`.

- `render_transcription.py`
  - Builds the render-only transcription from the label transcription.
  - Applies render-only augmentations such as pedals, dynamics, tempo, and similar notation injections.

- `renderer.py`
  - Renders notation from transcription.
  - Extracts SVG-derived diagnostics before raster postprocessing.
  - Returns both rendered assets and layout diagnostics.

- `truncation.py`
  - Implements truncation candidate generation and truncation selection.
  - Owns the policy for when truncation is required, preferred, or forbidden.

- `augmentation.py`
  - Applies offline image augmentation to already accepted rendered samples.
  - Uses layout-aware augmentation strength, not a one-size-fits-all transform policy.

- `acceptance.py`
  - Encodes the acceptance rules for render success, layout quality, and truncation eligibility.

- `records.py`
  - Converts accepted samples into the final dataset row format.

- `executor.py`
  - Runs the pipeline over many samples.
  - Can start simple in V1 and later grow multiprocessing, resume, and progress features.

- `io.py`
  - Writes images and metadata rows to disk.

## Core Types

These are suggested models, not mandatory exact names.

```python
@dataclass(frozen=True)
class SourceEntry:
    path: Path
    source_id: str
    measure_count: int
    non_empty_line_count: int
    has_header: bool


@dataclass(frozen=True)
class SourceSegment:
    source_id: str
    path: Path
    order: int


@dataclass(frozen=True)
class SamplePlan:
    sample_id: str
    seed: int
    segments: tuple[SourceSegment, ...]
    label_transcription: str
    source_measure_count: int
    source_non_empty_line_count: int
    segment_count: int


@dataclass(frozen=True)
class SvgLayoutDiagnostics:
    system_count: int
    page_count: int
    system_bbox_stats: dict[str, float] | None = None


@dataclass(frozen=True)
class RenderResult:
    image: np.ndarray
    render_layers: RenderedPage | None
    svg_diagnostics: SvgLayoutDiagnostics
    bottom_whitespace_ratio: float | None
    vertical_fill_ratio: float | None


@dataclass(frozen=True)
class AcceptedSample:
    sample_id: str
    label_transcription: str
    image_bytes: bytes
    segment_count: int
    source_ids: tuple[str, ...]
    source_measure_count: int
    system_count: int
    truncation_applied: bool
    truncation_reason: str | None
    truncation_ratio: float | None
    bottom_whitespace_ratio: float | None
    vertical_fill_ratio: float | None
```

Important point:

- `SamplePlan` is the planned sample before render.
- `RenderResult` is the render attempt and its diagnostics.
- `AcceptedSample` is the final sample that can be serialized.

This separation is important and should be preserved.

## End-to-End Pipeline

The preferred top-level flow is:

1. Build a `SourceIndex` from normalized `.krn` files.
2. Repeatedly ask the recipe to create a `SamplePlan`.
3. Build a render-only transcription from the plan's label transcription.
4. Render to SVG and raster outputs.
5. Extract SVG layout diagnostics, especially `system_count`.
6. Decide whether truncation is forbidden, preferred, or required.
7. If truncation is needed or preferred, search truncation candidates and render the best accepted prefix.
8. Once a render is accepted, apply layout-aware offline augmentation.
9. Serialize the final clean dataset record.

Pseudo-flow:

```python
source_index = build_source_index(input_dir)
recipe = ProductionRecipe()

for sample_idx in range(target_samples):
    plan = compose_sample(source_index, recipe, sample_idx)
    render_text = build_render_transcription(plan, recipe)
    render = render_sample(render_text, recipe)
    accepted = accept_or_truncate(plan, render, recipe)
    augmented = augment_accepted_sample(accepted, recipe)
    record = build_dataset_record(augmented)
    write_record(record)
```

## Composition Design

### Scope

Composition only needs to combine whole normalized `.krn` files.

There is no need in V1 to support:

- arbitrary measure slices from a file
- chunk-level search over arbitrary spans
- cross-file restructuring beyond literal concatenation

### Composition rules

When combining multiple `.krn` files:

- keep the first file as the anchor
- remove Humdrum exclusive interpretation headers from every subsequent file before concatenation
- preserve valid `.krn` structure
- do not insert synthetic separators unless required for `.krn` correctness

The default composition output should be the literal transcription that the model is meant to predict.

### Recommendation

Implement composition as a dedicated pure function:

```python
def compose_label_transcription(entries: Sequence[SourceEntry]) -> str:
    ...
```

This should be easy to unit test independently from rendering.

## Rendering Design

### Renderer contract

The renderer should return:

- raster image
- optional separated render layers
- SVG-derived system count
- page count
- frame/layout diagnostics used by acceptance and augmentation

The renderer should not decide truncation policy. It should only report facts.

### System count source of truth

The source of truth for `system_count` should be the SVG structure, for example by counting the relevant system classes or equivalent system-level SVG grouping emitted by the notation renderer.

This is a critical design decision:

- truncation policy should use SVG-derived system count
- output metadata should store that SVG-derived system count
- offline augmentation should not redefine layout facts

## Truncation Policy

This policy is fixed for the rewrite.

### Policy

- `system_count <= 4`
  - truncation is forbidden
  - accept or reject based on normal render-quality criteria only

- `system_count in {5, 6}`
  - truncation is preferred
  - the pipeline should attempt truncation and compare candidates
  - it may still keep the untruncated version if the recipe later decides that some 5/6-system samples are desirable, but the current default should bias toward truncation

- `system_count > 6`
  - truncation is required
  - the full render should never be emitted as a final sample

### Candidate search

The rewrite should preserve the useful idea from the current implementation: truncation should search longest valid prefixes first and should prefer keeping as much content as possible while satisfying layout policy.

Suggested approach:

- build prefix candidates aligned to valid file boundaries or valid line/chunk boundaries
- render candidates longest-first, optionally with bracketed search if needed
- accept the longest candidate whose SVG-derived `system_count` satisfies policy

### Important nuance

For composed samples, truncation must preserve the validity of `.krn`.

That means truncation logic should operate on a representation that understands legal cut points, not just raw string slicing.

## Acceptance Policy

The acceptance stage should return one of:

- `accept_without_truncation`
- `accept_with_truncation`
- `reject`

It should make that decision from:

- render success/failure
- SVG page count
- SVG system count
- frame fit diagnostics
- optional fill/whitespace diagnostics

Suggested rules:

- reject clear renderer failures
- reject multi-page renders unless truncation rescue succeeds
- reject visually degenerate renders
- let truncation policy decide how to respond to system count

Acceptance logic should live in one place and should not be spread across the executor and augmentation code.

## Augmentation Design

### Guiding idea

Offline augmentation should become layout-aware.

Longer or more crowded samples have less geometric slack and should receive less aggressive transforms.

### Inputs to augmentation policy

The augmentation policy should use:

- `segment_count`
- `source_measure_count`
- `system_count`
- `vertical_fill_ratio`
- `bottom_whitespace_ratio`

### Recommended strategy

Instead of passing many independent geometric knobs through the system, define a small number of internal augmentation bands.

For example:

- `roomy`
  - stronger rotation/translation/perspective allowed

- `balanced`
  - default augmentation strength

- `tight`
  - weaker rotation/translation, weaker squeeze, more conservative transforms

Band selection can be based on layout diagnostics, for example:

- if `system_count >= 6`, use `tight`
- if `vertical_fill_ratio` is high, use `tight`
- if `system_count <= 3` and fill is moderate, use `balanced` or `roomy`

The exact thresholds belong in the recipe, not in public config.

### Important rule

Augmentation should never be the stage that silently makes a valid accepted sample invalid.

If an aggressive transform would push notation out of bounds, the stage should:

- retry with a more conservative transform, or
- fall back to the accepted unaugmented render

## Recipe Design

The recipe should be a code object, not a CLI/config object.

Suggested structure:

```python
@dataclass(frozen=True)
class ProductionRecipe:
    composition: CompositionPolicy
    render_only_aug: RenderOnlyAugmentationPolicy
    truncation: TruncationPolicy
    offline_aug: OfflineAugmentationPolicy
    acceptance: AcceptancePolicy
```

The important point is not the exact class names. The important point is that the recipe owns policy and constants in code, and the rest of the pipeline depends on the recipe rather than on dozens of ad hoc parameters.

## Cleaner Dataset Record

The rewrite may define a cleaner record schema.

Suggested fields:

- `sample_id`
- `image`
- `transcription`
- `source_ids`
- `segment_count`
- `source_measure_count`
- `source_non_empty_line_count`
- `svg_system_count`
- `truncation_applied`
- `truncation_reason`
- `truncation_ratio`
- `vertical_fill_ratio`
- `bottom_whitespace_ratio`
- `recipe_version`

This is preferable to tightly coupling record shape to the old generator internals.

## V1 Scope

V1 should include:

- source indexing
- sample planning and composition
- render-only transcription building
- rendering
- SVG-derived system-count extraction
- truncation policy
- layout-aware offline augmentation
- writing a clean dataset record

V1 may omit or simplify:

- multiprocessing
- resume state
- progress heartbeat files
- quarantine logic
- per-sample profiling
- advanced scheduling heuristics

The first goal is a correct, readable, testable core.

## Testing Strategy

At minimum, add tests for:

- whole-file `.krn` concatenation validity
- header removal for second and later files
- deterministic `SamplePlan` creation from a seed
- SVG-derived system-count extraction
- truncation policy behavior for `<=4`, `5/6`, and `>6` systems
- truncation preserving valid `.krn`
- augmentation fallback when transforms would move notation out of bounds
- final dataset record schema

Recommended test split:

- pure unit tests for composition, truncation planning, and record building
- renderer integration tests for system-count extraction and truncation acceptance
- a small end-to-end smoke test over a few normalized `.krn` files

## Migration Notes

The engineer performing the rewrite should not try to preserve current code structure unless it is genuinely useful.

In particular, the rewrite should avoid:

- a giant constructor with many knobs
- a central mutable generator class that owns all concerns
- duplicated config plumbing between orchestrator and worker
- sample semantics encoded implicitly in `file_path + variant_idx`

The rewrite should preserve these useful ideas from the current system:

- explicit render-only augmentation versus label transcription
- truncation as a rescue/quality tool
- storing layout diagnostics in sample metadata
- conservative fallback behavior when augmentation would create out-of-bounds content

## Recommended Implementation Order

1. Define `types.py` and the recipe object.
2. Implement source indexing and whole-file composition.
3. Implement render transcription construction.
4. Implement rendering plus SVG system-count extraction.
5. Implement truncation policy and candidate search.
6. Implement layout-aware offline augmentation.
7. Implement record writing.
8. Add a simple single-process executor.
9. Only after that, consider multiprocessing and operational features.

## Open Questions Deferred on Purpose

These do not need to block the rewrite:

- the exact probabilities used by the single recipe
- the exact thresholds for augmentation bands
- whether some 5/6-system renders should still be kept untruncated in special cases
- whether future composition should move from whole-file concatenation to span-based composition

Those are tuning questions. The architecture should make them easy to adjust without reopening the core design.
