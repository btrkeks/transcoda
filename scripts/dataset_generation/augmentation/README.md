# Kern Augmentation

Kern-side augmentation utilities used by dataset generation.

## Current Pipeline Behavior

The orchestrator is [`pipeline.py`](./pipeline.py). At the moment, it actively applies:

- `apply_xtuplet(...)`
- `apply_invisible_rests(...)`
- `apply_remove_beginning_time_signature(...)`

Other augmentation modules exist in this package and can be enabled by wiring them into `augment_transcription(...)`.

## Config

[`config.py`](./config.py) defines `AugmentationConfig`, where each field is a probability in `[0.0, 1.0]` controlling whether that augmentation runs.

## Adding a New Augmentation

1. Add a new module under `scripts/dataset_generation/augmentation/`, for example `my_augmentation.py`.
2. Add a probability field to `AugmentationConfig` in [`config.py`](./config.py) and include it in validation.
3. Call the new function conditionally from `augment_transcription(...)` in [`pipeline.py`](./pipeline.py).
4. Export it in [`__init__.py`](./__init__.py) if it should be public.
5. Add tests under `tests/synthetic/kern_augmentation/`, for example `test_my_augmentation.py`.

## Shared Utilities

[`kern_utils.py`](./kern_utils.py) includes helpers for token targeting and spine-safe insertion:

- `find_note_tokens(krn)`
- `find_barline_indices(krn)`
- `sample_positions(positions, prob)`
- `append_to_token(token, suffix)`
- `apply_suffix_to_notes(krn, positions, suffix)`
- `insert_interpretation_line(krn, idx, tokens)`
- `get_spine_count(krn)`
