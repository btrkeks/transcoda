# Parser interface

The release depends on `vendor/kern_parser` by local path. The generation helpers
are outside that directory, in `crates/generation_helpers`. Replacing the vendor
directory preserves that boundary. The public subset is commit
`38563829dbcbd33e9df8aaca6286ad359406123a`, derived from
`4234a4f3d62948550bc2cfeb28d95f9dd231b1b0`. Retain its LICENSE, NOTICE,
THIRD_PARTY_NOTICES.md and README, plus the workspace-root verovioxide patch.
Update the lockfile, rebuild and rerun all
parity checks after replacement. Scoring inputs must not change.

## CLI commands used by the release

- `normalize INPUT OUTPUT --output-dialect humdrum-kern --workers N`: source normalization.
- `quality-filter-dir INPUT OUTPUT --workers N`: synthetic quality filtering.
- `concatenate-dir INPUT OUTPUT --manifest-jsonl PATH [--target-count N]`: source concatenation.
- `render-settings-search INPUT --output-jsonl PATH --output-svg-dir DIR --svg-overlap-gate reject --workers N`: rendering and injected-mark search.
- `generate-layout-targets INPUT SETTINGS TARGETS REPORT --skip-render-failures --skip-target-failures`: image/label projection.
- `serialize-jsonl --preset humdrum-kern --input-dialect transcoda-v2 --workers 1 --text-field TEXT --id-field ID`: frozen reference and prediction conversion. Reads and writes JSONL on stdin/stdout. Includes parser validity checks.
- `serialize-jsonl --preset transcoda-v2-training --input-dialect humdrum-kern`: continuation integration-test target conversion, with the source text in `transcription`.
- `serialize-dir INPUT OUTPUT --preset PRESET`: directory serialization.
- `__render-settings-search-worker`: hidden child process of render-settings-search.

`scripts/smoke.py` also uses `normalize`; evaluation invokes `serialize-jsonl`
through `pipelines/baselines/evaluation_inputs.py`. No Python binding is used.
JSONL rows must include `row_index` by default, or the field named by `--id-field`.
The command options in the staged scripts are authoritative.

## Library APIs used by generation_helpers

`continuation_prepare`:

- `KernDocument::parse`, `KernDocument::lines`, `KernLine::Data`.
- `document::Framing::Excerpt`, `score::Score::resolve`, `Score::topology`, `Score::state_at`.
- Topology `layers_at`, `track_of`, `track`, and track `staff`.
- State `track.meter`, `is_declared`, `display`, `meter::MeterDisplay::Hidden`.
- `normalization::multi_measure_rests::CompactMultiMeasureRests::new` and `normalization::NormalizationPass::normalize`.
- `transforms::split::{hide_opening_meter_except_staffs, musical_bar_count, split_trailing_bars_with_prefix_policy, SlicePrefixPolicy::{StandaloneExcerpt, PageRendering}}`.
- Split result `retained` and `removed`.
- `serialize::SerializePreset::HumdrumKern.try_serialize_document`.
- `dataset_manifest::{update_dataset_manifest, DatasetComponentUpdate}` with fields `name`, `producer`, `paths`, `summary`.

`meter_concat`:

- `KernDocument::{parse, clone}`.
- `normalization::{multi_measure_rests::CompactMultiMeasureRests::new, NormalizationPass::normalize}`.
- `transforms::{musical_bar_count, split_trailing_bars}`, split result `retained` and `removed`.
- `concatenation::concatenate_documents`.
- `serialize::SerializePreset::HumdrumKern.try_serialize_document`.

Both helper executables accept a JSON request on stdin. `continuation_prepare`
accepts `page` and `manifest` actions; the Python continuation module owns the
request contract. `meter_concat` accepts `parts` and `output`, including optional
part bar ranges. No other crate in this release imports the parser library.
