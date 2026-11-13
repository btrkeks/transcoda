# Regenerate synthetic training data

The V2.1 training snapshot contains 343,470 rows, including 11,391 KernScores rows.
It is not distributed or scheduled for hosting. `release/training-data/` contains
its sealed release and shard hashes for provenance only. Fetch the source files
yourself and follow their terms. The public test split has no KernScores rows.

## Acquire and convert sources

Obtain PDMX MusicXML and GrandStaff transcriptions from the cited upstream releases.
Record the downloaded revisions and archive hashes. The extraction entry points are:

```sh
uv run --no-sync python -m pipelines.datasets.generation.extract_raw_data.extract_mxl_member --help
uv run --no-sync python -m pipelines.datasets.generation.extract_raw_data.extract_grandstaff_transcriptions --help
uv run --no-sync python -m pipelines.datasets.generation.extract_raw_data.kernscores --help
```

For PDMX, extract each declared MusicXML member and use the patched musicxml2hum:

```sh
PYTHON="$PWD/.venv/bin/python" bash pipelines/datasets/generation/extract_raw_data/extract_mxl.sh data/raw/pdmx data/pdmx-xml
PATH="$PWD/vendor/musicxml2hum/bin:$PATH" WORKERS=4 bash scripts/dataset_generation/kern_conversions/convert_xml_to_kern.sh data/pdmx-xml data/converted/pdmx
```

For GrandStaff, convert its extended-kern transcriptions with
`scripts/dataset_generation/kern_conversions/convert_ekern_to_kern.sh`.
The KernScores acquisition CLI inventories and downloads files; it grants no new
rights to them. Do not redistribute source encodings or rendered KernScores pages.

## Build pages

```sh
uv run --no-sync python scripts/generate.py --input-kern data/converted/pdmx --source pdmx --work outputs/pdmx-build --output data/generated-pdmx --workers 4
```

The pipeline filters staves and rational durations, cleans source-specific
artifacts, normalizes notation, checks quality, concatenates sources, prepares
continuations, searches render settings, projects labels and applies scan degradation.
The vendored parser includes the V2.1 meter-display and injected-label fixes.
A preflight checks every continuation candidate. The deterministic continuation
selection fraction is 0.25 for training. Carried opening meters are hidden on
continuation pages while source-authored declarations stay visible.
Source-specific labelled-mark page-rate ranges are 0.20 to 0.80 for PDMX,
0.20 to 0.74 for KernScores and 0.20 to 0.28 for GrandStaff.

Use `--source grandstaff` or `--source kernscores-merged` for those corpora.
Use `--no-augmentation` for held-out rendered pages; it also sets the continuation
fraction to zero. Keep source splits separate. Record source IDs, content hashes,
concatenation manifests and all exclusion reports. Filter source/content overlaps
with your validation and test sets before generation. The private real-reference
matching database is not included, so exact historical exclusions cannot be rebuilt
from this repository alone. Do not train on real scans or on public test pages.

A small authored example exercises the full path:

```sh
uv run --no-sync python scripts/generate.py --input-kern examples --source toy --work outputs/example-build --output outputs/example-data --workers 1 --target-count 1
```

For multiple sources, adjust the relative layout-report paths in
`config/final_dataset_sources.json` and run the synthetic materializer:

```sh
uv run --no-sync python -m pipelines.datasets.materialization.main --registry config/final_dataset_sources.json --output_dir data/train-synthetic-v2.1 --num_workers 4 --synthetic_augraphy true
```

Train with `config/camera-ready-v2.1.json` as shown in the README. Supply your own
real validation dataset. The original 134-page validation snapshot remains private.
Exact retraining, data-byte regeneration and checkpoint selection are therefore
not established from public assets alone. The released checkpoint and frozen
predictions support direct evaluation without retraining.
