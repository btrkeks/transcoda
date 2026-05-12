# Benchmark Reproduction

This directory contains the benchmark entry points used to evaluate Transcoda. The final
paper table reports CER and OMR-NED as percentages; lower is better.

## Released Artifacts

| Artifact | Hugging Face |
| --- | --- |
| Verovio synthetic benchmark | [btrkeks/verovio-synth-omr](https://huggingface.co/datasets/btrkeks/verovio-synth-omr) |
| Polish historical-scan benchmark | [btrkeks/polish-scores](https://huggingface.co/datasets/btrkeks/polish-scores) |
| Transcoda 59M zero-shot weights | [btrkeks/transcoda-59M-zeroshot-v1](https://huggingface.co/btrkeks/transcoda-59M-zeroshot-v1) |

## Final Table

### Synthetic

Dataset: `data/datasets/validation/synth`

| Configuration          | CER ↓ | OMR-NED ↓ |
| ---------------------- | ----: | --------: |
| Transcoda (greedy)     |  4.38 |     18.71 |
| w/o target norm.       | 27.00 |     82.51 |
| + constrained decoding |  4.62 |     18.74 |
| + beam search          | **2.72** | **18.46** |

### Real Scans (Polish)

Dataset: `data/datasets/validation/polish`

All Polish samples are used for evaluation only. No Polish samples are used for training or
model selection.

| Configuration            | CER ↓ | OMR-NED ↓ |
| ------------------------ | ----: | --------: |
| Transcoda (greedy)       | 33.18 |     65.76 |
| w/o asym. augmentation   | 60.56 |     78.99 |
| w/o visual degradation   | 41.81 |     76.95 |
| w/o score concatenation  | 46.52 |     80.10 |
| FCMAE + CNX-V2-Base      | 27.44 | **61.11** |
| + constrained decoding   | 31.73 |     63.91 |
| + beam search            | **27.00** |  63.97 |

The final benchmark runs had zero conversion failures and zero metric-worker failures.

## Bundled Checkpoints

The Transcoda greedy baseline and inference ablations use:

```text
weights/transcoda-59M-zeroshot-v1.ckpt
sha256: 65512df87eb2b0805a97585e2f5d6fba6cbf6668199c1800e8cd05e69e032542
```

The released copy is available at
[btrkeks/transcoda-59M-zeroshot-v1](https://huggingface.co/btrkeks/transcoda-59M-zeroshot-v1).

The FCMAE ablation uses:

```text
weights/transcoda-120M-FCMAE-pretrain-v1.ckpt
sha256: 6e7d50aa0186c15d9264d3b92d734171e78108e7c9386b84faf58ea67caf2cef
```

`transcoda-59M-zeroshot-v1.ckpt` was trained on the `synth_310k` training dataset.

## Common Settings

All Transcoda benchmark rows use:

```text
max_length = 2048
length_penalty = 1.0
repetition_penalty = 1.1
```

The beam-search ablation additionally uses:

```text
num_beams = 3
```

The constrained-decoding rows enable the structural grammar constraints. The beam-search
row disables constraints because the current constrained logits processors are evaluated in
single-beam mode.

## Environment

Use `uv` from the repository root:

```bash
uv sync --group omr-ned --group grammar
source .venv/bin/activate
```

OMR-NED uses `musicdiff`, `music21`, and `converter21` from the `omr-ned` dependency
group. Grammar-constrained decoding uses `xgrammar` from the `grammar` dependency group.

The metric pipeline converts `**kern` output to MusicXML before scoring. Ensure `hum2xml`
is on `PATH`, or pass `--hum2xml-path`.

## Dataset Sanity Check

Before running the benchmark, verify the bundled validation datasets:

```bash
python - <<'PY'
from datasets import load_from_disk

for path in ["data/datasets/validation/synth", "data/datasets/validation/polish"]:
    ds = load_from_disk(path)
    print(path)
    print("  rows:", len(ds))
    print("  fingerprint:", ds._fingerprint)
    print("  columns:", ds.column_names)
PY
```

The synthetic validation dataset currently used for the final table has 6,862 samples and
fingerprint `443b1e2f8b00158e`.

## Commands

### Greedy Baseline

```bash
python scripts/benchmark/run.py \
  --dataset-root data/datasets/validation \
  --datasets synth,polish \
  --models ours \
  --metrics cer,omr_ned \
  --ours-checkpoint weights/transcoda-59M-zeroshot-v1.ckpt \
  --ours-strategy greedy \
  --ours-repetition-penalty 1.1 \
  --disable-constraints \
  --hum2xml-path hum2xml \
  --output-root outputs/benchmark/transcoda-greedy
```

### Constrained Decoding

```bash
python scripts/benchmark/run.py \
  --dataset-root data/datasets/validation \
  --datasets synth,polish \
  --models ours \
  --metrics cer,omr_ned \
  --ours-checkpoint weights/transcoda-59M-zeroshot-v1.ckpt \
  --ours-strategy greedy \
  --ours-repetition-penalty 1.1 \
  --hum2xml-path hum2xml \
  --output-root outputs/benchmark/transcoda-constrained
```

### Beam Search

```bash
python scripts/benchmark/run.py \
  --dataset-root data/datasets/validation \
  --datasets synth,polish \
  --models ours \
  --metrics cer,omr_ned \
  --ours-checkpoint weights/transcoda-59M-zeroshot-v1.ckpt \
  --ours-strategy beam \
  --ours-num-beams 3 \
  --ours-repetition-penalty 1.1 \
  --disable-constraints \
  --hum2xml-path hum2xml \
  --output-root outputs/benchmark/transcoda-beam
```

### FCMAE + ConvNeXt-V2-Base

```bash
python scripts/benchmark/run.py \
  --dataset-root data/datasets/validation \
  --datasets polish \
  --models ours \
  --metrics cer,omr_ned \
  --ours-checkpoint weights/transcoda-120M-FCMAE-pretrain-v1.ckpt \
  --ours-strategy greedy \
  --ours-repetition-penalty 1.1 \
  --disable-constraints \
  --hum2xml-path hum2xml \
  --output-root outputs/benchmark/transcoda-fcmae
```

## Output Layout

Each invocation writes:

```text
<output-root>/
├── run_manifest.json
├── summary.json
├── summary.csv
└── <dataset>/<model>/
    ├── raw_predictions.jsonl
    ├── per_sample_metrics.csv
    └── summary.json
```

Use `summary.json` or `summary.csv` for aggregate values. Per-sample files include any
conversion or metric-worker failures; final reported runs had none.

## TEDn Utility

`compute_tedn.py` scores a single predicted MusicXML file against a ground-truth MusicXML
file:

```bash
python scripts/benchmark/compute_tedn.py \
  --prediction path/to/prediction.xml \
  --ground-truth path/to/ground_truth.xml
```

Pass `--json` for machine-readable output.
