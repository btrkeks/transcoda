# Transcoda

**End-to-end zero-shot Optical Music Recognition via data-centric synthetic training.**

A compact 59M-parameter vision-encoder-decoder that turns raw score images directly into [`**kern`](https://www.humdrum.org/rep/kern/) symbolic transcriptions. Trained from scratch in about 6 hours on a single RTX 5090, on synthetic data only.

<p align="center">
  <img src="figures/architecture.png" alt="Transcoda architecture: ConvNeXt-V2 encoder, projector MLP, 2D sinusoidal positional encoding, and an 8-layer Transformer decoder with optional constrained decoding." width="100%">
</p>

## TL;DR

- **Compact and fast.** 59M parameters, 6h training on a single RTX 5090.
- **Data-centric.** A deterministic pipeline removes structural ambiguity from `**kern` targets so the decoder learns one canonical sequence per image.
- **Best score among compared systems** on a clean Verovio synthetic benchmark (18.46% OMR-NED) and on real-world Polish historical scans (63.97% OMR-NED), beating both compact and much larger baselines.
- **Optional constrained decoding** guarantees formally valid `**kern` output for downstream renderers, with small raw-metric changes in the current ablations.
- Releases standardized Verovio synthetic and Polish historical-scan benchmarks for future OMR evaluation.

## Released Artifacts

| Artifact | Link |
| --- | --- |
| Verovio synthetic benchmark | [btrkeks/verovio-synth-omr](https://huggingface.co/datasets/btrkeks/verovio-synth-omr) |
| Polish historical-scan benchmark | [btrkeks/polish-scores](https://huggingface.co/datasets/btrkeks/polish-scores) |
| Transcoda 59M zero-shot weights | [btrkeks/transcoda-59M-zeroshot-v1](https://huggingface.co/btrkeks/transcoda-59M-zeroshot-v1) |

## Headline results

OMR-NED (lower is better). Polish = 102 historical scanned scores; Verovio = cleanly rendered held-out synthetic scores.

| Model         |  Params | Polish (real) ↓ | Verovio (synthetic) ↓ |
| ------------- | ------: | --------------: | --------------------: |
| SMT++         |   11M |          80.16% |                92.23% |
| Legato        |  943M |          86.73% |                43.91% |
| **Transcoda** | **59M** |      **63.97%** |            **18.46%** |

Transcoda uses unconstrained beam search (width 3) for these headline numbers. Key decoding ablations are noted below.

## Benchmark settings

All headline numbers use the same filtered Polish split, the same Verovio benchmark, and the same metric pipeline. Baselines use public checkpoints without fine-tuning.

The benchmark datasets and evaluated Transcoda checkpoint are released on Hugging Face:

| Artifact | Hugging Face |
| --- | --- |
| Verovio synthetic benchmark | [btrkeks/verovio-synth-omr](https://huggingface.co/datasets/btrkeks/verovio-synth-omr) |
| Polish historical-scan benchmark | [btrkeks/polish-scores](https://huggingface.co/datasets/btrkeks/polish-scores) |
| Transcoda 59M zero-shot weights | [btrkeks/transcoda-59M-zeroshot-v1](https://huggingface.co/btrkeks/transcoda-59M-zeroshot-v1) |

| Model | Checkpoint | Decoding and runtime settings |
| ----- | ---------- | ----------------------------- |
| SMT++ | `PRAIG/smt-fp-grandstaff` | greedy decoding, maximum length 2048 |
| Legato | `guangyangmusic/legato` | beam width 3, repetition penalty 1.1, maximum length 2048 |
| Transcoda | released checkpoint | unconstrained beam search with width 3, maximum length 2048 |

Baselines use the public checkpoints without fine-tuning. Legato predictions are converted from ABC to MusicXML before OMR-NED evaluation. SMT++ predictions are evaluated directly as restored `**kern`.

## Method

**Architecture.** A pretrained ConvNeXt-V2 encodes a full-page score image into a 47×33 grid of 768-d patch features. A projector MLP and a 2D sinusoidal positional encoding lift these into a 1551×512 sequence. An 8-layer Transformer decoder with RoPE self-attention emits `**kern` tokens autoregressively over a 3,000-token vocabulary. End-to-end, no symbol detector, no staff segmenter.

**Target canonicalization is the key insight.** `**kern` lets the same score map to many syntactically different but semantically equivalent sequences. A deterministic pass collapses each score to one canonical form, so the decoder no longer has to model annotator style. The ablation is sharp:

| Configuration                          | OMR-NED ↓ |
| -------------------------------------- | --------: |
| Full Transcoda (canonicalized targets) |    18.71% |
| Same model, raw non-canonical targets  |    82.51% |

This is the main modeling result: target normalization prevents sequence collapse on clean synthetic pages.

**Constrained decoding (optional).** A stateful, layered grammar engine enforces local `**kern` syntax and global line-width consistency during inference. It guarantees formally valid output for downstream parsers and renderers. In the current ablations it is close to greedy decoding on clean Verovio pages (18.74% OMR-NED) and improves the Polish scan score to 63.91% OMR-NED, while beam search still gives the best CER.

## Data pipeline

Training uses 310,554 synthetic examples. Two augmentation families decouple visual diversity from target ambiguity.

**Asymmetric semantic augmentation** adds dynamics, pedal markings, and tempo text to the _rendered image_ without changing the canonical `**kern` target. The model sees richer engraving without inheriting transcription noise.

<p align="center">
  <img src="figures/semantic-base.png"      alt="Base render"      width="49%">
  <img src="figures/semantic-augmented.png" alt="Same target with added dynamics, pedal, and tempo markings" width="49%">
</p>

**Visual degradation** simulates print-and-scan reality: clean baseline, geometric warp, ink drop-out, and bleed-through.

<p align="center">
  <img src="figures/aug-clean.png"    alt="Clean render"   width="49%">
  <img src="figures/aug-warp.png"     alt="Geometric warp" width="49%">
</p>
<p align="center">
  <img src="figures/aug-poor-ink.jpg" alt="Low-ink degradation" width="49%">
  <img src="figures/aug-bleed.jpg"    alt="Bleed-through"       width="49%">
</p>

## Qualitative example

Bach excerpt, identical input. Transcoda preserves rhythm and pitch; both baselines drift on long-range structure.

<p align="center">
  <img src="figures/qualitative.png" alt="Side-by-side outputs of Transcoda, Legato, and SMT++ on a Bach excerpt." width="85%">
</p>

TEDn against ground truth: **Transcoda 1.5**, Legato 3.12, SMT++ 66.04.

## Quick Start

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone and install
git clone <repo-url> && cd Transcoda
uv sync

# Optional dependency groups
uv sync --group grammar   # xgrammar for constrained decoding
uv sync --group omr-ned   # semantic OMR evaluation metrics
uv sync --group dev       # development tools
```

### Inference

```bash
uv run scripts/inference.py \
    --weights ./weights/model.ckpt \
    --image path/to/score.png
```

Requires the `grammar` dependency group. Auto-detects CUDA/MPS/CPU.

## Training

```bash
source .venv/bin/activate

# Train from scratch
python train.py config/train.json

# Override config values via CLI (dot notation)
python train.py config/train.json --model.d_model=256 --training.max_epochs=10

# Resume from last checkpoint (enable auto_resume explicitly)
python train.py config/train.json --checkpoint.auto_resume=true

# Start fresh, ignoring existing checkpoints
python train.py config/train.json --fresh_run=true

# Validate a checkpoint without training
python train.py config/train.json --validate_only=true --checkpoint_path path/to/model.ckpt
```

See [`docs/TRAINING_AND_DATA.md`](docs/TRAINING_AND_DATA.md) for configuration details, Slurm
commands, FCMAE pretraining, dataset generation, profiling, and sequence trimming.

## Dataset Generation

The pipeline converts raw music scores through several stages into training-ready image-kern pairs.

### Stages

1. **Raw extraction** — pull source files (MusicXML, ekern)
2. **Kern conversion** — convert to `**kern` format
3. **Filtering** — drop structurally invalid kern files
4. **Normalization** — 21-pass canonicalization pipeline
5. **Manual fixes** — curated corrections (Polish scores)
6. **Rendering** — Verovio renders kern to SVG to PNG with augmentations
7. **Sequence-length filtering** — remove outliers

See [`docs/TRAINING_AND_DATA.md`](docs/TRAINING_AND_DATA.md) for dataset generation commands, run
artifacts, calibration notes, profiling, and sequence trimming.

See [`docs/normalization.md`](docs/normalization.md) for details on the normalization pipeline.

## Project Structure

```
src/
├── model/               # Vision-encoder-decoder architecture
├── training/            # Lightning module, optimizer setup
├── data/                # Datasets, collators, preprocessing
├── grammar/             # Grammar-constrained decoding (xgrammar)
├── core/                # Tokenizer utils, kern processing, metrics
├── evaluation/          # Evaluation harness, OMR-NED, wrappers
└── callbacks/           # W&B logging, progress tracking

scripts/
├── inference.py         # Single-image inference CLI
├── dataset_generation/  # Full data pipeline
└── benchmark/           # Evaluation & metric computation

grammars/kern.gbnf       # GBNF grammar for **kern notation
vocab/                   # BPE tokenizer (3k tokens)
config/                  # Training & fine-tuning configs
docs/                    # Architecture, normalization, constraint docs
```

## Documentation

- [Model Architecture](docs/MODEL_ARCHITECTURE.md) — encoder-decoder design, positional encoding, attention
- [Training and Data Operations](docs/TRAINING_AND_DATA.md) — Slurm commands, FCMAE pretraining, dataset generation, profiling, trimming
- [Normalization Pipeline](docs/normalization.md) — 21-pass kern canonicalization
- [Constrained Decoding](docs/constraint-decoding.md) — xgrammar integration and GBNF grammar
- [Configuration](config/README.md) — config structure, CLI overrides, per-section reference

## Paper

The paper is in [`paper.pdf`](./paper.pdf).

## Citation

```bibtex
@misc{dratschuk2026transcodaendtoendzeroshotoptical,
      title={Transcoda: End-to-End Zero-Shot Optical Music Recognition via Data-Centric Synthetic Training},
      author={Daniel Dratschuk and Paul Swoboda},
      year={2026},
      eprint={2605.10835},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.10835},
}
```
