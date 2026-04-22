# Transcoda

Optical Music Recognition (OMR) system that converts sheet music images into [`**kern`](https://www.humdrum.org/rep/kern/) (Humdrum) notation. Uses a vision-encoder-decoder architecture with grammar-constrained decoding to ensure structurally valid output.

## Architecture

```
Image (B, 3, H, W)
    │
    ▼
┌──────────────────────────────────────────────┐
│ ConvVisionFrontend                           │
│  - HuggingFace vision backbone (AutoModel)   │
│  - Token-space MLP projector                 │
│  - 2D sinusoidal positional stream           │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ Autoregressive Decoder                       │
│  - Token embedding + RoPE positions          │
│  - N pre-norm decoder layers                 │
│      self-attn → cross-attn → FFN            │
│  - Final LayerNorm + vocab projection        │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ Grammar-Constrained Decoding                 │
│  - xgrammar with GBNF grammar for **kern     │
│  - Spine structure & rhythm constraints      │
│  - Runaway guard                             │
└──────────────────────────────────────────────┘
    │
    ▼
**kern transcription
```

The vision encoder is pluggable — any HuggingFace vision model works (default: ConvNeXtV2-tiny-22k). The decoder is a custom transformer with cross-attention over encoder features. At inference time, an [xgrammar](https://github.com/mlc-ai/xgrammar)-based constrained decoding layer filters logits to guarantee syntactically valid `**kern` output.

## Features

- **Pluggable vision encoder** — swap in any HuggingFace vision backbone via config
- **Grammar-constrained decoding** — GBNF grammar enforces valid `**kern` syntax at every decoding step
- **21-pass normalization pipeline** — canonicalizes ground-truth kern for consistent training targets
- **Synthetic data generation** — renders kern scores to images via Verovio with configurable augmentations
- **PyTorch Lightning training** — tiered validation, auto-resume, model compilation, gradient accumulation
- **Comprehensive metrics** — CER, SER, LER, and OMR-NED (semantic music notation distance)
- **W&B integration** — experiment tracking, artifact logging, example visualization

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

# Resume from last checkpoint (auto_resume is on by default)
python train.py config/train.json

# Start fresh, ignoring existing checkpoints
python train.py config/train.json --fresh_run=true

# Validate a checkpoint without training
python train.py config/train.json --validate_only=true --checkpoint_path path/to/model.ckpt
```

### Configuration

Training configs live in `config/`. Key files:

| File                 | Purpose                                     |
| -------------------- | ------------------------------------------- |
| `train.json`         | Production training (synthetic grand staff) |
| `finetune_real.json` | Fine-tune on real scans                     |
| `debug.json`         | Quick sanity check                          |
| `profile.json`       | Performance profiling                       |

See [`config/README.md`](config/README.md) for full config documentation and all available options.

### Remote Training (Slurm)

```bash
./train.sh submit            # Submit job to cluster
./train.sh validate          # Validate job setup
./train.sh logs              # View job logs
./train.sh queue             # Show job queue
./train.sh cancel            # Cancel running job
```

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

### Running the Pipeline

```bash
source .venv/bin/activate

# Generate training dataset with the production rewrite
python -m scripts.dataset_generation.dataset_generation.main \
  data/interim/train/pdmx/3_normalized \
  data/interim/train/grandstaff/3_normalized \
  data/interim/train/openscore-lieder/3_normalized \
  data/interim/train/openscore-stringquartets/3_normalized \
  --output_dir data/datasets/train_rewrite \
  --target_samples 1000 \
  --num_workers 4

# Or use the convenience wrapper
bash scripts/dataset_generation/run_rewrite_train_pipeline.sh

# Validation generation remains on the old pipeline path for now
PIPELINE_TARGET=validation bash scripts/dataset_generation/run_full_pipeline.sh
```

The rewrite writes a Hugging Face dataset via `save_to_disk()` at `output_dir` and stores run artifacts under `data/datasets/_runs/<output_name>/<run_id>/` by default. Each run directory includes:

- `info.json` with the run configuration and summary
- `progress.json` with live counters
- resumable state under `<output_dir>/.resume/`

The rewrite dataset also includes clean-render layout diagnostics such as `vertical_fill_ratio`,
`bottom_whitespace_ratio`, `bottom_whitespace_px`, `top_whitespace_px`, and
`content_height_px`. These are measured before offline dirt/degradation augmentation so they
reflect page occupancy of the synthetic render itself rather than the augmented image.

Balanced generation is mandatory in the rewrite path. Production runs always use the bundled
token-length calibration spec checked into the repo, and generation aborts if that spec is
missing or incompatible with the current recipe/tokenizer. The calibration CLI remains available
as a maintenance tool for refreshing the bundled spec when those assumptions change.

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
- [Normalization Pipeline](docs/normalization.md) — 21-pass kern canonicalization
- [Constrained Decoding](docs/constraint-decoding.md) — xgrammar integration and GBNF grammar
- [Configuration](config/README.md) — config structure, CLI overrides, per-section reference

## Commands

### Generate dataset

```bash
source .venv/bin/activate && python -m scripts.dataset_generation.dataset_generation.main \
  data/interim/train/pdmx/3_normalized \
  data/interim/train/grandstaff/3_normalized \
  data/interim/train/musetrainer/3_normalized \
  data/interim/train/openscore-lieder/3_normalized \
  data/interim/train/openscore-stringquartets/3_normalized \
  --name test_v1 \
  --target_samples 100 \
  --num_workers 4 \
  --max_attempts 999 \
```
