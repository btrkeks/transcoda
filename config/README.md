# Configuration

Simple, flat config structure. Each config is self-contained.

## Files

```
config/
├── data_spec.json # Canonical dataset image geometry (generation + validation assembly)
├── train.json      # Production training
├── finetune_real.json # Real-scan fine-tuning (OlimpIC + retention validation)
├── finetune_polish_adapt_real_only.json # Real-only Polish adaptation finetuning
├── finetune_polish_adapt_mixed.json # Mixed synth+Polish adaptation finetuning
├── debug.json      # Quick sanity checks (~1 min)
└── profile.json    # Performance profiling
```

## Usage

```bash
# Production training
python train.py --config_path config/train.json

# Real-data fine-tuning from synthetic checkpoint
python train.py config/finetune_real.json --checkpoint_path /path/to/synth_pretrain.ckpt

# Leakage-safe Polish adaptation (mixed)
python train.py config/finetune_polish_adapt_mixed.json --checkpoint_path /path/to/synth_pretrain.ckpt

# Leakage-safe Polish adaptation (real-only)
python train.py config/finetune_polish_adapt_real_only.json --checkpoint_path /path/to/synth_pretrain.ckpt

# Quick debug run
python train.py --config_path config/debug.json

# With CLI overrides
python train.py --config_path config/train.json --training.batch_size=4 --training.max_epochs=5

# Validation-only pass on a checkpoint (no training)
python train.py --config_path config/train.json --validate_only=true --checkpoint_path /path/to/best.ckpt
```

## CLI Overrides

Any config value can be overridden from the command line using dot notation:

```bash
# Change model size
python train.py --config_path config/train.json --model.d_model=256

# Quick test with smaller batches
python train.py --config_path config/train.json --training.limit_train_batches=10 --training.max_epochs=1

# Different seed
python train.py --config_path config/train.json --seed=1337
```

## Config Sections

| Section      | Purpose                                    |
|--------------|-------------------------------------------|
| `data`       | Dataset paths and preprocessing            |
| `training`   | Batch sizes, epochs, hardware settings     |
| `model`      | Architecture (d_model, layers, encoder)    |
| `optimizer`  | Learning rate, weight decay, scheduler     |
| `checkpoint` | Model saving and W&B logging               |

## Current Production Defaults (`config/train.json`)

- Checkpoint monitor: `checkpoint.monitor="val/polish/CER_no_ties_beams"` (`mode="min"`)
- Auto-resume: off by default (`checkpoint.auto_resume=false`); set `checkpoint.auto_resume=true` to resume from `last.ckpt`
- Scheduler: `optimizer.lr_scheduler="cosine"` (warmup + single cosine decay)
- Tiered validation:
  - frequent: every `training.val_check_interval` (default 1000 steps) on `training.frequent_validation_set_names` (default `["polish"]`)
  - frequent subset metrics: every validation pass on deterministic subsets from `training.frequent_validation_subset_sizes` (default `{"synth": 256}`), logged separately as `val/synth_subset/CER`
  - full: every `training.full_validation_every_n_steps` (default 5000 steps) on all validation sets

To force a fresh run explicitly:

```bash
python train.py config/train.json --fresh_run=true
```

## Polish Adaptation Defaults

- Use `scripts/dataset_generation/run_polish_adapt_pipeline.sh` to build:
  - `data/datasets/train_polish_adapt`
  - `data/datasets/validation/polish_dev`
  - `data/datasets/validation/polish_test`
  - `data/datasets/train_polish_adapt_mixed`
- New Polish adaptation configs monitor `val/polish_dev/CER_no_ties_beams`.
- The legacy `data/datasets/validation/polish` artifact is all-splits PRAIG and should not be used as a clean benchmark for adaptation training.
