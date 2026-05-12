# Training and Data Operations

This page collects operational commands for training, FCMAE pretraining, dataset generation,
profiling, sequence trimming, cloud artifacts, and Slurm jobs.

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

## Configuration

Training configs live in `config/`. Key files:

| File                 | Purpose                                     |
| -------------------- | ------------------------------------------- |
| `train.json`         | Production training (synthetic grand staff) |
| `finetune_real.json` | Fine-tune on real scans                     |
| `debug.json`         | Quick sanity check                          |
| `profile.json`       | Performance profiling                       |

See [`../config/README.md`](../config/README.md) for full config documentation and all available options.

## Remote Training (Slurm)

```bash
./train.sh submit            # Submit job to cluster; auto-assigns and prints a unique run id by default
./train.sh validate          # Validate the last submitted run by default
./train.sh logs              # View job logs
./train.sh queue             # Show job queue
./train.sh cancel            # Cancel running job
```

### Training

```bash
./train.sh submit -- \
  --data.train_path=./data/datasets/train/train_400k \
  --model.max_seq_len=3000 \
  --model.encoder_model_name_or_path=weights/fcmae-real-scans/exported_encoder_3
```

### Validation

```bash
./train.sh validate
```

By default, `validate` auto-selects the last submitted run. To validate a specific checkpoint,
pass `--checkpoint_path` before any forwarded `train.py` overrides:

```bash
./train.sh validate --checkpoint_path weights/my-run/epoch=12-step=3456.ckpt --gpu-type rtx4090
```

`--checkpoint` is accepted as an alias. Put config overrides after `--`:

```bash
./train.sh validate \
  --checkpoint_path weights/my-run/epoch=12-step=3456.ckpt \
  --gpu-type rtx4090 \
  -- \
  --data.batch_size=4
```

## FCMAE Encoder Pretraining

FCMAE-style masked image pretraining uses `config/pretrain_fcmae_base.json` and writes regular
Lightning checkpoints. Local smoke runs keep W&B disabled by default:

```bash
source .venv/bin/activate
python scripts/pretrain_fcmae.py config/pretrain_fcmae_base.json training.max_steps=10
```

On Slurm, use the dedicated wrapper and put config overrides after `--` as `key=value` tokens.
New submit jobs auto-create a fresh timestamped checkpoint directory under the configured
`checkpoint.dirpath` parent:

```bash
./pretrain_fcmae.sh submit \
  --job-name fcmae-real-scans \
  --time 48:00:00 \
  --mem 64G \
  -- \
  data.manifest_path=data/fcmae_real_scans_manifest.txt \
  data.image_dir=null \
  logging.wandb_enabled=true
```

Resume with either `--resume weights/fcmae-real-scans/last.ckpt` or the equivalent
`training.resume_from_checkpoint=...` override. By default, resumed submit jobs fork into a new
timestamped output directory; pass `checkpoint.dirpath=weights/fcmae-real-scans` explicitly if you
want to continue writing into the original directory. `./pretrain_fcmae.sh logs` tails the most
recent remembered FCMAE job.

### Exporting a Pretrained Encoder

Export the ConvNeXtV2 encoder as a Hugging Face directory:

```bash
python scripts/export_fcmae_encoder.py \
  weights/fcmae-real-scans/last-v5.ckpt \
  weights/fcmae-real-scans/exported_encoder_3 \
  --validate
```

The exporter refuses to write into a non-empty directory unless `--overwrite` is passed and writes
`fcmae_export_metadata.json` next to the exported `config.json`.

For Slurm training, `export.export_on_train_end=true` defaults `export.output_dir` to
`exported_encoder` inside the generated checkpoint directory.

### Supervised Handoff

The exported encoder is consumed through the existing Transformers encoder provider:

```bash
./train.sh submit -- \
  model.encoder_provider=transformers \
  model.encoder_model_name_or_path=weights/fcmae-real-scans/exported_encoder \
  checkpoint.dirpath=weights/transcoda-fcmae-real-scans-finetune \
  checkpoint.run_name=transcoda-fcmae-real-scans-finetune
```

## Dataset Generation

The pipeline converts raw music scores through several stages into training-ready image-kern pairs.

### Stages

1. **Raw extraction** — pull source files (MusicXML, ekern)
2. **Kern conversion** — convert to `**kern` format. MusicXML inputs are converted with `musicxml2hum` from our humlib fork ([btrkeks/musicxml2hum](https://github.com/btrkeks/musicxml2hum)); the upstream binary is not interchangeable for reproducing the published datasets.
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

The normalization implementation lives under `scripts/dataset_generation/normalization/`.

### Generate Dataset

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
  --max_attempts 999
```

### Profile Dataset Generation

```bash
source .venv/bin/activate

python scripts/dataset_generation/profile_dataset_generation.py \
  data/interim/train/pdmx/3_normalized \
  data/interim/train/grandstaff/3_normalized \
  data/interim/train/musetrainer/3_normalized \
  data/interim/train/openscore-lieder/3_normalized \
  data/interim/train/openscore-stringquartets/3_normalized \
  --target_samples 100 \
  --num_workers 4
```

The harness writes a timestamped run directory under `/tmp/dataset_generation_profiles/` by default. Start with:

- `summary.json` for the full command, elapsed time, and artifact paths
- `pyspy.svg` for the subprocess flamegraph
- `generation.stdout.log` and `generation.stderr.log` for harness output
- `system_ps.log`, `system_vmstat.log`, and `system_top.log` for machine telemetry
- the dataset-generation `run_artifacts_dir` referenced from `summary.json`

### Trim Long Sequences

```bash
python ./scripts/dataset_generation/filter_by_seq_len.py ./data/datasets/train/train_20k ./vocab/bpe3k-splitspaces 3000
```

## Cloud Commands

```bash
./cloud pull 20260422T183615Z-12556
```

## Pretrain

```bash
./pretrain_fcmae.sh submit --sbatch-arg '--nodelist=tujestpolin' --gpus 2
```
