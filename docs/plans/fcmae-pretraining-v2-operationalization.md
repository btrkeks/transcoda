# FCMAE Pretraining V2 Operationalization Plan

V1 added the isolated dense masked-image-modeling pretraining lane for ConvNeXtV2. V2 makes that lane usable for real runs: Slurm submission, Weights & Biases logging, robust checkpoint export, and a clean handoff into supervised encoder-decoder training.

This plan assumes the V1 implementation from commit `57b2b0e`:

- `scripts/pretrain_fcmae.py`
- `scripts/export_fcmae_encoder.py`
- `src/pretraining/fcmae/`
- `config/pretrain_fcmae_base.json`
- `tests/test_fcmae_smoke.py`

---

## 0. Decisions Already Made

| # | Decision | Value |
| --- | --- | --- |
| D1 | V2 scope | Operational hardening only: Slurm, W&B, export, supervised-loading integration. No objective redesign. |
| D2 | Slurm interface | Add a dedicated `pretrain_fcmae.sh`, modeled after `train.sh` and `scripts/benchmark/submit_slurm.sh`. |
| D3 | W&B | Use Lightning `WandbLogger` when explicitly enabled; default is off for local/smoke runs. |
| D4 | Export format | Hugging Face encoder directory via `encoder.save_pretrained(export_dir)`. No custom `encoder_provider` for supervised training. |
| D5 | Supervised handoff | Supervised training consumes export by setting `model.encoder_model_name_or_path=<export_dir>` with existing `encoder_provider="transformers"`. |
| D6 | Export timing | Manual export script remains primary. Optional export-on-train-end is config-driven only through `config.export.*`; no separate Slurm `--export-dir` knob. |
| D7 | Checkpoint upload | Keep `log_model=False` for W&B by default. Local checkpoints remain canonical. |

---

## 1. Current V1 Baseline

V1 already provides:

- JSON config with dotlist overrides.
- `ModelCheckpoint(save_last=True)`.
- dense MIM model with `norm_pix_loss`.
- image-folder/manifest data input.
- offline smoke tests.
- export script that loads `FCMAEPretrainer` and calls `module.model.encoder.save_pretrained(...)`.

V2 should not disturb these contracts except to extend config surface and logging. If a change would alter the pretraining objective, move it to a separate v3/research plan.

---

## 2. Proposed File Changes

Add:

```text
pretrain_fcmae.sh
src/pretraining/fcmae/logging.py
tests/test_fcmae_export.py
tests/test_fcmae_wandb_config.py
tests/scripts/test_pretrain_fcmae_slurm_script.py
```

Modify:

```text
config/pretrain_fcmae_base.json
scripts/pretrain_fcmae.py
scripts/export_fcmae_encoder.py
src/pretraining/fcmae/config.py
src/pretraining/fcmae/lightning_module.py
src/pretraining/fcmae/model.py
README.md
```

Optional, only if useful:

```text
scripts/diagnostics/check_fcmae_export.py
```

---

## 3. Config Additions

Extend `src/pretraining/fcmae/config.py`.

### Logging Config

Add:

```python
class FCMAELoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wandb_enabled: bool = False
    project: str = "SMT-FCMAE"
    run_name: str | None = None
    group: str | None = None
    tags: list[str] = Field(default_factory=list)
    log_model: bool = False
    log_reconstructions: bool = True  # Active only when wandb_enabled=true.
    log_reconstruction_every_n_steps: int = 500
    log_reconstruction_max_batches: int | None = 20
    log_reconstruction_max_images: int = 4
```

Add `logging: FCMAELoggingConfig = Field(default_factory=FCMAELoggingConfig)` to `FCMAEConfig`.

Validation:

- `log_reconstruction_every_n_steps >= 1`.
- `log_reconstruction_max_batches` is `None` or `>= 1`.
- `log_reconstruction_max_images >= 1`.

### Export Config

Add:

```python
class FCMAEExportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str | None = None
    export_on_train_end: bool = False
    overwrite: bool = False
    validate_export: bool = True
```

Add `export: FCMAEExportConfig = Field(default_factory=FCMAEExportConfig)` to `FCMAEConfig`.

Rules:

- if `export_on_train_end=true`, `output_dir` must be set.
- export writes a Hugging Face directory suitable for `AutoModel.from_pretrained(output_dir)`.

### Slurm-Oriented Config Defaults

Do not encode Slurm resources in JSON config. Keep Slurm resource selection in `pretrain_fcmae.sh` so local and cluster runs share the same config file.

### Resume Support

Add one field to `FCMAETrainingConfig`:

```python
resume_from_checkpoint: str | None = None
```

`scripts/pretrain_fcmae.py` passes it to:

```python
trainer.fit(module, datamodule=datamodule, ckpt_path=config.training.resume_from_checkpoint)
```

The Slurm wrapper may expose a convenience `--resume PATH` flag that forwards `training.resume_from_checkpoint=PATH`.

---

## 4. Weights & Biases Logging

### Scalar Logging

`FCMAEPretrainer` already logs:

- `train/loss`
- `train/mask_ratio`
- `train/masked_foreground_ratio`
- `train/samples_skipped_no_valid_patches`
- `train/lr`

V2 should add one dynamic scalar:

- `train/valid_patch_ratio`

Add this to `MaskedImageModelingOutput` or compute it in `training_step` from `batch["valid_patch_mask"]`. Prefer logging it from the Lightning module with `self.log(...)` so it behaves under distributed training.

Static run values should be logged once via W&B hyperparameters/config, not every step:

- `effective_batch_size`
- `base_learning_rate`
- resolved initial learning rate
- `pixels_per_sample`

The scheduled LR is already logged step-wise as `train/lr`.

### W&B Logger Setup

Update `scripts/pretrain_fcmae.py`:

1. If `config.logging.wandb_enabled`, create `WandbLogger`; otherwise pass `logger=False` or omit logger.
2. Pass it into `L.Trainer(logger=logger)`.
3. Log the full config dict with `logger.log_hyperparams(...)`.
4. Use `log_model=config.logging.log_model`, default `False`.
5. Add Slurm metadata from `src.artifacts.collect_slurm()` to W&B config when available, using the same `slurm/...` key style as `train.py`.

Suggested group default:

- if `logging.group` is set, use it;
- else derive from `checkpoint.dirpath` basename, mirroring supervised `setup_logger`.

### Reconstruction Preview Logging

Add `src/pretraining/fcmae/logging.py` with a small Lightning callback, for example `FCMAEReconstructionLogger`.

Behavior:

- Every `log_reconstruction_every_n_steps`, log a preview from the current training batch output.
- Log at most `log_reconstruction_max_images`.
- Stop after `log_reconstruction_max_batches` preview batches if that value is set.
- Show original image, masked input, reconstruction preview, and mask overlay.
- Keep logging CPU-friendly and best-effort. If W&B is disabled, callback is inert.

Important details:

- `training_step` currently returns only loss, so the module must stash a detached preview payload during the actual training forward, e.g. `self._latest_preview = {...}` containing the current batch tensors and `MaskedImageModelingOutput`. The callback reads and clears this payload in `on_train_batch_end`.
- Do not re-run a no-grad forward in the callback for the default preview path; that would resample the mask and double the encoder work.
- Inputs are normalized `[-1, 1]`; convert to display by `(x + 1) / 2`, clamp `[0, 1]`.
- Patch predictions are normalized if `norm_pix_loss=True`, so a faithful reconstruction preview may require unnormalizing patch targets/predictions. For v2, it is acceptable to log:
  - original image;
  - masked input;
  - mask overlay;
  - optionally predicted patches if unnormalization is implemented safely.
- Do not make image logging block training. Wrap preview generation in `torch.no_grad()` and catch/log preview-only failures.

Acceptance gate: scalar W&B logging must work even if preview logging is disabled.

---

## 5. Slurm Script

Add `pretrain_fcmae.sh` at repo root.

It should mirror the useful ergonomics of `train.sh` without copying all supervised-specific checkpoint validation logic.

Commands:

```bash
./pretrain_fcmae.sh submit [options] -- [config overrides]
./pretrain_fcmae.sh local [options] -- [config overrides]
./pretrain_fcmae.sh queue
./pretrain_fcmae.sh cancel JOB_ID
./pretrain_fcmae.sh logs [JOB_ID] [--lines N] [--no-follow]
./pretrain_fcmae.sh export CHECKPOINT_PATH OUTPUT_DIR
./pretrain_fcmae.sh help
```

Default config:

```bash
config/pretrain_fcmae_base.json
```

Slurm options:

- `--partition`
- `--gpus`
- `--gpu-type`
- `--cpus-per-task`
- `--mem`
- `--time`
- `--job-name`
- `--sbatch-arg` repeatable
- `--resume PATH`
- `--no-sync`
- `--dry-run`

Override grammar:

- after `--`, forward overrides to `scripts/pretrain_fcmae.py` as bare `key=value` tokens, matching V1;
- do not copy `train.sh` helpers that assume `--key=value` without adapting them;
- if the wrapper accepts `--key=value` for user convenience, strip the leading `--` before calling Python.

Execution requirements:

- Use `bash -lc 'source .venv/bin/activate && ...'` inside the Slurm job body.
- Do not rely on `/bin/sh` sourcing `.venv/bin/activate`.
- If running `uv sync`, do it under `bash -lc` or export the expected user-local bin path first.
- Write Slurm logs under `logs/`.
- Write a small last-run state file, e.g. `logs/pretrain_fcmae-last-run.env`, storing job id, config path, checkpoint dir, and optional config export dir.

Suggested job body:

```bash
cd "${ROOT_DIR}"
source .venv/bin/activate
python scripts/pretrain_fcmae.py "${CONFIG}" "${OVERRIDES[@]}"
```

Optional export-on-train-end:

- If `config.export.export_on_train_end=true`, `scripts/pretrain_fcmae.py` exports after `trainer.fit(...)` returns successfully:

```bash
python scripts/export_fcmae_encoder.py "${LAST_CKPT}" "${config.export.output_dir}" --validate
```

where `LAST_CKPT` resolves to `${checkpoint.dirpath}/last.ckpt` after applying config overrides.

This is config-driven only. Do not add a separate Slurm `--export-dir` flag in v2. Manual export remains the primary supported path.

---

## 6. Checkpoint Export Hardening

Current `scripts/export_fcmae_encoder.py` is intentionally small. V2 should make it safer.

Add CLI flags:

```bash
python scripts/export_fcmae_encoder.py CKPT OUTPUT_DIR \
  --overwrite \
  --validate
```

Behavior:

1. Load `FCMAEPretrainer` from checkpoint on CPU.
2. Confirm `module.model.encoder` exposes `save_pretrained`.
3. Refuse to overwrite non-empty `OUTPUT_DIR` unless `--overwrite`.
4. Call `encoder.save_pretrained(OUTPUT_DIR)`.
5. Copy or write a small metadata file:

```json
{
  "source_checkpoint": "...",
  "source_encoder_model_name_or_path": "...",
  "pretraining_config": "...full FCMAEConfig.model_dump()...",
  "git_commit": "...",
  "v1_implementation_commit": "57b2b0e",
  "exported_at_utc": "..."
}
```

Suggested filename:

```text
fcmae_export_metadata.json
```

6. If `--validate`, run:

```python
AutoModel.from_pretrained(output_dir)
```

Then do a tiny no-grad forward on `(1, 3, 768, 544)` or the config image size if available, and assert the feature map shape is 4D. This validates export mechanics at pretraining resolution, not the supervised input contract.

Tests:

- exporting a dummy/small injected-encoder checkpoint should either use a tiny save-pretrained-capable fake encoder or test the overwrite/metadata logic separately.
- real HF export validation can be a diagnostics script or an optional test skipped without local weights.

---

## 7. Supervised Loading Integration

The desired steady-state user flow:

```bash
./pretrain_fcmae.sh submit \
  --job-name fcmae-real-scans \
  --time 48:00:00 \
  --mem 64G \
  -- \
  data.manifest_path=data/fcmae_real_scans_manifest.txt \
  checkpoint.dirpath=weights/fcmae-real-scans

python scripts/export_fcmae_encoder.py \
  weights/fcmae-real-scans/last.ckpt \
  weights/fcmae-real-scans/exported_encoder \
  --validate

./train.sh submit -- \
  model.encoder_model_name_or_path=weights/fcmae-real-scans/exported_encoder \
  checkpoint.dirpath=weights/smt-fcmae-real-scans-finetune \
  checkpoint.run_name=smt-fcmae-real-scans-finetune
```

No supervised code change should be necessary for the first handoff because `EncoderLoader._load_transformers` already calls `AutoModel.from_pretrained(...)`.

Add documentation in `README.md`:

- one section for pretraining;
- one section for exporting;
- one section for using the exported encoder in supervised training.

Optional safety check:

- add `scripts/diagnostics/check_supervised_encoder_load.py EXPORT_DIR`.
- It creates an `SMTConfig` or calls `EncoderLoader` with `encoder_model_name_or_path=EXPORT_DIR` and asserts output dimension detection succeeds.
- It runs a no-grad one-batch encoder/front-end probe at the supervised image contract `1485 x 1050`.
- It asserts the expected supervised pixel convention is still `[-1, 1]` with mean/std `0.5`, so no ImageNet preprocessing mismatch is introduced silently.

Do not add a new `encoder_provider` unless local HF export proves impossible.

---

## 8. Implementation Order

1. Add config classes for `logging` and `export`.
2. Update `config/pretrain_fcmae_base.json`.
3. Add W&B logger setup to `scripts/pretrain_fcmae.py`.
4. Add reconstruction logging callback with image logging disabled by default in tests.
5. Harden `scripts/export_fcmae_encoder.py` with overwrite, metadata, and validation.
6. Add `pretrain_fcmae.sh` with `submit`, `local`, `queue`, `cancel`, `logs`, `export`, and `--resume`.
7. Add tests for config validation and export overwrite/metadata behavior.
8. Add shell-script tests for help/dry-run command rendering.
9. Add README usage docs.
10. Run smoke tests locally.
11. On `antemurale`, run a short Slurm dry run with `training.max_steps=10`.
12. Export the dry-run checkpoint and run supervised encoder-load diagnostic.

---

## 9. Acceptance Criteria

Local:

- `pytest tests/test_fcmae_smoke.py` passes.
- New config/export/script tests pass.
- `python scripts/pretrain_fcmae.py config/pretrain_fcmae_base.json training.max_steps=1 ...` still works without W&B when disabled.
- `scripts/export_fcmae_encoder.py --validate` refuses unsafe overwrite and writes metadata.

Slurm:

- `./pretrain_fcmae.sh submit --dry-run ...` prints a valid job script.
- `./pretrain_fcmae.sh submit -- ... training.max_steps=10` completes on `antemurale`.
- `./pretrain_fcmae.sh logs` can follow the job output.
- The Slurm job logs W&B scalars when `logging.wandb_enabled=true`.
- Resuming with `--resume weights/.../last.ckpt` or `training.resume_from_checkpoint=...` works.

W&B:

- Run config is logged.
- Slurm metadata is logged when running under Slurm.
- Scalars appear: `train/loss`, `train/lr`, `train/masked_foreground_ratio`, `train/valid_patch_ratio`, `train/samples_skipped_no_valid_patches`.
- Reconstruction previews appear when enabled.
- `log_model=false` by default.

Export:

- `weights/.../exported_encoder/config.json` exists.
- `AutoModel.from_pretrained(exported_encoder)` works.
- Export metadata records source checkpoint, full pretraining config, current git commit, and V1 implementation commit.

Supervised handoff:

- `EncoderLoader` can load the exported encoder path.
- A one-batch supervised/front-end probe succeeds at `1485 x 1050` with `model.encoder_model_name_or_path=<exported_encoder>`.
- The supervised diagnostic confirms the expected `[-1, 1]` pixel convention.
- No new supervised encoder provider is required.

---

## 10. Deferred Work

- automatic upload of exported encoder as a W&B artifact;
- full Slurm array support for multi-resolution pretraining runs;
- validation split and reconstruction validation metrics;
- visual comparison dashboard for pretraining checkpoints;
- automatic supervised fine-tune launch after export;
- sparse FCMAE/MinkowskiEngine feasibility spike;
- uploading the exported encoder to Hugging Face Hub.

---

## 11. Risks and Notes

- W&B image logging can be slow. Keep it cadence-limited and easy to disable.
- Export validation with real ConvNeXtV2 weights may be memory-heavy on CPU for large rectangular inputs. Use no-grad and allow a smaller validation shape if needed.
- Slurm wrapped jobs must use `bash -lc 'source .venv/bin/activate && ...'` per repo guidance.
- Do not use W&B model artifact upload for every checkpoint in v2; local checkpoints are the source of truth.
- The exported encoder inherits the V1 `[-1, 1]` pixel convention expected by supervised training, even though the original HF model card uses ImageNet-style preprocessing.
