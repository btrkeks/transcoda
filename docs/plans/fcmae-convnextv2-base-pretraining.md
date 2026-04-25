# FCMAE ConvNeXtV2-Base Pretraining Plan

Add a minimal, isolated FCMAE-style continued pretraining lane for a `ConvNeXtV2-base` encoder, using image-only sheet music data and a smoke-testable training loop. The first implementation should prove the objective, data path, checkpointing, and future supervised-loading handoff without disturbing the existing OMR encoder-decoder training path.

This plan intentionally starts with a dense PyTorch implementation rather than the exact upstream sparse implementation from Meta's ConvNeXt V2 repository. The upstream FCMAE code uses sparse ConvNeXt modules and `MinkowskiEngine`; that is useful for faithful reproduction, but a heavier operational dependency for this repo and the remote Slurm environment. We can revisit sparse FCMAE after the dense loop works.

Because the encoder sees zeroed pixel regions instead of sparse-encoded visible-only tokens, what we implement in v1 is a **SimMIM-style masked image modeling** objective with the FCMAE decoder shape, not literal FCMAE. We keep the `fcmae/` directory name since the decoder design and borrowed mask/loss math come from the FCMAE reference, but class names and docstrings must be accurate about the objective (see Section 6 and Section 15).

---

## 0. Decisions Already Made

| # | Decision | Value |
| --- | --- | --- |
| D1 | Pretraining backbone | `facebook/convnextv2-base-22k-224`, continued from the Hugging Face checkpoint |
| D2 | First implementation style | Dense PyTorch masked image modeling loop inspired by FCMAE, not upstream sparse `MinkowskiEngine` FCMAE |
| D3 | Scope of first pass | Minimal trainable loop plus smoke test |
| D4 | Data source shape | Image-only dataset, initially easiest as a folder of image files |
| D5 | Pretraining image size | Configurable rectangular target size; default should be moderate for throughput |
| D6 | Supervised integration | Deferred until the pretraining loop is proven |
| D7 | Export/load format | Hugging Face directory via `encoder.save_pretrained(dir)`; consumed unchanged by `EncoderLoader._load_transformers` at `src/model/encoder.py:65` |
| D8 | Existing OMR training | Keep untouched except for a future explicit encoder-load integration |
| D9 | Pixel normalization | Inherit supervised `[-1, 1]` convention (`mean=std=0.5`) from `src/data/preprocessing.py`, not ImageNet stats. Deliberate: matches the supervised pipeline the encoder will feed into. |
| D10 | Pad color | White (255 uint8 → +1.0 post-norm), matching `preprocess_pil_image` behavior. Note: this diverges from the `NORMALIZED_PAD_VALUE = -1.0` constant in `src/data/preprocessing.py`, which is used elsewhere by the collator; we intentionally follow the *resize* behavior, not the *collate* constant. |
| D11 | Masking objective class name | `DenseMaskedImageModelingConvNeXtV2` — accurate to what it does. Folder stays `fcmae/` to signal design lineage. |
| D12 | `norm_pix_loss` | Enabled by default, ported from upstream FCMAE. |
| D13 | Valid-patch rule | A patch is valid iff **all** of its pixels are non-padding. No fractional-threshold variants in v1. |
| D14 | Borrowed code | Port ~60 LOC from `docs/external/ConvNeXt-V2/models/fcmae.py` (patchify / unpatchify / gen_random_mask / upsample_mask / forward_loss), adapted for rectangular inputs. See Section 15. |

---

## 1. Motivation

The existing supervised model uses a Hugging Face ConvNeXtV2 encoder loaded through `EncoderLoader`, then adapts its feature map into token sequences through `ConvVisionFrontend`. FCMAE-style continued pretraining should adapt the vision encoder to sheet music before full image-to-`**kern` training, especially once we have many real-world sheet music scans without transcriptions.

`facebook/convnextv2-base-22k-224` is already FCMAE-pretrained and ImageNet-22K fine-tuned. This plan is therefore domain-adaptive continued pretraining, not pretraining from scratch.

Local upstream reference clone:

- `docs/external/ConvNeXt-V2/`
- FCMAE reference: `docs/external/ConvNeXt-V2/models/fcmae.py`
- ConvNeXtV2 reference: `docs/external/ConvNeXt-V2/models/convnextv2.py`

Use the local clone for implementation intent, masking/reconstruction terminology, and architecture details. Do not import *at runtime* from `docs/external/ConvNeXt-V2` — that path is reference material, not a package. Instead, **port specific small functions** (patchify/unpatchify/masking/loss) into `src/pretraining/fcmae/` with a source-file/line-range citation in each function's docstring. See Section 15 for the exact list. This keeps runtime dependencies explicit and avoids pulling in MinkowskiEngine, apex, or timm==0.3.2.

The cleanest design is to treat FCMAE as a separate pretraining task:

- it consumes images only;
- it optimizes reconstruction over masked regions;
- it saves pretraining checkpoints;
- a later export step makes the pretrained encoder consumable by supervised OMR training.

This keeps pretraining experimentation isolated from the already complex autoregressive decoder, grammar validation, and sequence metrics.

---

## 2. Key Concern: Image Size

Using a smaller image size can affect downstream OMR quality because sheet music contains small details: augmentation dots, accidentals, ledger lines, stems, beams, articulations, fingering, and other tiny marks. Aggressive downscaling may teach useful page/layout features while weakening symbol-level visual acuity.

However, full-size `1485 x 1050` FCMAE with ConvNeXtV2-base will be expensive. The first implementation should therefore make image size a config knob rather than a hard-coded assumption.

Recommended first default:

```json
"image_height": 768,
"image_width": 544
```

This preserves a rectangular page-like aspect while keeping smoke tests and early runs practical. For v1, all configured sizes must be divisible by `patch_size=32`. For real pretraining, move toward larger divisible sizes after verifying throughput and memory. Candidate larger settings:

- `896 x 640`
- `1024 x 736`
- `1472 x 1056`

Do not force square ImageNet-style crops into the core design. Sheet music page geometry is meaningful. The current supervised contract, `1485 x 1050`, is not divisible by 32, so using that exact size for FCMAE should wait until the implementation supports internal pad/crop to stride multiples and unpads or masks the reconstruction loss correctly.

---

## 3. Proposed File Layout

Add a new pretraining package:

```text
src/pretraining/
  __init__.py
  fcmae/
    __init__.py
    config.py
    data.py
    masking.py
    model.py
    lightning_module.py
    export.py

scripts/
  pretrain_fcmae.py

config/
  pretrain_fcmae_base.json

tests/
  test_fcmae_smoke.py
```

Rationale:

- `src/pretraining/fcmae/` keeps the task isolated from `src/training/`, which currently means supervised sequence training.
- `scripts/pretrain_fcmae.py` mirrors `train.py` conceptually, but avoids mixing config and execution paths.
- `export.py` can remain small at first or even be a stub until the first checkpoint exists.
- `tests/test_fcmae_smoke.py` gives us a fast guard against shape drift, non-finite loss, and broken image folder loading.

---

## 4. Config Design

Create a small dedicated config model instead of reusing `ExperimentConfig`. FCMAE has no tokenizer, no decoder, no validation text metrics, and no generation settings.

Suggested file: `src/pretraining/fcmae/config.py`

Suggested config sections:

```python
from pydantic import BaseModel, Field, model_validator


class FCMAEDataConfig(BaseModel):
    image_dir: str | None = None
    manifest_path: str | None = None
    image_height: int = 768
    image_width: int = 544
    extensions: list[str] = Field(default_factory=lambda: [".png", ".jpg", ".jpeg", ".webp"])


class FCMAEModelConfig(BaseModel):
    encoder_model_name_or_path: str = "facebook/convnextv2-base-22k-224"
    patch_size: int = 32
    mask_ratio: float = 0.6
    decoder_dim: int = 512
    decoder_depth: int = 2
    norm_pix_loss: bool = True  # Ported from upstream FCMAE; normalizes target patches per-patch before MSE.


class FCMAETrainingConfig(BaseModel):
    batch_size: int = 2
    num_workers: int = 4
    max_steps: int = 1000
    accumulate_grad_batches: int = 4
    # Base LR at effective batch size 256. Actual LR is scaled at runtime:
    #   lr = base_learning_rate * (batch_size * accumulate_grad_batches * world_size) / 256
    # Matches the MAE/FCMAE linear-scaling rule.
    base_learning_rate: float = 1.5e-4
    weight_decay: float = 0.05
    warmup_steps: int = 100
    precision: str = "bf16-mixed"
    seed: int = 0


class FCMAECheckpointConfig(BaseModel):
    dirpath: str = "weights/fcmae"
    filename: str = "fcmae-{step:06d}"
    save_last: bool = True
    save_top_k: int = 0
```

Validation requirements:

- `image_height`, `image_width`, and `patch_size` must be positive.
- `image_height % patch_size == 0`.
- `image_width % patch_size == 0`.
- `0 < mask_ratio < 1`.
- `batch_size`, `num_workers`, `max_steps`, and `accumulate_grad_batches` must be positive, except `num_workers` may be `0`.
- `base_learning_rate` and `weight_decay` must be non-negative; `warmup_steps` must be >= 0.
- exactly one of `image_dir` and `manifest_path` must be set.

Keep the config intentionally smaller than the supervised training config. More knobs can be added after the loop proves itself.

---

## 5. Image Dataset

Suggested file: `src/pretraining/fcmae/data.py`

Responsibilities:

1. Load image paths from either `manifest_path` or by recursively collecting image files from `image_dir`.
2. Load each image as RGB.
3. Preserve aspect ratio while fitting the full image inside the configured rectangular target.
4. Pad remaining area with white pixels.
5. Normalize with the same image convention used by supervised training.
6. Return:

```python
{
    "pixel_values": Tensor,  # (3, H, W)
    "valid_pixel_mask": Tensor,  # bool (H, W), True for original resized content, False for pad
    "valid_patch_mask": Tensor,  # bool (H / patch_size, W / patch_size)
    "path": str,
}
```

Use existing preprocessing where practical:

- `src/data/preprocessing.py` already defines the image normalization convention and supervised page geometry behavior.
- Do not call `preprocess_pil_image(image, image_width, fixed_size=(image_height, image_width))` for arbitrary FCMAE image folders in v1. That helper resizes to fixed width and top-crops overflow, which can silently remove lower systems from real scans.
- Add an FCMAE-specific helper that fits the full image within `(image_height, image_width)`, preserving aspect ratio and padding only. The helper should also return `valid_pixel_mask` so reconstruction loss can ignore padding.

Avoid both direct resize-to-`H x W` distortion and silent crop-by-default. Later runs may add random crop, random scale, or document-scanner augmentations, but those should be explicit training augmentations rather than accidental geometry or content loss.

### Input Sources

Support both input modes from day one:

- `image_dir`: recursively scan for files with configured `extensions`; good for smoke tests and ad hoc local runs.
- `manifest_path`: newline-delimited image paths, optionally relative to the manifest file's parent directory; preferred for large real-world scan corpora on the cluster because it makes the run reproducible and avoids repeated expensive directory walks.

### Valid Masks

`valid_patch_mask` is derived from `valid_pixel_mask` by max-pooling the *invalid* (padding) mask at `patch_size` stride: a patch is valid iff **all** of its pixels are non-padding. No fractional-threshold variants in v1 — this removes a hyperparameter and matches the rectangular-content-on-rectangular-canvas shape of our renders.

The reconstruction loss must not include padded-only patches. Padded whitespace should not become an easy dominant pretraining target.

---

## 6. Dense Masked Image Modeling Model

Suggested file: `src/pretraining/fcmae/model.py`

The production model should:

1. Load `AutoModel.from_pretrained("facebook/convnextv2-base-22k-224")`.
2. Create a patch mask over the image grid.
3. Replace masked input regions with a learned `mask_token` (shape `(1, 3, patch_size, patch_size)` or broadcastable equivalent), initialized via `torch.nn.init.normal_(std=0.02)`. Do not use a constant fill — this is a dense SimMIM-style setup and the learned token is the conventional choice.
4. Run the masked image through ConvNeXtV2.
5. Decode encoder features to patch-vector reconstructions.
6. Compute reconstruction loss only on masked valid patches.

Minimal class shape:

```python
class DenseMaskedImageModelingConvNeXtV2(nn.Module):
    def __init__(
        self,
        config: FCMAEModelConfig,
        encoder: nn.Module | None = None,
        encoder_output_dim: int | None = None,
        encoder_stride: int | None = None,
    ) -> None: ...

    def forward(
        self,
        pixel_values: torch.Tensor,
        valid_patch_mask: torch.Tensor | None = None,
    ) -> MaskedImageModelingOutput:
        ...
```

Class naming note: `DenseMaskedImageModelingConvNeXtV2` reflects what the module actually does — zero-fill masked pixels, run the dense HF encoder, reconstruct with a shallow decoder. It is **not** Meta's sparse FCMAE. A one-paragraph module docstring at the top of `model.py` must say so, cite `docs/external/ConvNeXt-V2/models/fcmae.py`, and list what was ported vs. replaced.

Suggested output:

```python
@dataclass
class MaskedImageModelingOutput:
    loss: torch.Tensor
    pred_patches: torch.Tensor  # (B, grid_h * grid_w, patch_size * patch_size * 3)
    target_patches: torch.Tensor
    mask: torch.Tensor  # bool (B, grid_h, grid_w), selected reconstruction patches (upstream 1=remove convention)
    valid_patch_mask: torch.Tensor | None
    masked_foreground_ratio: torch.Tensor
```

Dependency injection is part of the constructor contract, not a testing afterthought:

- real training passes `encoder=None`, causing the class to load the configured Hugging Face encoder;
- smoke tests pass a tiny dummy encoder;
- `encoder_output_dim` tells the decoder how many channels the encoder emits;
- `encoder_stride` should match the feature-map stride used for mask/reconstruction geometry.

If `encoder` is injected, `encoder_output_dim` and `encoder_stride` must be provided explicitly. This keeps tests offline and avoids hidden `AutoModel.from_pretrained(...)` downloads.

### Patch Size

Use `patch_size=32` by default, matching the effective stride of ConvNeXtV2 feature maps. This makes the decoder naturally reconstruct one patch per final feature-map position.

For an input `(H, W)`, require divisibility by `patch_size` in the first implementation:

```text
H % patch_size == 0
W % patch_size == 0
```

This is one reason to default to `768 x 544`: both are divisible by 32.

### Masking

Suggested file: `src/pretraining/fcmae/masking.py`

Create a random boolean mask of shape:

```text
(B, H / patch_size, W / patch_size)
```

Then upsample or repeat it to image space when masking pixels:

```text
(B, 1, H, W)
```

Keep the mask generator pure and testable:

```python
def random_patch_mask(
    batch_size: int,
    grid_h: int,
    grid_w: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    ...
```

When `valid_patch_mask` is provided, the random mask should sample from valid patches by default. For each sample with `num_valid_patches > 0`, select exactly `k_i = max(1, round(mask_ratio * num_valid_patches))` valid patches; never select invalid patches as reconstruction targets. This per-sample variable count is the authoritative rule — all tests and log metrics derive from it. Guard against degenerate samples:

- if a sample has **zero** valid patches (all-padding image), drop its loss contribution and increment a `train/samples_skipped_no_valid_patches` counter;
- the final loss denominator must be `max(1, masked_valid_patch_count_total)` to avoid NaN.

The Lightning module should log `train/masked_foreground_ratio` or equivalent so we know whether the objective is seeing notation-heavy regions or mostly blank page.

### Decoder

First-pass decoder should stay close to the FCMAE reference and predict one patch vector per encoder-grid position:

- reshape encoder feature map to tokens `(B, grid_h * grid_w, encoder_output_dim)`;
- apply a small MLP or `1x1` token projection stack;
- output `pred_patches` with shape `(B, grid_h * grid_w, patch_size * patch_size * 3)`;
- compare against patchified input pixels with the same shape.

Loss should be MSE or smooth L1 over masked valid patches only. MSE is fine for the smoke loop. A full-resolution ConvTranspose reconstruction path is deferred; it is more memory-heavy and makes rectangular padding semantics easier to get wrong.

Important: this is not claiming exact equivalence to Meta's sparse FCMAE. It is a practical dense pretraining objective inspired by FCMAE.

---

## 7. Lightning Module

Suggested file: `src/pretraining/fcmae/lightning_module.py`

Responsibilities:

- instantiate `DenseMaskedImageModelingConvNeXtV2`;
- implement `training_step`;
- log `train/loss`;
- log `train/mask_ratio`;
- log `train/masked_foreground_ratio` or `train/masked_valid_patch_ratio`;
- log `train/samples_skipped_no_valid_patches`;
- log the resolved effective learning rate (`train/lr`) so the batch-scaling rule is observable;
- configure AdamW with MAE-style LR linear scaling:
  `lr = base_learning_rate * (batch_size * accumulate_grad_batches * world_size) / 256`;
- add warmup + cosine scheduler by default (MAE/FCMAE standard, not optional).

Minimal class:

```python
class FCMAEPretrainer(L.LightningModule):
    def __init__(
        self,
        model_config: FCMAEModelConfig,
        training_config: FCMAETrainingConfig,
    ) -> None: ...
```

Keep it independent from `SMTTrainer`.

---

## 8. Entrypoint

Suggested file: `scripts/pretrain_fcmae.py`

The entrypoint should:

1. Load JSON config.
2. Seed reproducibly.
3. Build image-folder datamodule.
4. Build `FCMAEPretrainer`.
5. Build a basic Lightning `Trainer`.
6. Add `ModelCheckpoint(dirpath=checkpoint.dirpath, filename=checkpoint.filename, save_last=checkpoint.save_last, save_top_k=checkpoint.save_top_k)`.
7. Run `trainer.fit(...)`.

CLI shape can be simple:

```bash
python scripts/pretrain_fcmae.py config/pretrain_fcmae_base.json
```

If cheap, support the same dotlist override style as `train.py` so users can do:

```bash
python scripts/pretrain_fcmae.py config/pretrain_fcmae_base.json training.max_steps=10 data.image_dir=/tmp/fcmae-smoke
```

Do not require W&B in the first implementation. If the existing training setup helpers are easy to reuse without pulling in decoder-specific assumptions, reuse them selectively; otherwise keep this entrypoint lean. The tiny run should still write `weights/fcmae/last.ckpt` by default so the checkpointing path is proven from day one.

---

## 9. Smoke Test

Suggested file: `tests/test_fcmae_smoke.py`

The smoke test should not download remote HF weights. Use dependency injection so the FCMAE model can accept a tiny local/dummy encoder in tests.

The test should:

1. Create a temporary folder with 2-4 synthetic RGB PNGs.
2. Build the image-folder dataset at a small divisible size, for example `128 x 96`.
3. Build the FCMAE module with a tiny dummy encoder, passing explicit `encoder_output_dim` and `encoder_stride`.
4. Run one forward pass with `valid_patch_mask`.
5. Assert:

```python
torch.isfinite(output.loss)
output.pred_patches.shape[-1] == patch_size * patch_size * 3
output.target_patches.shape == output.pred_patches.shape
output.mask.dtype == torch.bool
output.valid_patch_mask is not None
```

Optional but useful:

- run one `training_step`;
- run `loss.backward()`;
- assert at least one trainable parameter receives a finite gradient.

### Real Encoder Shape Probe

Add a separate optional diagnostic check that uses the real Hugging Face encoder. It should be skipped by default when weights are unavailable or network access is disabled.

Suggested file: `scripts/diagnostics/check_fcmae_convnextv2_shape.py`

The check should:

1. Load `facebook/convnextv2-base-22k-224`.
2. Run one no-grad forward on a rectangular divisible input, for example `(1, 3, 768, 544)`.
3. Assert `last_hidden_state.ndim == 4` for this ConvNeXtV2 backbone.
4. Normalize HF output layout into `(B, C, H', W')` if needed.
5. Assert `H' == image_height / 32` and `W' == image_width / 32`.
6. Print the detected output channel count and feature stride.

This probe is not a replacement for offline tests; it catches real-weight shape/layout assumptions before a long pretraining run.

---

## 10. Future Export and Supervised Loading

Do not block the first PR on export if the smoke loop is not yet proven. But design with this handoff in mind.

Preferred later flow:

```bash
python scripts/pretrain_fcmae.py config/pretrain_fcmae_base.json
python scripts/export_fcmae_encoder.py weights/fcmae/last.ckpt weights/fcmae/exported_encoder
python train.py config/train.json model.encoder_model_name_or_path=weights/fcmae/exported_encoder
```

**Export format is decided (D7)**: the export step calls `encoder.save_pretrained(exported_encoder_dir)` on the HF encoder wrapped inside the pretraining module. That writes `config.json` + safetensors, which `EncoderLoader._load_transformers` at `src/model/encoder.py:65` already loads unchanged — no new `encoder_provider` path, no state-dict shim. The first integration is still explicit and opt-in (just point `encoder_model_name_or_path` at the local dir); the default supervised encoder path is unchanged.

---

## 11. Implementation Order

0. **License check for Section 15 port.** Confirm CC BY-NC 4.0 compatibility and add attribution file. Blocker for steps 3-4.
1. Create `src/pretraining/fcmae/config.py`.
2. Create image-folder dataset and datamodule in `data.py`.
3. Create `masking.py` with the four ported helpers (see Section 15) plus unit tests.
4. Create dense masked-image-modeling model in `model.py` as `DenseMaskedImageModelingConvNeXtV2`, using the ported `_reconstruction_loss`.
5. Create Lightning wrapper in `lightning_module.py` with MAE LR scaling and warmup+cosine scheduler.
6. Add smoke test + ported-function unit tests.
7. Run smoke test (gate before entrypoint wiring).
8. Create `scripts/pretrain_fcmae.py`.
9. Add `config/pretrain_fcmae_base.json`.
10. Add optional real-encoder shape diagnostic.
11. Run a tiny local fit on a temporary image folder; confirm `weights/fcmae/last.ckpt` is written.
12. Stub export script (`scripts/export_fcmae_encoder.py`) — calls `encoder.save_pretrained(dir)` and nothing else.

---

## 12. Acceptance Criteria

The first implementation is complete when:

- `pytest tests/test_fcmae_smoke.py` passes;
- a tiny local run can complete at least one optimizer step;
- the tiny run writes `weights/fcmae/last.ckpt` or a configured equivalent;
- the FCMAE package does not import tokenizer, grammar, generation, or supervised validation code;
- existing supervised training imports still work;
- image size, mask ratio, and encoder name are configurable;
- preprocessing preserves full image content by default and returns valid pixel/patch masks;
- masked reconstruction loss excludes padded-only patches;
- the model can run with a dummy encoder in tests and the real HF encoder in actual pretraining;
- the optional real-encoder shape probe passes in an environment with Hugging Face weights available;
- ported functions from Section 15 each carry the required attribution header and pass their dedicated unit tests;
- the Section 15 license compatibility check is resolved (either attribution added, or code reimplemented from the paper);
- `norm_pix_loss` is wired through and defaults to `True`;
- LR linear-scaling rule is implemented and the resolved `train/lr` is logged;
- `train/samples_skipped_no_valid_patches` is logged and remains `0` for the normal smoke-test fixture.

---

## 13. Deferred Work

- exact sparse FCMAE reproduction with `MinkowskiEngine`;
- W&B reconstruction image logging;
- validation split and validation reconstruction metrics;
- Slurm script for real pretraining;
- supervised encoder loading from FCMAE checkpoint/export at scale (the export stub from Section 10 is in v1; wiring it into a real supervised run is deferred);
- real-world scan augmentations;
- multi-resolution pretraining schedule;
- full-size `1485 x 1050` performance/memory profiling;
- staff-line-aware contiguous mask spans (Open Question 3).

---

## 14. Open Questions for Later

These should not block the first smoke-testable implementation:

1. Should real FCMAE pretraining mix global page views and local high-resolution crops?
2. Should reconstruction loss be computed in normalized pixel space, raw pixel space, or a perceptual/edge-enhanced space?
3. Should masks be random patches only, or include staff-line-aware contiguous spans?
4. Should exported encoder weights include any normalization-stat metadata from pretraining?
5. Should supervised fine-tuning freeze early ConvNeXt stages for the first few thousand steps after FCMAE import?

---

## 15. Borrowed Code From Upstream FCMAE

We port a small set of functions from `docs/external/ConvNeXt-V2/models/fcmae.py` rather than reinvent them. This section is the contract: what moves, where it lands, what changes, what stays.

### Why port instead of depend

- upstream requires `MinkowskiEngine` + `apex` + `timm==0.3.2` + Python 3.8 (see `docs/external/ConvNeXt-V2/INSTALL.md`). All three are operationally painful on `antemurale`.
- upstream's `patchify` asserts square inputs (`fcmae.py:97`); sheet music is rectangular.
- upstream trains from scratch into a custom `SparseConvNeXtV2`; we want continued pretraining from HF `facebook/convnextv2-base-22k-224` and export back to HF format with no state-dict remapping.

### Functions to port

Each function below lands in the named local file with (a) a module docstring header citing the upstream file and commit hash, and (b) a per-function docstring with a `Ported from: docs/external/ConvNeXt-V2/models/fcmae.py:<lines>` line.

| Upstream (line range) | Lands in | Adaptation |
| --- | --- | --- |
| `FCMAE.patchify` (`fcmae.py:91-103`) | `src/pretraining/fcmae/masking.py` as `patchify(imgs, patch_size)` | Drop the square assertion. Use `h = H // p, w = W // p` separately. Assert `H % p == 0 and W % p == 0`. |
| `FCMAE.unpatchify` (`fcmae.py:105-117`) | `src/pretraining/fcmae/masking.py` as `unpatchify(x, grid_h, grid_w, patch_size)` | Take `grid_h, grid_w` as explicit args instead of deriving from `sqrt(L)`. |
| `FCMAE.gen_random_mask` (`fcmae.py:119-135`) | `src/pretraining/fcmae/masking.py` as `random_patch_mask(batch_size, grid_h, grid_w, mask_ratio, device, valid_patch_mask=None)` | Generalize `L = grid_h * grid_w`. Add `valid_patch_mask` support: (a) assign `+inf` noise to invalid positions so they sort last; **and** (b) cap the per-sample selection count at `k_i = max(1, round(mask_ratio * num_valid_patches_i))` per Section 6 — the +inf trick alone is insufficient when `round(mask_ratio * L) > num_valid_patches`, which would otherwise select invalid positions as "least-bad" picks. Apply the degenerate-sample guards from Section 6. |
| `FCMAE.upsample_mask` (`fcmae.py:137-142`) | `src/pretraining/fcmae/masking.py` as `upsample_mask(mask, scale)` | Accept `(B, grid_h, grid_w)` directly instead of flat `(B, L)`; drop the square reshape. |
| `FCMAE.forward_loss` (`fcmae.py:164-184`) | `src/pretraining/fcmae/model.py`, private method `_reconstruction_loss` | Keep `norm_pix_loss` verbatim. Replace the denominator `mask.sum()` with `max(1, (mask * valid_patch_mask).sum())`. Multiply `loss * mask * valid_patch_mask` instead of `loss * mask`. Keep the upstream `0 = keep, 1 = remove` mask convention throughout our module so ported code stays readable. |

### What we do **not** port

- `SparseConvNeXtV2` and the `Block` decoder implementation (`models/convnextv2.py`, `models/convnextv2_sparse.py`). Replace with: HF `AutoModel.from_pretrained(...)` for the encoder; `transformers.models.convnextv2.modeling_convnextv2.ConvNextV2Layer` instances for the decoder blocks. Shape is still `1x1 proj → N × Layer → 1x1 pred`.

  Decoder `ConvNextV2Layer` config source: construct a fresh `transformers.ConvNextV2Config` locally inside `model.py` with `hidden_sizes=[decoder_dim]`, `depths=[decoder_depth]`, `drop_path_rate=0.0`, and otherwise the HF defaults (`hidden_act="gelu"`, `layer_scale_init_value=1e-6`). Do not reuse the encoder's config — its `hidden_sizes`, `depths`, and `drop_path_rate` describe the encoder, not the decoder, and mutating it in place would corrupt the encoder's saved config on export. Pass `dim=decoder_dim` and `drop_path=0.0` per-layer.
- `_init_weights` — the Minkowski-specific branches are dead weight here; HF weights are already initialized, the decoder uses Kaiming/LayerNorm defaults, and the `mask_token` gets `torch.nn.init.normal_(std=0.02)` directly.
- `main_pretrain.py` / `engine_pretrain.py` / upstream `optim_factory.py` / `submitit_pretrain.py`. Replaced by our Lightning module, entrypoint, and (eventually) Slurm wrapper.
- `datasets.py` — we have our own rectangular, aspect-preserving dataset in Section 5.

### License and attribution

Upstream `LICENSE` (in `docs/external/ConvNeXt-V2/LICENSE`) is the Attribution-NonCommercial 4.0 International license (CC BY-NC 4.0) from Meta. Before porting:

1. Confirm that license is compatible with this repo's license and intended use. **This is a blocker — do not port code until confirmed.** If incompatible, reimplement the math from the FCMAE paper without copying the code verbatim.
2. If compatible, each ported function carries a header comment: `# Ported from facebookresearch/ConvNeXt-V2, fcmae.py:<lines>, commit <hash>. Licensed under CC BY-NC 4.0.`
3. Add a `NOTICE` or `THIRD_PARTY_LICENSES` entry at repo root if one does not already exist, listing the upstream repo, commit, and license.

### Test coverage for ported code

Because these functions are small and pure, add unit tests in `tests/test_fcmae_smoke.py` (or a sibling `tests/test_fcmae_masking.py`) that are independent of the encoder:

- `patchify(unpatchify(x)) == x` round-trip on random rectangular input.
- `random_patch_mask` with `mask_ratio=0.6` returns exactly `round(0.6 * grid_h * grid_w)` masked positions per sample.
- `random_patch_mask` with a `valid_patch_mask` that has only `k` valid positions never selects an invalid one, and selects exactly `max(1, round(0.6*k))` positions when `k > 0` (per the Section 6 rule).
- `upsample_mask` shape is `(B, grid_h * scale, grid_w * scale)` for rectangular inputs.
- `_reconstruction_loss` with `norm_pix_loss=True` produces a finite scalar when half the patches are invalid.
