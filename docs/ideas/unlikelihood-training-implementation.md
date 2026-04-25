# Unlikelihood Training — Implementation Plan

## Purpose

Add **token-level unlikelihood (UL) loss** to the synthetic training loop to reduce the
decoder's propensity to emit repetitive loops at inference on the polish real-score
dataset.

Primary reference: Welleck et al., 2019, "Neural Text Generation With Unlikelihood
Training."

Mechanism in one line: on top of the existing NLL, penalize the model for putting
probability mass on tokens that already appeared earlier in the same target sequence.

### Non-goals for v1

- Sequence-level UL (requires in-loop generation during training; defer to v2).
- Any change to inference (`scripts/inference.py`).
- Any change to the dataset pipeline or augmentation stack.
- Any interaction with CTC aux loss beyond "losses sum." The two are independently
  toggled and compose cleanly.

---

## 1. Pre-committed decisions

These are locked. Do not re-debate during implementation.

1. **Token-level only.** Sequence-level UL (Welleck §3.2) is out of scope.
2. **Context set `C_t` = UNIQUE non-pad tokens in `labels[b, 0:t]` minus `labels[b, t]`
   minus `{pad, bos, eos}`.** Unique, not frequency-weighted; stable across sequence
   lengths.
3. **`ignore_index = -100`.** This matches `self.model.loss` (set at
   `src/model/modeling_smt.py:796`). Positions where `labels == -100` contribute neither
   to the presence mask nor to the denominator.
4. **Special-token exclusion on by default** (`pad_token_id`, `bos_token_id`,
   `eos_token_id`). These are valid vocab ids distinct from `-100`.
5. **Loss location: Lightning `training_step`, NOT inside `model.forward`.** Mirrors the
   CTC plan. Model stays agnostic to training regularizers. Runs outside the
   `torch.compile` boundary; UL ops stay eager.
6. **Use the existing NLL unchanged.** `outputs.loss` already carries
   label-smoothed CE. We read `outputs.logits` for the UL computation; no double
   softmax.
7. **fp32 cast for `log_softmax`.** Mandatory under AMP/bf16 — `log1p(-p)` goes NaN
   otherwise.
8. **Clamp `p.clamp(max=1 - 1e-6)` before `log1p(-p)`.** 1e-6 is deliberate; smaller
   values lose precision in downstream ops, larger values under-penalize.
9. **No interaction with label smoothing.** LS stays on NLL; UL acts independently on
   raw logits via its own log-softmax. Keep `label_smoothing=0.1` in v1.
10. **Default `alpha = 1.0`.** Per reference paper. Sweep order if needed: `{1.0, 0.5,
    2.0}`.
11. **No warmup in v1.** Expose a `unlikelihood_warmup_steps` knob defaulting to `0`.
    Enable only if stability issues appear.
12. **No context window cap in v1.** Full sequence. Expose `context_window` knob (None
    = full), default None. Window fallback documented in pitfalls.
13. **Config lives in `Training` pydantic model and `config/train.json`.** NOT in
    `SMTConfig`; UL is training-only and not part of model state.
14. **Artifact captures `unlikelihood_*` fields in `experiment_config["training"]`.**
    Falls out automatically from pydantic serialization; verify with round-trip test.
15. **Metrics logged: `train/loss_nll`, `train/loss_ul`, `train/ul_alpha`; mirrored on
    validation.** `TRAIN_LOSS` continues to report the summed total for
    back-compat.

---

## 2. Background

### Definition

Given targets `y_1, ..., y_T` and context `x` (image features), for each position `t`:

```
C_t = ({ unique v : v ∈ y_{<t} } \ { y_t }) \ { pad, bos, eos }
```

Token-level UL loss at position `t`:

```
L_UL(t) = -Σ_{c ∈ C_t}  log(1 - p_θ(c | y_{<t}, x))
```

Total training loss:

```
L = L_NLL + α · L_UL    ( + β · L_CTC if CTC aux is also enabled )
```

### Why this attacks the observed failure mode

The encoder-ablation diagnostic (prefix 220, polish loops) showed median KL(zeroed) <
KL(shuffled) and 71% argmax agreement with encoder zeroed — i.e. once a loop establishes,
the decoder's next-token distribution is dominated by the LM prior over recent tokens.

- NLL pushes mass onto `y_t`. It has no direct pressure against mass on tokens already in
  the prefix.
- UL explicitly pushes mass OFF previously-seen non-target tokens.
- Net effect on the decoder's attractor landscape: narrower basins around loop states,
  faster escape velocity when inference-time drift pushes state near a basin.

Applied to synthetic training, UL generalizes to polish because it's a regularizer on
decoder output distributions, not on input distributions.

### Interaction with CTC (if both enabled)

Orthogonal. Losses add. No tensor is shared. The only joint consideration is gradient
norm: with NLL + α·UL + β·CTC, peak grad norm may grow by 20–30%. Monitor
`gradient_clip_val` (currently `1.0`) and raise if clipping fires much more often.

---

## 3. File-by-file change list

### 3.1 `src/config.py` — `Training` pydantic model (line 42)

Add after the `label_smoothing` field (line 130):

```python
# Unlikelihood training (Welleck et al. 2019, token-level only).
# Penalizes probability mass on tokens that already appeared in the prefix.
unlikelihood_enabled: bool = False
unlikelihood_alpha: float = 1.0
unlikelihood_warmup_steps: int = 0
unlikelihood_context_window: int | None = None  # None = full prefix
unlikelihood_exclude_special_tokens: bool = True  # exclude pad/bos/eos
```

Defaults are **off**; existing runs are bit-identical.

### 3.2 `config/train.json`

Add under `"training"`:

```json
"unlikelihood_enabled": false,
"unlikelihood_alpha": 1.0,
"unlikelihood_warmup_steps": 0,
"unlikelihood_context_window": null,
"unlikelihood_exclude_special_tokens": true
```

### 3.3 New file: `src/training/unlikelihood.py`

```python
"""Token-level unlikelihood loss (Welleck et al. 2019).

Consumed by SMTTrainer.training_step. Must run OUTSIDE torch.compile — do not import
from modeling_smt.py.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


def build_context_presence_mask(
    labels: torch.Tensor,
    vocab_size: int,
    ignore_index: int = -100,
    exclude_ids: Iterable[int] | None = None,
    context_window: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the UL candidate mask.

    Returns
    -------
    presence : (B, T, V) bool tensor.
        presence[b, t, v] = True iff v is a UL candidate at (b, t):
            - v appeared at some position s in labels[b, :t] with labels[b, s] != ignore_index
            - v != labels[b, t]
            - v not in exclude_ids
            - labels[b, t] != ignore_index (else presence row is all False)
    valid_positions : (B, T) bool.
        True where labels[b, t] != ignore_index.
    """
    B, T = labels.shape
    device = labels.device
    valid_positions = labels.ne(ignore_index)  # (B, T)

    # Safe indices for one_hot: -100 -> 0 (masked out by valid_positions below).
    safe_labels = labels.clamp(min=0)
    # int16 keeps memory manageable; max possible count per (b, v) = T, which fits.
    one_hot = F.one_hot(safe_labels, num_classes=vocab_size).to(torch.int16)  # (B, T, V)
    # Zero-out contributions from ignored positions so they don't pollute the cumsum.
    one_hot = one_hot * valid_positions.unsqueeze(-1).to(torch.int16)

    if context_window is None:
        # Strict prefix: positions 0..t-1.
        cum = one_hot.cumsum(dim=1) - one_hot  # (B, T, V)
    else:
        # Prefix window [max(0, t - W), t - 1].
        cum_full = one_hot.cumsum(dim=1) - one_hot  # strict prefix
        # Shift cum_full forward by `context_window` to subtract away tokens that
        # fell out of the window.
        W = int(context_window)
        cum_shifted = F.pad(cum_full, (0, 0, W, 0), value=0)[:, :T, :]
        cum = cum_full - cum_shifted

    presence = cum > 0

    # Exclude current target at each position (use safe_labels; invalid positions
    # get gated by valid_positions at the caller).
    current_one_hot = F.one_hot(safe_labels, num_classes=vocab_size).bool()
    presence = presence & ~current_one_hot

    # Exclude specified special ids.
    if exclude_ids:
        ignore_vocab = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        for i in exclude_ids:
            if 0 <= int(i) < vocab_size:
                ignore_vocab[int(i)] = True
        presence = presence & ~ignore_vocab.view(1, 1, vocab_size)

    # Invalid positions contribute nothing.
    presence = presence & valid_positions.unsqueeze(-1)

    return presence, valid_positions


def compute_token_unlikelihood_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    exclude_ids: Iterable[int] | None = None,
    context_window: int | None = None,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Token-level unlikelihood loss.

    Parameters
    ----------
    logits : (B, T, V). Raw (pre-softmax). Any dtype supported by AMP.
    labels : (B, T). Integer targets. `ignore_index` marks positions to skip.
    ignore_index : int. Matches CrossEntropyLoss ignore_index; default -100.
    exclude_ids : iterable of vocab ids to exclude as UL candidates (e.g.
        {pad_id, bos_id, eos_id}).
    context_window : None for full prefix, else an int window size.
    epsilon : clamp margin below 1.0 for numerical stability.

    Returns
    -------
    scalar torch.Tensor — mean UL loss over valid positions.
    """
    B, T, V = logits.shape

    with torch.no_grad():
        presence, valid_positions = build_context_presence_mask(
            labels=labels,
            vocab_size=V,
            ignore_index=ignore_index,
            exclude_ids=exclude_ids,
            context_window=context_window,
        )

    # fp32 is mandatory under AMP. Do not skip this cast.
    log_probs = F.log_softmax(logits.float(), dim=-1)  # (B, T, V)
    probs = log_probs.exp()
    one_minus_p_log = torch.log1p(-probs.clamp(max=1.0 - epsilon))  # (B, T, V)

    # -sum over candidate vocab positions
    ul_per_pos = -(one_minus_p_log * presence.to(one_minus_p_log.dtype)).sum(dim=-1)
    # Mask positions where labels == ignore_index
    ul_per_pos = ul_per_pos * valid_positions.to(ul_per_pos.dtype)

    denom = valid_positions.sum().clamp(min=1).to(ul_per_pos.dtype)
    return ul_per_pos.sum() / denom
```

**Memory budget.** For `B=8, T=2048, V≈3000` the largest tensors are:

- `one_hot` (int16): 8 · 2048 · 3000 · 2 B ≈ 98 MB
- `cum` (int16): ~98 MB (materialized transiently, freed before log_probs alloc)
- `log_probs` / `probs` / `one_minus_p_log` (fp32, (B, T, V)): ~200 MB each, up to three
  simultaneously live

Peak ~1 GB extra. Acceptable on the training GPU (H100-class). If a smaller GPU OOMs,
drop `context_window` to 256 (same memory) or switch to the gather-based alternative in
§9.

### 3.4 `src/training/lightning_module.py`

**In `SMTTrainer.__init__`** (near line 117 where label_smoothing is wired), after the
label-smoothing block:

```python
# Unlikelihood config cache
self._ul_enabled = training.unlikelihood_enabled
self._ul_alpha = float(training.unlikelihood_alpha)
self._ul_warmup = int(training.unlikelihood_warmup_steps)
self._ul_context_window = training.unlikelihood_context_window
self._ul_exclude_ids: set[int] = set()
if training.unlikelihood_exclude_special_tokens:
    # Resolve ids — these are the three vocab ids, distinct from ignore_index=-100.
    self._ul_exclude_ids = {
        int(self.pad_token_id),
        int(self.bos_token_id),
        int(self.eos_token_id),
    }
```

(Confirm `self.pad_token_id` / `bos_token_id` / `eos_token_id` are already attributes on
SMTTrainer. If not, derive them once from the tokenizer or hparams.)

**Add two helper methods** on `SMTTrainer`:

```python
def _current_ul_alpha(self) -> float:
    if self._ul_warmup <= 0:
        return self._ul_alpha
    progress = min(1.0, float(self.global_step) / float(self._ul_warmup))
    return self._ul_alpha * progress

def _compute_ul(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    from src.training.unlikelihood import compute_token_unlikelihood_loss
    return compute_token_unlikelihood_loss(
        logits=logits,
        labels=labels,
        ignore_index=-100,
        exclude_ids=self._ul_exclude_ids,
        context_window=self._ul_context_window,
    )
```

**In `training_step`** (line 647), replace the current:

```python
loss = outputs.loss
```

with:

```python
nll = outputs.loss
loss = nll
if self._ul_enabled:
    ul = self._compute_ul(outputs.logits, labels)
    alpha = self._current_ul_alpha()
    loss = nll + alpha * ul
    self.log("train/loss_nll", nll, on_step=True, on_epoch=True,
             batch_size=x.size(0), prog_bar=False)
    self.log("train/loss_ul", ul, on_step=True, on_epoch=True,
             batch_size=x.size(0), prog_bar=False)
    self.log("train/ul_alpha", alpha, on_step=True, prog_bar=False)
```

Keep the existing `self.log(TRAIN_LOSS, loss, ...)` line — it now reports the combined
total, which is the right default for early-stopping / checkpoint selection.

**In `validation_step`** (line 684), after `loss = outputs.loss`, mirror the logging:

```python
if self._ul_enabled:
    ul = self._compute_ul(outputs.logits, labels)
    self.log(
        self._validation_set_metric_name(val_set_name, "loss_nll"),
        outputs.loss, on_epoch=True, batch_size=x.size(0),
    )
    self.log(
        self._validation_set_metric_name(val_set_name, "loss_ul"),
        ul, on_epoch=True, batch_size=x.size(0),
    )
```

Do **not** change the primary `val/*/loss` metric — let it continue to report NLL for
comparability across runs with UL on/off. (Alternatively, add a config knob
`validation_loss_includes_ul`; v1: off.)

### 3.5 `src/artifacts.py`

No explicit code change if `RunArtifact` already serializes the `Training` pydantic
model wholesale. Add a round-trip test (see §5 tests) that asserts the five new fields
survive JSON serialization.

### 3.6 `src/model/checkpoint_loader.py`

No changes. UL is training-time only; it is not in model state and not required at
inference.

### 3.7 `scripts/inference.py`

No changes.

---

## 4. Execution order (stages)

Land each stage in its own commit. Do not merge stages.

### Stage 1 — config plumbing (no behavior change)

- Add the five fields to `Training` pydantic model.
- Add the five entries to `config/train.json` with defaults.
- Add a round-trip test that asserts the fields serialize/deserialize via
  `RunArtifact`.
- **Gate:** `pytest tests/` must pass. Full training smoke (5 steps, fixed seed) must
  be bit-identical to pre-change main.

### Stage 2 — loss implementation (no Lightning wiring)

- Add `src/training/unlikelihood.py` with `build_context_presence_mask` and
  `compute_token_unlikelihood_loss`.
- Land unit tests 1–9 (see §5).
- **Gate:** `pytest tests/training/test_unlikelihood.py` passes.

### Stage 3 — Lightning integration

- Wire config into `SMTTrainer.__init__`.
- Wire `_compute_ul` into `training_step` and `validation_step`.
- Land integration test 10.
- **Gate 1:** With `unlikelihood_enabled=false`, `TRAIN_LOSS` over 5 seeded steps is
  bit-identical to pre-change main.
- **Gate 2:** With `unlikelihood_enabled=true`, 5 steps run without NaN/inf and
  `train/loss_ul` appears in the logger.

### Stage 4 — local smoke run

- 500 training steps on a single GPU, small batch, `unlikelihood_enabled=true`,
  `alpha=1.0`.
- Verify:
  - `train/loss_ul` declines from its initial value (typically 5–20) monotonically in
    trend.
  - `train/loss_nll` behaves similarly to a NLL-only run at the same step count.
  - No NaN/inf under AMP.
  - Step time increase ≤ 15% vs NLL-only.

### Stage 5 — antemurale A/B run

- Submit two full training runs: NLL-only baseline (`unlikelihood_enabled=false`) and
  UL (`alpha=1.0`). Same seed, same data, same schedule.
- Checkpoint at matched training steps.

### Stage 6 — evaluation

- On each checkpoint:
  - Synth val CER/SER/WER (existing).
  - Polish val CER/SER (existing).
  - Polish catastrophic-loop rate via `analyze_catastrophic_repetition` from
    `src/core/metrics/runaway_monitor.py`.
- Compare against acceptance criteria (§7).

---

## 5. Unit + integration tests

New file `tests/training/test_unlikelihood.py`:

1. **`test_presence_mask_basic`** — hand-built (1, 5, 10) labels, assert presence
   equals expected boolean tensor.
2. **`test_presence_mask_excludes_current_target`** — when `labels[b, t] = v` and `v`
   also appears at `s < t`, presence at `(b, t, v)` is `False`.
3. **`test_presence_mask_excludes_specials`** — pad/bos/eos never in presence
   regardless of labels.
4. **`test_presence_mask_context_window`** — with W=3 and a token at `t−5` not
   reoccurring in `[t−3, t−1]`, presence at that `(t, v)` is `False`.
5. **`test_presence_mask_respects_ignore_index`** — positions with `labels == -100`
   yield all-False presence rows AND do not contribute to other rows' cumsum.
6. **`test_ul_zero_when_distribution_is_peaked_on_target`** — logits with argmax = target
   and near-zero elsewhere → UL ≈ 0.
7. **`test_ul_positive_when_mass_on_prior_token`** — construct logits with high
   probability on a previously-seen non-target token → UL > some positive threshold.
8. **`test_ul_gradient_flows_to_candidate_logits`** — backward; assert grad at
   `(b, t, v)` for `v ∈ C_t` is nonzero and negative (pushing mass down), grad at
   non-candidate positions is zero.
9. **`test_ul_bf16_stable`** — logits cast to bf16; loss is finite and non-NaN.

New file `tests/training/test_lightning_unlikelihood_integration.py`:

10. **`test_training_step_with_ul_enabled`** — small 2-sample batch, `ul_enabled=True`;
    assert `train/loss_nll` and `train/loss_ul` both logged, total loss > NLL alone,
    `loss.backward()` proceeds.

Also add to an appropriate artifact test file (e.g. `tests/test_artifacts.py`):

11. **`test_artifact_roundtrips_ul_config`** — set `unlikelihood_enabled=True`,
    `alpha=0.7`, etc.; serialize `RunArtifact` to JSON and back; assert fields match.

---

## 6. Hyperparameter recipe

### Default starting point (first UL run)

```
unlikelihood_enabled = true
unlikelihood_alpha = 1.0
unlikelihood_context_window = null          # full prefix
unlikelihood_warmup_steps = 0
unlikelihood_exclude_special_tokens = true
label_smoothing = 0.1                       # unchanged
```

Optimizer, LR, schedule, batch size: **unchanged**.

### Dial tree

- **Synth val CER degrades > 5% vs baseline** → lower `alpha` to 0.5. If still
  degrading, enable `warmup_steps = 2000`.
- **Polish loop rate drops < 30% relative** → raise `alpha` to 2.0. If still
  insufficient, set `context_window = 256` (focuses UL on loop-scale recent prefix).
- **Grad clipping fires on > 20% of steps** → raise `gradient_clip_val` from 1.0 to 1.5
  (UL adds real gradient magnitude, especially early in training).
- **Training NaNs under AMP** (shouldn't happen with the fp32 cast, but if it does) →
  first verify `epsilon = 1e-6` clamp is applied; then set `alpha = 0.5` and
  `warmup_steps = 5000`.

### Sweep order (one full run per point)

`{1.0, 0.5, 2.0}`. If polish loop rate is not meaningfully reduced at any of these, UL
has not worked; move to CTC (per the other plan).

---

## 7. Rollout on antemurale

Standard flow:

1. Land Stages 1–3 on a feature branch; merge to main with UL disabled by default.
2. Push to GitHub; on antemurale: `git pull`.
3. Create experiment config: copy `config/train.json` → `config/train_ul_alpha1.json`,
   flip `unlikelihood_enabled` and set `alpha=1.0`.
4. Submit via repo's existing `./train.sh` interface (baseline + UL, same seed).
5. Monitor in wandb:
   - `train/loss_nll` — should track baseline `train/loss`.
   - `train/loss_ul` — expect initial value in the 5–20 range, declining trend.
   - `val/polish/loss_nll`, `val/polish/loss_ul` — similar shape.

---

## 8. Acceptance criteria

1. `unlikelihood_enabled=false` path is **bit-identical** to pre-change main for 5
   seeded training steps. (No regression when disabled.)
2. `unlikelihood_enabled=true` training runs 5k steps with no NaN/inf in any loss term
   under AMP.
3. `train/loss_nll`, `train/loss_ul`, `train/ul_alpha` all appear in wandb with
   sensible magnitudes.
4. Step-time (`perf/step_time_ms`) regression ≤ 15% vs baseline.
5. Polish catastrophic-loop rate (from `analyze_catastrophic_repetition`) decreases by
   **≥ 30% relative** vs NLL-only baseline at matched synth val CER.
6. Synth val CER does **not** increase by more than 5% relative.
7. A checkpoint saved with UL enabled loads via `load_model_from_checkpoint` and
   produces identical inference output to a non-UL checkpoint given the same weights
   (UL is training-only; no inference path change).
8. Unit tests 1–9, integration test 10, artifact round-trip test 11 all pass.
9. `experiment_config["training"]` in the run artifact contains all five new fields.

If #5 fails even at `alpha = 2.0`, UL has not worked for this workload; proceed to CTC.

---

## 9. Common pitfalls

1. **`ignore_index = -100` vs `pad_token_id`.** Labels use `-100` for padding (see
   `modeling_smt.py:796, 950`). `pad_token_id` is a valid vocab id distinct from -100.
   UL's `ignore_index` argument is -100; `exclude_ids` uses the vocab id. Crossing
   these will either crash one_hot (negative index) or silently corrupt the mask.
   Covered by test 5.
2. **Forgetting to exclude the current target `y_t`.** `**kern` targets contain many
   repeated tokens (tabs, newlines) by design. If you don't exclude `labels[b, t]`
   from `C_t`, you penalize the model for correctly predicting a token that legitimately
   recurs. A single-bit bug that will ruin the run. Covered by test 2.
3. **AMP / bf16 NaN in `log1p(-p)`.** If you skip the `.float()` cast on logits, `p`
   near 1 in bf16 produces -inf in `log1p`. The cast is mandatory; do not remove it
   for perf. Covered by test 9.
4. **Clamp value drift.** Use `epsilon = 1e-6`. Going smaller (1e-8) breaks bf16
   downstream; going larger under-penalizes confident wrong predictions.
5. **Memory at `(B, T, V)`.** The one_hot and cumsum tensors are the dominant allocation.
   Keep them in `int16`; cast only where needed. Don't accidentally promote to fp32 by
   multiplying against a float before the cumsum.
6. **Compile boundary.** UL runs in Lightning step (eager). Do NOT move it into
   `SMTModelForCausalLM.forward` without re-profiling compile — `F.one_hot` at
   `V=3000` plus cumsum along a dynamic-shape dim can blow up `torch.compile` cache
   entries.
7. **Logits detachment.** `outputs.logits` must carry grad for UL to flow gradients to
   the decoder's output projection. Do NOT add `.detach()` anywhere in the UL path.
8. **NLL vs UL denominator mismatch.** Both must aggregate over the same valid
   positions (non-`-100` labels). If you accidentally divide UL by `B*T` instead of
   `valid.sum()`, `alpha = 1.0` no longer corresponds to the paper's scale.
9. **Gradient only reaches the output projection.** UL's gradient path is logits →
   projection → decoder hidden states → decoder layers → cross-attention → encoder.
   The gradient magnitude on the encoder is typically small compared to NLL's.
   This is expected; don't interpret it as a bug.
10. **Validation step logging.** `validation_step` runs under `torch.no_grad()`; UL's
    internals use `with torch.no_grad()` for the mask and don't require grad for the
    scalar. Ensure `self.log(..., on_epoch=True)` works from val — it does by default
    in Lightning, but double-check for the multi-dataloader case (per-set metric
    naming via `_validation_set_metric_name`).
11. **BOS/EOS policy sanity.** The model predicts `[y_1, ..., y_T, EOS]` from input
    `[BOS, y_1, ..., y_{T-1}, EOS]`. The **target at each position** is what goes into
    UL. Use `batch["labels"]` (already shifted by the collator), NOT
    `batch["decoder_input_ids"]`.
12. **Interaction with grammar-constrained decoding at eval.** UL affects training
    only. Evaluating with `use_grammar_constraints=true` vs `false` is orthogonal; run
    both variants on the polish diagnostic to isolate the UL effect from the
    grammar's.
13. **`train/loss` vs `train/loss_nll` when UL is on.** `TRAIN_LOSS` continues to
    report the **combined total** so checkpoint selection still has a single number.
    When comparing UL-on vs baseline runs, compare `train/loss_nll` directly;
    `train/loss` is not apples-to-apples.

### Alternative implementation (if `(B, T, V)` memory is a problem)

Swap the one_hot/cumsum path for gather-based `(B, T, T)`:

```python
# For each (b, t), candidate indices are labels[b, 0:t]; gather log_probs at those.
prev = labels.unsqueeze(1).expand(B, T, T)                # [b, t, s] = labels[b, s]
time = torch.tril(torch.ones(T, T, dtype=torch.bool, device=labels.device),
                  diagonal=-1).unsqueeze(0).expand(B, T, T)
neq  = prev != labels.unsqueeze(2)
nign = prev != ignore_index
# plus exclude_ids mask on prev
cand_mask = time & neq & nign  # (B, T, T)

log_probs = F.log_softmax(logits.float(), dim=-1)         # (B, T, V)
gathered = log_probs.gather(2, prev.clamp(min=0))          # (B, T, T)
one_minus = torch.log1p(-gathered.exp().clamp(max=1 - 1e-6))
ul = -(one_minus * cand_mask.to(one_minus.dtype)).sum(-1)  # (B, T)
```

Memory: `B · T · T · 4 B` vs one_hot's `B · T · V · 2 B`. At V=3000, T=2048, gather
uses ~128 MB per (B=8) fp32 tensor vs one_hot's ~50 MB; but gather has fewer live
fp32 tensors. Pick whichever fits. Do NOT ship both; commit to one in code.

---

## 10. FAQ / scope boundary

**Q: Can we add sequence-level UL in this same PR?**
A: No. Sequence-level requires inline generation during training (expensive, changes
throughput, has its own hyperparameters). Ship token-level first; evaluate; decide
sequence-level separately.

**Q: Can UL replace label smoothing?**
A: Not in v1. Their mechanisms overlap in spirit but not in form. Keep both for v1;
ablate label smoothing only after token-level UL is proven.

**Q: Can UL replace grammar constraints at inference?**
A: No. UL is a soft regularizer applied during training; grammar is a hard constraint
at decode time. They compose.

**Q: What if we want UL and CTC simultaneously?**
A: Fine. Losses add: `L = L_NLL + α·L_UL + β·L_CTC`. Each toggles via its own config
flag. Expect ~20–30% higher peak grad norm; raise `gradient_clip_val` if clipping
fires much more often.

**Q: Should we A/B `alpha` inside a single run via schedule?**
A: No. Confounds attribution. Run separate jobs per alpha.

**Q: Why unique-tokens instead of Welleck's list-with-duplicates?**
A: Unique is more stable across sequence lengths (T up to 2048 here vs ~256–512 in the
paper's LM setting). Frequency-weighted penalty at long T over-penalizes structural
repeats like `\t`. If we ever want to reproduce the paper exactly, the gather
implementation gives list-with-duplicates for free — swap the presence mask for a
count mask.

**Q: Do we need changes to `inference.py` or `checkpoint_loader.py`?**
A: No. UL is training-only.

**Q: Does UL require retokenization or vocab changes?**
A: No. Uses the existing vocab as-is. No special tokens added.

**Q: What's the expected loss magnitude?**
A: Early in training, `L_UL` is typically 5–20 (depending on how diffuse the initial
distribution is). It decays faster than NLL as the model sharpens. By mid-training
expect `L_UL` well below `L_NLL`.

**Q: What if polish loop rate doesn't improve?**
A: Two possibilities. (a) Loop rate is bottlenecked by encoder representation quality
and UL can't help — move to CTC. (b) `alpha` is too low — sweep up. The dial tree in
§6 covers (b); criterion #5 rules on (a).
