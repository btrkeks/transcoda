# CTC Auxiliary Loss — Implementation Plan

Adds a CTC auxiliary head on top of the vision encoder, a blended training loss, and a CTC greedy-decode diagnostic path. Intended to reduce decoder attention-collapse loops on the `polish` real-score split by forcing the encoder to produce directly decodable features.

**Important revision (v2):** the CTC target is a **filtered, content-token-only** subsequence of the decoder target — not the full token stream. This is required because at the fixed `1485×1050` image size the encoder produces `Tenc = 1551` tokens, and roughly 30% of polish targets exceed that length when taken verbatim. Filtering out purely-structural tokens (tabs, newlines, nulls) compresses targets ~2–4× and brings the overwhelming majority of samples under `Tenc`, with a per-sample length mask as safety net for the residual. See D5, D6, D16, D17, and §1.2.

This plan is **implementation-ready**, with explicit measurement gates. If a gate fails
(especially target-length coverage or CTC exclusion rate), do not force the remaining
stages through unchanged; follow the decision points in §3 and §8.

---

## 0. Decisions already made

All decisions below are final. Do not re-open them in PR review unless you have concrete evidence the decision fails.

| # | Decision | Value |
| --- | --- | --- |
| D1 | Blank-token handling | Reserved slot at index `out_categories` in the CTC head only. Not added to tokenizer, not added to decoder vocab. |
| D2 | CTC head input | `encoder_features.encoder_tokens_raw` (pre-positional, post-projector). Shape: `(B, Tenc, d_model)`. |
| D3 | CTC head architecture | Single `nn.Linear(d_model, out_categories + 1)`. No non-linearity, no dropout. |
| D4 | CTC input "time" axis | Row-major flattened encoder grid (already computed by the frontend). For a 1485×1050 image with stride-32 ConvNeXtV2-tiny, `Tenc ≈ 47 × 33 = 1551`. Do not re-flatten column-major. Do not height-pool. |
| D5 | Target preparation | Strip (a) `<bos>`, `<eos>`, and positions where `labels == -100`; **then** (b) apply the content-token filter (D16). The resulting sequence is the CTC target. |
| D6 | Samples where `filtered_target_len + adjacent_dupe_count > Tenc` | Exclude from the CTC loss for that batch (mask out, average over surviving samples only). Log both the count and the fraction excluded per step. With the D16 filter this should be < 5% of synthetic training samples; > 10% indicates a filter misconfiguration. **`adjacent_dupe_count`** is the number of positions in the *filtered* target where `target[i] == target[i-1]`; CTC requires a blank between adjacent identical symbols, so each such pair consumes one extra input frame. |
| D7 | Loss blending | `total_loss = ce_loss + ctc_weight(step) * ctc_loss`, where `ctc_weight(step) = ctc_aux_weight` if `ctc_aux_warmup_steps == 0`, otherwise `ctc_aux_weight * min(1, step / ctc_aux_warmup_steps)`. Default `ctc_aux_weight = 0.3`. |
| D8 | Warmup on CTC weight | Linear ramp over `ctc_aux_warmup_steps`. Default config value `0` (no warmup). **Recommended value for the first real run: `2000`.** CTC loss at step 0 on a near-random encoder is astronomically high; `zero_infinity=True` prevents NaN from overflow-to-inf but does not prevent the large *finite* gradient spike before clamping fires. Warmup sidesteps this. |
| D9 | Where CTC is computed | Logits inside `SMTModelForCausalLM.forward` (compiles well), loss inside `SMTTrainer.training_step` (kept out of compiled boundary because `F.ctc_loss` interacts poorly with `max-autotune-no-cudagraphs`). |
| D10 | Numerical safety | `nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)`. CTC head logits and log-softmax computed in **fp32** regardless of AMP. |
| D11 | Validation metric | Separate `val/<set>/CER_ctc_content` and `val/<set>/ctc_loss` metrics using CTC greedy-decoded output and raw CTC loss respectively. CTC targets are filtered content-only targets, so the CTC CER must compare against the same filtered target stream, not the full decoder target with tabs/newlines/dots. Attention remains the primary metric; CTC is a diagnostic. **Primary `val/<set>/loss` stays as NLL-only** — do not add CTC into it, to preserve comparability with pre-CTC baselines and keep checkpoint selection on a stable metric. |
| D12 | Inference | CTC decoder is **not** the default inference path. Attention decoder remains primary. A `scripts/inference_ctc.py` diagnostic entrypoint is added for side-by-side comparison. |
| D13 | Vocab projection sharing | Do **not** share weights between CTC head and decoder `vocab_projection`. Separate heads learn independently. |
| D14 | Config surface | New fields on `Training`: `ctc_aux_enabled`, `ctc_aux_weight`, `ctc_aux_warmup_steps`, `ctc_target_exclude_patterns`. Persisted through `RunArtifact.experiment_config["training"]` so checkpoints remember their CTC configuration; do not refer to a nonexistent `artifact.training` typed field. |
| D15 | Backward compatibility | When `ctc_aux_enabled=false` (default), the model builds no CTC head and has zero runtime overhead. Existing (pre-CTC) checkpoints load without changes. **CTC-trained checkpoints require `use_ctc_aux=True` in the rebuilt `SMTConfig` to avoid strict-load failures on the extra `ctc_head.*` keys.** `load_model_from_checkpoint` is the single authoritative load path and reads `use_ctc_aux` from the run artifact (§2.7). Anyone loading a state_dict manually outside that path must either match the artifact config or pass `strict=False`. |
| D16 | Content-token filter | A filter function removes purely-structural tokens (tab, newline, null-continuation) from CTC targets. Filter is built at Lightning `__init__` from a configurable **whitelist of decoded-string patterns** (D16a) by iterating the tokenizer vocabulary and matching each token's decoded string. Starting set: `["\t", "\n", "."]`. Filter is applied before the length check in D6. |
| D16a | Filter pattern matching rule | A token is filtered if its **decoded string (via `tokenizer.decode([token_id])`) after stripping surrounding whitespace** is exactly equal to one of the configured patterns, OR is an empty string. BPE tokens that mix structural and content characters (e.g. a hypothetical `4e\t` token) are **kept** — they carry content. |
| D17 | CTC head vocab | Unchanged at `out_categories + 1`. No vocab remapping between full decoder vocab and CTC target space. Filtered structural ids simply never appear as CTC targets; the head learns to route blanks through their corresponding encoder columns. This keeps the implementation simple at the cost of a few hundred "dead" logits. |
| D18 | Optimizer group | `ctc_head` parameters must be explicitly added to `configure_optimizers()` as their own full-LR group. They are not part of `frontend.encoder`, `frontend.projector`, or `decoder`, so they will not train unless the optimizer group is added. |

---

## 1. Background (read before touching code)

### 1.1 What CTC needs

`nn.CTCLoss` requires:
- `log_probs`: `(T, N, C)` — input length T, batch N, class count C (including blank).
- `targets`: `(N, S)` or concatenated `(sum(target_lengths),)`.
- `input_lengths`: `(N,)`, one per sample.
- `target_lengths`: `(N,)`.

Key constraint: for every sample, the real CTC minimum is

```
input_length >= target_length + adjacent_duplicate_count(target)
```

where `adjacent_duplicate_count` is the number of positions `i` with `target[i] == target[i-1]`. Each such pair requires a blank frame between the two identical emissions in any valid alignment. PyTorch's `F.ctc_loss` returns `inf` when this fails, which `zero_infinity=True` then silently zeros — so violations are invisible unless you check explicitly. We compute this per sample on the filtered target and mask out violators (D6). Do **not** rely on the simpler `input_length >= target_length` check — adjacent dupes do occur on content-filtered kern targets (e.g. a pitch that repeats across a barline after tabs are stripped).

### 1.2 How the encoder shape feeds CTC

`ConvVisionFrontend.forward` (`src/model/frontends/conv_frontend.py:25`) produces:

- `encoder_tokens_raw`: `(B, H*W, d_model)` — row-major flatten of the `(B, C, H, W)` feature map. This is `Tenc = H*W`.
- `encoder_attention_mask`: `(B, H*W)` boolean — `True` = valid, `False` = padding. Use `mask.sum(dim=-1)` for `input_lengths`.

At the fixed `1485×1050` input size with ConvNeXtV2-tiny (cumulative stride 32), `Tenc = 47 * 33 = 1551`.

### 1.2.1 Target-length coverage and why we filter

Roughly 30% of polish targets exceed `Tenc = 1551` tokens when taken verbatim from the decoder labels. Because `nn.CTCLoss` requires `input_length >= target_length` for any valid alignment, any sample that violates this has its CTC loss silently zeroed by `zero_infinity=True` — which means the CTC signal is lost precisely on the longest, hardest samples (the ones most prone to looping).

**`**kern` targets are structurally repetitive.** The token stream contains tabs (`\t`, spine separators), newlines (`\n`, record separators) and dots (`.`, rhythmic-continuation nulls) at a density that typically accounts for 50–75% of all tokens. These tokens carry no per-column image content: they are layout artifacts of the grid-based encoding. A CTC head that tries to align them to encoder columns is solving a layout-prediction problem, not a vision-grounding problem.

**Consequence.** The CTC target is the decoder target *with* structural tokens filtered out (D5 + D16). Expected compression on the synthetic training set is 2–4×, which puts the overwhelming majority of samples well under `Tenc`. The residual (very long content-only targets) is still excluded by D6 as a safety net, but this should now be a rare event rather than a systematic 30% loss.

**Alignment validity after filtering.** Filtering tabs/newlines/dots does not break CTC's monotonicity requirement: the remaining content tokens (pitches, durations, interpretations, barlines, etc.) still appear in monotonic left-to-right, top-to-bottom spatial order on the page. CTC alignment emits blank at encoder columns corresponding to filtered structural regions, which is exactly the right behavior.

**Why not just make Tenc bigger?** Doubling `Tenc` via larger input or finer encoder
stride requires retraining from scratch and substantially increases decoder
cross-attention cost. Deferred (see §8).

### 1.3 Reading the existing codebase

- Model forward: `src/model/modeling_smt.py:904`. Already stashes `output["encoder_outputs"]` at line 985 — we will add `output["ctc_logits"]` and `output["ctc_input_lengths"]` there.
- Lightning training step: `src/training/lightning_module.py:647`. Reads `outputs.loss` — we extend to also read `outputs.ctc_logits` when present.
- Config: `src/config.py:42`. Add fields to `Training` (keep them at the bottom of the class).
- Model config: `src/model/configuration_smt.py` (66 lines — inspect before editing). Add `use_ctc_aux: bool` and propagate through `load_model_from_checkpoint`.
- Collator: `src/data/collators.py`. Emits `labels` padded with `-100`. We reuse that directly; no collator changes.

---

## 2. Change list by file

Each section is prescriptive. Follow it in the order given.

### 2.1 `src/config.py`

Add these fields at the bottom of class `Training` (before `validate_runaway_guard`),
with validation:

```python
# CTC auxiliary loss on encoder features. Helps prevent decoder
# attention collapse by forcing encoder features to be directly decodable.
ctc_aux_enabled: bool = False
ctc_aux_weight: float = 0.3
# Linear ramp-up over this many steps. 0 = no warmup. Recommended 2000 for
# the first real run (D8).
ctc_aux_warmup_steps: int = 0
# Decoded-string patterns whose matching vocab tokens are stripped from CTC
# targets (D16). See plan §1.2.1 for rationale. Default strips tabs,
# newlines, and kern null-continuation dots.
ctc_target_exclude_patterns: list[str] = Field(default_factory=lambda: ["\t", "\n", "."])
```

Add to `validate_runaway_guard` (rename is fine if the engineer prefers, but keep it cohesive — append after existing checks):

```python
if self.ctc_aux_enabled and self.ctc_aux_weight <= 0:
    raise ValueError("training.ctc_aux_weight must be > 0 when ctc_aux_enabled=true")
if self.ctc_aux_enabled and self.ctc_aux_warmup_steps < 0:
    raise ValueError("training.ctc_aux_warmup_steps must be >= 0")
if self.ctc_aux_enabled and not isinstance(self.ctc_target_exclude_patterns, list):
    raise ValueError("training.ctc_target_exclude_patterns must be a list of strings")
```

### 2.2 `src/model/configuration_smt.py`

Add two fields to `SMTConfig`:

```python
use_ctc_aux: bool = False
ctc_blank_token_id: int | None = None  # Resolved to out_categories at model-build time.
```

In the constructor (or the `__init__`/`@dataclass` body, whichever is used), default `ctc_blank_token_id` to `out_categories` when `use_ctc_aux=True`:

```python
if self.use_ctc_aux and self.ctc_blank_token_id is None:
    self.ctc_blank_token_id = int(self.out_categories)
```

### 2.3 `src/model/modeling_smt.py`

**Three edits.**

**(a)** In `SMTModelForCausalLM.__init__` (after the decoder is built at line 785–794), add:

```python
self.ctc_head: nn.Linear | None = None
if bool(getattr(config, "use_ctc_aux", False)):
    # +1 for the blank token at index out_categories
    self.ctc_head = nn.Linear(config.d_model, config.out_categories + 1)
    # Zero-init the blank logit bias so early training doesn't collapse to blank.
    nn.init.zeros_(self.ctc_head.bias)
```

**(b)** In `SMTModelForCausalLM.forward`, immediately after the existing line
```python
output["encoder_outputs"] = encoder_features
output["encoder_attention_mask"] = memory_key_padding_mask
```
(at line 985), add:

```python
if self.ctc_head is not None and labels is not None:
    # CTC logits computed in fp32 for numerical stability under AMP.
    # Only emit these in teacher-forced train/validation forwards. Generation should
    # not pay the CTC-head cost on every decode step.
    raw_tokens = encoder_features.encoder_tokens_raw
    device_type = raw_tokens.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        ctc_logits = self.ctc_head(raw_tokens.to(torch.float32))
    output["ctc_logits"] = ctc_logits  # (B, Tenc, C+1)

    enc_mask = encoder_features.encoder_attention_mask
    if enc_mask is None:
        # All-valid fallback for backward compatibility.
        enc_mask = torch.ones(
            ctc_logits.shape[:2], dtype=torch.bool, device=ctc_logits.device
        )
    output["ctc_input_lengths"] = enc_mask.sum(dim=-1).to(torch.long)
```

**(c)** No changes to generation code paths. Because CTC logits are emitted only when
`labels is not None`, the CTC head is inert during `.generate()`.

### 2.4 New file: `src/training/ctc_loss.py`

Create this file. It encapsulates target preparation and loss computation — keeps `training_step` readable and lets us unit-test in isolation.

```python
"""CTC auxiliary loss for the SMT vision-encoder-decoder model.

Targets are derived from the standard decoder `labels` tensor:
  - strip BOS if present (decoder labels usually do not include BOS; decoder inputs do)
  - strip EOS if present
  - drop positions where labels == -100 (padded targets)
  - drop any token id present in `structural_token_ids` (D5 + D16).

Samples whose filtered target cannot fit in the available CTC input frames are
excluded from the batch loss. Feasibility is checked with:
  filtered_target_len + adjacent_duplicate_count(filtered_target) <= input_len
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CTCLossResult:
    loss: torch.Tensor                # scalar, 0.0 if no sample qualified
    num_samples_used: int             # samples contributing to the loss
    num_samples_excluded: int         # non-empty samples failing the CTC feasibility check
    num_samples_empty_after_filter: int  # samples whose filtered target is empty
    mean_target_length: float         # filtered length, over used samples; 0 if none
    max_target_length: int            # filtered length, over used samples; 0 if none
    mean_required_input_length: float  # target_len + adjacent_dupes, over used samples; 0 if none
    mean_input_length: float          # over used samples; 0 if none
    mean_filter_compression: float    # filtered_len / unfiltered_len, over used samples; 0 if none


def resolve_structural_token_ids(
    tokenizer,
    exclude_patterns: Iterable[str],
) -> set[int]:
    """Resolve decoded-string patterns to a set of vocab ids (D16a).

    A token is structural iff:
      - `tokenizer.decode([id]).strip()` matches exactly one of `exclude_patterns`, OR
      - the decoded-stripped string is empty (pure whitespace / artifacts).

    BPE tokens that mix structural and content characters are NOT matched and are
    therefore KEPT as CTC targets.
    """
    patterns = {p for p in exclude_patterns}
    structural: set[int] = set()
    for token_str, token_id in tokenizer.get_vocab().items():
        decoded = tokenizer.decode([int(token_id)])
        stripped = decoded.strip()
        if stripped == "" or stripped in patterns:
            structural.add(int(token_id))
    return structural


def extract_ctc_targets(
    labels: torch.Tensor,
    *,
    pad_ignore_index: int,
    bos_token_id: int,
    eos_token_id: int,
    structural_token_ids: set[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strip BOS/EOS, ignore_index, and structural-token entries from labels.

    Args:
        labels: (B, Tdec) with -100 at padded targets; may contain BOS at start and EOS at end.
        pad_ignore_index: typically -100.
        bos_token_id, eos_token_id: special token IDs.
        structural_token_ids: optional set of vocab ids to filter (D16).

    Returns:
        flat_targets: (sum(filtered_lengths),) long
        filtered_lengths: (B,) long — length AFTER filtering
        unfiltered_lengths: (B,) long — length BEFORE structural filtering (but after
            BOS/EOS/ignore-index removal). Used for compression diagnostics.
    """
    assert labels.dim() == 2, f"expected (B, Tdec), got {labels.shape}"
    device = labels.device
    batch_size = labels.size(0)
    structural = structural_token_ids or set()
    structural_tensor: torch.Tensor | None = None
    if structural:
        structural_tensor = torch.tensor(
            sorted(structural), dtype=labels.dtype, device=device
        )

    cleaned: list[torch.Tensor] = []
    filtered_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    unfiltered_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    for i in range(batch_size):
        row = labels[i]
        base_keep = (
            (row != pad_ignore_index)
            & (row != bos_token_id)
            & (row != eos_token_id)
        )
        unfiltered_lengths[i] = int(base_keep.sum().item())
        if structural_tensor is not None:
            not_structural = ~torch.isin(row, structural_tensor)
            keep = base_keep & not_structural
        else:
            keep = base_keep
        kept = row[keep]
        cleaned.append(kept)
        filtered_lengths[i] = kept.numel()

    if filtered_lengths.sum().item() == 0:
        flat = torch.zeros(0, dtype=torch.long, device=device)
    else:
        flat = torch.cat(cleaned, dim=0).to(torch.long)
    return flat, filtered_lengths, unfiltered_lengths


def _adjacent_dupe_counts(
    flat_targets: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    """Count adjacent-duplicate pairs within each concatenated target chunk.

    For a target [a, a, b, b, b, c]: adjacent pairs at positions (0,1), (2,3), (3,4) → 3.
    Each such pair consumes one extra input frame in any valid CTC alignment.
    """
    batch_size = target_lengths.numel()
    counts = torch.zeros(batch_size, dtype=torch.long, device=target_lengths.device)
    cursor = 0
    for i in range(batch_size):
        length = int(target_lengths[i].item())
        if length >= 2:
            chunk = flat_targets[cursor : cursor + length]
            counts[i] = int((chunk[1:] == chunk[:-1]).sum().item())
        cursor += length
    return counts


def compute_ctc_loss(
    *,
    ctc_logits: torch.Tensor,       # (B, Tenc, C+1), fp32
    ctc_input_lengths: torch.Tensor, # (B,) long
    labels: torch.Tensor,            # (B, Tdec)
    blank_token_id: int,
    pad_ignore_index: int,
    bos_token_id: int,
    eos_token_id: int,
    structural_token_ids: set[int] | None = None,
) -> CTCLossResult:
    assert ctc_logits.dtype == torch.float32, "CTC logits must be fp32"

    flat_targets, target_lengths, unfiltered_lengths = extract_ctc_targets(
        labels,
        pad_ignore_index=pad_ignore_index,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        structural_token_ids=structural_token_ids,
    )

    # Real CTC constraint: input_len >= target_len + adjacent_dupes(filtered_target).
    # Computed on the filtered target — adjacent dupes in the raw (pre-filter) target
    # are irrelevant because those tokens aren't CTC emissions.
    input_lengths = ctc_input_lengths.to(torch.long)
    adj_dupes = _adjacent_dupe_counts(flat_targets, target_lengths)
    required_input = target_lengths + adj_dupes
    non_empty = target_lengths > 0  # CTC on an empty target is undefined
    empty_after_filter = int((~non_empty).sum().item())
    valid_mask = (required_input <= input_lengths) & non_empty
    num_used = int(valid_mask.sum().item())
    num_excluded = int(((required_input > input_lengths) & non_empty).sum().item())

    if num_used == 0:
        return CTCLossResult(
            loss=ctc_logits.new_zeros(()),
            num_samples_used=0,
            num_samples_excluded=num_excluded,
            num_samples_empty_after_filter=empty_after_filter,
            mean_target_length=0.0,
            max_target_length=0,
            mean_required_input_length=0.0,
            mean_input_length=0.0,
            mean_filter_compression=0.0,
        )

    # Re-pack only the valid samples for CTCLoss.
    # CTCLoss accepts concatenated targets, so we just need to rebuild flat_targets.
    valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(-1)
    cursor = 0
    kept_chunks: list[torch.Tensor] = []
    kept_lengths: list[int] = []
    for i in range(target_lengths.numel()):
        length = int(target_lengths[i].item())
        if valid_mask[i]:
            kept_chunks.append(flat_targets[cursor : cursor + length])
            kept_lengths.append(length)
        cursor += length
    flat_targets_valid = torch.cat(kept_chunks, dim=0) if kept_chunks else flat_targets.new_zeros(0)
    target_lengths_valid = torch.tensor(kept_lengths, dtype=torch.long, device=labels.device)

    ctc_logits_valid = ctc_logits[valid_idx]        # (N', Tenc, C+1)
    input_lengths_valid = input_lengths[valid_idx]   # (N',)
    required_input_valid = required_input[valid_idx]  # (N',)

    # CTCLoss expects (T, N, C)
    device_type = ctc_logits_valid.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        log_probs = F.log_softmax(ctc_logits_valid.float(), dim=-1).transpose(0, 1)  # (Tenc, N', C+1)
        loss = F.ctc_loss(
            log_probs,
            flat_targets_valid,
            input_lengths_valid,
            target_lengths_valid,
            blank=blank_token_id,
            reduction="mean",
            zero_infinity=True,
        )

    # Compression ratio = filtered / unfiltered, over used samples.
    unfiltered_valid = unfiltered_lengths[valid_idx].float()
    ratios = target_lengths_valid.float() / unfiltered_valid.clamp(min=1)
    mean_compression = float(ratios.mean().item()) if ratios.numel() else 0.0

    return CTCLossResult(
        loss=loss,
        num_samples_used=num_used,
        num_samples_excluded=num_excluded,
        num_samples_empty_after_filter=empty_after_filter,
        mean_target_length=float(target_lengths_valid.float().mean().item()),
        max_target_length=int(target_lengths_valid.max().item()),
        mean_required_input_length=float(required_input_valid.float().mean().item()),
        mean_input_length=float(input_lengths_valid.float().mean().item()),
        mean_filter_compression=mean_compression,
    )
```

### 2.5 `src/training/lightning_module.py`

**Three edits.**

**(a)** In `SMTTrainer.__init__`, after the existing `self.model = SMTModelForCausalLM(self.config)` line (around 114), cache CTC config and resolve the structural-token filter:

```python
self._ctc_aux_enabled = bool(training.ctc_aux_enabled)
self._ctc_aux_weight = float(training.ctc_aux_weight)
self._ctc_aux_warmup_steps = int(training.ctc_aux_warmup_steps)
self._ctc_blank_token_id = (
    int(self.config.out_categories) if self._ctc_aux_enabled else None
)
self._ctc_structural_token_ids: set[int] = set()
if self._ctc_aux_enabled:
    from transformers import PreTrainedTokenizerFast
    from src.training.ctc_loss import resolve_structural_token_ids
    self._ctc_tokenizer = PreTrainedTokenizerFast.from_pretrained(vocab_dir)
    self._ctc_structural_token_ids = resolve_structural_token_ids(
        self._ctc_tokenizer,
        exclude_patterns=list(training.ctc_target_exclude_patterns),
    )
    # Fail loud if the filter resolves to nothing despite being configured.
    if not self._ctc_structural_token_ids and training.ctc_target_exclude_patterns:
        raise RuntimeError(
            "ctc_target_exclude_patterns configured but no matching tokenizer ids found. "
            f"Patterns={training.ctc_target_exclude_patterns}. Verify tokenizer vocab."
        )
    # One-time log of the filter for reproducibility.
    self._log_ctc_filter_summary(training.ctc_target_exclude_patterns)
```

Add the helper:

```python
def _log_ctc_filter_summary(self, patterns: list[str]) -> None:
    """Log the resolved structural-token id set once at init."""
    n = len(self._ctc_structural_token_ids)
    vocab = int(self.config.out_categories)
    logger.info(
        "CTC structural filter: patterns=%s, matched %d / %d vocab ids (%.1f%%)",
        patterns, n, vocab, 100.0 * n / max(vocab, 1),
    )
```

Do not assume `SMTTrainer` already has `self.tokenizer`; in the current repo it receives
`w2i`, `i2w`, and `vocab_dir`. Load `PreTrainedTokenizerFast.from_pretrained(vocab_dir)`
only when CTC is enabled, as shown above.

**(b)** Also in `__init__`, when building the `SMTConfig` above, pass through `use_ctc_aux`:

```python
self.config = SMTConfig(
    ...existing fields...,
    use_ctc_aux=bool(training.ctc_aux_enabled),
)
```

**(c)** In `training_step` (line 647) and `validation_step` (line 684), extend the loss computation. Add this helper on the class:

```python
def _maybe_apply_ctc_aux(
    self,
    outputs,
    labels: torch.Tensor,
    batch_size: int,
    *,
    log_prefix: str,
) -> torch.Tensor | None:
    """Compute CTC aux loss if enabled. Returns the scaled contribution to add to CE, or None."""
    if not self._ctc_aux_enabled:
        return None
    ctc_logits = getattr(outputs, "ctc_logits", None)
    ctc_input_lengths = getattr(outputs, "ctc_input_lengths", None)
    if ctc_logits is None or ctc_input_lengths is None:
        # Model was built without CTC head despite config — fail loud.
        raise RuntimeError(
            "ctc_aux_enabled=true but model.forward did not return ctc_logits. "
            "Ensure SMTConfig.use_ctc_aux is set."
        )

    from src.training.ctc_loss import compute_ctc_loss
    result = compute_ctc_loss(
        ctc_logits=ctc_logits,
        ctc_input_lengths=ctc_input_lengths,
        labels=labels,
        blank_token_id=self._ctc_blank_token_id,
        pad_ignore_index=-100,
        bos_token_id=self.config.bos_token_id,
        eos_token_id=self.config.eos_token_id,
        structural_token_ids=self._ctc_structural_token_ids,
    )
    self.log(f"{log_prefix}/ctc_loss", result.loss, on_step=True, on_epoch=True, batch_size=batch_size)
    self.log(f"{log_prefix}/ctc_samples_used", result.num_samples_used, on_step=True, batch_size=batch_size)
    self.log(f"{log_prefix}/ctc_samples_excluded", result.num_samples_excluded, on_step=True, batch_size=batch_size)
    self.log(f"{log_prefix}/ctc_excluded_fraction", result.num_samples_excluded / max(batch_size, 1), on_step=True, batch_size=batch_size)
    self.log(f"{log_prefix}/ctc_samples_empty_filtered", result.num_samples_empty_after_filter, on_step=True, batch_size=batch_size)
    self.log(f"{log_prefix}/ctc_mean_target_len", result.mean_target_length, on_step=True, batch_size=batch_size)
    self.log(f"{log_prefix}/ctc_max_target_len", result.max_target_length, on_step=True, batch_size=batch_size)
    self.log(f"{log_prefix}/ctc_mean_required_input_len", result.mean_required_input_length, on_step=True, batch_size=batch_size)
    self.log(f"{log_prefix}/ctc_mean_input_len", result.mean_input_length, on_step=True, batch_size=batch_size)
    self.log(f"{log_prefix}/ctc_filter_compression", result.mean_filter_compression, on_step=True, batch_size=batch_size)

    if self._ctc_aux_warmup_steps <= 0:
        weight = self._ctc_aux_weight
    else:
        progress = min(1.0, float(self.global_step) / float(self._ctc_aux_warmup_steps))
        weight = self._ctc_aux_weight * progress
    self.log(f"{log_prefix}/ctc_weight", weight, on_step=True, batch_size=batch_size)
    return weight * result.loss
```

In `training_step`, after `loss = outputs.loss` (line 670), before `self.log(TRAIN_LOSS, ...)`:

```python
ctc_contribution = self._maybe_apply_ctc_aux(outputs, labels, batch_size=x.size(0), log_prefix="train")
if ctc_contribution is not None:
    loss = loss + ctc_contribution
```

In `validation_step`, after `loss = outputs.loss` (line 710):

```python
_ = self._maybe_apply_ctc_aux(outputs, labels, batch_size=x.size(0), log_prefix=f"val/{base_set_name}")
```

Validation logs CTC metrics, but the primary `val/<set>/loss` must remain NLL-only
(D11). Do not add the CTC contribution to validation loss unless a future plan adds a
separate `validation_loss_includes_ctc` knob.

For validation CER, decode `outputs.ctc_logits` with `ctc_greedy_decode`, build the
filtered content-only target sequences from `labels` using the same
`structural_token_ids`, and update a separate metric named
`val/<set>/CER_ctc_content`. Do **not** compare CTC output against full decoder labels:
CTC intentionally omits filtered structural tokens, so full-target CER would measure
the filter design rather than CTC learning.

Implementation detail: add a separate `CharacterErrorRate` instance/collection for CTC
content CER, or update/log it manually, rather than mixing CTC content predictions into
the existing full-target `val_metrics_by_set`. Convert the variable-length decoded CTC
lists and filtered target lists to padded tensors using `pad_token_id` before calling
the metric.

**(d)** In `configure_optimizers`, explicitly add the CTC head parameters as a full-LR
group. The current optimizer only includes `frontend.encoder`, `frontend.projector`,
and `decoder`; without this edit the CTC head will receive gradients but never update.
Add after the decoder group:

```python
if self._ctc_aux_enabled and self.model.ctc_head is not None:
    param_groups.extend(
        split_named_params_for_weight_decay(
            self.model.ctc_head.named_parameters(),
            lr=decoder_lr,
            weight_decay=weight_decay,
            name_prefix="ctc_head",
        )
    )
```

Add a test or assertion that optimizer parameter ids cover every trainable
`ctc_head` parameter when `ctc_aux_enabled=true`.

### 2.6 `src/artifacts.py`

No new typed artifact dataclass is required. `RunArtifact` already persists the complete
JSON config in `experiment_config`, so the new `Training` config fields will be captured
there automatically as long as they are added to `src/config.py` and `config/train.json`.

Add a round-trip test that creates an artifact from a config containing:

```json
"training": {
  "ctc_aux_enabled": true,
  "ctc_aux_weight": 0.3,
  "ctc_aux_warmup_steps": 2000,
  "ctc_target_exclude_patterns": ["\t", "\n", "."]
}
```

Then assert those values survive `to_json()` / `from_json()` under
`artifact.experiment_config["training"]`. Verify `"\t"` and `"\n"` semantically as
strings; do not hand-roll escaping.

### 2.7 `src/model/checkpoint_loader.py`

When reconstructing `SMTConfig` (currently at line 82–100), read CTC enablement from
the artifact's persisted experiment config:

```python
training_cfg = artifact.experiment_config.get("training", {})
```

Then pass it into `SMTConfig`:

```python
use_ctc_aux=bool(training_cfg.get("ctc_aux_enabled", False)),
```

### 2.8 New file: `src/model/ctc_decode.py`

Diagnostic CTC greedy decoder. Used by the diagnostic inference script and validation metric.

```python
"""Greedy CTC decoding: argmax per frame, merge repeats, drop blanks."""
from __future__ import annotations

import torch


def ctc_greedy_decode(
    ctc_logits: torch.Tensor,        # (B, Tenc, C+1)
    input_lengths: torch.Tensor,     # (B,) long
    blank_token_id: int,
) -> list[list[int]]:
    assert ctc_logits.dim() == 3
    argmax = ctc_logits.argmax(dim=-1)  # (B, Tenc)
    decoded: list[list[int]] = []
    for i in range(argmax.size(0)):
        T = int(input_lengths[i].item())
        seq = argmax[i, :T].tolist()
        out: list[int] = []
        prev: int | None = None
        for tok in seq:
            if tok != prev:  # merge repeats
                if tok != blank_token_id:
                    out.append(int(tok))
                prev = tok
        decoded.append(out)
    return decoded
```

### 2.9 New file: `scripts/inference_ctc.py`

Minimal diagnostic. Mirrors `scripts/inference.py` but runs CTC greedy decoding instead of `.generate()`.

```python
"""Diagnostic CTC greedy-decode inference.

Usage:
    uv run scripts/inference_ctc.py --weights <ckpt> --image <img.png>

Prints CTC greedy-decoded output for diagnostic comparison against the standard
attention inference script.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fire
import torch
from PIL import Image

from src.data.preprocessing import preprocess_pil_image, LayoutNormalizationConfig
from src.model.checkpoint_loader import load_model_from_checkpoint
from src.model.ctc_decode import ctc_greedy_decode


@torch.no_grad()
def run(weights: str, image: str, device: str = "cuda"):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    loaded = load_model_from_checkpoint(weights, dev)
    if loaded.model.ctc_head is None:
        raise SystemExit("This checkpoint was not trained with CTC aux — nothing to decode.")
    loaded.model.eval()

    with Image.open(image).convert("RGB") as img:
        tensor, input_size = preprocess_pil_image(
            image=img, image_width=loaded.image_width,
            fixed_size=loaded.fixed_size,
            layout_normalization=LayoutNormalizationConfig(enabled=False),
        )
    pv = tensor.unsqueeze(0).to(dev)
    sz = torch.tensor([input_size], device=dev)

    enc = loaded.model.forward_encoder(pv, image_sizes=sz)
    with torch.autocast(device_type=enc.encoder_tokens_raw.device.type, enabled=False):
        ctc_logits = loaded.model.ctc_head(enc.encoder_tokens_raw.to(torch.float32))
    if enc.encoder_attention_mask is None:
        input_lengths = torch.full(
            (ctc_logits.size(0),),
            ctc_logits.size(1),
            dtype=torch.long,
            device=ctc_logits.device,
        )
    else:
        input_lengths = enc.encoder_attention_mask.sum(dim=-1).to(torch.long)
    blank = int(loaded.model.config.out_categories)
    decoded = ctc_greedy_decode(ctc_logits, input_lengths, blank_token_id=blank)[0]

    text = "".join(loaded.i2w.get(int(t), "") for t in decoded)
    print("[ctc-greedy]")
    print(text)


if __name__ == "__main__":
    fire.Fire(run)
```

### 2.10 Tests

Add the following. All use pytest.

**`tests/unit/training/test_ctc_loss.py`** — covers `extract_ctc_targets`, `resolve_structural_token_ids`, and `compute_ctc_loss`:

- `test_extract_targets_strips_bos_eos_and_ignore_index`
- `test_extract_targets_handles_all_padded_row` (all -100 → length 0)
- `test_extract_targets_strips_structural_token_ids` (given a set, those ids are removed from the flat target AND reduce filtered_lengths; unfiltered_lengths is preserved)
- `test_extract_targets_returns_empty_when_filter_drops_everything` (target composed entirely of structural ids → filtered_lengths = 0, unfiltered_lengths > 0)
- `test_resolve_structural_from_tokenizer` (load the project tokenizer at `vocab/bpe3k-splitspaces-tokenizer.json`, pass patterns `["\t", "\n", "."]`, assert the resolved set is nonempty, contains ids whose `decode()` is in the pattern set, and excludes ids for mixed content tokens like `4e` or any multi-character pitch token)
- `test_compute_ctc_loss_returns_zero_when_all_excluded` (construct a batch where every filtered target fails `target_len + adjacent_dupes <= input_len`)
- `test_compute_ctc_loss_counts_empty_after_filter_separately` (all-structural target rows count toward `num_samples_empty_after_filter`, not `num_samples_excluded`)
- `test_compute_ctc_loss_matches_reference_on_toy_example` (hand-construct a trivial `(T=10, N=2, C=5)` case with filter disabled, compare against raw `F.ctc_loss`)
- `test_compute_ctc_loss_fp32_under_amp_input` (pass fp16 logits and assert assertion error; we require fp32)
- `test_compute_ctc_loss_compression_metric` (filter drops half the tokens → mean_filter_compression ≈ 0.5)

**`tests/unit/model/test_ctc_head.py`**:

- `test_ctc_head_output_shape` (B=2, verify `(B, Tenc, out_categories+1)`)
- `test_no_ctc_head_when_disabled` (model built with `use_ctc_aux=False` has `ctc_head is None` and `forward` returns no `ctc_logits`)
- `test_forward_exposes_ctc_logits_and_lengths` (sanity check the stashed outputs)

**`tests/unit/model/test_ctc_decode.py`**:

- `test_greedy_merges_repeats_and_drops_blanks`
- `test_greedy_respects_input_lengths` (later frames beyond `input_length` must not leak into the output)

**`tests/unit/training/test_ctc_lightning_integration.py`**:

- `test_ctc_head_parameters_are_in_optimizer` (with `ctc_aux_enabled=true`, collect
  optimizer parameter ids and assert every trainable `model.ctc_head` parameter is
  present exactly once)
- `test_validation_logs_ctc_content_cer` (small batch, CTC enabled; assert the metric
  name is `CER_ctc_content` or the logged scalar is `val/<set>/CER_ctc_content`, and
  that targets are filtered before CER computation)

Target test count: **~17 unit tests**. Total added runtime: ≤5 seconds.

---

## 3. Execution order

Do the edits in this order. Each stage should end with a clean commit.

0. **Target-length feasibility diagnostic** (no model changes). Add or run a small
   script that loads train/polish labels, applies the proposed CTC target filter, and
   reports raw length, filtered length, adjacent-dupe count, required input length, and
   excluded fraction against `Tenc=1551`. Gate: filtered exclusion should be <5% on
   synthetic train and not obviously pathological on polish. If it is >10%, stop and
   revisit §8 before implementing CTC training.
1. **Config plumbing** (2.1, 2.2, 2.6, 2.7). Flip no behavior. Run existing tests — they must still pass.
2. **CTC loss module** (2.4) and its unit tests (2.10 first file). Loss module in isolation.
3. **Model head and forward wiring** (2.3) and its unit tests (2.10 second file). Verify `ctc_logits` shows up in model output when enabled.
4. **Lightning integration** (2.5). Run a dry-run training on 10 batches of the `debug` config (`config/debug.json` exists per `ls config/`) to confirm nothing crashes.
5. **CTC decode + diagnostic script** (2.8, 2.9) and their unit tests (2.10 third file).
6. **Config enablement** (toggle `ctc_aux_enabled: true` in a new experimental config — see §5).

One PR per stage, or one PR with stage-sized commits. Do **not** squash before review — the stage boundaries are review checkpoints.

---

## 4. Hyperparameter recipe

| Param | Value | Notes |
| --- | --- | --- |
| `ctc_aux_enabled` | `true` | Obviously. |
| `ctc_aux_weight` | `0.3` | Standard hybrid-ASR default. Sweep `{0.1, 0.3, 0.5}` only if the 0.3 run does not improve polish CER. |
| `ctc_aux_warmup_steps` | `2000` for first real run | Config default remains `0` for backwards compatibility, but warmup is recommended for the first CTC experiment. |
| `label_smoothing` (existing) | `0.1` | Leave alone. CTC and label smoothing compose fine. |
| `learning_rate` (existing) | `0.001` | Leave alone. |
| `encoder_lr_factor` (existing) | `0.3` | Leave alone. CTC head must **not** inherit this reduced encoder LR; it gets its own full-LR optimizer group (D18). |
| `warmup_steps` (existing optimizer warmup) | `500` | Leave alone. This is separate from `ctc_aux_warmup_steps`. |
| `gradient_clip_val` (existing) | `1.0` | Leave alone. CTC gradients are well-behaved with `zero_infinity`. |
| `ctc_target_exclude_patterns` | `["\t", "\n", "."]` | Starting set per §1.2.1. If `train/ctc_samples_excluded` > 10% or `train/ctc_filter_compression` > 0.6 on the smoke run, expand to include spine markers (`*^`, `*v`, `*-`) one at a time and re-check. If compression < 0.2, remove `.` first — it may be over-aggressive. |

The CTC head parameters go into their own optimizer group at the decoder/full learning
rate, with the same weight-decay splitting policy as the decoder. Verify by inspecting
`configure_optimizers()` after changes: CTC head must not be absent, and must not be in
the reduced-LR encoder group. If it accidentally ends up with 0.3× LR, it will learn
too slowly.

---

## 5. Rollout on `antemurale`

Follow `CLAUDE.md` conventions — push to GitHub, pull on remote.

1. Create `config/train_ctc.json` as a copy of `config/train.json` with:
   ```json
   {
     "training": {
       "...": "...",
       "ctc_aux_enabled": true,
       "ctc_aux_weight": 0.3,
       "ctc_aux_warmup_steps": 2000,
       "ctc_target_exclude_patterns": ["\t", "\n", "."]
     }
   }
   ```
   Keep the `checkpoint.run_name` distinct (e.g., `"SMT-System-level-es-polish-v1-ctc"`) and write to a new `checkpoint.dirpath`.
2. Smoke-test locally for 100 steps with `config/debug.json` + the CTC overrides. Confirm:
   - No NaN in `train/loss` or `train/ctc_loss`.
   - `train/ctc_samples_used` ≈ batch size and `train/ctc_excluded_fraction` < 5% after filtering (A10).
   - `train/ctc_filter_compression` lands in the 0.25–0.55 band (A11). If not, stop and tune `ctc_target_exclude_patterns` before continuing.
   - Init-time log shows `CTC structural filter: patterns=[...], matched N / V vocab ids` with N > 0.
   - Both losses decrease over the 100 steps.
3. Submit a 2000-step dry run on `antemurale` with `train_ctc.json`. Compare `val/polish/CER` and `val/polish/CER_no_ties_beams` against the most recent non-CTC baseline at the same step count.
4. If step-3 CER is not worse, launch the full-length run. Log both CE and CTC losses in wandb.
5. After training converges, run `scripts/diagnostics/loop_rate.py` (see `docs/guides/loop-degeneration-diagnostics.md`) on `polish` for both the baseline and CTC checkpoints. The CTC checkpoint's loop rate is the primary success metric.

Slurm wrap gotcha (per `CLAUDE.md`): if submitting ad hoc with `sbatch --wrap`, use `bash -lc 'source .venv/bin/activate && ...'`.

---

## 6. Acceptance criteria

A PR may merge when **all** of the following hold:

| # | Criterion | How verified |
| --- | --- | --- |
| A1 | All new unit tests pass | `pytest tests/unit/training/test_ctc_loss.py tests/unit/training/test_ctc_lightning_integration.py tests/unit/model/test_ctc_head.py tests/unit/model/test_ctc_decode.py` |
| A2 | Full existing test suite still passes with `ctc_aux_enabled=false` | `pytest` |
| A3 | 100-step debug run completes without NaN | Check wandb run |
| A4 | With `ctc_aux_enabled=true`, forward pass overhead ≤10% wall-clock per step | `perf/step_time_ms` in wandb compared against baseline |
| A5 | Polish CER does not regress vs. a matched-step baseline checkpoint | Compare `val/polish/CER` and `val/polish/CER_no_ties_beams` at a common step |
| A6 | Polish loop rate (per `analyze_catastrophic_repetition`) measurably decreases | Run `scripts/diagnostics/loop_rate.py` before/after |
| A7 | `CER_ctc_content` on polish is within `2×` of attention CER after applying the same content-token filter to targets | Sanity check that the CTC head actually learned something; do not compare content-only CTC output to full decoder targets. |
| A8 | A checkpoint trained with CTC can be loaded by `scripts/inference.py` | Manual run |
| A9 | `scripts/inference_ctc.py` produces plausible (not empty, not garbage) output on a polish sample | Manual run |
| A10 | Filter coverage: `train/ctc_samples_excluded` / batch_size < 5% on the synthetic training set after warmup (step 500+) | Inspect wandb. Value > 10% means the filter is not compressing enough — adjust `ctc_target_exclude_patterns` before launching the full run. |
| A11 | Filter compression: `train/ctc_filter_compression` ≈ 0.25–0.55 on synthetic training (i.e. filtered target is 25–55% the length of the raw target) | Inspect wandb. Values near 1.0 mean the filter isn't matching anything; values near 0.0 mean it's over-aggressive and stripping content. |
| A12 | CTC feasibility diagnostic passes before training changes are trusted | Stage 0 report shows filtered exclusion <5% on train, and reports adjacent-dupe-adjusted required lengths rather than raw target lengths only. |

If A5 fails but A4 and A3 pass, the implementation is likely correct but the hyperparameter is wrong — try `ctc_aux_weight=0.1`. If A7 fails (CTC head is severely worse than attention), the CTC head is not learning — verify blank index, fp32 logits, and that the head parameters are in the correct optimizer group.

---

## 7. Common pitfalls

Read these before you write code. Each has caused a lost day for someone.

1. **Wrong CTC tensor layout.** `F.ctc_loss` expects `(T, N, C)` for log-probs. The head produces `(B, Tenc, C+1)`. You must `.transpose(0, 1)` after `log_softmax`. Getting this wrong produces a plausible-looking loss that does not learn.
2. **Forgetting `log_softmax`.** `F.ctc_loss` wants log-probabilities, not logits. It will not raise — it will silently compute garbage.
3. **Wrong blank index.** The blank is at index `out_categories` (the `+1` slot), not index 0, not `pad_token_id`. Passing `blank=0` will make the model emit the first vocabulary token in place of blanks. Verify in `test_compute_ctc_loss_matches_reference_on_toy_example`.
4. **AMP + CTC.** Under `max-autotune-no-cudagraphs` with AMP, CTC loss silently NaNs when logits are fp16. The `.to(torch.float32)` in §2.3(b) is non-negotiable.
5. **Target sequences that are too long.** Naive `F.ctc_loss` on `target_len > input_len` gives `inf` or NaN. `zero_infinity=True` hides the NaN but also hides the bug. The D6 filter is what actually fixes it. Log `ctc_samples_excluded` and watch it — if it's > 5% of the batch, something is wrong with the data or with `Tenc`.
6. **Mistreating BOS/EOS in targets.** The decoder is trained to emit `<eos>`. The CTC head should *not* — leaving EOS in the CTC target forces the head to learn an end-of-sequence symbol it has no reason to emit. Strip both BOS and EOS (D5).
7. **Wrong encoder attention mask.** `ctc_input_lengths = encoder_attention_mask.sum(dim=-1)` only works if `True = valid`. Confirm in `ConvVisionFrontend.build_memory_key_padding_mask` (`src/model/frontends/conv_frontend.py:80`) — it uses the convention `True = valid`. Do not negate.
8. **Config drift between train and artifact.** If you enable `ctc_aux_enabled` in
   `Training` but the saved `run_artifact_json` does not preserve that value under
   `experiment_config["training"]`, the checkpoint loader will rebuild the model without
   the CTC head and fail on `ctc_head.*` state-dict keys. The round-trip artifact test
   in §2.6 is the protection.
9. **Compile boundary.** `F.ctc_loss` does not play well with `torch.compile(mode="max-autotune-no-cudagraphs")`. The design places CTC loss in the Lightning `training_step` (outside the compiled `self.model`), not inside the model forward. Do not move the loss call into `model.forward`.
10. **Gradient flow to the frozen encoder.** The config exposes `freeze_encoder_stages`. If the frontend is partially frozen, the CTC head still trains, but gradients to frozen encoder blocks are no-ops — meaning the CTC signal only reshapes the projector and later stages. This is intentional; don't "fix" it.
11. **Filter resolves to zero ids (silent drop).** If `ctc_target_exclude_patterns` doesn't match any vocab tokens (e.g. the tokenizer encodes `\t` only as part of longer multi-character BPE tokens), `resolve_structural_token_ids` returns an empty set and the filter becomes a no-op — you're back to the 30% exclusion regime. The Lightning init raises if patterns are configured but none match; do not silence that error. Validate with `test_resolve_structural_from_tokenizer`.
12. **Over-aggressive filter (stripping content).** If `ctc_target_exclude_patterns` accidentally matches content tokens (e.g. adding `"4"` strips every quarter-note duration token), the CTC target becomes semantically damaged. The `stripped == pattern` match rule in `resolve_structural_token_ids` is intentionally exact — do not switch it to `startswith` or `in` without thinking very hard. Acceptance criterion A11 (compression in 0.25–0.55 band) is the main guard.
13. **Filter drift vs. decoder vocab.** The decoder still predicts the full vocab including structural tokens. Do NOT filter the decoder CE targets — only the CTC targets. The filter lives entirely inside `extract_ctc_targets`; nothing touches `output.loss` (the attention CE).
14. **BPE encoding of `\t` and `\n`.** Tokenizers built with whitespace pre-tokenization may not have standalone ids for tab/newline — they may appear only as part of larger tokens (e.g. `\t4e`). In that case the filter won't find them and compression will be near 1.0. Check A11 on the debug run before submitting the antemurale job. If pure-whitespace tokens don't exist, the filter needs to be redesigned (e.g. match tokens whose decoded string *ends* with a structural run), but that is out of scope for v2 — flag and escalate.

---

## 8. What this plan does not cover

Out of scope, explicitly:

- Joint CTC-attention beam search at inference. The CTC head here is auxiliary and diagnostic; it does not rescore attention beams.
- CTC-only inference as the primary path. Attention decoder remains primary.
- Changes to the tokenizer or vocabulary.
- FCMAE pretraining on unlabeled real scores. That is a separate, independent plan.
- Cross-attention entropy regularization, unlikelihood training, decoder input-token dropout. These are complementary but independent workstreams.
- Handling encoder backbones other than ConvNeXtV2 — if `encoder_model_name_or_path` changes, revisit D4.
- Increasing `Tenc` by enlarging the input image or reducing encoder stride. Both are plausible follow-ups if §1.2.1 coverage turns out insufficient under the content-only filter, but they require re-training from scratch and are a separate plan.
- A CTC-only upsampling adapter. If Stage 0 or A10 shows too many samples are excluded,
  prefer a CTC-only sequence/map upsampler before reducing ConvNeXt stride: it gives
  the CTC head more alignment frames without increasing decoder cross-attention length.
- Redesigning the filter rule to handle tokenizers that only encode structural characters as part of larger BPE tokens (see pitfall 14). V2 relies on the existence of standalone structural-token ids.

---

## 9. Questions that are explicitly answered

- *"Should we add blank to the tokenizer?"* No (D1).
- *"Should we use encoder_tokens_pos or encoder_tokens_raw?"* Raw (D2).
- *"Should we share weights with the decoder vocab projection?"* No (D13).
- *"What if some samples are too long?"* Two-tier mitigation. First, the CTC target filter (D5 + D16) strips purely-structural tokens (tabs, newlines, nulls) — typical compression is 2–4× on `**kern`, which brings the great majority of polish targets under `Tenc = 1551`. Second, the residual (samples still too long after filtering) is masked out of the CTC loss (D6). Acceptance criteria A10 and A11 verify that filter-only coverage is sufficient and that the mask is only a safety net, not a primary mechanism.
- *"Why not just make `Tenc` bigger?"* Doubling `Tenc` via larger input image or finer encoder stride requires retraining the encoder from scratch (and may lose the clean ConvNeXtV2 pretrained geometry) while substantially increasing cross-attention cost. Deferred — revisit only if content-only CTC proves insufficient. If more CTC frames are needed, try a CTC-only upsampling adapter first.
- *"Why filter on decoded strings instead of raw token strings?"* BPE token strings can contain artifacts (leading `Ġ`, byte-level markers). Decoding forces the comparison to be over the actual surface form the model is learning to emit, which is the right invariant.
- *"Do we need a warmup on ctc_weight?"* The config default is `0` for no behavior change, but the first real run should use `ctc_aux_warmup_steps=2000` (D8).
- *"What weight should we use?"* 0.3. Sweep `{0.1, 0.3, 0.5}` only if 0.3 fails (see §4).
- *"Do we log a CTC error rate at validation?"* Yes, `val/<set>/CER_ctc_content` (D11).
- *"Does this affect inference speed?"* Not for the default attention inference path. The CTC head is a single Linear that runs during `forward_encoder` but its logits are ignored at generate time.
- *"Does this change the checkpoint format?"* It adds `ctc_head.weight` and `ctc_head.bias` parameters. Old checkpoints still load (the new parameters initialize fresh). New checkpoints loaded by old code will error on the extra keys — accepted cost.
