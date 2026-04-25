# Diagnosing Decoder Loop Degeneration on Real Scores

Symptom: the model emits an infinite alternation like

```
.	64ddd
.	64ccc#
.	64ddd
.	64ccc#
...
```

on the `data/datasets/validation/polish/` (real-score) split, while `synth` is fine.

This guide defines five diagnostics that, together, determine **which** underlying mechanism is driving the loop, so the next fix can be chosen on evidence rather than intuition. Work through them top to bottom.

## What we are trying to distinguish

Two very different root causes can produce the same visible failure:

1. **Encoder features are OOD on real scans.** Cross-attention has something to look at, but the features are noisy or unfamiliar, so the decoder picks up the LM prior.
2. **The decoder ignores the encoder entirely once a loop begins.** Cross-attention has collapsed; the output is purely an LM bigram cycle regardless of what the image contains.

The fixes differ:

| Dominant cause | Best next move |
| --- | --- |
| Encoder features OOD | FCMAE pretraining on unlabeled real scores; stronger domain-invariant training |
| Decoder ignores encoder | CTC auxiliary loss; cross-attention entropy regularizer; decoder input dropout; unlikelihood training |

Most real systems have both. The diagnostics below tell you the ratio.

## Prerequisites

- Working venv (`source .venv/bin/activate`) per `CLAUDE.md`.
- A trained checkpoint (`.ckpt`) known to reproduce the loop on `polish`.
- Access to `data/datasets/validation/polish/` (HF arrow) and `data/datasets/validation/synth/`.
- Tokenizer at `vocab/bpe3k-splitspaces-tokenizer.json` (loaded automatically from artifact).

All diagnostic scripts below belong under `scripts/diagnostics/`. Create the directory if it doesn't exist.

**Shared setup** — every diagnostic starts from the same loader scaffold:

```python
# scripts/diagnostics/_common.py
from pathlib import Path
import torch
from datasets import load_from_disk
from src.model.checkpoint_loader import load_model_from_checkpoint
from src.data.preprocessing import preprocess_pil_image, LayoutNormalizationConfig

def load(weights: str, device: str = "cuda"):
    dev = torch.device(device)
    loaded = load_model_from_checkpoint(weights, dev)
    loaded.model.eval()
    return loaded, dev

def encode_image_from_dataset(loaded, row, device):
    """Convert a dataset row into the pixel_values tensor the model expects."""
    img = row["image"].convert("RGB") if hasattr(row["image"], "convert") else row["image"]
    tensor, model_input_size = preprocess_pil_image(
        image=img,
        image_width=loaded.image_width,
        fixed_size=loaded.fixed_size,
        layout_normalization=LayoutNormalizationConfig(enabled=False),
    )
    return tensor.unsqueeze(0).to(device), torch.tensor([model_input_size], device=device)
```

Every diagnostic produces an artifact in `docs/reports/diag-<name>-<run_id>.{json,png}` for the record. Keep these — you need them to compare runs.

---

## Diagnostic 1 — Encoder ablation (is the decoder even reading the image?)

**Purpose.** Determine whether, mid-loop, the encoder output still influences the decoder's next-token distribution. If it doesn't, the loop is a pure LM attractor regardless of feature quality.

**How it works.** Build a *looping* prefix by running generation on a failing Polish image until the loop is well established (e.g., 40–60 tokens into the repetition). Then, at the next step, run the decoder twice:

- **Control:** with the real encoder output.
- **Ablation:** with the encoder output zeroed / shuffled / swapped from an unrelated score.

If next-token predictions are **unchanged** under ablation, the decoder is not reading the encoder. If they change meaningfully, the encoder signal is still getting through and the loop is driven by noisy features rather than attention collapse.

**Where to hook in.** The encoder output is a `VisionFrontendOutput` (`src/model/vision_frontend.py:11`) with fields `encoder_tokens_raw`, `encoder_tokens_pos`, `encoder_attention_mask`. It is passed into the decoder in `SMTModelForCausalLM.forward()` at `src/model/modeling_smt.py:904`. You can construct or replace it freely.

**Script — `scripts/diagnostics/encoder_ablation.py`:**

```python
"""Encoder ablation: compare next-token distributions with real vs ablated encoder outputs."""
from __future__ import annotations
import json, sys
from dataclasses import asdict, dataclass
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import fire, torch
import torch.nn.functional as F
from datasets import load_from_disk
from src.model.vision_frontend import VisionFrontendOutput
from scripts.diagnostics._common import load, encode_image_from_dataset

@dataclass
class AblationResult:
    sample_index: int
    prefix_len: int
    top1_control: str
    top1_zeroed: str
    top1_shuffled: str
    top1_swapped: str
    kl_zeroed: float
    kl_shuffled: float
    kl_swapped: float
    argmax_agreement_zeroed: bool
    argmax_agreement_shuffled: bool
    argmax_agreement_swapped: bool

def _kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    return F.kl_div(q, p, reduction="batchmean", log_target=True).item()

def _ablate(enc: VisionFrontendOutput, mode: str, donor: VisionFrontendOutput | None):
    if mode == "zero":
        return VisionFrontendOutput(
            encoder_tokens_raw=torch.zeros_like(enc.encoder_tokens_raw),
            encoder_tokens_pos=torch.zeros_like(enc.encoder_tokens_pos),
            encoder_attention_mask=enc.encoder_attention_mask,
        )
    if mode == "shuffle":
        perm = torch.randperm(enc.encoder_tokens_raw.size(1), device=enc.encoder_tokens_raw.device)
        return VisionFrontendOutput(
            encoder_tokens_raw=enc.encoder_tokens_raw[:, perm],
            encoder_tokens_pos=enc.encoder_tokens_pos[:, perm],
            encoder_attention_mask=enc.encoder_attention_mask,
        )
    if mode == "swap":
        assert donor is not None
        return VisionFrontendOutput(
            encoder_tokens_raw=donor.encoder_tokens_raw,
            encoder_tokens_pos=donor.encoder_tokens_pos,
            encoder_attention_mask=donor.encoder_attention_mask,
        )
    raise ValueError(mode)

def _build_looping_prefix(model, pixel_values, image_sizes, device, n_tokens: int = 200) -> torch.Tensor:
    """Run unconstrained greedy generation to get a prefix that is inside a loop."""
    with torch.no_grad():
        out = model.generate(
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            do_sample=False,
            num_beams=1,
            max_length=n_tokens,
            use_cache=True,
        )
    return out  # (1, T)

@torch.no_grad()
def _next_token_logits(model, prefix_ids: torch.Tensor, enc: VisionFrontendOutput) -> torch.Tensor:
    """One forward pass through the decoder, using supplied encoder features. No KV cache."""
    out = model.forward(
        input_ids=prefix_ids,
        encoder_outputs=enc,
        encoder_attention_mask=enc.encoder_attention_mask,
        use_cache=False,
    )
    return out.logits[:, -1, :]  # (1, vocab)

def run(weights: str, polish_path: str = "./data/datasets/validation/polish",
        sample_index: int = 0, donor_index: int = 1, prefix_len: int = 120,
        out_path: str = "docs/reports/diag-encoder-ablation.json"):
    loaded, device = load(weights)
    model, i2w = loaded.model, loaded.i2w
    ds = load_from_disk(polish_path)

    victim = ds[sample_index]
    donor = ds[donor_index]
    pv_v, sz_v = encode_image_from_dataset(loaded, victim, device)
    pv_d, sz_d = encode_image_from_dataset(loaded, donor, device)

    # Encoder outputs
    enc_v = model.forward_encoder(pv_v, image_sizes=sz_v)
    enc_d = model.forward_encoder(pv_d, image_sizes=sz_d)

    # Looping prefix — truncate to the first `prefix_len` tokens of an unconstrained generation
    prefix = _build_looping_prefix(model, pv_v, sz_v, device, n_tokens=prefix_len + 20)[:, :prefix_len]

    logits_real = _next_token_logits(model, prefix, enc_v)
    logits_zero = _next_token_logits(model, prefix, _ablate(enc_v, "zero", None))
    logits_shuf = _next_token_logits(model, prefix, _ablate(enc_v, "shuffle", None))
    logits_swap = _next_token_logits(model, prefix, _ablate(enc_v, "swap", enc_d))

    def top1(l): return i2w[int(l.argmax(dim=-1).item())]

    result = AblationResult(
        sample_index=sample_index, prefix_len=prefix_len,
        top1_control=top1(logits_real),
        top1_zeroed=top1(logits_zero),
        top1_shuffled=top1(logits_shuf),
        top1_swapped=top1(logits_swap),
        kl_zeroed=_kl(logits_real, logits_zero),
        kl_shuffled=_kl(logits_real, logits_shuf),
        kl_swapped=_kl(logits_real, logits_swap),
        argmax_agreement_zeroed=top1(logits_real) == top1(logits_zero),
        argmax_agreement_shuffled=top1(logits_real) == top1(logits_shuf),
        argmax_agreement_swapped=top1(logits_real) == top1(logits_swap),
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(asdict(result), indent=2))
    print(json.dumps(asdict(result), indent=2))

if __name__ == "__main__":
    fire.Fire(run)
```

**Run it across many samples.** Loop over the first 10–20 failing Polish samples and aggregate. One sample is not a signal.

```bash
for i in $(seq 0 19); do
  uv run scripts/diagnostics/encoder_ablation.py \
    --weights weights/GrandStaff/es_polish_v1/smt-model.ckpt \
    --sample_index $i --donor_index $(( (i + 7) % 30 )) \
    --out_path docs/reports/diag-encoder-ablation/${i}.json
done
```

**How to read the numbers.**

| Signal across ≥10 samples | Interpretation | Implied next fix |
| --- | --- | --- |
| `argmax_agreement_*` ≈ 100%, `kl_* < 0.05` | Decoder ignores encoder entirely | CTC aux loss, attention-entropy reg, unlikelihood, decoder input dropout |
| `argmax_agreement_*` mixed (40–80%), `kl_* ∈ [0.1, 1.0]` | Encoder signal present but weak | FCMAE + CTC aux (combined) |
| `argmax_agreement_*` ≈ 0%, `kl_* > 2.0` | Encoder signal strong, features OOD | FCMAE pretraining dominates |

Expect row 1 or row 2 in practice. `swap` (donor) is the most informative variant — if even a *completely different score's* encoder output produces the same next token, the cross-attention is not being used at all.

**Failure modes when running this.**
- `prefix_len` too small → you're measuring the healthy regime, not the loop. Confirm the prefix is inside the repetition by printing `i2w[tok]` for the last 20 tokens of the prefix before measuring.
- `donor` image is too similar to `victim` → `swap` becomes uninformative. Pick a donor from a different engraving style.

---

## Diagnostic 2 — Cross-attention entropy and map

**Purpose.** Measure whether cross-attention distributions collapse (near-zero entropy, single peak) during loops, and whether the attended region **freezes or drifts off-staff** when the loop begins.

**Implementation note.** `CrossAttention` uses `F.scaled_dot_product_attention` (`src/model/modeling_smt.py:374`), which does not expose weights. For diagnostic runs, monkey-patch the forward to compute softmax explicitly and record weights. Only do this in eval scripts — it's slower and numerically slightly different from SDPA.

**Script — `scripts/diagnostics/xattn_capture.py`:**

```python
"""Capture per-step cross-attention weights and compute entropy trace."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import fire, torch, math
import torch.nn.functional as F
import numpy as np
from datasets import load_from_disk
from src.model.modeling_smt import CrossAttention
from scripts.diagnostics._common import load, encode_image_from_dataset

_CAPTURE: list[torch.Tensor] = []

def _patched_cross_attn_forward(self: CrossAttention, query, key, value,
                                 key_padding_mask=None, past_key_value=None, use_cache=False):
    q = self.q_proj(query)
    if past_key_value is not None:
        k, v = past_key_value
        q = self._split_heads(q)
    else:
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))
        q = self._split_heads(q)

    # Explicit softmax to expose weights
    scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, Tq, Tk)
    if key_padding_mask is not None:
        mask = self._build_attention_mask(key_padding_mask=key_padding_mask,
                                          query_length=q.size(-2), key_length=k.size(-2),
                                          is_causal=False)
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    _CAPTURE.append(weights.detach().cpu())
    out = torch.matmul(weights, v)
    out = self._merge_heads(out)
    out = self.out_proj(out)
    return out, (k, v) if use_cache else None

def _entropy(weights: torch.Tensor) -> torch.Tensor:
    """Entropy per query position; weights shape: (B, H, Tq, Tk)."""
    eps = 1e-12
    return -(weights.clamp_min(eps) * weights.clamp_min(eps).log()).sum(dim=-1)

@torch.no_grad()
def run(weights: str, polish_path: str = "./data/datasets/validation/polish",
        sample_index: int = 0, max_tokens: int = 300,
        out_path: str = "docs/reports/diag-xattn-entropy.json"):
    loaded, device = load(weights)
    model = loaded.model
    CrossAttention.forward = _patched_cross_attn_forward  # monkey-patch

    ds = load_from_disk(polish_path)
    pv, sz = encode_image_from_dataset(loaded, ds[sample_index], device)

    # Manual greedy loop so we can tag which step each weight tensor belongs to
    bos = torch.tensor([[model.config.bos_token_id]], device=device)
    enc = model.forward_encoder(pv, image_sizes=sz)

    ids = bos
    step_entropy_mean: list[float] = []
    step_entropy_max_head: list[float] = []
    step_max_weight: list[float] = []
    tokens: list[str] = []

    past = None
    for step in range(max_tokens):
        _CAPTURE.clear()
        out = model.forward(
            input_ids=ids if step == 0 else ids[:, -1:],
            encoder_outputs=enc,
            encoder_attention_mask=enc.encoder_attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        tokens.append(loaded.i2w[int(nxt.item())])

        if _CAPTURE:
            # Take last decoder layer's weights for the current query step
            w = _CAPTURE[-1]  # (B, H, 1, Tk)
            ent = _entropy(w)[0, :, 0]  # (H,)
            step_entropy_mean.append(ent.mean().item())
            step_entropy_max_head.append(ent.max().item())
            step_max_weight.append(w[0, :, 0].max().item())

        if nxt.item() == model.config.eos_token_id:
            break

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({
        "sample_index": sample_index,
        "tokens": tokens,
        "entropy_mean": step_entropy_mean,
        "entropy_max_head": step_entropy_max_head,
        "max_attention_weight": step_max_weight,
        "n_encoder_tokens": int(enc.encoder_tokens_raw.size(1)),
    }, indent=2))

if __name__ == "__main__":
    fire.Fire(run)
```

**Optional — attention-map overlay.** For a handful of samples, also save per-step attention maps reshaped to the encoder's 2D grid (the frontend records `self._last_encoder_hw` at `src/model/frontends/conv_frontend.py:41`). Overlay on the preprocessed image and dump PNGs per step to `docs/reports/diag-xattn-maps/<sample>/step_<t>.png`. Visual inspection is often decisive — a frozen cursor is immediately obvious.

**How to read it.**

- Plot `entropy_mean` over decoding steps. Mark where the loop begins (you can find this by looking for the first 8-token run in `tokens` where the same two lines alternate).
- **Entropy stays flat and low across the loop** → attention has collapsed (bucket 1 above).
- **Entropy rises or becomes noisy inside the loop** → attention is searching but not finding; features are OOD (bucket 2).
- **`max_attention_weight` near 1.0 inside loops** → single-token cursor that's not advancing. In the 2D map, check whether that cursor is **on a staff or off it** — off-staff confirms attention has wandered into padding/background.

Thresholds (approximate, based on typical behavior):

| `entropy_mean` in loop vs healthy | Call |
| --- | --- |
| Loop entropy < 0.3 × healthy entropy | Collapse |
| Loop entropy in [0.3, 0.8] × healthy | Partial collapse |
| Loop entropy ≈ healthy (but wrong tokens) | Not a collapse — features OOD |

Compute `healthy entropy` by running the same diagnostic on a sample from `data/datasets/validation/synth/` where generation does not loop.

---

## Diagnostic 3 — Next-token output-distribution entropy

**Purpose.** Cheap second opinion on Diagnostic 2. Output-distribution entropy is trivial to log and often tells the same story.

**Procedure.** Extend Diagnostic 2's manual greedy loop to also record `F.softmax(logits, dim=-1)` entropy per step. No patching needed. Add inside the step loop:

```python
probs = F.softmax(out.logits[:, -1, :], dim=-1)
entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1).item()
step_output_entropy.append(entropy)
```

Save alongside the cross-attention trace in the same JSON.

**Interpretation.** Output entropy near zero (model is highly confident) across loop steps is the classical "attractor" signature — the model is *certain* about each next token. Combined with low cross-attention entropy it's conclusive: the decoder has decided and the image is no longer in the loop.

---

## Diagnostic 4 — Loop-rate comparison: synth vs polish

**Purpose.** Quantify whether loops are domain-specific or global. The original premise is that only `polish` loops; this diagnostic confirms the premise with numbers, and gives a baseline to compare against after any fix.

**Reuse what exists.** `src/core/metrics/runaway_monitor.py:212` already has `analyze_catastrophic_repetition`. Run inference on each split and apply it.

**Script — `scripts/diagnostics/loop_rate.py`:**

```python
"""Run inference over a split and compute loop rate + per-sample diagnostics."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import fire, torch
from datasets import load_from_disk
from src.core.metrics.runaway_monitor import analyze_catastrophic_repetition
from src.grammar.semantic_sequence_finalizer import finalize_generated_kern_sequence
from scripts.diagnostics._common import load, encode_image_from_dataset

@torch.no_grad()
def run(weights: str, split_path: str,
        limit: int | None = None,
        max_length: int = 2048,
        out_path: str = "docs/reports/diag-loop-rate.json"):
    loaded, device = load(weights)
    model = loaded.model
    ds = load_from_disk(split_path)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    rows = []
    for i in range(len(ds)):
        pv, sz = encode_image_from_dataset(loaded, ds[i], device)
        out = model.generate(pixel_values=pv, image_sizes=sz, do_sample=False,
                             num_beams=1, max_length=max_length, use_cache=True,
                             repetition_penalty=1.0)  # Note: no repetition penalty to expose raw behavior
        ids = out[0].tolist()
        finalized = finalize_generated_kern_sequence(
            token_ids=ids, i2w=loaded.i2w,
            bos_token_id=model.config.bos_token_id,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=loaded.pad_token_id,
            max_length=max_length, rule_factories=[],
        )
        diag = analyze_catastrophic_repetition(finalized.text)
        rows.append({
            "index": i,
            "tokens_generated": len(ids),
            "repeat_loop": diag.repeat_loop,
            "reason": diag.repeat_loop_reason,
            "max_identical_line_run": diag.max_identical_line_run,
            "max_ngram_occurrences": diag.max_repeated_ngram_occurrences,
            "coverage": diag.repeated_ngram_line_coverage,
        })

    loop_rate = sum(r["repeat_loop"] for r in rows) / len(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({
        "split_path": split_path,
        "n": len(rows),
        "loop_rate": loop_rate,
        "rows": rows,
    }, indent=2))
    print(f"loop rate on {split_path}: {loop_rate:.1%}  (n={len(rows)})")

if __name__ == "__main__":
    fire.Fire(run)
```

**Run on both splits:**

```bash
uv run scripts/diagnostics/loop_rate.py \
  --weights weights/GrandStaff/es_polish_v1/smt-model.ckpt \
  --split_path data/datasets/validation/polish \
  --out_path docs/reports/diag-loop-rate-polish.json

uv run scripts/diagnostics/loop_rate.py \
  --weights weights/GrandStaff/es_polish_v1/smt-model.ckpt \
  --split_path data/datasets/validation/synth --limit 256 \
  --out_path docs/reports/diag-loop-rate-synth.json
```

**Reading.**

| Observation | Interpretation |
| --- | --- |
| polish ≫ synth (e.g., 15% vs 0.5%) | Domain-specific. FCMAE-shaped problem. |
| polish ≈ synth, both high | Not domain-specific; decoder is generically collapse-prone. Decoder-side fixes. |
| polish ≈ synth, both low | You may be looking at a rare edge case; re-examine failing sample selection. |

Record `loop_rate` as your baseline. Every subsequent fix must beat this.

---

## Diagnostic 5 — Sequence-length distribution: train vs polish

**Purpose.** Invalidate (or confirm) one specific confound: that `polish` requires sequence lengths the model never trained on, pushing RoPE into extrapolation territory.

You have already said "train lengths cover polish lengths" — this diagnostic **verifies** that claim with numbers and protects against drift as the training corpus changes.

**Script — `scripts/diagnostics/length_distribution.py`:**

```python
"""Compare tokenized sequence-length distributions across splits."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import fire
import numpy as np
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast

def _lengths(path: str, tok: PreTrainedTokenizerFast, text_col: str = "transcription", limit: int | None = None):
    ds = load_from_disk(path)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [len(tok(t, add_special_tokens=True)["input_ids"]) for t in ds[text_col]]

def run(train_path: str = "./data/datasets/train/train_full",
        polish_path: str = "./data/datasets/validation/polish",
        synth_path: str = "./data/datasets/validation/synth",
        vocab_dir: str = "./vocab/bpe3k-splitspaces",
        out_path: str = "docs/reports/diag-length-distribution.json"):
    tok = PreTrainedTokenizerFast.from_pretrained(vocab_dir)
    result = {}
    for name, p in [("train", train_path), ("polish", polish_path), ("synth", synth_path)]:
        lens = _lengths(p, tok, limit=5000 if name == "train" else None)
        lens_arr = np.asarray(lens)
        result[name] = {
            "n": len(lens),
            "min": int(lens_arr.min()), "max": int(lens_arr.max()),
            "mean": float(lens_arr.mean()),
            "p50": float(np.percentile(lens_arr, 50)),
            "p95": float(np.percentile(lens_arr, 95)),
            "p99": float(np.percentile(lens_arr, 99)),
        }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    fire.Fire(run)
```

**Reading.** Compare `polish.p99` to `train.p95`. If `polish.p99 > train.p95`, the tail of real scores exceeds what you trained on — a contributing factor to instability even if not the primary cause. `config/train.json` caps `max_seq_len: 2048`; anything close to that is already risky.

---

## Decision Matrix

After all five diagnostics, fill in this table and decide.

| Diagnostic | Signal for "decoder ignores encoder" | Signal for "encoder features OOD" | Your measurement |
| --- | --- | --- | --- |
| D1 encoder ablation | argmax agreement ≈ 100% under swap | agreement ≈ 0%, high KL | |
| D2 xattn entropy | flat and very low in loop | high or noisy in loop | |
| D3 output entropy | near zero in loop | moderate in loop | |
| D4 synth vs polish | loops on both | loops only on polish | |
| D5 length tails | irrelevant | irrelevant (this is a confound check) | |

- **D1 + D2 + D3 all point to "decoder ignores encoder"** → the single highest-leverage next change is **CTC auxiliary loss** (adds a discriminative head the decoder cannot bypass), followed by cross-attention entropy regularization.
- **D1 argmax flips under swap, D4 shows a large polish/synth gap** → **FCMAE pretraining on unlabeled real sheet music** is the single highest-leverage next change. Pair with **pseudo-labeling** on the same corpus.
- **Mixed signals** → both. Run FCMAE first (it's the foundation) and add CTC in the supervised fine-tune on top of it.

## Hygiene

- Run every diagnostic on ≥10 samples and report distributions, not single examples.
- Save raw JSON to `docs/reports/diag-<name>-<run_id>.json`. Keep these under version control; they are your before/after evidence when a fix lands.
- Re-run D1 and D4 after any structural change to training (objective, pretraining, architecture). They're the fastest way to confirm the fix actually moved the mechanism you targeted.
- **Gotcha:** the ConvNeXtV2 encoder is wrapped with gradient checkpointing enabled by default (`src/model/modeling_smt.py:777`). Diagnostics run under `torch.no_grad()` so it is inert, but if you adapt these into training-time probes, disable the wrapper or you will see non-reproducible outputs.
- **Gotcha:** `CrossAttention` KV-cache short-circuits computation when `past_key_value` is set (`src/model/modeling_smt.py:358`). The patched forward in Diagnostic 2 preserves this path — do not "simplify" it away or entropy traces over long prefixes will be wrong.
