# SMT Model Architecture

Vision-to-sequence architecture for Optical Music Recognition (OMR), converting sheet music images into `**kern` token sequences.

This document reflects the current implementation in `src/model/` and current usage (RoPE in decoder).

## High-Level Overview

```
Image (B, 3, H, W)
    │
    ▼
┌──────────────────────────────────────────────┐
│ ConvVisionFrontend                           │
│  - HuggingFace vision backbone (AutoModel)   │
│  - Token-space MLP projector                 │
│  - 2D sinusoidal positional stream           │
│  - Optional encoder attention mask           │
└──────────────────────────────────────────────┘
    │
    ├── encoder_tokens_raw: (B, S_enc, d_model)
    ├── encoder_tokens_pos: (B, S_enc, d_model)
    └── encoder_attention_mask: (B, S_enc) bool, True=valid
    │
    ▼
┌──────────────────────────────────────────────┐
│ Autoregressive Decoder                       │
│  - Token embedding                           │
│  - RoPE self-attention positions             │
│  - N pre-norm decoder layers                 │
│      self-attn -> cross-attn -> FFN          │
│  - Final LayerNorm + vocab projection        │
└──────────────────────────────────────────────┘
    │
    ▼
Logits (B, T, vocab_size)
```

Cross-attention uses:
- Keys from `encoder_tokens_pos`
- Values from `encoder_tokens_raw`

## Vision Frontend

### `ConvVisionFrontend`

Current supported frontend is `vision_frontend="conv"` only.

Pipeline:
1. Run encoder via `EncoderLoader` (`AutoModel.from_pretrained`).
2. Convert encoder feature map `(B, C_enc, H', W')` to token sequence `(B, H'*W', C_enc)`.
3. Project to decoder width with `ProjectorMLP`: `Linear -> GELU -> Linear`.
4. Build positional key stream by:
   - Reshaping projected tokens to `(B, d_model, H', W')`
   - Adding 2D sinusoidal positional encoding
   - Flattening back to `(B, H'*W', d_model)`
5. Return `VisionFrontendOutput` with raw tokens, positional tokens, and optional mask.

### Encoder Loader

`EncoderLoader` currently supports `encoder_provider="transformers"` and detects output channels from encoder config with heuristic checks:
- `hidden_sizes[-1]`
- `vision_config.hidden_size`
- `hidden_size`
- `num_channels`
- `embed_dim`

Optional `freeze_encoder_stages > 0` applies ConvNeXt-style freezing:
- freeze `embeddings`
- freeze first `N` blocks in `encoder.stages`

### Encoder Attention Mask

Variable image sizes are handled in `ConvVisionFrontend.build_memory_key_padding_mask(...)`.

Inputs:
- `image_sizes`: original `(H, W)` per sample
- `encoder_hw`: encoder map size `(H', W')`
- `input_hw`: padded input size `(H_pad, W_pad)`

Output:
- boolean mask `(B, H'*W')`, `True = valid`

Valid extents are computed with ceil-style scaling:
- `valid_h = ceil(orig_h * H' / H_pad)`
- `valid_w = ceil(orig_w * W' / W_pad)`

## Decoder

### Positional Encoding (Current Usage)

The decoder supports two modes in code:
- `absolute`: learned token embedding + 1D sinusoidal addition
- `rope`: learned token embedding + RoPE on self-attention Q/K

Current usage is RoPE (`positional_encoding="rope"`).

RoPE details:
- Implemented via `torchtune.modules.RotaryPositionalEmbeddings`
- Applied to self-attention Q/K before KV-cache concatenation
- Position indices are offset by cached sequence length during generation

### Decoder Block

Each `DecoderLayer` is pre-norm with residual connections:
1. `LayerNorm -> SelfAttention -> Dropout -> Residual add`
2. `LayerNorm -> CrossAttention -> Dropout -> Residual add`
3. `LayerNorm -> FFN(Linear -> GELU -> Dropout -> Linear) -> Dropout -> Residual add`

`DecoderStack` applies `N` layers, then final `LayerNorm`.

### Attention and Caching

Base behavior (`_BaseAttention`):
- head split/merge helpers
- combined mask creation (causal + padding key mask)
- attention kernel uses `torch.nn.functional.scaled_dot_product_attention`

Important implementation note:
- Manual non-SDPA attention path is not implemented.

Self-attention:
- fused QKV projection
- supports KV cache

Cross-attention:
- separate Q/K/V projections
- supports KV cache during generation (cached projected encoder K/V reused per step)

KV cache tensor layout:
- `(batch, num_heads, seq_len, head_dim)`

Per decoder layer cache structure:
- `((self_k, self_v), (cross_k, cross_v))`

## Model Interface and Generation

`SMTModelForCausalLM` extends `PreTrainedModel` + `GenerationMixin`.

### Forward Contract

Primary inputs:
- `pixel_values` for first step / training
- `input_ids` or `labels` for decoder
- optional `encoder_outputs` for reuse
- optional `encoder_attention_mask`
- optional `past_key_values`

Training conveniences:
- If `input_ids` missing and `labels` present, decoder inputs are built by right shift + BOS.
- Loss is `CrossEntropyLoss(ignore_index=-100)`.

### Generation Flow

Implemented hooks:
- `prepare_inputs_for_generation`
- `_update_model_kwargs_for_generation`
- `_reorder_cache`

Behavior:
1. First generation step runs encoder (unless `encoder_outputs` already provided).
2. Encoder outputs and encoder mask are preserved in `model_kwargs`.
3. Later steps pass only last token (`input_ids[:, -1:]`) plus cache.
4. Beam search cache reorder is applied to both self and cross caches.

Compatibility handling:
- Accepts legacy tuple cache and newer Transformers cache objects (checks cache content via `get_seq_length()` when available).

## Effective Configuration Surface

Key architecture parameters:
- `encoder_model_name_or_path`
- `encoder_provider` (currently `transformers`)
- `freeze_encoder_stages`
- `vision_frontend` (currently `conv`)
- `d_model`
- `num_hidden_layers`
- `num_attn_heads`
- `dim_ff`
- `projector_hidden_mult`
- `out_categories`
- `maxlen`
- `positional_encoding` (current usage: `rope`)
- `rope_theta`

Token IDs:
- `pad_token_id`
- `bos_token_id`
- `eos_token_id`

## Training-Relevant Implementation Notes

- Encoder input is converted to channels-last memory format before backbone forward.
- Encoder gradient checkpointing is enabled when supported by the selected backbone.
- Decoder stack has a gradient-checkpointing code path, but enabling it is separate from encoder checkpointing.
- Attention execution requires PyTorch SDPA (`torch.nn.functional.scaled_dot_product_attention`).
- Legacy checkpoints with only `encoder_to_decoder_projection.*` keys are rejected; current bridge is `frontend.projector.*` (2-layer MLP).
