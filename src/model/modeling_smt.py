import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.checkpoint import checkpoint
from torchtune.modules import RotaryPositionalEmbeddings
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithCrossAttentions

from .configuration_smt import SMTConfig
from .frontends import ConvVisionFrontend
from .vision_frontend import VisionFrontendOutput


class PositionalEncoding2D(nn.Module):
    def __init__(self, dim, h_max, w_max):
        assert dim % 4 == 0, "PositionalEncoding2D expects dim divisible by 4."
        super().__init__()
        self.h_max = h_max
        self.dim = dim

        self.pe: torch.Tensor
        self.register_buffer(
            "pe",
            torch.zeros((dim, h_max, w_max), requires_grad=False),
            persistent=False,
        )

        div = torch.exp(
            -torch.arange(0.0, dim // 2, 2) / dim * torch.log(torch.tensor(1e4))
        ).unsqueeze(1)
        w_pos = torch.arange(0.0, w_max) * div
        h_pos = torch.arange(0.0, h_max) * div
        self.pe[: dim // 2 : 2] = torch.sin(h_pos).unsqueeze(2).repeat(1, 1, w_max)
        self.pe[1 : dim // 2 : 2] = torch.cos(h_pos).unsqueeze(2).repeat(1, 1, w_max)
        self.pe[dim // 2 :: 2] = torch.sin(w_pos).unsqueeze(1).repeat(1, h_max, 1)
        self.pe[dim // 2 + 1 :: 2] = torch.cos(w_pos).unsqueeze(1).repeat(1, h_max, 1)

    def forward(self, x):
        """
        Add 2D positional encoding to x
        x: Tensor(B, C, H, W)
        returns:
        - Tensor(B, C, H, W)
        """
        return x + self.get_pe_by_size(x.size(-2), x.size(-1)).to(dtype=x.dtype, device=x.device)

    def get_pe_by_size(self, h, w):
        return self.pe[:, :h, :w]


class PositionalEncoding1D(nn.Module):
    def __init__(self, dim, len_max):
        super().__init__()
        self.len_max = len_max
        self.dim = dim
        self.pe: torch.Tensor
        self.register_buffer(
            "pe", torch.zeros((len_max, dim), requires_grad=False), persistent=False
        )

        div = torch.exp(-torch.arange(0.0, dim, 2) / dim * torch.log(torch.tensor(1e4)))
        l_pos = torch.arange(0.0, len_max).unsqueeze(1) * div
        self.pe[:, ::2] = torch.sin(l_pos)
        self.pe[:, 1::2] = torch.cos(l_pos)

    def forward(self, x, start=0):
        """
        Add 1D positional encoding to x
        x: Tensor(B, L, C)
        start: index for x[:, 0, :]
        returns:
        - Tensor(B, L, C)
        """
        return x + self.pe[start : start + x.size(-2)]


class _BaseAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        use_fused_qkv: bool = False,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            logger.error("The embeddings depth must be divisible by the number of heads")
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = self.d_head**-0.5
        self.use_fused_qkv = use_fused_qkv
        if not hasattr(F, "scaled_dot_product_attention"):
            raise RuntimeError(
                "scaled_dot_product_attention is required. "
                "This model no longer supports non-SDPA attention paths."
            )

        if self.use_fused_qkv:
            self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        else:
            self.q_proj = nn.Linear(d_model, d_model, bias=bias)
            self.k_proj = nn.Linear(d_model, d_model, bias=bias)
            self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self._init_parameters()

        self.dropout = nn.Dropout(dropout)

    def _init_parameters(self) -> None:
        if self.use_fused_qkv:
            nn.init.xavier_uniform_(self.qkv_proj.weight)
            if self.qkv_proj.bias is not None:
                nn.init.zeros_(self.qkv_proj.bias)
        else:
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            if self.q_proj.bias is not None:
                nn.init.zeros_(self.q_proj.bias)
            if self.k_proj.bias is not None:
                nn.init.zeros_(self.k_proj.bias)
            if self.v_proj.bias is not None:
                nn.init.zeros_(self.v_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Split the heads and put them into a batch-first format."""
        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.d_head)
        return tensor.transpose(1, 2)  # (batch_size, num_heads, seq_len, d_head)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Merge heads and transpose back to batch-first format."""
        batch_size = tensor.shape[0]
        tensor = tensor.transpose(1, 2)
        return tensor.reshape(batch_size, -1, self.d_model).contiguous()

    def _apply_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        rope: RotaryPositionalEmbeddings | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply attention mechanism with optional KV caching.

        Args:
            q, k, v: Query, key, value tensors before head splitting
            key_padding_mask: Optional padding mask
            is_causal: Whether to apply causal masking
            past_key_value: Optional cached keys/values in head-split format
            rope: Optional RoPE module for rotary positional embeddings
            input_pos: Position indices for RoPE (shape: [seq_len])

        Returns:
            Tuple of (output, present_key_value)
        """
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Apply RoPE BEFORE KV cache concatenation
        # RoPE rotates Q and K based on position indices
        if rope is not None and input_pos is not None:
            # Torchtune expects [b, s, n_h, h_d], we have [b, n_h, s, h_d]
            q = rope(q.transpose(1, 2), input_pos=input_pos).transpose(1, 2)
            k = rope(k.transpose(1, 2), input_pos=input_pos).transpose(1, 2)

        # Concatenate cached keys/values AFTER head splitting and RoPE
        # This ensures we only split heads once per token, not repeatedly
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)  # Concatenate on sequence dimension
            v = torch.cat([past_v, v], dim=2)

        # Cache the head-split tensors for next iteration
        present_key_value = (k, v)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = self._build_attention_mask(
                key_padding_mask=key_padding_mask,
                query_length=q.size(-2),
                key_length=k.size(-2),
                is_causal=is_causal,
            )

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            # Dropout is not automatically turned off during eval
            dropout_p=self.dropout.p if self.training else 0.0,
            # PyTorch docs: An error is thrown if both attn_mask and is_causal are set.
            # TODO: attn_mask will likely always be set. Could try to optimize to use is_causal when possible.
            is_causal=is_causal if attn_mask is None else False,
        )

        output = self._merge_heads(attn_output)
        output = self.out_proj(output)

        return output, present_key_value

    def _build_attention_mask(
        self,
        key_padding_mask: torch.Tensor | None,
        query_length: int,
        key_length: int,
        is_causal: bool,
    ) -> torch.Tensor | None:
        """Build a proper attention mask from key padding mask and causal requirements."""
        # PyTorch docs:
        # Attention mask; shape must be broadcastable to the shape of attention weights, which is (N,...,L,S)
        # N: batch size
        # L: target sequence length
        # S: source sequence length

        device = key_padding_mask.device if key_padding_mask is not None else None

        # Start with causal mask if needed
        if is_causal:
            # Create causal mask [seq_len, seq_len]
            causal_mask = torch.tril(
                torch.ones(query_length, key_length, device=device), diagonal=0
            ).bool()
            # True = keep (attend to), matching PyTorch convention
            attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
        else:
            attn_mask = None

        # Add key padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: [batch, seq_len], True = valid
            # Convert to attention mask format
            key_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]

            if attn_mask is not None:
                # Combine with causal mask
                batch_size = key_padding_mask.shape[0]
                attn_mask = attn_mask.expand(batch_size, -1, -1, -1) & key_mask
            else:
                # Just use key mask, expanded for query positions
                attn_mask = key_mask.expand(-1, -1, query_length, -1)

        return attn_mask


class SelfAttention(_BaseAttention):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            use_fused_qkv=True,
        )

    def forward(
        self,
        query: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        rope: RotaryPositionalEmbeddings | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Forward pass for self-attention with optional KV caching.

        Args:
            query: Input tensor of shape (batch_size, seq_len, d_model)
            key_padding_mask: Optional mask for padding positions
            is_causal: Whether to apply causal masking
            past_key_value: Optional tuple of cached (keys, values) in head-split format
                           Shape: (batch_size, num_heads, past_seq_len, head_dim)
            use_cache: Whether to return the current key-value cache
            rope: Optional RoPE module for rotary positional embeddings
            input_pos: Position indices for RoPE (shape: [seq_len])

        Returns:
            A tuple of (attention_output, present_key_value).
            present_key_value is returned only if use_cache is True, and contains
            head-split tensors of shape (batch_size, num_heads, seq_len, head_dim).
        """
        qkv = self.qkv_proj(query)
        q, k, v = qkv.chunk(3, dim=-1)

        # Delegate cache concatenation to _apply_attention (after head splitting)
        attn_output, present_key_value = self._apply_attention(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            past_key_value=past_key_value,
            rope=rope,
            input_pos=input_pos,
        )

        # Only return cache if use_cache is True
        return attn_output, (present_key_value if use_cache else None)


class CrossAttention(_BaseAttention):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Forward pass for cross-attention with optional KV caching.

        During generation, encoder outputs never change, so we can cache the projected
        K/V tensors after the first step and reuse them on subsequent steps.

        Args:
            query: Query tensor from decoder (batch_size, seq_len, d_model)
            key: Key tensor from encoder (batch_size, enc_seq_len, d_model)
            value: Value tensor from encoder (batch_size, enc_seq_len, d_model)
            key_padding_mask: Optional mask for padding positions
            past_key_value: Optional tuple of cached (keys, values) in head-split format
            use_cache: Whether to return the current key-value cache

        Returns:
            Tuple of (attention_output, present_key_value).
            present_key_value is returned only if use_cache is True.
        """
        q = self.q_proj(query)

        # Use cached K/V if available, otherwise project from encoder outputs
        if past_key_value is not None:
            # Cached values are already head-split, bypass projection and _apply_attention's
            # head splitting by directly computing attention here
            k, v = past_key_value
            q = self._split_heads(q)

            # Compute attention directly with cached K/V
            attn_mask = None
            if key_padding_mask is not None:
                attn_mask = self._build_attention_mask(
                    key_padding_mask=key_padding_mask,
                    query_length=q.size(-2),
                    key_length=k.size(-2),
                    is_causal=False,
                )

            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )

            output = self._merge_heads(attn_output)
            output = self.out_proj(output)

            # Return the same cached values
            return output, (k, v) if use_cache else None

        # No cache available, project K/V from encoder outputs
        if key is None or value is None:
            raise ValueError("Both key and value must be provided for cross-attention")

        k = self.k_proj(key)
        v = self.v_proj(value)

        attn_output, present_key_value = self._apply_attention(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
            is_causal=False,
        )
        return attn_output, present_key_value if use_cache else None


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = SelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.cross_attn = CrossAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

        self.activation = nn.GELU()

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )

        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(3)])

    def forward(
        self,
        x: torch.Tensor,
        encoder_output_key: torch.Tensor,
        encoder_output_value: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        past_self_attn_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_cross_attn_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        rope: RotaryPositionalEmbeddings | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None,
    ]:
        """
        Forward pass for decoder layer with optional KV caching.

        Args:
            x: Input tensor
            encoder_output_key: Encoder output for cross-attention keys
            encoder_output_value: Encoder output for cross-attention values
            tgt_key_padding_mask: Mask for valid positions in target sequence (True = valid)
            memory_key_padding_mask: Padding mask for encoder output
            past_self_attn_key_value: Optional past key-value cache for self-attention
            past_cross_attn_key_value: Optional past key-value cache for cross-attention
            use_cache: Whether to return the current key-value cache
            rope: Optional RoPE module for rotary positional embeddings
            input_pos: Position indices for RoPE (shape: [seq_len])

        Returns:
            Tuple of (output, (present_self_kv, present_cross_kv))
        """
        # 1. Self-Attention (Causal)
        norm_x = self.norm_layers[0](x)
        attn_output, present_self_kv = self.self_attn(
            query=norm_x,
            key_padding_mask=tgt_key_padding_mask,
            is_causal=not use_cache,  # Causal mask is only needed for full sequences
            past_key_value=past_self_attn_key_value,
            use_cache=use_cache,
            rope=rope,
            input_pos=input_pos,
        )
        x = x + self.dropout_layers[0](attn_output)

        # 2. Cross-Attention
        norm_x = self.norm_layers[1](x)
        attn_output, present_cross_kv = self.cross_attn(
            query=norm_x,
            key=encoder_output_key,
            value=encoder_output_value,
            key_padding_mask=memory_key_padding_mask,
            past_key_value=past_cross_attn_key_value,
            use_cache=use_cache,
        )
        x = x + self.dropout_layers[1](attn_output)

        # 3. Feed-Forward Network
        norm_x = self.norm_layers[2](x)
        ffn_output = self.ffn(norm_x)
        x = x + self.dropout_layers[2](ffn_output)

        present_key_value = (present_self_kv, present_cross_kv) if use_cache else None
        return x, present_key_value


class DecoderStack(nn.Module):
    def __init__(
        self,
        num_hidden_layers: int,
        d_model: int,
        dim_ff: int,
        num_heads: int,
        dropout: float,
        gradient_checkpointing: bool = False,
        use_rope: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.use_rope = use_rope
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model=d_model, num_heads=num_heads, dim_ff=dim_ff, dropout=dropout)
                for _ in range(num_hidden_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output_2D: torch.Tensor,
        encoder_output_raw: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        past_key_values: tuple[tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]], ...] | None = None,
        use_cache: bool = False,
        rope: RotaryPositionalEmbeddings | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        tuple[tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]], ...]
        | None,
    ]:
        """
        Forward pass for decoder stack with optional KV caching.

        Args:
            x: Input tensor
            encoder_output_2D: 2D encoder output for cross-attention keys
            encoder_output_raw: Raw encoder output for cross-attention values
            tgt_key_padding_mask: Mask for valid positions in target sequence (True = valid)
            memory_key_padding_mask: Padding mask for encoder output
            past_key_values: Optional past key-value caches for all layers.
                Structure: tuple of ((self_k, self_v), (cross_k, cross_v)) per layer
            use_cache: Whether to return the current key-value caches
            rope: Optional RoPE module for rotary positional embeddings
            input_pos: Position indices for RoPE (shape: [seq_len])

        Returns:
            Tuple of (output, next_key_values)
            next_key_values structure: tuple of ((self_k, self_v), (cross_k, cross_v)) per layer
        """
        output = x
        next_key_values = [] if use_cache else None

        for i, dec_layer in enumerate(self.layers):
            # Unpack layer cache into self-attention and cross-attention caches
            if past_key_values is not None:
                layer_past = past_key_values[i]
                past_self_attn_kv = layer_past[0]
                past_cross_attn_kv = layer_past[1]
            else:
                past_self_attn_kv = None
                past_cross_attn_kv = None

            if self.gradient_checkpointing and self.training:
                # --- Gradient Checkpointing Path ---
                def create_custom_forward(module, rope_module, pos_tensor):
                    def custom_forward(*inputs):
                        # inputs[0]: x
                        # inputs[1]: encoder_output_key
                        # inputs[2]: encoder_output_value
                        # inputs[3]: tgt_key_padding_mask
                        # inputs[4]: memory_key_padding_mask
                        # inputs[5]: past_self_attn_key_value (usually None in training)
                        # inputs[6]: past_cross_attn_key_value (usually None in training)
                        return module(
                            x=inputs[0],
                            encoder_output_key=inputs[1],
                            encoder_output_value=inputs[2],
                            tgt_key_padding_mask=inputs[3],
                            memory_key_padding_mask=inputs[4],
                            past_self_attn_key_value=inputs[5],
                            past_cross_attn_key_value=inputs[6],
                            use_cache=False,  # Must be False during checkpointing
                            rope=rope_module,
                            input_pos=pos_tensor,
                        )

                    return custom_forward

                # checkpoint requires at least one input to have requires_grad=True
                layer_outputs = checkpoint(
                    create_custom_forward(dec_layer, rope, input_pos),
                    output,
                    encoder_output_2D,
                    encoder_output_raw,
                    tgt_key_padding_mask,
                    memory_key_padding_mask,
                    past_self_attn_kv,
                    past_cross_attn_kv,
                    use_reentrant=False,  # Modern default
                )

                assert layer_outputs is not None, "Checkpointing returned None"

                # Unpack the tuple (output, present_key_value)
                output, present_key_value = layer_outputs
            else:
                # --- Standard Path ---
                output, present_key_value = dec_layer(
                    x=output,
                    encoder_output_key=encoder_output_2D,
                    encoder_output_value=encoder_output_raw,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    past_self_attn_key_value=past_self_attn_kv,
                    past_cross_attn_key_value=past_cross_attn_kv,
                    use_cache=use_cache,
                    rope=rope,
                    input_pos=input_pos,
                )

            if use_cache and next_key_values is not None:
                next_key_values.append(present_key_value)

        output = self.final_norm(output)
        next_key_values = (
            tuple(next_key_values) if (use_cache and next_key_values is not None) else None
        )

        return output, next_key_values


class Decoder(nn.Module):
    def __init__(
        self,
        num_hidden_layers: int,
        d_model: int,
        dim_ff: int,
        n_heads: int,
        max_seq_length: int,
        out_categories: int,
        dropout: float = 0.1,
        positional_encoding: str = "absolute",
        rope_theta: float = 10000.0,
    ):
        super().__init__()

        self.positional_encoding_type = positional_encoding

        self.decoder = DecoderStack(
            num_hidden_layers=num_hidden_layers,
            d_model=d_model,
            dim_ff=dim_ff,
            num_heads=n_heads,
            dropout=dropout,
            use_rope=(positional_encoding == "rope"),
        )

        self.embedding = nn.Embedding(num_embeddings=out_categories, embedding_dim=d_model)

        # Initialize position encoding based on config
        if positional_encoding == "absolute":
            self.position_encoding = PositionalEncoding1D(dim=d_model, len_max=max_seq_length)
            self.rope = None
        elif positional_encoding == "rope":
            self.position_encoding = None
            # RoPE operates on head dimension (d_model // n_heads)
            head_dim = d_model // n_heads
            self.rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_length, base=rope_theta)
        else:
            raise ValueError(f"Unknown positional_encoding: {positional_encoding}. Use 'absolute' or 'rope'.")

        self.vocab_projection = nn.Linear(in_features=d_model, out_features=out_categories)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        decoder_input: torch.Tensor,
        encoder_output_2D: torch.Tensor,
        encoder_output_raw: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        past_key_values: tuple | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple | None]:
        """
        Forward pass for decoder with optional KV caching.

        Args:
            decoder_input: Input token IDs
            encoder_output_2D: 2D encoder output for cross-attention
            encoder_output_raw: Raw encoder output for cross-attention
            tgt_key_padding_mask: Mask for valid positions in target sequence (True = valid)
            memory_key_padding_mask: Padding mask for encoder output
            past_key_values: Optional past key-value caches.
                Structure: tuple of ((self_k, self_v), (cross_k, cross_v)) per layer
            use_cache: Whether to return the current key-value caches

        Returns:
            Tuple of (output, predictions, next_key_values)
        """
        # Determine the position offset for positional encoding
        past_length = 0
        if past_key_values is not None:
            # Cache structure: ((self_k, self_v), (cross_k, cross_v)) per layer
            # Self-attention keys are stored in head-split shape (batch_size, num_heads, seq_len, head_dim)
            # Index 2 is the sequence length dimension
            past_length = past_key_values[0][0][0].shape[2]

        seq_len = decoder_input.size(1)
        decoder_input = self.embedding(decoder_input)

        # Apply positional encoding based on type
        if self.positional_encoding_type == "absolute":
            decoder_input = self.position_encoding(decoder_input, start=past_length)
            input_pos = None
        else:
            # RoPE: compute position indices for the current sequence
            input_pos = torch.arange(
                past_length, past_length + seq_len,
                device=decoder_input.device, dtype=torch.long
            )

        output, next_key_values = self.decoder(
            x=decoder_input,
            encoder_output_2D=encoder_output_2D,
            encoder_output_raw=encoder_output_raw,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            rope=self.rope,
            input_pos=input_pos,
        )

        output = self.dropout(output)

        predictions = self.vocab_projection(output)

        return output, predictions, next_key_values


class SMTOutput(CausalLMOutputWithCrossAttentions):
    """
    Output wrapper for the SMT model.

    Inherits from CausalLMOutputWithCrossAttentions and includes:
        - loss: Optional language modeling loss
        - logits: Prediction scores
        - past_key_values: Optional cached key-value states
        - hidden_states: Optional hidden states
        - attentions: Always None (attention weights are not returned)
        - cross_attentions: Always None (attention weights are not returned)
    """


class SMTModelForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = SMTConfig

    def __init__(self, config: SMTConfig):
        super().__init__(config)

        # --- Vision frontend ---
        if config.vision_frontend == "conv":
            self.frontend = ConvVisionFrontend(config)
        else:
            raise ValueError(
                f"Unsupported vision_frontend: {config.vision_frontend}. Supported: ['conv']"
            )

        try:
            self.frontend.encoder.gradient_checkpointing_enable()
        except (AttributeError, ValueError):
            # Some encoder models don't support gradient checkpointing
            logger.warning(
                f"Encoder {config.encoder_model_name_or_path} does not support gradient checkpointing"
            )

        # --- Decoder ---
        self.decoder = Decoder(
            num_hidden_layers=config.num_hidden_layers,
            d_model=config.d_model,
            dim_ff=config.dim_ff,
            n_heads=config.num_attn_heads,
            max_seq_length=config.maxlen,
            out_categories=config.out_categories,
            positional_encoding=config.positional_encoding,
            rope_theta=config.rope_theta,
        )

        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

        self.w2i = config.w2i
        self.i2w = config.i2w
        self.maxlen = int(config.maxlen)

    def forward_encoder(
        self,
        x: torch.Tensor,
        image_sizes: torch.Tensor | None = None,
    ) -> VisionFrontendOutput:
        """Forward pass through encoder.

        Args:
            x: Normalized image tensor (B, C, H, W) in range [-1, 1].
               Normalization is handled by the dataloader.
        """
        return self.frontend(images=x, image_sizes=image_sizes)

    def forward_decoder(
        self,
        encoder_outputs: VisionFrontendOutput,
        last_predictions,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        past_key_values=None,
        use_cache=False,
    ):
        """
        Forward pass through the decoder.

        Args:
            encoder_outputs: Vision frontend output
            last_predictions: Input token IDs for decoder
            tgt_key_padding_mask: Mask for valid positions in target sequence (True = valid)
            memory_key_padding_mask: Padding mask for encoder output
            past_key_values: Optional past key-value caches
            use_cache: Whether to return the current key-value caches

        Returns:
            SMTOutput containing logits and optional cache
        """
        output, predictions, next_key_values = self.decoder(
            decoder_input=last_predictions,
            encoder_output_2D=encoder_outputs.encoder_tokens_pos,
            encoder_output_raw=encoder_outputs.encoder_tokens_raw,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        return SMTOutput(
            logits=predictions,
            past_key_values=next_key_values,
            hidden_states=(output,),
            attentions=None,
            cross_attentions=None,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor) -> torch.Tensor:
        # labels: (B, T) with -100 at padded targets
        pad_id = self.config.pad_token_id
        bos_id = self.config.bos_token_id

        # replace ignore_index with pad for decoder input construction
        lab = torch.where(labels.eq(-100), torch.full_like(labels, pad_id), labels)
        dec_in = torch.empty_like(lab)
        dec_in[:, 0] = bos_id
        dec_in[:, 1:] = lab[:, :-1]
        return dec_in

    def _coerce_encoder_outputs(
        self,
        encoder_outputs: VisionFrontendOutput | BaseModelOutput,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> VisionFrontendOutput:
        if isinstance(encoder_outputs, VisionFrontendOutput):
            if encoder_outputs.encoder_attention_mask is None and encoder_attention_mask is not None:
                return VisionFrontendOutput(
                    encoder_tokens_raw=encoder_outputs.encoder_tokens_raw,
                    encoder_tokens_pos=encoder_outputs.encoder_tokens_pos,
                    encoder_attention_mask=encoder_attention_mask,
                )
            return encoder_outputs

        if isinstance(encoder_outputs, BaseModelOutput):
            raw_tokens = encoder_outputs.last_hidden_state
            pos_tokens = getattr(encoder_outputs, "encoder_tokens_pos", None)
            if pos_tokens is None and encoder_outputs.hidden_states:
                pos_tokens = encoder_outputs.hidden_states[0]
            if pos_tokens is None:
                pos_tokens = raw_tokens

            mask = encoder_attention_mask
            if mask is None:
                mask = getattr(encoder_outputs, "encoder_attention_mask", None)

            return VisionFrontendOutput(
                encoder_tokens_raw=raw_tokens,
                encoder_tokens_pos=pos_tokens,
                encoder_attention_mask=mask,
            )

        raise TypeError(
            f"encoder_outputs must be VisionFrontendOutput or BaseModelOutput, got: {type(encoder_outputs)}"
        )

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_outputs: VisionFrontendOutput | BaseModelOutput | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        image_sizes: torch.Tensor | None = None,
        **kwargs,
    ):
        """
        Forward pass of the SMT model.

        Args:
            pixel_values: Input images (batch, 3, height, width)
            input_ids: Decoder input token IDs (batch, seq_len)
            labels: Ground truth labels for training (batch, seq_len) with -100 for padding
            attention_mask: Attention mask for decoder (True = valid positions)
            encoder_outputs: Optional pre-computed encoder outputs (for generation)
            encoder_attention_mask: Optional pre-computed encoder padding mask
            past_key_values: Optional cached key-value states from previous generation steps
            use_cache: Whether to return key-value cache for generation
            output_hidden_states: Whether to return hidden states (not used currently)
            return_dict: Whether to return dict output (always True)
            image_sizes: Original image sizes (batch, 2) for building encoder padding mask

        Returns:
            SMTOutput containing logits, loss, and optional cache
        """
        use_cache = use_cache if use_cache is not None else False
        return_dict = return_dict if return_dict is not None else True

        # Handle decoder input construction
        if input_ids is None:
            if labels is None:
                raise ValueError("Provide either input_ids or labels.")
            input_ids = self.prepare_decoder_input_ids_from_labels(labels)

        # Build decoder attention mask if not provided
        if attention_mask is None and labels is not None:
            # True = keep
            attention_mask = labels.ne(-100)

        # Encode images (or use pre-computed encoder outputs for generation)
        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError(
                    "You have to specify either pixel_values or encoder_outputs for the first generation step."
                )
            encoder_features = self.forward_encoder(pixel_values, image_sizes=image_sizes)
        else:
            encoder_features = self._coerce_encoder_outputs(
                encoder_outputs=encoder_outputs,
                encoder_attention_mask=encoder_attention_mask,
            )

        # Build or use memory padding mask for encoder features
        memory_key_padding_mask = encoder_attention_mask
        if memory_key_padding_mask is None:
            memory_key_padding_mask = encoder_features.encoder_attention_mask

        # Pass through decoder with caching support
        output = self.forward_decoder(
            encoder_features,
            input_ids,
            tgt_key_padding_mask=attention_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # Compute loss if labels provided
        if labels is not None:
            output.loss = self.loss(output.logits.permute(0, 2, 1), labels)

        # Expose encoder state so generation kwargs can persist it between steps.
        output["encoder_outputs"] = encoder_features
        output["encoder_attention_mask"] = memory_key_padding_mask

        return output

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        pixel_values=None,
        image_sizes=None,
        encoder_outputs=None,
        encoder_attention_mask=None,
        **kwargs,
    ):
        """
        Prepare inputs for each generation step.

        This method is called by HuggingFace's generate() to prepare the inputs
        for each forward pass during autoregressive generation.

        Args:
            input_ids: Current sequence of generated token IDs
            past_key_values: Cached key-value states from previous steps
            attention_mask: Attention mask for decoder (can be long or bool dtype)
            pixel_values: Input images (batch, 3, height, width) - only used on first step
            image_sizes: Original image sizes for building encoder padding mask
            encoder_outputs: Cached encoder outputs (from first step)
            encoder_attention_mask: Cached encoder attention mask (from first step)
            **kwargs: Additional arguments

        Returns:
            dict: Model inputs ready for forward pass
        """
        # Convert attention_mask to bool if provided (HF often uses long dtype)
        if attention_mask is not None:
            attention_mask = attention_mask.bool()

        # SUBSEQUENT STEPS: past_key_values has content, just slice input_ids and pass through state
        # Note: In transformers 4.50+, past_key_values may be an empty DynamicCache object
        # instead of None, so we check if it actually has content using get_seq_length()
        has_past = past_key_values is not None and (
            # For Cache objects (transformers 4.50+), check if they have content
            (hasattr(past_key_values, "get_seq_length") and past_key_values.get_seq_length() > 0)
            # For legacy tuple format, check if non-empty
            or (isinstance(past_key_values, tuple) and len(past_key_values) > 0)
        )
        if has_past:
            # Only pass the last token to the decoder for incremental generation
            input_ids = input_ids[:, -1:]
            if encoder_outputs is None:
                raise ValueError("encoder_outputs must be preserved across generation steps")

            return {
                **kwargs,  # Forward remaining kwargs (required for transformers 4.50+)
                "input_ids": input_ids,
                "encoder_outputs": encoder_outputs,
                "encoder_attention_mask": encoder_attention_mask,
                "past_key_values": past_key_values,
                "cache_position": cache_position,
                "use_cache": True,
                "attention_mask": None,  # Not needed for single token
            }

        # FIRST STEP: No cache yet, need to encode images and build masks
        if pixel_values is None and encoder_outputs is None:
            raise ValueError("pixel_values must be provided for the first generation step")

        # Encode images if encoder_outputs not already provided
        if encoder_outputs is None:
            # If input_ids is None, create initial BOS tokens
            if input_ids is None:
                batch_size = pixel_values.shape[0]
                input_ids = torch.full(
                    (batch_size, 1),
                    self.config.bos_token_id,
                    dtype=torch.long,
                    device=pixel_values.device,
                )

            # Encode images
            encoder_outputs = self.forward_encoder(pixel_values, image_sizes=image_sizes)
        else:
            encoder_outputs = self._coerce_encoder_outputs(
                encoder_outputs=encoder_outputs,
                encoder_attention_mask=encoder_attention_mask,
            )

        # Reuse pre-built encoder mask from frontend outputs when available
        if encoder_attention_mask is None:
            encoder_attention_mask = encoder_outputs.encoder_attention_mask

        # Create decoder attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        # Return all state for first step
        return {
            **kwargs,  # Forward remaining kwargs (required for transformers 4.50+)
            "input_ids": input_ids,
            "pixel_values": pixel_values,  # Include for forward() on first step
            "image_sizes": image_sizes,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": encoder_attention_mask,
            "past_key_values": None,
            "cache_position": cache_position,
            "use_cache": True,
            "attention_mask": attention_mask,
        }

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder=False,
        **kwargs,
    ):
        """
        Update model kwargs for generation to ensure encoder_outputs and encoder_attention_mask
        are passed through to subsequent generation steps.

        We delegate standard updates to the parent implementation and then persist
        custom encoder keys required by this model.

        Args:
            outputs: Model outputs from the previous step
            model_kwargs: Current model kwargs
            is_encoder_decoder: Whether this is an encoder-decoder model
            **kwargs: Additional arguments

        Returns:
            dict: Updated model_kwargs with past_key_values and encoder state
        """
        updated_model_kwargs = super()._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )

        # Preserve custom encoder state required by this model's forward signature.
        if "encoder_outputs" in model_kwargs:
            updated_model_kwargs["encoder_outputs"] = model_kwargs["encoder_outputs"]
        elif "encoder_outputs" in outputs:
            updated_model_kwargs["encoder_outputs"] = outputs["encoder_outputs"]
        if "encoder_attention_mask" in model_kwargs:
            updated_model_kwargs["encoder_attention_mask"] = model_kwargs["encoder_attention_mask"]
        elif "encoder_attention_mask" in outputs:
            updated_model_kwargs["encoder_attention_mask"] = outputs["encoder_attention_mask"]

        # Pixel tensors are only needed for the first generation step (when encoder runs).
        updated_model_kwargs.pop("pixel_values", None)

        return updated_model_kwargs

    @staticmethod
    def _assert_no_legacy_projector_keys(state_dict: dict[str, torch.Tensor]) -> None:
        has_legacy_projector = any(
            key.startswith("encoder_to_decoder_projection.") for key in state_dict
        )
        has_new_projector = any(key.startswith("frontend.projector.") for key in state_dict)

        if has_legacy_projector and not has_new_projector:
            raise RuntimeError(
                "Checkpoint uses legacy single-layer projector keys "
                "('encoder_to_decoder_projection.*') and is incompatible with the current "
                "2-layer MLP projector ('frontend.projector.*')."
            )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self._assert_no_legacy_projector_keys(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorder cached key-value states for beam search.

        This method is called by HuggingFace's beam search to reorder the cache
        when beam hypotheses are rearranged.

        Args:
            past_key_values: Cached key-value states from previous steps
                Structure: tuple of ((self_k, self_v), (cross_k, cross_v)) per layer
                Each key/value has shape (batch_size, num_heads, seq_len, head_dim)
            beam_idx: Tensor of beam indices to select (batch_size,)

        Returns:
            tuple: Reordered past_key_values with the same structure
        """
        reordered_past = ()
        for layer_past in past_key_values:
            # layer_past is ((self_k, self_v), (cross_k, cross_v))
            self_attn_cache, cross_attn_cache = layer_past

            # Reorder self-attention cache
            reordered_self = tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in self_attn_cache
            )

            # Reorder cross-attention cache
            reordered_cross = tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in cross_attn_cache
            )

            reordered_past += ((reordered_self, reordered_cross),)
        return reordered_past

    def _build_memory_key_padding_mask(self, image_sizes, encoder_output, encoder_input):
        """
        Build a boolean mask marking valid encoder positions based on runtime feature map ratios.

        Args:
            image_sizes (torch.Tensor): Tensor of original (H, W) of each image in the batch.
            encoder_output (torch.Tensor): The output feature map from the encoder.
            encoder_input (torch.Tensor): The padded input tensor fed to the encoder.

        Returns:
            torch.Tensor: A boolean mask of shape (batch_size, num_encoder_tokens).
                        `True` indicates a valid position.
        """
        if isinstance(encoder_output, VisionFrontendOutput):
            encoder_output = encoder_output.encoder_tokens_raw

        if isinstance(encoder_output, torch.Tensor) and encoder_output.ndim == 4:
            encoder_hw = (encoder_output.shape[-2], encoder_output.shape[-1])
        else:
            encoder_hw = getattr(self.frontend, "_last_encoder_hw", None)

        if encoder_hw is None:
            raise ValueError(
                "Cannot infer encoder feature map size for memory mask. "
                "Run forward_encoder first or provide 4D encoder_output."
            )

        return self.frontend.build_memory_key_padding_mask(
            image_sizes=image_sizes,
            encoder_hw=encoder_hw,
            input_hw=(encoder_input.shape[-2], encoder_input.shape[-1]),
            device=encoder_input.device,
        )
