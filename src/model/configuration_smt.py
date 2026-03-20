from transformers import PretrainedConfig


class SMTConfig(PretrainedConfig):
    model_type = "SMT"

    def __init__(
        self,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        maxh=2512,
        maxw=2512,
        maxlen=8000,
        out_categories=2512,
        vocab_size: int | None = None,
        w2i: dict | None = None,
        i2w: dict | None = None,
        out_dir="out_smt",
        d_model=256,
        dim_ff=256,
        num_hidden_layers=8,
        num_attn_heads=4,
        encoder_model_name_or_path: str = "facebook/convnextv2-tiny-22k-224",
        encoder_provider: str = "transformers",
        freeze_encoder_stages: int = 0,
        vision_frontend: str = "conv",
        projector_hidden_mult: float = 4.0,
        positional_encoding: str = "absolute",
        rope_theta: float = 10000.0,
        **kwargs,
    ):
        # Initialize parent class first with special token IDs
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        # Set custom attributes
        self.architectures = ["SMT"]
        self.maxh = maxh
        self.maxw = maxw
        self.maxlen = maxlen
        resolved_out_categories = int(out_categories)
        resolved_vocab_size = resolved_out_categories if vocab_size is None else int(vocab_size)
        if resolved_vocab_size != resolved_out_categories:
            raise ValueError("vocab_size and out_categories must match for SMTConfig")

        self.out_categories = resolved_out_categories
        self.vocab_size = resolved_vocab_size
        self.w2i = w2i if w2i is not None else {}
        self.i2w = i2w if i2w is not None else {}
        self.out_dir = out_dir
        self.d_model = d_model
        self.dim_ff = dim_ff
        self.num_attn_heads = num_attn_heads
        self.num_hidden_layers = num_hidden_layers
        self.encoder_model_name_or_path = encoder_model_name_or_path
        self.encoder_provider = encoder_provider
        self.freeze_encoder_stages = freeze_encoder_stages
        self.vision_frontend = vision_frontend
        self.projector_hidden_mult = projector_hidden_mult
        self.positional_encoding = positional_encoding
        self.rope_theta = rope_theta
