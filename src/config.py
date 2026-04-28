from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.metrics_schema import DEFAULT_CHECKPOINT_MONITOR


class TokenizerConfig(BaseModel):
    """Configuration for the tokenizer (word-level or BPE)."""

    model_config = ConfigDict(extra="forbid")

    type: str = "word"  # "word" | "bpe"
    vocab_size: int | None = None
    min_freq: int = 2


class Data(BaseModel):
    """Configuration for dataset paths and preprocessing."""

    model_config = ConfigDict(extra="forbid")

    train_path: str  # Direct path to training dataset (arrow files)
    validation_paths: dict[str, str]  # Named validation sets: {"synth": "...", "polish": "..."}
    vocab_dir: str = ""  # Computed from vocab_name via model_validator
    image_width: int = 1050  # Target image width in pixels
    fixed_image_height: int = 1485  # Fixed collate image height (H) for compile-stable batches
    fixed_image_width: int = 1050  # Fixed collate image width (W) for compile-stable batches

    @model_validator(mode="before")
    @classmethod
    def compute_vocab_dir(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Compute vocab_dir from vocab_name before validation."""
        if isinstance(data, dict) and "vocab_name" in data:
            vocab_name = data.pop("vocab_name")
            data["vocab_dir"] = f"./vocab/{vocab_name}"
        return data


class Training(BaseModel):
    """Configuration for training loop and data loading parameters."""

    model_config = ConfigDict(extra="forbid")

    # Data loading (hardware-dependent)
    batch_size: int = 1
    val_batch_size: int | None = None  # Defaults to batch_size if not specified
    num_workers: int = 0  # Train dataloader workers
    train_prefetch_factor: int = 8
    train_persistent_workers: bool = True
    train_pin_memory: bool = True
    # Validation dataloader settings (conservative defaults to avoid host RAM spikes)
    # val_num_workers=None means derive from train workers as min(2, num_workers)
    val_num_workers: int | None = None
    val_prefetch_factor: int = 2
    val_persistent_workers: bool = False
    val_pin_memory: bool = False

    # Training loop
    max_epochs: int = 10000
    max_steps: int | None = None
    min_steps: int | None = None
    val_check_interval: int = 2000  # Validate every N training steps
    limit_train_batches: int | None = None  # For profiling - set to small number like 10
    limit_val_batches: int | float | None = None
    enable_profiling: bool = False  # Enable PyTorch profiler
    profiler_dirpath: str | None = None  # Profiler output directory (default: ./profiler_logs)
    gradient_clip_val: float | None = 1.0
    gradient_clip_algorithm: str = "norm"
    accumulate_grad_batches: int | None = 8
    # Temporarily freeze the full vision encoder for the first N optimizer steps.
    # Permanent stage freezes remain controlled by model.freeze_encoder_stages.
    freeze_encoder_steps: int = 0
    # Slurm-friendly progress logging cadence and verbosity controls.
    progress_train_interval_seconds: float = 30.0
    progress_train_every_n_steps: int = 50
    progress_val_percent_interval: int = 10
    progress_enable_ascii_bar: bool = False
    # When True, logs a wandb.Table with validation examples containing input images, ground truth,
    # predictions, per-sample metrics (SER, CER), and an HTML diff for detailed analysis.
    log_example_images: bool = True
    # torch.compile settings for improved performance (10-25% speedup expected)
    compile_model: bool | None = None  # None = off, True = enable torch.compile
    compile_mode: str = "reduce-overhead"  # Options: "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
    # Skip the test phase after training completes
    skip_test: bool = False
    # OMR-NED metric computation during validation (slower, requires musicdiff).
    # Provides semantically-aware music notation comparison. Parse failures count
    # as 100.0 in the aggregate score and are tracked separately.
    # Install dependencies with: uv sync --group omr-ned
    compute_omr_ned: bool = False
    # Grammar-constrained decoding during validation.
    # Ensures syntactically valid **kern output but adds overhead.
    # Disable during training for faster validation; enable for final evaluation.
    use_grammar_constraints: bool = False
    # Structural bias for multi-spine line starts that should enter interpretation mode.
    use_interpretation_transition_constraints: bool = True
    interpretation_transition_non_spine_bonus: float = 1.5
    interpretation_transition_null_interp_bonus: float = 1.0
    interpretation_transition_data_start_penalty: float = -1.0
    interpretation_transition_barline_start_penalty: float = -1.0
    # Semantic decoder constraint preserving active spine width during constrained decoding.
    use_spine_structure_constraints: bool = True
    # Backward-compatible flag; inference currently ignores RhythmRule wiring.
    use_rhythm_constraints: bool = False
    # Grammar file used when use_grammar_constraints=True.
    grammar_path: str = "grammars/kern.gbnf"
    # Secondary decode-time guardrail to reduce control-token runaway loops.
    # Only active in validation/test generation when grammar constraints are enabled.
    runaway_guard_enabled: bool = True
    # Guardrail preset strictness: "lenient" | "moderate" | "strict".
    runaway_guard_strictness: str = "moderate"
    # Optional per-threshold overrides (None = use strictness preset values).
    runaway_guard_max_same_control_token: int | None = None
    runaway_guard_max_control_lines_streak: int | None = None
    runaway_guard_max_spine_splits: int | None = None
    runaway_guard_max_spine_merges: int | None = None
    runaway_guard_max_ottava_markers: int | None = None
    runaway_guard_max_tuplet_markers: int | None = None
    runaway_guard_max_tremolo_markers: int | None = None
    # Scalar runaway diagnostics for validation/test loops.
    runaway_monitor_enabled: bool = True
    runaway_monitor_strictness: str = "moderate"
    runaway_monitor_max_len_ratio: float | None = None
    runaway_monitor_repeat_ngram_size: int | None = None
    runaway_monitor_repeat_ngram_max_occurrences: int | None = None
    runaway_monitor_max_identical_line_run: int | None = None
    runaway_monitor_flag_no_eos_at_max_length: bool = True
    # Label smoothing for CrossEntropyLoss. Helps combat exposure bias by preventing
    # overconfident predictions that cascade during autoregressive decoding.
    label_smoothing: float = 0.0
    # Tiered validation: run frequent checks on a focused subset of validation sets
    # and periodic full checks on all sets.
    tiered_validation_enabled: bool = True
    full_validation_every_n_steps: int = 5000
    frequent_validation_set_names: list[str] = Field(default_factory=lambda: ["polish"])
    frequent_validation_subset_sizes: dict[str, int] = Field(default_factory=dict)
    frequent_validation_subset_seed: int = 0
    # Early stopping controls (monitor/mode come from checkpoint config)
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.01

    # DDP / Trainer plumbing
    devices: int = 1
    strategy: str = "auto"
    num_nodes: int = 1

    @model_validator(mode="after")
    def validate_ddp(self) -> Training:
        if self.devices < 1:
            raise ValueError("training.devices must be >= 1")
        if self.num_nodes < 1:
            raise ValueError("training.num_nodes must be >= 1")
        if self.strategy == "ddp_spawn":
            raise ValueError(
                "training.strategy='ddp_spawn' is not supported; use 'ddp' instead. "
                "ddp_spawn re-pickles the datamodule/tokenizer per subprocess and is "
                "fragile with our setup."
            )
        if self.devices * self.num_nodes > 1 and self.strategy == "auto":
            self.strategy = "ddp"
        return self

    @model_validator(mode="after")
    def validate_runaway_guard(self) -> Training:
        valid_strictness = {"lenient", "moderate", "strict"}
        if self.runaway_guard_strictness not in valid_strictness:
            raise ValueError(
                "training.runaway_guard_strictness must be one of "
                f"{sorted(valid_strictness)}"
            )

        if not self.use_grammar_constraints:
            explicit_transition = (
                "use_interpretation_transition_constraints" in self.model_fields_set
            )
            explicit_spine = "use_spine_structure_constraints" in self.model_fields_set
            explicit_rhythm = "use_rhythm_constraints" in self.model_fields_set
            if (
                explicit_transition and self.use_interpretation_transition_constraints
            ) or (explicit_spine and self.use_spine_structure_constraints) or (
                explicit_rhythm and self.use_rhythm_constraints
            ):
                raise ValueError(
                    "training.use_interpretation_transition_constraints, "
                    "training.use_spine_structure_constraints and "
                    "training.use_rhythm_constraints require "
                    "training.use_grammar_constraints=true"
                )
            self.use_interpretation_transition_constraints = False
            self.use_spine_structure_constraints = False
            self.use_rhythm_constraints = False

        if self.interpretation_transition_non_spine_bonus < 0:
            raise ValueError("training.interpretation_transition_non_spine_bonus must be >= 0")
        if self.interpretation_transition_null_interp_bonus < 0:
            raise ValueError("training.interpretation_transition_null_interp_bonus must be >= 0")
        if self.interpretation_transition_data_start_penalty > 0:
            raise ValueError("training.interpretation_transition_data_start_penalty must be <= 0")
        if self.interpretation_transition_barline_start_penalty > 0:
            raise ValueError(
                "training.interpretation_transition_barline_start_penalty must be <= 0"
            )

        override_values = (
            self.runaway_guard_max_same_control_token,
            self.runaway_guard_max_control_lines_streak,
            self.runaway_guard_max_spine_splits,
            self.runaway_guard_max_spine_merges,
            self.runaway_guard_max_ottava_markers,
            self.runaway_guard_max_tuplet_markers,
            self.runaway_guard_max_tremolo_markers,
        )
        for value in override_values:
            if value is not None and value < 1:
                raise ValueError("runaway_guard threshold overrides must be >= 1")

        valid_monitor_strictness = {"lenient", "moderate", "strict"}
        if self.runaway_monitor_strictness not in valid_monitor_strictness:
            raise ValueError(
                "training.runaway_monitor_strictness must be one of "
                f"{sorted(valid_monitor_strictness)}"
            )
        if (
            self.runaway_monitor_max_len_ratio is not None
            and self.runaway_monitor_max_len_ratio <= 1.0
        ):
            raise ValueError("training.runaway_monitor_max_len_ratio must be > 1.0")
        if (
            self.runaway_monitor_repeat_ngram_size is not None
            and self.runaway_monitor_repeat_ngram_size < 2
        ):
            raise ValueError("training.runaway_monitor_repeat_ngram_size must be >= 2")
        if (
            self.runaway_monitor_repeat_ngram_max_occurrences is not None
            and self.runaway_monitor_repeat_ngram_max_occurrences < 2
        ):
            raise ValueError("training.runaway_monitor_repeat_ngram_max_occurrences must be >= 2")
        if (
            self.runaway_monitor_max_identical_line_run is not None
            and self.runaway_monitor_max_identical_line_run < 2
        ):
            raise ValueError("training.runaway_monitor_max_identical_line_run must be >= 2")

        if self.progress_train_interval_seconds <= 0:
            raise ValueError("training.progress_train_interval_seconds must be > 0")
        if self.progress_train_every_n_steps < 1:
            raise ValueError("training.progress_train_every_n_steps must be >= 1")
        if not (1 <= self.progress_val_percent_interval <= 50):
            raise ValueError("training.progress_val_percent_interval must be between 1 and 50")
        if self.full_validation_every_n_steps < 1:
            raise ValueError("training.full_validation_every_n_steps must be >= 1")
        for set_name, subset_size in self.frequent_validation_subset_sizes.items():
            if subset_size < 1:
                raise ValueError(
                    "training.frequent_validation_subset_sizes values must be >= 1 "
                    f"(got {subset_size} for {set_name!r})"
                )
        if self.early_stopping_patience < 1:
            raise ValueError("training.early_stopping_patience must be >= 1")
        if self.early_stopping_min_delta < 0:
            raise ValueError("training.early_stopping_min_delta must be >= 0")
        if self.freeze_encoder_steps < 0:
            raise ValueError("training.freeze_encoder_steps must be >= 0")

        return self


class Generation(BaseModel):
    """Configuration for autoregressive decoding during inference/validation."""

    model_config = ConfigDict(extra="forbid")

    strategy: str = "beam"  # "greedy" | "beam"
    num_beams: int = 4
    length_penalty: float = 1.0
    repetition_penalty: float = 1.3
    early_stopping: bool | str = True
    max_length: int | None = None
    num_return_sequences: int = 1
    use_cache: bool = True
    do_sample: bool = False

    @model_validator(mode="after")
    def validate_generation(self) -> Generation:
        if self.strategy not in {"greedy", "beam"}:
            raise ValueError("generation.strategy must be 'greedy' or 'beam'")

        if self.strategy == "greedy":
            self.num_beams = 1
        elif self.num_beams <= 1:
            raise ValueError("generation.num_beams must be > 1 when strategy='beam'")

        if self.num_return_sequences < 1:
            raise ValueError("generation.num_return_sequences must be >= 1")

        if self.repetition_penalty <= 0:
            raise ValueError("generation.repetition_penalty must be > 0")

        if self.do_sample:
            raise ValueError("generation.do_sample=True is not supported in this project")

        return self


class Checkpoint(BaseModel):
    """Configuration for model checkpointing and W&B logging."""

    model_config = ConfigDict(extra="forbid")

    dirpath: str = "weights/GrandStaff/"
    filename: str = "GrandStaff_SMT_NexT"
    monitor: str = DEFAULT_CHECKPOINT_MONITOR
    mode: str = "min"
    save_top_k: int = 1
    save_last: bool = True
    verbose: bool = True
    project: str = "SMT-FP"
    run_name: str = "SMT-System-level"
    tags: list[str] = Field(  # W&B tags for filtering (e.g., ["scale-study", "data-50k"])
        default_factory=list
    )
    auto_resume: bool = False


class ModelConfig(BaseModel):
    """Configuration for the SMT model architecture."""

    model_config = ConfigDict(extra="forbid")

    d_model: int = 256
    dim_ff: int = 256
    num_hidden_layers: int = 8
    num_attn_heads: int = 4
    encoder_model_name_or_path: str = "facebook/convnextv2-tiny-22k-224"
    encoder_provider: str = "transformers"
    freeze_encoder_stages: int = 0
    vision_frontend: str = "conv"
    projector_hidden_mult: float = 4.0

    # Positional encoding configuration
    positional_encoding: str = "absolute"  # "absolute" | "rope"
    rope_theta: float = 10000.0  # Base frequency for RoPE (only used when positional_encoding="rope")

    # Input/output dimension constraints
    max_image_height: int = 2512  # Maximum input image height
    max_image_width: int = 2512  # Maximum input image width
    max_seq_len: int = 8000  # Maximum output sequence length


class OptimizerConfig(BaseModel):
    """Configuration for the optimizer and learning rate scheduler."""

    model_config = ConfigDict(extra="forbid")

    learning_rate: float = 3e-4
    encoder_lr_factor: float = 0.1
    weight_decay: float = 3e-3
    warmup_steps: int = 1000
    cosine_eta_min_factor: float = 0.1

    # LR scheduler: "cosine" (single decay) or "cosine_warm_restarts" (periodic restarts)
    lr_scheduler: str = "cosine"
    # CosineAnnealingWarmRestarts parameters (only when lr_scheduler="cosine_warm_restarts")
    cosine_restart_period: int = 5000  # T_0: steps in the first restart cycle
    cosine_restart_mult: int = 2  # T_mult: multiply period after each restart

    # Layer-wise learning rate decay (LLRD) settings
    layerwise_decay_gamma: float | None = None  # e.g., 0.75; None disables LLRD
    layerwise_target: str = "encoder"  # Which module to apply LLRD to (only "encoder" supported)


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    data: Data
    checkpoint: Checkpoint
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    training: Training = Field(default_factory=Training)
    generation: Generation = Field(default_factory=Generation)

    @model_validator(mode="after")
    def validate_validation_subset_config(self) -> ExperimentConfig:
        unknown_names = sorted(
            set(self.training.frequent_validation_subset_sizes) - set(self.data.validation_paths)
        )
        if unknown_names:
            raise ValueError(
                "training.frequent_validation_subset_sizes keys must exist in "
                f"data.validation_paths; unknown keys: {unknown_names}"
            )
        return self


def experiment_config_from_dict(s: Any) -> ExperimentConfig:
    """Load ExperimentConfig from a dictionary (backward compatibility wrapper)."""
    return ExperimentConfig.model_validate(s)


def experiment_config_to_dict(x: ExperimentConfig) -> dict[str, Any]:
    """Convert ExperimentConfig to a dictionary (backward compatibility wrapper)."""
    return x.model_dump(exclude_none=True)
