from __future__ import annotations

import math
import re
from decimal import Decimal
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from transcoda.metrics_schema import DEFAULT_CHECKPOINT_MONITOR
from transcoda.validation.metrics.verosim_options import (
    DEFAULT_VEROSIM_MODE,
    normalize_verosim_mode,
)

SPINE_STRUCTURE_BARLINE_WIDTH_MODES = {"per_spine", "compact", "both"}
SPINE_STRUCTURE_EMPTY_FIELD_MODES = {"forbid", "data_only"}


class TokenizerConfig(BaseModel):
    """Configuration for the tokenizer (word-level or BPE)."""

    model_config = ConfigDict(extra="forbid")

    type: str = "word"  # "word" | "bpe"
    vocab_size: int | None = None
    min_freq: int = 2


_RELEASE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class TrainSource(BaseModel):
    """One train release the sampler draws from, with its share of every epoch.

    ``release`` names the release (``data/releases/<release>/``; under lab, a
    pinned train alias). Every row of the source carries it as
    ``train_source``. ``path`` is the release's HF snapshot directory; when
    omitted it is read from ``data/releases/<release>/release.json``. A zero
    ``weight`` keeps the source loaded and pinned but never draws it.
    """

    model_config = ConfigDict(extra="forbid")

    release: str
    weight: float = Field(ge=0, le=1)
    path: str | None = None
    max_repeats_per_epoch: int | None = Field(default=None, ge=1)

    @field_validator("release")
    @classmethod
    def _release_name(cls, value: str) -> str:
        if not _RELEASE_NAME.fullmatch(value):
            raise ValueError(
                f"train source release must be letters, digits, '.', '_' or '-': {value!r}"
            )
        return value

    @field_validator("weight")
    @classmethod
    def _finite_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("train source weights must be finite")
        return value


class Data(BaseModel):
    """Configuration for dataset paths and preprocessing."""

    model_config = ConfigDict(extra="forbid")

    # Exactly one of train_path (one HF dataset, sampled by mixture_path or
    # the legacy flags) and train_sources (several releases, sampled by their
    # weights).
    train_path: str | None = None  # Direct path to training dataset (arrow files)
    train_sources: list[TrainSource] | None = None
    validation_paths: dict[str, str]  # Named validation sets: {"synth": "...", "real": "..."}
    mixture_path: str | None = None  # Optional tracked transcoda.mixture.v1 policy
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

    @model_validator(mode="after")
    def validate_train_sources(self) -> Data:
        if (self.train_path is None) == (self.train_sources is None):
            raise ValueError("set exactly one of data.train_path and data.train_sources")
        if self.train_sources is None:
            return self
        if not self.train_sources:
            raise ValueError("data.train_sources must list at least one release")
        names = [source.release for source in self.train_sources]
        if len(names) != len(set(names)):
            raise ValueError(f"data.train_sources releases must be unique: {names}")
        total = sum(Decimal(str(source.weight)) for source in self.train_sources)
        if total != Decimal("1"):
            raise ValueError(f"data.train_sources weights must sum to 1, got {total}")
        if self.mixture_path is not None:
            raise ValueError("data.train_sources carries its own weights; drop data.mixture_path")
        return self

    @model_serializer(mode="wrap")
    def _omit_absent_train_sources(self, handler: SerializerFunctionWrapHandler) -> Any:
        # Configs without train_sources serialize exactly as before the field
        # existed, so their config digests and fingerprints do not move.
        payload = handler(self)
        if self.train_sources is None and isinstance(payload, dict):
            payload.pop("train_sources", None)
        return payload


class Training(BaseModel):
    """Configuration for training loop and data loading parameters."""

    model_config = ConfigDict(extra="forbid")

    # Data loading (hardware-dependent)
    batch_size: int = 1
    val_batch_size: int | None = None  # Defaults to batch_size if not specified
    num_workers: int = 0  # Train dataloader workers
    train_prefetch_factor: int = 4
    train_persistent_workers: bool = True
    train_pin_memory: bool = True
    # Ship uint8 pixels through the DataLoader and normalize on the GPU
    # (TranscodaTrainer.on_after_batch_transfer). Float32 normalization in
    # workers was the train_batch_wait_s bottleneck: 4x the IPC/pin bytes and
    # ~3x the worker CPU per sample.
    normalize_on_gpu: bool = True
    # Pad decoder batches to the smallest bucket that fits the batch instead
    # of always model.max_seq_len. Numerically exact (padding is trailing and
    # attention is causal); measured ~1.35x train-step compute at batch 12.
    # Each bucket adds one static torch.compile shape variant (one-time
    # max-autotune warmup each) and requires
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (train.py sets it).
    # None disables bucketing (single fixed length).
    decoder_length_buckets: list[int] | None = [768, 1024, 1536]
    # Optional source-domain weighting for mixed synthetic/real final datasets.
    # When set, the train sampler gives rows with source_domain="real" this
    # total probability mass per epoch while keeping the materialized dataset
    # provenance intact.
    real_scan_sample_probability: float | None = None
    # Optional separate mass for real no-music pages (curation_stage ==
    # "manual_review_no_music", empty targets). When set, real music rows get
    # real_scan_sample_probability, no-music rows get this mass, and synthetic
    # rows get the remainder. When None, all real rows share
    # real_scan_sample_probability uniformly (legacy behavior).
    real_scan_no_music_sample_probability: float | None = None
    # Optional oversampling of tuplet-rich rows in mixed training datasets.
    # When set, rows whose kern transcription carries a meaningful share of
    # non-power-of-two durations get this sampling-weight multiplier within
    # their source-domain group (the real/synth mass split from
    # real_scan_sample_probability is preserved). Counters the tuplet
    # exposure gap between the pdmx train mix and real validation scores.
    tuplet_oversample_factor: float | None = None
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
    limit_train_samples: int | None = None  # For profiling/debug setup without full-dataset map
    limit_train_batches: int | None = None  # For profiling - set to small number like 10
    limit_val_batches: int | float | None = None
    enable_profiling: bool = False  # Enable PyTorch profiler
    profiler_dirpath: str | None = None  # Profiler output directory (default: ./profiler_logs)
    profiler_wait_steps: int = 1
    profiler_warmup_steps: int = 1
    profiler_active_steps: int = 3
    profiler_repeat: int = 1
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
    # Training-step mask elision for fixed-size, teacher-forced batches. Decoder
    # padding is trailing and ignored by the loss; fixed image batches have no
    # encoder padding. Keeping these true avoids unnecessary attention masks in
    # the compiled training forward path while validation/generation remain masked.
    omit_train_decoder_attention_mask: bool = True
    omit_train_encoder_attention_mask: bool = True
    # Skip the automatic post-fit full validation pass. Useful for profiling and
    # short smoke runs that only need training-step measurements.
    skip_final_validation: bool = False
    # Skip the test phase after training completes
    skip_test: bool = False
    # OMR-NED metric computation for validate-only runs. This uses the external
    # VeroSim backend after validation predictions have been written.
    compute_omr_ned: bool = True
    omr_ned_kern_parser_binary: str | None = None
    omr_ned_verosim_binary: str | None = None
    omr_ned_mode: str = DEFAULT_VEROSIM_MODE
    omr_ned_detail: str | None = None
    omr_ned_jobs: int = 0
    # Grammar-constrained decoding during validation.
    # Ensures syntactically valid **kern output but adds overhead.
    # Disable during training for faster validation; enable for final evaluation.
    use_grammar_constraints: bool = False
    # Semantic decoder constraint preserving active spine width during constrained decoding.
    use_spine_structure_constraints: bool = True
    # Target-dialect width policy for barline records under spine-structure constraints.
    spine_structure_barline_width: str = "per_spine"
    # Target-dialect policy for omitted null data fields under spine-structure constraints.
    spine_structure_empty_field_mode: str = "forbid"
    # Grammar file used when use_grammar_constraints=True.
    grammar_path: str | None = None
    # Hard decode-time runaway guard: stop a sequence once the same complete
    # line has been generated this many times in a row. Unlike the
    # runaway_monitor_* diagnostics below, this changes generation. Gold
    # Polish validation scores top out at 5 consecutive identical lines;
    # observed runaway loops repeat one line 23-200 times.
    decode_max_identical_line_run: int | None = None
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
    min_fit_validation_step: int = 0
    tiered_validation_enabled: bool = True
    full_validation_every_n_steps: int = 5000
    frequent_validation_set_names: list[str] = Field(default_factory=lambda: ["real"])
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

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_omr_ned_detail(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        migrated = dict(data)
        detail = migrated.get("omr_ned_detail")
        if detail is not None and "omr_ned_mode" not in migrated:
            migrated["omr_ned_mode"] = normalize_verosim_mode(detail=str(detail))
        return migrated

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
    def validate_training_options(self) -> Training:
        if self.grammar_path is not None:
            self.grammar_path = self.grammar_path.strip() or None

        if self.use_grammar_constraints and self.grammar_path is None:
            raise ValueError(
                "training.grammar_path is required when "
                "training.use_grammar_constraints=true"
            )

        if not self.use_grammar_constraints:
            explicit_spine = "use_spine_structure_constraints" in self.model_fields_set
            if explicit_spine and self.use_spine_structure_constraints:
                raise ValueError(
                    "training.use_spine_structure_constraints requires "
                    "training.use_grammar_constraints=true"
                )
            self.use_spine_structure_constraints = False

        if self.spine_structure_barline_width not in SPINE_STRUCTURE_BARLINE_WIDTH_MODES:
            raise ValueError(
                "training.spine_structure_barline_width must be one of "
                f"{sorted(SPINE_STRUCTURE_BARLINE_WIDTH_MODES)}"
            )
        if self.spine_structure_empty_field_mode not in SPINE_STRUCTURE_EMPTY_FIELD_MODES:
            raise ValueError(
                "training.spine_structure_empty_field_mode must be one of "
                f"{sorted(SPINE_STRUCTURE_EMPTY_FIELD_MODES)}"
            )

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
        if self.min_fit_validation_step < 0:
            raise ValueError("training.min_fit_validation_step must be >= 0")
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
        if self.limit_train_samples is not None and self.limit_train_samples < 1:
            raise ValueError("training.limit_train_samples must be >= 1")
        if self.profiler_wait_steps < 0:
            raise ValueError("training.profiler_wait_steps must be >= 0")
        if self.profiler_warmup_steps < 0:
            raise ValueError("training.profiler_warmup_steps must be >= 0")
        if self.profiler_active_steps < 1:
            raise ValueError("training.profiler_active_steps must be >= 1")
        if self.profiler_repeat < 1:
            raise ValueError("training.profiler_repeat must be >= 1")
        if self.real_scan_sample_probability is not None and not (
            0 < self.real_scan_sample_probability < 1
        ):
            raise ValueError("training.real_scan_sample_probability must be between 0 and 1")
        if self.real_scan_no_music_sample_probability is not None:
            if not 0 < self.real_scan_no_music_sample_probability < 1:
                raise ValueError(
                    "training.real_scan_no_music_sample_probability must be between 0 and 1"
                )
            if self.real_scan_sample_probability is None:
                raise ValueError(
                    "training.real_scan_no_music_sample_probability requires "
                    "training.real_scan_sample_probability to be set"
                )
            if (
                self.real_scan_sample_probability + self.real_scan_no_music_sample_probability
                >= 1
            ):
                raise ValueError(
                    "training.real_scan_sample_probability plus "
                    "training.real_scan_no_music_sample_probability must be less than 1"
                )
        if self.tuplet_oversample_factor is not None and self.tuplet_oversample_factor <= 1.0:
            raise ValueError("training.tuplet_oversample_factor must be > 1.0")
        if self.decode_max_identical_line_run is not None and (
            self.decode_max_identical_line_run < 2
        ):
            raise ValueError("training.decode_max_identical_line_run must be >= 2")
        try:
            self.omr_ned_mode = normalize_verosim_mode(
                mode=self.omr_ned_mode,
                detail=self.omr_ned_detail,
            )
        except ValueError as exc:
            raise ValueError(
                "training.omr_ned_mode must be one of active, experimental; "
                "legacy training.omr_ned_detail must be one of tierA, tierAB, tierAB_dir"
            ) from exc
        if self.omr_ned_jobs < 0:
            raise ValueError("training.omr_ned_jobs must be >= 0")

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
    validation_length_multiplier: float = 1.5
    validation_length_buffer: int = 32
    # Absolute validation/test decode cap (Hugging Face max_length: counts the
    # BOS prompt token and EOS). When set it replaces the target-relative cap,
    # so a page's decode no longer depends on its target or batch neighbours.
    # None keeps the relative cap and existing config digests.
    validation_absolute_max_length: int | None = None
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
        if self.num_return_sequences != 1:
            raise ValueError(
                "generation.num_return_sequences must be 1; validation and inference "
                "currently expect exactly one generated sequence per input sample"
            )

        if self.repetition_penalty <= 0:
            raise ValueError("generation.repetition_penalty must be > 0")

        if self.validation_length_multiplier <= 0:
            raise ValueError("generation.validation_length_multiplier must be > 0")

        if self.validation_length_buffer < 0:
            raise ValueError("generation.validation_length_buffer must be >= 0")

        if (
            self.validation_absolute_max_length is not None
            and self.validation_absolute_max_length < 2
        ):
            raise ValueError("generation.validation_absolute_max_length must be >= 2")

        if self.do_sample:
            raise ValueError("generation.do_sample=True is not supported in this project")

        return self


class Checkpoint(BaseModel):
    """Configuration for model checkpointing and W&B logging."""

    model_config = ConfigDict(extra="forbid")

    dirpath: str = "weights/transcoda/"
    filename: str = "transcoda-model"
    monitor: str = DEFAULT_CHECKPOINT_MONITOR
    mode: str = "min"
    save_top_k: int = 1
    save_last: bool = True
    verbose: bool = True
    project: str = "Transcoda-FP"
    run_name: str = "Transcoda-System-level"
    tags: list[str] = Field(  # W&B tags for filtering (e.g., ["scale-study", "data-50k"])
        default_factory=list
    )
    auto_resume: bool = False
    # Commit a durable checkpoint every N optimizer steps and at fit end
    # (transcoda/callbacks/durable_checkpoint.py). None keeps resolved configs
    # and their digests unchanged.
    durable_every_steps: int | None = Field(default=None, ge=1)


class ValidationDecoding(BaseModel):
    """Validation-time constrained decoding settings."""

    model_config = ConfigDict(extra="forbid")

    use_grammar_constraints: bool = False
    use_spine_structure_constraints: bool = True
    spine_structure_barline_width: str = "per_spine"
    spine_structure_empty_field_mode: str = "forbid"
    grammar_path: str | None = None

    @model_validator(mode="after")
    def validate_decoding(self) -> ValidationDecoding:
        if self.grammar_path is not None:
            self.grammar_path = self.grammar_path.strip() or None

        if self.use_grammar_constraints and self.grammar_path is None:
            raise ValueError(
                "validation.decoding.grammar_path is required when "
                "validation.decoding.use_grammar_constraints=true"
            )

        if not self.use_grammar_constraints:
            explicit_spine = "use_spine_structure_constraints" in self.model_fields_set
            if explicit_spine and self.use_spine_structure_constraints:
                raise ValueError(
                    "validation.decoding.use_spine_structure_constraints requires "
                    "validation.decoding.use_grammar_constraints=true"
                )
            self.use_spine_structure_constraints = False

        if self.spine_structure_barline_width not in SPINE_STRUCTURE_BARLINE_WIDTH_MODES:
            raise ValueError(
                "validation.decoding.spine_structure_barline_width must be one of "
                f"{sorted(SPINE_STRUCTURE_BARLINE_WIDTH_MODES)}"
            )
        if self.spine_structure_empty_field_mode not in SPINE_STRUCTURE_EMPTY_FIELD_MODES:
            raise ValueError(
                "validation.decoding.spine_structure_empty_field_mode must be one of "
                f"{sorted(SPINE_STRUCTURE_EMPTY_FIELD_MODES)}"
            )

        return self


class ValidationVeroSim(BaseModel):
    """VeroSim OMR-NED validation metric settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    kern_parser_binary: str | None = None
    verosim_binary: str | None = None
    mode: str = DEFAULT_VEROSIM_MODE
    jobs: int = 0
    # Score VeroSim gives a page it cannot score (`--failure-score`). None
    # keeps VeroSim's default (100) and existing config digests.
    failure_score: float | None = None
    # OMR-NED target: the tokenizer round trip of the label ids (None or
    # "round_trip", the historical behaviour) or the dataset's stored label
    # ("stored"). None keeps existing config digests.
    target_text: Literal["round_trip", "stored"] | None = None
    # Validation sets VeroSim never scores: no-music sets, whose empty targets
    # have nothing to score. None keeps existing config digests.
    exclude_val_sets: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_detail(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        migrated = dict(data)
        detail = migrated.pop("detail", None)
        if detail is not None:
            migrated["mode"] = normalize_verosim_mode(
                mode=migrated.get("mode"),
                detail=str(detail),
            )
        return migrated

    @model_validator(mode="after")
    def validate_verosim(self) -> ValidationVeroSim:
        try:
            self.mode = normalize_verosim_mode(mode=self.mode)
        except ValueError as exc:
            raise ValueError(
                "validation.metrics.verosim.mode must be one of active, experimental"
            ) from exc
        if self.jobs < 0:
            raise ValueError("validation.metrics.verosim.jobs must be >= 0")
        return self


class ValidationMetrics(BaseModel):
    """Validation metric backend settings."""

    model_config = ConfigDict(extra="forbid")

    verosim: ValidationVeroSim = Field(default_factory=ValidationVeroSim)


class ValidationSchedule(BaseModel):
    """Validation scheduling settings for tiered and expensive passes."""

    model_config = ConfigDict(extra="forbid")

    tiered_validation_enabled: bool = True
    min_fit_validation_step: int = 0
    full_validation_every_n_steps: int = 5000
    frequent_validation_set_names: list[str] = Field(default_factory=lambda: ["real"])
    frequent_validation_subset_sizes: dict[str, int] = Field(default_factory=dict)
    frequent_validation_subset_seed: int = 0

    @model_validator(mode="after")
    def validate_schedule(self) -> ValidationSchedule:
        if self.full_validation_every_n_steps < 1:
            raise ValueError("validation.schedule.full_validation_every_n_steps must be >= 1")
        if self.min_fit_validation_step < 0:
            raise ValueError("validation.schedule.min_fit_validation_step must be >= 0")
        for set_name, subset_size in self.frequent_validation_subset_sizes.items():
            if subset_size < 1:
                raise ValueError(
                    "validation.schedule.frequent_validation_subset_sizes values must be >= 1 "
                    f"(got {subset_size} for {set_name!r})"
                )
        return self


class ValidationSelection(BaseModel):
    """Configuration for OMR-NED-first checkpoint selection."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    primary: str = "val/real/OMR_NED"
    fallback: str | None = None
    mode: str = "min"
    omr_ned_min_step: int = 20_000
    min_samples: int = 1
    max_verosim_failure_rate: float | None = 0.10
    max_prediction_truncation_rate: float | None = 0.02
    max_runaway_rate: float | None = 0.02

    @model_validator(mode="after")
    def validate_selection(self) -> ValidationSelection:
        if not self.primary.strip():
            raise ValueError("validation_selection.primary must be non-empty")
        self.primary = self.primary.strip()

        if self.fallback is not None:
            self.fallback = self.fallback.strip() or None

        if self.mode not in {"min", "max"}:
            raise ValueError("validation_selection.mode must be one of ['max', 'min']")
        if self.omr_ned_min_step < 0:
            raise ValueError("validation_selection.omr_ned_min_step must be >= 0")
        if self.min_samples < 1:
            raise ValueError("validation_selection.min_samples must be >= 1")

        for field_name in (
            "max_verosim_failure_rate",
            "max_prediction_truncation_rate",
            "max_runaway_rate",
        ):
            value = getattr(self, field_name)
            if value is not None and not (0.0 <= value <= 1.0):
                raise ValueError(f"validation_selection.{field_name} must be between 0 and 1")

        return self


class ValidationConfig(BaseModel):
    """Top-level validation behavior, metrics, schedule, and selection."""

    model_config = ConfigDict(extra="forbid")

    decoding: ValidationDecoding = Field(default_factory=ValidationDecoding)
    metrics: ValidationMetrics = Field(default_factory=ValidationMetrics)
    schedule: ValidationSchedule = Field(default_factory=ValidationSchedule)
    selection: ValidationSelection = Field(default_factory=ValidationSelection)


class ModelConfig(BaseModel):
    """Configuration for the Transcoda model architecture."""

    model_config = ConfigDict(extra="forbid")

    d_model: int = 256
    dim_ff: int = 256
    num_hidden_layers: int = 8
    num_attn_heads: int = 4
    encoder_model_name_or_path: str = "facebook/convnextv2-tiny-22k-224"
    encoder_provider: str = "transformers"
    freeze_encoder_stages: int = 0
    vision_frontend: str = "conv"  # "conv" | "conv_multiscale" | "swin"
    projector_hidden_mult: float = 4.0
    multiscale_hidden_state_index: int = -2
    multiscale_fusion_init_logit: float = -4.0

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


def _dict_value(parent: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = parent.get(key)
    if value is None:
        value = {}
    if not isinstance(value, dict):
        return None
    copied = dict(value)
    parent[key] = copied
    return copied


def _copy_aliases(
    source: dict[str, Any],
    target: dict[str, Any],
    aliases: dict[str, str],
) -> None:
    for source_key, target_key in aliases.items():
        if source_key in source:
            target[target_key] = source[source_key]


def _drop_ignored_keys(mapping: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        mapping.pop(key, None)


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=42, ge=0)
    data: Data
    checkpoint: Checkpoint
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    training: Training = Field(default_factory=Training)
    generation: Generation = Field(default_factory=Generation)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    validation_selection: ValidationSelection = Field(default_factory=ValidationSelection)

    @model_validator(mode="before")
    @classmethod
    def migrate_validation_config(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        migrated = dict(data)
        training = migrated.get("training")
        if isinstance(training, dict):
            training_values = dict(training)
            migrated["training"] = training_values
            _drop_ignored_keys(training_values, ("use_rhythm_constraints",))
        else:
            training_values = {}
        validation = _dict_value(migrated, "validation")
        if validation is None:
            return migrated

        if "validation_selection" in migrated:
            validation["selection"] = migrated["validation_selection"]

        decoding = _dict_value(validation, "decoding")
        if decoding is not None:
            _drop_ignored_keys(decoding, ("use_rhythm_constraints",))
            _copy_aliases(
                training_values,
                decoding,
                {
                    "use_grammar_constraints": "use_grammar_constraints",
                    "use_spine_structure_constraints": "use_spine_structure_constraints",
                    "spine_structure_barline_width": "spine_structure_barline_width",
                    "spine_structure_empty_field_mode": "spine_structure_empty_field_mode",
                    "grammar_path": "grammar_path",
                },
            )

        metrics = _dict_value(validation, "metrics")
        if metrics is not None:
            verosim = _dict_value(metrics, "verosim")
            if verosim is not None:
                _copy_aliases(
                    training_values,
                    verosim,
                    {
                        "compute_omr_ned": "enabled",
                        "omr_ned_kern_parser_binary": "kern_parser_binary",
                        "omr_ned_verosim_binary": "verosim_binary",
                        "omr_ned_mode": "mode",
                        "omr_ned_detail": "detail",
                        "omr_ned_jobs": "jobs",
                    },
                )

        schedule = _dict_value(validation, "schedule")
        if schedule is not None:
            _copy_aliases(
                training_values,
                schedule,
                {
                    "tiered_validation_enabled": "tiered_validation_enabled",
                    "min_fit_validation_step": "min_fit_validation_step",
                    "full_validation_every_n_steps": "full_validation_every_n_steps",
                    "frequent_validation_set_names": "frequent_validation_set_names",
                    "frequent_validation_subset_sizes": "frequent_validation_subset_sizes",
                    "frequent_validation_subset_seed": "frequent_validation_subset_seed",
                },
            )

        return migrated

    @model_validator(mode="after")
    def validate_validation_subset_config(self) -> ExperimentConfig:
        legacy_mixture_fields = {
            "training.real_scan_sample_probability": self.training.real_scan_sample_probability,
            "training.real_scan_no_music_sample_probability": (
                self.training.real_scan_no_music_sample_probability
            ),
            "training.tuplet_oversample_factor": self.training.tuplet_oversample_factor,
        }
        configured_legacy = [
            name for name, value in legacy_mixture_fields.items() if value is not None
        ]
        if self.data.mixture_path is not None and configured_legacy:
            raise ValueError(
                "data.mixture_path cannot be combined with legacy sampling flags: "
                f"{configured_legacy}"
            )
        if self.data.train_sources is not None and configured_legacy:
            raise ValueError(
                "data.train_sources cannot be combined with legacy sampling flags: "
                f"{configured_legacy}"
            )
        unknown_names = sorted(
            set(self.validation.schedule.frequent_validation_subset_sizes)
            - set(self.data.validation_paths)
        )
        if unknown_names:
            raise ValueError(
                "validation.schedule.frequent_validation_subset_sizes keys must exist in "
                f"data.validation_paths; unknown keys: {unknown_names}"
            )
        if self.validation.selection.fallback is None:
            self.validation.selection.fallback = self.checkpoint.monitor
        self.validation_selection = self.validation.selection
        self._sync_validation_to_training()
        return self

    def _sync_validation_to_training(self) -> None:
        decoding = self.validation.decoding
        self.training.use_grammar_constraints = decoding.use_grammar_constraints
        self.training.use_spine_structure_constraints = decoding.use_spine_structure_constraints
        self.training.spine_structure_barline_width = decoding.spine_structure_barline_width
        self.training.spine_structure_empty_field_mode = decoding.spine_structure_empty_field_mode
        self.training.grammar_path = decoding.grammar_path

        verosim = self.validation.metrics.verosim
        self.training.compute_omr_ned = verosim.enabled
        self.training.omr_ned_kern_parser_binary = verosim.kern_parser_binary
        self.training.omr_ned_verosim_binary = verosim.verosim_binary
        self.training.omr_ned_mode = verosim.mode
        self.training.omr_ned_detail = None
        self.training.omr_ned_jobs = verosim.jobs

        schedule = self.validation.schedule
        self.training.tiered_validation_enabled = schedule.tiered_validation_enabled
        self.training.min_fit_validation_step = schedule.min_fit_validation_step
        self.training.full_validation_every_n_steps = schedule.full_validation_every_n_steps
        self.training.frequent_validation_set_names = list(schedule.frequent_validation_set_names)
        self.training.frequent_validation_subset_sizes = dict(
            schedule.frequent_validation_subset_sizes
        )
        self.training.frequent_validation_subset_seed = schedule.frequent_validation_subset_seed


def experiment_config_from_dict(s: Any) -> ExperimentConfig:
    """Load ExperimentConfig from a dictionary (backward compatibility wrapper)."""
    return ExperimentConfig.model_validate(s)


def experiment_config_to_dict(x: ExperimentConfig) -> dict[str, Any]:
    """Convert ExperimentConfig to a dictionary (backward compatibility wrapper)."""
    return x.model_dump(exclude_none=True)
