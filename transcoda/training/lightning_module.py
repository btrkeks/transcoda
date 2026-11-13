import math
from collections.abc import Callable

import lightning.pytorch as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch.optim.lr_scheduler import (
    LinearLR,
    SequentialLR,
)
from torchinfo import summary
from torchmetrics import Metric, MetricCollection

from transcoda.config import Generation, ModelConfig, OptimizerConfig, Training
from transcoda.core.metrics import (
    CharacterErrorRate,
    LineErrorRate,
    RunawayMonitorTracker,
    SymbolErrorRate,
    resolve_runaway_monitor_config,
)
from transcoda.grammar import GrammarProvider
from transcoda.grammar.constraint_factory import ConstrainedDecodingFactory
from transcoda.metrics_schema import (
    FINAL_VAL_PREFIX,
    TRAIN_LOSS,
    TRAIN_STAGE,
    VAL_PREFIX,
    base_val_set_name,
    build_test_metric_key,
    final_val_aggregate_metric,
    final_val_set_metric,
    is_subset_val_set_name,
    val_aggregate_metric,
    val_set_metric,
    val_subset_metric,
)
from transcoda.model import TranscodaConfig, TranscodaModelForCausalLM
from transcoda.model.generation_policy import settings_from_generation_config
from transcoda.preprocessing import normalize_uint8_pixel_values
from transcoda.training.optim.cosine import (
    make_cosine_annealing_lambda_lr,
    make_cosine_warm_restarts_lambda_lr,
)
from transcoda.training.optim.layerwise import (
    build_llrd_param_groups_for_convnextv2,
    split_named_params_for_weight_decay,
)
from transcoda.validation.decoding import (
    DecodePolicy,
    ValidationDecoder,
    target_relative_generation_cap,
    validation_generation_cap,
)

_ALLOWED_COMPILE_MODES = {
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}


def _finite_estimated_stepping_batches(trainer: L.Trainer) -> int | float:
    total_steps = getattr(trainer, "estimated_stepping_batches", None)
    if total_steps is None:
        raise RuntimeError("Trainer not attached yet: cannot infer total steps.")
    if not math.isfinite(float(total_steps)):
        raise RuntimeError(
            "Trainer estimated_stepping_batches must be finite to configure the "
            f"cosine scheduler; got {total_steps!r}."
        )
    return total_steps


def repeat_loop_filtered_set_names_for(
    val_set_names: list[str], *, runaway_monitor_enabled: bool
) -> list[str]:
    """Full validation sets that also log CER_no_repeat_loops (every set, whatever its name)."""
    if not runaway_monitor_enabled:
        return []
    return [name for name in val_set_names if not is_subset_val_set_name(name)]


def _validation_decode_policy_name(training: Training) -> str:
    parts = ["validation"]
    if training.use_grammar_constraints:
        parts.append("grammar")
        if training.use_spine_structure_constraints:
            parts.append("spine")
    else:
        parts.append("unconstrained")
    if training.decode_max_identical_line_run is not None:
        parts.append(f"lineguard{training.decode_max_identical_line_run}")
    return "+".join(parts)


def _target_relative_generation_cap(
    target_lengths: torch.Tensor,
    *,
    multiplier: float,
    buffer: int,
    maxlen: int,
    configured_max_length: int | None,
) -> int:
    return target_relative_generation_cap(
        target_lengths,
        multiplier=multiplier,
        buffer=buffer,
        maxlen=maxlen,
        configured_max_length=configured_max_length,
    )


class _TargetSymbolCount(Metric):
    """Target tokens a validation set scored, summed over ranks; zero for a no-music set."""

    full_state_update = False

    def __init__(self, *, pad_id: int, bos_id: int, eos_id: int) -> None:
        super().__init__()
        self._special_ids = (int(pad_id), int(bos_id), int(eos_id))
        self.add_state("symbols", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, target: torch.Tensor) -> None:  # type: ignore[override]
        content = torch.ones_like(target, dtype=torch.bool)
        for special_id in self._special_ids:
            content &= target != special_id
        self.symbols += content.sum().to(device=self.symbols.device, dtype=self.symbols.dtype)

    def compute(self) -> torch.Tensor:  # type: ignore[override]
        return self.symbols


def _scored_no_target_symbols(module, set_name: str) -> bool:
    """Whether a validation set's targets were all empty (no-music pages)."""
    counts = getattr(module, "val_target_symbols_by_set", None)
    if counts is None or set_name not in counts:
        return False
    return float(counts[set_name].compute().detach().cpu().item()) == 0.0


class _RepeatLoopFilteredSampleCounts(Metric):
    """Counts samples included/excluded from repeat-loop-filtered validation metrics."""

    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("included", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("excluded", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, included: int, excluded: int) -> None:  # type: ignore[override]
        device = self.included.device
        dtype = self.included.dtype
        self.included += torch.tensor(float(included), device=device, dtype=dtype)
        self.excluded += torch.tensor(float(excluded), device=device, dtype=dtype)

    def compute(self) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "CER_no_repeat_loops_samples": self.included,
            "repeat_loop_excluded_samples": self.excluded,
        }


class TranscodaTrainer(L.LightningModule):
    def __init__(
        self,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        training: Training,
        generation: Generation,
        maxh,
        maxw,
        maxlen,
        out_categories,
        pad_token_id,
        bos_token_id,
        eos_token_id,
        w2i,
        i2w,
        vocab_dir: str,
        val_set_names: list[str] | None = None,
        base_val_set_names: list[str] | None = None,
        show_model_summary: bool = False,
        run_artifact_json: str | None = None,
        total_training_steps: int | None = None,
    ):
        super().__init__()
        # Accepted only so Lightning can load older checkpoints whose saved
        # hyperparameters still contain this removed constructor argument.
        _ = total_training_steps

        self.config = TranscodaConfig(
            maxh=maxh,
            maxw=maxw,
            maxlen=maxlen,
            out_categories=out_categories,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            w2i=w2i,
            i2w=i2w,
            d_model=model_config.d_model,
            dim_ff=model_config.dim_ff,
            num_attn_heads=model_config.num_attn_heads,
            num_hidden_layers=model_config.num_hidden_layers,
            encoder_model_name_or_path=model_config.encoder_model_name_or_path,
            encoder_provider=model_config.encoder_provider,
            freeze_encoder_stages=model_config.freeze_encoder_stages,
            vision_frontend=model_config.vision_frontend,
            projector_hidden_mult=model_config.projector_hidden_mult,
            multiscale_hidden_state_index=model_config.multiscale_hidden_state_index,
            multiscale_fusion_init_logit=model_config.multiscale_fusion_init_logit,
            positional_encoding=model_config.positional_encoding,
            rope_theta=model_config.rope_theta,
        )
        self.model = TranscodaModelForCausalLM(self.config)
        self._scheduled_encoder_trainable_params = [
            param for param in self.model.frontend.encoder.parameters() if param.requires_grad
        ]
        self._encoder_freeze_schedule_active = False

        # Apply label smoothing if configured (combats exposure bias)
        if training.label_smoothing > 0:
            self.model.loss = torch.nn.CrossEntropyLoss(
                ignore_index=-100, label_smoothing=training.label_smoothing
            )

        # Store validation set names for dataloader_idx mapping
        self._val_set_names = val_set_names or ["validation"]
        self._base_val_set_names = base_val_set_names or [
            name for name in self._val_set_names if not is_subset_val_set_name(name)
        ]

        # Initialize TorchMetrics for validation and testing
        # Create per-validation-set metrics (one MetricCollection per set)
        def create_metrics(set_name: str):
            if is_subset_val_set_name(set_name):
                return MetricCollection(
                    {
                        "CER": CharacterErrorRate(
                            pad_id=pad_token_id,
                            bos_id=bos_token_id,
                            eos_id=eos_token_id,
                            i2w=i2w,
                        )
                    }
                )

            metric_map: dict[str, torch.nn.Module] = {
                "SER": SymbolErrorRate(
                    pad_id=pad_token_id, bos_id=bos_token_id, eos_id=eos_token_id
                ),
                "CER": CharacterErrorRate(
                    pad_id=pad_token_id, bos_id=bos_token_id, eos_id=eos_token_id, i2w=i2w
                ),
                "LER": LineErrorRate(
                    pad_id=pad_token_id, bos_id=bos_token_id, eos_id=eos_token_id, i2w=i2w
                ),
            }
            return MetricCollection(metric_map)

        # Per-validation-set metrics (e.g., val/synth/SER, val/real/CER)
        self.val_metrics_by_set = torch.nn.ModuleDict(
            {name: create_metrics(name) for name in self._val_set_names}
        )
        self.val_target_symbols_by_set = torch.nn.ModuleDict(
            {
                name: _TargetSymbolCount(
                    pad_id=pad_token_id, bos_id=bos_token_id, eos_id=eos_token_id
                )
                for name in self._val_set_names
            }
        )
        self._val_batches_seen_by_set = {name: 0 for name in self._val_set_names}

        # Test metrics (single set)
        self.test_metrics = create_metrics("test").clone(prefix="test_")

        # Runaway monitor trackers for validation/test scalar diagnostics (optional)
        self._runaway_monitor_enabled = bool(training.runaway_monitor_enabled)
        runaway_monitor_config = (
            resolve_runaway_monitor_config(training) if self._runaway_monitor_enabled else None
        )
        if self._runaway_monitor_enabled and runaway_monitor_config is not None:
            self._val_runaway_tracker_by_set = {
                name: RunawayMonitorTracker(
                    pad_id=pad_token_id,
                    bos_id=bos_token_id,
                    eos_id=eos_token_id,
                    i2w=i2w,
                    config=runaway_monitor_config,
                )
                for name in self._base_val_set_names
            }
            self._test_runaway_tracker = RunawayMonitorTracker(
                pad_id=pad_token_id,
                bos_id=bos_token_id,
                eos_id=eos_token_id,
                i2w=i2w,
                config=runaway_monitor_config,
            )
        else:
            self._val_runaway_tracker_by_set = None
            self._test_runaway_tracker = None

        repeat_loop_filtered_set_names = repeat_loop_filtered_set_names_for(
            self._val_set_names, runaway_monitor_enabled=self._runaway_monitor_enabled
        )
        self.val_cer_no_repeat_loops_by_set = torch.nn.ModuleDict(
            {
                name: CharacterErrorRate(
                    pad_id=pad_token_id,
                    bos_id=bos_token_id,
                    eos_id=eos_token_id,
                    i2w=i2w,
                )
                for name in repeat_loop_filtered_set_names
            }
        )
        self.val_repeat_loop_filtered_counts_by_set = torch.nn.ModuleDict(
            {name: _RepeatLoopFilteredSampleCounts() for name in repeat_loop_filtered_set_names}
        )

        # Grammar-constrained decoding provider for inference (optional)
        # Compiles grammar eagerly to fail fast on errors
        # Requires xgrammar to be installed (uv sync --group grammar)
        if training.use_grammar_constraints:
            if not training.grammar_path:
                raise ValueError(
                    "training.grammar_path is required when training.use_grammar_constraints=true"
                )
            self._grammar_provider = GrammarProvider(
                grammar_path=training.grammar_path,
                vocab_dir=vocab_dir,
                vocab_size=out_categories,
            )
        else:
            self._grammar_provider = None
        self._constraint_factory = ConstrainedDecodingFactory(
            grammar_provider=self._grammar_provider,
            i2w=self.config.i2w,
            bos_token_id=getattr(self.model.config, "bos_token_id", None),
            eos_token_id=getattr(self.model.config, "eos_token_id", None),
            pad_token_id=getattr(self.model.config, "pad_token_id", None),
            use_spine_structure_constraints=training.use_spine_structure_constraints,
            spine_structure_barline_width=training.spine_structure_barline_width,
            spine_structure_empty_field_mode=training.spine_structure_empty_field_mode,
            max_identical_line_run=training.decode_max_identical_line_run,
        )

        self._generation_settings = settings_from_generation_config(generation)
        self._generation_max_length = generation.max_length
        self._validation_decode_policy = DecodePolicy(
            name=_validation_decode_policy_name(training),
            generation_settings=self._generation_settings,
            configured_max_length=self._generation_max_length,
        )
        self._validation_decoder = ValidationDecoder(
            model=self.model,
            constraint_factory=self._constraint_factory,
            policy=self._validation_decode_policy,
        )
        self._generation_validation_length_multiplier = generation.validation_length_multiplier
        self._generation_validation_length_buffer = generation.validation_length_buffer
        self._generation_validation_absolute_max_length = (
            generation.validation_absolute_max_length
        )
        self._compiled_forward_model = None
        self._compile_initialized = False
        self._validation_metric_prefix = VAL_PREFIX
        self._validation_example_logging_override: bool | None = None
        self._validation_prediction_recording = False

        self.save_hyperparameters(ignore=["w2i", "i2w", "total_training_steps"])
        if show_model_summary:
            summary(
                self.model,
                input_size=[
                    (1, 3, self.config.maxh, self.config.maxw),
                    (1, self.config.maxlen),
                ],
                dtypes=[torch.float, torch.long],
            )
        # Callers may opt into stage logging for curricula or scheduled training
        # experiments. The default stage is intentionally not logged because it
        # is constant for ordinary runs.
        self.current_stage: int = 1
        self.stage_calculator: Callable[[int], int] = lambda x: self.current_stage
        self._log_train_stage = False

    def _forward_model(self):
        """Return compiled forward handle when enabled, otherwise eager model."""
        if self._compiled_forward_model is not None:
            return self._compiled_forward_model
        return self.model

    def disable_compiled_forward_model(self) -> bool:
        """Drop the compiled wrapper so checkpoint restore uses canonical model keys."""
        if self._compiled_forward_model is None:
            return False

        self._compiled_forward_model = None
        self._compile_initialized = False
        self.print(
            "Disabled torch.compile forward wrapper; subsequent checkpoint restores run eagerly."
        )
        return True

    def set_validation_metric_prefix(self, prefix: str) -> None:
        if prefix not in {VAL_PREFIX, FINAL_VAL_PREFIX}:
            raise ValueError(
                f"Unsupported validation metric prefix {prefix!r}; "
                f"expected one of {[VAL_PREFIX, FINAL_VAL_PREFIX]!r}."
            )
        self._validation_metric_prefix = prefix

    def set_validation_example_logging_override(self, enabled: bool | None) -> None:
        self._validation_example_logging_override = enabled

    def should_log_validation_examples(self) -> bool:
        if self._validation_example_logging_override is not None:
            return bool(self._validation_example_logging_override)
        return bool(self.hparams.training.log_example_images)

    def set_validation_prediction_recording(self, enabled: bool) -> None:
        self._validation_prediction_recording = bool(enabled)

    def is_recording_validation_predictions(self) -> bool:
        return bool(getattr(self, "_validation_prediction_recording", False))

    @property
    def validation_metric_prefix(self) -> str:
        return self._validation_metric_prefix

    def _should_emit_per_sample_validation_outputs(self) -> bool:
        """Per-sample outputs feed both W&B example logging and prediction recording."""
        if self.should_log_validation_examples():
            return True
        return TranscodaTrainer.is_recording_validation_predictions(self)

    def _validation_set_metric_name(self, set_name: str, metric_name: str) -> str:
        if self._validation_metric_prefix == FINAL_VAL_PREFIX:
            return final_val_set_metric(set_name, metric_name)
        return val_set_metric(set_name, metric_name)

    def _validation_aggregate_metric_name(self, metric_name: str) -> str:
        if self._validation_metric_prefix == FINAL_VAL_PREFIX:
            return final_val_aggregate_metric(metric_name)
        return val_aggregate_metric(metric_name)

    def _mark_compiled_step_begin(self) -> None:
        """Mark CUDAGraph step boundary before each compiled model invocation."""
        if self._compiled_forward_model is None:
            return

        compiler_mod = getattr(torch, "compiler", None)
        mark_step_begin = (
            getattr(compiler_mod, "cudagraph_mark_step_begin", None) if compiler_mod else None
        )
        if callable(mark_step_begin):
            mark_step_begin()

    def _encoder_freeze_steps(self) -> int:
        training = getattr(getattr(self, "hparams", None), "training", None)
        return int(getattr(training, "freeze_encoder_steps", 0) or 0)

    def _set_scheduled_encoder_requires_grad(self, requires_grad: bool) -> None:
        params = getattr(self, "_scheduled_encoder_trainable_params", [])
        for param in params:
            param.requires_grad = requires_grad

    def _update_encoder_freeze_schedule(self) -> bool:
        """Apply the step-based encoder freeze schedule and return frozen state."""
        freeze_steps = TranscodaTrainer._encoder_freeze_steps(self)
        if freeze_steps <= 0:
            if getattr(self, "_encoder_freeze_schedule_active", False):
                TranscodaTrainer._set_scheduled_encoder_requires_grad(self, True)
                self._encoder_freeze_schedule_active = False
            return False

        should_freeze = int(getattr(self, "global_step", 0) or 0) < freeze_steps
        is_active = bool(getattr(self, "_encoder_freeze_schedule_active", False))

        if should_freeze and not is_active:
            TranscodaTrainer._set_scheduled_encoder_requires_grad(self, False)
            self._encoder_freeze_schedule_active = True
            print_fn = getattr(self, "print", None)
            if callable(print_fn):
                print_fn(
                    "Temporarily froze encoder parameters for "
                    f"training.freeze_encoder_steps={freeze_steps}."
                )
        elif not should_freeze and is_active:
            TranscodaTrainer._set_scheduled_encoder_requires_grad(self, True)
            self._encoder_freeze_schedule_active = False
            print_fn = getattr(self, "print", None)
            if callable(print_fn):
                print_fn("Unfroze encoder parameters after freeze_encoder_steps schedule.")

        return should_freeze

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Clean up checkpoints saved with torch.compile enabled.

        When torch.compile wraps self.model, the compiled wrapper is registered as a
        submodule (_compiled_forward_model) whose _orig_mod points back to self.model.
        This causes state_dict to contain duplicate entries:
          - model.frontend.encoder...  (canonical, from self.model)
          - _compiled_forward_model._orig_mod.frontend.encoder...  (duplicate)
        We drop the duplicates so Lightning can load the canonical keys normally.

        The compiled model's optimizer may also have a different number of parameter
        groups, so we drop optimizer/scheduler state to avoid restoration errors.
        Training resumes with a fresh optimizer while keeping the loaded model weights.
        """
        _COMPILE_PREFIX = "_compiled_forward_model._orig_mod."
        state_dict = checkpoint.get("state_dict", {})
        if any(_COMPILE_PREFIX in k for k in state_dict):
            checkpoint["state_dict"] = {
                k: v for k, v in state_dict.items() if _COMPILE_PREFIX not in k
            }
            checkpoint["optimizer_states"] = []
            checkpoint["lr_schedulers"] = []
            self.print(
                "Stripped torch.compile artifacts from checkpoint. "
                "Optimizer and scheduler state reset (model weights preserved)."
            )

    def on_fit_start(self) -> None:
        """Initialize torch.compile once at fit start if requested."""
        TranscodaTrainer._update_encoder_freeze_schedule(self)

        if self._compile_initialized:
            return
        self._compile_initialized = True

        if not bool(self.hparams.training.compile_model):
            return

        compile_mode = self.hparams.training.compile_mode
        if compile_mode not in _ALLOWED_COMPILE_MODES:
            raise ValueError(
                f"Invalid training.compile_mode='{compile_mode}'. "
                f"Expected one of: {sorted(_ALLOWED_COMPILE_MODES)}."
            )

        # Compile + CUDAGraph paths can conflict with checkpointed encoder recomputation.
        # Disable encoder checkpointing proactively for compile stability.
        encoder = getattr(getattr(self.model, "frontend", None), "encoder", None)
        disable_gc = getattr(encoder, "gradient_checkpointing_disable", None)
        if callable(disable_gc):
            disable_gc()
            self.print("Disabled encoder gradient checkpointing for torch.compile stability.")

        if getattr(self.hparams.training, "decoder_length_buckets", None):
            # One static graph per decoder-length bucket (see
            # on_after_batch_transfer); make sure the variant cache fits the
            # bucket count plus validation batch-size variants.
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 16)

        self.print(f"Compiling forward path with torch.compile(mode='{compile_mode}')...")
        try:
            self._compiled_forward_model = torch.compile(self.model, mode=compile_mode)
        except Exception as exc:
            raise RuntimeError(
                "torch.compile initialization failed with "
                f"training.compile_mode='{compile_mode}'. "
                "Set training.compile_model=false to continue in eager mode."
            ) from exc

        self.print("torch.compile enabled for training/validation loss forward path.")

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        decoder_lr = self.hparams.optimizer_config.learning_rate
        encoder_lr = decoder_lr * self.hparams.optimizer_config.encoder_lr_factor
        weight_decay = self.hparams.optimizer_config.weight_decay

        cfg = self.hparams.optimizer_config

        # Determine if LLRD should be enabled
        use_llrd = (
            cfg.layerwise_target == "encoder"
            and cfg.layerwise_decay_gamma is not None
            and cfg.layerwise_decay_gamma > 0.0
        )

        param_groups = []

        # Build encoder parameter groups (flat or layer-wise)
        if use_llrd:
            # Use layer-wise learning rate decay for encoder
            encoder_groups = build_llrd_param_groups_for_convnextv2(
                encoder=self.model.frontend.encoder,
                base_encoder_lr=encoder_lr,
                weight_decay=weight_decay,
                gamma=float(cfg.layerwise_decay_gamma),
                name_prefix="encoder",
            )
            param_groups.extend(encoder_groups)
        else:
            param_groups.extend(
                split_named_params_for_weight_decay(
                    self.model.frontend.encoder.named_parameters(),
                    lr=encoder_lr,
                    weight_decay=weight_decay,
                    name_prefix="encoder",
                )
            )

        # Add bridge and decoder groups with global no-decay policy.
        param_groups.extend(
            split_named_params_for_weight_decay(
                self.model.frontend.bridge_named_parameters(),
                lr=decoder_lr,
                weight_decay=weight_decay,
                name_prefix="bridge",
            )
        )
        param_groups.extend(
            split_named_params_for_weight_decay(
                self.model.decoder.named_parameters(),
                lr=decoder_lr,
                weight_decay=weight_decay,
                name_prefix="decoder",
            )
        )

        adamw_kwargs = {
            "weight_decay": weight_decay,
        }
        if torch.cuda.is_available():
            # Use fused=True for faster optimizer steps on CUDA.
            # Fused optimizer combines multiple operations into single CUDA kernels,
            # significantly reducing kernel launch overhead and memory bandwidth.
            # Note: Requires manual gradient clipping (see configure_gradient_clipping).
            optimizer = torch.optim.AdamW(param_groups, fused=True, **adamw_kwargs)
        else:
            optimizer = torch.optim.AdamW(param_groups, **adamw_kwargs)

        # Apply the temporary encoder freeze only after optimizer groups have been
        # built, since the grouping utilities intentionally skip frozen params.
        TranscodaTrainer._update_encoder_freeze_schedule(self)

        # --- Log Parameter Group Summary (rank 0 only) ---
        if getattr(self, "global_rank", 0) == 0:
            try:
                self.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                self.print("  Optimizer Parameter Groups")
                self.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                for i, group in enumerate(optimizer.param_groups):
                    group_name = group.get("name", f"group{i}")
                    group_lr = group["lr"]
                    group_wd = group.get("weight_decay", 0.0)
                    num_params = sum(p.numel() for p in group["params"])
                    self.print(
                        f"  [{i:2d}] {group_name:32s} lr={group_lr:.3e}  "
                        f"wd={group_wd:.3f}  params={num_params / 1e6:.2f}M"
                    )
                self.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            except Exception:
                # Silently ignore logging errors
                pass

        # --- Learning Rate Scheduler Setup ---
        # Cosine schedulers apply a shared multiplier to each group's base_lr,
        # so every param group decays toward its own per-group floor at
        # `cosine_eta_min_factor * group.initial_lr`. This is required for LLRD:
        # otherwise shallow encoder groups (whose initial LR can be smaller than
        # a global eta_min) would see their LR rise over training.
        warmup_steps = self.hparams.optimizer_config.warmup_steps
        eta_min_factor = float(self.hparams.optimizer_config.cosine_eta_min_factor)

        total_steps = _finite_estimated_stepping_batches(self.trainer)

        lr_scheduler_type = cfg.lr_scheduler
        if lr_scheduler_type == "cosine_warm_restarts":
            main_scheduler = make_cosine_warm_restarts_lambda_lr(
                optimizer,
                T_0=cfg.cosine_restart_period,
                T_mult=cfg.cosine_restart_mult,
                eta_min_factor=eta_min_factor,
            )
        else:
            t_max_value = int(max(1, total_steps - warmup_steps))
            main_scheduler = make_cosine_annealing_lambda_lr(
                optimizer,
                T_max=t_max_value,
                eta_min_factor=eta_min_factor,
            )

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | int | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Manual gradient clipping for compatibility with fused optimizers.

        Fused optimizers (fused=True) combine multiple operations into single CUDA kernels.
        This override ensures gradient clipping works correctly with fused AdamW.
        """
        if gradient_clip_val is None:
            return

        gradient_clip_val = float(gradient_clip_val)
        gradient_clip_algorithm = gradient_clip_algorithm or "norm"
        if gradient_clip_algorithm == "norm":
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_val)
        elif gradient_clip_algorithm == "value":
            torch.nn.utils.clip_grad_value_(self.parameters(), gradient_clip_val)
        else:
            raise ValueError(
                "Unsupported gradient_clip_algorithm: "
                f"{gradient_clip_algorithm!r}. Expected 'norm' or 'value'."
            )

    def set_stage(self, stage: int) -> None:
        self.current_stage = int(stage)
        self._log_train_stage = True

    def set_stage_calculator(self, stage_calculator: Callable[[int], int]) -> None:
        self.stage_calculator = stage_calculator
        self._log_train_stage = True

    def _generate_with_constraints(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor | None,
        max_length: int,
    ) -> torch.Tensor:
        """Backward-compatible wrapper around the validation decoder."""
        decoder = getattr(self, "_validation_decoder", None)
        if decoder is None:
            policy = DecodePolicy(
                name="validation+compat",
                generation_settings=self._generation_settings,
                configured_max_length=getattr(self, "_generation_max_length", None),
            )
            decoder = ValidationDecoder(
                model=self.model,
                constraint_factory=self._constraint_factory,
                policy=policy,
            )
        return decoder.generate(
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            max_length=max_length,
        )

    def _generate_with_grammar(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor | None,
        max_length: int,
    ) -> torch.Tensor:
        """Backward-compatible alias for the shared constrained generation path."""
        generate_with_constraints = getattr(self, "_generate_with_constraints", None)
        if generate_with_constraints is None:
            generate_with_constraints = TranscodaTrainer._generate_with_constraints.__get__(
                self, type(self)
            )
        return generate_with_constraints(
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            max_length=max_length,
        )

    def _target_relative_generation_max_length(self, labels: torch.Tensor) -> int:
        target_lengths = (labels != -100).sum(dim=1)
        return _target_relative_generation_cap(
            target_lengths,
            multiplier=float(getattr(self, "_generation_validation_length_multiplier", 1.5)),
            buffer=int(getattr(self, "_generation_validation_length_buffer", 32)),
            maxlen=int(self.config.maxlen),
            configured_max_length=getattr(self, "_generation_max_length", None),
        )

    def _validation_generation_max_length(self, labels: torch.Tensor) -> int:
        """Decode cap for a validation/test batch; absolute when configured."""
        absolute = getattr(self, "_generation_validation_absolute_max_length", None)
        if absolute is None:
            return self._target_relative_generation_max_length(labels)
        return validation_generation_cap(
            labels,
            absolute_max_length=int(absolute),
            multiplier=float(getattr(self, "_generation_validation_length_multiplier", 1.5)),
            buffer=int(getattr(self, "_generation_validation_length_buffer", 32)),
            maxlen=int(self.config.maxlen),
            configured_max_length=getattr(self, "_generation_max_length", None),
        )

    def forward(self, inputs, last_preds):
        return self.model(inputs, last_preds)

    @staticmethod
    def _training_config_bool(module, name: str, default: bool) -> bool:
        hparams = getattr(module, "hparams", None)
        training = getattr(hparams, "training", None)
        return bool(getattr(training, name, default))

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        """Normalize uint8 pixels on-device.

        The datamodule ships uint8 CHW through the DataLoader when
        training.normalize_on_gpu is set (4x less worker IPC/pinning than
        normalized float32); dtype-driven so float32 batches pass through
        unchanged and older pipelines keep working.
        """
        pixel_values = batch.get("pixel_values") if isinstance(batch, dict) else None
        if pixel_values is not None and pixel_values.dtype == torch.uint8:
            normalized = normalize_uint8_pixel_values(pixel_values)
            # The precision plugin's input cast (e.g. bf16-true) only converts
            # floating tensors and has already run, so match the module dtype
            # here.
            batch["pixel_values"] = normalized.to(self.dtype)
        training_cfg = getattr(getattr(self, "hparams", None), "training", None)
        if isinstance(batch, dict) and getattr(training_cfg, "decoder_length_buckets", None):
            # Bucketed padding varies the decoder length between batches. Keep
            # that dim static so dynamo compiles one graph per bucket instead
            # of a dynamic-shape graph (whose flash-attention backward hits
            # stride-assertion failures in torch 2.9).
            for key in ("labels", "decoder_attention_mask"):
                value = batch.get(key)
                if isinstance(value, torch.Tensor) and value.dim() >= 2:
                    torch._dynamo.mark_static(value, 1)
        return batch

    def on_train_batch_start(self, batch, batch_idx):
        """Apply the encoder freeze schedule before each training batch."""
        encoder_frozen = TranscodaTrainer._update_encoder_freeze_schedule(self)
        if TranscodaTrainer._encoder_freeze_steps(self) > 0:
            self.log(
                "train/encoder_frozen",
                float(encoder_frozen),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

    def training_step(self, batch, _batch_idx):
        x = batch["pixel_values"]
        labels = batch["labels"]
        dec_mask = batch.get("decoder_attention_mask", None)
        image_sizes = batch.get("image_sizes", None)
        build_decoder_attention_mask_from_labels = True

        if TranscodaTrainer._training_config_bool(
            self,
            "omit_train_decoder_attention_mask",
            True,
        ):
            dec_mask = None
            build_decoder_attention_mask_from_labels = False

        if TranscodaTrainer._training_config_bool(
            self,
            "omit_train_encoder_attention_mask",
            True,
        ):
            image_sizes = None

        self._mark_compiled_step_begin()
        outputs = self._forward_model()(
            pixel_values=x,
            labels=labels,
            image_sizes=image_sizes,
            attention_mask=dec_mask,  # True = valid
            build_decoder_attention_mask_from_labels=build_decoder_attention_mask_from_labels,
        )
        loss = outputs.loss

        self.log(
            TRAIN_LOSS,
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=x.size(0),
            prog_bar=False,
            sync_dist=True,
        )
        if getattr(self, "_log_train_stage", False):
            stage = self.stage_calculator(self.global_step)
            self.log(TRAIN_STAGE, stage, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, val_batch, batch_idx, dataloader_idx: int = 0):
        x = val_batch["pixel_values"]
        labels = val_batch["labels"]
        dec_mask = val_batch.get("decoder_attention_mask", None)
        image_sizes = val_batch.get("image_sizes", None)
        sample_ids = val_batch.get("sample_ids", None)
        dataset_sample_ids = val_batch.get("dataset_sample_ids", None)
        sample_revision_digests = val_batch.get("sample_revision_digests", None)
        sources = val_batch.get("source", None)

        # Get validation set name from dataloader index
        val_set_name = self._val_set_names[dataloader_idx]
        is_subset_loader = is_subset_val_set_name(val_set_name)
        seen_by_set = getattr(self, "_val_batches_seen_by_set", None)
        if seen_by_set is None:
            seen_by_set = {name: 0 for name in self._val_set_names}
            self._val_batches_seen_by_set = seen_by_set
        seen_by_set[val_set_name] = seen_by_set.get(val_set_name, 0) + 1

        # Calculate validation loss
        self._mark_compiled_step_begin()
        outputs = self._forward_model()(
            pixel_values=x,
            labels=labels,
            image_sizes=image_sizes,
            attention_mask=dec_mask,
        )
        loss = outputs.loss

        # Log validation loss per set (Lightning aggregates automatically across batches)
        if not is_subset_loader:
            self.log(
                self._validation_set_metric_name(val_set_name, "loss"),
                loss,
                on_step=False,
                on_epoch=True,
                batch_size=x.size(0),
                prog_bar=False,
                add_dataloader_idx=False,
                sync_dist=True,
            )

        # Allow semantically valid but more verbose transcriptions while retaining a
        # bounded decode for bad checkpoints and runaway validation cases.
        generation_max_length = self._validation_generation_max_length(labels)

        # Get predicted token IDs with dynamic max_length
        # Uses grammar-constrained decoding if enabled
        preds = self._generate_with_grammar(
            pixel_values=x,
            image_sizes=image_sizes,
            max_length=generation_max_length,
        )

        labels_for_metric = labels.masked_fill(labels == -100, self.config.pad_token_id)

        # Update metrics for this validation set
        self.val_metrics_by_set[val_set_name].update(preds, labels_for_metric)
        target_symbols_by_set = getattr(self, "val_target_symbols_by_set", None)
        if target_symbols_by_set is not None and val_set_name in target_symbols_by_set:
            target_symbols_by_set[val_set_name].update(labels_for_metric)

        # Update runaway monitor diagnostics for this validation set (if enabled).
        runaway_diagnostics = None
        runaway_trackers = getattr(self, "_val_runaway_tracker_by_set", None)
        if runaway_trackers is not None and not is_subset_loader:
            runaway_tracker = runaway_trackers.get(val_set_name)
            if runaway_tracker is not None:
                runaway_diagnostics = runaway_tracker.update_batch(
                    preds=preds,
                    targets=labels_for_metric,
                    max_length_cap=generation_max_length,
                )
                filtered_cer_by_set = getattr(self, "val_cer_no_repeat_loops_by_set", None)
                filtered_counts_by_set = getattr(
                    self, "val_repeat_loop_filtered_counts_by_set", None
                )
                if (
                    filtered_cer_by_set is not None
                    and filtered_counts_by_set is not None
                    and val_set_name in filtered_cer_by_set
                    and val_set_name in filtered_counts_by_set
                ):
                    included_indices = [
                        idx for idx, diag in enumerate(runaway_diagnostics) if not diag.repeat_loop
                    ]
                    excluded_count = len(runaway_diagnostics) - len(included_indices)
                    filtered_counts_by_set[val_set_name].update(
                        included=len(included_indices), excluded=excluded_count
                    )
                    if included_indices:
                        filtered_cer_by_set[val_set_name].update(
                            preds[included_indices],
                            labels_for_metric[included_indices],
                        )

        # Return per-sample outputs for downstream callbacks (WandbVisualizerCallback,
        # ValidationPredictionWriter) when any of them needs them.
        if (
            TranscodaTrainer._should_emit_per_sample_validation_outputs(self)
            and not is_subset_loader
        ):
            return TranscodaTrainer._build_per_sample_validation_outputs(
                self,
                val_set_name=val_set_name,
                preds=preds,
                labels_for_metric=labels_for_metric,
                sample_ids=sample_ids,
                dataset_sample_ids=dataset_sample_ids,
                sample_revision_digests=sample_revision_digests,
                sources=sources,
                batch_size=x.size(0),
                generation_max_length=generation_max_length,
            )
        return None

    def _build_per_sample_validation_outputs(
        self,
        *,
        val_set_name: str,
        preds: torch.Tensor,
        labels_for_metric: torch.Tensor,
        sample_ids: torch.Tensor | None,
        dataset_sample_ids: list[str] | None,
        sample_revision_digests: list[str] | None,
        sources: list | None,
        batch_size: int,
        generation_max_length: int,
    ) -> dict:
        """Build per-sample prediction records with exact CER values."""
        if sample_ids is None:
            raise ValueError("Validation batch is missing sample_ids required for exact ranking.")
        if sources is None:
            sources = [None] * batch_size

        cers = [
            CharacterErrorRate.compute_single(
                pred=preds[i].tolist(),
                target=labels_for_metric[i].tolist(),
                pad_id=self.config.pad_token_id,
                i2w=self.config.i2w,
            )
            for i in range(batch_size)
        ]
        return {
            "val_set_name": val_set_name,
            "sample_ids": sample_ids.detach().cpu(),
            "dataset_sample_ids": (
                list(dataset_sample_ids) if dataset_sample_ids is not None else None
            ),
            "sample_revision_digests": (
                list(sample_revision_digests) if sample_revision_digests is not None else None
            ),
            "sources": list(sources),
            "cers": cers,
            "pred_ids": [preds[i].detach().cpu().tolist() for i in range(batch_size)],
            "gt_ids": [labels_for_metric[i].detach().cpu().tolist() for i in range(batch_size)],
            "generation_max_length": int(generation_max_length),
            "decode_policy_name": getattr(
                getattr(self, "_validation_decoder", None),
                "policy_name",
                None,
            ),
        }

    def on_validation_epoch_start(self) -> None:
        self._val_batches_seen_by_set = {name: 0 for name in self._val_set_names}
        target_symbols_by_set = getattr(self, "val_target_symbols_by_set", None)
        if target_symbols_by_set is not None:
            for metric in target_symbols_by_set.values():
                metric.reset()
        runaway_trackers = getattr(self, "_val_runaway_tracker_by_set", None)
        if runaway_trackers is not None:
            for tracker in runaway_trackers.values():
                tracker.reset()
        filtered_cer_by_set = getattr(self, "val_cer_no_repeat_loops_by_set", None)
        if filtered_cer_by_set is not None:
            for metric in filtered_cer_by_set.values():
                metric.reset()
        filtered_counts_by_set = getattr(self, "val_repeat_loop_filtered_counts_by_set", None)
        if filtered_counts_by_set is not None:
            for metric in filtered_counts_by_set.values():
                metric.reset()

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics at the end of each epoch.

        Visualization and example logging is now handled by WandbVisualizerCallback.
        """
        # Compute and log metrics for each validation set. A set without target
        # symbols (no-music pages, whose target is empty) has no error rate and
        # no runaway rate: every prediction would count as a length blow-up.
        # It logs only its loss and stays out of the val/SER and runaway
        # aggregates.
        all_ser_values = []
        runaway_set_metrics: list[dict[str, float | int]] = []
        for set_name in self._val_set_names:
            metrics = self.val_metrics_by_set[set_name]
            seen_batches = self._val_batches_seen_by_set.get(set_name, 0)
            if seen_batches > 0 and _scored_no_target_symbols(self, set_name):
                seen_batches = 0
            if seen_batches > 0:
                computed = metrics.compute()
                if is_subset_val_set_name(set_name):
                    cer_value = computed.get("CER")
                    if cer_value is not None:
                        self.log(
                            val_subset_metric(base_val_set_name(set_name), "CER"),
                            cer_value,
                            on_epoch=True,
                            prog_bar=False,
                            sync_dist=True,
                        )
                else:
                    # Log metrics with per-set path (e.g., val/synth/SER, val/real/CER)
                    for metric_name, value in computed.items():
                        self.log(
                            self._validation_set_metric_name(set_name, metric_name),
                            value,
                            on_epoch=True,
                            prog_bar=False,
                            sync_dist=True,
                        )
                        if metric_name == "SER":
                            all_ser_values.append(value)

            metrics.reset()
            target_symbols_by_set = getattr(self, "val_target_symbols_by_set", None)
            if target_symbols_by_set is not None and set_name in target_symbols_by_set:
                target_symbols_by_set[set_name].reset()

            filtered_cer_by_set = getattr(self, "val_cer_no_repeat_loops_by_set", None)
            filtered_counts_by_set = getattr(self, "val_repeat_loop_filtered_counts_by_set", None)
            if (
                filtered_cer_by_set is not None
                and filtered_counts_by_set is not None
                and set_name in filtered_cer_by_set
                and set_name in filtered_counts_by_set
            ):
                if seen_batches > 0:
                    count_values = filtered_counts_by_set[set_name].compute()
                    included_samples = count_values["CER_no_repeat_loops_samples"]
                    if float(included_samples.detach().cpu().item()) > 0:
                        filtered_cer = filtered_cer_by_set[set_name].compute()
                    else:
                        filtered_cer = torch.zeros_like(included_samples)
                    self.log(
                        self._validation_set_metric_name(set_name, "CER_no_repeat_loops"),
                        filtered_cer,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )
                filtered_cer_by_set[set_name].reset()
                filtered_counts_by_set[set_name].reset()

            runaway_trackers = getattr(self, "_val_runaway_tracker_by_set", None)
            if runaway_trackers is not None:
                tracker = runaway_trackers.get(set_name)
                if tracker is not None:
                    if seen_batches > 0:
                        computed_runaway = tracker.compute()
                        runaway_set_metrics.append(computed_runaway)
                        runaway_rate = computed_runaway.get("runaway_rate")
                        if runaway_rate is not None:
                            self.log(
                                self._validation_set_metric_name(set_name, "runaway_rate"),
                                float(runaway_rate),
                                on_epoch=True,
                                prog_bar=False,
                                sync_dist=True,
                            )
                    tracker.reset()

        # Log aggregate validation metrics (weighted by active sets).
        if all_ser_values:
            overall_ser = sum(all_ser_values) / len(all_ser_values)
            self.log(
                self._validation_aggregate_metric_name("SER"),
                overall_ser,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        if runaway_set_metrics:
            total_runaway_samples = sum(
                int(metrics.get("runaway_samples", 0)) for metrics in runaway_set_metrics
            )
            if total_runaway_samples > 0:
                total_runaway_rate = (
                    sum(
                        float(metrics.get("runaway_rate", 0.0))
                        * int(metrics.get("runaway_samples", 0))
                        for metrics in runaway_set_metrics
                    )
                    / total_runaway_samples
                )
                self.log(
                    self._validation_aggregate_metric_name("runaway_rate"),
                    total_runaway_rate,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )

    def test_step(self, test_batch, _batch_idx):
        x = test_batch["pixel_values"]
        labels = test_batch["labels"]
        image_sizes = test_batch.get("image_sizes", None)

        # Allow semantically valid but more verbose transcriptions while retaining a
        # bounded decode for bad checkpoints and runaway test cases.
        generation_max_length = self._validation_generation_max_length(labels)

        # Get predicted token IDs with dynamic max_length
        # Uses grammar-constrained decoding if enabled
        preds = self._generate_with_grammar(
            pixel_values=x,
            image_sizes=image_sizes,
            max_length=generation_max_length,
        )

        # Update metrics (MetricCollection updates all metrics in one call)
        self.test_metrics.update(preds, labels)

        # Update test runaway diagnostics (if enabled).
        test_runaway_tracker = getattr(self, "_test_runaway_tracker", None)
        if test_runaway_tracker is not None:
            labels_for_monitor = labels.masked_fill(labels == -100, self.config.pad_token_id)
            test_runaway_tracker.update_batch(
                preds=preds,
                targets=labels_for_monitor,
                max_length_cap=generation_max_length,
            )

    def on_test_epoch_end(self) -> None:
        # Compute and log all test metrics (MetricCollection handles compute + returns dict)
        for metric_name, value in self.test_metrics.compute().items():
            normalized_name = metric_name.removeprefix("test_")
            self.log(
                build_test_metric_key(normalized_name),
                value,
                on_epoch=True,
                prog_bar=(normalized_name == "SER"),
                sync_dist=True,
            )
        test_runaway_tracker = getattr(self, "_test_runaway_tracker", None)
        if test_runaway_tracker is not None:
            computed = test_runaway_tracker.compute()
            for metric_name, value in computed.items():
                self.log(
                    build_test_metric_key(metric_name),
                    float(value),
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )
            test_runaway_tracker.reset()
