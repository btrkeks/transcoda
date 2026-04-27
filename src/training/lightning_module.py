import math
import time
from collections.abc import Callable

import lightning.pytorch as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch.optim.lr_scheduler import (
    LinearLR,
    SequentialLR,
)
from torchinfo import summary
from torchmetrics import MetricCollection

from src.config import Generation, ModelConfig, OptimizerConfig, Training
from src.core.kern_utils import strip_tie_beam_markers_from_kern_text
from src.core.metrics import (
    CharacterErrorRate,
    LineErrorRate,
    RunawayMonitorTracker,
    SymbolErrorRate,
    resolve_runaway_monitor_config,
)
from src.grammar import GrammarProvider
from src.grammar.constraint_factory import ConstrainedDecodingFactory
from src.grammar.interpretation_transition_rule import (
    resolve_interpretation_transition_config,
)
from src.grammar.runaway_guard import resolve_runaway_guard_config
from src.metrics_schema import (
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
from src.model import SMTConfig, SMTModelForCausalLM, VisionFrontendOutput
from src.model.generation_policy import (
    build_generate_kwargs,
    settings_from_generation_config,
)
from src.training.optim.cosine import (
    make_cosine_annealing_lambda_lr,
    make_cosine_warm_restarts_lambda_lr,
)
from src.training.optim.layerwise import (
    build_llrd_param_groups_for_convnextv2,
    split_named_params_for_weight_decay,
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


def _is_polish_style_validation_set(set_name: str) -> bool:
    return set_name == "polish" or set_name.startswith("polish_")


class SMTTrainer(L.LightningModule):
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
        total_training_steps: int | None = None,
        show_model_summary: bool = False,
        run_artifact_json: str | None = None,
    ):
        super().__init__()

        self.config = SMTConfig(
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
            positional_encoding=model_config.positional_encoding,
            rope_theta=model_config.rope_theta,
        )
        self.model = SMTModelForCausalLM(self.config)

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

            base_set_name = base_val_set_name(set_name)
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
            if _is_polish_style_validation_set(base_set_name):
                metric_map["CER_no_ties_beams"] = CharacterErrorRate(
                    pad_id=pad_token_id,
                    bos_id=bos_token_id,
                    eos_id=eos_token_id,
                    i2w=i2w,
                    text_normalizer=strip_tie_beam_markers_from_kern_text,
                )
            return MetricCollection(metric_map)

        # Per-validation-set metrics (e.g., val/synth/SER, val/polish/CER)
        self.val_metrics_by_set = torch.nn.ModuleDict(
            {name: create_metrics(name) for name in self._val_set_names}
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

        # OMR-NED tracker for semantic music notation comparison (optional)
        self._compute_omr_ned = training.compute_omr_ned
        self._omr_ned_tracker = None
        if self._compute_omr_ned:
            from src.core.metrics import OMRNEDTracker

            tracker = OMRNEDTracker()
            if tracker.enabled:
                self._omr_ned_tracker = tracker
                # OMR-NED metric is enabled (no output to avoid TQDM interference)
            else:
                self._compute_omr_ned = False
                # Print warning using plain print to avoid TQDM interference
                print(
                    "\n"
                    "╔══════════════════════════════════════════════════════════════════╗\n"
                    "║  OMR-NED METRIC DISABLED - musicdiff not installed!              ║\n"
                    "║                                                                  ║\n"
                    "║  You set compute_omr_ned=true in config, but the required        ║\n"
                    "║  dependencies are missing. The metric will NOT be computed.      ║\n"
                    "║                                                                  ║\n"
                    "║  To fix, run:  uv sync --group omr-ned                           ║\n"
                    "╚══════════════════════════════════════════════════════════════════╝"
                )

        # Grammar-constrained decoding provider for inference (optional)
        # Compiles grammar eagerly to fail fast on errors
        # Requires xgrammar to be installed (uv sync --group grammar)
        if training.use_grammar_constraints:
            self._grammar_provider = GrammarProvider(
                grammar_path=training.grammar_path,
                vocab_dir=vocab_dir,
                vocab_size=out_categories,
            )
        else:
            self._grammar_provider = None
        self._runaway_guard_enabled = bool(training.runaway_guard_enabled)
        self._runaway_guard_config = (
            resolve_runaway_guard_config(training) if self._runaway_guard_enabled else None
        )
        self._constraint_factory = ConstrainedDecodingFactory(
            grammar_provider=self._grammar_provider,
            i2w=self.config.i2w,
            bos_token_id=getattr(self.model.config, "bos_token_id", None),
            eos_token_id=getattr(self.model.config, "eos_token_id", None),
            pad_token_id=getattr(self.model.config, "pad_token_id", None),
            use_interpretation_transition_constraints=(
                training.use_interpretation_transition_constraints
            ),
            use_spine_structure_constraints=training.use_spine_structure_constraints,
            # Preserve RhythmRule in code, but keep it out of validation/test inference.
            use_rhythm_constraints=False,
            interpretation_transition_config=resolve_interpretation_transition_config(training),
            runaway_guard_enabled=self._runaway_guard_enabled,
            runaway_guard_config=self._runaway_guard_config,
        )

        self._generation_settings = settings_from_generation_config(generation)
        self._generation_max_length = generation.max_length
        self._compiled_forward_model = None
        self._compile_initialized = False
        self._validation_metric_prefix = VAL_PREFIX
        self._validation_example_logging_override: bool | None = None

        self.save_hyperparameters(ignore=["w2i", "i2w"])
        if show_model_summary:
            summary(
                self.model,
                input_size=[
                    (1, 3, self.config.maxh, self.config.maxw),
                    (1, self.config.maxlen),
                ],
                dtypes=[torch.float, torch.long],
            )
        self.current_stage: int = 1
        self.stage_calculator: Callable[[int], int] = lambda x: self.current_stage

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
                self.model.frontend.projector.named_parameters(),
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

    def set_stage(self, stage):
        self.stage = stage

    def set_stage_calculator(self, stage_calculator: Callable[[int], int]):
        self.stage_calculator = stage_calculator

    def _generate_with_constraints(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor | None,
        max_length: int,
    ) -> torch.Tensor:
        """Generate sequences with optional grammar constraints.

        If a grammar provider is configured, adds a LogitsProcessor to enforce
        valid **kern syntax. Decoding strategy is controlled by centralized
        generation config (beam or greedy).

        Args:
            pixel_values: Input images (batch, 3, height, width).
            image_sizes: Original image sizes for encoder padding mask.
            max_length: Maximum sequence length to generate.

        Returns:
            Generated token IDs (batch, seq_len).
        """
        effective_max_length = max_length
        if self._generation_max_length is not None:
            effective_max_length = min(max_length, int(self._generation_max_length))

        bundle = self._constraint_factory.build(self._generation_settings)
        logits_processor = bundle.logits_processors
        stopping_criteria = bundle.stopping_criteria
        generation_settings = bundle.generation_settings

        # Precompute encoder state and pass it explicitly to generate().
        # This avoids relying on internal model_kwargs propagation across
        # beam-search steps in newer transformers versions.
        encoder_outputs = self.model.forward_encoder(pixel_values, image_sizes=image_sizes)
        num_beams = max(1, int(getattr(generation_settings, "num_beams", 1)))
        num_return_sequences = max(
            1, int(getattr(generation_settings, "num_return_sequences", 1))
        )
        do_sample = bool(getattr(generation_settings, "do_sample", False))
        if num_beams > 1 and do_sample:
            expand_size = num_beams * num_return_sequences
        elif num_beams > 1:
            expand_size = num_beams
        else:
            expand_size = num_return_sequences
        if expand_size > 1:
            encoder_outputs = VisionFrontendOutput(
                encoder_tokens_raw=encoder_outputs.encoder_tokens_raw.repeat_interleave(
                    expand_size, dim=0
                ),
                encoder_tokens_pos=encoder_outputs.encoder_tokens_pos.repeat_interleave(
                    expand_size, dim=0
                ),
                encoder_attention_mask=(
                    None
                    if encoder_outputs.encoder_attention_mask is None
                    else encoder_outputs.encoder_attention_mask.repeat_interleave(expand_size, dim=0)
                ),
            )
        generate_kwargs = build_generate_kwargs(
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            max_length=effective_max_length,
            settings=generation_settings,
            logits_processor=logits_processor,
        )
        if stopping_criteria is not None:
            generate_kwargs["stopping_criteria"] = stopping_criteria
        generate_kwargs.pop("pixel_values", None)
        generate_kwargs.pop("image_sizes", None)
        generate_kwargs["input_ids"] = torch.full(
            (pixel_values.shape[0], 1),
            self.model.config.bos_token_id,
            dtype=torch.long,
            device=pixel_values.device,
        )
        generate_kwargs["encoder_outputs"] = encoder_outputs
        preds = self.model.generate(**generate_kwargs)
        return preds

    def _generate_with_grammar(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor | None,
        max_length: int,
    ) -> torch.Tensor:
        """Backward-compatible alias for the shared constrained generation path."""
        generate_with_constraints = getattr(self, "_generate_with_constraints", None)
        if generate_with_constraints is None:
            generate_with_constraints = SMTTrainer._generate_with_constraints.__get__(
                self, type(self)
            )
        return generate_with_constraints(
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            max_length=max_length,
        )

    def forward(self, inputs, last_preds):
        return self.model(inputs, last_preds)

    def on_train_batch_start(self, batch, batch_idx):
        """Mark when the batch is ready for timing metrics."""
        self._batch_ready_time = time.perf_counter()

    def training_step(self, batch, _batch_idx):
        # Mark step start for timing metrics
        step_start = time.perf_counter()

        # Log data latency if batch_ready_time was recorded
        if hasattr(self, "_batch_ready_time") and self._batch_ready_time is not None:
            data_latency_ms = (step_start - self._batch_ready_time) * 1000.0
            self.log(
                "perf/data_to_step_ms", data_latency_ms, on_step=True, prog_bar=False, logger=True
            )

        x = batch["pixel_values"]
        labels = batch["labels"]
        dec_mask = batch.get("decoder_attention_mask", None)
        image_sizes = batch.get("image_sizes", None)

        self._mark_compiled_step_begin()
        outputs = self._forward_model()(
            pixel_values=x,
            labels=labels,
            image_sizes=image_sizes,
            attention_mask=dec_mask,  # True = valid
        )
        loss = outputs.loss

        stage = self.stage_calculator(self.global_step)

        self.log(
            TRAIN_LOSS,
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=x.size(0),
            prog_bar=False,
            sync_dist=True,
        )
        self.log(TRAIN_STAGE, stage, on_epoch=True, prog_bar=False)

        # Log step execution time
        step_end = time.perf_counter()
        step_time_ms = (step_end - step_start) * 1000.0
        self.log("perf/step_time_ms", step_time_ms, on_step=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, val_batch, batch_idx, dataloader_idx: int = 0):
        x = val_batch["pixel_values"]
        labels = val_batch["labels"]
        dec_mask = val_batch.get("decoder_attention_mask", None)
        image_sizes = val_batch.get("image_sizes", None)
        sample_ids = val_batch.get("sample_ids", None)
        sources = val_batch.get("source", None)

        # Get validation set name from dataloader index
        val_set_name = self._val_set_names[dataloader_idx]
        base_set_name = base_val_set_name(val_set_name)
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

        # Calculate the maximum actual sequence length in the batch (excluding -100 padding)
        # Add a small buffer (+2) to allow for EOS token and safety margin
        max_gt_length = (labels != -100).sum(dim=1).max().item()
        max_length_with_buffer = min(max_gt_length + 2, self.config.maxlen)

        # Get predicted token IDs with dynamic max_length
        # Uses grammar-constrained decoding if enabled
        preds = self._generate_with_grammar(
            pixel_values=x,
            image_sizes=image_sizes,
            max_length=max_length_with_buffer,
        )

        labels_for_metric = labels.masked_fill(labels == -100, self.config.pad_token_id)

        # Update metrics for this validation set
        self.val_metrics_by_set[val_set_name].update(preds, labels_for_metric)

        # Update runaway monitor diagnostics for this validation set (if enabled).
        runaway_trackers = getattr(self, "_val_runaway_tracker_by_set", None)
        if runaway_trackers is not None and not is_subset_loader:
            runaway_tracker = runaway_trackers.get(val_set_name)
            if runaway_tracker is not None:
                runaway_tracker.update_batch(
                    preds=preds,
                    targets=labels_for_metric,
                    max_length_cap=max_length_with_buffer,
                )

        # Accumulate samples for OMR-NED computation (if enabled)
        if self._compute_omr_ned and self._omr_ned_tracker is not None and not is_subset_loader:
            from src.core.text_processing import token_ids_to_string

            for i in range(len(preds)):
                pred_str = token_ids_to_string(
                    preds[i].tolist(), self.config.i2w, self.config.pad_token_id
                )
                target_str = token_ids_to_string(
                    labels_for_metric[i].tolist(), self.config.i2w, self.config.pad_token_id
                )
                self._omr_ned_tracker.update(pred_str, target_str, base_set_name)

        # Return outputs for WandbVisualizerCallback if logging is enabled.
        # Provide exact per-sample CER values over the full validation set.
        if self.should_log_validation_examples() and not is_subset_loader:
            if sample_ids is None:
                raise ValueError("Validation batch is missing sample_ids required for exact ranking.")
            if sources is None:
                sources = [None] * x.size(0)

            cers = [
                CharacterErrorRate.compute_single(
                    pred=preds[i].tolist(),
                    target=labels_for_metric[i].tolist(),
                    pad_id=self.config.pad_token_id,
                    i2w=self.config.i2w,
                )
                for i in range(x.size(0))
            ]
            cers_no_ties_beams = None
            if _is_polish_style_validation_set(base_set_name):
                cers_no_ties_beams = [
                    CharacterErrorRate.compute_single(
                        pred=preds[i].tolist(),
                        target=labels_for_metric[i].tolist(),
                        pad_id=self.config.pad_token_id,
                        i2w=self.config.i2w,
                        text_normalizer=strip_tie_beam_markers_from_kern_text,
                    )
                    for i in range(x.size(0))
                ]

            return {
                "val_set_name": val_set_name,
                "sample_ids": sample_ids.detach().cpu(),
                "sources": list(sources),
                "cers": cers,
                "cers_no_ties_beams": cers_no_ties_beams,
                "pred_ids": [preds[i].detach().cpu().tolist() for i in range(x.size(0))],
                "gt_ids": [labels_for_metric[i].detach().cpu().tolist() for i in range(x.size(0))],
            }
        return None

    def on_validation_epoch_start(self) -> None:
        self._val_batches_seen_by_set = {name: 0 for name in self._val_set_names}
        runaway_trackers = getattr(self, "_val_runaway_tracker_by_set", None)
        if runaway_trackers is not None:
            for tracker in runaway_trackers.values():
                tracker.reset()

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics at the end of each epoch.

        Visualization and example logging is now handled by WandbVisualizerCallback.
        """
        # Compute and log metrics for each validation set
        active_set_count = 0
        all_ser_values = []
        runaway_set_metrics: list[dict[str, float | int]] = []
        base_set_names = getattr(
            self,
            "_base_val_set_names",
            [name for name in self._val_set_names if not is_subset_val_set_name(name)],
        )
        runaway_rate_keys = (
            "runaway_rate",
            "runaway_length_blowup_rate",
            "runaway_repeat_loop_rate",
            "runaway_no_eos_at_max_length_rate",
            "runaway_max_length_hit_rate",
        )
        for set_name in self._val_set_names:
            metrics = self.val_metrics_by_set[set_name]
            seen_batches = self._val_batches_seen_by_set.get(set_name, 0)
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
                    active_set_count += 1
                    # Log metrics with per-set path (e.g., val/synth/SER, val/polish/CER)
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

            runaway_trackers = getattr(self, "_val_runaway_tracker_by_set", None)
            if runaway_trackers is not None:
                tracker = runaway_trackers.get(set_name)
                if tracker is not None:
                    if seen_batches > 0:
                        computed_runaway = tracker.compute()
                        runaway_set_metrics.append(computed_runaway)
                        for metric_name, value in computed_runaway.items():
                            self.log(
                                self._validation_set_metric_name(set_name, metric_name),
                                float(value),
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
        if self._val_set_names:
            self.log(
                self._validation_aggregate_metric_name("active_set_count"),
                float(active_set_count),
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                self._validation_aggregate_metric_name("is_full_pass"),
                1.0 if active_set_count == len(base_set_names) else 0.0,
                on_epoch=True,
                prog_bar=False,
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
                for key in runaway_rate_keys[1:]:
                    weighted_value = (
                        sum(
                            float(metrics.get(key, 0.0)) * int(metrics.get("runaway_samples", 0))
                            for metrics in runaway_set_metrics
                        )
                        / total_runaway_samples
                    )
                    self.log(
                        self._validation_aggregate_metric_name(key),
                        weighted_value,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )
                self.log(
                    self._validation_aggregate_metric_name("runaway_samples"),
                    float(total_runaway_samples),
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )

        # Compute and log OMR-NED metrics (if enabled)
        if self._compute_omr_ned and self._omr_ned_tracker is not None:
            omr_ned_metrics = self._omr_ned_tracker.compute()
            self.log(
                self._validation_aggregate_metric_name("OMR_NED"),
                omr_ned_metrics.overall_score,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                self._validation_aggregate_metric_name("OMR_NED_failures"),
                float(omr_ned_metrics.overall_failures),
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            for source, score in omr_ned_metrics.by_source_score.items():
                self.log(
                    self._validation_set_metric_name(source, "OMR_NED"),
                    score,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    self._validation_set_metric_name(source, "OMR_NED_failures"),
                    float(omr_ned_metrics.by_source_failures.get(source, 0)),
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )
            self._omr_ned_tracker.reset()

    def test_step(self, test_batch, _batch_idx):
        x = test_batch["pixel_values"]
        labels = test_batch["labels"]
        image_sizes = test_batch.get("image_sizes", None)

        # Calculate the maximum actual sequence length in the batch (excluding -100 padding)
        # Add a small buffer (+2) to allow for EOS token and safety margin
        max_gt_length = (labels != -100).sum(dim=1).max().item()
        max_length_with_buffer = min(max_gt_length + 2, self.config.maxlen)

        # Get predicted token IDs with dynamic max_length
        # Uses grammar-constrained decoding if enabled
        preds = self._generate_with_grammar(
            pixel_values=x,
            image_sizes=image_sizes,
            max_length=max_length_with_buffer,
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
                max_length_cap=max_length_with_buffer,
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
