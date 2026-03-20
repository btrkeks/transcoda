"""Model adapters used by the benchmark runner."""

from __future__ import annotations

import sys
import warnings
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, AutoTokenizer, GenerationConfig

if TYPE_CHECKING:
    from src.data.preprocessing import LayoutNormalizationConfig

from src.core.metrics.runaway_monitor import (
    CatastrophicLoopConfig,
    CatastrophicLoopDiagnostics,
    analyze_catastrophic_repetition,
)

RawFormat = Literal["kern", "abc"]


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device=device)


def _count_generated_tokens(
    *,
    token_ids: list[int],
    bos_token_id: int | None,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> int:
    count = 0
    for token_id in token_ids:
        if pad_token_id is not None and token_id == pad_token_id:
            break
        if bos_token_id is not None and token_id == bos_token_id:
            continue
        if eos_token_id is not None and token_id == eos_token_id:
            break
        count += 1
    return count


@dataclass(frozen=True)
class AdapterProfilingContext:
    collect_profile: bool = False
    trace_path: Path | None = None


@runtime_checkable
class BenchmarkModelAdapter(Protocol):
    """Common inference interface for all benchmarked models."""

    @property
    def name(self) -> str:
        ...

    @property
    def raw_format(self) -> RawFormat:
        ...

    def predict_batch(self, images: list[Image.Image]) -> list[str]:
        ...


def resolve_device(device: str) -> torch.device:
    """Resolve CLI device selection into a torch device."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_legacy_transformers_config_compat(config_class: type | None) -> None:
    """Backfill config defaults expected by newer transformers releases."""
    if config_class is None:
        return
    defaults = {
        "is_encoder_decoder": False,
        "tie_word_embeddings": False,
        "_attn_implementation_internal": None,
    }
    for name, value in defaults.items():
        if not hasattr(config_class, name):
            setattr(config_class, name, value)


def _looks_like_local_path(reference: str) -> bool:
    expanded = Path(reference).expanduser()
    return expanded.is_absolute() or reference.startswith(("./", "../", "~"))


class OurCheckpointAdapter:
    """Adapter for this repo's SMT checkpoint with grammar-enabled decoding."""

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        *,
        layout_normalization: "LayoutNormalizationConfig | None" = None,
        strategy: str | None = None,
        num_beams: int | None = None,
        repetition_penalty: float | None = None,
        use_constraints: bool = True,
        profiling_enabled: bool = False,
        loop_recovery_enabled: bool = False,
        loop_recovery_repetition_penalty: float = 1.35,
    ) -> None:
        from src.grammar.constraint_factory import ConstrainedDecodingFactory
        from src.grammar.interpretation_transition_rule import InterpretationTransitionConfig
        from src.grammar.provider import GrammarProvider
        from src.model.checkpoint_loader import load_model_from_checkpoint
        from src.model.generation_policy import (
            apply_generation_overrides,
            enforce_constraint_safe_settings,
            settings_from_decoding_spec,
        )

        loaded = load_model_from_checkpoint(checkpoint_path, device)
        self.device = device
        self.model = loaded.model
        self.i2w = loaded.i2w
        self.pad_token_id = loaded.pad_token_id
        self.artifact = loaded.artifact
        self.image_width = loaded.image_width
        self.fixed_size = loaded.fixed_size
        self.checkpoint_generation_settings = settings_from_decoding_spec(loaded.artifact.decoding)
        self.requested_generation_settings = apply_generation_overrides(
            self.checkpoint_generation_settings,
            strategy=strategy,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )
        self.use_constraints = bool(use_constraints)
        self.effective_generation_settings = enforce_constraint_safe_settings(
            self.requested_generation_settings,
            has_constraints=self.use_constraints,
        )
        self.generation_settings = self.requested_generation_settings
        self.max_length = loaded.artifact.decoding.max_len or self.model.config.maxlen
        self._warned_constraint_beam_downgrade = False
        if self.use_constraints and self.requested_generation_settings.strategy == "beam":
            warnings.warn(
                "Benchmark constrained decoding for our adapter is not beam-safe; "
                "downgrading the effective decode policy to greedy single-beam. "
                "Pass --disable-constraints to benchmark unconstrained beam search.",
                UserWarning,
                stacklevel=2,
            )
            self._warned_constraint_beam_downgrade = True
        self.grammar_provider = None
        if self.use_constraints:
            self.grammar_provider = GrammarProvider(
                grammar_path="grammars/kern.gbnf",
                vocab_dir=loaded.vocab_dir,
                vocab_size=self.model.config.out_categories,
            )
        self.constraint_factory = ConstrainedDecodingFactory(
            grammar_provider=self.grammar_provider,
            i2w=self.i2w,
            bos_token_id=getattr(self.model.config, "bos_token_id", None),
            eos_token_id=getattr(self.model.config, "eos_token_id", None),
            pad_token_id=self.pad_token_id,
            use_interpretation_transition_constraints=self.use_constraints,
            use_spine_structure_constraints=self.use_constraints,
            # Preserve RhythmRule in code, but keep it out of benchmark inference.
            use_rhythm_constraints=False,
            interpretation_transition_config=InterpretationTransitionConfig(),
            runaway_guard_enabled=False,
            runaway_guard_config=None,
            collect_stats=profiling_enabled,
        )
        self.layout_normalization = layout_normalization
        self.profiling_enabled = bool(profiling_enabled)
        self.loop_recovery_enabled = bool(loop_recovery_enabled)
        self.loop_recovery_config = CatastrophicLoopConfig()
        recovery_penalty = float(loop_recovery_repetition_penalty)
        if recovery_penalty <= self.generation_settings.repetition_penalty:
            recovery_penalty = self.generation_settings.repetition_penalty + 0.05
        self.loop_recovery_settings = apply_generation_overrides(
            self.requested_generation_settings,
            repetition_penalty=recovery_penalty,
        )
        self._profiling_context = AdapterProfilingContext()
        self._last_profile: dict[str, Any] | None = None
        self._name = "ours"
        self._raw_format: RawFormat = "kern"
        self._bos_token_id = getattr(self.model.config, "bos_token_id", None)
        self._eos_token_id = getattr(self.model.config, "eos_token_id", None)

    @staticmethod
    def _serialize_generation_settings(settings) -> dict[str, Any]:
        return {
            "strategy": settings.strategy,
            "num_beams": int(settings.num_beams),
            "length_penalty": float(settings.length_penalty),
            "repetition_penalty": float(settings.repetition_penalty),
            "early_stopping": settings.early_stopping,
            "num_return_sequences": int(settings.num_return_sequences),
            "use_cache": bool(settings.use_cache),
            "do_sample": bool(settings.do_sample),
        }

    @property
    def decode_policy(self) -> dict[str, Any]:
        return {
            "requested": {
                "constraints_enabled": self.use_constraints,
                "checkpoint": self._serialize_generation_settings(self.checkpoint_generation_settings),
                "overrides": {
                    "strategy": self.requested_generation_settings.strategy
                    if self.requested_generation_settings.strategy
                    != self.checkpoint_generation_settings.strategy
                    else None,
                    "num_beams": self.requested_generation_settings.num_beams
                    if self.requested_generation_settings.num_beams
                    != self.checkpoint_generation_settings.num_beams
                    else None,
                },
                "resolved": self._serialize_generation_settings(self.requested_generation_settings),
            },
            "effective": {
                "constraints_enabled": self.use_constraints,
                "settings": self._serialize_generation_settings(self.effective_generation_settings),
                "downgraded_to_greedy": self._warned_constraint_beam_downgrade,
            },
        }

    @property
    def name(self) -> str:
        return self._name

    @property
    def raw_format(self) -> RawFormat:
        return self._raw_format

    def prepare_profile(
        self,
        *,
        collect_profile: bool,
        trace_path: Path | None = None,
    ) -> None:
        self._profiling_context = AdapterProfilingContext(
            collect_profile=bool(collect_profile),
            trace_path=trace_path,
        )
        self._last_profile = None

    def consume_last_profile(self) -> dict[str, Any] | None:
        profile = self._last_profile
        self._last_profile = None
        return profile

    def _collect_constraint_stats(self, logits_processors: list[Any] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for processor in logits_processors or []:
            stats_fn = getattr(processor, "stats", None)
            if not callable(stats_fn):
                continue
            stats = stats_fn()
            if processor.__class__.__name__ == "GrammarConstrainedLogitsProcessor":
                payload["grammar"] = stats
            elif processor.__class__.__name__ == "StatefulKernLogitsProcessor":
                payload["semantic"] = stats
        return payload

    def _run_generation_batch(
        self,
        images: list[Image.Image],
        *,
        generation_settings,
        trace_path: Path | None = None,
    ) -> dict[str, Any]:
        from src.data.preprocessing import preprocess_pil_image
        from src.grammar.semantic_sequence_finalizer import finalize_generated_kern_sequence
        from src.model.generation_policy import build_generate_kwargs

        preprocess_started_at = perf_counter()
        pixel_values = []
        image_sizes = []
        for image in images:
            tensor, model_input_size = preprocess_pil_image(
                image=image,
                image_width=self.image_width,
                fixed_size=self.fixed_size,
                layout_normalization=self.layout_normalization,
            )
            pixel_values.append(tensor)
            image_sizes.append(model_input_size)
        preprocess_ms = (perf_counter() - preprocess_started_at) * 1000.0

        tensor_started_at = perf_counter()
        batch = torch.stack(pixel_values, dim=0).to(self.device)
        sizes = torch.tensor(image_sizes, dtype=torch.long, device=self.device)
        _synchronize_device(self.device)
        tensor_setup_ms = (perf_counter() - tensor_started_at) * 1000.0

        bundle_started_at = perf_counter()
        constraint_bundle = self.constraint_factory.build(generation_settings)
        constraint_bundle_ms = (perf_counter() - bundle_started_at) * 1000.0

        generate_started_at = perf_counter()
        with torch.no_grad():
            generate_kwargs = build_generate_kwargs(
                pixel_values=batch,
                image_sizes=sizes,
                max_length=self.max_length,
                settings=constraint_bundle.generation_settings,
                logits_processor=constraint_bundle.logits_processors,
            )
            if constraint_bundle.stopping_criteria is not None:
                generate_kwargs["stopping_criteria"] = constraint_bundle.stopping_criteria
            if trace_path is not None:
                from torch.profiler import ProfilerActivity, profile

                activities = [ProfilerActivity.CPU]
                if self.device.type == "cuda":
                    activities.append(ProfilerActivity.CUDA)
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                with profile(activities=activities, record_shapes=True) as profiler:
                    generated = self.model.generate(**generate_kwargs)
                profiler.export_chrome_trace(str(trace_path))
            else:
                generated = self.model.generate(**generate_kwargs)
        _synchronize_device(self.device)
        generate_ms = (perf_counter() - generate_started_at) * 1000.0

        finalize_started_at = perf_counter()
        outputs: list[str] = []
        generated_lengths: list[int] = []
        for sequence in generated:
            token_ids = sequence.tolist()
            finalized = finalize_generated_kern_sequence(
                token_ids=token_ids,
                i2w=self.i2w,
                bos_token_id=self._bos_token_id,
                eos_token_id=self._eos_token_id,
                pad_token_id=self.pad_token_id,
                max_length=self.max_length,
                rule_factories=constraint_bundle.semantic_rule_factories,
            )
            outputs.append(finalized.text)
            generated_lengths.append(
                _count_generated_tokens(
                    token_ids=token_ids,
                    bos_token_id=self._bos_token_id,
                    eos_token_id=self._eos_token_id,
                    pad_token_id=self.pad_token_id,
                )
            )
        finalize_ms = (perf_counter() - finalize_started_at) * 1000.0

        return {
            "outputs": outputs,
            "generated_lengths": generated_lengths,
            "preprocess_ms": preprocess_ms,
            "tensor_setup_ms": tensor_setup_ms,
            "constraint_bundle_ms": constraint_bundle_ms,
            "generate_ms": generate_ms,
            "finalize_ms": finalize_ms,
            "constraint_stats": self._collect_constraint_stats(constraint_bundle.logits_processors),
            "trace_path": None if trace_path is None else str(trace_path),
        }

    @staticmethod
    def _diagnostics_to_dict(diagnostics: Any) -> dict[str, Any]:
        if is_dataclass(diagnostics):
            return asdict(diagnostics)
        return {
            "max_identical_line_run": int(getattr(diagnostics, "max_identical_line_run")),
            "max_repeated_ngram_occurrences": int(
                getattr(diagnostics, "max_repeated_ngram_occurrences")
            ),
            "repeated_ngram_size": int(getattr(diagnostics, "repeated_ngram_size")),
            "repeated_ngram_line_coverage": float(
                getattr(diagnostics, "repeated_ngram_line_coverage")
            ),
            "repeat_loop": bool(getattr(diagnostics, "repeat_loop")),
            "repeat_loop_reason": getattr(diagnostics, "repeat_loop_reason"),
        }

    @staticmethod
    def _build_loop_recovery_payload(
        primary: CatastrophicLoopDiagnostics,
        *,
        rerun_attempted: bool,
        rerun: CatastrophicLoopDiagnostics | None = None,
        replaced_prediction: bool = False,
    ) -> dict[str, Any]:
        return {
            "primary_detected": primary.repeat_loop,
            "primary_reason": primary.repeat_loop_reason,
            "primary_metrics": OurCheckpointAdapter._diagnostics_to_dict(primary),
            "rerun_attempted": rerun_attempted,
            "rerun_detected": None if rerun is None else rerun.repeat_loop,
            "rerun_reason": None if rerun is None else rerun.repeat_loop_reason,
            "rerun_metrics": None if rerun is None else OurCheckpointAdapter._diagnostics_to_dict(rerun),
            "replaced_prediction": replaced_prediction,
        }

    def predict_batch_rows(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        trace_path = self._profiling_context.trace_path
        primary = self._run_generation_batch(
            images,
            generation_settings=self.generation_settings,
            trace_path=trace_path,
        )
        outputs = list(primary["outputs"])
        generated_lengths = list(primary["generated_lengths"])
        total_preprocess_ms = float(primary["preprocess_ms"])
        total_tensor_setup_ms = float(primary["tensor_setup_ms"])
        total_constraint_bundle_ms = float(primary["constraint_bundle_ms"])
        total_generate_ms = float(primary["generate_ms"])
        total_finalize_ms = float(primary["finalize_ms"])

        rows: list[dict[str, Any]] = []
        for index, prediction in enumerate(outputs):
            row: dict[str, Any] = {"prediction": prediction}
            if not self.loop_recovery_enabled:
                rows.append(row)
                continue

            primary_diag = analyze_catastrophic_repetition(
                prediction,
                config=self.loop_recovery_config,
            )
            rerun_diag: CatastrophicLoopDiagnostics | None = None
            replaced_prediction = False
            if primary_diag.repeat_loop:
                rerun_result = self._run_generation_batch(
                    [images[index]],
                    generation_settings=self.loop_recovery_settings,
                    trace_path=None,
                )
                total_preprocess_ms += float(rerun_result["preprocess_ms"])
                total_tensor_setup_ms += float(rerun_result["tensor_setup_ms"])
                total_constraint_bundle_ms += float(rerun_result["constraint_bundle_ms"])
                total_generate_ms += float(rerun_result["generate_ms"])
                total_finalize_ms += float(rerun_result["finalize_ms"])
                generated_lengths.extend(rerun_result["generated_lengths"])
                rerun_prediction = rerun_result["outputs"][0]
                rerun_diag = analyze_catastrophic_repetition(
                    rerun_prediction,
                    config=self.loop_recovery_config,
                )
                if not rerun_diag.repeat_loop:
                    row["prediction"] = rerun_prediction
                    replaced_prediction = True

            row["loop_recovery"] = self._build_loop_recovery_payload(
                primary_diag,
                rerun_attempted=primary_diag.repeat_loop,
                rerun=rerun_diag,
                replaced_prediction=replaced_prediction,
            )
            rows.append(row)

        if self.profiling_enabled:
            total_generated_tokens = int(sum(generated_lengths))
            self._last_profile = {
                "preprocess_ms": total_preprocess_ms,
                "tensor_setup_ms": total_tensor_setup_ms,
                "constraint_bundle_ms": total_constraint_bundle_ms,
                "generate_ms": total_generate_ms,
                "finalize_ms": total_finalize_ms,
                "generated_tokens": total_generated_tokens,
                "generated_tokens_per_sample": generated_lengths,
                "generate_tokens_per_second": (
                    total_generated_tokens / (total_generate_ms / 1000.0)
                ) if total_generate_ms > 0 else 0.0,
                "constraint_stats": primary["constraint_stats"],
                "trace_path": primary["trace_path"],
            }
        return rows

    def predict_batch(self, images: list[Image.Image]) -> list[str]:
        rows = self.predict_batch_rows(images)
        return [str(row["prediction"]) for row in rows]


class SMTPlusPlusAdapter:
    """Adapter for PRAIG's full-page SMT++ model."""

    def __init__(
        self,
        model_id: str,
        device: torch.device,
        *,
        max_length: int | None = None,
        praig_module_path: str = "models/external/praig-smt",
    ) -> None:
        praig_path = Path(praig_module_path).resolve()
        if str(praig_path) not in sys.path:
            sys.path.insert(0, str(praig_path))

        from smt_model import SMTModelForCausalLM as PRAIGModel

        ensure_legacy_transformers_config_compat(getattr(PRAIGModel, "config_class", None))

        self.device = device
        self.model = PRAIGModel.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()
        self.decode_max_length = max_length
        if self.decode_max_length is not None:
            self.model.maxlen = min(self.model.maxlen, self.decode_max_length)
        self._name = "smtpp"
        self._raw_format: RawFormat = "kern"

    @property
    def name(self) -> str:
        return self._name

    @property
    def raw_format(self) -> RawFormat:
        return self._raw_format

    def predict_batch(self, images: list[Image.Image]) -> list[str]:
        tensors: list[torch.Tensor] = []
        image_sizes: list[tuple[int, int]] = []
        max_height = 0
        max_width = 0

        for image in images:
            grayscale = image.convert("L")
            tensor = torch.from_numpy(np.array(grayscale)).float().div(255.0).unsqueeze(0)
            _, height, width = tensor.shape
            tensors.append(tensor)
            image_sizes.append((height, width))
            max_height = max(max_height, height)
            max_width = max(max_width, width)

        batch = torch.ones(
            (len(tensors), 1, max_height, max_width),
            dtype=torch.float32,
        )
        for index, tensor in enumerate(tensors):
            _, height, width = tensor.shape
            batch[index, :, :height, :width] = tensor

        image_sizes_tensor = torch.tensor(image_sizes, dtype=torch.long)
        with torch.no_grad():
            text_sequences, _ = self.model.predict(
                batch.to(self.device),
                image_sizes=image_sizes_tensor.to(self.device),
                convert_to_str=True,
            )

        if len(images) == 1:
            text_sequences = [text_sequences]

        outputs: list[str] = []
        for text_sequence in text_sequences:
            kern = "".join(text_sequence)
            kern = kern.replace("<b>", "\n").replace("<s>", " ").replace("<t>", "\t")
            outputs.append(kern)

        return outputs


class LegatoAdapter:
    """Adapter for LEGATO ABC inference."""

    def __init__(
        self,
        model_id: str,
        device: torch.device,
        *,
        encoder_path: str | None = None,
        max_length: int = 2048,
        num_beams: int = 10,
        legato_module_path: str = "models/external/legato",
    ) -> None:
        legato_path = Path(legato_module_path).resolve()
        if str(legato_path) not in sys.path:
            sys.path.insert(0, str(legato_path))

        from legato.models import LegatoModel
        from legato.models.processing_legato import LegatoProcessor

        self.device = device
        if encoder_path is not None and _looks_like_local_path(encoder_path):
            resolved_encoder_path = Path(encoder_path).expanduser()
            if not resolved_encoder_path.exists():
                raise FileNotFoundError(
                    "LEGATO encoder override path does not exist: "
                    f"{resolved_encoder_path}"
                )

        if encoder_path is None:
            self.model = LegatoModel.from_pretrained(model_id)
        else:
            config = AutoConfig.from_pretrained(model_id)
            config.encoder_pretrained_model_name_or_path = encoder_path
            self.model = LegatoModel.from_pretrained(model_id, config=config)
        self.model.to(device=device)
        self.model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        image_processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
        self.processor = LegatoProcessor(image_processor=image_processor, tokenizer=tokenizer)
        self.processor.tokenizer = tokenizer
        self.processor.image_processor = image_processor
        self.generation_config = GenerationConfig(
            max_length=max_length,
            num_beams=num_beams,
            repetition_penalty=1.1,
        )
        self._name = "legato"
        self._raw_format: RawFormat = "abc"

    @property
    def name(self) -> str:
        return self._name

    @property
    def raw_format(self) -> RawFormat:
        return self._raw_format

    def predict_batch(self, images: list[Image.Image]) -> list[str]:
        inputs = self.processor(images=images, truncation=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                use_model_defaults=False,
            )

        return self.processor.batch_decode(outputs.tolist(), skip_special_tokens=True)
