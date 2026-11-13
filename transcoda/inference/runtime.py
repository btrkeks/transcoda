"""Runtime inference composition for Transcoda checkpoints."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import transformers
from PIL import Image

from transcoda.artifacts import RunArtifact
from transcoda.grammar.constraint_factory import ConstraintBundle, ConstrainedDecodingFactory
from transcoda.grammar.provider import GrammarProvider
from transcoda.grammar.semantic_sequence_finalizer import finalize_generated_kern_sequence
from transcoda.model import TranscodaModelForCausalLM
from transcoda.model.checkpoint_loader import load_model_from_checkpoint
from transcoda.model.generation_confidence import compute_batch_generation_confidences
from transcoda.model.generation_policy import (
    GenerationSettings,
    apply_generation_overrides,
    build_generate_kwargs,
    enforce_constraint_safe_settings,
    settings_from_decoding_spec,
)
from transcoda.preprocessing import (
    ImagePreprocessor,
    LayoutNormalizationConfig,
    WhitespaceTrimConfig,
)


@dataclass(frozen=True)
class PreprocessingOptions:
    """User-facing preprocessing toggles for inference."""

    normalize_layout: bool = False
    normalize_layout_top_margin_px: int = 36
    normalize_layout_side_margin_px: int = 36
    normalize_layout_threshold: int = 100
    normalize_layout_min_component_area_fraction: float = 0.00002
    whitespace_trim: bool = True
    whitespace_trim_threshold: int = 235
    whitespace_trim_margin_px: int = 36
    whitespace_trim_min_component_area_fraction: float = 0.00002

    def layout_config(self) -> LayoutNormalizationConfig:
        return LayoutNormalizationConfig(
            enabled=self.normalize_layout,
            top_margin_px=self.normalize_layout_top_margin_px,
            side_margin_px=self.normalize_layout_side_margin_px,
            threshold=self.normalize_layout_threshold,
            min_component_area_fraction=self.normalize_layout_min_component_area_fraction,
        )

    def whitespace_trim_config(self) -> WhitespaceTrimConfig:
        return WhitespaceTrimConfig(
            enabled=self.whitespace_trim,
            threshold=self.whitespace_trim_threshold,
            margin_px=self.whitespace_trim_margin_px,
            min_component_area_fraction=self.whitespace_trim_min_component_area_fraction,
        )


@dataclass(frozen=True)
class GenerationOptions:
    """Generation settings that may override checkpoint decoding defaults."""

    max_length: int | None = None
    strategy: str | None = None
    num_beams: int | None = None
    length_penalty: float | None = None
    repetition_penalty: float | None = None
    early_stopping: bool | str | None = None
    include_confidence: bool = False


def _resolve_spine_structure_enabled(
    *,
    use_grammar: bool,
    use_spine_structure_constraints: bool | None,
) -> bool:
    """Preserve legacy grammar+spine behavior while allowing either constraint alone."""
    if use_spine_structure_constraints is None:
        return bool(use_grammar)
    return bool(use_spine_structure_constraints)


@dataclass(frozen=True)
class InferenceResult:
    """Complete output for one image inference run."""

    kern_text: str
    token_count: int
    elapsed_s: float
    original_size: tuple[int, int]
    model_input_size: tuple[int, int]
    preprocessing_metadata: dict[str, Any]
    generation_metadata: dict[str, Any]
    confidence: dict[str, Any] | None
    model_input_tensor: torch.Tensor = field(repr=False)


def beam_count_override(
    generation_settings: GenerationSettings,
    generation: GenerationOptions,
) -> int | None:
    """Resolve a valid beam count when callers request beam search by name."""
    if generation.num_beams is not None:
        return generation.num_beams
    if generation.strategy == "beam" and generation_settings.num_beams <= 1:
        return 4
    return None


def get_device() -> torch.device:
    """Auto-detect the best available inference device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_input_image(image_tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized model-input tensor back to a viewable RGB image."""
    image_chw = image_tensor.detach().cpu()
    if image_chw.ndim == 4:
        image_chw = image_chw[0]
    if image_chw.ndim != 3 or image_chw.shape[0] != 3:
        raise ValueError(
            f"Expected image tensor shape (3,H,W) or (1,3,H,W), got {tuple(image_tensor.shape)}"
        )

    image_hwc = image_chw.clamp(-1.0, 1.0).permute(1, 2, 0)
    image_uint8 = ((image_hwc + 1.0) * 127.5).round().to(torch.uint8).numpy()
    return Image.fromarray(image_uint8)


class InferenceRuntime:
    """Loaded checkpoint runtime for repeated image inference."""

    def __init__(
        self,
        *,
        model: TranscodaModelForCausalLM,
        tokenizer: Any | None = None,
        tokenizer_load_path: str | Path | None = None,
        i2w: dict[int, str],
        pad_token_id: int,
        image_width: int,
        fixed_size: tuple[int, int] | None,
        artifact: RunArtifact,
        device: torch.device,
        weights_path: Path,
        vocab_dir: str | Path | None = None,
        grammar_provider: GrammarProvider | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_load_path = (
            None if tokenizer_load_path is None else str(tokenizer_load_path)
        )
        self.i2w = i2w
        self.pad_token_id = pad_token_id
        self.image_width = image_width
        self.fixed_size = fixed_size
        self.artifact = artifact
        self.device = device
        self.weights_path = weights_path
        self.vocab_dir = None if vocab_dir is None else str(vocab_dir)
        self.grammar_provider = grammar_provider

    @classmethod
    def load(
        cls,
        weights_path: str | Path,
        *,
        device: torch.device | None = None,
        use_grammar: bool = False,
        grammar_path: str | Path | None = None,
    ) -> InferenceRuntime:
        if use_grammar and (grammar_path is None or not str(grammar_path).strip()):
            raise ValueError("grammar_path is required when use_grammar=True")

        resolved_device = device or get_device()
        resolved_weights = Path(weights_path).expanduser().resolve()
        loaded = load_model_from_checkpoint(str(resolved_weights), resolved_device)
        grammar_vocab_dir = getattr(loaded, "tokenizer_load_path", None) or loaded.vocab_dir
        grammar_provider = None
        if use_grammar:
            grammar_provider = GrammarProvider(
                grammar_path=str(grammar_path),
                vocab_dir=grammar_vocab_dir,
                vocab_size=loaded.model.config.out_categories,
            )
        return cls(
            model=loaded.model,
            tokenizer=getattr(loaded, "tokenizer", None),
            tokenizer_load_path=getattr(loaded, "tokenizer_load_path", None),
            i2w=loaded.i2w,
            pad_token_id=loaded.pad_token_id,
            image_width=loaded.image_width,
            fixed_size=loaded.fixed_size,
            artifact=loaded.artifact,
            device=resolved_device,
            weights_path=resolved_weights,
            vocab_dir=grammar_vocab_dir,
            grammar_provider=grammar_provider,
        )

    def preprocess(
        self,
        image: Image.Image,
        options: PreprocessingOptions | None = None,
    ):
        resolved_options = options or PreprocessingOptions()
        preprocessor = ImagePreprocessor.from_checkpoint_metadata(
            image_width=self.image_width,
            fixed_size=self.fixed_size,
            layout_normalization=resolved_options.layout_config(),
            whitespace_trim=resolved_options.whitespace_trim_config(),
        )
        return preprocessor(image)

    def resolve_max_length(self, generation: GenerationOptions | None = None) -> int:
        generation = generation or GenerationOptions()
        return generation.max_length or self.artifact.decoding.max_len or self.model.config.maxlen

    def resolve_generation_settings(
        self,
        generation: GenerationOptions | None = None,
        *,
        use_grammar: bool = False,
        use_spine_structure_constraints: bool | None = None,
    ) -> GenerationSettings:
        generation = generation or GenerationOptions()
        generation_settings = settings_from_decoding_spec(self.artifact.decoding)
        num_beams_override = beam_count_override(generation_settings, generation)
        generation_settings = apply_generation_overrides(
            generation_settings,
            strategy=generation.strategy,
            num_beams=num_beams_override,
            length_penalty=generation.length_penalty,
            repetition_penalty=generation.repetition_penalty,
            early_stopping=generation.early_stopping,
        )
        use_spine_structure = _resolve_spine_structure_enabled(
            use_grammar=use_grammar,
            use_spine_structure_constraints=use_spine_structure_constraints,
        )
        generation_settings = enforce_constraint_safe_settings(
            generation_settings,
            has_constraints=use_grammar or use_spine_structure,
        )
        return generation_settings

    def _constraint_bundle(
        self,
        generation_settings: GenerationSettings,
        *,
        use_grammar: bool,
        use_spine_structure_constraints: bool | None,
        spine_structure_barline_width: str | None,
        spine_structure_empty_field_mode: str | None,
    ) -> ConstraintBundle:
        use_spine_structure = _resolve_spine_structure_enabled(
            use_grammar=use_grammar,
            use_spine_structure_constraints=use_spine_structure_constraints,
        )
        if not use_grammar and not use_spine_structure:
            return ConstraintBundle(
                logits_processors=None,
                stopping_criteria=None,
                generation_settings=generation_settings,
                semantic_rule_factories=(),
            )
        if self.grammar_provider is None:
            if use_grammar:
                raise RuntimeError("Grammar provider was not initialized.")
        constraint_factory = ConstrainedDecodingFactory(
            grammar_provider=self.grammar_provider if use_grammar else None,
            i2w=self.i2w,
            bos_token_id=getattr(self.model.config, "bos_token_id", None),
            eos_token_id=getattr(self.model.config, "eos_token_id", None),
            pad_token_id=self.pad_token_id,
            use_spine_structure_constraints=use_spine_structure,
            spine_structure_barline_width=(
                spine_structure_barline_width
                or self.artifact.decoding.spine_structure_barline_width
            ),
            spine_structure_empty_field_mode=(
                spine_structure_empty_field_mode
                or self.artifact.decoding.spine_structure_empty_field_mode
            ),
        )
        return constraint_factory.build(generation_settings)

    def _autocast_context(self, autocast_dtype: torch.dtype | None):
        if autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=autocast_dtype)

    def run(
        self,
        image: Image.Image,
        *,
        preprocessing: PreprocessingOptions | None = None,
        generation: GenerationOptions | None = None,
        use_grammar: bool = False,
        use_spine_structure_constraints: bool | None = None,
        spine_structure_barline_width: str | None = None,
        spine_structure_empty_field_mode: str | None = None,
        autocast_dtype: torch.dtype | None = None,
    ) -> InferenceResult:
        return self.run_batch(
            [image],
            preprocessing=preprocessing,
            generation=generation,
            use_grammar=use_grammar,
            use_spine_structure_constraints=use_spine_structure_constraints,
            spine_structure_barline_width=spine_structure_barline_width,
            spine_structure_empty_field_mode=spine_structure_empty_field_mode,
            autocast_dtype=autocast_dtype,
        )[0]

    def run_batch(
        self,
        images: Sequence[Image.Image],
        *,
        preprocessing: PreprocessingOptions | None = None,
        generation: GenerationOptions | None = None,
        use_grammar: bool = False,
        use_spine_structure_constraints: bool | None = None,
        spine_structure_barline_width: str | None = None,
        spine_structure_empty_field_mode: str | None = None,
        autocast_dtype: torch.dtype | None = None,
    ) -> list[InferenceResult]:
        if not images:
            return []
        generation = generation or GenerationOptions()
        preprocessed_batch = [self.preprocess(image, preprocessing) for image in images]
        image_tensor = torch.cat(
            [preprocessed.pixel_values.unsqueeze(0) for preprocessed in preprocessed_batch],
            dim=0,
        ).to(self.device)
        image_sizes = torch.tensor(
            [preprocessed.model_input_size for preprocessed in preprocessed_batch],
            device=self.device,
        )
        resolved_max_length = self.resolve_max_length(generation)
        generation_settings = self.resolve_generation_settings(
            generation,
            use_grammar=use_grammar,
            use_spine_structure_constraints=use_spine_structure_constraints,
        )
        constraint_bundle = self._constraint_bundle(
            generation_settings,
            use_grammar=use_grammar,
            use_spine_structure_constraints=use_spine_structure_constraints,
            spine_structure_barline_width=spine_structure_barline_width,
            spine_structure_empty_field_mode=spine_structure_empty_field_mode,
        )
        active_settings = constraint_bundle.generation_settings

        start_time = time.perf_counter()
        with torch.no_grad(), self._autocast_context(autocast_dtype):
            generate_kwargs = build_generate_kwargs(
                pixel_values=image_tensor,
                image_sizes=image_sizes,
                max_length=resolved_max_length,
                settings=active_settings,
                logits_processor=constraint_bundle.logits_processors,
            )
            if constraint_bundle.stopping_criteria is not None:
                generate_kwargs["stopping_criteria"] = transformers.StoppingCriteriaList(
                    constraint_bundle.stopping_criteria
                )
            if generation.include_confidence:
                generate_kwargs["return_dict_in_generate"] = True
                generate_kwargs["output_scores"] = True
            generation_output = self.model.generate(**generate_kwargs)
            predicted_ids = (
                generation_output.sequences if generation.include_confidence else generation_output
            )
        elapsed_s = time.perf_counter() - start_time

        batch_token_ids = [sequence.tolist() for sequence in predicted_ids]
        finalized_list = [
            finalize_generated_kern_sequence(
                token_ids=token_ids,
                i2w=self.i2w,
                bos_token_id=getattr(self.model.config, "bos_token_id", None),
                eos_token_id=getattr(self.model.config, "eos_token_id", None),
                pad_token_id=self.pad_token_id,
                max_length=resolved_max_length,
                rule_factories=constraint_bundle.semantic_rule_factories,
            )
            for token_ids in batch_token_ids
        ]
        confidences: list[dict[str, Any] | None]
        if generation.include_confidence:
            confidences = compute_batch_generation_confidences(
                model=self.model,
                generation_output=generation_output,
                batch_token_ids=batch_token_ids,
                i2w=self.i2w,
                pad_token_id=self.pad_token_id,
                eos_token_id=getattr(self.model.config, "eos_token_id", None),
                finalized_list=finalized_list,
                max_length=resolved_max_length,
                use_grammar=use_grammar,
                use_spine_structure_constraints=(
                    "spine_structure" in constraint_bundle.active_constraints
                ),
            )
        else:
            confidences = [None for _ in batch_token_ids]

        results: list[InferenceResult] = []
        for index, (preprocessed, token_ids, finalized, confidence) in enumerate(
            zip(preprocessed_batch, batch_token_ids, finalized_list, confidences, strict=True)
        ):
            token_count = sum(
                1
                for token_id in token_ids
                if token_id != self.pad_token_id
                and self.i2w.get(token_id) not in ("<bos>", "<eos>")
            )
            kern_text = finalized.text if finalized.text else ""
            results.append(
                InferenceResult(
                    kern_text=kern_text,
                    token_count=token_count,
                    elapsed_s=elapsed_s,
                    original_size=preprocessed.original_size,
                    model_input_size=preprocessed.model_input_size,
                    preprocessing_metadata=preprocessed.debug_metadata,
                    generation_metadata={
                        "max_length": resolved_max_length,
                        "strategy": active_settings.strategy,
                        "num_beams": active_settings.num_beams,
                        "device": str(self.device),
                        "use_grammar": use_grammar,
                        "use_spine_structure_constraints": (
                            "spine_structure" in constraint_bundle.active_constraints
                        ),
                        "active_constraints": list(constraint_bundle.active_constraints),
                        "spine_structure_barline_width": (
                            spine_structure_barline_width
                            or self.artifact.decoding.spine_structure_barline_width
                        ),
                        "spine_structure_empty_field_mode": (
                            spine_structure_empty_field_mode
                            or self.artifact.decoding.spine_structure_empty_field_mode
                        ),
                        "saw_eos": finalized.saw_eos,
                        "hit_max_length": finalized.hit_max_length,
                        "truncated": finalized.truncated,
                        "trimmed_incomplete_tail": finalized.trimmed_incomplete_tail,
                        "stopped_on_padding": finalized.stopped_on_padding,
                        "weights_path": str(self.weights_path),
                    },
                    confidence=confidence,
                    model_input_tensor=image_tensor[index : index + 1],
                )
            )
        return results
