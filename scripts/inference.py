"""Inference script with grammar-constrained decoding for OMR.

Usage:
    uv run scripts/inference.py \
        --weights ./weights/FP-Grandstaff-system-level.ckpt \
        --image path/to/image.png

Requires xgrammar: uv sync --group grammar
"""

import sys
import tempfile
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

import fire
import torch
from PIL import Image

from src.artifacts import RunArtifact
from src.grammar.constraint_factory import ConstrainedDecodingFactory
from src.core.kern_postprocess import append_terminator_if_missing
from src.data.preprocessing import LayoutNormalizationConfig, preprocess_pil_image
from src.grammar.provider import GrammarProvider
from src.grammar.semantic_sequence_finalizer import finalize_generated_kern_sequence
from src.model import SMTModelForCausalLM
from src.model.checkpoint_loader import LoadedCheckpoint, load_model_from_checkpoint
from src.model.generation_policy import (
    apply_generation_overrides,
    build_generate_kwargs,
    enforce_grammar_safe_settings,
    settings_from_decoding_spec,
)

# Constants
_PROJECT_ROOT = Path(__file__).parent.parent
GRAMMAR_PATH = str(_PROJECT_ROOT / "grammars" / "kern.gbnf")


def _get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _log(msg: str) -> None:
    """Print a log message."""
    print(f"[inference] {msg}")


def load_model_and_grammar(
    weights_path: str, device: torch.device
) -> tuple[
    SMTModelForCausalLM,
    dict,
    int,
    GrammarProvider,
    int,
    tuple[int, int] | None,
    RunArtifact,
]:
    """Load model, tokenizer, and grammar provider from checkpoint."""
    _log(f"Loading checkpoint: {weights_path}")
    loaded: LoadedCheckpoint = load_model_from_checkpoint(weights_path, device)
    _log(f"Loading tokenizer: {loaded.vocab_dir}")
    _log(f"Model loaded on {device}")

    # Create grammar provider
    _log("Compiling grammar...")
    grammar_provider = GrammarProvider(
        grammar_path=GRAMMAR_PATH,
        vocab_dir=loaded.vocab_dir,
        vocab_size=loaded.model.config.out_categories,
    )

    return (
        loaded.model,
        loaded.i2w,
        loaded.pad_token_id,
        grammar_provider,
        loaded.image_width,
        loaded.fixed_size,
        loaded.artifact,
    )


def preprocess_image(
    image_path: str,
    image_width: int,
    fixed_size: tuple[int, int] | None,
    layout_normalization: LayoutNormalizationConfig | None,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    """Load and preprocess image to target width.

    Args:
        image_path: Path to the image file.
        image_width: Target width in pixels (from artifact).
        device: Device to load tensor on.

    Returns:
        Tuple of (image_tensor, model_input_size_hw, original_size_hw)
    """
    with Image.open(image_path).convert("RGB") as img:
        original_size = (img.height, img.width)
        tensor, model_input_size = preprocess_pil_image(
            image=img,
            image_width=image_width,
            fixed_size=fixed_size,
            layout_normalization=layout_normalization,
        )
        _log(
            f"Image: {original_size[1]}x{original_size[0]} -> "
            f"{model_input_size[1]}x{model_input_size[0]}"
        )
        return tensor.unsqueeze(0).to(device), model_input_size, original_size


def _resolve_debug_image_path(image_path: Path, debug_preprocessed_image: str) -> Path:
    """Resolve output path for a saved model-input debug image."""
    debug_value = debug_preprocessed_image.strip()
    if debug_value.lower() == "tmp":
        return Path(tempfile.gettempdir()) / f"{image_path.stem}.model_input.png"
    return Path(debug_value)


def _save_debug_preprocessed_image(image_tensor: torch.Tensor, output_path: Path) -> None:
    """Save the exact model-input image tensor as a viewable PNG.

    The model consumes normalized float32 CHW data in [-1, 1]. For debugging,
    this inverts the normalization back to uint8 RGB so the post-resize /
    post-pad geometry can be inspected visually.
    """
    image_chw = image_tensor.detach().cpu()
    if image_chw.ndim == 4:
        image_chw = image_chw[0]
    if image_chw.ndim != 3 or image_chw.shape[0] != 3:
        raise ValueError(f"Expected image tensor shape (3,H,W) or (1,3,H,W), got {tuple(image_tensor.shape)}")

    image_hwc = image_chw.clamp(-1.0, 1.0).permute(1, 2, 0)
    image_uint8 = ((image_hwc + 1.0) * 127.5).round().to(torch.uint8).numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_uint8).save(output_path)
    _log(f"Saved preprocessed model-input image: {output_path}")


def inference(
    weights: str,
    image: str,
    output: str | None = None,
    debug_preprocessed_image: str | None = None,
    normalize_layout: bool = False,
    normalize_layout_top_margin_px: int = 36,
    normalize_layout_side_margin_px: int = 36,
    normalize_layout_threshold: int = 100,
    normalize_layout_min_component_area_fraction: float = 0.00002,
    max_length: int | None = None,
    strategy: str | None = None,
    num_beams: int | None = None,
    length_penalty: float | None = None,
    repetition_penalty: float | None = None,
    early_stopping: bool | str | None = None,
):
    """Run inference on an image with grammar enforcement.

    Args:
        weights: Path to model checkpoint (.ckpt file).
        image: Path to input image.
        output: Path to output .krn file. If not specified, uses <image_name>.krn
        debug_preprocessed_image: Optional path for saving the exact model-input
            image after resize/pad/crop. Use "tmp" to write under the system temp dir.
        normalize_layout: Reframe detected notation onto a synthetic white page
            canvas before standard resize/pad preprocessing.
        normalize_layout_top_margin_px: Target top margin on the reframed page.
        normalize_layout_side_margin_px: Minimum symmetric side margin target on
            the reframed page.
        normalize_layout_threshold: Darkness threshold used to detect notation.
        normalize_layout_min_component_area_fraction: Minimum connected-component
            area fraction retained in the notation mask.
        max_length: Maximum number of tokens to generate. Defaults to artifact setting.
        strategy: Optional decoding strategy override ("greedy" or "beam").
        num_beams: Optional beam width override.
        length_penalty: Optional beam search length penalty override.
        repetition_penalty: Optional repetition penalty override.
        early_stopping: Optional beam search early stopping override.
    """
    # Auto-detect device
    device = _get_device()
    _log(f"Using device: {device}")

    # Auto-generate output filename if not specified
    image_path = Path(image)
    if output is None:
        output_path = image_path.with_name(f"{image_path.stem}_pred.krn")
    else:
        output_path = Path(output)

    # Load model and grammar
    model, i2w, pad_token_id, grammar_provider, image_width, fixed_size, artifact = load_model_and_grammar(
        weights, device
    )
    generation_settings = settings_from_decoding_spec(artifact.decoding)
    generation_settings = apply_generation_overrides(
        generation_settings,
        strategy=strategy,
        num_beams=num_beams,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        early_stopping=early_stopping,
    )
    safe_settings = enforce_grammar_safe_settings(generation_settings)
    if safe_settings != generation_settings:
        _log(
            "Grammar-constrained decoding currently uses greedy mode for reliability; "
            "overriding beam settings (num_beams=1)."
        )
    generation_settings = safe_settings

    layout_normalization = LayoutNormalizationConfig(
        enabled=normalize_layout,
        top_margin_px=normalize_layout_top_margin_px,
        side_margin_px=normalize_layout_side_margin_px,
        threshold=normalize_layout_threshold,
        min_component_area_fraction=normalize_layout_min_component_area_fraction,
    )
    if layout_normalization.enabled:
        _log(
            "Layout normalization enabled "
            f"(top_margin_px={layout_normalization.top_margin_px}, "
            f"side_margin_px={layout_normalization.side_margin_px}, "
            f"threshold={layout_normalization.threshold}, "
            "min_component_area_fraction="
            f"{layout_normalization.min_component_area_fraction})"
        )

    # Preprocess image
    image_tensor, model_input_size, _original_size = preprocess_image(
        image,
        image_width,
        fixed_size,
        layout_normalization,
        device,
    )
    if debug_preprocessed_image:
        debug_output_path = _resolve_debug_image_path(image_path, debug_preprocessed_image)
        _save_debug_preprocessed_image(image_tensor, debug_output_path)

    # Generate with grammar constraints
    _log("Generating...")
    start_time = time.perf_counter()
    resolved_max_length = max_length
    if resolved_max_length is None:
        resolved_max_length = artifact.decoding.max_len or model.config.maxlen

    constraint_factory = ConstrainedDecodingFactory(
        grammar_provider=grammar_provider,
        i2w=i2w,
        bos_token_id=getattr(model.config, "bos_token_id", None),
        eos_token_id=getattr(model.config, "eos_token_id", None),
        pad_token_id=pad_token_id,
        use_spine_structure_constraints=True,
        # Preserve RhythmRule in code, but keep it out of live inference.
        use_rhythm_constraints=False,
        runaway_guard_enabled=False,
        runaway_guard_config=None,
    )
    constraint_bundle = constraint_factory.build(generation_settings)

    with torch.no_grad():
        generate_kwargs = build_generate_kwargs(
            pixel_values=image_tensor,
            image_sizes=torch.tensor([model_input_size], device=device),
            max_length=resolved_max_length,
            settings=constraint_bundle.generation_settings,
            logits_processor=constraint_bundle.logits_processors,
        )
        if constraint_bundle.stopping_criteria is not None:
            generate_kwargs["stopping_criteria"] = constraint_bundle.stopping_criteria
        predicted_ids = model.generate(**generate_kwargs)
    elapsed = time.perf_counter() - start_time

    # Count tokens (excluding special tokens)
    token_ids = predicted_ids[0].tolist()
    num_tokens = sum(1 for t in token_ids if t != pad_token_id and i2w.get(t) not in ("<bos>", "<eos>"))
    _log(f"Generated {num_tokens} tokens in {elapsed:.2f}s")

    # Convert to kern and write output
    finalized = finalize_generated_kern_sequence(
        token_ids=token_ids,
        i2w=i2w,
        bos_token_id=getattr(model.config, "bos_token_id", None),
        eos_token_id=getattr(model.config, "eos_token_id", None),
        pad_token_id=pad_token_id,
        max_length=resolved_max_length,
        rule_factories=constraint_bundle.semantic_rule_factories,
    )
    kern_text = "**kern\t**kern\n" + finalized.text if finalized.text else ""
    kern_text = append_terminator_if_missing(kern_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(kern_text)

    _log(f"Output: {output_path}")


if __name__ == "__main__":
    fire.Fire(inference)
