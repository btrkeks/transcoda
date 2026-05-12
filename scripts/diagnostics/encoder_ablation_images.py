"""Encoder ablation diagnostic for a folder of failing score images.

This answers: once a loop-like prefix has been generated, does replacing the
encoder output change the decoder's next-token distribution?
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.diagnostics._common import encode_image_path, load
from src.model.vision_frontend import VisionFrontendOutput

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class SampleResult:
    image: str
    donor_image: str
    generated_len: int
    prefix_len: int
    prefix_tail_tokens: list[str]
    top1_control: str
    top1_zeroed: str
    top1_shuffled: str
    top1_swapped: str
    kl_zeroed: float
    kl_shuffled: float
    kl_swapped: float
    argmax_agreement_zeroed: bool
    argmax_agreement_shuffled: bool
    argmax_agreement_swapped: bool


def _image_paths(image_dir: Path) -> list[Path]:
    paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return paths


def _kl(control_logits: torch.Tensor, ablated_logits: torch.Tensor) -> float:
    control_log_probs = F.log_softmax(control_logits, dim=-1)
    ablated_log_probs = F.log_softmax(ablated_logits, dim=-1)
    return F.kl_div(
        ablated_log_probs,
        control_log_probs,
        reduction="batchmean",
        log_target=True,
    ).item()


def _top1(logits: torch.Tensor, i2w: dict[int, str]) -> str:
    return i2w.get(int(logits.argmax(dim=-1).item()), "")


def _tokens(ids: torch.Tensor, i2w: dict[int, str]) -> list[str]:
    return [i2w.get(int(token_id), "") for token_id in ids.squeeze(0).tolist()]


def _ablate(
    enc: VisionFrontendOutput,
    mode: str,
    donor: VisionFrontendOutput | None = None,
) -> VisionFrontendOutput:
    if mode == "zero":
        return VisionFrontendOutput(
            encoder_tokens_raw=torch.zeros_like(enc.encoder_tokens_raw),
            encoder_tokens_pos=torch.zeros_like(enc.encoder_tokens_pos),
            encoder_attention_mask=enc.encoder_attention_mask,
        )
    if mode == "shuffle":
        perm = torch.randperm(enc.encoder_tokens_raw.size(1), device=enc.encoder_tokens_raw.device)
        return VisionFrontendOutput(
            # Keep positional tokens fixed while moving visual content between positions.
            # Shuffling keys/values and positions together is mostly permutation-invariant
            # for cross-attention, so it is not a useful ablation.
            encoder_tokens_raw=enc.encoder_tokens_raw[:, perm],
            encoder_tokens_pos=enc.encoder_tokens_pos,
            encoder_attention_mask=enc.encoder_attention_mask,
        )
    if mode == "swap":
        if donor is None:
            raise ValueError("swap ablation requires donor encoder output")
        return VisionFrontendOutput(
            encoder_tokens_raw=donor.encoder_tokens_raw,
            encoder_tokens_pos=donor.encoder_tokens_pos,
            encoder_attention_mask=donor.encoder_attention_mask,
        )
    raise ValueError(f"Unknown ablation mode: {mode}")


@torch.no_grad()
def _build_prefix(
    model,
    pixel_values: torch.Tensor,
    image_sizes: torch.Tensor,
    prefix_len: int,
    generation_margin: int,
) -> torch.Tensor:
    generated = model.generate(
        pixel_values=pixel_values,
        image_sizes=image_sizes,
        do_sample=False,
        num_beams=1,
        max_length=prefix_len + generation_margin,
        use_cache=True,
        repetition_penalty=1.0,
    )
    if generated.size(1) < 2:
        raise RuntimeError("Generation produced fewer than two tokens; cannot ablate next token.")
    if generated.size(1) < prefix_len:
        return generated
    return generated[:, :prefix_len]


@torch.no_grad()
def _next_token_logits(model, prefix_ids: torch.Tensor, enc: VisionFrontendOutput) -> torch.Tensor:
    out = model.forward(
        input_ids=prefix_ids,
        encoder_outputs=enc,
        encoder_attention_mask=enc.encoder_attention_mask,
        use_cache=False,
    )
    return out.logits[:, -1, :]


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": math.nan, "median": math.nan}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {"mean": sum(values) / len(values), "median": median}


def _aggregate(results: list[SampleResult]) -> dict[str, object]:
    n = len(results)
    return {
        "n": n,
        "agreement_rate_zeroed": sum(r.argmax_agreement_zeroed for r in results) / n,
        "agreement_rate_shuffled": sum(r.argmax_agreement_shuffled for r in results) / n,
        "agreement_rate_swapped": sum(r.argmax_agreement_swapped for r in results) / n,
        "kl_zeroed": _summarize([r.kl_zeroed for r in results]),
        "kl_shuffled": _summarize([r.kl_shuffled for r in results]),
        "kl_swapped": _summarize([r.kl_swapped for r in results]),
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    image_dir = Path(args.image_dir)
    images = _image_paths(image_dir)
    if len(images) < 2:
        raise ValueError("Need at least two images so swap ablation can use a donor image.")

    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded, device = load(args.weights, args.device)
    model = loaded.model

    encoded: dict[Path, tuple[torch.Tensor, torch.Tensor, VisionFrontendOutput]] = {}
    for image_path in images:
        pixel_values, image_sizes = encode_image_path(loaded, image_path, device)
        enc = model.forward_encoder(pixel_values, image_sizes=image_sizes)
        encoded[image_path] = (pixel_values, image_sizes, enc)

    results: list[SampleResult] = []
    for index, image_path in enumerate(images):
        donor_path = images[(index + args.donor_offset) % len(images)]
        if donor_path == image_path:
            donor_path = images[(index + 1) % len(images)]

        pixel_values, image_sizes, enc = encoded[image_path]
        donor_enc = encoded[donor_path][2]

        prefix = _build_prefix(
            model=model,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            prefix_len=args.prefix_len,
            generation_margin=args.generation_margin,
        )

        logits_real = _next_token_logits(model, prefix, enc)
        logits_zero = _next_token_logits(model, prefix, _ablate(enc, "zero"))
        logits_shuf = _next_token_logits(model, prefix, _ablate(enc, "shuffle"))
        logits_swap = _next_token_logits(model, prefix, _ablate(enc, "swap", donor_enc))

        top1_control = _top1(logits_real, loaded.i2w)
        top1_zeroed = _top1(logits_zero, loaded.i2w)
        top1_shuffled = _top1(logits_shuf, loaded.i2w)
        top1_swapped = _top1(logits_swap, loaded.i2w)
        prefix_tail_tokens = _tokens(prefix[:, -args.prefix_tail :], loaded.i2w)

        result = SampleResult(
            image=str(image_path),
            donor_image=str(donor_path),
            generated_len=int(prefix.size(1)),
            prefix_len=int(prefix.size(1)),
            prefix_tail_tokens=prefix_tail_tokens,
            top1_control=top1_control,
            top1_zeroed=top1_zeroed,
            top1_shuffled=top1_shuffled,
            top1_swapped=top1_swapped,
            kl_zeroed=_kl(logits_real, logits_zero),
            kl_shuffled=_kl(logits_real, logits_shuf),
            kl_swapped=_kl(logits_real, logits_swap),
            argmax_agreement_zeroed=top1_control == top1_zeroed,
            argmax_agreement_shuffled=top1_control == top1_shuffled,
            argmax_agreement_swapped=top1_control == top1_swapped,
        )
        results.append(result)

        per_image_path = out_dir / f"{image_path.stem}.json"
        per_image_path.write_text(json.dumps(asdict(result), indent=2))
        print(
            f"{image_path.name}: control={top1_control!r} "
            f"zero={top1_zeroed!r} shuffle={top1_shuffled!r} swap={top1_swapped!r} "
            f"kl_swap={result.kl_swapped:.4f}"
        )

    report = {
        "weights": args.weights,
        "image_dir": str(image_dir),
        "device": str(device),
        "prefix_len_requested": args.prefix_len,
        "donor_offset": args.donor_offset,
        "aggregate": _aggregate(results),
        "results": [asdict(r) for r in results],
    }
    report_path = out_dir / "aggregate.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {report_path}")
    print(json.dumps(report["aggregate"], indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Checkpoint path to evaluate.")
    parser.add_argument("--image-dir", default="failing_polish_scores", help="Folder of failing score images.")
    parser.add_argument("--out-dir", default="docs/reports/diag-encoder-ablation-images")
    parser.add_argument(
        "--run-id",
        default=datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Subdirectory name under --out-dir.",
    )
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', or 'cpu'.")
    parser.add_argument("--prefix-len", type=int, default=120)
    parser.add_argument("--generation-margin", type=int, default=20)
    parser.add_argument("--prefix-tail", type=int, default=40)
    parser.add_argument("--donor-offset", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
