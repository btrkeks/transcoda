from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel

from src.pretraining.fcmae.lightning_module import FCMAEPretrainer

EXPORT_METADATA_FILENAME = "fcmae_export_metadata.json"
V1_IMPLEMENTATION_COMMIT = "57b2b0e"


def _collect_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _refuse_unsafe_overwrite(output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise FileExistsError(f"output path exists and is not a directory: {output_dir}")
    if any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}. Pass --overwrite to replace files."
        )


def _metadata(module: FCMAEPretrainer, checkpoint_path: Path) -> dict[str, Any]:
    pretraining_config = getattr(module, "full_config", None) or {
        "model": module.model_config.model_dump(),
        "training": module.training_config.model_dump(),
    }
    return {
        "source_checkpoint": str(checkpoint_path),
        "source_encoder_model_name_or_path": module.model_config.encoder_model_name_or_path,
        "pretraining_config": pretraining_config,
        "git_commit": _collect_git_commit(),
        "v1_implementation_commit": V1_IMPLEMENTATION_COMMIT,
        "exported_at_utc": datetime.now(UTC).isoformat(),
    }


def _validate_export(output_dir: Path, *, image_height: int, image_width: int) -> None:
    model = AutoModel.from_pretrained(output_dir)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, image_height, image_width))
    features = getattr(output, "last_hidden_state", output)
    if isinstance(features, tuple):
        features = features[0]
    if not torch.is_tensor(features) or features.ndim != 4:
        raise ValueError("export validation expected a 4D encoder feature map")


def export_fcmae_encoder(
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    validate: bool = False,
    validation_image_height: int = 768,
    validation_image_width: int = 544,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    _refuse_unsafe_overwrite(output_dir, overwrite=overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    module = FCMAEPretrainer.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    encoder = module.model.encoder
    if not hasattr(encoder, "save_pretrained"):
        raise TypeError("wrapped encoder does not expose save_pretrained")
    encoder.save_pretrained(output_dir)

    metadata = _metadata(module, checkpoint_path)
    (output_dir / EXPORT_METADATA_FILENAME).write_text(json.dumps(metadata, indent=2))

    if validate:
        _validate_export(
            output_dir,
            image_height=validation_image_height,
            image_width=validation_image_width,
        )

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the FCMAE encoder as a HF directory")
    parser.add_argument("checkpoint_path")
    parser.add_argument("output_dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validation-image-height", type=int, default=768)
    parser.add_argument("--validation-image-width", type=int, default=544)
    args = parser.parse_args()

    export_fcmae_encoder(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        validate=args.validate,
        validation_image_height=args.validation_image_height,
        validation_image_width=args.validation_image_width,
    )


if __name__ == "__main__":
    main()
