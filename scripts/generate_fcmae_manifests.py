from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def _normalized_relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _stable_key(seed: int, relative_path: str, *, salt: str = "split") -> str:
    payload = f"{salt}\0{seed}\0{relative_path}".encode()
    return hashlib.sha256(payload).hexdigest()


def collect_images(image_root: Path, extensions: tuple[str, ...]) -> list[Path]:
    suffixes = {extension.lower() if extension.startswith(".") else f".{extension.lower()}" for extension in extensions}
    return sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes and path.stat().st_size > 0
    )


def split_images(
    image_paths: list[Path],
    *,
    image_root: Path,
    validation_size: int,
    preview_size: int,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    if validation_size < 1:
        raise ValueError("validation_size must be >= 1")
    if preview_size < 1:
        raise ValueError("preview_size must be >= 1")
    if validation_size >= len(image_paths):
        raise ValueError(
            f"validation_size={validation_size} leaves no training images from {len(image_paths)} inputs"
        )

    by_split_key = sorted(
        image_paths,
        key=lambda path: _stable_key(seed, _normalized_relative_path(path, image_root)),
    )
    validation = sorted(by_split_key[:validation_size])
    train = sorted(by_split_key[validation_size:])
    preview = sorted(
        validation,
        key=lambda path: _stable_key(seed, _normalized_relative_path(path, image_root), salt="preview"),
    )[: min(preview_size, len(validation))]
    return train, validation, sorted(preview)


def _manifest_line(path: Path, manifest_dir: Path) -> str:
    return Path(os.path.relpath(path, manifest_dir)).as_posix()


def write_manifest(path: Path, image_paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_manifest_line(image_path, path.parent) for image_path in image_paths]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_split_metadata(
    path: Path,
    *,
    image_root: Path,
    train: list[Path],
    validation: list[Path],
    preview: list[Path],
    validation_size: int,
    preview_size: int,
    seed: int,
    extensions: tuple[str, ...],
) -> None:
    metadata = {
        "image_root": str(image_root),
        "extensions": list(extensions),
        "seed": seed,
        "validation_size_requested": validation_size,
        "preview_size_requested": preview_size,
        "train_count": len(train),
        "validation_count": len(validation),
        "preview_count": len(preview),
        "split_hash": hashlib.sha256(
            "\n".join(_normalized_relative_path(path, image_root) for path in validation).encode(
                "utf-8"
            )
        ).hexdigest(),
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_manifests(
    *,
    image_root: Path,
    output_dir: Path,
    validation_size: int,
    preview_size: int,
    seed: int,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
) -> dict[str, Path]:
    image_root = image_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not image_root.exists():
        raise FileNotFoundError(f"image_root does not exist: {image_root}")
    image_paths = collect_images(image_root, extensions)
    if not image_paths:
        raise ValueError(f"no images found under {image_root}")

    train, validation, preview = split_images(
        image_paths,
        image_root=image_root,
        validation_size=validation_size,
        preview_size=preview_size,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "train.txt",
        "validation": output_dir / "validation.txt",
        "preview": output_dir / "preview_fixed.txt",
        "metadata": output_dir / "split_metadata.json",
    }
    write_manifest(paths["train"], train)
    write_manifest(paths["validation"], validation)
    write_manifest(paths["preview"], preview)
    write_split_metadata(
        paths["metadata"],
        image_root=image_root,
        train=train,
        validation=validation,
        preview=preview,
        validation_size=validation_size,
        preview_size=preview_size,
        seed=seed,
        extensions=extensions,
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic FCMAE split manifests")
    parser.add_argument("image_root", type=Path, help="Directory containing FCMAE images")
    parser.add_argument("output_dir", type=Path, help="Directory to write manifest files")
    parser.add_argument("--validation-size", type=int, default=2000)
    parser.add_argument("--preview-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extensions", nargs="+", default=list(DEFAULT_EXTENSIONS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_manifests(
        image_root=args.image_root,
        output_dir=args.output_dir,
        validation_size=args.validation_size,
        preview_size=args.preview_size,
        seed=args.seed,
        extensions=tuple(args.extensions),
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
