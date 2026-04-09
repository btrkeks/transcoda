"""Source indexing for normalized .krn inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.dataset_generation.dataset_generation.source_stats import compute_kern_source_stats
from scripts.dataset_generation.dataset_generation_new.types import SourceEntry
from src.core.kern_utils import is_spinemerge_line, is_spinesplit_line


@dataclass(frozen=True)
class SourceIndex:
    root_dirs: tuple[Path, ...]
    entries: tuple[SourceEntry, ...]


def build_source_index(*input_dirs: str | Path) -> SourceIndex:
    if not input_dirs:
        raise ValueError("build_source_index requires at least one input directory")

    root_dirs = tuple(Path(input_dir).expanduser().resolve() for input_dir in input_dirs)
    root_labels = _build_root_labels(root_dirs)
    entries: list[SourceEntry] = []
    for root_dir, root_label in zip(root_dirs, root_labels, strict=True):
        if not root_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {root_dir}")
        if not root_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {root_dir}")

        paths = sorted(path for path in root_dir.rglob("*.krn") if path.is_file())
        if not paths:
            raise ValueError(f"No .krn files found under {root_dir}")

        for path in paths:
            stats = compute_kern_source_stats(path)
            rel = path.relative_to(root_dir)
            source_id = (Path(root_label) / rel.with_suffix("")).as_posix()
            initial_spine_count, terminal_spine_count = _compute_boundary_spine_counts(path)
            entries.append(
                SourceEntry(
                    path=path,
                    source_id=source_id,
                    root_dir=root_dir,
                    root_label=root_label,
                    measure_count=stats.measure_count,
                    non_empty_line_count=stats.non_empty_line_count,
                    has_header=_has_explicit_header(path),
                    initial_spine_count=initial_spine_count,
                    terminal_spine_count=terminal_spine_count,
                )
            )
    return SourceIndex(root_dirs=root_dirs, entries=tuple(entries))


def _has_explicit_header(path: Path) -> bool:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("!!"):
                continue
            return line.startswith("**")
    return False


def _compute_boundary_spine_counts(path: Path) -> tuple[int, int]:
    initial_spine_count: int | None = None
    terminal_spine_count: int | None = None
    current_spine_count: int | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("!!"):
                continue

            spine_count = line.count("\t") + 1
            if initial_spine_count is None:
                initial_spine_count = spine_count
            if current_spine_count is None:
                current_spine_count = spine_count

            if all(token == "*-" for token in line.split("\t")):
                continue

            if is_spinesplit_line(line):
                current_spine_count += sum(1 for token in line.split("\t") if token == "*^")
            elif is_spinemerge_line(line):
                current_spine_count = _apply_merge_count(current_spine_count, line.split("\t"))
            else:
                current_spine_count = spine_count

            terminal_spine_count = current_spine_count

    if initial_spine_count is None:
        raise ValueError(f"Cannot infer spine counts from empty source file: {path}")
    if terminal_spine_count is None:
        terminal_spine_count = initial_spine_count
    return initial_spine_count, terminal_spine_count


def _apply_merge_count(current_spine_count: int, tokens: list[str]) -> int:
    merge_groups = 0
    idx = 0
    while idx < len(tokens):
        if tokens[idx] != "*v":
            idx += 1
            continue
        merge_groups += 1
        while idx < len(tokens) and tokens[idx] == "*v":
            idx += 1
    merge_token_count = sum(1 for token in tokens if token == "*v")
    return current_spine_count - merge_token_count + merge_groups


def _build_root_labels(root_dirs: tuple[Path, ...]) -> tuple[str, ...]:
    if len(root_dirs) == 1:
        return (root_dirs[0].name,)

    parts_by_root = [root_dir.parts for root_dir in root_dirs]
    suffix_len = 1
    while True:
        labels = []
        for parts in parts_by_root:
            start_idx = max(0, len(parts) - suffix_len)
            labels.append(Path(*parts[start_idx:]).as_posix())
        if len(set(labels)) == len(labels):
            return tuple(labels)
        suffix_len += 1
