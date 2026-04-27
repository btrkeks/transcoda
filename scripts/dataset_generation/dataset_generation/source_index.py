"""Source indexing for normalized .krn inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from scripts.dataset_generation.dataset_generation.source_stats import compute_kern_source_stats
from scripts.dataset_generation.dataset_generation.types_domain import SourceEntry
from src.core.kern_concatenation import (
    SpineTopologyDiagnostic,
    diagnose_spine_topology,
    restore_terminal_spine_count_before_final_barline,
    summarize_spine_topology,
)


@dataclass(frozen=True)
class InvalidSourceDiagnostic:
    path: Path
    root_dir: Path
    root_label: str
    reason_code: str
    message: str


class InvalidSourceFileError(ValueError):
    def __init__(self, *, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class SourceIndex:
    root_dirs: tuple[Path, ...]
    entries: tuple[SourceEntry, ...]
    entry_idx_by_path: dict[Path, int]
    entry_indices_by_initial_spine_count: dict[int, tuple[int, ...]]
    invalid_sources: tuple[InvalidSourceDiagnostic, ...] = ()
    compatible_entry_ids_by_measure_count_cache: dict[int, dict[int, tuple[int, ...]]] = field(
        default_factory=dict
    )
    entry_indices_by_spine_class_cache: dict[str, tuple[int, ...]] = field(default_factory=dict)


def build_source_index(*input_dirs: str | Path) -> SourceIndex:
    if not input_dirs:
        raise ValueError("build_source_index requires at least one input directory")

    root_dirs = tuple(Path(input_dir).expanduser().resolve() for input_dir in input_dirs)
    root_labels = _build_root_labels(root_dirs)
    entries: list[SourceEntry] = []
    entry_idx_by_path: dict[Path, int] = {}
    invalid_sources: list[InvalidSourceDiagnostic] = []
    next_entry_idx = 0
    for root_dir, root_label in zip(root_dirs, root_labels, strict=True):
        if not root_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {root_dir}")
        if not root_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {root_dir}")

        paths = sorted(path for path in root_dir.rglob("*.krn") if path.is_file())
        if not paths:
            raise ValueError(f"No .krn files found under {root_dir}")

        for path in paths:
            rel = path.relative_to(root_dir)
            source_id = (Path(root_label) / rel.with_suffix("")).as_posix()
            try:
                stats = compute_kern_source_stats(path)
                (
                    initial_spine_count,
                    terminal_spine_count,
                    restored_terminal_spine_count,
                ) = _compute_boundary_spine_counts(path)
                entries.append(
                    SourceEntry(
                        entry_idx=next_entry_idx,
                        path=path,
                        source_id=source_id,
                        root_dir=root_dir,
                        root_label=root_label,
                        measure_count=stats.measure_count,
                        non_empty_line_count=stats.non_empty_line_count,
                        has_header=_has_explicit_header(path),
                        initial_spine_count=initial_spine_count,
                        terminal_spine_count=terminal_spine_count,
                        restored_terminal_spine_count=restored_terminal_spine_count,
                    )
                )
                entry_idx_by_path[path.resolve()] = next_entry_idx
                next_entry_idx += 1
            except InvalidSourceFileError as exc:
                invalid_sources.append(
                    InvalidSourceDiagnostic(
                        path=path,
                        root_dir=root_dir,
                        root_label=root_label,
                        reason_code=exc.reason_code,
                        message=str(exc),
                    )
                )
    entry_indices_by_initial_spine_count: dict[int, tuple[int, ...]] = {
        spine_count: tuple(entry.entry_idx for entry in grouped_entries)
        for spine_count, grouped_entries in _group_entries_by_initial_spine_count(entries).items()
    }
    return SourceIndex(
        root_dirs=root_dirs,
        entries=tuple(entries),
        entry_idx_by_path=entry_idx_by_path,
        entry_indices_by_initial_spine_count=entry_indices_by_initial_spine_count,
        invalid_sources=tuple(invalid_sources),
    )


def _group_entries_by_initial_spine_count(
    entries: list[SourceEntry],
) -> dict[int, list[SourceEntry]]:
    grouped: dict[int, list[SourceEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.initial_spine_count, []).append(entry)
    return grouped


def _has_explicit_header(path: Path) -> bool:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("!!"):
                continue
            return line.startswith("**")
    return False


def _compute_boundary_spine_counts(path: Path) -> tuple[int, int, int]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    original_topology = summarize_spine_topology(text)
    if (
        original_topology.initial_spine_count is None
        or original_topology.terminal_spine_count is None
    ):
        diagnostic = diagnose_spine_topology(text)
        raise InvalidSourceFileError(
            reason_code=diagnostic.reason_code if diagnostic is not None else "invalid_source",
            message=_format_topology_failure(
                path=path,
                diagnostic=diagnostic,
                prefix="Cannot infer spine counts",
            ),
        )

    restored_text = restore_terminal_spine_count_before_final_barline(text)
    restored_topology = summarize_spine_topology(restored_text)
    if restored_topology.terminal_spine_count is None:
        diagnostic = diagnose_spine_topology(restored_text)
        raise InvalidSourceFileError(
            reason_code=(
                diagnostic.reason_code if diagnostic is not None else "invalid_restored_terminal_spine_count"
            ),
            message=_format_topology_failure(
                path=path,
                diagnostic=diagnostic,
                prefix="Cannot infer restored terminal spine count",
            ),
        )

    return (
        original_topology.initial_spine_count,
        original_topology.terminal_spine_count,
        restored_topology.terminal_spine_count,
    )


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


def _format_topology_failure(
    *,
    path: Path,
    diagnostic: SpineTopologyDiagnostic | None,
    prefix: str,
) -> str:
    if diagnostic is None:
        return f"{prefix} from source file: {path}"

    detail = diagnostic.message
    if diagnostic.line_text is not None:
        detail = f"{detail}; line={diagnostic.line_text!r}"
    return f"{prefix} from source file: {path} ({diagnostic.reason_code}: {detail})"
