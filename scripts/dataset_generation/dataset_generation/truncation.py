"""Helpers for truncating normalized **kern transcriptions to single-page prefixes."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.kern_utils import split_into_same_spine_nr_chunks_and_measures


@dataclass(frozen=True)
class PrefixTruncationCandidate:
    """A prefix-based truncation candidate aligned to measure/chunk boundaries."""

    transcription: str
    chunk_count: int
    total_chunks: int
    ratio: float


@dataclass(frozen=True)
class PrefixTruncationSpace:
    """Cached chunk-aligned truncation search space for one transcription."""

    chunks: tuple[str, ...]

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def candidate_for_chunk_count(
        self,
        chunk_count: int,
    ) -> PrefixTruncationCandidate | None:
        """Return a chunk-aligned prefix candidate for the requested chunk count."""
        total_chunks = self.total_chunks
        if total_chunks == 0:
            return None
        if not 1 <= chunk_count <= total_chunks:
            raise ValueError(
                f"chunk_count must be in [1, {total_chunks}], got {chunk_count}"
            )

        transcription = "".join(self.chunks[:chunk_count]).rstrip("\n")
        if not transcription.strip():
            return None
        ratio = float(chunk_count) / float(total_chunks)
        return PrefixTruncationCandidate(
            transcription=transcription,
            chunk_count=chunk_count,
            total_chunks=total_chunks,
            ratio=ratio,
        )


def build_prefix_truncation_space(kern_text: str) -> PrefixTruncationSpace:
    """Build a cached chunk-aligned truncation space for a normalized **kern file."""
    chunks = tuple(split_into_same_spine_nr_chunks_and_measures(kern_text))
    return PrefixTruncationSpace(chunks=chunks)


def truncate_by_chunk_count(kern_text: str, chunk_count: int) -> tuple[str, float]:
    """Return a chunk-aligned prefix truncation and retained-chunk ratio."""
    space = build_prefix_truncation_space(kern_text)
    candidate = space.candidate_for_chunk_count(chunk_count)
    if candidate is None:
        return "", 0.0
    return candidate.transcription, candidate.ratio


def build_prefix_truncation_candidates(
    kern_text: str,
    *,
    max_trials: int,
) -> list[PrefixTruncationCandidate]:
    """Build longest-first chunk-aligned prefix truncation candidates."""
    if max_trials < 1:
        raise ValueError(f"max_trials must be >= 1, got {max_trials}")

    space = build_prefix_truncation_space(kern_text)
    total_chunks = space.total_chunks
    if total_chunks <= 1:
        return []

    candidates: list[PrefixTruncationCandidate] = []
    seen_transcriptions: set[str] = set()

    for chunk_count in range(total_chunks - 1, 0, -1):
        candidate = space.candidate_for_chunk_count(chunk_count)
        if candidate is None:
            continue
        if candidate.transcription in seen_transcriptions:
            continue
        seen_transcriptions.add(candidate.transcription)
        candidates.append(candidate)
        if len(candidates) >= max_trials:
            break

    return candidates
