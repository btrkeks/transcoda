from scripts.dataset_generation.dataset_generation.truncation import (
    TruncationProbeResult,
    build_canonical_prefix_candidates,
    build_prefix_truncation_candidates,
    build_prefix_truncation_space,
    find_best_truncation_candidate,
    truncate_by_chunk_count,
    validate_truncation_candidate_terminal_state,
)


def _example_kern() -> str:
    return "\n".join(
        [
            "*clefG2",
            "*M4/4",
            "=1",
            "4c",
            "=2",
            "4d",
            "=3",
            "4e",
            "=4",
            "4f",
        ]
    )


def test_truncate_by_chunk_count_returns_prefix_and_ratio():
    transcription, ratio = truncate_by_chunk_count(_example_kern(), chunk_count=3)
    assert "=4" not in transcription
    assert "4f" not in transcription
    assert transcription.strip()
    assert 0.0 < ratio < 1.0


def test_build_prefix_truncation_candidates_longest_first_and_capped():
    candidates = build_prefix_truncation_candidates(_example_kern(), max_trials=2)
    assert len(candidates) == 2
    assert candidates[0].ratio > candidates[1].ratio
    assert candidates[0].chunk_count > candidates[1].chunk_count


def test_build_prefix_truncation_candidates_returns_empty_for_single_chunk():
    candidates = build_prefix_truncation_candidates("4c\n4d\n4e", max_trials=5)
    assert candidates == []


def test_build_prefix_truncation_space_builds_chunk_aligned_candidates():
    space = build_prefix_truncation_space(_example_kern())

    candidate = space.candidate_for_chunk_count(2)

    assert candidate is not None
    assert candidate.total_chunks == space.total_chunks
    assert candidate.chunk_count == 2
    assert candidate.ratio == 2 / 5
    assert "=3" not in candidate.transcription


def test_prefix_candidate_strips_trailing_split_line():
    space = build_prefix_truncation_space("**kern\n=1\n4c\n*^\n4d\t4e\n=2\t=2")

    candidate = space.candidate_for_chunk_count(2)

    assert candidate is not None
    assert candidate.transcription.endswith("4c")
    assert "*^" not in candidate.transcription.splitlines()[-1]


def test_prefix_candidate_strips_trailing_merge_line():
    space = build_prefix_truncation_space("**kern\n=1\n4c\n*^\n4d\t4e\n*v\t*v\n4f\n=2")

    candidate = space.candidate_for_chunk_count(3)

    assert candidate is not None
    assert candidate.transcription.endswith("4d\t4e")
    assert "*v\t*v" not in candidate.transcription.splitlines()[-1]


def test_build_canonical_prefix_candidates_deduplicates_and_keeps_larger_chunk_count():
    candidates = build_canonical_prefix_candidates("**kern\n=1\n*^\n4c\t4e\n=2\t=2")

    assert [candidate.chunk_count for candidate in candidates] == [2]
    assert candidates[0].transcription == "**kern\n=1"


def test_validate_truncation_candidate_terminal_state_rejects_wrong_terminator_width():
    text = "**kern\t**kern\t**kern\n4c\t4e\t4g\n*\t*v\t*v\n*-\t*-\t*-"
    assert validate_truncation_candidate_terminal_state(text) == "invalid_terminal_spine_state"


def test_find_best_truncation_candidate_matches_exhaustive_search_for_monotone_profile():
    candidates = build_prefix_truncation_candidates(_example_kern(), max_trials=10)
    accepted_chunk_count = 3

    result = find_best_truncation_candidate(
        _example_kern(),
        max_trials=10,
        probe_candidate=lambda candidate: TruncationProbeResult(
            candidate=candidate,
            accepted=candidate.chunk_count <= accepted_chunk_count,
            rejection_reason=None if candidate.chunk_count <= accepted_chunk_count else "too_long",
            decision_reason=None,
        ),
    )

    assert result.selected_candidate is not None
    assert result.selected_candidate.chunk_count == accepted_chunk_count
    assert next(candidate.chunk_count for candidate in candidates if candidate.chunk_count <= accepted_chunk_count) == accepted_chunk_count


def test_find_best_truncation_candidate_refines_around_non_monotone_midpoint():
    accepted_chunk_counts = {2, 4}

    result = find_best_truncation_candidate(
        _example_kern(),
        max_trials=10,
        probe_candidate=lambda candidate: TruncationProbeResult(
            candidate=candidate,
            accepted=candidate.chunk_count in accepted_chunk_counts,
            rejection_reason=None if candidate.chunk_count in accepted_chunk_counts else "too_long",
            decision_reason=None,
        ),
    )

    assert result.selected_candidate is not None
    assert result.selected_candidate.chunk_count == 4


def test_find_best_truncation_candidate_reports_no_valid_candidate():
    result = find_best_truncation_candidate(
        _example_kern(),
        max_trials=10,
        probe_candidate=lambda candidate: TruncationProbeResult(
            candidate=candidate,
            accepted=False,
            rejection_reason="too_long",
            decision_reason="post_truncation_required",
        ),
    )

    assert result.selected_candidate is None
    assert result.selected_probe is None
    assert result.exhausted_budget is False
    assert result.probes


def test_find_best_truncation_candidate_honors_probe_budget():
    result = find_best_truncation_candidate(
        _example_kern(),
        max_trials=1,
        probe_candidate=lambda candidate: TruncationProbeResult(
            candidate=candidate,
            accepted=False,
            rejection_reason="too_long",
            decision_reason="post_truncation_required",
        ),
    )

    assert len(result.probes) == 1
    assert result.exhausted_budget is True
