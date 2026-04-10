from scripts.dataset_generation.dataset_generation.truncation import (
    build_prefix_truncation_space,
    build_prefix_truncation_candidates,
    truncate_by_chunk_count,
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
