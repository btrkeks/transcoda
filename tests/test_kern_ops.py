"""Tests for kern_ops functions."""

import pytest

from scripts.dataset_generation.dataset_generation.image_generation.kern_ops import (
    is_spinemerge_line,
    is_spinesplit_line,
    split_into_same_splite_nr_chunks_and_measures,
)


def test_is_spinesplit_line_basic():
    """Test detection of basic spine split lines."""
    assert is_spinesplit_line("*\t*^") is True
    assert is_spinesplit_line("*\t*^\t*") is True
    assert is_spinesplit_line("*^") is False  # Single column
    assert is_spinesplit_line("*clefF4\t*clefG2") is False  # Not spine split


def test_is_spinemerge_line_basic():
    """Test detection of basic spine merge lines."""
    assert is_spinemerge_line("*v\t*v") is True
    assert is_spinemerge_line("*v\t*v\t*") is True
    assert is_spinemerge_line("*\t*v\t*v\t*") is True


def test_split_into_same_splite_nr_chunks_simple():
    """Test splitting simple kern without spine manipulations."""
    krn = """**kern	**kern
*clefF4	*clefG2
*k[]	*k[]
*M4/4	*M4/4
4c	4cc
4d	4dd
=	=
4e	4ee
4f	4ff
=	=
*-	*-"""

    chunks = split_into_same_splite_nr_chunks_and_measures(krn)

    # Should split at each bar line
    assert len(chunks) >= 3


def test_split_into_same_splite_nr_chunks_with_spine_split():
    """Test splitting with spine split - this should expose Bug #2."""
    # Minimal example from Ballade file
    krn = """**kern	**kern	**dynam
*clefF4	*clefG2	*
*k[b-e-]	*k[b-e-]	*
*M6/4	*M6/4	*
*	*^	*
2.CC 2.C	2.a	4e- 4g	.
.	.	4e- 4g	.
=	=	=	=
*-	*-	*-	*-"""

    # This should not raise AssertionError
    chunks = split_into_same_splite_nr_chunks_and_measures(krn)

    # Verify chunks were created
    assert len(chunks) > 0


def test_split_with_empty_lines():
    """Test that empty lines cause the bug - Bug #2 root cause."""
    krn = """**kern	**kern
*clefF4	*clefG2
*k[]	*k[]
*M4/4	*M4/4
4c	4cc
4d	4dd

4e	4ee
=	=
*-	*-"""

    # This should fail because of the empty line
    with pytest.raises(AssertionError, match="Inconsistent number of spines"):
        split_into_same_splite_nr_chunks_and_measures(krn)


def test_split_with_spine_split_debug():
    """Debug test to see what chunks are created and where the error occurs."""
    krn = """**kern	**kern	**dynam
*clefF4	*clefG2	*
*k[b-e-]	*k[b-e-]	*
*M6/4	*M6/4	*
*	*^	*
2.CC 2.C	2.a	4e- 4g	.
.	.	4e- 4g	.
=	=	=	=
*-	*-	*-	*-"""

    # Manually do what the function does to see where it breaks
    result: list[str] = []
    current_chunk: list[str] = []

    for line in krn.splitlines():
        current_chunk.append(line)

        if is_spinesplit_line(line) or is_spinemerge_line(line):
            chunk_str = "\n".join(current_chunk) + "\n"
            result.append(chunk_str)
            print(f"Chunk after spine split/merge:\n{chunk_str}")
            current_chunk = []

    if current_chunk:
        chunk_str = "\n".join(current_chunk).rstrip()
        result.append(chunk_str)
        print(f"Final chunk:\n{chunk_str}")

    # Check each chunk for consistency
    for i, part in enumerate(result):
        lines = part.splitlines()
        if not lines:
            continue

        print(f"\nChunk {i}:")
        for line in lines:
            tab_count = line.count("\t")
            print(f"  Tabs={tab_count}: {line}")

        num_tabs = lines[0].count("\t")
        inconsistent_lines = [line for line in lines if line.count("\t") != num_tabs]

        if inconsistent_lines:
            print("  ERROR: Inconsistent tab counts!")
            print(f"  Expected {num_tabs} tabs, but found:")
            for line in inconsistent_lines:
                tab_count = line.count("\t")
                print(f"    {tab_count} tabs: {line}")
            pytest.fail(f"Inconsistent spines in chunk {i}")
