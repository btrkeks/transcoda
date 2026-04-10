import pytest

from scripts.dataset_generation.normalization.presets import (
    normalize_kern_transcription,
    normalize_kern_transcription_with_context,
)
from src.core.kern_concatenation import diagnose_spine_topology
from src.core.spine_state import InvalidSpineOperationError


def test_standard_pipeline_on_original_28():
    transcription = """*clefF4\t*clefG2
*k[b-e-a-d-]\t*k[b-e-a-d-]
*M6/8\t*M6/8
4.AA--[ 4.AAA--[\t4.a--[ 4.A--[
4AA--] 4AAA--]\t4a--] 4A--]
8AA-- 8AAA--\t8a-- 8A--
=\t=
4GG- 4GGG-\t4g- 4G-
8r\t8r
4.r\t4.r
=\t=
8FFL\t8ee--L 8cc-L 8a-L 8fL
8AA-\t8ee-- 8cc- 8a- 8f
8C-J\t8ee--J 8cc-J 8a-J 8fJ
8E--L\t8ee--L 8cc-L 8a-L 8fL
8F\t8ee-- 8cc- 8a- 8f
8A-J\t8ee--J 8cc-J 8a-J 8fJ
=\t=
8G-L\t8dd-L 8b--L 8g-L
8C\t8dd- 8b-- 8g-
8D-J\t8dd-J 8b--J 8g-J
8BB--L\t8dd-L 8b--L 8g-L
8GG-\t8dd- 8b-- 8g-
8DD-J\t8dd-J 8b--J 8g-J
=\t=
8FFL\t8ee--L 8cc-L 8a-L 8fL
8AA-\t8ee-- 8cc- 8a- 8f
8C-J\t8ee--J 8cc-J 8a-J 8fJ
8E--L\t8ee--L 8cc-L 8a-L 8fL
8F\t8ee-- 8cc- 8a- 8f
8A-J\t8ee--J 8cc-J 8a-J 8fJ
=\t=
*-\t*-
"""

    normalized = normalize_kern_transcription(transcription)

    expected_normalized = """*clefF4\t*clefG2
*k[b-e-a-d-]\t*k[b-e-a-d-]
*M6/8\t*M6/8
4.AAA--[ 4.AA--[\t4.A--[ 4.a--[
4AAA--] 4AA--]\t4A--] 4a--]
8AAA-- 8AA--\t8A-- 8a--
=\t=
4GGG- 4GG-\t4G- 4g-
8r\t8r
4.r\t4.r
=\t=
8FFL\t8f 8a- 8cc- 8ee--L
8AA-\t8f 8a- 8cc- 8ee--
8C-J\t8f 8a- 8cc- 8ee--J
8E--L\t8f 8a- 8cc- 8ee--L
8F\t8f 8a- 8cc- 8ee--
8A-J\t8f 8a- 8cc- 8ee--J
=\t=
8G-L\t8g- 8b-- 8dd-L
8C\t8g- 8b-- 8dd-
8D-J\t8g- 8b-- 8dd-J
8BB--L\t8g- 8b-- 8dd-L
8GG-\t8g- 8b-- 8dd-
8DD-J\t8g- 8b-- 8dd-J
=\t=
8FFL\t8f 8a- 8cc- 8ee--L
8AA-\t8f 8a- 8cc- 8ee--
8C-J\t8f 8a- 8cc- 8ee--J
8E--L\t8f 8a- 8cc- 8ee--L
8F\t8f 8a- 8cc- 8ee--
8A-J\t8f 8a- 8cc- 8ee--J
=\t=
""".strip()

    assert normalized == expected_normalized


def test_standard_pipeline_collapses_repeated_ties():
    transcription = """*clefF4
*k[]
*M4/4
4B[[
2a__
4f]]
4g((
="""

    normalized = normalize_kern_transcription(transcription)

    expected_normalized = """*clefF4
*k[]
*M4/4
4B[
2a_
4f]
4g((
=""".strip()

    assert normalized == expected_normalized


def test_standard_pipeline_rejects_orphan_merge():
    transcription = """*clefF4\t*clefG2
*k[]\t*k[]
*M4/4\t*M4/4
*\t*v
4c\t4e
"""

    with pytest.raises(InvalidSpineOperationError, match=r"Line 4: .*Invalid merge operation"):
        normalize_kern_transcription(transcription)


def test_standard_pipeline_accepts_consecutive_valid_manipulator_rows():
    transcription = """*clefF4\t*clefG2
*k[]\t*k[]
*M4/4\t*M4/4
*^\t*
*\t*^\t*
4c\t4e\t4g\t4b
*v\t*v\t*\t*
*\t*v\t*v
4c\t4g
=\t=
*-\t*-
"""

    normalized = normalize_kern_transcription(transcription)
    assert "4c\t4g" in normalized


def test_standard_pipeline_preserves_valid_triple_merge_excerpt_from_grandstaff_003867():
    transcription = """**kern\t**kern
*clefF4\t*clefG2
*k[b-e-a-d-g-]\t*k[b-e-a-d-g-]
*M2/4\t*M2/4
*^\t*
*\t*^\t*
8d-L\t2d-\t2BB-\t16.fLL
.\t.\t.\t32eJJk
8E\t.\t.\t4.g
8G\t.\t.\t.
8EJ\t.\t.\t.
*\t*v\t*v\t*
=\t=\t=
*\t*^\t*
8d-L\t2d-\t2BB-\t16.fLL
.\t.\t.\t32eJJk
8E\t.\t.\t4g
8G\t.\t.\t.
8EJ\t.\t.\t16.gLL
.\t.\t.\t32dd-JJk
*v\t*v\t*v\t*
=\t=
*\t*^
8r\t8dd-L\t8r
8AA 8F\t16.ccL\t8c
.\t32fJJk\t.
"""

    normalized, _ = normalize_kern_transcription_with_context(transcription)

    assert diagnose_spine_topology(normalized) is None
    assert "*v\t*v\t*v\t*" in normalized
    assert "=\t=\n*\t*^" in normalized
