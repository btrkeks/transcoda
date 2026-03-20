from scripts.dataset_generation.normalization.presets import normalize_kern_transcription


def test_versioned_kern_header_is_removed():
    transcription = """**kern_1.0\t**kern_1.0
*clefF4\t*clefG2
*k[]\t*k[]
*M4/4\t*M4/4
4C\t4c
*-\t*-
"""

    normalized = normalize_kern_transcription(transcription)

    expected = """*clefF4\t*clefG2
*k[]\t*k[]
*M4/4\t*M4/4
4C\t4c"""

    assert normalized == expected


def test_versioned_kern_spines_keep_only_kern_columns():
    transcription = """**kern_1.0\t**text\t**kern_1.0
*clefF4\t*Ilyrics\t*clefG2
*k[]\t*\t*k[]
*M4/4\t*\t*M4/4
4C\tla\t4c
*-\t*-\t*-
"""

    normalized = normalize_kern_transcription(transcription)

    expected = """*clefF4\t*clefG2
*k[]\t*k[]
*M4/4\t*M4/4
4C\t4c"""

    assert normalized == expected
