from src.core.kern_utils import project_kern_to_pitches


def test_project_kern_to_pitches_user_example():
    source = (
        "*clefF4\t*clefG2\n"
        "*k[f#c#]\t*k[f#c#]\n"
        "*^\t*^\n"
        "4d\t4D 4A\t8aa(L\t4f# 4dd\n"
        ".\t.\t16ff#L\t.\n"
        ".\t.\t16dd\t.\n"
        "8e\t8D 8A\t16gg\t8g 8cc#\n"
        ".\t.\t16ee)J\t.\n"
        "=\t=\t=\t=\n"
        "4d\t4D 4A\t8aa(L\t4f# 4dd\n"
        ".\t.\t16ff#L\t.\n"
        ".\t.\t16dd\t.\n"
        "8e\t8D 8A\t16gg\t8g 8cc#\n"
        ".\t.\t16ee)J\t.\n"
        "=\t=\t=\t="
    )
    expected = (
        "d\tD A\taa\tf# dd\n"
        ".\t.\tff#\t.\n"
        ".\t.\tdd\t.\n"
        "e\tD A\tgg\tg cc#\n"
        ".\t.\tee\t.\n"
        "d\tD A\taa\tf# dd\n"
        ".\t.\tff#\t.\n"
        ".\t.\tdd\t.\n"
        "e\tD A\tgg\tg cc#\n"
        ".\t.\tee\t."
    )

    assert project_kern_to_pitches(source) == expected


def test_project_kern_to_pitches_strips_note_syntax_and_preserves_accidentals():
    source = "4f#\t[8BB-)L\t16cn]J\t4ee-"
    expected = "f#\tBB-\tcn\tee-"

    assert project_kern_to_pitches(source) == expected


def test_project_kern_to_pitches_preserves_chord_spaces_and_null_tokens():
    source = "4D 4A\t.\t8g 8cc#"
    expected = "D A\t.\tg cc#"

    assert project_kern_to_pitches(source) == expected


def test_project_kern_to_pitches_represents_rests_as_r():
    source = "4r\t8rL 16cc\t.\t[2r]"
    expected = "r\tr cc\t.\tr"

    assert project_kern_to_pitches(source) == expected


def test_project_kern_to_pitches_drops_non_data_and_empty_projected_lines():
    source = (
        "!!!COM: Example\n"
        "**kern\t**kern\n"
        "*clefG2\t*clefF4\n"
        "=1\t=1\n"
        "!LO:T:a:t=rit.\t!\n"
        "yy\t@\n"
        "*-\t*-\n"
        "4c\t4E"
    )
    expected = "c\tE"

    assert project_kern_to_pitches(source) == expected


def test_project_kern_to_pitches_preserves_empty_fields_on_retained_lines():
    source = "yy\t4c\t@\t."
    expected = "\tc\t\t."

    assert project_kern_to_pitches(source) == expected


def test_project_kern_to_pitches_empty_input_returns_empty_string():
    assert project_kern_to_pitches("") == ""
