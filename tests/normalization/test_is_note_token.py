from src.core.kern_utils import is_note_token


def test_comment_is_not_note_token():
    comment = "!LO:T:a:t=ritentuo"
    result = is_note_token(comment)
    assert result is False
