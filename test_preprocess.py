from preprocess import clean_text


def test_lowercase():
    assert clean_text("HELLO World") == "hello world"


def test_url_removed():
    assert "http" not in clean_text("Visit http://spam.com now")
    assert "www" not in clean_text("go to www.spam.com today")


def test_symbols_removed():
    assert clean_text("Win!!! $$$ now???") == "win now"


def test_extra_spaces_collapsed():
    assert clean_text("a    b \n c") == "a b c"


def test_empty_string():
    assert clean_text("") == ""
