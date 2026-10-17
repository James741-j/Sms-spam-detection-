import pytest

from predict import predict_message


def test_empty_string_rejected():
    with pytest.raises(ValueError):
        predict_message("")


def test_whitespace_only_rejected():
    with pytest.raises(ValueError):
        predict_message("     ")


def test_too_long_rejected():
    with pytest.raises(ValueError):
        predict_message("a" * 1001)


def test_none_rejected():
    with pytest.raises(ValueError):
        predict_message(None)


def test_max_length_accepted():
    assert "label" in predict_message("a" * 1000)
