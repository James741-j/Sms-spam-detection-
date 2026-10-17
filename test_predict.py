from predict import predict_message


def test_obvious_spam():
    msg = "WINNER!! Claim your free prize now, call 09061701461 to collect your cash reward"
    assert predict_message(msg)["label"] == "spam"


def test_normal_message():
    assert predict_message("Are we meeting at 5 today?")["label"] == "ham"


def test_output_format():
    result = predict_message("Hello, how are you?")
    assert set(result) == {"label", "confidence"}
    assert result["label"] in {"spam", "ham"}
    assert 0 <= result["confidence"] <= 1
