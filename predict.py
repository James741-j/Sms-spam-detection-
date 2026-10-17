"""Load the trained model and predict whether a message is spam."""

from functools import lru_cache

import joblib

from config import MAX_TEXT_LENGTH, MODEL_PATH
from preprocess import clean_text


@lru_cache(maxsize=1)
def load_model():
    """Load the model once and reuse it."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python train.py"
        )
    return joblib.load(MODEL_PATH)


def validate_text(text) -> str:
    """Return the stripped text or raise ValueError for invalid input."""
    if not isinstance(text, str):
        raise ValueError("Message must be text.")
    text = text.strip()
    if not text:
        raise ValueError("Message cannot be empty.")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"Message is too long (max {MAX_TEXT_LENGTH} characters).")
    return text


def predict_message(text) -> dict:
    """Return {'label': 'spam'|'ham', 'confidence': float between 0 and 1}."""
    text = validate_text(text)
    model = load_model()
    cleaned = clean_text(text)
    proba = model.predict_proba([cleaned])[0]
    spam_prob = float(proba[1])
    if spam_prob >= 0.5:
        return {"label": "spam", "confidence": round(spam_prob, 4)}
    return {"label": "ham", "confidence": round(1 - spam_prob, 4)}
