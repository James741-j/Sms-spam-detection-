"""Text cleaning and dataset loading helpers."""

import re

import pandas as pd

from config import DATA_PATH

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_SPACES_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, remove URLs and symbols, and collapse extra spaces."""
    text = str(text).lower()
    text = _URL_RE.sub(" ", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    return _SPACES_RE.sub(" ", text).strip()


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    """Load the SMS Spam Collection CSV and return columns: label (0/1), text."""
    df = pd.read_csv(path, encoding="latin-1")
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    df = df.dropna(subset=["label", "text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df.drop_duplicates()
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].apply(clean_text)
    return df.reset_index(drop=True)
