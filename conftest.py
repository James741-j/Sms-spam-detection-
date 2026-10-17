"""Shared fixtures: make sure a trained model exists before tests run."""

import pytest

from config import MODEL_PATH


@pytest.fixture(scope="session", autouse=True)
def ensure_model():
    if not MODEL_PATH.exists():
        import train

        train.main()
