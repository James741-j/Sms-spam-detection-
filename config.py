"""Central configuration for the SMS spam detection project."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "spam.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"
REPORTS_DIR = BASE_DIR
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_TEXT_LENGTH = 1000
F1_THRESHOLD = 0.90  # quality gate: training fails if the best F1 is lower
