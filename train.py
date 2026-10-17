"""Train, compare and save the spam classifier. Run: python train.py"""

import json
import sys

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

import config  # noqa: E402
from preprocess import load_dataset  # noqa: E402


def _tfidf() -> TfidfVectorizer:
    # Stop-word removal was tested with 5-fold CV on the training set and
    # lowered F1 (it removes words like "you"/"your" that are common in spam).
    return TfidfVectorizer(max_features=3000, ngram_range=(1, 2))


def build_models() -> dict:
    """Return the two candidate pipelines."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("tfidf", _tfidf()),
                (
                    "clf",
                    LogisticRegression(
                        C=10.0,
                        solver="lbfgs",
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Naive Bayes": Pipeline([("tfidf", _tfidf()), ("clf", MultinomialNB())]),
    }


def evaluate(model, x_test, y_test) -> dict:
    """Compute the evaluation metrics for one fitted model."""
    pred = model.predict(x_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def main() -> int:
    df = load_dataset()
    print(f"Dataset: {len(df)} messages | spam share: {df['label'].mean():.1%}")

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=config.TEST_SIZE,
        stratify=df["label"],
        random_state=config.RANDOM_STATE,
    )

    results, fitted = {}, {}
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        fitted[name] = model
        results[name] = evaluate(model, x_test, y_test)

    best_name = max(results, key=lambda n: results[n]["f1"])
    best = results[best_name]

    joblib.dump(fitted[best_name], config.MODEL_PATH)

    output = {
        "best_model_name": best_name,
        "dataset_size": int(len(df)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        **best,
        "all_models": results,
    }
    config.METRICS_PATH.write_text(json.dumps(output, indent=2))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=np.array(best["confusion_matrix"]),
        display_labels=["ham", "spam"],
    )
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n{'Model':<22}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}")
    for name, m in results.items():
        print(
            f"{name:<22}{m['accuracy']:>8.4f}{m['precision']:>8.4f}"
            f"{m['recall']:>8.4f}{m['f1']:>8.4f}"
        )
    print(f"\nBest model: {best_name}  (F1 = {best['f1']:.4f})")

    if best["f1"] < config.F1_THRESHOLD:
        print(f"QUALITY GATE FAILED: F1 {best['f1']:.4f} < {config.F1_THRESHOLD}")
        return 1
    print(f"Quality gate passed (F1 >= {config.F1_THRESHOLD}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
