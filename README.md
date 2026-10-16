# SMS Spam Detection - A DevSecMLOps Pipeline

![CI](https://github.com/YOUR_USERNAME/sms-spam-detection/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A machine learning app that classifies an SMS message as **spam** or **not spam (ham)**.
The model (TF-IDF + Logistic Regression) is shipped through a simple, automated
**DevSecMLOps** pipeline: tests, security scans, a model quality gate and a Docker build
run on every push.

## DevSecMLOps coverage

| Stage | What is used |
|-------|--------------|
| **Dev** | Modular Python package (`src/`), pytest unit tests, flake8 + black, pre-commit hooks, Makefile |
| **Sec** | Bandit (code scan), pip-audit (dependency scan), Gitleaks (secret scan), input validation, non-root Docker user, Dependabot |
| **ML** | TF-IDF features, Logistic Regression vs Naive Bayes, evaluation (accuracy, precision, recall, F1, confusion matrix) |
| **Ops** | GitHub Actions CI, F1 quality gate, Docker image, Streamlit web app |

## Pipeline

```
Code push --> Lint --> Tests --> Security scans --> Train model --> Quality gate --> Docker build
              flake8   pytest   bandit/pip-audit    TF-IDF + LR     F1 >= 0.90      non-root image
                                gitleaks
```

## Dataset

SMS Spam Collection (UCI / Kaggle): 5,572 labelled messages in the raw file (`data/spam.csv`).
After removing duplicates, **5,158 messages** are used
(4,126 train / 1,032 test, stratified 80:20). About 12.4% are spam.

## Model

- `TfidfVectorizer(max_features=3000, ngram_range=(1, 2))`
- `LogisticRegression(C=10.0, solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=42)`
- Multinomial Naive Bayes is trained for comparison.
- Stop-word removal and `C` were chosen using 5-fold cross-validation on the **training set only**
  (the test set was not used for tuning).

## Results (test set)

| Model | Accuracy | Precision | Recall | F1 |
|-------|---------:|----------:|-------:|---:|
| **Logistic Regression** | 0.9874 | 0.9528 | 0.9453 | 0.9490 |
| Naive Bayes | 0.9748 | 0.9904 | 0.8047 | 0.8879 |

Confusion matrix (Logistic Regression): 898 true ham, 6 false spam,
7 missed spam, 121 true spam. See `reports/confusion_matrix.png`.

Metrics for spam class. These numbers come from `python -m src.train` (saved in `metrics.json`).

**Quality gate:** training exits with an error if F1 < 0.90 (`F1_THRESHOLD` in `src/config.py`).

## Folder structure

```
sms-spam-detection/
├── .github/workflows/ci.yml   # CI pipeline
├── .github/dependabot.yml
├── data/spam.csv
├── models/model.joblib        # trained model
├── reports/confusion_matrix.png
├── src/                       # config, preprocess, train, predict
├── tests/                     # pytest tests
├── app.py                     # Streamlit app
├── Dockerfile
├── Makefile
├── requirements.txt / requirements-dev.txt
├── SECURITY.md  LICENSE  README.md
```

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python -m src.train                # train + metrics + quality gate
pytest -q                          # run tests
streamlit run app.py               # open http://localhost:8501
```

### Docker

```bash
docker build -t sms-spam-detection .
docker run --rm -p 8501:8501 sms-spam-detection
```

### Security scans (local)

```bash
bandit -r src app.py
pip-audit -r requirements.txt
```

## Limitations and future work

- Trained on an English SMS dataset from 2000s-era messages; modern spam may look different.
- Spam class is only about 12% of the data, so recall on new spam styles may be lower.
- Ideas: add drift monitoring, try character n-grams, add MLflow experiment tracking.

## License

MIT - see [LICENSE](LICENSE).
