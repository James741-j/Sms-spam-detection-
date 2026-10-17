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
| **Dev** | Modular Python modules, pytest unit tests, flake8 + black, pre-commit hooks, Makefile |
| **Sec** | Bandit (code scan), pip-audit (dependency scan), Gitleaks (secret scan), input validation, non-root Docker user |
| **ML** | TF-IDF features, Logistic Regression vs Naive Bayes, evaluation (accuracy, precision, recall, F1, confusion matrix) |
| **Ops** | GitHub Actions CI, F1 quality gate, Docker image, Streamlit web app |

## Pipeline

```
Code push --> Lint --> Tests --> Security scans --> Train model --> Quality gate --> Docker build
              flake8   pytest   bandit/pip-audit    TF-IDF + LR     F1 >= 0.90      non-root image
                                gitleaks
```

## Dataset

SMS Spam Collection (UCI / Kaggle): 5,572 labelled messages in the raw file (`spam.csv`).
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
7 missed spam, 121 true spam. See `confusion_matrix.png`.

Metrics for spam class. These numbers come from `python train.py` (saved in `metrics.json`).

**Quality gate:** training exits with an error if F1 < 0.90 (`F1_THRESHOLD` in `config.py`).

## Folder structure

All project files are in one folder (no subfolders), except the CI workflow, which GitHub
requires to be in `.github/workflows/`.

```
sms-spam-detection/
├── .github/workflows/ci.yml   # CI pipeline (must stay in this path)
├── app.py                     # Streamlit app
├── config.py                  # settings and quality-gate threshold
├── preprocess.py              # text cleaning and dataset loading
├── train.py                   # training, evaluation, quality gate
├── predict.py                 # input validation and prediction
├── conftest.py                # shared pytest fixture
├── test_preprocess.py  test_predict.py  test_validation.py
├── spam.csv                   # dataset
├── model.joblib               # trained model
├── metrics.json               # results of the last training run
├── confusion_matrix.png
├── Dockerfile  .dockerignore  Makefile
├── requirements.txt  requirements-dev.txt
├── .gitignore  .env.example  .flake8  .pre-commit-config.yaml
└── SECURITY.md  LICENSE  README.md
```

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python train.py                # train + metrics + quality gate
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
bandit config.py preprocess.py train.py predict.py app.py
pip-audit -r requirements.txt
```

## Limitations and future work

- Trained on an English SMS dataset from 2000s-era messages; modern spam may look different.
- Spam class is only about 12% of the data, so recall on new spam styles may be lower.
- Ideas: add drift monitoring, try character n-grams, add MLflow experiment tracking.

## License

MIT - see [LICENSE](LICENSE).
