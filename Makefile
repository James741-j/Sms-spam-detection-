.PHONY: install train test scan run docker-build docker-run

install:
	pip install -r requirements-dev.txt

train:
	python -m src.train

test:
	pytest -q

scan:
	bandit -r src app.py
	pip-audit -r requirements.txt

run:
	streamlit run app.py

docker-build:
	docker build -t sms-spam-detection .

docker-run:
	docker run --rm -p 8501:8501 sms-spam-detection
