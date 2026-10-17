.PHONY: setup data eda train test lint run docker-build docker-run minikube-deploy

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
data:
	python scripts/download_data.py
eda:
	python -m src.eda
train:
	python -m src.train
test:
	pytest -q
lint:
	ruff check src tests scripts
run:
	uvicorn src.api:app --host 0.0.0.0 --port 8000
docker-build:
	docker build -t heart-disease-api:1.0.0 .
docker-run:
	docker run --rm -p 8000:8000 heart-disease-api:1.0.0
minikube-deploy:
	kubectl apply -f k8s/namespace.yaml -f k8s/deployment.yaml -f k8s/service.yaml