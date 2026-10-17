# Heart Disease MLOps Assignment

An original, end-to-end reference implementation for AIMLCZG523. 

## What is included
- UCI/OpenML acquisition, cleaning, EDA plots
- Logistic Regression and Random Forest comparison with 5-fold CV
- MLflow metrics, parameters, plots and model artifacts
- FastAPI `/predict`, `/health`, `/metrics`; structured request logs
- Pytest, Ruff, GitHub Actions, GHCR image workflow
- Docker Compose with Prometheus + Grafana
- Kubernetes deployment and service manifests

## Local setup (execute in this exact order)
```bash
git clone <YOUR_REPOSITORY_URL>
cd heart-disease-mlops
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/download_data.py
python -m src.eda
mlflow ui --port 5000                  # open http://localhost:5000 in a second terminal
python -m src.train
pytest -q
ruff check src tests scripts
```
Open `artifacts/metrics.json` and `http://localhost:5000` to capture real evidence for the report.

## Run and test API locally
```bash
uvicorn src.api:app --reload --port 8000
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"age":55,"sex":1,"cp":2,"trestbps":130,"chol":240,"fbs":0,"restecg":1,"thalach":150,"exang":0,"oldpeak":1.2,"slope":1,"ca":0,"thal":2}'
curl http://localhost:8000/metrics
```

## Docker + monitoring
`best_model.joblib` must exist before a Docker build.
```bash
docker compose up --build
```
Capture: API response at `http://localhost:8000/docs`, Prometheus at `http://localhost:9090/targets`, and Grafana at `http://localhost:3000` (default `admin` / `admin`). Create a panel using `http_requests_total` and `http_request_duration_seconds`.

## GitHub + GHCR
1. Create a **private or public** GitHub repository named `heart-disease-mlops` and push this project.
2. In GitHub Actions, enable workflow permissions: **Settings -> Actions -> General -> Workflow permissions -> Read and write permissions**.
3. Push to `main`. The workflow runs lint, tests, download, training, artifact upload, image build and image push.
4. Confirm the image under the repository's **Packages** section. If private, either make it public or create an imagePullSecret for Kubernetes.

## Minikube deployment
```bash
minikube start
eval $(minikube docker-env)            # uses your locally built image, alternative to GHCR
docker build -t heart-disease-api:1.0.0 .
# For local image, change the image line in k8s/deployment.yaml to heart-disease-api:1.0.0
kubectl apply -f k8s/namespace.yaml -f k8s/deployment.yaml -f k8s/service.yaml
kubectl -n heart-mlops get pods,svc
minikube service heart-disease-api -n heart-mlops --url
kubectl -n heart-mlops scale deployment/heart-disease-api --replicas=3
```

## Final submission checklist
- [ ] GitHub repository link works; meaningful commits show your stages.
- [ ] `requirements.txt` supports clean setup; data script, EDA, training and tests have been rerun.
- [ ] Screenshots show EDA, MLflow, GitHub Actions, Docker API response, Kubernetes pods/service/scaling, Prometheus target, Grafana panel.
- [ ] Report is completed in your own words and exported to a 10-page PDF/DOCX.
- [ ] Short 3-5 minute video demonstrates the sequence: code -> MLflow -> API -> Docker -> Kubernetes -> monitoring.
- [ ] Do not submit template placeholders, invented metrics, or screenshots from another student.

## Suggested evidence folder naming
Save report images under `screenshots/` using: `01_eda.png`, `02_mlflow.png`, `03_ci.png`, `04_api.png`, `05_docker.png`, `06_k8s_pods.png`, `07_k8s_scale.png`, `08_prometheus.png`, `09_grafana.png`.
