# Capstone-Project-MLOps-02

An End-to-End MLOps Capstone Project (for educational purposes): an IMDB movie-review **Sentiment Analysis** service, taken all the way from raw data to a monitored, auto-scaled production deployment on Kubernetes.

The pipeline ingests review text, cleans it, vectorizes it with Bag-of-Words, trains a Logistic Regression classifier, evaluates and registers it in an MLflow Model Registry, serves it through a Flask app, and ships it via a fully automated CI/CD pipeline to AWS EKS — with Prometheus + Grafana monitoring the live service.

## Tech Stack

| Concern | Tool |
|---|---|
| Project scaffolding | Cookiecutter Data Science |
| Data versioning | DVC (local remote → AWS S3 remote) |
| Pipeline orchestration | DVC pipeline (`dvc.yaml` / `dvc repro`) |
| Experiment tracking & model registry | MLflow, hosted on DagsHub |
| Modeling | scikit-learn (`CountVectorizer` + `LogisticRegression`) |
| Serving | Flask + Gunicorn |
| Dependency management | `pip`, `pipreqs` (for the slim app-only requirements) |
| CI/CD | GitHub Actions |
| Containerization | Docker → Amazon ECR |
| Orchestration | Kubernetes on AWS EKS (`eksctl`, `kubectl`) |
| Monitoring | `prometheus_client` (app metrics) → Prometheus (EC2) → Grafana (EC2) |
| Cloud storage/creds | AWS S3, IAM, boto3 |

## Repository Structure

```
├── data/
│   ├── raw/                 # train.csv, test.csv (post train/test split)
│   ├── interim/              # cleaned text (train_processed.csv, test_processed.csv)
│   ├── processed/             # BoW features (train_bow.csv, test_bow.csv)
│   └── external/              # placeholder for external data sources
├── models/
│   ├── model.pkl              # trained LogisticRegression classifier
│   └── vectorizer.pkl          # fitted CountVectorizer
├── notebooks/                  # exploratory experiment notebooks (exp1.ipynb, exp2_bow_vs_tfidf.py, exp3_lor_bow_hp.py)
├── reports/
│   ├── metrics.json            # accuracy / precision / recall / auc of the last run
│   ├── experiment_info.json    # MLflow run_id + model_path, feeds registration
│   └── figures/
├── src/
│   ├── logger/                 # rotating file + console logger, shared across the pipeline
│   ├── connections/
│   │   ├── s3_connection.py    # boto3 wrapper to pull raw CSVs from S3
│   │   └── ssms_connection.py  # SQL Server connection helper
│   ├── data/
│   │   ├── data_ingestion.py   # downloads source data, splits train/test, writes data/raw
│   │   └── data_preprocessing.py  # text cleaning (lowercase, stopwords, punctuation, lemmatization)
│   ├── features/
│   │   └── feature_engineering.py # fits CountVectorizer (Bag-of-Words), writes models/vectorizer.pkl
│   ├── model/
│   │   ├── model_building.py   # trains LogisticRegression, writes models/model.pkl
│   │   ├── model_evaluation.py # scores the model, logs metrics/params/model to MLflow
│   │   ├── register_model.py   # registers the MLflow run as a Model Registry version ("Staging")
│   │   ├── predict_model.py
│   │   └── train_model.py
│   └── visualization/
│       └── visualize.py
├── flask_app/
│   ├── app.py                  # Flask API: "/" (UI), "/predict", "/metrics" (Prometheus)
│   ├── preprocessing_utility.py
│   ├── load_model_test.py
│   ├── templates/index.html
│   └── requirements.txt        # slim, app-only deps (generated with pipreqs)
├── scripts/
│   └── promote_model.py        # promotes the latest "Staging" model version to "Production"
├── tests/
│   ├── test_model.py           # model load / signature / performance thresholds
│   └── test_flask_app.py       # Flask route tests
├── .github/workflows/ci.yaml   # full CI/CD pipeline (see below)
├── dvc.yaml                    # DVC pipeline stage graph
├── dvc.lock                    # DVC-tracked stage/output hashes
├── params.yaml                 # pipeline hyperparameters
├── Dockerfile                  # builds the Flask serving image
├── deployment.yaml             # Kubernetes Deployment + LoadBalancer Service
├── requirements.txt            # full dev/pipeline dependencies
├── setup.py / tox.ini / Makefile / test_environment.py   # cookiecutter-ds scaffolding
├── docs/                       # Sphinx docs scaffold
└── projectflow.txt             # step-by-step build log this README is generated from
```

## DVC Pipeline (`dvc.yaml`)

Five DVC stages, wired together as a DAG, each with tracked deps/outs:

1. **`data_ingestion`** — `src/data/data_ingestion.py`
   Downloads the raw dataset (currently a public CSV URL; an S3-backed loader via `src/connections/s3_connection.py` is stubbed in for swapping in private data), keeps only `positive`/`negative` sentiment rows, maps them to `1`/`0`, and splits into train/test using `data_ingestion.test_size` from `params.yaml`. Outputs → `data/raw/`.
2. **`data_preprocessing`** — `src/data/data_preprocessing.py`
   Cleans the `review` text column: strips URLs/numbers/punctuation, lowercases, removes stopwords, lemmatizes (NLTK). Outputs → `data/interim/`.
3. **`feature_engineering`** — `src/features/feature_engineering.py`
   Fits a `CountVectorizer` (Bag-of-Words) with `feature_engineering.max_features` features from `params.yaml`, transforms train/test, and persists the vectorizer. Outputs → `data/processed/`, `models/vectorizer.pkl`.
4. **`model_building`** — `src/model/model_building.py`
   Trains a `LogisticRegression(C=1, solver='liblinear', penalty='l2')` on the BoW features. Outputs → `models/model.pkl`.
5. **`model_evaluation`** — `src/model/model_evaluation.py`
   Scores accuracy/precision/recall/AUC on the test set, logs metrics + model params + the model itself to MLflow (via DagsHub tracking), and writes the run's `run_id`/`model_path` to `reports/experiment_info.json`. Outputs (metrics) → `reports/metrics.json`.
6. **`model_registration`** — `src/model/register_model.py`
   Reads `reports/experiment_info.json` and registers that MLflow run as a new version of the `my_model` registry entry, transitioning it straight to the **Staging** stage.

**Current tracked parameters** (`params.yaml`):
```yaml
data_ingestion:
  test_size: 0.25
feature_engineering:
  max_features: 70
```

**Latest tracked metrics** (`reports/metrics.json`):
```json
{
  "accuracy": 0.64,
  "precision": 0.674,
  "recall": 0.483,
  "auc": 0.727
}
```

Run the whole pipeline with:
```bash
dvc repro       # (re)runs any stage whose deps/params changed
dvc status      # check what's out of sync
dvc push        # push data/model artifacts to the configured remote
```

## Experiment Tracking (MLflow on DagsHub)

- Tracking URI: `https://dagshub.com/1.arpanchandra/Capstone-Project-MLOps-02.mlflow`
- Every `model_evaluation` run creates an MLflow run under the `my-dvc-pipeline` experiment, logging params, metrics, the sklearn model, and the metrics artifact.
- Authentication is via a DagsHub token stored in the `CAPSTONE_TEST` environment variable (loaded from `.env` locally, or injected as a GitHub Actions secret / Kubernetes secret in CI/CD — never committed to the repo).
- Model lifecycle: `register_model.py` registers a new version into **Staging**; `scripts/promote_model.py` archives the current **Production** version and promotes the latest **Staging** version to **Production**. The Flask app always serves whichever version is currently in `Production` (falling back to `None` stage if nothing is promoted yet).

## Data & Model Versioning (DVC)

- `dvc init` initializes DVC tracking for `data/`, `models/`, and pipeline outputs (all gitignored — only their `.dvc`/`dvc.lock` pointers are versioned).
- Two DVC remotes were configured over the project's life:
  - `mylocal` → a local folder (`local_s3/`), used for the initial DVC setup/testing.
  - `myremote` → an AWS S3 bucket (`s3://cosplaints-datagrid-01`), the production remote, accessed via an IAM user and `dvc[s3]`.
- Standard flow: `dvc status` → `dvc commit` → `dvc push` (data) alongside `git add/commit/push` (code + `.dvc` pointer files).

## Serving — Flask App (`flask_app/`)

`app.py` exposes three routes:
- **`GET /`** — renders `templates/index.html` (the sentiment-analysis UI).
- **`POST /predict`** — takes free-text input, applies the same normalization pipeline used in training (lowercase → stopword removal → digit/punctuation/URL stripping → lemmatization), vectorizes it with the persisted `vectorizer.pkl`, and returns the model's prediction (`Positive`/`Negative`).
- **`GET /metrics`** — exposes Prometheus-format metrics via `prometheus_client`:
  - `app_request_count{method, endpoint}` — request counter
  - `app_request_latency_seconds{endpoint}` — request latency histogram
  - `model_prediction_count{prediction}` — prediction-class counter

On startup, the app connects to the DagsHub-hosted MLflow registry, fetches the latest `Production` (or `None`-stage) version of `my_model`, and loads it via `mlflow.pyfunc.load_model`.

Dependencies for the container are kept minimal (`flask_app/requirements.txt`), generated with `pipreqs . --force` to avoid shipping the full pipeline/dev dependency set into production.

## Containerization (Docker)

`Dockerfile` (built from the repo root):
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY flask_app/ /app/
COPY models/vectorizer.pkl /app/models/vectorizer.pkl
RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords wordnet
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
```
- Only `flask_app/` and the vectorizer are baked into the image — the trained classifier itself is loaded at runtime from the MLflow registry, not copied into the image.
- Local build/run:
  ```bash
  docker build -t capstone-app:latest .
  docker run -p 8888:5000 -e CAPSTONE_TEST=<your-dagshub-token> capstone-app:latest
  ```

## CI/CD Pipeline (`.github/workflows/ci.yaml`)

Triggered on every `push`. Job **`project-testing`** runs on `ubuntu-latest`:

1. Checkout code, set up Python 3.10, cache pip deps.
2. `pip install -r requirements.txt`.
3. `dvc repro` — re-executes the DVC pipeline end to end (ingestion → registration).
4. `python -m unittest tests/test_model.py` — validates the newly-registered model loads correctly, has the expected input/output signature, and clears minimum performance thresholds (accuracy/precision/recall/F1 ≥ 0.40) against holdout data.
5. `python scripts/promote_model.py` — on success, promotes the model from **Staging** to **Production** in the MLflow registry.
6. `python -m unittest tests/test_flask_app.py` — smoke-tests the Flask routes (`/` renders the UI, `/predict` returns a Positive/Negative result).
7. **AWS ECR login** — authenticates Docker against `<AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com` using AWS secrets.
8. **Build → tag → push** the Docker image to ECR (`ECR_REPOSITORY:latest`).
9. **`kubectl` setup** (`azure/setup-kubectl@v3`) and `aws eks update-kubeconfig --region us-east-1 --name flask-app-cluster` to target the EKS cluster.
10. **Create/refresh a Kubernetes Secret** (`capstone-secret`) holding `CAPSTONE_TEST`, applied idempotently via `--dry-run=client -o yaml | kubectl apply -f -`.
11. **Deploy** — `kubectl apply -f deployment.yaml`, rolling out the new image to the cluster.

Required GitHub Actions secrets: `CAPSTONE_TEST` (DagsHub token), `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_ACCOUNT_ID`, `ECR_REPOSITORY`.

## Kubernetes Deployment (`deployment.yaml`)

- **Deployment** `flask-app` — 2 replicas, image `<AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/flask-app:latest`, container port `5000`.
  - Resource requests: `256Mi` memory / `250m` CPU. Limits: `512Mi` memory / `1` CPU.
  - `CAPSTONE_TEST` injected from the `capstone-secret` Kubernetes Secret (never hardcoded).
- **Service** `flask-app-service` — type `LoadBalancer`, forwards port `5000` → `5000`, fronting the two pods with an AWS ELB.

### EKS cluster provisioning
```bash
eksctl create cluster --name flask-app-cluster --region us-east-1 --version 1.36 \
  --nodegroup-name flask-app-nodes --node-type t3.small \
  --nodes 1 --nodes-min 1 --nodes-max 1 --managed
```
Notable real-world gotchas hit while standing this up (see `projectflow.txt` for the full log):
- A stale Python/Conda-based `awscli` conflicted with the official AWS CLI v2 (`C:\Program Files\Amazon\AWSCLIV2\aws.exe`) — resolved via `pip uninstall awscli` + installing `AWSCLIV2.msi`.
- An outdated `eksctl` (`0.158.0`) auto-selected Kubernetes **1.25**, which AWS now rejects (`unsupported Kubernetes version 1.25`), leaving the CloudFormation stack in `ROLLBACK_COMPLETE`. Fixed by upgrading `eksctl` to `0.230.0` and pinning `--version 1.36` explicitly. The failed stack had to be manually deleted (`aws cloudformation delete-stack` + `wait stack-delete-complete`) before retrying.
- The EKS control plane and node group are each backed by their own CloudFormation stack (`eksctl-<cluster>-cluster`, `eksctl-<cluster>-nodegroup-<name>`); node groups scale via an Auto Scaling Group, which is subject to AWS's per-account **Fleet Request** quota.
- Security group inbound rules had to be opened for port `5000` for the Flask service.

Useful verification commands:
```bash
aws eks --region us-east-1 update-kubeconfig --name flask-app-cluster
aws eks list-clusters
kubectl get nodes
kubectl get namespaces
kubectl get pods -A
kubectl get svc flask-app-service      # grab the LoadBalancer external DNS
curl http://<external-elb-dns>:5000
```

## Monitoring (Prometheus + Grafana)

- **Prometheus** (dedicated Ubuntu EC2, `t3.medium`, 20GB SSD, ports `9090`/`22` open): installed from the official release tarball, configured via `/etc/prometheus/prometheus.yml` with a `flask-app` scrape job pointed at the EKS LoadBalancer's DNS on port `5000` (scraping the app's `/metrics` endpoint), and run as `/usr/local/bin/prometheus --config.file=/etc/prometheus/prometheus.yml`.
- **Grafana** (dedicated Ubuntu EC2, `t3.medium`, 20GB SSD, ports `3000`/`22` open): installed via the `.deb` package, run as a systemd service (`grafana-server`, enabled on boot). Prometheus is added as a Grafana data source (`http://<prometheus-ec2-ip>:9090`), and dashboards are built on top of the custom app metrics (`app_request_count`, `app_request_latency_seconds`, `model_prediction_count`).

## Local Development

```bash
# 1. Environment
conda create -n MLFlow-Capstone python=3.10
conda activate MLFlow-Capstone
pip install -r requirements.txt

# 2. Secrets (not committed — create your own .env)
#    CAPSTONE_TEST=<your DagsHub token>

# 3. Run the DVC pipeline
dvc repro

# 4. Serve locally
cd flask_app
python app.py          # http://localhost:5000
```

## Tests

```bash
python -m unittest tests/test_model.py       # model load/signature/performance thresholds
python -m unittest tests/test_flask_app.py   # Flask route smoke tests
```

## Notes on Secrets

`.env`, `flask_app/.env`, and `credential.txt` hold local secrets (DagsHub token, AWS keys) and are excluded via `.gitignore` — they are **not** tracked in git. In CI/CD, the same values live only as GitHub Actions secrets and a Kubernetes Secret (`capstone-secret`), never in source.