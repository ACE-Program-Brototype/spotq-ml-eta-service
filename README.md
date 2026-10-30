# SpotQ ETA Microservice

A Python 3.11, FastAPI-based microservice for restaurant queue waiting-time prediction. SpotQ ETA uses XGBoost model artifacts, MongoDB for queue data, and Redis for historical waiting-time features and asynchronous event processing.

The service follows Clean Architecture principles and includes an **Automated ETA Model Training & Retraining Pipeline**, reproducible machine-learning bootstrap pipeline, data cleaning, leakage-aware canonical feature engineering, MLflow experiment tracking, candidate eligibility gates, historical-feature pre-warming, and a BullMQ consumer for completed queue events.

> **Current status (SCRUM-1197):** Implemented the **Automated ETA Model Training & Retraining Pipeline** using validated historical queue outcomes from MongoDB, MLflow experiment tracking with persistent run metadata, candidate eligibility thresholds (MAE/RMSE gates, baseline improvement, synthetic data safeguards), and contract-safe candidate verification without overwriting or disrupting the active production model. Ingestion via the BullMQ consumer (`ml-eta-queue`) feeds continuous retraining datasets.

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Local Setup](#local-setup)
- [Configuration and Secrets](#configuration-and-secrets)
- [Automated ETA Model Training & Retraining Pipeline](#automated-eta-model-training--retraining-pipeline)
- [Synthetic Data Generation](#synthetic-data-generation)
- [Data Cleaning and Feature Engineering](#data-cleaning-and-feature-engineering)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Redis Historical Feature Pre-warming](#redis-historical-feature-pre-warming)
- [Running the API](#running-the-api)
- [BullMQ Event Consumer](#bullmq-event-consumer)
- [Testing and Code Quality](#testing-and-code-quality)
- [Docker](#docker)
- [Branching and CI](#branching-and-ci)
- [Current Limitations and Next Steps](#current-limitations-and-next-steps)

---

## Architecture

SpotQ ETA separates HTTP presentation, application logic, infrastructure integrations, and machine-learning workflows.

```text
                       ┌──────────────────────────┐
                       │     FastAPI Application  │
                       │       app/main.py        │
                       └────────────┬─────────────┘
                                    │
                       ┌────────────▼─────────────┐
                       │ Presentation / HTTP API  │
                       │ Health and Prediction    │
                       └────────────┬─────────────┘
                                    │
                       ┌────────────▼─────────────┐
                       │ Application & Domain     │
                       │ Use Cases / Business     │
                       │ Rules                    │
                       └────────────┬─────────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  │                 │                 │
         ┌────────▼────────┐ ┌──────▼────────┐ ┌──────▼────────┐
         │ MongoDB         │ │ Redis         │ │ XGBoost       │
         │ Queue Events    │ │ Historical ETA│ │ Model         │
         │                 │ │ Features      │ │ Artifacts     │
         └─────────────────┘ └───────────────┘ └───────────────┘


                 Asynchronous Queue Ingestion

         ┌────────────────────┐
         │ MERN Backend        │
         │ BullMQ Producer     │
         └──────────┬─────────┘
                    │ Enqueue completed queue events
         ┌──────────▼─────────┐
         │ Redis 7.x           │
         │ BullMQ Queue        │
         │ ml-eta-queue        │
         └──────────┬─────────┘
                    │ Consume jobs
         ┌──────────▼─────────┐
         │ Python BullMQ       │
         │ Consumer Worker     │
         └──────────┬─────────┘
                    │ Validate and persist
         ┌──────────▼─────────┐
         │ MongoDB             │
         │ queue_events        │
         └────────────────────┘


              Offline ML Bootstrap Pipeline

         ┌───────────────────────────────┐
         │ Synthetic Data Generation     │
         │ Cleaning and Feature Building │
         │ Chronological Evaluation      │
         │ XGBoost Candidate Training    │
         │ Redis Feature Pre-warming     │
         └───────────────────────────────┘
```

The API and queue consumer are separate processes. The consumer receives events through BullMQ and writes validated queue records to MongoDB. It does not require an HTTP bridge between the MERN producer and the Python service.

The offline training pipeline prepares model candidates and historical features. It does not introduce an automated model-promotion mechanism.

## Project Structure

```text
spotq-ml-eta-service/
├── app/
│   ├── application/
│   │   ├── schemas/
│   │   │   └── queue_event.py
│   │   └── queue_event_processor.py
│   ├── domain/
│   ├── infrastructure/
│   │   ├── config/
│   │   ├── cache/
│   │   ├── database/
│   │   ├── logging/
│   │   ├── observability/
│   │   ├── queue/
│   │   │   └── bullmq_worker.py
│   │   └── ml_models/
│   │       ├── artifacts/
│   │       │   └── candidates/
│   │       ├── data_cleaning.py
│   │       ├── dataset.py
│   │       ├── diagnostics.py
│   │       ├── features.py
│   │       ├── loader.py
│   │       ├── redis_history.py
│   │       └── tracking.py
│   ├── presentation/
│   │   ├── admin/
│   │   ├── api/v1/endpoints/
│   │   ├── health/
│   │   └── predict/
│   ├── shared/
│   └── main.py
├── config/
│   └── restaurant_timezones.json
├── docs/
│   ├── model-lifecycle.md
│   └── model-training-workflow.md
├── scripts/
│   ├── generate_dummy_model.py
│   ├── seed_training_data.py
│   ├── train_eta_model.py
│   ├── prewarm_redis_history.py
│   ├── run_queue_worker.py
│   ├── produce_test_queue_event.py
│   ├── produce_duplicate_test_event.py
│   ├── produce_conflicting_test_event.py
│   └── produce_invalid_test_event.py
├── tests/
│   ├── unit/
│   └── integration/
├── Dockerfile
├── docker-compose.dev.yml
├── pyproject.toml
├── uv.lock
└── README.md
```

The queue-event schema is shared between the HTTP ingestion endpoint and the application processor. The processor contains event validation and persistence logic, while the infrastructure worker handles BullMQ job processing and failure classification.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| API framework | FastAPI |
| Dependency management | uv |
| Database | MongoDB |
| Feature cache and queue transport | Redis |
| Queue processing | BullMQ Python worker |
| Node.js producer | BullMQ v6, as reported by the MERN team |
| Python queue library | `bullmq==3.3.4` |
| Primary model | XGBoost |
| Benchmark model | Random Forest |
| Experiment tracking | MLflow |
| Data processing | Pandas and NumPy |
| Validation | Pydantic |
| Testing | Pytest |
| Configuration | Pydantic Settings |
| Logging | Structlog |
| Observability | Prometheus instrumentation |

---

## Prerequisites

- Python 3.11+
- `uv`
- Git
- Docker and Docker Compose, if using containers
- Access to a configured MongoDB instance
- Access to Redis
- Access to MLflow tracking server (default: `http://localhost:5000`)
- Infisical CLI and authorized project access, if required by your environment

For the BullMQ consumer, the development environment must also use the queue name and prefix agreed with the MERN team.

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/ACE-Program-Brototype/spotq-ml-eta-service.git
cd spotq-ml-eta-service
```

### 2. Install dependencies

```bash
uv sync --locked
```

### 3. Configure environment variables

Configure the required environment variables using the project's approved secret-management method.

### 4. Verify the environment

```bash
uv run python --version
uv run pytest tests/unit -q
```

## Configuration and Secrets

SpotQ uses Pydantic Settings to load application configuration. MongoDB and Redis connectivity has been verified in the development environment.

When using Infisical, authenticate with your authorized account:

```bash
infisical login
```

Run the API with the project and environment configuration appropriate for your setup:

```bash
infisical run --projectId <your-project-id> --env=dev -- uv run uvicorn app.main:app --reload --port 8000
```

For local execution without Infisical, configure the required variables using the names defined in `app/infrastructure/config/settings.py`.

### BullMQ configuration

| Variable | Default | Purpose |
|---|---|---|
| `BULLMQ_QUEUE_NAME` | `ml-eta-queue` | BullMQ queue consumed by the Python worker |
| `BULLMQ_PREFIX` | `bull` | BullMQ Redis key prefix |
| `BULLMQ_CONCURRENCY` | `5` | Maximum concurrent jobs; valid range 1–50 |

The MERN team has reported the following development configuration:

- Queue name: `ml-eta-queue`
- BullMQ version: `^6.0.9`
- ioredis version: `^6.0.0`
- Redis server: Redis 7.x using the official Docker image
- Key prefix: `bull`

Redis keys therefore follow the expected pattern `bull:ml-eta-queue:*`.

The Python consumer uses its configured Redis connection settings. Ensure its Redis hostname, port, authentication, queue name, and prefix match the development environment used by the producer. Inside Docker Compose, a service hostname may differ from the host machine's `localhost`.

The Python package version is pinned separately in `pyproject.toml`. Cross-language interoperability with the MERN producer has not yet been verified.

### Security guidelines

- Never commit `.env` files, access tokens, database credentials, or service tokens.
- Do not print secret values when debugging configuration.
- Use separate credentials for development, staging, and production.
- Confirm the target database and Redis environment before running scripts that write data.

---

## Automated ETA Model Training & Retraining Pipeline

- **Script:** `scripts/train_eta_model.py`
- **Tracking Context:** `app/infrastructure/ml_models/tracking.py`
- **Detailed Guide:** [`docs/model-training-workflow.md`](file:///Users/jeswin07/Documents/Data%20Science%20AI%20ML/Data%20Science%20Btype/SpotQ_ML/spotq-ml-eta-service/docs/model-training-workflow.md)

The automated training and retraining pipeline builds validated candidate model artifacts using historical completed queue outcomes from MongoDB. It operates completely out-of-band without disrupting active inference.

### Core Pipeline Capabilities

1. **Historical Data Validation:** Pulls completed queue outcomes from MongoDB (`queue_events`), verifies required fields, timestamp chronology, non-negative wait durations $\le 120$ min, duration consistency, deduplication, and enforces a configurable `--min-sample-size` (default: 50).
2. **Canonical 6-Feature Matrix:** Builds leakage-free feature vectors matching the real-time inference contract: `party_size`, `total_people_ahead`, `hour_of_day`, `day_of_week`, `is_weekend`, and `historical_average_wait`.
3. **Chronological Evaluation & Hyperparameter Search:** Purged expanding-window cross-validation and chronological train/test split with embargo for in-flight queues. Tunes XGBoost via `RandomizedSearchCV` and benchmarks against training-median and historical-average baselines.
4. **Candidate Eligibility Gates:** Configurable gates (`--max-test-mae`, `--max-test-rmse`, baseline outperformance). Guardrail strictly marks models trained on synthetic data as ineligible for production.
5. **Contract-Safe Candidate Verification:** Serializes candidates to `app/infrastructure/ml_models/artifacts/candidates/eta_model_candidate.json`, reloads the booster, validates feature names, ordering, and count, checks metadata, and runs smoke inference without touching or overwriting the active production model.
6. **MLflow Experiment Tracking:** Persistently logs run parameters, dataset provenance, metrics, code revision git SHA, tags, and candidate artifacts to the MLflow tracking server. Unhandled failures are explicitly recorded as `FAILED`.

### Executing the Training & Retraining Pipeline

#### 1. Retrain on Historical Real Queue Data
```bash
uv run python -m scripts.train_eta_model \
  --dataset-mode real \
  --collection queue_events \
  --min-sample-size 100 \
  --max-test-mae 4.5 \
  --max-test-rmse 7.5
```

#### 2. Train on Synthetic Data (CI / Local Verification)
```bash
uv run python -m scripts.train_eta_model \
  --dataset-mode synthetic \
  --min-sample-size 50 \
  --n-iter 12
```

### MLflow Experiment Tracking

The pipeline automatically connects to the configured MLflow server (default: `http://localhost:5000`):

| Variable | Default | Purpose |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow tracking server endpoint |
| `MLFLOW_EXPERIMENT_NAME` | `spotq-eta-service` | Experiment grouping for runs |

Start MLflow locally using Docker Compose:
```bash
cd spotq-infra
docker compose up -d mlflow
```
Access the tracking UI at `http://localhost:5000` to inspect candidate metrics, parameters, and model artifacts.

---

## Synthetic Data Generation

**Script:** `scripts/seed_training_data.py`

The generator creates synthetic queue records across multiple mock restaurants. The implemented dataset contains 3,000 records across five mock restaurants.

The synthetic demand model incorporates:

- Lunch peaks between 12:00 and 14:00.
- Dinner peaks between 19:00 and 21:30.
- Increased weekend demand.
- Nonlinear effects from party size and queue workload.
- Restaurant-specific service characteristics and variability.
- Chronologically consistent join and completion timestamps.

Each generated record includes synthetic-data identification fields. Stable event identifiers and idempotent insertion logic help prevent duplicate events on repeated seeding.

Before executing the script, inspect its supported arguments and verify the target database and collection. The script writes to MongoDB.

---

## Data Cleaning and Feature Engineering

### Data cleaning

**Module:** `app/infrastructure/ml_models/data_cleaning.py`

The cleaning stage validates queue records and produces diagnostics for accepted and rejected records.

Checks include:

- Missing required values and invalid timestamps.
- Completion before queue joining.
- Non-positive waiting times and waiting times exceeding 120 minutes.
- Invalid party sizes and workload values.
- Duplicate stable identifiers.

Rejected records must be accounted for with exclusion reasons.

### Canonical feature contract

**Module:** `app/infrastructure/ml_models/features.py`

The model uses six input features in this exact order:

| Position | Feature | Type |
|---:|---|---|
| 1 | `party_size` | Integer |
| 2 | `total_people_ahead` | Integer |
| 3 | `hour_of_day` | Integer, 0–23 |
| 4 | `day_of_week` | Integer, Monday = 0 |
| 5 | `is_weekend` | Integer, 0 or 1 |
| 6 | `historical_average_wait` | Float |

**Target:** `actual_wait_time_minutes`

Restaurant identity is used to isolate historical statistics but is not automatically included as a numeric model input.

Historical features must only use eligible completed records available strictly before the prediction timestamp. Future observations and the current record's target must not influence feature values.

Training and inference must follow the same feature schema, ordering, data types, and fallback semantics.

---

## Model Training and Evaluation

**Script:** `scripts/train_eta_model.py`

The training pipeline explicitly selects either synthetic or real data mode. It evaluates a historical-average baseline, a training-median baseline, a Random Forest benchmark, and a tuned XGBoost model.

Validation uses chronological holdout evaluation and time-aware cross-validation rather than randomly shuffled time-series splits.

### Train a synthetic candidate

```bash
uv run python scripts/train_eta_model.py \
  --dataset-mode synthetic \
  --timezone-map config/restaurant_timezones.json \
  --output-dir app/infrastructure/ml_models/artifacts/candidates
```

This command can regenerate candidate outputs. Review the configured environment and arguments before running it.

### Verified synthetic results

| Model | Holdout MAE | Holdout RMSE | R² |
|---|---:|---:|---:|
| Training median | 10.86 min | 15.36 min | -0.0420 |
| Historical average | 10.80 min | 14.03 min | 0.1309 |
| Random Forest | 3.08 min | 3.97 min | 0.9306 |
| Tuned XGBoost | 2.89 min | 3.73 min | 0.9385 |

The XGBoost candidate reduced holdout MAE by approximately 73.26% relative to the historical-average baseline on this synthetic dataset.

These results validate pipeline mechanics and model learning against generated data. They must not be presented as real-world ETA accuracy.

### Candidate artifacts

Training outputs are stored under:

```text
app/infrastructure/ml_models/artifacts/candidates/
├── eta_model_candidate.json
└── eta_training_report.json
```

The report includes training configuration, cleaning diagnostics, feature information, evaluation results, and model metadata.

The candidate is not automatically promoted to the serving artifact. Its six-feature contract must be checked against the existing inference loader and serving contract before promotion.

---

## Redis Historical Feature Pre-warming

**Script:** `scripts/prewarm_redis_history.py`

**Reader helper:** `app/infrastructure/ml_models/redis_history.py`

The pre-warm script aggregates completed queue records by restaurant, local weekday, and local hour, using the configured timezone mapping.

### Redis key format

```text
eta:history:{restaurant_id}:{day_of_week}:{hour_of_day}
```

Values are numeric historical-average waiting times. Supporting metadata records the source and sample count.

### Fallback hierarchy

When an exact historical bucket is unavailable, the reader attempts:

1. Exact restaurant, weekday, and hour.
2. Same restaurant and weekday.
3. Same restaurant across available buckets.
4. Global historical average.
5. Configured cold-start default of 20 minutes.

The pre-warm script retries Redis connections up to three attempts and reports failure if the connection cannot be established.

### Run pre-warming

```bash
uv run python scripts/prewarm_redis_history.py \
  --collection spotq_production \
  --timezones config/restaurant_timezones.json
```

### Inspect the report

```bash
cat app/infrastructure/ml_models/artifacts/candidates/redis_prewarm_report.json
```

### Verified result

| Metric | Result |
|---|---:|
| Input records | 3,000 |
| Rejected records | 0 |
| Redis buckets written | 649 |
| Synthetic-only buckets | 649 |
| Execution status | Completed |

All synthetic aggregates remain identifiable as synthetic and must not be interpreted as validated live restaurant history.

---

## Running the API

The FastAPI application entry point is `app.main:app`.

For local development, with the required environment variables configured:

```bash
uv run uvicorn app.main:app --reload --port 8000
```

With Infisical:

```bash
infisical run --projectId <your-project-id> --env=dev -- uv run uvicorn app.main:app --reload --port 8000
```

The verified development endpoints are:

- `GET /healthz`
- `GET /api/v1/eta/predict`
- Swagger UI at `/docs`
- OpenAPI schema at `/openapi.json`

These are local development URLs when the service runs on `127.0.0.1:8000`.

Development serving verification: the six-feature synthetic candidate loaded successfully, and the prediction endpoint returned HTTP 200. The existing seven-feature `eta_model_v1.json` artifact is incompatible with the canonical six-feature serving contract and remains uncommitted. The successful development test does not establish real-world model accuracy.

---

## BullMQ Event Consumer

**Worker:** `app/infrastructure/queue/bullmq_worker.py`

**Entrypoint:** `scripts/run_queue_worker.py`

The BullMQ consumer runs as a standalone Python process, separate from FastAPI. It consumes completed queue events from Redis and persists validated records to MongoDB.

### Architecture

```text
MERN Backend
    │
    │ Enqueue completed queue event
    ▼
BullMQ Queue: ml-eta-queue
    │
    ▼
Redis 7.x
Prefix: bull
    │
    ▼
Python BullMQ Consumer
    │
    ├── Validate event schema
    ├── Validate timestamps and wait duration
    ├── Persist to MongoDB
    ├── Handle identical duplicates idempotently
    └── Permanently reject invalid/conflicting events
    │
    ▼
MongoDB: queue_events
```

### Run the consumer

Ensure Redis and MongoDB are reachable and the environment is configured to match the producer.

Set the queue configuration:

```bash
export BULLMQ_QUEUE_NAME=ml-eta-queue
export BULLMQ_PREFIX=bull
export BULLMQ_CONCURRENCY=5
```

Then run the worker:

```bash
uv run python scripts/run_queue_worker.py
```

The worker initializes MongoDB, ensures the unique queue-event index exists, creates the BullMQ consumer, and begins processing jobs.

When using Infisical, use the authorized project configuration for the worker process:

```bash
infisical run --projectId <your-project-id> --env=dev -- uv run python scripts/run_queue_worker.py
```

Run the API and consumer as separate processes. In containerized development, ensure both services can reach the same Redis instance and use compatible connection settings.

### Event validation

The shared schema is defined in `app/application/schemas/queue_event.py`.

Each event must include:

| Field | Validation |
|---|---|
| `queue_entry_id` | Non-empty string, maximum 128 characters |
| `restaurant_id` | Non-empty string, maximum 128 characters |
| `party_size` | Integer, 1–100 |
| `joined_at` | Valid ISO 8601 timestamp with timezone |
| `completed_at` | Valid ISO 8601 timestamp with timezone, after joining |
| `actual_wait_time_minutes` | Positive finite number, maximum 120 |
| `total_people_ahead_at_join` | Integer, 0–1000 |

The processor normalizes timestamps to UTC and checks that the difference between `joined_at` and `completed_at` matches `actual_wait_time_minutes` within a tolerance of 0.51 minutes.

### Persistence and idempotency

The processor writes events to the MongoDB `queue_events` collection.

A unique index named `uq_queue_entry_id` on `queue_entry_id` prevents duplicate event records, provided existing data satisfies the uniqueness constraint.

The processor handles redelivery as follows:

- **New event:** validates and persists the event.
- **Identical duplicate:** confirms that the existing event matches and returns a successful duplicate result without inserting another record.
- **Conflicting duplicate:** rejects reuse of the same event ID with different event data and preserves the existing record.
- **Invalid event:** permanently rejects schema-invalid or business-invalid payloads.
- **Transient failure:** propagates unexpected database or processing errors so BullMQ can apply its retry policy.

The worker only returns success after MongoDB persistence succeeds or an identical existing event has been confirmed.

### Retry and failed-job handling

The consumer classifies invalid payloads and conflicting duplicates as permanent failures using BullMQ's `UnrecoverableError`. Unexpected errors are propagated to the queue library for retry handling.

Retry attempts, backoff, and failed-job retention must be configured by the producer when it creates jobs. The Python consumer does not retroactively define those job options.

Failed jobs should remain available according to the agreed retention policy so they can be inspected and, when appropriate, replayed.

### Local verification

The following scenarios have been tested locally using a Python BullMQ producer and the development Redis and MongoDB services:

- Valid event persisted successfully and the job reached the completed state.
- Identical duplicate did not create a second MongoDB record.
- Conflicting duplicate failed permanently, and the original record remained unchanged.
- Invalid payload failed permanently.
- Unit tests confirmed that unexpected processing errors propagate for retry handling.

**Integration status:** Python-side queue processing and persistence are verified. The MERN producer is still under development, so cross-language integration has not yet been confirmed. The actual Node.js job name, payload, retry/backoff configuration, and failed-job retention must be checked when the producer becomes available.

---

## Testing and Code Quality

### Run unit tests

```bash
uv run pytest tests/unit -q
```

The latest verified unit-test run completed with **130 passed** and four non-blocking warnings.

### Run the full test suite

```bash
uv run pytest -q
```

The full suite may include integration tests requiring external services.

### Run integration tests

```bash
uv run pytest tests/integration -v
```

Ensure the required MongoDB and Redis credentials are configured before executing integration tests.

### Run linting

```bash
uv run ruff check app/ scripts/ tests/
```

### Check whitespace errors

```bash
git diff --check
```

The latest SCRUM-1197 verification confirmed that Ruff, Black formatting, and `git diff --check` pass, with all 130 unit tests passing.

---

## Docker

The repository includes a `Dockerfile` and `docker-compose.dev.yml`.

### Build the image

```bash
docker build -t spotq-ml-eta-service:latest .
```

### Run the API container

```bash
docker run --rm -p 8000:8000 spotq-ml-eta-service:latest
```

Supply the required configuration through the approved secret-management mechanism or runtime environment. Do not put service tokens directly in the README or commit them to source control.

Development verification confirmed `GET /healthz`, `GET /api/v1/eta/predict`, and `/openapi.json` returned HTTP 200. The Docker API container previously failed to start because its Infisical runtime configuration did not provide the required project configuration/authentication. Container startup remains unresolved; do not consider Docker deployment verified.

The BullMQ consumer must also be deployed as a separately managed process with access to the configured Redis and MongoDB instances.

---

## Branching and CI

The repository uses protected integration and release branches:

- `main` — production release branch.
- `staging` — pre-production validation branch.
- `development` — integration branch for active feature work.

Root-level branch protection JSON files document repository governance:

- `Development Protection.json`
- `Staging Protection.json`
- `Main Protection.json`

### Branch naming conventions

| Work type | Convention |
|---|---|
| Features | `feat/<feature-name>` |
| Bug fixes | `fix/<issue-name>` |
| Refactoring | `refactor/<module-name>` |
| Documentation | `docs/<topic>` |
| Maintenance | `chore/<task>` |

The documented CI workflow is `.github/workflows/ci.yml`. When present and configured, CI should validate dependencies, linting, unit tests, and the container build for the configured target branches.

---

## Current Limitations and Next Steps

The SCRUM-1602 implementation establishes the data-to-model workflow. SCRUM-1608 adds a locally verified BullMQ consumer, but the MERN producer integration remains pending.

The following items must be verified before treating the model and ingestion integration as production-ready:

1. **Artifact compatibility:** The six-feature development candidate loaded successfully against the canonical serving contract. The separate seven-feature serving artifact remains incompatible and must not be promoted without deliberate schema reconciliation.
2. **Artifact promotion:** Promote a candidate only after validation succeeds. A failed candidate must not overwrite an existing valid serving artifact.
3. **Training provenance:** Regenerate the candidate and report together when the report must identify the current implementation commit as the exact training code version.
4. **Real-data evaluation:** Evaluate on representative, trustworthy restaurant outcomes before making real-world accuracy claims.
5. **Data integration:** Verify the configured MongoDB collection, Redis keys, fallback behavior, and intended service loading path in the target environment.
6. **Performance verification:** Twenty sequential loopback requests were below 50 ms, but a representative concurrent load test is still required before claiming the latency target is met.
7. **Fallback investigation:** The live prediction returned `fallback_used: true`; investigate the historical-feature lookup and confirm the intended Redis instance and fallback behavior.
8. **Container startup:** Resolve Infisical configuration for the Docker API container and repeat health and prediction checks inside the container.
9. **BullMQ interoperability:** Validate the Node.js producer against the Python consumer using `ml-eta-queue`, the `bull` prefix, the actual job name and payload, and the producer's retry/backoff and failed-job retention settings.
10. **Queue deployment:** Run the worker as a separately managed process with appropriate Redis and MongoDB configuration.
11. **MongoDB index readiness:** Confirm existing production records contain no duplicate `queue_entry_id` values before creating the unique index.
12. **Operational resilience:** Test worker recovery after Redis or MongoDB interruptions, graceful shutdown, and replay of retained failed jobs.

### Verification summary

- MongoDB connectivity: verified in development.
- Redis connectivity: verified in development.
- Synthetic pre-warm: completed, 649 buckets written.
- Candidate XGBoost model loading: verified.
- SCRUM-1608 unit tests: latest run, 100 passed.
- Ruff and whitespace checks: passed on the current feature branch.
- Queue event persistence: verified using a Python producer and consumer.
- Duplicate and invalid event handling: verified locally.
- MERN producer interoperability: pending.
- Real-world ETA accuracy: not established.
- Six-feature candidate loading in development: verified.
- Seven-feature artifact compatibility: unresolved; artifact remains uncommitted.
- Docker API startup: unresolved.
- Live prediction historical-feature fallback: observed; root cause not yet confirmed.
