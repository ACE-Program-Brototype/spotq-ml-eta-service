# SpotQ ETA Microservice

A Python 3.11, FastAPI-based microservice foundation for restaurant queue waiting-time prediction. SpotQ ETA uses XGBoost model artifacts, MongoDB for queue data, and Redis for historical waiting-time features.

The service follows Clean Architecture principles and includes a reproducible machine-learning bootstrap pipeline for synthetic data generation, data cleaning, leakage-aware feature engineering, model evaluation, and historical-feature pre-warming.

> **Current ML status:** The SCRUM-1602 pipeline has been implemented and its unit tests pass. The verified training candidate and Redis aggregates use synthetic data. Real-world prediction accuracy, production serving compatibility, and sub-50 ms prediction latency have not yet been established.

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Local Setup](#local-setup)
- [Configuration and Secrets](#configuration-and-secrets)
- [ETA Machine Learning Pipeline](#eta-machine-learning-pipeline)
- [Synthetic Data Generation](#synthetic-data-generation)
- [Data Cleaning and Feature Engineering](#data-cleaning-and-feature-engineering)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Redis Historical Feature Pre-warming](#redis-historical-feature-pre-warming)
- [Running the API](#running-the-api)
- [Testing and Code Quality](#testing-and-code-quality)
- [Docker](#docker)
- [Branching and CI](#branching-and-ci)
- [Current Limitations and Next Steps](#current-limitations-and-next-steps)

---

## Architecture

SpotQ ETA separates business logic, application orchestration, infrastructure integrations, and HTTP presentation.

```text
                    ┌──────────────────────────┐
                    │     FastAPI Application  │
                    │       app/main.py        │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │ Presentation / HTTP API  │
                    │ Health and Prediction   │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │ Application & Domain     │
                    │ Use Cases / Business     │
                    │ Rules                    │
                    └────────────┬─────────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             │                   │                   │
    ┌────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
    │ MongoDB         │ │ Redis           │ │ ML Model        │
    │ Queue Records   │ │ Historical ETA  │ │ XGBoost         │
    │                 │ │ Features        │ │ Artifacts       │
    └─────────────────┘ └─────────────────┘ └─────────────────┘

                 Offline ML Bootstrap Pipeline
                                 │
                 ┌───────────────▼───────────────┐
                 │ Synthetic Data Generation     │
                 │ Cleaning and Feature Building │
                 │ Chronological Evaluation      │
                 │ XGBoost Candidate Training    │
                 │ Redis Feature Pre-warming    │
                 └───────────────────────────────┘
```

The offline training pipeline and Redis pre-warming workflow prepare data and model assets. They do not introduce a new real-time prediction endpoint or an automated model-promotion mechanism.

## Project Structure

```text
spotq-eta-service/
├── app/
│   ├── application/                  # Application use cases
│   ├── domain/                       # Core business rules
│   ├── infrastructure/
│   │   ├── config/                   # Pydantic settings
│   │   ├── cache/                    # Redis client
│   │   ├── database/                 # MongoDB client
│   │   ├── logging/                  # Structured logging
│   │   ├── observability/             # Metrics
│   │   └── ml_models/
│   │       ├── artifacts/
│   │       │   └── candidates/
│   │       │       ├── eta_model_candidate.json
│   │       │       ├── eta_training_report.json
│   │       │       └── redis_prewarm_report.json
│   │       ├── data_cleaning.py
│   │       ├── dataset.py
│   │       ├── diagnostics.py
│   │       ├── features.py
│   │       ├── loader.py
│   │       └── redis_history.py
│   ├── presentation/
│   │   ├── api/v1/endpoints/
│   │   ├── health/
│   │   └── predict/
│   ├── shared/
│   └── main.py
├── config/
│   └── restaurant_timezones.json
├── scripts/
│   ├── generate_dummy_model.py
│   ├── seed_training_data.py
│   ├── train_eta_model.py
│   └── prewarm_redis_history.py
├── tests/
│   ├── unit/
│   └── integration/
├── Development Protection.json
├── Staging Protection.json
├── Main Protection.json
├── Dockerfile
├── docker-compose.dev.yml
├── pyproject.toml
├── uv.lock
└── README.md
```

Model candidates and reports are stored separately from the existing serving artifact. Generated Python cache files are omitted from the structure above.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| API framework | FastAPI |
| Dependency management | uv |
| Database | MongoDB |
| Feature cache | Redis |
| Primary model | XGBoost |
| Benchmark model | Random Forest |
| Data processing | Pandas and NumPy |
| Validation | Pytest |
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
- Access to a configured Redis instance
- Infisical CLI and authorized project access, if your environment uses Infisical for secrets

## Local Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd spotq-eta-service
```

### 2. Install dependencies

This project uses `uv` and the committed lockfile.

```bash
uv sync --locked
```

If the project defines a separate development dependency group, install it according to the dependency groups declared in `pyproject.toml`.

### 3. Configure environment variables

Use your authorized environment configuration method. See [Configuration and Secrets](#configuration-and-secrets).

### 4. Verify the environment

```bash
uv run python --version
uv run pytest tests/unit -q
```

---

## Configuration and Secrets

SpotQ uses Pydantic Settings to load application configuration. MongoDB and Redis connectivity has been verified in the development environment.

When using Infisical, authenticate with your authorized account and inject the appropriate project environment variables into the process.

```bash
infisical login
```

To run a command with injected secrets, use the project and environment configuration appropriate for your setup:

```bash
infisical run -- uvicorn app.main:app --reload --port 8000
```

For local execution without Infisical, configure the required environment variables using the names defined in `app/infrastructure/config/settings.py`.

**Security guidelines**

- Never commit `.env` files, access tokens, database credentials, or service tokens.
- Do not print secret values when debugging configuration.
- Use separate credentials for development, staging, and production.
- Confirm the target database and Redis environment before running scripts that write data.

---

## ETA Machine Learning Pipeline

The bootstrap pipeline is implemented in three main scripts.

| Script | Responsibility |
|---|---|
| `scripts/seed_training_data.py` | Generate reproducible synthetic queue records and seed MongoDB |
| `scripts/train_eta_model.py` | Clean data, engineer features, evaluate baselines, and train an XGBoost candidate |
| `scripts/prewarm_redis_history.py` | Calculate historical wait-time aggregates and write them to Redis |

The pipeline supports development and integration testing without requiring live restaurant traffic.

### Pipeline workflow

1. Generate synthetic queue records using a configurable seed.
2. Insert records into the configured MongoDB collection using idempotent insertion logic.
3. Validate timestamps, waiting times, party sizes, workload values, and duplicate identifiers.
4. Build the canonical feature matrix using only historical information available at prediction time.
5. Split the data chronologically and evaluate historical-average, Random Forest, and XGBoost models.
6. Save the candidate model and training report.
7. Aggregate historical waiting times by restaurant, weekday, and hour.
8. Pre-warm Redis and generate a report describing the results.

### Inspect script options

Check the available arguments before running a script:

```bash
uv run python scripts/seed_training_data.py --help
uv run python scripts/train_eta_model.py --help
uv run python scripts/prewarm_redis_history.py --help
```

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

### Run seeding

First inspect the supported arguments:

```bash
uv run python scripts/seed_training_data.py --help
```

Then execute the command using the configured database environment and the intended collection, `spotq_production`.

The script writes to MongoDB. Verify the target environment before execution.

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

The model uses the following six input features in this exact order:

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

Historical features must only use eligible completed records available strictly before the prediction timestamp. Future observations and the current record's target must not influence the feature values.

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

The previously generated training report contains the following results:

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

The candidate is not automatically promoted to the serving artifact. The candidate's six-feature contract must be checked against the existing inference loader and serving contract before promotion.

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
infisical run -- uvicorn app.main:app --reload --port 8000
```

The repository contains health and prediction routing modules. Verify the routes registered by the current application before relying on any particular endpoint path.

The presence of a trained candidate does not, by itself, establish that the candidate is loaded by the live serving path. Model promotion and serving compatibility are separate steps.

---

## Testing and Code Quality

### Run unit tests

```bash
uv run pytest tests/unit -q
```

The latest verified unit-test run completed with **71 passed** and one non-blocking Starlette deprecation warning.

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

If Ruff is installed in the project environment:

```bash
uv run ruff check app/ tests/ scripts/
```

### Check whitespace errors

```bash
git diff --check
```

---

## Docker

The repository includes a `Dockerfile` and `docker-compose.dev.yml`.

### Build the image

```bash
docker build -t spotq-eta-service:latest .
```

### Run the container

Supply the required configuration through your approved secret-management mechanism or runtime environment:

```bash
docker run --rm -p 8000:8000 spotq-eta-service:latest
```

If the container is configured to launch through Infisical, provide the appropriate authorized runtime configuration. Do not place real service tokens directly in the README or commit them to source control.

Verify the health endpoint against the route actually registered by the application.

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

The SCRUM-1602 implementation establishes the data-to-model workflow, but the following items must be verified before treating the candidate as a production-ready model:

1. **Serving compatibility:** Confirm that the candidate's six features match the intended inference loader and feature contract. The existing serving artifact has a different feature schema.
2. **Artifact promotion:** Promote a candidate only after validation succeeds. A failed candidate must not overwrite an existing valid serving artifact.
3. **Training provenance:** Regenerate the candidate and report together when the report must identify the current implementation commit as the exact training code version.
4. **Real-data evaluation:** Evaluate on representative, trustworthy restaurant outcomes before making real-world accuracy claims.
5. **Integration verification:** Verify the configured MongoDB collection, Redis keys, fallback behavior, and intended service loading path in the target environment.
6. **Performance verification:** Benchmark the complete prediction path before claiming a sub-50 ms latency target has been met.

### SCRUM-1602 verification summary

- MongoDB connectivity: verified.
- Redis connectivity: verified.
- Synthetic pre-warm: completed, 649 buckets written.
- Candidate XGBoost model loading: verified.
- Unit tests: 71 passed.
- Real-world ETA accuracy: not established.
- Candidate promotion and serving compatibility: pending verification.

---

## License and Contribution

Follow the repository's contribution process and branch protection rules. Submit changes through a feature branch and pull request, with tests and relevant evaluation reports included for review.
