
# SpotQ ETA Microservice

A production-ready, sub-50ms wait-time prediction microservice built with Python 3.11, FastAPI, and Clean Architecture principles. This service evaluates live restaurant queue volumes against an XGBoost model and manages historical feature baselines via Redis and MongoDB.

---

## Folder Structure

```text
spotq-eta-service/
├── app/
│   ├── domain/                  # Core Business Rules (Pure Python)
│   ├── application/             # Use Cases & Orchestration
│   ├── infrastructure/          # External Tools & Drivers
│   │   ├── config/              # Pydantic Settings loading Infiscial env vars
│   │   ├── cache/               # Redis Feature Store implementation
│   │   ├── database/            # MongoDB Ingestion Buffer implementation
│   │   ├── ml_models/           # XGBoost model loader (.json artifact)
│   │   ├── logger/              # Structlog JSON logger targeting Loki
│   │   └── observability/       # Prometheus metrics instrumentator
│   ├── presentation/            # Entrypoints & HTTP Layer
│   └── shared/                  # Cross-Cutting Utilities
├── tests/                       # Unit and integration test suites
├── Dockerfile                   # Multi-stage production container build
├── docker-compose.dev.yml       # Local development multi-container setup
├── pyproject.toml               # Project metadata and dependencies
└── README.md                    # Project documentation

```

---

## Local Setup Instructions

### Prerequisites

* Python 3.11+
* `uv` package manager (recommended) or `pip`
* Docker and Docker Compose

### Virtual Environment Setup

1. Clone the repository and navigate into the project directory:
```bash
git clone <repository-url>
cd spotq-eta-service

```


2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate

```


3. Install dependencies in editable mode with development tools:
```bash
uv pip install -e ".[dev]"

```



---

## Infiscial Configuration

This service uses Infiscial for secure environment variable management.

1. Install the Infiscial CLI and log in:
```bash
infiscial login

```


2. Run the application injected with Infiscial secrets:
```bash
infiscial run -- uvicorn main:app --reload --port 8000

```



---

## Docker Usage

To spin up the complete local development stack including Redis, MongoDB, and the ETA microservice:

1. Generate the model artifact stub (if required):
```bash
python scripts/generate_dummy_model.py

```


2. Build and run containers in detached mode:
```bash
docker compose -f docker-compose.dev.yml up --build -d

```


3. Verify container health:
```bash
curl http://localhost:8000/healthz

```



---

## Running Tests and Linting

Execute the verification suite locally to check code quality and unit tests:

```bash
ruff check app/ tests/
pytest -v

```

---

## Branching Strategy

* **`main`**: Production release branch. Protected.
* **`testing`**: Pre-production staging and integration testing branch. Protected.
* **`development`**: Integration branch for active feature development. Protected.

### Working Branch Naming Conventions

* Feature work: `feat/<feature-name>`
* Bug fixes: `fix/<issue-name>`
* Refactoring: `refactor/<module-name>`
* Documentation: `docs/<topic>`
* Tasks and maintenance: `chore/<task>`

---

## CI Workflow

The GitHub Actions pipeline (`.github/workflows/ci.yml`) automatically executes on pushes and pull requests targeting `main`, `testing`, and `development`.

It performs the following validation steps:

1. Installs dependencies using `uv` on Python 3.11.
2. Runs the `ruff` linter and formatter checks.
3. Executes the full `pytest` unit test suite.
4. Builds the container image via Docker Buildx with GitHub Actions cache integration.