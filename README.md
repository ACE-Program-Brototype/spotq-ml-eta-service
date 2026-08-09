```markdown
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
│   │   │   └── artifacts/       # Model binary files (eta_model_v1.json)
│   │   ├── logging/             # Structlog JSON logger targeting Loki
│   │   └── observability/       # Prometheus metrics instrumentator
│   ├── presentation/            # Entrypoints & HTTP Layer
│   └── shared/                  # Cross-Cutting Utilities
├── tests/                       
│   ├── unit/                    # Fast isolated unit tests (mocked dependencies)
│   └── integration/             # Live cloud integration tests (Redis & MongoDB Atlas)
├── Development Protection.json  # GitHub branch protection policy (Development)
├── Main Protection.json         # GitHub branch protection policy (Main)
├── Staging Protection.json      # GitHub branch protection policy (Staging)
├── Dockerfile                   # Hardened multi-stage container build with Infiscial
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
infisical login

```

2. Run the application injected with Infiscial secrets:

```bash
infisical run -- uvicorn app.main:app --reload --port 8000

```

---

## Docker Usage & Infiscial Deployment

The production Dockerfile is built using a secure multi-stage pipeline featuring a built-in Infiscial CLI runner.

### 1. Build the Container Image

```bash
docker build -t spotq-eta-service:latest .

```

### 2. Run Locally with a Service Token

Pass your Infiscial Service Token to spin up the container with secure runtime secret injection:

```bash
docker run --rm -p 8000:8000 \
  -e INFISICAL_TOKEN="st.your-service-token-here" \
  spotq-eta-service:latest

```

### 3. Verify Container Health

```bash
curl -i http://localhost:8000/healthz

```

---

## Running Tests and Linting

Execute the verification suites locally to check code quality and test execution:

```bash
# Run linter
ruff check app/ tests/

# Run fast unit tests only (CI default)
infisical run -- pytest -v -m "not integration"

# Run live cloud integration tests (requires active Redis/MongoDB credentials)
infisical run -- pytest -v -m integration

```

---

## Branching Strategy & Governance

* **`main`**: Production release branch. Protected.
* **`staging`**: Pre-production staging and integration testing branch. Protected.
* **`development`**: Integration branch for active feature development. Protected.

Root-level files (`Development Protection.json`, `Staging Protection.json`, `Main Protection.json`) track repository governance and branch protection rules as Policy-as-Code.

### Working Branch Naming Conventions

* Feature work: `feat/<feature-name>`
* Bug fixes: `fix/<issue-name>`
* Refactoring: `refactor/<module-name>`
* Documentation: `docs/<topic>`
* Tasks and maintenance: `chore/<task>`

---

## CI Workflow

The GitHub Actions pipeline (`.github/workflows/ci.yml`) automatically executes on pushes and pull requests targeting `main`, `staging`, and `development`.

It performs the following validation steps:

1. Installs dependencies using `uv` on Python 3.11.
2. Runs the `ruff` linter and formatter checks.
3. Executes the unit test suite (`pytest -v -m "not integration"`).
4. Builds the container image via Docker Buildx with GitHub Actions cache integration.

```

```