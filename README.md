# SpotQ ETA Microservice (`spotq-eta-service`)

Sub-50ms Wait-Time Prediction Engine for the SpotQ Queue Management Ecosystem.

---

## Architecture Overview

Built using **Clean Architecture** principles:

* **Presentation Layer:** FastAPI HTTP routes, Health check diagnostics, Prometheus `/metrics` route.
* **Application Layer:** Prediction & Ingestion use cases, Interface definitions.
* **Domain Layer:** Pure Python Entities (`QueueItem`, `PartySize`) with zero framework dependencies.
* **Infrastructure Layer:** Redis Feature Store, MongoDB Ingestion Buffer, XGBoost model loaders, Structlog JSON logging, Doppler config injection.

---

## Quickstart (Local Development)

### Prerequisites
* Python 3.11+
* [`uv`](https://github.com/astral-sh/uv) package manager
* Docker & Docker Desktop

### 1. Environment Setup
```bash
# Create virtual environment and install dependencies
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"