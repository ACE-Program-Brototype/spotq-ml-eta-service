#!/usr/bin/env bash
set -e

# If INFISICAL_TOKEN is set and non-empty, use Infisical CLI to inject secrets
if [ -n "$INFISICAL_TOKEN" ]; then
    echo "Starting spotq-ml-eta-service with Infisical secret injection..."
    exec infisical run -- uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
else
    echo "Starting spotq-ml-eta-service directly with environment variables..."
    exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
fi

