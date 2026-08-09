# ==========================================
# Stage 1: Build dependencies with uv
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv for fast wheel compilation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency manifests and application code
COPY pyproject.toml README.md ./
COPY app ./app

# Install dependencies into system site-packages
RUN uv pip install --system --no-cache .

# ==========================================
# Stage 2: Hardened Runtime Container with Infiscial
# ==========================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Install curl, bash, and Infiscial CLI for Debian/Ubuntu slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    bash \
    && curl -1sLf 'https://artifacts-cli.infisical.com/setup.deb.sh' | bash \
    && apt-get install -y infisical \
    && rm -rf /var/lib/apt/lists/*

# Create non-root system user for security
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1001 appuser

# Copy installed site-packages and binaries from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code and .infiscial.json configuration
COPY --chown=appuser:appgroup . .

# Switch to non-root user
USER appuser

# Expose microservice HTTP port
EXPOSE 8000

# Built-in container health diagnostic probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    INFISICAL_DISABLE_UPDATE_CHECK=true

# Entrypoint using Infisical to inject secrets into Uvicorn
CMD ["infisical", "run", "--", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]