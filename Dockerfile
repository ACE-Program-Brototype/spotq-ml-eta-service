# ==========================================
# Stage 1: Build dependencies with uv
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv for fast wheel compilation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency manifests
COPY pyproject.toml README.md ./
COPY app ./app

# Install dependencies into system site-packages
RUN uv pip install --system --no-cache .

# ==========================================
# Stage 2: Hardened Runtime Container
# ==========================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Create non-root system user for security
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1001 appuser

# Copy installed site-packages and binaries from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY --chown=appuser:appgroup . .

# Switch to non-root user
USER appuser

# Expose microservice HTTP port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Entrypoint using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]