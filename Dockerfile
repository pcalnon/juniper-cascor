# =============================================================================
# JuniperCascor — Cascade Correlation Neural Network Training Service
# Multi-stage Dockerfile for production deployment
# =============================================================================
# Build: docker build -t juniper-cascor:latest .
# Run:   docker run -p 8200:8200 -e JUNIPER_DATA_URL=http://localhost:8100 juniper-cascor:latest
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder — Install dependencies
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tools
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install CPU-only PyTorch first (avoids pulling CUDA which is ~4 GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install juniper-data-client (not yet on PyPI)
RUN pip install --no-cache-dir \
    "juniper-data-client @ git+https://github.com/pcalnon/juniper-data-client.git@main"

# Install project with all extras
RUN pip install --no-cache-dir -e ".[all]"

# -----------------------------------------------------------------------------
# Stage 2: Runtime — Minimal production image
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="JuniperCascor"
LABEL org.opencontainers.image.description="Cascade Correlation Neural Network training service"
LABEL org.opencontainers.image.authors="Paul Calnon"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/pcalnon/juniper-cascor"

# Create non-root user
RUN groupadd --gid 1000 juniper && \
    useradd --uid 1000 --gid juniper --shell /bin/bash --create-home juniper

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY --chown=juniper:juniper src/ ./src/

# Create required directories
RUN mkdir -p logs reports/junit data && chown -R juniper:juniper /app

USER juniper

# PYTHONPATH so imports from src/ resolve correctly
ENV PYTHONPATH=/app/src

# Service configuration
ENV CASCOR_HOST=0.0.0.0
ENV CASCOR_PORT=8200
ENV CASCOR_LOG_LEVEL=INFO
ENV JUNIPER_DATA_URL=http://localhost:8100

EXPOSE 8200

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8200/v1/health', timeout=5)" || exit 1

CMD ["python", "src/server.py"]
