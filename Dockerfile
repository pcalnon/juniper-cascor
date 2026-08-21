# =============================================================================
# JuniperCascor — Cascade Correlation Neural Network Training Service
# Multi-stage Dockerfile for production deployment
# =============================================================================
# Build: docker build -t juniper-cascor:latest .
# Run:   docker run -p 127.0.0.1:8200:8200 -e JUNIPER_CASCOR_HOST=0.0.0.0 -e JUNIPER_CASCOR_LOOPBACK_PUBLISH_ATTESTED=true -e JUNIPER_DATA_URL=http://localhost:8100 juniper-cascor:latest
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder — Install dependencies
# -----------------------------------------------------------------------------
FROM python:3.14-slim AS builder

WORKDIR /build

# Install build tools
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install CPU-only PyTorch first (avoids pulling CUDA which is ~4 GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install pinned dependencies from lockfile (best layer caching)
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

# Copy project files and install without deps (already installed above)
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps .

# -----------------------------------------------------------------------------
# Stage 2: Runtime — Minimal production image
# -----------------------------------------------------------------------------
FROM python:3.14-slim AS runtime

# Build provenance (juniper-ml notes/BUILD_PROVENANCE_DESIGN_2026-06-14.md):
# the deploy Makefile passes this repo's own git SHA, an ISO-8601 build
# timestamp, and the package version at build time. They are stamped as OCI
# labels and exported as env vars (below) so the running service reports them
# on /v1/health and `make doctor` can detect stale-image drift. Default empty
# when the image is built bare (read back as None by the app).
ARG GIT_SHA=""
ARG BUILD_DATE=""
ARG APP_VERSION=""

LABEL org.opencontainers.image.title="JuniperCascor"
LABEL org.opencontainers.image.description="Cascade Correlation Neural Network training service"
LABEL org.opencontainers.image.authors="Paul Calnon"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/pcalnon/juniper-cascor"
LABEL org.opencontainers.image.revision="${GIT_SHA}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.version="${APP_VERSION}"

# Create non-root user
RUN groupadd --gid 1000 juniper && \
    useradd --uid 1000 --gid juniper --shell /bin/bash --create-home juniper

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY --chown=juniper:juniper src/ ./src/

# Create required directories
RUN mkdir -p logs reports/junit data cascor-snapshots && chown -R juniper:juniper /app

USER juniper

# PYTHONPATH so imports from src/ resolve correctly
ENV PYTHONPATH=/app/src

# Service configuration
# Safe image default: bind loopback so the SEC-F22 bind guard does not crash a
# bare container. Published-container deployments must opt in explicitly with
# JUNIPER_CASCOR_HOST=0.0.0.0 plus a bind attestation —
# JUNIPER_CASCOR_LOOPBACK_PUBLISH_ATTESTED=true for a loopback-only host publish
# (the compose default) or JUNIPER_CASCOR_AUTH_PROXY_ATTESTED=true when a
# fronting authenticating reverse proxy terminates access — after constraining
# the host-side publish/proxy.
ENV JUNIPER_CASCOR_HOST=127.0.0.1
ENV JUNIPER_CASCOR_PORT=8200
ENV JUNIPER_CASCOR_LOG_LEVEL=INFO
# Declared in the IMAGE so every launch path is correct by default -- a bare `docker run`
# (which this file's own header documents) and the Helm deployment, not just compose.
# Left underived on purpose: the service tier used to compute this as parents[3]/"snapshots"
# = /app/snapshots while compose mounted /app/data, one directory away, so every
# containerized snapshot was written to the container's writable layer and lost on recreate.
# Nothing errored. Orchestrators now only have to MOUNT over a path that is already right.
ENV JUNIPER_CASCOR_SNAPSHOTS_DIR=/app/cascor-snapshots
ENV JUNIPER_DATA_URL=http://localhost:8100

# Build provenance (see the ARG block in the runtime stage above): exported so
# the app process can read its own source revision / build date and report them
# on /v1/health. Empty when built bare (read back as None).
ENV JUNIPER_CASCOR_GIT_SHA=${GIT_SHA}
ENV JUNIPER_CASCOR_BUILD_DATE=${BUILD_DATE}

EXPOSE 8200

# Health check for container orchestration (liveness + readiness)
# start-period=15s: PyTorch + numpy initialization adds ~10s startup overhead
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8200/v1/health/ready', timeout=5)" || exit 1

CMD ["python", "src/server.py"]
