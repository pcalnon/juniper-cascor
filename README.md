<!-- markdownlint-disable MD013 MD033 MD041 -->
<!--
  MD013 (line-length): README contains prose paragraphs that intentionally
                       exceed the 512-char ecosystem limit. Disabled file-wide
                       since wrapping mid-sentence harms PyPI rendering.
  MD033 (no-inline-html): The right-aligned logo + spacing rely on HTML.
  MD041 (first-line-heading): The HTML logo is the first line by design.
-->
<div align="right" width="150px" height="150px" align="right" valign="top"> <img src="images/Juniper_Logo_150px.png" alt="Juniper" align="right" valign="top" width="150px" /></div>
<br /> <br /> <br /> <br />

# Juniper: Dynamic Neural Network Research Platform

Juniper is an AI/ML research platform for investigating dynamic neural network architectures and novel learning paradigms.  The project emphasizes ground-up implementations from primary literature, enabling a more transparent exploration of fundamental algorithms.

## Juniper Cascor

`juniper-cascor` is the **Cascade-Correlation training service** of the Juniper platform. It implements the Fahlman & Lebiere (1990) algorithm from first principles and exposes that implementation as a long-running FastAPI service with a REST interface and two WebSocket channels — a training stream (`/ws/v1/training`) over which clients receive live training events, and a worker stream (`/ws/v1/workers`) over which `juniper-cascor-worker` instances connect to take part in distributed candidate-pool training. The service is dataset-aware: it consumes the named-version dataset registry of `juniper-data` and emits structured training events that `juniper-canopy` renders in real time. `juniper-cascor` is the place where the platform's training algorithms actually run; the other components surround it as data, monitoring, and parallelism layers.

## Distribution

`juniper-cascor` is published on PyPI as **[`juniper-cascor`](https://pypi.org/project/juniper-cascor/)**.
The package is also surfaced through the platform meta-distribution
**[`juniper-ml`](https://pypi.org/project/juniper-ml/)**, which installs
the full client stack via `pip install juniper-ml[all]`.

```bash
pip install juniper-cascor
```

## Ecosystem Compatibility

This service is part of the [Juniper](https://github.com/pcalnon/juniper-ml) ecosystem.
Verified compatible versions:

| juniper-data | juniper-cascor | juniper-canopy | data-client | cascor-client | cascor-worker |
|--------------|----------------|----------------|-------------|---------------|---------------|
| 0.6.x        | 0.4.x          | 0.4.x          | >=0.4.1     | >=0.4.0       | >=0.3.0       |

For full-stack Docker deployment and integration tests, see [`juniper-deploy`](https://github.com/pcalnon/juniper-deploy).

## Architecture

`juniper-cascor` is the **training service** of the Juniper ecosystem. It depends on `juniper-data` for datasets, is monitored in real time by `juniper-canopy`, and is parallelised across hosts by `juniper-cascor-worker`.

```text
┌─────────────────────┐     REST+WS      ┌──────────────────────┐
│   juniper-canopy    │ ◄──────────────► │   juniper-cascor     │
│   Dashboard         │                  │   Training Svc       │
│   Port 8050         │                  │   Port 8200 ◄── here │
└──────────┬──────────┘                  └──────┬──────┬────────┘
           │ REST                               │ REST │ WS
           ▼                                    ▼      │
┌──────────────────────────────────────────────────┐   │
│                   juniper-data                   │   │
│              Dataset Service · Port 8100         │   │
└──────────────────────────────────────────────────┘   │
                                                       ▼
                                       ┌─────────────────────────┐
                                       │ juniper-cascor-worker   │
                                       │ Distributed candidate   │
                                       │ training over           │
                                       │ /ws/v1/workers          │
                                       └─────────────────────────┘
```

**API surface**: REST (`/v1/...`) plus two WebSocket channels — `/ws/v1/training` for live training events and `/ws/v1/workers` for remote worker connections. All REST responses use the `{status, data, meta}` envelope.

## Related Services

| Service | Relationship | Notes |
|---------|-------------|-------|
| [juniper-data](https://github.com/pcalnon/juniper-data) | `juniper-cascor` fetches datasets from here | Set `JUNIPER_DATA_URL` |
| [juniper-canopy](https://github.com/pcalnon/juniper-canopy) | Monitors CasCor training in real time | Connects to `/ws/v1/training` |
| [juniper-cascor-client](https://github.com/pcalnon/juniper-cascor-client) | REST + WebSocket client library | `pip install juniper-cascor-client` |
| [juniper-cascor-worker](https://github.com/pcalnon/juniper-cascor-worker) | Distributed candidate-training worker | Connects to `/ws/v1/workers` |

## Service Configuration

Configuration is sourced from `src/api/settings.py` (Pydantic `BaseSettings`, `env_prefix="JUNIPER_CASCOR_"`). The variables most relevant to typical deployment are listed below; the full set (rate-limiting, worker rate-limiting, anomaly detection, WebSocket-control cooldown, and observability) is documented in [`docs/install/REFERENCE.md`](docs/install/REFERENCE.md).

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `JUNIPER_DATA_URL` | Yes | `http://localhost:8100` | `juniper-data` service URL |
| `JUNIPER_CASCOR_HOST` | No | `127.0.0.1` | Bind address (override to `0.0.0.0` for Docker) |
| `JUNIPER_CASCOR_PORT` | No | `8200` | Service port (container side) |
| `JUNIPER_CASCOR_LOG_LEVEL` | No | `INFO` | Log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `JUNIPER_CASCOR_LOG_FORMAT` | No | `text` | `text` or `json` (structured logging) |
| `JUNIPER_CASCOR_CORS_ORIGINS` | No | `[]` | Allowed CORS origins |
| `JUNIPER_CASCOR_API_KEYS` | No | `None` | Comma-separated API keys; authentication disabled when unset |
| `JUNIPER_CASCOR_API_KEYS_FILE` | No | — | Docker-secrets path for the API-keys list |
| `JUNIPER_DATA_API_KEY` | No | — | API key for outbound calls to `juniper-data` |
| `JUNIPER_CASCOR_METRICS_ENABLED` | No | `false` | Expose `/metrics` for Prometheus scraping |
| `JUNIPER_CASCOR_SENTRY_DSN` | No | `None` | Sentry DSN for error tracking |
| `CASCOR_LOG_LEVEL` | No | — | Legacy fallback for `JUNIPER_CASCOR_LOG_LEVEL` |

## Docker Deployment

```bash
# Full stack (recommended) — see juniper-deploy:
git clone https://github.com/pcalnon/juniper-deploy.git  # (private repository)
cd juniper-deploy && docker compose up --build

# Standalone:
docker build -t juniper-cascor:latest .
docker run --rm -p 8200:8200 \
  -e JUNIPER_CASCOR_HOST=0.0.0.0 \
  -e JUNIPER_DATA_URL=http://host.docker.internal:8100 \
  juniper-cascor:latest
```

The Dockerfile is multi-stage (Python 3.14-slim builder + runtime); the runtime image installs PyTorch from the CPU-only index (`https://download.pytorch.org/whl/cpu`) outside the lockfile to avoid the multi-gigabyte CUDA wheels. Container health is probed against `/v1/health/ready`.

## Dependency Lockfile

The `requirements.lock` file pins exact dependency versions for reproducible Docker builds. The `pyproject.toml` retains flexible `>=` ranges for local development. A parallel `uv.lock` covers `uv`-native resolution.

Regenerate after changing dependencies in `pyproject.toml`:

```bash
uv pip compile pyproject.toml \
  --extra ml --extra api --extra observability --extra juniper-data \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  --no-emit-package torch \
  -o requirements.lock
```

PyTorch is excluded from the lockfile and installed separately in the Dockerfile from the CPU-only index. The ecosystem-wide lockfile-freshness gate enforces this regeneration on every PR that touches `pyproject.toml`; if a regenerated lockfile triggers the self-pin trap of `uv pip compile -o requirements.lock` reading the existing file, compile to `/tmp/requirements.lock` and `mv` into place.

## Active Research Components

`juniper-cascor` houses four research components of the Juniper platform: the **Cascade-Correlation reference implementation** itself (Fahlman & Lebiere, 1990), kept inspectable at the level of candidate units, correlation objectives, and weight-freezing semantics; the **candidate-pool training protocol** that parallelises candidate-unit selection across processes and (via `juniper-cascor-worker`) across hosts; the **multi-network orchestration** machinery being developed for population-level comparative experiments (design notes: [`juniper-ml/notes/PHASE_6E_DESIGN.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/PHASE_6E_DESIGN.md), [`PHASE_6E_MULTI_NETWORK_DESIGN.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/PHASE_6E_MULTI_NETWORK_DESIGN.md)); and the **WebSocket training-event protocol** consumed by `juniper-canopy` for live introspection of network growth. The METRICS-MON R3.7 macOS CI integration soak (completed 2026-05-15) supports cross-platform reproducibility claims for these components.

## Quick Start Guide

### Prerequisites

- Python ≥ 3.12 (Docker image uses 3.14)
- Conda environment `JuniperCascor`
- A running `juniper-data` instance reachable at `JUNIPER_DATA_URL` (typically `http://localhost:8100`)

### Installation

```bash
git clone https://github.com/pcalnon/juniper-cascor.git
cd juniper-cascor
conda env create -f conf/conda_environment.yaml
conda activate JuniperCascor
pip install -e ".[ml,api,observability,test]"
```

The PyPI release is installable via `pip install juniper-cascor`; the editable-clone form above is the standard for active development.

### Verification

Start the service:

```bash
python src/server.py
```

Confirm the service is responding:

```bash
curl http://localhost:8200/v1/health
curl http://localhost:8200/v1/health/ready
```

The library is also usable directly from Python, bypassing the service:

```python
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

config = CascadeCorrelationConfig(input_size=2, output_size=2, random_seed=42)
network = CascadeCorrelationNetwork(config=config)
history = network.fit(x_train, y_train, epochs=100)
print(f"Test accuracy: {network.get_accuracy(x_test, y_test):.2%}")
```

Run the test suite:

```bash
cd src/tests && bash scripts/run_tests.bash
# or
pytest -m "not slow" -v
```

### Next Steps

- [`docs/install/QUICK_START.md`](docs/install/QUICK_START.md) — complete installation guide
- [`docs/install/USER_MANUAL.md`](docs/install/USER_MANUAL.md) — comprehensive usage guide
- [`docs/api/API_REFERENCE.md`](docs/api/API_REFERENCE.md) — complete REST and WebSocket API documentation
- [`juniper-deploy`](https://github.com/pcalnon/juniper-deploy) — Docker Compose orchestration for the full-stack platform
- [`juniper-ml`](https://pypi.org/project/juniper-ml/) — platform meta-package on PyPI

## Research Philosophy

The Juniper platform exists to study learning algorithms whose network architecture is not fixed in advance. Its initial anchor is the Cascade-Correlation algorithm of Fahlman and Lebiere (1990), implemented from the primary literature without recourse to higher-level abstractions that elide the algorithm's operational detail. The organising commitment is that algorithm implementations remain inspectable at the level at which they were originally specified: candidate units, correlation objectives, weight-freezing semantics, and the structural events that grow the network are first-class artifacts of the codebase rather than internal details of a library wrapper. This permits comparative work — across algorithms, datasets, and hyperparameter regimes — to be conducted on a known and reproducible substrate.

The current platform comprises a Cascade-Correlation training service exposing a REST and WebSocket interface, a dataset-generation service with a named-version registry that includes the ARC-AGI families, a real-time monitoring dashboard for inspecting training dynamics as they occur, and a distributed worker that parallelises candidate-unit training across hosts. Near-term work extends the architectural-growth catalogue beyond Cascade-Correlation, introduces multi-network orchestration for comparative experiments at the level of network populations rather than individual runs, and tightens the dataset–training–monitoring loop into a reproducible research workbench. The longer-term direction is the systematic empirical study of constructive and architecture-growing learning algorithms, with first-class infrastructure for the ablation, comparison, and replication that such a study requires.

## Documentation

| Document | Purpose |
|----------|---------|
| [`docs/DOCUMENTATION_OVERVIEW.md`](docs/DOCUMENTATION_OVERVIEW.md) | Navigation index for all `juniper-cascor` documentation |
| [`docs/INDEX.md`](docs/INDEX.md) | Quick documentation index |
| [`docs/install/QUICK_START.md`](docs/install/QUICK_START.md) | Complete installation guide |
| [`docs/install/USER_MANUAL.md`](docs/install/USER_MANUAL.md) | Comprehensive usage guide |
| [`docs/api/API_REFERENCE.md`](docs/api/API_REFERENCE.md) | Complete REST and WebSocket API reference |
| [`docs/testing/QUICK_START.md`](docs/testing/QUICK_START.md) | Testing instructions |
| [`docs/ci_cd/QUICK_START.md`](docs/ci_cd/QUICK_START.md) | Continuous integration guide |
| [`docs/source/QUICK_START.md`](docs/source/QUICK_START.md) | Contributor documentation |
| [`docs/overview/CONSTANTS_GUIDE.md`](docs/overview/CONSTANTS_GUIDE.md) | Configuration constants reference |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history |

## License

MIT License — see [`LICENSE`](LICENSE) for details.
