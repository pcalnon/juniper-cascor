# Juniper: Dynamic Neural Network Research Platform

[![CI/CD Pipeline](https://github.com/pcalnon/juniper-cascor/actions/workflows/ci.yml/badge.svg)](https://github.com/pcalnon/juniper-cascor/actions/workflows/ci.yml)

**Version**: 0.3.17 | [Changelog](CHANGELOG.md) | [Full Documentation](docs/DOCUMENTATION_OVERVIEW.md)

Juniper is an AI/ML research platform for investigating dynamic neural network architectures and novel learning paradigms. The project emphasizes ground-up implementations from primary literature, enabling a more transparent exploration of fundamental algorithms.

## Ecosystem Compatibility

This service is part of the [Juniper](https://github.com/pcalnon/juniper-ml) ecosystem.
Verified compatible versions:

| juniper-data | juniper-cascor | juniper-canopy | data-client | cascor-client | cascor-worker |
|---|---|---|---|---|---|
| 0.4.x | 0.3.x | 0.2.x | >=0.3.1 | >=0.1.0 | >=0.1.0 |

For full-stack Docker deployment and integration tests, see `juniper-deploy`.

## Architecture

JuniperCascor is the **training service** of the Juniper ecosystem. It depends on JuniperData for datasets and is monitored by juniper-canopy in real-time.

```text
┌─────────────────────┐     REST+WS      ┌──────────────────────┐
│   juniper-canopy     │ ◄──────────────► │  JuniperCascor       │
│   Dashboard         │                  │  Training Svc        │
│   Port 8050         │                  │  Port 8200  ◄── here │
└──────────┬──────────┘                  └──────────┬───────────┘
           │ REST                                    │ REST
           ▼                                         ▼
┌──────────────────────────────────────────────────────────────┐
│                      JuniperData                              │
│                   Dataset Service  ·  Port 8100               │
└──────────────────────────────────────────────────────────────┘
```

**API**: REST + WebSocket (`/ws/training`, `/ws/control`). All responses use a `{status, data, meta}` envelope.

## Related Services

| Service | Relationship | Notes |
|---------|-------------|-------|
| [juniper-data](https://github.com/pcalnon/juniper-data) | JuniperCascor fetches datasets from here | Set `JUNIPER_DATA_URL` |
| [juniper-canopy](https://github.com/pcalnon/juniper-canopy) | Monitors CasCor training in real-time | Connects to `/ws/training` |
| [juniper-cascor-client](https://github.com/pcalnon/juniper-cascor-client) | PyPI REST+WS client library | `pip install juniper-cascor-client` |

### Service Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `JUNIPER_DATA_URL` | Yes | `http://localhost:8100` | JuniperData service URL |
| `CASCOR_HOST` | No | `0.0.0.0` | Listen address |
| `CASCOR_PORT` | No | `8200` | Service port |
| `CASCOR_LOG_LEVEL` | No | `INFO` | Log verbosity (`DEBUG`, `INFO`, `WARNING`) |

### Docker Deployment

```bash
# Standalone:
docker build -t juniper-cascor:latest .
docker run -p 8200:8200 -e JUNIPER_DATA_URL=http://host.docker.internal:8100 juniper-cascor:latest

# Full stack:
git clone https://github.com/pcalnon/juniper-deploy.git
cd juniper-deploy && docker compose up --build
```

## Dependency Lockfile

The `requirements.lock` file pins exact dependency versions for reproducible Docker builds. The `pyproject.toml` retains flexible `>=` ranges for local development.

**Regenerate after changing dependencies in `pyproject.toml`:**

```bash
uv pip compile pyproject.toml --extra ml --extra api --extra observability --extra juniper-data \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  --no-emit-package torch -o requirements.lock
```

PyTorch is excluded from the lockfile and installed separately in the Dockerfile from the CPU-only index.

## Quick Start

### Prerequisites

- Python 3.11 or later (3.14 recommended)
- Conda package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/pcalnon/juniper-cascor.git
cd juniper-cascor

# Create and activate conda environment
conda env create -f conf/conda_environment.yaml
conda activate JuniperCascor

# Run the spiral problem evaluation
cd src && python main.py
```

### Run Tests

```bash
# Fast tests (recommended for development)
cd src/tests && bash scripts/run_tests.bash

# Or using pytest directly
pytest -m "not slow" -v
```

For detailed installation instructions, see the [Quick Start Guide](docs/install/quick-start.md).

## Active Research Components

**juniper_cascor**: Cascade Correlation Neural Network

- Reference implementation from foundational research (Fahlman & Lebiere, 1990)
- Designed for flexibility, modularity, and scalability
- Enables investigation of constructive learning algorithms

**juniper_canopy**: Interactive Research Interface

- Research-driven monitoring and visualization environment
- Delivers novel observations through real-time network introspection
- Transforms metrics into insights, accelerating experimental iteration

## Documentation

| Category                                                 | Description                           |
| -------------------------------------------------------- | ------------------------------------- |
| [Documentation Overview](docs/DOCUMENTATION_OVERVIEW.md) | Complete navigation guide to all docs |
| [Documentation Index](docs/index.md)                     | Quick documentation index             |
| [Quick Start](docs/install/quick-start.md)               | Get up and running quickly            |
| [User Manual](docs/install/user-manual.md)               | Comprehensive usage guide             |
| [API Reference](docs/api/api-reference.md)               | Complete API documentation            |
| [Testing Guide](docs/testing/quick-start.md)             | Testing instructions                  |
| [CI/CD Guide](docs/ci/quick-start.md)                    | Continuous integration                |
| [Source Code Guide](docs/source/quick-start.md)          | Contributor documentation             |
| [Constants Reference](docs/overview/constants-guide.md)  | Configuration constants               |

## Basic Usage

```python
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
import torch

# Create configuration
config = CascadeCorrelationConfig(
    input_size=2,
    output_size=2,
    random_seed=42
)

# Create and train network
network = CascadeCorrelationNetwork(config=config)
history = network.fit(x_train, y_train, epochs=100)

# Evaluate
accuracy = network.get_accuracy(x_test, y_test)
print(f"Test accuracy: {accuracy:.2%}")

# Save/Load
network.save_to_hdf5("model.h5")
loaded = CascadeCorrelationNetwork.load_from_hdf5("model.h5")
```

## Research Philosophy

Juniper prioritizes **transparency over convenience** and **understanding over abstraction**. By implementing algorithms from first principles, the platform provides researchers with increased visibility into network behavior, enabling a more rigorous and more controlled investigation of learning dynamics and architectural innovations.

## Important Notices

### Thread Safety Warning

**The `CascadeCorrelationNetwork` class is NOT thread-safe.** Do not share network instances between threads without proper synchronization. For concurrent training scenarios, create separate network instances per thread. The internal multiprocessing for candidate training is handled within the class and does not require external synchronization.

## Juniper Ecosystem

| Repository | Description |
|-----------|-------------|
| [juniper-cascor](https://github.com/pcalnon/juniper-cascor) | CasCor neural network training service (this repo) |
| [juniper-canopy](https://github.com/pcalnon/juniper-canopy) | Real-time monitoring dashboard |
| [juniper-data](https://github.com/pcalnon/juniper-data) | Dataset generation service |
| [juniper-data-client](https://github.com/pcalnon/juniper-data-client) | PyPI: `juniper-data-client` |
| [juniper-cascor-client](https://github.com/pcalnon/juniper-cascor-client) | PyPI: `juniper-cascor-client` |
| [juniper-cascor-worker](https://github.com/pcalnon/juniper-cascor-worker) | PyPI: `juniper-cascor-worker` |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome. Please see the [Source Code Guide](docs/source/quick-start.md) for development setup and coding conventions.
