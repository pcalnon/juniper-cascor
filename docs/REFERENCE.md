# Technical Reference — juniper-cascor

**Project**: juniper-cascor — Cascade Correlation Neural Network backend
**Author**: Paul Calnon
**License**: MIT License
**Last Updated**: 2026-08-30

Reference material relocated **verbatim** out of `AGENTS.md` under the shared-session-memory plan
(juniper-ml plan §P5 step e, Tier A). `AGENTS.md` is loaded into every session; this file is read on
demand. Nothing here was rewritten — each section carries a provenance line naming where it came
from.

**Hazards are deliberately NOT here.** Sections carrying directives whose *non-application destroys
work* were left resident in `AGENTS.md` rather than relocated — `## CI/CD Pipelines` and
`## Middleware Stack` were both excluded from this cut for that reason.

See also [`docs/INDEX.md`](INDEX.md), the documentation index for this repository.

---

## Table of Contents

- [Directory Structure Reference](#directory-structure-reference)
- [REST API Reference](#rest-api-reference)
- [WebSocket Protocol Reference](#websocket-protocol-reference)
- [Core Components Reference](#core-components-reference)
- [Constants Configuration Reference](#constants-configuration-reference)
- [Documentation Files Reference](#documentation-files-reference)
- [Key Dependencies Reference](#key-dependencies-reference)
- [Further Reading](#further-reading)

---

## Directory Structure Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

```text
juniper-cascor/
├── src/                              # Application source code
│   ├── main.py                       # CLI entry point (standalone training)
│   ├── server.py                     # FastAPI server entry point
│   ├── __init__.py
│   ├── api/                          # FastAPI REST/WebSocket service layer
│   │   ├── app.py                    #   Application factory with lifespan
│   │   ├── settings.py               #   Pydantic-based configuration
│   │   ├── security.py               #   API key auth and rate limiting
│   │   ├── middleware.py             #   Security headers, body limits, auth
│   │   ├── observability.py          #   Logging, Prometheus, Sentry, request IDs
│   │   ├── service_launcher.py       #   Auto-start companion services
│   │   ├── secrets.py                #   Docker secrets file loader
│   │   ├── lifecycle/                #   Training lifecycle management
│   │   │   ├── manager.py            #     TrainingLifecycleManager
│   │   │   ├── monitor.py            #     TrainingMonitor (metrics/callbacks)
│   │   │   └── state_machine.py      #     TrainingStateMachine
│   │   ├── routes/                   #   REST API endpoint handlers
│   │   │   ├── health.py             #     Health and readiness probes
│   │   │   ├── network.py            #     Network CRUD and inspection
│   │   │   ├── training.py           #     Training control
│   │   │   ├── dataset.py            #     Dataset metadata and arrays
│   │   │   ├── decision_boundary.py  #     Decision boundary visualization
│   │   │   ├── metrics.py            #     Prometheus metrics endpoint
│   │   │   ├── snapshots.py          #     Network snapshot management
│   │   │   └── workers.py            #     Remote worker registry
│   │   ├── websocket/                #   WebSocket real-time channels
│   │   │   ├── manager.py            #     WebSocketManager (connections, broadcast)
│   │   │   ├── control_stream.py     #     /ws/control (training commands)
│   │   │   ├── training_stream.py    #     /ws/training (metrics stream)
│   │   │   ├── worker_stream.py      #     /ws/v1/workers (remote worker protocol)
│   │   │   └── messages.py           #     Message builders
│   │   ├── workers/                  #   Remote WebSocket worker system
│   │   │   ├── registry.py           #     WorkerRegistry (pool management)
│   │   │   ├── coordinator.py        #     WorkerCoordinator (task dispatch)
│   │   │   ├── protocol.py           #     Wire protocol (JSON + binary)
│   │   │   ├── security.py           #     Worker authentication
│   │   │   └── audit.py              #     Worker audit logging
│   │   └── models/                   #   Pydantic request/response models
│   │       ├── common.py             #     ResponseEnvelope, ErrorResponse
│   │       ├── health.py             #     ReadinessResponse
│   │       ├── network.py            #     NetworkCreateRequest
│   │       └── training.py           #     TrainingStartRequest, TrainingParamUpdateRequest
│   ├── cascade_correlation/          # Core CasCor neural network
│   │   ├── cascade_correlation.py    #   CascadeCorrelationNetwork class
│   │   ├── cascade_correlation_config/
│   │   │   └── cascade_correlation_config.py  # Network configuration dataclass
│   │   └── cascade_correlation_exceptions/
│   │       └── cascade_correlation_exceptions.py  # Custom exceptions
│   ├── candidate_unit/               # Candidate hidden unit training
│   │   └── candidate_unit.py         #   CandidateUnit, ActivationWithDerivative
│   ├── parallelism/                  # Distributed task scheduling
│   │   └── task_distributor.py       #   TaskDistributor (local-first policy)
│   ├── spiral_problem/               # Two-spiral classification problem
│   │   ├── spiral_problem.py         #   SpiralProblem orchestration
│   │   ├── data_provider.py          #   Spiral data generation
│   │   └── check.py                  #   Validation utilities
│   ├── snapshots/                    # Network serialization
│   │   ├── snapshot_serializer.py    #   Serialize/deserialize networks
│   │   ├── snapshot_common.py        #   Shared snapshot utilities
│   │   ├── snapshot_utils.py         #   Helper functions
│   │   └── snapshot_cli.py           #   CLI for snapshot management
│   ├── cascor_constants/             # Configuration constants
│   │   ├── constants.py              #   Master constants file
│   │   ├── constants_activation/     #   Activation function definitions
│   │   ├── constants_candidates/     #   Candidate training parameters
│   │   ├── constants_model/          #   Network architecture parameters
│   │   ├── constants_logging/        #   Logging configuration
│   │   ├── constants_problem/        #   Problem-specific settings
│   │   └── constants_hdf5/           #   Serialization paths
│   ├── cascor_plotter/               # Visualization
│   │   └── cascor_plotter.py
│   ├── log_config/                   # Logging system
│   │   ├── log_config.py             #   LogConfig (logger configuration)
│   │   └── logger/
│   │       └── logger.py             #   Logger wrapper with color support
│   ├── remote_client/                # HTTP client for API interaction
│   │   └── remote_client.py
│   ├── profiling/                    # Performance analysis
│   │   ├── memory.py                 #   Memory profiling
│   │   ├── deterministic.py          #   cProfile profiling
│   │   └── logging_utils.py          #   Profiling log helpers
│   ├── utils/                        # General utilities
│   │   └── utils.py
│   └── tests/                        # Test suite
│       ├── conftest.py               #   Fixtures and configuration
│       ├── pytest.ini                #   pytest settings
│       ├── scripts/
│       │   ├── run_tests.bash        #   Test runner
│       │   └── run_benchmarks.bash   #   Benchmark runner
│       ├── helpers/                  #   Test helper utilities
│       ├── mocks/                    #   Test mock objects
│       ├── unit/                     #   Unit tests (70+ files)
│       │   └── api/                  #   API unit tests (30+ files)
│       ├── integration/              #   Integration tests
│       │   └── api/                  #   API integration tests
│       └── performance/              #   Performance benchmarks
├── conf/                             # Configuration files
│   ├── logging_config.yaml           #   Logging configuration
│   ├── docker-compose.yaml           #   Docker Compose (deprecated, use juniper-deploy)
│   ├── conda_environment.yaml        #   Conda environment definition
│   ├── requirements.txt              #   pip requirements
│   └── *.conf                        #   Shell configuration files
├── docs/                             # User documentation
│   ├── INDEX.md                      #   Documentation navigation
│   ├── DOCUMENTATION_OVERVIEW.md     #   Standards and usage guide
│   ├── DEVELOPER_CHEATSHEET.md       #   Quick-reference for developers
│   ├── USER_MANUAL.md
│   ├── ENVIRONMENT_SETUP.md
│   ├── api/                          #   API reference and schemas
│   ├── ci_cd/                        #   CI/CD pipeline documentation
│   ├── install/                      #   Installation guides
│   ├── overview/                     #   Architecture and constants guides
│   ├── source/                       #   Source code documentation
│   └── testing/                      #   Testing guides
├── notes/                            # Development documentation
│   ├── history/                      #   Archived development notes (50+ files)
│   ├── prompts/                      #   Archived Claude Code session prompts
│   ├── pull_requests/                #   PR documentation
│   ├── setup_config_guides/          #   Setup and configuration guides
│   └── templates/                    #   Document templates
├── scripts/                          # Automation scripts
│   ├── check_doc_links.py            #   Documentation link checker
│   ├── generate_dep_docs.sh          #   Dependency documentation generator
│   └── tls/
│       └── generate_certs.bash       #   TLS certificate generation
├── util/                             # Shell utility scripts
│   ├── juniper_cascor.bash           #   Main launcher
│   ├── run_all_tests.bash            #   Test runner
│   ├── profile_training.bash         #   py-spy profiler
│   ├── get_code_stats.bash           #   Code statistics
│   └── (30+ additional utility scripts)
├── data/                             # Spiral datasets (.pt files)
├── dist/                             # Build artifacts
├── .github/                          # GitHub configuration
│   ├── workflows/
│   │   ├── ci.yml                    #   Main CI pipeline
│   │   ├── golden-regression.yml     #   WS-6 OUT-12 golden / snapshot gate (serial)
│   │   ├── conformance.yml           #   WS-6 OUT-13 model-core conformance gate (serial)
│   │   ├── ci-protocol.yml           #   Path-filtered CI for juniper-cascor-protocol
│   │   ├── ci-cascor-model.yml       #   Path-filtered CI for juniper-cascor-model
│   │   ├── scheduled-tests.yml       #   Scheduled test runs
│   │   ├── publish.yml               #   PyPI publish (juniper-cascor, tag v*)
│   │   ├── publish-protocol.yml      #   PyPI publish (juniper-cascor-protocol)
│   │   ├── publish-cascor-model.yml  #   PyPI publish (juniper-cascor-model)
│   │   ├── lockfile-update.yml       #   Dependency lockfile updates
│   │   ├── codeql.yml                #   CodeQL semantic SAST (Python; soak)
│   │   ├── security-scan.yml         #   Scheduled Bandit + pip-audit
│   │   ├── sequence-safety.yml       #   Per-PR compositional-loss screens (ADVISORY, standalone)
│   │   └── main-verify.yml           #   Post-merge compositional-loss net (catch-up base + stable-title issue)
│   ├── CODEOWNERS                    #   Code ownership rules
│   └── dependabot.yml                #   Dependency update automation
├── .serena/memories/                 # Serena MCP server context
├── pyproject.toml                    # Package metadata and tool configuration
├── Dockerfile                        # Multi-stage production container
├── requirements.lock                 # Pinned dependency lockfile
├── .env.example                      # Environment variable template
├── .pre-commit-config.yaml           # Pre-commit hook configuration
├── AGENTS.md                         # This file
├── CHANGELOG.md                      # Release changelog
├── README.md                         # Project overview
└── LICENSE                           # MIT License
```

---

---

## REST API Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

The juniper-cascor service exposes a versioned REST API at the `/v1/` prefix. All responses use the `ResponseEnvelope` wrapper pattern.

### Health Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/v1/health` | No | Simple status check |
| `GET` | `/v1/health/live` | No | Kubernetes liveness probe |
| `GET` | `/v1/health/ready` | No | Readiness probe with dependency checks |

### Network Management

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/v1/network` | Yes | Create new network with configuration (409 while STARTED / PAUSED / REPLAYING / INVESTIGATING) |
| `GET` | `/v1/network` | Yes | Get current network info |
| `DELETE` | `/v1/network` | Yes | Delete current network (409 while STARTED / PAUSED / REPLAYING / INVESTIGATING) |
| `GET` | `/v1/network/topology` | Yes | Network topology for visualization |
| `GET` | `/v1/network/stats` | Yes | Weight statistics |

### Training Control

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/v1/training/start` | Yes | Start training (inline data or dataset generator) |
| `POST` | `/v1/training/stop` | Yes | Stop training (409 while INVESTIGATING / REPLAYING) |
| `POST` | `/v1/training/pause` | Yes | Pause training |
| `POST` | `/v1/training/resume` | Yes | Resume paused training |
| `POST` | `/v1/training/reset` | Yes | Reset training state |
| `GET` | `/v1/training/status` | Yes | Get current training status |
| `GET` | `/v1/training/params` | Yes | Get training parameters |
| `PATCH` | `/v1/training/params` | Yes | Update runtime parameters (typed `InvalidCandidatePoolError` → 422, other `ValueError` → 404) |
| `POST` | `/v1/training/dataset` | Yes | Stage canopy-dialect dataset config for next start |
| `DELETE` | `/v1/training/dataset` | Yes | Cancel staged dataset config |
| `GET` | `/v1/training/dataset/pending` | Yes | Read staged dataset config (or `null`) |
| `POST` | `/v1/training/dataset/live` | Yes | In-flight live dataset swap (experimental gate) |

### Dataset & Visualization

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/v1/dataset` | Yes | Dataset metadata |
| `GET` | `/v1/dataset/data` | Yes | Dataset arrays for visualization |
| `GET` | `/v1/decision-boundary` | Yes | Decision boundary visualization |

### Snapshots

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/v1/snapshots` | Yes | List network snapshots |
| `POST` | `/v1/snapshots` | Yes | Save network snapshot |
| `GET` | `/v1/snapshots/{snapshot_id}` | Yes | Get metadata for a specific snapshot |
| `POST` | `/v1/snapshots/{snapshot_id}/restore` | Yes | Restore a network from a snapshot |

### Workers

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/v1/workers` | Yes | Remote worker registry status |
| `GET` | `/v1/workers/stats` | Yes | Aggregate worker statistics |
| `GET` | `/v1/workers/{worker_id}` | Yes | Get details for a specific worker |

### Metrics

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/v1/metrics` | No | Prometheus metrics (if enabled) |
| `GET` | `/v1/metrics/history` | Yes | Training metrics history |

### Request/Response Models

| Model | Module | Purpose |
|-------|--------|---------|
| `ResponseEnvelope` | `api.models.common` | Standard response wrapper (status, data, error) |
| `ErrorResponse` | `api.models.common` | Error detail model |
| `NetworkCreateRequest` | `api.models.network` | Network creation parameters |
| `TrainingStartRequest` | `api.models.training` | Training start parameters (data source, config); nested `InlineDataset` rejects train/val length mismatches and half-specified val splits at the request boundary (`422`) |
| `TrainingParamUpdateRequest` | `api.models.training` | Runtime parameter update |
| `ReadinessResponse` | `api.models.health` | Readiness with dependency status |

---

---

## WebSocket Protocol Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

Three WebSocket channels provide real-time communication.

### `/ws/control` -- Training Command Channel

- **Direction**: Client to server
- **Authentication**: X-API-Key header
- **Purpose**: Send training commands (start, stop, pause, resume, reset)
- **Message format**: JSON command messages (must be a JSON **object**)
- **Non-object JSON**: In-band `invalid_message` ack (`Invalid JSON: expected object`); the recv loop continues and the connection stays open (parity with `/ws/training`'s `isinstance(msg, dict)` guard). Malformed JSON (parse failure) still closes with `1003`.
- **Handler**: `api.websocket.control_stream.control_stream_handler()`

### `/ws/training` -- Metrics Stream

- **Direction**: Server to client (broadcast)
- **Authentication**: X-API-Key header
- **Purpose**: Real-time training metrics, epoch-end events, cascade additions, candidate progress
- **Message format**: JSON metrics messages
- **Handler**: `api.websocket.training_stream.training_stream_handler()`

### `/ws/v1/workers` -- Remote Worker Protocol

- **Direction**: Bidirectional (machine-to-machine)
- **Authentication**: API key; rejects browser Origin headers
- **Purpose**: Worker registration, heartbeat, task assignment, result collection
- **Message format**: JSON envelope + binary numpy frames (up to 100MB)
- **Handler**: `api.websocket.worker_stream.worker_stream_handler()`

### WebSocketManager

- Thread-safe broadcasting via `asyncio.run_coroutine_threadsafe()`
- Connection lifecycle management with bounded limit (default: 50)
- Automatic heartbeat/keepalive
- `/ws/control` per-identity admission uses `ws_identity_key` (`src/api/websocket/manager.py`): a truncated (16-char) per-process HMAC-SHA256 of the **stripped** `X-API-Key`. Blank / whitespace-only keys return `None` (anonymous) so they cannot collapse onto one shared SEC-F19 D4b identity bucket -- those callers rely on the global and per-IP caps only.

### Defensive numeric settings (`_numeric_setting`)

`/ws/training` and `/ws/control` read heartbeat (and control idle) timeouts through `_numeric_setting(obj, name, fallback)` before `asyncio.sleep` / `asyncio.wait_for`.

- Attributes: `ws_heartbeat_interval_sec`, `ws_heartbeat_pong_timeout_sec`, and (control only) `ws_control_idle_timeout_sec`.
- Only real `int`/`float` values are accepted; `None`, missing attributes, strings, and `MagicMock` stubs fall back (`30` / `10` / the process `Settings.ws_control_idle_timeout_sec`, default `120`).
- Prevents non-numeric leaks from raising `TypeError` and tearing down the heartbeat/idle loops when tests stub `app.state.settings`.
- Details: [`docs/api/JUNIPER_CASCOR_API_REFERENCE.md`](../docs/api/JUNIPER_CASCOR_API_REFERENCE.md) § Defensive numeric settings.

---

---

## Core Components Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

### CascadeCorrelationNetwork

Main neural network class (`src/cascade_correlation/cascade_correlation.py`).

**Key Methods**:

- `fit()` -- Train the network (full cascade correlation loop)
- `forward()` -- Forward pass through the network
- `train_output_layer()` -- Output layer weight training
- `train_candidates()` -- Candidate unit training (parallel)
- `get_accuracy()` -- Classification accuracy
- `save_to_hdf5()` / `load_from_hdf5()` -- HDF5 serialization
- `create_snapshot()` -- Network state snapshot

**Training Process**:

1. Train output layer on current network (quickprop/backprop)
2. Generate candidate pool (configurable size, default 8)
3. Train candidates in parallel on residual error (decorrelated)
4. Select best candidate (highest correlation, threshold: 0.4)
5. Freeze candidate weights, add to network
6. Repeat until convergence or max hidden units reached

### CandidateUnit

Single candidate hidden neuron (`src/candidate_unit/candidate_unit.py`).

- Configurable activation (tanh, sigmoid, relu)
- Decorrelated training: maximizes correlation with output residuals
- Early stopping with patience parameter
- Picklable via `ActivationWithDerivative` wrapper (solves multiprocessing serialization)

### SpiralProblem

Two-spiral classification problem (`src/spiral_problem/`).

- `SpiralProblem` -- Orchestration and training workflow
- `DataProvider` -- Spiral data generation
- Configurable: number of spirals, rotations, noise

### Configuration

Network configuration via `CascadeCorrelationConfig` dataclass:

```python
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

config = CascadeCorrelationConfig(input_size=2, output_size=2)
network = CascadeCorrelationNetwork(config=config)
network.fit(x_train, y_train, epochs=100)
accuracy = network.get_accuracy(x_test, y_test)
```

Application/API configuration via Pydantic Settings:

- Class: `api.settings.Settings` (inherits `pydantic_settings.BaseSettings`)
- Prefix: `JUNIPER_CASCOR_`
- Loads from environment variables, `.env` file, or Docker secrets
- Cached via `get_settings()`

### Custom Exceptions

Defined in `cascade_correlation_exceptions/`:

- `ConfigurationError` -- Invalid configuration
- `TrainingError` -- Training failures
- `ValidationError` -- Input validation failures

---

---

## Constants Configuration Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

Hierarchical structure in `src/cascor_constants/`:

| Module | Purpose |
|--------|---------|
| `constants.py` | Master constants file |
| `constants_model/` | Network architecture defaults (input_size, output_size, max_hidden_units) |
| `constants_candidates/` | Candidate training parameters (learning_rate, epochs, pool_size, patience) |
| `constants_activation/` | Activation functions and their derivatives |
| `constants_logging/` | Logging formatters, handlers, levels |
| `constants_problem/` | Problem-specific settings (spiral parameters) |
| `constants_hdf5/` | Serialization paths and keys |
| `constants_api/` | API-layer defaults — `constants_api_defaults.py` exposes 49 constants prefixed `_PROJECT_API_*` covering Pydantic field defaults for `NetworkCreateRequest` / `TrainingStartRequest`, lifecycle defaults, middleware body/rate-limit defaults, observability defaults, TLS minimum versions, decision-boundary resolution bounds, juniper-data integration timeouts, and inter-service URL templates |

### Cross-repo alignment (Wave 5 verified)

| Source of truth | Alignment requirement |
|-----------------|-----------------------|
| `cascor_constants/constants_api/constants_api_defaults.py` | `_PROJECT_API_NETWORK_*_DEFAULT` values must equal the corresponding `Field(default=...)` in `src/api/models/network.py` and `src/api/models/training.py` |
| `src/api/workers/protocol.py` `MessageType(StrEnum)` | Canonical wire-protocol message types — `juniper-cascor-worker/constants.py` and `juniper-cascor-client/constants.py` mirror these strings bit-identically |
| `src/api/security.py` `APIKeyHeader(name="X-API-Key")` | The literal `"X-API-Key"` is the canonical header name; all clients must use the same string |
| `src/api/workers/protocol.py` `BinaryFrame.encode/decode` | Header struct format `<I` and dtype encoding `utf-8` define the binary frame layout — workers must use the same struct format and encoding |

---

---

## Documentation Files Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

### User Documentation (`docs/`)

| Directory | Contents |
|-----------|----------|
| `docs/` (root) | INDEX.md, DOCUMENTATION_OVERVIEW.md, DEVELOPER_CHEATSHEET.md, USER_MANUAL.md, ENVIRONMENT_SETUP.md |
| `docs/api/` | API_REFERENCE.md, API_SCHEMAS.md |
| `docs/ci_cd/` | QUICK_START.md, ENVIRONMENT_SETUP.md, MANUAL.md, REFERENCE.md, BRANCH_PROTECTION.md |
| `docs/install/` | QUICK_START.md, ENVIRONMENT_SETUP.md, USER_MANUAL.md, REFERENCE.md |
| `docs/overview/` | CONSTANTS_GUIDE.md |
| `docs/source/` | QUICK_START.md, ENVIRONMENT_SETUP.md, MANUAL.md, REFERENCE.md |
| `docs/testing/` | QUICK_START.md, ENVIRONMENT_SETUP.md, MANUAL.md, REFERENCE.md, SELECTIVE_TESTING_GUIDE.md |

### Development Documentation (`notes/`)

| File | Description |
|------|-------------|
| `notes/ARCHITECTURE_GUIDE.md` | System architecture overview |
| `notes/FEATURES_GUIDE.md` | Feature documentation and usage |
| `notes/API_REFERENCE.md` | API reference (development version) |
| `notes/PRE-DEPLOYMENT_ROADMAP-2.md` | Pre-deployment roadmap (consolidated) |
| `notes/JUNIPER-CASCOR_POST-RELEASE_DEVELOPMENT-ROADMAP.md` | Post-release development roadmap |
| `notes/INTEGRATION_ROADMAP-01.md` | Cascor-Canopy integration tracker |
| `notes/PHASE2_SERVICE_API_PLAN.md` | Phase 2 service API plan |
| `notes/PERFORMANCE_TESTING_PLAN.md` | Performance testing plan |
| `notes/DEPENDENCY_UPDATE_WORKFLOW.md` | Dependency management workflow |
| `notes/CI_PIPELINE_DEVELOPMENT_PLAN.md` | CI/CD pipeline plan |
| `notes/setup_config_guides/` | Setup guides (Serena, Exa, forkserver, CI/CD) |
| `notes/templates/` | Templates (roadmap, issue, PR, release notes) |
| `notes/history/` | Archived development notes (50+ files) |

---

---

## Key Dependencies Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

### Core (always installed)

| Package | Purpose |
|---------|---------|
| `numpy>=1.24.0` | Numerical computations |
| `pydantic>=2.0.0` | Data validation and settings |
| `sentry-sdk>=2.0.0` | Error tracking |
| `python-dotenv>=1.0.0` | Environment file loading |

### ML Extra (`pip install juniper-cascor[ml]`)

| Package | Purpose |
|---------|---------|
| `torch>=2.0.0` | Neural network tensors and operations |
| `h5py>=3.0.0` | HDF5 file serialization |
| `matplotlib>=3.5.0` | Plotting and visualization |
| `PyYAML>=6.0` | YAML configuration parsing |

### API Extra (`pip install juniper-cascor[api]`)

| Package | Purpose |
|---------|---------|
| `fastapi>=0.100.0` | REST API framework |
| `uvicorn[standard]>=0.20.0` | ASGI server |
| `pydantic-settings>=2.0.0` | Environment-based configuration |

### Observability Extra (`pip install juniper-cascor[observability]`)

| Package | Purpose |
|---------|---------|
| `prometheus-client>=0.20.0` | Prometheus metrics |

### Test Extra (`pip install juniper-cascor[test]`)

| Package | Purpose |
|---------|---------|
| `pytest>=6.0` | Test framework |
| `pytest-asyncio>=0.21.0` | Async test support |
| `pytest-cov` | Coverage reporting |
| `pytest-timeout` | Test timeout enforcement |
| `pytest-xdist` | Parallel test execution |
| `coverage[toml]` | Coverage with TOML configuration support |
| `httpx>=0.24.0` | Async HTTP client (API testing) |
| `responses>=0.23.0` | HTTP mocking |
| `dill>=0.3.6` | Extended pickling |
| `psutil>=5.9.0` | System metrics for benchmarks |

### All Extras (`pip install juniper-cascor[all]`)

Installs: `ml`, `api`, `observability`, `test`, `juniper-data`

---

---

## Further Reading

- [`AGENTS.md`](../AGENTS.md) — the resident agent guide this material was relocated from.
- [`docs/INDEX.md`](INDEX.md) — index of this repository's documentation.
- [`docs/DEVELOPER_CHEATSHEET.md`](DEVELOPER_CHEATSHEET.md) — quick-reference card for development.
