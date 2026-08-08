# AGENTS.md - Juniper Cascor Project Guide

**Project**: Juniper Cascade Correlation Neural Network
**Repository**: pcalnon/juniper-cascor
**Author**: Paul Calnon
**License**: MIT License
**Version**: 0.7.0
**Last Updated**: 2026-08-07

---

## Quick Reference

### Conda Environment

> **Required:** Activate the live `JuniperCascor1` conda environment before running any commands. The env name is **versioned** — rebuilds increment the suffix and rename the old env `*-DEPRECATED` (never activate those). Discover yours with `conda env list | grep JuniperCascor`.

```bash
conda activate JuniperCascor1   # live env; see the note above
```

### Essential Commands

```bash
# --- Server (primary operational mode) ---
cd src && python server.py                                        # Development server
uvicorn api.app:create_app --factory --host 0.0.0.0 --port 8200  # Production server

# --- CLI Training (standalone mode) ---
cd src && python main.py                                          # Train on two-spiral problem

# --- Docker ---
docker build -t juniper-cascor:latest .                           # Build container
docker run -p 8200:8200 juniper-cascor:latest                     # Run container

# --- Testing ---
bash src/tests/scripts/run_tests.bash                # Run all tests
cd src && python -m pytest tests/ -v                 # Verbose test output
cd src && python -m pytest tests/unit/ -v            # Unit tests only
cd src && python -m pytest tests/unit/api/ -v        # API unit tests
cd src && python -m pytest tests/integration/ -v     # Integration tests
cd src && python -m pytest tests/performance/ -v     # Performance benchmarks
cd src && python -m pytest tests/ -m "unit" -v       # By marker
cd src && python -m pytest tests/ -k "spiral" -v     # By keyword
cd src && python -m pytest tests/ --run-long         # Include long-running tests
cd src && python -m pytest tests/ --cov=. --cov-report=html:tests/reports/htmlcov  # Coverage

# --- Performance Benchmarks ---
bash src/tests/scripts/run_benchmarks.bash           # Run performance benchmarks

# --- Profiling ---
cd src && python -m cProfile -o profile.out main.py  # cProfile deterministic profiler
cd src && python -m profiling.memory                 # Memory profiling
bash util/profile_training.bash                      # py-spy sampling profiler

# --- Type Checking & Linting ---
cd src && python -m mypy .                           # Type checking
cd src && python -m flake8 .                         # Linting
cd src && python -m black . --check                  # Format check
cd src && python -m isort . --check-only             # Import sort check
cd src && python -m bandit -r . -c ../pyproject.toml # Security scan

# --- Pre-commit ---
pre-commit run --all-files                           # Run all hooks
pre-commit install                                   # Install hooks
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| **Server Configuration** | | |
| `JUNIPER_CASCOR_HOST` | API listen address | `127.0.0.1` |
| `JUNIPER_CASCOR_PORT` | API listen port | `8200` |
| `JUNIPER_CASCOR_LOG_LEVEL` | Log level (TRACE, DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `JUNIPER_CASCOR_LOG_FORMAT` | Log output format (`text` or `json`) | `text` |
| `JUNIPER_CASCOR_CORS_ORIGINS` | CORS allowed origins (JSON list) | `[]` (none) |
| **Authentication & Security** | | |
| `JUNIPER_CASCOR_API_KEYS` | Comma-separated API keys for authentication | `None` (auth disabled) |
| `JUNIPER_CASCOR_API_KEYS_FILE` | Docker-secrets path for API keys; an existing empty/whitespace file yields open auth in the compose `_FILE`-only pattern (`get_secret()` does not fall back to the plain env var) | `None` |
| `JUNIPER_CASCOR_REQUIRE_AUTH` | SEC-F01 intended auth posture: `false` = WARN and run open when keys missing/blank; `true` = refuse boot with `AuthPostureError`. Set `true` wherever secrets are provisioned (composed juniper-deploy). Bypass with `JUNIPER_SKIP_AUTH_POSTURE_CHECK=1` (logged loudly). | `false` |
| `JUNIPER_CASCOR_RATE_LIMIT_ENABLED` | Enable rate limiting | `false` |
| `JUNIPER_CASCOR_RATE_LIMIT_REQUESTS_PER_MINUTE` | Fixed-window budget; keyed `key:<api_key>` when REST auth succeeds, `ip:<client>` when auth is disabled | `60` |
| **WebSocket** | | |
| `JUNIPER_CASCOR_WS_MAX_CONNECTIONS` | Maximum WebSocket connections | `50` |
| `JUNIPER_WS_HEARTBEAT_INTERVAL_SEC` | WebSocket heartbeat interval (`Settings` `AliasChoices` name -- **not** `JUNIPER_CASCOR_`-prefixed) | `30` |
| `JUNIPER_WS_HEARTBEAT_PONG_TIMEOUT_SEC` | WebSocket heartbeat pong/liveness window (same `AliasChoices` binding) | `10` |
| **Observability** | | |
| `JUNIPER_CASCOR_SENTRY_DSN` | Sentry DSN for error tracking | `None` (disabled) |
| `JUNIPER_CASCOR_METRICS_ENABLED` | Enable Prometheus metrics | `false` |
| `JUNIPER_CASCOR_EVAL_METRICS_ENABLED` | Compute C7 scalar evaluation metrics (F1/precision/recall/ROC-AUC) per training step over the eval split; surfaced on `/v1/metrics`, `/v1/metrics/history`, and the WS `metrics` frames. Distinct from `JUNIPER_CASCOR_METRICS_ENABLED` (Prometheus). Set `0`/`false` to disable. | `true` |
| **Auto-Start** | | |
| `JUNIPER_CASCOR_AUTO_START` | Auto-start training on server startup | `true` |
| `JUNIPER_CASCOR_AUTO_DATASET` | Default dataset generator | `spiral` |
| `JUNIPER_CASCOR_AUTO_DATASET_PARAMS` | Dataset generator parameters (JSON) | `{}` |
| `JUNIPER_CASCOR_AUTO_NETWORK` | Network creation parameters (JSON) | `{}` |
| `JUNIPER_CASCOR_AUTO_TRAIN_EPOCHS` | Auto-start training max epochs | `200` |
| `JUNIPER_CASCOR_AUTO_START_DATA_SERVICE` | Auto-start juniper-data companion | `false` |
| `JUNIPER_CASCOR_AUTO_START_DATA_SERVICE_COMMAND` | Command to launch juniper-data service | `python -m juniper_data` |
| `JUNIPER_CASCOR_AUTO_START_CANOPY` | Auto-start juniper-canopy companion | `false` |
| `JUNIPER_CASCOR_AUTO_START_CANOPY_COMMAND` | Command to launch juniper-canopy service | `python -m juniper_canopy` |
| **Remote Workers** | | |
| `JUNIPER_CASCOR_REMOTE_WORKERS_HEARTBEAT_TIMEOUT` | Worker heartbeat stale timeout, seconds (CONC-10 reap) | `30.0` |
| `JUNIPER_CASCOR_REMOTE_WORKERS_TASK_REASSIGNMENT_TIMEOUT` | Fallback reassignment for orphaned in-flight tasks, seconds. All four immediate-requeue paths (reject, soft abort, clean disconnect, dispatch send failure) bypass this timeout -- reaching it means none of them fired. | `120.0` |
| **Legacy / Integration** | | |
| `CASCOR_LOG_LEVEL` | Override log level at runtime | `INFO` |
| `JUNIPER_DATA_URL` | JuniperData service URL | `http://localhost:8100` |
| `JUNIPER_DATA_API_KEY` | API key for JuniperData authentication | (none) |

### Key Entry Points

| File | Purpose |
|------|---------|
| `src/server.py` | **FastAPI server entry point** (primary production mode) |
| `src/main.py` | CLI entry point for standalone two-spiral training |
| `src/api/app.py` | FastAPI application factory with lifespan management |
| `src/api/settings.py` | Pydantic-based application configuration |
| `src/cascade_correlation/cascade_correlation.py` | Core neural network implementation |
| `src/candidate_unit/candidate_unit.py` | Candidate unit for network growth |
| `src/spiral_problem/spiral_problem.py` | Two-spiral problem solver |
| `src/parallelism/task_distributor.py` | Distributed task scheduling (local + remote workers) |
| `src/profiling/` | Profiling infrastructure (memory, deterministic) |
| `src/tests/scripts/run_tests.bash` | Test runner script |
| `src/tests/conftest.py` | Test configuration and fixtures |
| `util/profile_training.bash` | py-spy sampling profiler |

---

## Project Overview

Juniper Cascor is an AI/ML research platform implementing the **Cascade Correlation Neural Network** algorithm from foundational research (Fahlman & Lebiere, 1990). The project provides both a **REST/WebSocket service** for integrated deployment and a **standalone CLI** for direct experimentation.

**Research Philosophy**:

- Transparency over convenience: Algorithms implemented from first principles
- Understanding over abstraction: Full visibility into network behavior
- Modularity and scalability: Designed for research flexibility

**Operational Modes**:

| Mode | Entry Point | Purpose |
|------|-------------|---------|
| **Service** | `src/server.py` | FastAPI REST/WebSocket server for production and integration use |
| **CLI** | `src/main.py` | Standalone training on the two-spiral problem |

---

## Directory Structure

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
│   ├── sequence_safety/              #   Compositional-loss screens (symbol-loss + docs-additions)
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
│   │   ├── security-scan.yml         #   Security scanning
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

## REST API

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

## WebSocket Protocol

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
- Details: [`docs/api/JUNIPER_CASCOR_API_REFERENCE.md`](docs/api/JUNIPER_CASCOR_API_REFERENCE.md) § Defensive numeric settings.

---

## Training Lifecycle Management

The lifecycle system coordinates network training through deterministic state transitions.

| Component | Module | Purpose |
|-----------|--------|---------|
| `TrainingLifecycleManager` | `api.lifecycle.manager` | Central orchestrator (thread-safe via locks) |
| `TrainingStateMachine` | `api.lifecycle.state_machine` | Deterministic state transitions (idle, training, paused, etc.) |
| `TrainingMonitor` | `api.lifecycle.monitor` | Callback-based metrics collection |

**Training Events** (emitted via callbacks):

- `training_start` / `training_end`
- `epoch_end`
- `cascade_add`
- `candidate_progress`

The lifecycle manager wraps the network's training methods non-intrusively to emit events without modifying the core algorithm.

**Staged dataset dialect** -- Canopy stages `dataset_type` values in its own dialect (`spirals` / `moons` / `xor`). `TrainingLifecycleManager._translate_staged_config` aliases them to juniper-data generator keys (`spiral` / `moon`) only at the `_reload_dataset` / live-swap fetch boundary, remaps sample counts (`n_points_per_spiral`, `n_points_per_quadrant`), clamps zero divisors, and strips spiral-only fields for non-spiral generators. Stored configs keep the canopy names.

**C2b progress pairs** -- `output_epoch`/`output_total_epochs` and `candidate_epoch`/`candidate_total_epochs` are zeroed at run start (`_run_training`) and at growth-phase exit (the `training_end` handler after a grow), so UI bars never keep the previous pass's terminal values across those boundaries.

---

## Remote Worker System

Distributed candidate training via WebSocket workers.

| Component | Module | Purpose |
|-----------|--------|---------|
| `WorkerRegistry` | `api.workers.registry` | Worker pool management (register, deregister, health) |
| `WorkerCoordinator` | `api.workers.coordinator` | Task dispatch, monitoring, result collection |
| Worker Protocol | `api.workers.protocol` | Wire format: JSON envelope + binary numpy frames |
| Worker Security | `api.workers.security` | Authentication, token management |
| Worker Audit | `api.workers.audit` | Audit logging for worker operations |

**Worker Lifecycle**:

1. Worker connects via `/ws/v1/workers` with API key
2. Worker sends `register` with `worker_id` + `capabilities` (CPU/GPU, pool size)
3. Server validates `worker_id` against `^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$` (else close `4008`), stores it as `client_name`, and assigns `worker-<12 hex>` as the authoritative registry id (CR-026)
4. Coordinator assigns candidate training tasks
5. Worker returns `task_result` envelopes plus binary numpy frames; the typed parse requires `task_id` / `candidate_id` / `correlation` ∈ [0, 1] / `success` / `epochs_completed`, and rejects JSON `true`/`false` for the int/float fields (`isinstance(True, int)` is true in Python)
6. Heartbeat keepalive (default 30s timeout)
7. Auto-deregistration on heartbeat timeout
8. Task reassignment on worker failure (default 120s timeout)
9. Session teardown calls `coordinator.handle_worker_disconnect(worker_id)` -- which requeues any in-flight task, then deregisters the worker and drops its send callback -- and clears `AnomalyDetector` history for that `worker_id` (`clear_worker`) so reconnect churn cannot grow `_worker_history` without bound or let a recycled id inherit stale anomaly signals

**Result ownership & tensor validation** (`WorkerCoordinator.submit_result` / `WorkerProtocol.validate_tensors`):

- Only the worker recorded in `PendingTask.assigned_worker_id` may complete a task. A peer / stale / wrong `worker_id` is rejected (`False`); the task stays incomplete under the original assignee, and `registry.complete_task(worker_id, success=False)` is charged to the *submitting* worker.
- A result for a task whose `assigned_worker_id` is `None` (the pre-dispatch / post-requeue window) is rejected **without** freeing the submitter's busy slot, so an unrelated active assignment is not wiped.
- `validate_tensors` returns validation errors (does not raise) for non-dict manifest entries, missing `shape` / `dtype`, empty `weights` (`Tensor weights: empty array` instead of a zero-size `np.max`), shape/dtype mismatches, and NaN/Inf / magnitude violations -- keeping the WebSocket result path from crashing the coordinator session.
- A `success=True` result with a missing or empty `weights` tensor is rejected, checked **before** `validate_tensors` so an empty / absent `tensor_manifest` cannot skip the guard. Otherwise `_dispatch_to_remote_workers` rebuilds a `CandidateUnit` at random-init parameters that still carries the worker's claimed correlation, poisoning N-best selection. `success=False` may legitimately omit weights. This guard frees the worker but does **not** requeue -- the task falls back to the 120s sweep.

**In-flight task recovery** -- four distinct immediate-requeue paths on `WorkerCoordinator`, each with its own trigger and log line (do not assume any failure waits 120s):

| Trigger | Method | Worker released via | Log line |
|---------|--------|---------------------|----------|
| Schema / tensor-manifest reject inside `submit_result` | `_reject_and_requeue_task` | `complete_task(..., success=False)` | `Task <id> requeued after rejected result from worker <w>` |
| Soft `task_result` binary-frame abort (text instead of bytes, frame over 100 MB, `BinaryFrame.decode` `ValueError`) | `abort_in_flight_result` | `complete_task(..., success=False)` | `Task <id> requeued after soft result-frame abort from worker <w>` |
| Clean WebSocket close, including mid-binary-frame | `handle_worker_disconnect` | `registry.deregister` + `unregister_send_callback` | `Task <id> requeued after worker <w> disconnect` |
| `task_assign` never delivered (`send_json` / `send_bytes` raised after `get_next_assignment`) | `requeue_after_dispatch_failure` | `complete_task(..., success=False)` | `Task <id> requeued after dispatch send failure to worker <w>` |

- All four clear `assigned_worker_id`, refresh `dispatched_at`, and append to `_unassigned_tasks`; each is a no-op if the task is already completed or already queued.
- `handle_worker_disconnect` is called from the worker-stream session `finally` and holds `self._lock` across the requeue **and** the deregister (CONC-10 lock discipline), so a concurrent `get_next_assignment` cannot land a task on a worker about to disappear.
- `_try_dispatch_task` wraps `send_json` **and** the `send_bytes` frame loop in one `try`, so a partial-send failure is caught too; it logs `Dispatch send failed for task <id> to worker <w> — requeueing` before the rollback. The per-connection `_make_send_callback` only returns `False` on failure and performs no coordinator rollback.
- Receive-site guards reject non-object JSON and a malformed / over-32-entry `tensor_manifest` before binary receive; those header-level rejects return an in-band error only and do **not** requeue (unlike the per-frame guards, which route through `_abort_soft_result_frame`), so recovery there still falls back to disconnect or the 120s timeout.
- Heartbeat / stale-worker reaping (`_check_stale_workers`) fires after `JUNIPER_CASCOR_REMOTE_WORKERS_HEARTBEAT_TIMEOUT` (default 30s); the orphaned-assignment fallback (`_check_task_timeouts`) after `JUNIPER_CASCOR_REMOTE_WORKERS_TASK_REASSIGNMENT_TIMEOUT` (default 120s).
- **Round cancellation** -- `cancel_round` captures every worker still holding an in-flight assignment **before** clearing the coordinator maps, then calls `registry.complete_task(worker_id, success=False)` for each. Without that release the worker stays busy forever: `assign_task` keeps returning `False`, `get_next_assignment` refuses it work, and `_check_task_timeouts` cannot help because pending tracking is already gone.
- Details: [`docs/api/JUNIPER_CASCOR_API_REFERENCE.md`](docs/api/JUNIPER_CASCOR_API_REFERENCE.md) § WS `/ws/v1/workers`.

---

## Middleware Stack

Registered in `src/api/app.py` via successive `app.add_middleware(...)` calls. Starlette/FastAPI middleware runs **LIFO** (last added = first executed), so the outer-to-inner request order when all layers are enabled is:

| Order (outer → inner) | Middleware | Module | Purpose |
|-----------------------|-----------|--------|---------|
| 1 | `RequestIdMiddleware` | `api.observability` | X-Request-ID propagation (always added last, so runs first) |
| 2 | `PrometheusMiddleware` | `api.observability` | Metrics (only when `metrics_enabled`) |
| 3 | `SecurityMiddleware` | `api.middleware` | API key auth + rate limiting (exempt paths) |
| 4 | `SecurityHeadersMiddleware` | `api.middleware` | CSP, HSTS, X-Frame-Options, etc. |
| 5 | `RequestBodyLimitMiddleware` | `api.middleware` | 10 MiB request body limit (CR-024 stream cap) |
| 6 | `CORSMiddleware` | FastAPI/Starlette | CORS headers (only if origins are configured) |

WebSocket upgrade requests are **not** intercepted by `BaseHTTPMiddleware`, so `/ws/*` paths skip the body-limit and security-middleware HTTP paths entirely; they use WebSocket auth and message validation instead.

`RequestBodyLimitMiddleware` uses `Content-Length` only as an early-reject fast path. For `POST`/`PUT`/`PATCH` it must always stream-read with a cumulative byte cap so under-declared headers cannot bypass the limit (CR-024). Cap constant: `_PROJECT_API_MAX_REQUEST_BODY_BYTES`. Tests: `tests/unit/api/test_api_middleware.py::TestRequestBodyLimitMiddleware`.

---

## Security

### API Key Authentication

- Header: `X-API-Key`
- Comparison: HMAC-based (timing-safe)
- When `JUNIPER_CASCOR_API_KEYS` is unset/blank, authentication is disabled
- Docker secrets support: `JUNIPER_CASCOR_API_KEYS_FILE` for container deployments — `get_secret()` returns an existing file's stripped contents with no env fallback; an empty file leaves keys unset in the compose `_FILE`-only pattern

### Boot-Time Auth Posture (SEC-F01)

- Lifespan calls `juniper_service_core.enforce_auth_posture` after the bind guard, before serving (`src/api/app.py`)
- `JUNIPER_CASCOR_REQUIRE_AUTH=false` (default): missing/blank keys → loud WARNING, service continues (bare/dev)
- `JUNIPER_CASCOR_REQUIRE_AUTH=true`: missing/blank keys → CRITICAL + `AuthPostureError` (fail-closed for provisioned stacks)
- Escape hatch: `JUNIPER_SKIP_AUTH_POSTURE_CHECK=1` (logged loudly)

### Rate Limiting

- Fixed-window (thread-safe); optional via `JUNIPER_CASCOR_RATE_LIMIT_ENABLED` (default off)
- Default: 60 requests/minute when enabled (`JUNIPER_CASCOR_RATE_LIMIT_REQUESTS_PER_MINUTE`)
- Keying: `key:<api_key>` when REST API-key auth succeeds; `ip:<client>` when auth is disabled (`API_KEYS` unset / `[]`)
- Ordering: `SecurityMiddleware` authenticates **before** rate limiting -- a 401 never burns a budget slot
- 429 responses carry `Retry-After` plus `X-RateLimit-Limit` / `-Remaining` / `-Reset`, preserved through the middleware's `JSONResponse` rebuild
- Exempt paths (`EXEMPT_PATHS`): `/v1/health`, `/v1/health/live`, `/v1/health/ready`, `/docs`, `/openapi.json`, `/redoc`, `/metrics`, `/metrics/`

### Request Body Limits (CR-024)

- Default cap: 10 MiB (`_PROJECT_API_MAX_REQUEST_BODY_BYTES`)
- Mutating methods only (`POST` / `PUT` / `PATCH`)
- Oversized declared `Content-Length` → HTTP 413; invalid header → HTTP 400
- Stream path always enforces the cumulative cap (do **not** gate on absent `Content-Length` -- that reopens the under-declared bypass)
- Under-limit bodies are cached on `request._body` for downstream handlers (BUG-CC-15)
- WebSocket upgrades are not covered by this middleware (`BaseHTTPMiddleware` skip)

### Security Headers

`SecurityHeadersMiddleware` (`api.middleware`) injects on every HTTP response, including health probes:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy: camera=(), microphone=(), geolocation=()`
- `Content-Security-Policy: default-src 'none'; frame-ancestors 'none'` (constructor override supported)

HSTS (`Strict-Transport-Security: max-age=31536000; includeSubDomains`) is added **only** when the request carries `X-Forwarded-Proto: https`. A TLS terminator that does not forward that header will silently omit HSTS on an otherwise-HTTPS public URL.

### TLS

Certificate generation script: `scripts/tls/generate_certs.bash`

---

## Observability

### Logging

- **Text format** (development): Colored console output with custom formatters
- **JSON format** (production): Structured JSON logging for log aggregation
- **Request IDs**: Propagated via `ContextVar` and `X-Request-ID` header
- **Custom levels**: TRACE, VERBOSE, DEBUG, INFO, WARNING, ERROR, CRITICAL, FATAL

### Prometheus Metrics

When `JUNIPER_CASCOR_METRICS_ENABLED=true`:

- `http_requests_total` -- Request counter by method, path, status
- `http_request_duration_seconds` -- Request latency histogram
- Build info labels
- Endpoint: `GET /v1/metrics`

### Sentry Integration

When `JUNIPER_CASCOR_SENTRY_DSN` is set:

- Automatic error reporting
- Release tracking with API version

### Health Probes

- **Liveness** (`/v1/health/live`): Process is running
- **Readiness** (`/v1/health/ready`): Dependencies available, training system initialized

---

## Core Components

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

## Parallelism and Distribution

### TaskDistributor

Central task scheduler (`src/parallelism/task_distributor.py`):

- **Local-first scheduling**: Assigns tasks to local multiprocessing workers as primary capacity
- **Remote overflow**: Routes excess tasks to registered WebSocket workers
- **Fallback**: Reverts to local execution if remote workers fail or timeout

### Local Multiprocessing

- `multiprocessing.Pool` with forkserver context (safer for CUDA, avoids GIL clone issues)
- Queue-based task distribution via custom manager
- Shared memory support (numpy arrays via `multiprocessing.shared_memory`)
- BLAS thread limits set before imports to prevent contention with workers

### Pickling

Classes implement `__getstate__` and `__setstate__` to handle non-picklable objects (loggers, closures). `ActivationWithDerivative` stores function type by name and reconstructs on unpickling.

---

## Serialization System

### HDF5 Snapshots

Save/load network state including architecture, weights, activation functions, training history, random state, UUID, and checksums.

**CLI Tools**:

```bash
cd src
python -m snapshots.snapshot_cli save network.pkl snapshot.h5
python -m snapshots.snapshot_cli load snapshot.h5
python -m snapshots.snapshot_cli verify snapshot.h5
python -m snapshots.snapshot_cli list ./snapshots/
```

---

## Programming Conventions

**Naming**:

- Constants: Uppercase with underscores, component-prefixed (e.g., `_CASCADE_CORRELATION_NETWORK_*`)
- Classes: PascalCase
- Methods/Functions: snake_case with private methods prefixed by underscore
- Constructor Parameters: Name-mangled style

**File Headers**: Standardized headers with Project, Sub-Project, Application, Author, Version, File Path, Dates, License, Description, References.

**Imports**: Standard library, Third-party, Local application (enforced by isort with `profile = "black"`).

**Type Hints**: Extensive use throughout (e.g., `def forward(self, x: torch.Tensor = None) -> torch.Tensor:`).

**Logging**: Custom system with levels: TRACE, VERBOSE, DEBUG, INFO, WARNING, ERROR, CRITICAL, FATAL.

**Documentation**: Structured docstrings with Description, Args, Returns, Raises, Notes sections.

**Line Length**: 512 for all linters (black, isort, flake8).

**Prometheus Collectors**: Use the canonical helpers from `juniper-observability` (`>=0.2.0`) for any new `Counter` / `Gauge` / `Histogram` / `Summary` / `Info` / `Enum`:

- `register_or_reuse(factory, name, *args, **kwargs)` — adopt-existing on duplicate (the default for almost every call site; preserves accumulated samples across in-process re-init).
- `register_fresh(...)` — drop-and-recreate (only when args genuinely differ; the legacy local `_register_or_reuse` shape).
- `register_info_or_update(name, description, **labels)` — sugar for the `Info` two-step register-then-`.info({...})` pattern.
- `lazy_register_or_reuse(...)` — for the lazy-init-with-`None`-sentinel pattern.

Tests touching these collectors should use `juniper_observability.testing.reset_prometheus_registry`. Existing examples: `src/api/observability.py:_ensure_training_metrics` / `_ensure_ws_metrics` (22+ call sites via the `_register_or_reuse` alias) and `src/api/websocket/control_stream.py:_get_command_counter`. See [the design doc in juniper-ml](https://github.com/pcalnon/juniper-ml/blob/main/notes/observability/JUNIPER_2026-05-05_JUNIPER-ML_REGISTER-OR-REUSE-HELPER-DESIGN.md) for the rationale.

---

## Testing Infrastructure

**Framework**: pytest
**Location**: `src/tests/`

### Test Categories (Markers)

| Marker | Description |
|--------|-------------|
| `unit` | Unit tests for individual components |
| `integration` | Integration tests for full workflows |
| `performance` | Performance and benchmarking tests |
| `slow` | Long-running tests |
| `long` | Long-running correctness tests (use `--run-long`) |
| `gpu` | GPU/CUDA tests |
| `multiprocessing` | Multiprocessing tests |
| `spiral` | Spiral problem tests |
| `correlation` | Correlation calculation tests |
| `network_growth` | Network growth algorithm tests |
| `candidate_training` | Candidate unit training tests |
| `validation` | Input validation tests |
| `accuracy` | Accuracy calculation tests |
| `early_stopping` | Early stopping logic tests |
| `golden` | Golden / snapshot regression (OUT-12; needs `--golden`, serial WS-6 lane) |
| `conformance` | model-core GrowableModel conformance (OUT-13; needs `--conformance`, serial WS-6 lane) |
| `requires_juniper_data` | Tests requiring juniper-data package |

### Test Directory Structure

```text
src/tests/
├── conftest.py                  # Global fixtures
├── pytest.ini                   # pytest configuration
├── scripts/
│   ├── run_tests.bash           # Test runner
│   └── run_benchmarks.bash      # Benchmark runner
├── conformance/                 # WS-6 OUT-13 model-core conformance suite
├── fixtures/golden/             # WS-6 OUT-12 checked-in golden artifacts
├── helpers/
│   ├── assertions.py            # Custom assertion helpers
│   └── utilities.py             # Test utility functions
├── mocks/
│   └── mock_candidate.py        # Mock objects
├── unit/                        # Unit tests (70+ files)
│   ├── test_cascade_correlation_*.py   # CasCor network tests
│   ├── test_candidate_unit_*.py        # Candidate unit tests
│   ├── test_spiral_*.py                # Spiral problem tests
│   ├── test_snapshot_*.py              # Serialization tests
│   ├── test_log_*.py                   # Logging tests
│   ├── test_server_coverage.py         # Server entry point tests
│   ├── test_main_*.py                  # CLI entry point tests
│   └── api/                            # API layer tests (30+ files)
│       ├── test_api_app.py             #   App factory tests
│       ├── test_api_health.py          #   Health endpoint tests
│       ├── test_api_middleware.py       #   Middleware tests
│       ├── test_api_security.py        #   Security tests
│       ├── test_api_settings.py        #   Settings tests
│       ├── test_lifecycle_*.py         #   Lifecycle tests
│       ├── test_websocket_*.py         #   WebSocket tests
│       ├── test_worker_*.py            #   Worker system tests
│       ├── test_*_route*.py            #   Route handler tests
│       └── test_monitoring_hooks.py    #   Monitoring tests
├── integration/                 # Integration tests
│   ├── test_serialization.py
│   ├── test_spiral_problem.py
│   ├── test_comprehensive_serialization.py
│   ├── test_juniper_data_e2e.py
│   └── api/                     # API integration tests
│       ├── conftest.py
│       ├── test_api_full_lifecycle.py
│       └── test_websocket_streaming.py
└── performance/                 # Performance benchmarks
    ├── conftest.py
    ├── test_baselines.py
    ├── test_concurrency_scaling.py
    ├── test_endtoend_profiling.py
    ├── test_micro_autograd.py
    ├── test_micro_candidate.py
    ├── test_micro_correlation.py
    ├── test_micro_forward_pass.py
    ├── test_micro_output_training.py
    └── test_shared_memory.py
```

### Test Output

- HTML Coverage: `src/tests/reports/htmlcov/index.html`
- XML Coverage: `src/tests/reports/coverage.xml`
- JUnit XML: `src/tests/reports/junit.xml`

### Coverage

Reproduce the CI coverage gate locally (full suite):

```bash
make coverage                 # convenience wrapper
bash util/run_coverage.bash   # source of truth (mirrors .github/workflows/ci.yml)
```

Gate: 80% aggregate (override with `COVERAGE_FAIL_UNDER=<n>`). Coverage runs in parallel mode with a custom data_file (see pyproject `[tool.coverage.run]`); the script reproduces the CI sequence exactly. Full suite by design; for a narrower run use plain `pytest`.

---

## CI/CD Pipelines

### GitHub Actions Workflows

| Workflow | File | Triggers | Purpose |
|----------|------|----------|---------|
| CI/CD Pipeline | `.github/workflows/ci.yml` | Push (main, develop, feature/**, fix/**), PR, dispatch | Pre-commit, unit tests, integration tests, security scanning |
| Golden Regression (WS-6) | `.github/workflows/golden-regression.yml` | Push `main`, PR `main`/`develop`, dispatch | Serial OUT-12 golden / snapshot regression (Python 3.13 + torch 2.11.0) |
| Conformance (WS-6) | `.github/workflows/conformance.yml` | Push `main`, PR `main`/`develop`, dispatch | Serial OUT-13 model-core GrowableModel conformance |
| CI — protocol | `.github/workflows/ci-protocol.yml` | Path-filtered on `juniper-cascor-protocol/**`, dispatch | Package tests + build/`twine check` |
| CI — cascor-model | `.github/workflows/ci-cascor-model.yml` | Path-filtered on `juniper-cascor-model/**`, dispatch | Package tests (incl. drift-guard) + build/`twine check` |
| Scheduled Long Tests | `.github/workflows/scheduled-tests.yml` | Cron schedule (nightly), dispatch | Slow and long-running correctness tests |
| Publish | `.github/workflows/publish.yml` | Release (`v*`) | PyPI publish for `juniper-cascor` (TestPyPI → verify → PyPI) |
| Publish protocol | `.github/workflows/publish-protocol.yml` | Release (`juniper-cascor-protocol-v*`) + `workflow_dispatch` | PyPI publish for `juniper-cascor-protocol` |
| Publish model | `.github/workflows/publish-cascor-model.yml` | Release (`juniper-cascor-model-v*`) + `workflow_dispatch` | PyPI publish for `juniper-cascor-model` |
| Lockfile Update | `.github/workflows/lockfile-update.yml` | Push to dependabot/** branches | Dependency lockfile refresh |
| Security Scan | `.github/workflows/security-scan.yml` | Schedule/dispatch | Gitleaks, Bandit, pip-audit |
| Sequence Safety (Advisory) | `.github/workflows/sequence-safety.yml` | PR (`main`/`develop`) | Per-PR symbol-loss + docs-deletion screens over base..HEAD (ADVISORY, standalone, never a required check) |
| Post-Merge Main Verification | `.github/workflows/main-verify.yml` | Push `main`, dispatch | Bypass-proof post-merge compositional-loss net (catch-up base; stable-title tracking issue on failure) |

### Lockfile Update PAT Gate

`lockfile-update.yml` checks out and pushes with `CROSS_REPO_DISPATCH_TOKEN` so the lock commit re-triggers CI. Dependabot runs use the Dependabot secret store — a PAT registered only under Actions secrets is empty there.

| Condition | Behavior |
|-----------|----------|
| PAT present | Auto-regen + `[dependabot skip]` push |
| PAT absent + `dependabot[bot]` | Green no-op (`::notice::`); **Lockfile Freshness** in `ci.yml` still blocks stale locks |
| PAT absent + other actor | Hard fail (secret misconfiguration) |

Optional: register the same PAT under **Settings → Secrets → Dependabot** to restore Dependabot auto-regen. Operator narrative: [`notes/DEPENDENCY_UPDATE_WORKFLOW.md`](notes/DEPENDENCY_UPDATE_WORKFLOW.md).

### CI Pipeline Jobs (ci.yml)

- Pre-commit hooks (black, isort, flake8, mypy, bandit)
- Unit tests with coverage enforcement
- Integration tests
- Security scanning (Gitleaks, Bandit SARIF, pip-audit)
- Lockfile Freshness (`lockfile-check`) — required quality-gate input
- Dependency caching for performance
- Concurrency: one pipeline per branch, cancel-in-progress

### Additional GitHub Configuration

- `.github/CODEOWNERS` -- Code ownership rules
- `.github/dependabot.yml` -- Automated dependency updates

### Sequence Safety (Compositional-Loss Net)

Ported from juniper-ml (ml#873 / #880 / #928; the flood-remediation program) after the 2026-08-05 storm triage found *compositional loss* — a silently deleted def/class/method, a gutted body, or a net docs-section deletion that no per-PR check sees because a deleted test cannot fail — to be cascor's one remaining gap. Two pure-stdlib git-diff screens live in `util/sequence_safety/`:

- `symbol_loss_check.py` — AST symbol inventory of BASE vs HEAD over `src/**/*.py` (includes `src/tests/**`); FAIL on a LOST / WEAKENED / DUPLICATED def, with a qualified-name / body-similarity relocation downgrade. Escape hatch: an `Allow-Symbol-Loss: <qualified.symbol>` commit trailer. (`@property`/`@x.setter` accessor pairs are disambiguated so they are never a false DUPLICATED — a cascor-src adaptation over the juniper-ml original.)
- `docs_additions_check.py` — markdown deletion-magnitude screen over `AGENTS.md` + `docs/**` + `notes/**`; FAIL on a deleted heading or a run of ≥ N consecutive deleted lines, WARN on small in-place swaps. Escape hatch: an `Allow-Docs-Rewrite: <path>` commit trailer.

Everything is **ADVISORY** — neither workflow is a required status check and this makes **no branch-ruleset change**.
`sequence-safety.yml` surfaces findings per-PR at review (with WARN-only `allow-symbol-loss` / `docs-rewrite` label hatches); `main-verify.yml` is the bypass-proof post-merge net that fires on every merge to `main` (catch-up base sweeps any `[skip ci]` window; a stable-title tracking issue is upserted on failure).
v1 defers the post-merge regression battery (cascor's suite is heavy) and Slack notify (no webhook secret) — see the workflow header comments.
The screens are behaviourally identical to juniper-ml's, whose `tests/test_symbol_loss_check.py` + `tests/test_docs_additions_check.py` remain the authoritative regression suite.

---

## Deployment

### Docker

Multi-stage Dockerfile (`Dockerfile`):

```bash
# Build
docker build -t juniper-cascor:latest .

# Run
docker run -p 8200:8200 \
  -e JUNIPER_DATA_URL=http://localhost:8100 \
  -e JUNIPER_CASCOR_API_KEYS=your-key-here \
  juniper-cascor:latest
```

- **Base image**: `python:3.14-slim`
- **CPU-only PyTorch** (avoids ~4GB CUDA dependency)
- **Non-root user** in production stage

### Docker Compose

Per-service compose file in `conf/docker-compose.yaml` is **deprecated**. Use the unified orchestration in `juniper-deploy`:

```bash
cd ../juniper-deploy && make up
```

### Service Ports

| Service | Port | Health Endpoint |
|---------|------|-----------------|
| juniper-cascor | 8200 | `/v1/health` |

### Service Launcher

When running outside containers, juniper-cascor can auto-start companion services (`src/api/service_launcher.py`):

- `JUNIPER_CASCOR_AUTO_START_DATA_SERVICE=true` -- Start juniper-data
- `JUNIPER_CASCOR_AUTO_START_CANOPY=true` -- Start juniper-canopy

The launcher probes health endpoints before declaring readiness.

**Failed-start cleanup** (fail-closed registry / FD hygiene):

- `ManagedService.terminate` always closes the companion log handle in a `finally` (even when wait/kill raises).
- Health-probe exceptions are treated as unhealthy; the subprocess is terminated and removed from `_active_services`.
- If `terminate()` itself raises after a failed health check, the launcher still drops the stale `_active_services` entry (the removal is in a `finally`), so atexit / shutdown cannot chase an orphaned companion or leave port conflicts.

---

## Key Dependencies

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

## Constants Configuration

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

## Documentation Files

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

## MCP Server Availability

### Serena

The project includes Serena MCP server configuration in `.serena/memories/`:

| File | Purpose |
|------|---------|
| `project_overview.md` | Project context for Serena |
| `code_style_conventions.md` | Coding standards context |
| `suggested_commands.md` | Suggested development commands |
| `task_completion_checklist.md` | Task completion guidelines |

Serena provides semantic code analysis tools for navigating the codebase, finding symbols, and understanding architecture through symbolic tools rather than raw file reads.

---

## Known Issues and Workarounds

- **Logger Pickling**: Loggers excluded from `__getstate__` for multiprocessing
- **GPU Support**: Tests disable GPU by default; use `--gpu` flag for GPU tests
- **Long-Running Tests**: Skipped by default; use `--run-long` to run them
- **Random Reproducibility**: Set `random_seed` in config for deterministic training
- **BLAS Thread Contention**: BLAS thread limits (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`) set before imports to prevent contention with multiprocessing workers

---

## Development Workflow

**Adding Features**:

1. Create feature in appropriate module
2. Add constants to `src/cascor_constants/`
3. Add tests in `src/tests/unit/` (or `src/tests/unit/api/` for API features)
4. Update documentation in `notes/` or `docs/`
5. Run tests: `cd src && python -m pytest tests/ -v`
6. Run linting: `pre-commit run --all-files`

**Adding Tests**:

1. Create test file following `test_<feature>.py` naming
2. Use appropriate markers (`@pytest.mark.unit`, `@pytest.mark.integration`, etc.)
3. Use fixtures from `conftest.py`
4. Follow Arrange-Act-Assert pattern
5. For API tests, use `httpx.AsyncClient` with the `TestClient` pattern

---

## Performance Considerations

**Training Tips**:

- Optimize `candidate_pool_size` for CPU core count
- Use N-best candidate selection for faster convergence
- Tune `patience` for speed vs. accuracy tradeoff
- Configure BLAS thread limits for multiprocessing environments

**Benchmarks**:

Performance micro-benchmarks in `src/tests/performance/` cover:

- Forward pass latency
- Candidate training throughput
- Correlation calculation
- Output training
- Concurrency scaling
- Shared memory performance
- Autograd overhead
- End-to-end profiling

---

## Security Notes

- No secrets or API keys in codebase
- Sensitive files excluded via `.gitignore`
- `.env.example` provided as template; `.env` is gitignored
- Docker secrets supported for container deployments
- SOPS encryption available (`.sops.yaml`, `.env.enc`)
- Log files may contain training data -- handle appropriately
- Pre-commit hooks include `bandit` security scanner
- CI pipeline includes Gitleaks and pip-audit scans

---

## Script Placement

**Permanent utilities** live in `util/`. **Single-use / temporary / unfinished scripts** go in `util/ad-hoc/` (create on first use). See [`util/ad-hoc/README.md`](util/ad-hoc/README.md) for the per-script header / lifecycle conventions.

`/tmp/` is **prohibited** as the home for any script that produces, modifies, or analyzes repository content. `/tmp/` is reaped when sessions / sandboxes / containers end, and scripts placed there are lost (irrecoverable). `/tmp/` remains fine as a scratch *workspace* for intermediate artifacts the script itself creates and reads — the prohibition is on script *source files*.

This is an ecosystem-wide rule restated in the parent `Juniper/AGENTS.md` "Cross-Project Conventions" section. Motivating incident: irrecoverable loss of `phase4_consolidate.py` and `v2_citation_validate.py` from the juniper-ml requirements-snapshot effort.

---

## Worktree Procedures (Mandatory -- Task Isolation)

Git worktrees allow multiple branches to be checked out simultaneously in separate directories. For the Juniper ecosystem, all worktrees are centralized in **`/home/pcalnon/Development/python/Juniper/worktrees/`** using standardized naming convention: `<repo-name>--<branch-name>--<YYYYMMDD-HHMM>--<short-hash>`

**Full procedures**:

- **`notes/WORKTREE_SETUP_PROCEDURE.md`** -- Creating a worktree
- **`notes/WORKTREE_CLEANUP_PROCEDURE_V2.md`** -- Merging, removing, pushing (V2 fixes CWD-trap bug)

**Rules**:

- Centralized location only
- Clean before starting
- Push before merging
- Prune after cleanup
- No stale worktrees

---

## Thread Handoff (Mandatory -- Replaces Thread Compaction)

Critical instruction: Thread handoff MUST replace thread compaction when context limits approach.

**Trigger threshold**: Initiate handoff at **95% to 99%** of compaction threshold (within 1-5% of compaction threshold).

**Additional triggers**:

| Condition | Indicator |
|-----------|-----------|
| Context saturation | 15+ tool calls or 5+ file edits |
| Phase boundary | Logical work phase complete |
| Degraded recall | Re-reading files or re-asking questions |
| Multi-module transition | Moving between major components |
| User request | User says "hand off", "new thread", or similar |

**Do NOT handoff** when:

- Task nearly complete (< 2 remaining steps)
- Current thread still sharp
- Work tightly coupled and splitting loses critical state

**How to execute**:

1. Checkpoint work done, remains, discovered, files in play
2. Compose concise, actionable handoff summary
3. Present to user and recommend new thread
4. Include verification commands
5. State git status

**Rules**:

- Not optional -- every Claude Code instance must follow
- Handoff early, not late
- Don't duplicate CLAUDE.md content
- Be specific: include file paths, decisions, test status

---

## Contact

For questions, refer to:

- User documentation in `docs/`
- Development documentation in `notes/`
- Test examples in `src/tests/`
- Constants definitions in `src/cascor_constants/`
- API reference in `docs/api/API_REFERENCE.md`
