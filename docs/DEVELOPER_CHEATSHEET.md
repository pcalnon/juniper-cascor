# Developer Cheatsheet — juniper-cascor

**Version**: 1.0.2  |  **Date**: 2026-08-05  |  **Project**: juniper-cascor

---

## Common Commands

**The following commands launch a full set of Juniper Project services: start services in the order listed below:**

- juniper-data: cd /home/pcalnon/Development/python/Juniper/juniper-data && conda activate JuniperData && pip install -e ".[all]" && PYTHON_GIL=0 uvicorn juniper_data.api.app:app --host 0.0.0.0 --port 8100
- juniper-cascor: cd /home/pcalnon/Development/python/Juniper/juniper-cascor/src && conda activate JuniperCascor1 && JUNIPER_CASCOR_PORT=8201 python server.py
- Public/container bind only when fronted: if `JUNIPER_CASCOR_HOST` is non-loopback (for example `0.0.0.0`), also set a bind attestation — `JUNIPER_CASCOR_LOOPBACK_PUBLISH_ATTESTED=true` after verifying a loopback-only host-publish, or `JUNIPER_CASCOR_AUTH_PROXY_ATTESTED=true` after verifying a fronting authenticating reverse proxy fronts the port.
- juniper-canopy: cd /home/pcalnon/Development/python/Juniper/juniper-canopy/src && conda activate JuniperCanopy1 && CASCOR_SERVICE_URL="<http://localhost:8201>" uvicorn main:app --host 0.0.0.0 --port 8050

> **Conda env naming:** the live envs are **versioned** — `JuniperCascor1`, `JuniperCanopy1` (the bare `JuniperCascor` / `JuniperCanopy` are now `*-DEPRECATED` with a broken toolchain; `JuniperData` is unversioned). Discover yours with `conda env list | grep Juniper<App>` and use that name; rebuilds increment the suffix.

**General list of useful Commands:**

| Task                                    | Command                                                                                                     |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------------|
| Run application, training only          | `cd src && python main.py`                                                                                  |
| Run API server and cascor training loop | cd src && JUNIPER_CASCOR_PORT=8201 uvicorn api.app:app                                                      |
|                                         | cd src && JUNIPER_CASCOR_PORT=8201 python server.py                                                         |
|                                         | Note: server.py — starts uvicorn on settings.host:settings.port (default 127.0.0.1:8200)                    |
| Run all tests                           | `cd src/tests && bash scripts/run_tests.bash`                                                               |
| Unit / integration tests                | `bash scripts/run_tests.bash -u` / `bash scripts/run_tests.bash -i`                                         |
| Tests with coverage                     | `bash scripts/run_tests.bash -v -c`                                                                         |
| Run by marker                           | `python -m pytest -m "spiral" -v`                                                                           |
| Long-running tests                      | `python -m pytest --run-long`                                                                               |
| Benchmarks (quiet)                      | `cd src/tests/scripts && bash run_benchmarks.bash -q -n 10`                                                 |
| Pre-commit                              | `pre-commit run --all-files`                                                                                |
| Type check                              | `cd src && python -m mypy cascade_correlation/ candidate_unit/ --ignore-missing-imports`                    |
| Lint / format / isort                   | `python -m flake8 . --max-line-length=512` / `python -m black --check .` / `python -m isort --check-only .` |
| Security scan                           | `bandit -r src/`                                                                                            |

> See: [AGENTS.md](../AGENTS.md) | [Configuration Reference](install/REFERENCE.md)

---

## Service Operations

```bash
conda activate JuniperCascor1 && cd juniper-cascor && pip install -e ".[all]"
cd src && python main.py                                       # native start
curl -s http://localhost:8200/v1/health | python -m json.tool  # health
docker compose --profile full up -d                            # Docker start
```

> See: [Ecosystem Service Ports](https://github.com/pcalnon/juniper-ml/blob/main/docs/REFERENCE.md#service-ports) | [juniper-deploy AGENTS.md](https://github.com/pcalnon/juniper-deploy/blob/main/AGENTS.md)

---

## API Endpoints

**Add an endpoint:** Create route under `/v1`, register in `app.py`, add client method in `juniper-cascor-client`.

**WebSocket:** `/ws/training` (metrics), `/ws/control` (commands), `/ws/v1/workers` (remote workers). Over-cap connections close with `1013`; update `ws_client.py`, canopy, and worker clients when adding or changing channels.

### Training Lifecycle Quick Paths

| Endpoint                       | Method | Purpose                                          |
|--------------------------------|--------|--------------------------------------------------|
| `/v1/training/start`           | POST   | Start async training (inline or generated data)  |
| `/v1/training/stop`            | POST   | Request stop                                     |
| `/v1/training/pause`           | POST   | Pause active training                            |
| `/v1/training/resume`          | POST   | Resume paused training                           |
| `/v1/training/reset`           | POST   | Reset lifecycle state and metric buffer          |
| `/v1/training/status`          | GET    | Combined FSM + monitor + training_state snapshot |
| `/v1/metrics`                  | GET    | Latest metric                                    |
| `/v1/metrics/history?count=50` | GET    | Recent metric history                            |

`training_state` now carries phase-granularity fields useful for UI/progress bars:

- `phase_detail` (`training_output`, `training_candidates`, `adding_candidate`, or empty)
- `grow_iteration`, `grow_max`
- `best_correlation`
- `candidates_trained`, `candidates_total`
- `phase_started_at` (ISO timestamp)
- `candidate_epoch`, `candidate_total_epochs` (real-time per-candidate progress)

Metrics nuance:

- `/v1/metrics/history` and `/ws/training` include callback-driven output-phase points (epoch 1, every 25 epochs, final epoch).
- `accuracy` can be `null` for those output-phase callback emissions.
- **C7:** rows always include nullable `f1`/`precision`/`recall`/`roc_auc` (populated on the terminal `training_step` row per drain). `GET /v1/metrics` also returns an `eval_metrics` metadata block. Toggle with `JUNIPER_CASCOR_EVAL_METRICS_ENABLED` (default on; not Prometheus).
- `/ws/training` can also emit `candidate_progress` messages (epoch 1, every 50 epochs, final epoch per candidate).
- Fresh `/ws/training` connects receive `initial_status`, `state`, and `initial_metrics`; resume requests use `{"type":"resume","data":{"last_seq":...,"server_instance_id":...}}` and replay only buffered broadcasts with higher `seq`.
- `/ws/control` rate limiting returns an in-band `command_response` with `status:"rate_limited"` and keeps the socket open; it does not close on normal command throttling.

**Middleware** (outermost first): CORS -> Security -> Prometheus -> RequestId. **Models:** Pydantic (API), dataclasses (config).

> See: [API Reference](api/API_REFERENCE.md) | [API Schemas](api/API_SCHEMAS.md)

---

## Configuration

**Add a constant:** Add to the appropriate module in `src/cascor_constants/`, follow `_UPPER_SNAKE` naming, import via `cascor_constants.constants`. Modules: `constants_model/` (architecture), `constants_candidates/` (pool/training), `constants_activation/` (functions), `constants_logging/` (levels), `constants_problem/` (defaults), `constants_hdf5/` (serialization).

> See: [Constants Guide](overview/CONSTANTS_GUIDE.md)

### Environment Variables

| Variable                         | Default                 | Description                            |
|----------------------------------|-------------------------|----------------------------------------|
| `CASCOR_LOG_LEVEL`               | `INFO`                  | Log level override (set before import) |
| `JUNIPER_DATA_URL`               | `http://localhost:8100` | JuniperData service URL                |
| `JUNIPER_DATA_API_KEY`           | --                      | API key for JuniperData                |
| `JUNIPER_CASCOR_HOST`            | `127.0.0.1`             | Bind host for the service; non-loopback requires a bind attestation (see the two flags below) |
| `JUNIPER_CASCOR_PORT`            | `8200`                  | Bind port for the service              |
| `JUNIPER_CASCOR_LOOPBACK_PUBLISH_ATTESTED` | `false`       | Bind attestation for non-loopback binds: port reachable only via a loopback-only host publish |
| `JUNIPER_CASCOR_AUTH_PROXY_ATTESTED` | `false`             | Bind attestation for non-loopback binds: a fronting authenticating reverse proxy terminates access |
| `JUNIPER_CASCOR_CORS_ORIGINS`    | `[]`                    | Allowed CORS origins                   |
| `JUNIPER_CASCOR_API_KEYS`        | --                      | API keys for authentication            |
| `JUNIPER_CASCOR_API_KEYS_FILE`   | --                      | Docker-secrets path for keys; existing empty file → open auth (compose `_FILE`-only pattern) |
| `JUNIPER_CASCOR_REQUIRE_AUTH`    | `false`                 | SEC-F01: `true` = refuse boot when keys missing/blank; default WARN-and-run-open |
| `JUNIPER_SKIP_AUTH_POSTURE_CHECK` | unset                 | Escape hatch skipping the boot posture check (`1` to bypass; logged loudly) |
| `JUNIPER_CASCOR_LOG_FORMAT`      | --                      | Set to `json` for JSON logging         |
| `JUNIPER_CASCOR_SENTRY_DSN`      | --                      | Sentry DSN                             |
| `JUNIPER_CASCOR_METRICS_ENABLED` | `false`                 | Enable Prometheus metrics              |
| `JUNIPER_CASCOR_EVAL_METRICS_ENABLED` | `true`             | C7 F1/precision/recall/ROC-AUC on `/v1/metrics`, history, and WS `metrics` (not Prometheus) |
| `JUNIPER_CASCOR_WS_MAX_CONNECTIONS_GLOBAL` | `200`        | Stack-wide WebSocket cap across training, control, and worker sockets |
| `JUNIPER_CASCOR_WS_MAX_CONNECTIONS_PER_IDENTITY` | `5`     | `/ws/control` cap per API-key identity |
| `JUNIPER_CASCOR_WS_MAX_CONNECTIONS_PER_IP` | `5`          | Per-source-IP cap; DoS dampening only and shared behind Docker NAT |
| `JUNIPER_CASCOR_WS_CONTROL_ALLOWED_ORIGINS` | `http://localhost:8050,http://127.0.0.1:8050,https://localhost:8050,https://127.0.0.1:8050` | `/ws/control` Origin allowlist. Accepts JSON-array or comma-CSV. Empty string disables (opt-out). For docker compose, add `http://juniper-canopy:8050` so canopy's `ControlStreamSupervisor` can connect. |
| `JUNIPER_WS_REPLAY_BUFFER_SIZE` | `1024` | `/ws/training` broadcast replay buffer size. `0` disables resume replay. |
| `JUNIPER_WS_INITIAL_METRICS_COUNT` | `100` | Recent metrics sent as `initial_metrics` on fresh `/ws/training` connect. `0` disables the automatic burst; clients can still send `subscribe_metrics`. |
| `JUNIPER_WS_SEND_TIMEOUT_SECONDS` | `0.5` | Per-client WebSocket send timeout before a slow consumer is dropped from fan-out. |
| `JUNIPER_WS_MAX_MESSAGE_SIZE_BYTES` | `60000` | Serialized JSON threshold for `chunked_message` envelopes. `0` disables chunking (tests only; oversized frames may be dropped by intermediaries). |
| `JUNIPER_WS_CHUNK_PAYLOAD_SIZE_BYTES` | `32000` | Payload slice size for each `chunked_message`. |
| `JUNIPER_CASCOR_WS_MAX_CONNECTIONS` | `50` | Global WebSocket connection cap, including pending `/ws/training` resume handshakes. |
| `JUNIPER_CASCOR_WS_HEARTBEAT_INTERVAL_SEC` | `30` | Training/control heartbeat ping interval. |
| `JUNIPER_CASCOR_WS_HEARTBEAT_PONG_TIMEOUT_SEC` | `10` | Training/control pong timeout before heartbeat close. |

---

## Training Lifecycle and Algorithm Parameters

1. **Output training** -- Train output weights on current architecture
2. **Candidate training** -- Train pool of candidates to maximize correlation with residual error
3. **N-best selection** -- Select candidate with highest |correlation| (above `correlation_threshold`)
4. **Install** -- Freeze candidate, add to network, re-initialize output weights
5. **Repeat** until `target_accuracy` reached or `max_hidden_units` exhausted

```python
config = CascadeCorrelationConfig(
    input_size=2, output_size=2, learning_rate=0.01,
    candidate_pool_size=50, max_hidden_units=50,
    correlation_threshold=0.0005, patience=50, target_accuracy=0.999,
)
network = CascadeCorrelationNetwork(config=config)
```

| Parameter               | Default  | Notes                    |
|-------------------------|----------|--------------------------|
| `learning_rate`         | `0.01`   | Output layer LR          |
| `candidate_pool_size`   | `50`     | Tune to CPU core count   |
| `candidate_epochs`      | `600`    | More = better candidates |
| `correlation_threshold` | `0.0005` | Minimum for selection    |
| `patience`              | `50`     | Early stopping epochs    |
| `max_hidden_units`      | `50`     | Network growth limit     |
| `target_accuracy`       | `0.999`  | Termination goal         |

> See: [API Reference](api/API_REFERENCE.md) | [AGENTS.md](../AGENTS.md#core-components)

---

## HDF5 Checkpoint Format

```python
serializer = CascadeHDF5Serializer()
serializer.save_network(network, "./snapshots/network.h5", include_training_state=True)
loaded = serializer.load_network("./snapshots/network.h5")
```

```bash
python -m snapshots.snapshot_cli verify snapshot.h5            # integrity
python -m snapshots.snapshot_cli list ./snapshots/             # list all
python -m snapshots.snapshot_cli cleanup ./snapshots/ --keep 5 # prune old
```

| HDF5 Group     | Contents                                                 |
|----------------|----------------------------------------------------------|
| `meta`         | UUID, version, timestamps, checksums                     |
| `arch`         | Input/output sizes, hidden units, activation functions   |
| `params`       | Trained weights and biases for all layers                |
| `history`      | Training history, loss, accuracy per epoch               |
| `random_state` | Python, NumPy, PyTorch RNG states (deterministic resume) |

Compression: gzip level 4 (default). Performance: Save <2s, Load <3s, Verify <200ms (100 units).

> See: [Constants Guide -- Serialization](overview/CONSTANTS_GUIDE.md#serialization-constants)

---

## Testing

**Coverage gate:** 80%. Reports: `src/tests/reports/htmlcov/`, `coverage.xml`, `junit.xml`.

| Marker            | Description           | Marker                        | Description                |
|-------------------|-----------------------|-------------------------------|----------------------------|
| `unit`            | Individual components | `spiral`                      | Spiral problem             |
| `integration`     | Full workflows        | `correlation`                 | Correlation calculation    |
| `performance`     | Benchmarking          | `network_growth`              | Growth algorithm           |
| `slow`            | Long-running          | `candidate_training`          | Candidate training         |
| `long`            | Needs `--run-long`    | `validation`                  | Input validation           |
| `gpu`             | Needs GPU/CUDA        | `accuracy` / `early_stopping` | Accuracy / stopping        |
| `multiprocessing` | Multi-process tests   | `requires_juniper_data`       | Needs juniper-data service |
| `golden`          | Needs `--golden`      | `conformance`                 | Needs `--conformance`      |

**WS-6 tip:** Golden and conformance lanes are serial-only (no xdist), pin Python 3.13 + torch 2.11.0 in CI, and require `--golden` / `--conformance` plus `--slow --integration`. Reproduce with `CASCOR_NUM_PROCESSES=1` and single-thread BLAS. Regenerate goldens only with `GOLDEN_CAPTURE=1` after intentional behavior change — see `src/tests/fixtures/golden/README.md` and [CI Quick Start § WS-6](ci_cd/QUICK_START.md#ws-6-gates-golden--conformance).

**API version wiring (BUG-CC-04):** assert `app.version`, `/v1/health` `version`, and `set_build_info` against `api.app._API_VERSION` (`importlib.metadata.version("juniper-cascor")`) — never a pinned `"0.x.y"`. Fixture-only model construction may still use literals. `ResponseEnvelope.meta.version` currently uses `api.models.common._API_VERSION` (literal; may lag).

> See: [Testing Quick Start](testing/QUICK_START.md) | [Testing Reference](testing/REFERENCE.md) | [API version assertions](testing/MANUAL.md#api-version-assertions-bug-cc-04)

---

## Logging and Observability

Levels: TRACE(5) -> VERBOSE(7) -> DEBUG(10) -> INFO(20) -> WARNING(30) -> ERROR(40) -> CRITICAL(50) -> FATAL(60). Set `CASCOR_LOG_LEVEL` before import: `export CASCOR_LOG_LEVEL=DEBUG` or inline `CASCOR_LOG_LEVEL=WARNING python main.py`.

**Prometheus metrics:** Namespace `juniper_cascor_*`. Pattern: `juniper_cascor_<subsystem>_<name>_<unit>`. Grafana: **JuniperCascor** (UID `juniper-cascor`).

> See: [Observability Guide](https://github.com/pcalnon/juniper-deploy/blob/main/docs/OBSERVABILITY_GUIDE.md) | [Configuration Reference](install/REFERENCE.md#logging-configuration)

---

## Dependencies and CI/CD

```bash
# Add dep: edit pyproject.toml, then regenerate requirements.lock
uv pip compile pyproject.toml \
  --extra ml --extra api --extra observability --extra juniper-data \
  --index-strategy unsafe-best-match --no-emit-package torch \
  --upgrade -o requirements.lock
# Conda env
conda create --name JuniperCascor1 --file conf/conda_environment.yaml
```

Core: `torch`, `numpy`, `h5py`, `matplotlib`, `PyYAML`, `requests`

**Pre-commit hooks:** black, isort, flake8, mypy, bandit, yamllint, shellcheck, markdownlint, pytest-unit (local), coverage-gate (local, 80%), no-unencrypted-env (SOPS guard). **Line length:** 512.

**CI pipeline:** pre-commit -> unit-tests -> integration-tests -> build -> security -> lockfile-check -> required-checks. Separate serial gates: `golden-regression.yml` (OUT-12) and `conformance.yml` (OUT-13). Path-filtered package CI: `ci-protocol.yml`, `ci-cascor-model.yml`.

**Dependabot lockfile:** `lockfile-update.yml` auto-regens when `CROSS_REPO_DISPATCH_TOKEN` is visible to the run. Dependabot uses a separate secret store — missing PAT there is a green no-op; **Lockfile Freshness** still blocks stale locks. Register the PAT under Dependabot secrets to restore auto-push.

**PyPI publish:** cut a GitHub Release (not a bare tag). Tags: `v*` → `publish.yml` (`juniper-cascor`); `juniper-cascor-protocol-v*` / `juniper-cascor-model-v*` → matching sub-package workflows. TestPyPI verify uses `--no-deps` and TestPyPI index only. Keep `pypa/gh-action-pypi-publish` SHA-pinned (Dependabot bumps all three workflows together).

> See: [CI Quick Start](ci_cd/QUICK_START.md#dependabot-lockfile-updates) | [CI Manual — Lockfile](ci_cd/MANUAL.md#lockfile-update-workflow) | [CI Manual — PyPI Publishing](ci_cd/MANUAL.md#pypi-publishing) | [CI Reference](ci_cd/REFERENCE.md#publish-workflows) | [Dependency Update Workflow](../notes/DEPENDENCY_UPDATE_WORKFLOW.md) | [Environment Setup](install/ENVIRONMENT_SETUP.md)

---

## Troubleshooting

| Symptom                                                               | Cause                                                       | Fix                                                                                                       |
|-----------------------------------------------------------------------|-------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| Unit tests fail with `assert '0.7.0' == '0.6.0'` (or similar SemVer)   | Wiring test pins a literal package version                  | Assert against `api.app._API_VERSION` (BUG-CC-04); do not hard-code SemVer in health/app/build-info tests |
| `CASCOR_LOG_LEVEL` no effect                                          | Set after import                                            | Set env var before any `import`                                                                           |
| Logger pickle error                                                   | Logger in `__getstate__`                                    | Exclude logger from pickle state                                                                          |
| `Unrecognized activation function name during deserialization`        | Activation name missing from `ActivationWithDerivative` map | Add matching key to `src/utils/activation.py` `ACTIVATION_MAP` (function `__name__` or module class name) |
| HDF5/pickle restore changed activation unexpectedly (legacy behavior) | Previous fallback-to-ReLU behavior no longer applies        | Use only supported activation names; unknown names now fail fast with `ValueError`                        |
| GPU tests skipped                                                     | No CUDA or flag missing                                     | `pytest --gpu` on GPU machine                                                                             |
| Long tests skipped                                                    | Flag not passed                                             | `pytest --run-long`                                                                                       |
| HDF5 load fails                                                       | Corrupted or version mismatch                               | `python -m snapshots.snapshot_cli verify snapshot.h5`                                                     |
| NaN in training                                                       | LR too high or bad data                                     | Reduce `learning_rate`, check tensors                                                                     |
| C7 `f1`/`roc_auc` always `null` on history rows                       | Within-pass `output_epoch` rows, or eval metrics disabled   | Read terminal `kind="training_step"` rows; ensure `JUNIPER_CASCOR_EVAL_METRICS_ENABLED` is not `0`/`false` |
| C7 scalars `null` with `eval_metrics.undefined` set                   | `empty_batch` / `single_class` / `invalid_output` on eval split | Check labels have ≥2 classes and finite outputs; see [C7 API notes](api/JUNIPER_CASCOR_API_REFERENCE.md#c7-scalar-evaluation-metrics) |
| Server refuses to start with `NonLoopbackBindError`                   | `JUNIPER_CASCOR_HOST` is non-loopback without a bind attestation | Bind `127.0.0.1` for local/dev, or set `JUNIPER_CASCOR_LOOPBACK_PUBLISH_ATTESTED=true` (loopback-only host publish) or `JUNIPER_CASCOR_AUTH_PROXY_ATTESTED=true` (fronting authenticating proxy) after verifying it |
| Server refuses to start with `AuthPostureError` / CRITICAL auth posture | `JUNIPER_CASCOR_REQUIRE_AUTH=true` and keys missing/blank (incl. empty `_FILE`) | Provision a real `JUNIPER_CASCOR_API_KEYS` / non-empty `*_FILE`, or set `REQUIRE_AUTH=false` only for bare/dev |
| Protected routes open / boot WARNING "running OPEN" with compose secrets | Empty/whitespace `JUNIPER_CASCOR_API_KEYS_FILE` (compose `_FILE`-only); `get_secret()` returns `""` with no env fallback | Put a real key in the secret file; set `JUNIPER_CASCOR_REQUIRE_AUTH=true` so empty secrets fail boot |
| WebSocket closes during connect with `1013`                           | Global, per-IP, or `/ws/control` per-identity cap reached   | Raise the relevant `JUNIPER_CASCOR_WS_MAX_CONNECTIONS_*` cap only after checking expected clients and worker fleet size |
| Golden / conformance tests collected but all skipped                  | Missing `--golden` / `--conformance` opt-in flags           | Pass the matching flag with `--slow --integration` (see `conftest.py`)                                                  |
| Golden lane red locally after green unit CI                           | Wrong interpreter/torch or multi-thread / xdist             | Match CI pins (3.13 + torch 2.11.0, serial, `CASCOR_NUM_PROCESSES=1`, single-thread BLAS)                               |
| Lockfile Freshness red; Update Lockfile green with no `[dependabot skip]` commit | `CROSS_REPO_DISPATCH_TOKEN` empty in Dependabot secret store | Register PAT under Dependabot secrets, or push a local `uv pip compile ... -o requirements.lock` |
| Update Lockfile hard-fails on a human `pyproject.toml` PR             | Actions PAT missing/expired                                 | Restore Actions `CROSS_REPO_DISPATCH_TOKEN` or commit the regen in the PR |
| Publish workflow skipped / wrong package                              | Release tag prefix does not match workflow guard            | Use `v*`, `juniper-cascor-protocol-v*`, or `juniper-cascor-model-v*` — see [PyPI Publishing](ci_cd/MANUAL.md#pypi-publishing) |
| TestPyPI `400 File already exists` on publish                         | Dual trigger or concurrent upload of same version           | Publish via Release only (no `push: tags`); bump version to re-upload |

---

## Cross-References

[Ecosystem Cheatsheet](https://github.com/pcalnon/juniper-ml/blob/main/docs/DEVELOPER_CHEATSHEET_JUNIPER-ML.md) | [juniper-cascor-client](https://github.com/pcalnon/juniper-cascor-client/blob/main/docs/DEVELOPER_CHEATSHEET.md) | [juniper-deploy](https://github.com/pcalnon/juniper-deploy/blob/main/docs/DEVELOPER_CHEATSHEET.md) | [Parent Ecosystem Guide](https://github.com/pcalnon/juniper-ml/blob/main/CLAUDE.md)

---
