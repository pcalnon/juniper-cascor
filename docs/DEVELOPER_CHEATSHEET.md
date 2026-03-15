# Developer Cheatsheet — juniper-cascor

**Version**: 1.0.0  |  **Date**: 2026-03-15  |  **Project**: juniper-cascor

---

## Common Commands

| Task | Command |
|------|---------|
| Run application | `cd src && python main.py` |
| Run all tests | `cd src/tests && bash scripts/run_tests.bash` |
| Unit / integration tests | `bash scripts/run_tests.bash -u` / `bash scripts/run_tests.bash -i` |
| Tests with coverage | `bash scripts/run_tests.bash -v -c` |
| Run by marker | `python -m pytest -m "spiral" -v` |
| Long-running tests | `python -m pytest --run-long` |
| Benchmarks (quiet) | `cd src/tests/scripts && bash run_benchmarks.bash -q -n 10` |
| Pre-commit | `pre-commit run --all-files` |
| Type check | `cd src && python -m mypy cascade_correlation/ candidate_unit/ --ignore-missing-imports` |
| Lint / format / isort | `python -m flake8 . --max-line-length=512` / `python -m black --check .` / `python -m isort --check-only .` |
| Security scan | `bandit -r src/` |

> See: [AGENTS.md](../AGENTS.md) | [Configuration Reference](install/REFERENCE.md)

---

## Service Operations

```bash
conda activate JuniperCascor && cd juniper-cascor && pip install -e ".[all]"
cd src && python main.py                                       # native start
curl -s http://localhost:8200/v1/health | python -m json.tool  # health
docker compose --profile full up -d                            # Docker start
```

> See: [Ecosystem Service Ports](../../CLAUDE.md#service-ports) | [juniper-deploy AGENTS.md](../../juniper-deploy/AGENTS.md)

---

## API Endpoints

**Add an endpoint:** Create route under `/v1`, register in `app.py`, add client method in `juniper-cascor-client`.

**WebSocket:** `/ws/training` (metrics), `/ws/control` (commands). Update `ws_client.py` and canopy when adding.

**Middleware** (outermost first): CORS -> Security -> Prometheus -> RequestId. **Models:** Pydantic (API), dataclasses (config).

> See: [API Reference](api/API_REFERENCE.md) | [API Schemas](api/API_SCHEMAS.md)

---

## Configuration

**Add a constant:** Add to the appropriate module in `src/cascor_constants/`, follow `_UPPER_SNAKE` naming, import via `cascor_constants.constants`. Modules: `constants_model/` (architecture), `constants_candidates/` (pool/training), `constants_activation/` (functions), `constants_logging/` (levels), `constants_problem/` (defaults), `constants_hdf5/` (serialization).

> See: [Constants Guide](overview/CONSTANTS_GUIDE.md)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CASCOR_LOG_LEVEL` | `INFO` | Log level override (set before import) |
| `JUNIPER_DATA_URL` | `http://localhost:8100` | JuniperData service URL |
| `JUNIPER_DATA_API_KEY` | -- | API key for JuniperData |
| `JUNIPER_CASCOR_HOST` | `0.0.0.0` | Bind host for the service |
| `JUNIPER_CASCOR_PORT` | `8200` | Bind port for the service |
| `JUNIPER_CASCOR_CORS_ORIGINS` | `[]` | Allowed CORS origins |
| `JUNIPER_CASCOR_API_KEYS` | -- | API keys for authentication |
| `JUNIPER_CASCOR_LOG_FORMAT` | -- | Set to `json` for JSON logging |
| `JUNIPER_CASCOR_SENTRY_DSN` | -- | Sentry DSN |
| `JUNIPER_CASCOR_METRICS_ENABLED` | `false` | Enable Prometheus metrics |

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

| Parameter | Default | Notes |
|-----------|---------|-------|
| `learning_rate` | `0.01` | Output layer LR |
| `candidate_pool_size` | `50` | Tune to CPU core count |
| `candidate_epochs` | `600` | More = better candidates |
| `correlation_threshold` | `0.0005` | Minimum for selection |
| `patience` | `50` | Early stopping epochs |
| `max_hidden_units` | `50` | Network growth limit |
| `target_accuracy` | `0.999` | Termination goal |

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

| HDF5 Group | Contents |
|------------|----------|
| `meta` | UUID, version, timestamps, checksums |
| `arch` | Input/output sizes, hidden units, activation functions |
| `params` | Trained weights and biases for all layers |
| `history` | Training history, loss, accuracy per epoch |
| `random_state` | Python, NumPy, PyTorch RNG states (deterministic resume) |

Compression: gzip level 4 (default). Performance: Save <2s, Load <3s, Verify <200ms (100 units).

> See: [Constants Guide -- Serialization](overview/CONSTANTS_GUIDE.md#serialization-constants)

---

## Testing

**Coverage gate:** 80%. Reports: `src/tests/reports/htmlcov/`, `coverage.xml`, `junit.xml`.

| Marker | Description | Marker | Description |
|--------|-------------|--------|-------------|
| `unit` | Individual components | `spiral` | Spiral problem |
| `integration` | Full workflows | `correlation` | Correlation calculation |
| `performance` | Benchmarking | `network_growth` | Growth algorithm |
| `slow` | Long-running | `candidate_training` | Candidate training |
| `long` | Needs `--run-long` | `validation` | Input validation |
| `gpu` | Needs GPU/CUDA | `accuracy` / `early_stopping` | Accuracy / stopping |
| `multiprocessing` | Multi-process tests | `requires_juniper_data` | Needs juniper-data service |

> See: [Testing Quick Start](testing/QUICK_START.md) | [Testing Reference](testing/REFERENCE.md)

---

## Logging and Observability

Levels: TRACE(5) -> VERBOSE(7) -> DEBUG(10) -> INFO(20) -> WARNING(30) -> ERROR(40) -> CRITICAL(50) -> FATAL(60). Set `CASCOR_LOG_LEVEL` before import: `export CASCOR_LOG_LEVEL=DEBUG` or inline `CASCOR_LOG_LEVEL=WARNING python main.py`.

**Prometheus metrics:** Namespace `juniper_cascor_*`. Pattern: `juniper_cascor_<subsystem>_<name>_<unit>`. Grafana: **JuniperCascor** (UID `juniper-cascor`).

> See: [Observability Guide](../../juniper-deploy/docs/OBSERVABILITY_GUIDE.md) | [Configuration Reference](install/REFERENCE.md#logging-configuration)

---

## Dependencies and CI/CD

```bash
# Add dep: edit pyproject.toml, then regenerate lockfile
uv pip compile pyproject.toml -o requirements.txt
# Conda env
conda create --name JuniperCascor --file conf/conda_environment.yaml
```

Core: `torch`, `numpy`, `h5py`, `matplotlib`, `PyYAML`, `requests`

**Pre-commit hooks:** black, isort, flake8, mypy, bandit, yamllint, shellcheck, markdownlint, pytest-unit (local), coverage-gate (local, 80%), no-unencrypted-env (SOPS guard). **Line length:** 512.

**CI pipeline:** pre-commit -> unit-tests -> integration-tests -> build -> security -> required-checks

> See: [CI Quick Start](ci_cd/QUICK_START.md) | [CI Reference](ci_cd/REFERENCE.md) | [Environment Setup](install/ENVIRONMENT_SETUP.md)

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CASCOR_LOG_LEVEL` no effect | Set after import | Set env var before any `import` |
| Logger pickle error | Logger in `__getstate__` | Exclude logger from pickle state |
| GPU tests skipped | No CUDA or flag missing | `pytest --gpu` on GPU machine |
| Long tests skipped | Flag not passed | `pytest --run-long` |
| HDF5 load fails | Corrupted or version mismatch | `python -m snapshots.snapshot_cli verify snapshot.h5` |
| NaN in training | LR too high or bad data | Reduce `learning_rate`, check tensors |

---

## Cross-References

[Ecosystem Cheatsheet](../../juniper-ml/notes/DEVELOPER_CHEATSHEET.md) | [juniper-cascor-client](../../juniper-cascor-client/docs/DEVELOPER_CHEATSHEET.md) | [juniper-deploy](../../juniper-deploy/docs/DEVELOPER_CHEATSHEET.md) | [Parent Ecosystem Guide](../../CLAUDE.md)
