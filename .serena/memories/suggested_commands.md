# Suggested Commands for Juniper Cascor

## Running the Application

```bash
cd src && python main.py

# With profiling
cd src && python main.py --profile                    # cProfile deterministic profiling
cd src && python main.py --profile-memory             # tracemalloc memory profiling
```

## Testing

```bash
# Run all unit tests
cd src/tests && bash scripts/run_tests.bash

# Run specific test categories
cd src/tests && bash scripts/run_tests.bash -u              # Unit tests only
cd src/tests && bash scripts/run_tests.bash -i              # Integration tests
cd src/tests && bash scripts/run_tests.bash -m "spiral"     # Spiral problem tests

# Run tests with coverage
cd src/tests && bash scripts/run_tests.bash -v -c

# Run a specific test file
cd src/tests && python -m pytest unit/test_forward_pass.py -v

# Run by marker
python -m pytest -m "unit and accuracy" -v

# Performance benchmarks
cd src/tests/scripts && bash run_benchmarks.bash
```

## Code Quality (Pre-commit Hooks)

```bash
# Install and setup (one-time)
pip install pre-commit
pre-commit install

# Run all hooks
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
```

## Individual Linting/Formatting Tools

```bash
# Format with Black
cd src && python -m black .

# Check formatting only
cd src && python -m black --check --diff .

# Sort imports with isort
cd src && python -m isort .

# Check import sorting only
cd src && python -m isort --check-only --diff .

# Lint with Flake8
cd src && python -m flake8 . --max-line-length=512 --extend-ignore=E203,E266,E501,W503

# Type check with MyPy
cd src && python -m mypy cascade_correlation/ candidate_unit/ --ignore-missing-imports
```

## Security Scanning

```bash
bandit -r src/                    # Run Bandit SAST scan
pip-audit                         # Check for dependency vulnerabilities
```

## Serialization CLI

```bash
python -m snapshots.snapshot_cli save network.pkl snapshot.h5
python -m snapshots.snapshot_cli load snapshot.h5
python -m snapshots.snapshot_cli verify snapshot.h5
python -m snapshots.snapshot_cli list ./snapshots/
```

## Environment Variables

| Variable | Description | Values |
|----------|-------------|--------|
| `CASCOR_LOG_LEVEL` | Override log level | `TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR` |

## System Utilities (Linux)

Standard Linux commands: `git`, `ls`, `cd`, `grep`, `find`, `cat`, `head`, `tail`
