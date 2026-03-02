# Testing Environment Setup

This guide covers the complete setup for the Juniper Cascor testing environment, including dependencies, configuration, GPU testing, coverage, and IDE integration.

## Test Dependencies

### Required Packages

The test suite requires the following pytest plugins:

| Package | Purpose |
|---------|---------|
| `pytest` | Core testing framework (≥6.0) |
| `pytest-cov` | Code coverage reporting |
| `pytest-timeout` | Prevent test hangs with timeouts |
| `pytest-xdist` | Parallel test execution (optional) |

### Installing Test Dependencies

**Using pip:**

```bash
pip install pytest pytest-cov pytest-timeout pytest-xdist
```

**Using conda (recommended):**

```bash
conda install pytest pytest-cov pytest-timeout pytest-xdist -c conda-forge
```

**From the project environment:**

```bash
conda env create -f conf/conda_environment.yaml
conda activate JuniperCascor
```

### Additional Dependencies

Tests also require the core project dependencies:

- `torch` - PyTorch for tensor operations
- `numpy` - Numerical computations
- `h5py` - HDF5 serialization testing

---

## Test Configuration

### pytest.ini Settings

The test configuration is defined in `src/tests/pytest.ini`:

```ini
[pytest]
minversion = 6.0
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Timeout configuration (CASCOR-P0-002)
timeout = 60
timeout_method = signal

addopts =
    -ra                              # Show extra test summary for all except passed
    -q                               # Quiet mode
    -p no:warnings                   # Disable warning capture
    --strict-markers                 # Error on unknown markers
    --strict-config                  # Error on config issues
    --continue-on-collection-errors  # Continue if collection fails
    --tb=short                       # Short traceback format
    --cov=cascade_correlation        # Coverage for cascade_correlation
    --cov=candidate_unit             # Coverage for candidate_unit
    --cov-report=term-missing        # Terminal report with missing lines
    --cov-report=html:htmlcov        # HTML report
    --cov-report=xml                 # XML report for CI
```

#### Key Settings Explained

| Setting | Description |
|---------|-------------|
| `minversion = 6.0` | Minimum pytest version required |
| `timeout = 60` | Default 60-second timeout per test |
| `timeout_method = signal` | Use signal-based timeout (Linux/macOS) |
| `--strict-markers` | Fail on unregistered markers |
| `-ra` | Show summary for all test results |
| `--tb=short` | Condensed traceback output |

### Timeout Configuration

- **Default timeout**: 60 seconds for standard tests
- **Slow tests**: 300 seconds (5 minutes) for training-intensive tests

To run slow tests without timeout:

```bash
pytest -m slow --timeout=0
```

To run only fast tests (CI default):

```bash
pytest -m "not slow"
```

### conftest.py Fixtures

The `src/tests/conftest.py` file provides shared fixtures for all tests.

#### Data Generation Fixtures

| Fixture | Description |
|---------|-------------|
| `simple_2d_data` | Simple 2D classification data (two classes) |
| `spiral_2d_data` | Two-spiral classification data |
| `n_spiral_data` | Parameterized N-spiral data generator |
| `regression_data` | Regression data with non-linear targets |

#### Network Configuration Fixtures

| Fixture | Description |
|---------|-------------|
| `simple_config` | Basic network configuration |
| `spiral_config` | Configuration optimized for spiral problems |
| `regression_config` | Configuration for regression tasks |

#### Network Instance Fixtures

| Fixture | Description |
|---------|-------------|
| `simple_network` | Fresh network with simple config |
| `spiral_network` | Fresh network for spiral problems |
| `regression_network` | Fresh network for regression |
| `trained_simple_network` | Pre-trained network on simple 2D data |

#### Candidate Unit Fixtures

| Fixture | Description |
|---------|-------------|
| `simple_candidate` | Basic CandidateUnit instance |

#### Validation Fixtures

| Fixture | Description |
|---------|-------------|
| `valid_tensor_2d` | Valid 2D input tensor |
| `valid_target_2d` | Valid one-hot target tensor |
| `invalid_tensors` | Collection of invalid tensors for error testing |

#### Mock Fixtures

| Fixture | Description |
|---------|-------------|
| `mock_logger` | Mock logger for testing |
| `mock_config` | Mock configuration object |

#### Utility Fixtures

| Fixture | Description |
|---------|-------------|
| `tolerance` | Standard floating-point tolerances |
| `device` | Appropriate device (cuda/cpu) |
| `fast_slow_mode` | Check if fast-slow mode is enabled |
| `fast_training_params` | Optimized params for fast execution |
| `training_scale` | Scale factor for fast-slow mode (0.1) |

#### Automatic Fixtures

| Fixture | Description |
|---------|-------------|
| `cleanup_temp_files` | Auto-cleanup after tests |
| `reset_random_seeds` | Reset seeds before each test |

---

## GPU Testing Setup

### Requirements for GPU Tests

1. **NVIDIA GPU** with CUDA support
2. **CUDA Toolkit** (version 12.x recommended)
3. **PyTorch with CUDA** support

### Verifying CUDA Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
```

**From command line:**

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Running Tests With/Without GPU

**Skip GPU tests (default behavior):**

```bash
# GPU tests are skipped by default
cd src/tests && bash scripts/run_tests.bash
```

**Run GPU tests:**

```bash
# Enable GPU tests with --gpu flag
cd src/tests && bash scripts/run_tests.bash -g

# Or directly with pytest
pytest --gpu
```

**Run only GPU tests:**

```bash
pytest -m gpu --gpu
```

### GPU Test Behavior

The `conftest.py` disables CUDA by default during tests:

```python
def pytest_configure(config):
    if not config.getoption("--gpu", default=False):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

This ensures consistent CPU-based test behavior unless GPU testing is explicitly requested.

---

## Coverage Configuration

### pytest-cov Setup

Coverage is configured in both `pytest.ini` and `pyproject.toml`:

**pyproject.toml (primary):**

```toml
[tool.coverage.run]
source = ["src/cascade_correlation", "src/candidate_unit", "src/snapshots"]
omit = [
    "*/tests/*",
    "*/__pycache__/*",
    "*/data/*",
    "*/logs/*",
]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
show_missing = true
precision = 2
```

### Coverage Source Modules

Coverage is collected for:

- `cascade_correlation/` - Core network implementation
- `candidate_unit/` - Candidate unit implementation
- `snapshots/` - Serialization system

### Report Formats

| Format | Location | Purpose |
|--------|----------|---------|
| Terminal | stdout | Quick overview with missing lines |
| HTML | `src/tests/reports/htmlcov/index.html` | Interactive browsing |
| XML | `src/tests/reports/coverage.xml` | CI/CD integration |

### Running with Coverage

```bash
# Default (coverage enabled)
cd src/tests && bash scripts/run_tests.bash

# Explicit coverage
cd src/tests && bash scripts/run_tests.bash -c

# Disable coverage for faster runs
cd src/tests && bash scripts/run_tests.bash --no-coverage
```

### Viewing Coverage Reports

```bash
# Open HTML report in browser (Linux)
xdg-open src/tests/reports/htmlcov/index.html

# Or navigate directly
firefox src/tests/reports/htmlcov/index.html
```

---

## Multiprocessing Considerations

### Fork Server vs Spawn Context

PyTorch and multiprocessing require careful handling:

```python
# conftest.py sets thread limits for parallel execution
if os.environ.get("PYTEST_XDIST_WORKER"):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    torch.set_num_threads(1)
```

### Known Issues by OS

| OS | Issue | Workaround |
|----|-------|------------|
| **Linux** | `fork` can cause CUDA issues | Use `forkserver` or `spawn` for GPU tests |
| **macOS** | Default `spawn` is slower | Accept performance trade-off |
| **Windows** | Only `spawn` available | No `signal` timeout method; use `thread` |

### Windows Timeout Configuration

On Windows, change the timeout method:

```ini
# pytest.ini or pyproject.toml
timeout_method = thread  # Instead of 'signal'
```

### Parallel Test Execution

```bash
# Run tests in parallel with pytest-xdist
cd src/tests && bash scripts/run_tests.bash -j

# Or directly
pytest -n auto  # Auto-detect CPU count
pytest -n 4     # Use 4 workers
```

**Note**: Coverage may be incomplete with parallel execution. For accurate coverage, run sequentially.

---

## IDE Integration

### VS Code pytest Configuration

Add to `.vscode/settings.json`:

```json
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestPath": "pytest",
    "python.testing.cwd": "${workspaceFolder}/src/tests",
    "python.testing.pytestArgs": [
        "--no-cov",
        "-v",
        "--tb=short"
    ],
    "python.testing.autoTestDiscoverOnSaveEnabled": true
}
```

#### Recommended VS Code Extensions

- **Python** (Microsoft) - Python language support
- **Python Test Explorer** - Test discovery and running
- **Coverage Gutters** - Display coverage in editor

### Running Tests from VS Code

1. **Test Explorer Panel**: Click the flask icon in the sidebar
2. **Run All Tests**: Click the play button at the top
3. **Run Single Test**: Click play next to individual tests
4. **Debug Test**: Right-click → "Debug Test"

### Launch Configuration for Debugging

Add to `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Pytest",
            "type": "debugpy",
            "request": "launch",
            "module": "pytest",
            "args": [
                "-v",
                "--no-cov",
                "-x",
                "${file}"
            ],
            "cwd": "${workspaceFolder}/src/tests",
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```

### PyCharm Configuration

1. **Settings** → **Tools** → **Python Integrated Tools**
2. Set **Default test runner** to `pytest`
3. Set **Working directory** to `src/tests`
4. Add pytest arguments: `--no-cov -v`

---

## Environment Variables

### Log Level Control

```bash
# Reduce logging for faster tests (default in conftest.py)
export CASCOR_LOG_LEVEL=WARNING

# Verbose logging for debugging
export CASCOR_LOG_LEVEL=DEBUG
```

### Fast-Slow Mode

```bash
# Enable fast-slow mode for quicker slow test execution
export JUNIPER_FAST_SLOW=1

# Or use command line
pytest --fast-slow
```

### CUDA Control

```bash
# Disable CUDA (done automatically unless --gpu)
export CUDA_VISIBLE_DEVICES=""

# Use specific GPU
export CUDA_VISIBLE_DEVICES=0
```

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Ensure `src/` is in PYTHONPATH or run from `src/tests/` |
| Tests hang | Check timeout settings; use `--timeout=0` to debug |
| Coverage missing | Don't use parallel execution (`-j`) for accurate coverage |
| GPU tests fail | Verify CUDA installation with `torch.cuda.is_available()` |
| Import errors | Run `pip install -e .` from project root |

### Verifying Installation

```bash
# Check pytest and plugins
pytest --version
pytest --co -q  # List all discovered tests

# Check coverage
coverage --version

# Verify project imports
cd src && python -c "from cascade_correlation import CascadeCorrelationNetwork"
```

---

## Quick Reference

```bash
# Run all unit tests
cd src/tests && bash scripts/run_tests.bash

# Run specific test file
pytest unit/test_forward_pass.py -v

# Run by marker
pytest -m "unit and accuracy" -v

# Run with GPU
pytest --gpu -v

# Run slow tests
pytest --slow --timeout=0 -v

# Fast execution mode
pytest --fast-slow -v

# Coverage report only
pytest --cov-report=html --no-cov-on-fail
```
