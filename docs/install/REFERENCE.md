# Configuration Reference

This document provides a comprehensive reference for all configuration options, environment variables, CLI arguments, and directory conventions in Juniper Cascor.

## Environment Variables

### CASCOR_LOG_LEVEL

Override the default log level at runtime without modifying code.

| Attribute | Value |
|-----------|-------|
| **Variable** | `CASCOR_LOG_LEVEL` |
| **Type** | String |
| **Default** | `INFO` |
| **Valid Values** | `TRACE`, `VERBOSE`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, `FATAL` |

**Log Level Hierarchy** (from most to least verbose):

| Level | Number | Description |
|-------|--------|-------------|
| `TRACE` | 1 | Maximum verbosity, detailed execution tracing |
| `VERBOSE` | 5 | Extended debug information |
| `DEBUG` | 10 | Debug messages for development |
| `INFO` | 20 | General informational messages |
| `WARNING` | 30 | Warning conditions |
| `ERROR` | 40 | Error conditions |
| `CRITICAL` | 50 | Critical errors requiring immediate attention |
| `FATAL` | 60 | Fatal errors causing application termination |

**Usage Examples:**

```bash
# Quiet mode for production/benchmarking (less verbose)
export CASCOR_LOG_LEVEL=WARNING
python main.py

# Debug mode for verbose output
export CASCOR_LOG_LEVEL=DEBUG
python main.py

# Maximum verbosity for tracing execution
export CASCOR_LOG_LEVEL=TRACE
python main.py

# Single command with environment override
CASCOR_LOG_LEVEL=WARNING python main.py
```

**Notes:**

- Must be set **before** importing cascor modules (the value is read at import time)
- Invalid values are ignored and the default `INFO` level is used
- The test suite sets `CASCOR_LOG_LEVEL=WARNING` by default to reduce output noise

## CLI Arguments

### Main Application (`src/main.py`)

Run the application from the `src/` directory:

```bash
cd src && python main.py [options]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--profile` | Flag | `false` | Enable cProfile deterministic profiling |
| `--profile-memory` | Flag | `false` | Enable tracemalloc memory profiling |
| `--profile-output` | String | `./profiles` | Directory for profile output files |
| `--profile-top-n` | Integer | `30` | Number of top functions to display in profile output |

### Profiling Options

**cProfile Deterministic Profiling:**

```bash
# Basic profiling
python main.py --profile

# Custom output directory
python main.py --profile --profile-output ./my_profiles

# Show top 50 functions
python main.py --profile --profile-top-n 50
```

**Memory Profiling:**

```bash
# Basic memory profiling
python main.py --profile-memory

# With custom top-N allocation display
python main.py --profile-memory --profile-top-n 50
```

**External Sampling Profiler (py-spy):**

```bash
# Generate flame graph
./util/profile_training.bash

# Generate SVG flame graph
./util/profile_training.bash --svg

# Generate Speedscope JSON format
./util/profile_training.bash --speedscope
```

### Test Runner (`src/tests/scripts/run_tests.bash`)

Run tests from the `src/tests/` directory:

```bash
cd src/tests && bash scripts/run_tests.bash [options]
```

| Argument | Description |
|----------|-------------|
| `-u` | Run unit tests only |
| `-i` | Run integration tests only |
| `-m "marker"` | Run tests matching a specific marker (e.g., `spiral`, `correlation`) |
| `-v` | Verbose output |
| `-c` | Generate coverage report |
| `-j` | Run tests in parallel |
| `-f` | Re-run failed tests only |

**Examples:**

```bash
# All unit tests with coverage
bash scripts/run_tests.bash -v -c

# Spiral problem tests only
bash scripts/run_tests.bash -m "spiral"

# Parallel execution
bash scripts/run_tests.bash -j
```

### Benchmark Runner (`src/tests/scripts/run_benchmarks.bash`)

Run benchmarks from the `src/tests/scripts/` directory:

```bash
cd src/tests/scripts && bash run_benchmarks.bash [options]
```

| Argument | Description |
|----------|-------------|
| `-s` | Run serialization benchmarks only |
| `-q`, `--quiet` | Quiet mode (sets `CASCOR_LOG_LEVEL=WARNING`) |
| `-n N` | Number of iterations |

**Examples:**

```bash
# All benchmarks
bash run_benchmarks.bash

# Serialization only
bash run_benchmarks.bash -s

# Quiet mode with 10 iterations
bash run_benchmarks.bash -q -n 10
```

## Configuration Files

### conf/conda_environment.yaml

Conda environment specification for the project.

**Purpose:** Defines all Python dependencies and their versions for reproducible environments.

**Usage:**

```bash
# Create environment from file
conda create --name JuniperCascor --file conf/conda_environment.yaml

# Export current environment
conda list --export > conf/conda_environment.yaml
```

**Key Dependencies:**

| Package | Purpose |
|---------|---------|
| `torch` | PyTorch for tensor operations and neural networks |
| `numpy` | Numerical computing |
| `pytest` | Test framework |
| `pyyaml` | YAML configuration parsing |

### conf/logging_config.yaml

YAML-based logging configuration loaded by Python's `logging.config.dictConfig`.

**Structure:**

```yaml
version: 1
formatters:
  formatter_console:
    format: "[%(filename)s:%(lineno)d] (%(asctime)s) [%(levelname)s] %(message)s"
    datefmt: "%y-%m-%d %H:%M:%S"
  formatter_file:
    format: "[%(filename)s: %(funcName)s:%(lineno)d] (%(asctime)s) [%(levelname)s] %(message)s"
    datefmt: "%Y-%m-%d %H:%M:%S"
handlers:
  handler_console:
    class: logging.StreamHandler
    level: TRACE
    formatter: formatter_console
    stream: ext://sys.stdout
  handler_file:
    class: logging.FileHandler
    level: TRACE
    formatter: formatter_file
    filename: juniper_cascor.log
    encoding: utf-8
    mode: a
loggers:
  juniper:
    level: TRACE
    handlers: [handler_console, handler_file]
    propagate: False
root:
  handlers: [handler_console, handler_file]
  level: TRACE
```

**Customization:**

- Change handler `level` values to adjust verbosity per output
- Modify `format` strings to include/exclude log fields
- Update `filename` to change log file location

### conf/script_util.cfg

Shell configuration file sourced by utility scripts.

**Purpose:** Defines global constants for bash utility scripts.

**Key Constants:**

| Constant | Description |
|----------|-------------|
| `JUNIPER_PROJECT_NAME` | Project name (`Juniper`) |
| `JUNIPER_APPLICATION_NAME` | Application name (`JuniperCascor`) |
| `JUNIPER_APPLICATION_VERSION` | Current version |
| `DATA_DIR_NAME` | Data directory name (`data`) |
| `LOGGING_DIR_NAME` | Logs directory name (`logs`) |
| `IMAGES_DIR_NAME` | Images directory name (`images`) |
| `SOURCE_DIR_NAME` | Source directory name (`src`) |

**Usage in scripts:**

```bash
#!/usr/bin/env bash
source "${PROJECT_ROOT}/conf/script_util.cfg"

echo "Project: ${JUNIPER_PROJECT_NAME}"
echo "Data directory: ${ROOT_PROJECT_DIR}/${DATA_DIR_NAME}"
```

## JuniperData Configuration

JuniperData is the dataset management service that provides training data to Juniper Cascor.

### JuniperDataClient Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_url` | `http://localhost:8100` | JuniperData API base URL |
| `timeout` | `30` | Request timeout in seconds |

### API Endpoints

The JuniperDataClient uses the following API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/datasets` | GET | List available datasets |
| `/v1/datasets/{id}/artifact` | GET | Download dataset artifact |

### Configuration Examples

**Default Configuration:**

```python
from juniper_data_client import JuniperDataClient

# Uses default base_url (http://localhost:8100) and timeout (30s)
client = JuniperDataClient()
```

**Custom Configuration:**

```python
from juniper_data_client import JuniperDataClient

# Custom base URL and timeout
client = JuniperDataClient(
    base_url="http://juniper-data.example.com:8100",
    timeout=60
)
```

### Notes

- The JuniperData URL is configured via the `JuniperDataClient` constructor
- Default timeout is 30 seconds for all API requests
- No environment variable override is currently available; configuration is done programmatically

## Directory Conventions

### Project Directory Structure

| Directory | Purpose | Git Status |
|-----------|---------|------------|
| `data/` | Training data, datasets | `.gitignore` |
| `logs/` | Application log files | `.gitignore` |
| `reports/` | Test and coverage reports | `.gitignore` |
| `images/` | Generated visualizations and plots | Tracked |
| `snapshots/` | HDF5 network snapshot files | `.gitignore` |
| `src/cascor_snapshots/` | Additional snapshot storage | `.gitignore` |
| `profiles/` | Profiling output files | `.gitignore` |

### data/

**Purpose:** Storage for training datasets and generated data files.

**Contents:**

- Spiral dataset files
- Custom training data
- Preprocessed tensors

**Notes:**

- Create directory if it doesn't exist: `mkdir -p data/`
- Files in this directory are not committed to git

### logs/

**Purpose:** Application log file storage.

**Default Log File:** `juniper_cascor.log`

**Location Configuration:**

The log file path is configured via constants in `src/cascor_constants/constants.py`:

```python
_CASCOR_LOG_FILE_NAME = "juniper_cascor.log"
_CASCOR_LOG_FILE_PATH = "./logs/"
```

### reports/

**Purpose:** Generated test and analysis reports.

**Contents:**

- `htmlcov/` - HTML coverage reports
- `coverage.xml` - XML coverage for CI/CD
- `junit.xml` - JUnit test results

**Generation:**

```bash
# Generate coverage reports
cd src/tests && bash scripts/run_tests.bash -c

# View HTML report
open reports/htmlcov/index.html
```

### images/

**Purpose:** Generated visualizations and plots.

**Contents:**

- Spiral dataset visualizations
- Training progress plots
- Network architecture diagrams

### snapshots/

**Purpose:** HDF5 network serialization files.

**Usage:**

```python
# Save network
network.save_to_hdf5("./snapshots/network.h5")

# Load network
loaded = CascadeCorrelationNetwork.load_from_hdf5("./snapshots/network.h5")
```

**CLI Tools:**

```bash
python -m snapshots.snapshot_cli save network.pkl snapshot.h5
python -m snapshots.snapshot_cli load snapshot.h5
python -m snapshots.snapshot_cli verify snapshot.h5
python -m snapshots.snapshot_cli list ./snapshots/
```

## Logging Configuration

### Custom Log Levels

Juniper Cascor extends Python's standard logging with custom levels:

| Level | Number | Method | Use Case |
|-------|--------|--------|----------|
| `TRACE` | 1 | `logger.trace()` | Detailed execution tracing |
| `VERBOSE` | 5 | `logger.verbose()` | Extended debug output |
| `FATAL` | 60 | `logger.fatal()` | Unrecoverable errors |

### Programmatic Configuration

```python
from log_config.log_config import LogConfig
from log_config.logger.logger import Logger

# Use class methods for early logging (before configuration)
Logger.info("Starting application")
Logger.debug("Debug message")

# Configure logging
log_config = LogConfig(
    _LogConfig__log_level_name="DEBUG",
    _LogConfig__log_file_name="custom.log",
    _LogConfig__log_file_path="./logs/"
)

# Get configured logger instance
logger = log_config.get_logger()
logger.trace("Trace level message")
logger.verbose("Verbose level message")
```

### Log Format Placeholders

| Placeholder | Description |
|-------------|-------------|
| `%(filename)s` | Source file name |
| `%(lineno)d` | Line number |
| `%(funcName)s` | Function name |
| `%(asctime)s` | Timestamp |
| `%(levelname)s` | Log level name |
| `%(message)s` | Log message |

### Console vs File Formatting

- **Console:** `[filename:line] (timestamp) [LEVEL] message`
- **File:** `[filename: function:line] (timestamp) [LEVEL] message`

The file format includes function names for better debugging context.

## Type Checking and Linting

### mypy Configuration

```bash
cd src && python -m mypy cascade_correlation/ candidate_unit/ --ignore-missing-imports
```

### flake8 Configuration

```bash
cd src && python -m flake8 . --max-line-length=512 --extend-ignore=E203,E266,E501,W503
```

### black (Format Checking)

```bash
cd src && python -m black --check --diff .
```

### isort (Import Sorting)

```bash
cd src && python -m isort --check-only --diff .
```

### trunk (All-in-One)

```bash
trunk check
```

## Quick Reference

### Common Commands

```bash
# Run application
cd src && python main.py

# Run with profiling
cd src && python main.py --profile

# Run tests
cd src/tests && bash scripts/run_tests.bash

# Run benchmarks (quiet mode)
cd src/tests/scripts && bash run_benchmarks.bash -q

# Set log level
export CASCOR_LOG_LEVEL=WARNING
```

### Environment Variable Summary

| Variable | Description | Example |
|----------|-------------|---------|
| `CASCOR_LOG_LEVEL` | Override log level | `WARNING`, `DEBUG`, `TRACE` |

### Default Paths

| Item | Default Path |
|------|--------------|
| Log file | `./logs/juniper_cascor.log` |
| Logging config | `./conf/logging_config.yaml` |
| Snapshots | `./snapshots/` |
| Profiles | `./profiles/` |
| Coverage HTML | `./src/tests/reports/htmlcov/` |
| JuniperData API | `http://localhost:8100` |
