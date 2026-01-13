# AGENTS.md - Juniper Cascor Project Guide

**Project**: Juniper Cascade Correlation Neural Network  
**Version**: 0.3.2 (0.7.3)  
**License**: MIT License  
**Author**: Paul Calnon  
**Last Updated**: 2025-01-12

---

## Quick Reference

### Essential Commands

```bash
# Run the application
cd src && python main.py

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

# Type checking (Python uses type hints - no dedicated type checker configured)
# Linting via trunk (if available)
trunk check
```

### Key Entry Points

| File                                             | Purpose                            |
| ------------------------------------------------ | ---------------------------------- |
| `src/main.py`                                    | Main application entry point       |
| `src/cascade_correlation/cascade_correlation.py` | Core neural network implementation |
| `src/spiral_problem/spiral_problem.py`           | Two-spiral problem solver          |
| `src/candidate_unit/candidate_unit.py`           | Candidate unit for network growth  |
| `src/tests/run_tests.bash`                       | Test runner script                 |
| `src/tests/conftest.py`                          | Test configuration and fixtures    |

---

## Project Overview

Juniper Cascor is an AI/ML research platform implementing the **Cascade Correlation Neural Network** algorithm from foundational research (Fahlman & Lebiere, 1990). The project emphasizes ground-up implementations from primary literature for transparent exploration of constructive learning algorithms.

### Research Philosophy

- **Transparency over convenience**: Algorithms implemented from first principles
- **Understanding over abstraction**: Full visibility into network behavior
- **Modularity and scalability**: Designed for research flexibility

---

## Directory Structure

```bash
juniper_cascor/
├── src/                              # Source code root
│   ├── main.py                       # Application entry point
│   ├── cascade_correlation/          # Core Cascor network implementation
│   │   ├── cascade_correlation.py    # Main CascadeCorrelationNetwork class
│   │   ├── cascade_correlation_config/  # Network configuration
│   │   └── cascade_correlation_exceptions/  # Custom exceptions
│   ├── candidate_unit/               # Candidate unit implementation
│   │   └── candidate_unit.py         # CandidateUnit class
│   ├── spiral_problem/               # Two-spiral problem implementation
│   │   └── spiral_problem.py         # SpiralProblem class
│   ├── constants/                    # All project constants
│   │   ├── constants.py              # Main constants aggregator
│   │   ├── constants_activation/     # Activation function constants
│   │   ├── constants_candidates/     # Candidate training constants
│   │   ├── constants_hdf5/           # HDF5 serialization constants
│   │   ├── constants_logging/        # Logging constants
│   │   ├── constants_model/          # Model architecture constants
│   │   └── constants_problem/        # Problem-specific constants
│   ├── log_config/                   # Logging configuration
│   │   ├── log_config.py             # LogConfig class
│   │   └── logger/                   # Custom Logger class
│   ├── cascor_plotter/               # Visualization utilities
│   │   └── cascor_plotter.py         # CascadeCorrelationPlotter class
│   ├── snapshots/                    # HDF5 serialization system
│   │   ├── snapshot_serializer.py    # Main serialization logic
│   │   ├── snapshot_utils.py         # Utility functions
│   │   ├── snapshot_cli.py           # CLI tools
│   │   └── snapshot_common.py        # Common serialization helpers
│   ├── remote_client/                # Remote multiprocessing client
│   ├── utils/                        # Utility functions
│   │   └── utils.py                  # Helper functions
│   └── tests/                        # Test suite
│       ├── conftest.py               # Pytest configuration and fixtures
│       ├── pytest.ini                # Pytest settings
│       ├── unit/                     # Unit tests
│       ├── integration/              # Integration tests
│       ├── helpers/                  # Test utilities
│       ├── mocks/                    # Mock objects
│       └── scripts/                  # Test runner scripts
├── conf/                             # Configuration files
│   ├── conda_environment.yaml        # Conda environment
│   ├── logging_config.yaml           # Logging configuration
│   └── *.conf                        # Various shell config files
├── util/                             # Shell utility scripts
├── notes/                            # Project documentation
├── data/                             # Data directory (gitignored)
├── logs/                             # Log files (gitignored)
├── images/                           # Generated images
├── reports/                          # Generated reports
└── README.md                         # Project README
```

---

## Core Components

### CascadeCorrelationNetwork

The main neural network class implementing the Cascade Correlation algorithm.

**Location**: `src/cascade_correlation/cascade_correlation.py`

**Key Methods**:

- `fit(x, y, epochs)` - Train the network
- `forward(x)` - Forward pass through network
- `train_output_layer(x, y, epochs)` - Train output layer only
- `train_candidates(x, y, residual_error)` - Train candidate pool
- `get_accuracy(x, y)` - Calculate classification accuracy
- `save_to_hdf5(filepath)` - Save network to HDF5
- `load_from_hdf5(filepath)` - Load network from HDF5
- `create_snapshot()` - Create network snapshot

**Key Dataclasses**:

- `TrainingResults` - Aggregated candidate training results
- `ValidateTrainingInputs` - Inputs for training validation
- `ValidateTrainingResults` - Results from validation

### CandidateUnit

Represents a candidate hidden unit in the network.

**Location**: `src/candidate_unit/candidate_unit.py`

**Key Methods**:

- `train(x, residual_error, epochs)` - Train the candidate
- `_calculate_correlation(output, residual_error)` - Calculate Pearson correlation
- `_update_weights_and_bias(params)` - Update using autograd

**Key Dataclasses**:

- `CandidateTrainingResult` - Training result for single candidate
- `CandidateParametersUpdate` - Parameters for weight updates
- `CandidateCorrelationCalculation` - Correlation calculation results

### SpiralProblem

Implements the classic two-spiral classification problem.

**Location**: `src/spiral_problem/spiral_problem.py`

**Key Methods**:

- `evaluate(...)` - Run full evaluation pipeline
- `generate_spiral_dataset(...)` - Generate spiral data
- `_create_input_features(...)` - Create input tensors
- `_create_one_hot_targets(...)` - Create target tensors

---

## Programming Conventions

### Naming Conventions

**Constants**:

- Uppercase with underscores, prefixed by component: `_CASCOR_LOG_LEVEL_NAME`
- Hierarchical naming: `_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION`

**Classes**:

- PascalCase: `CascadeCorrelationNetwork`, `CandidateUnit`

**Methods/Functions**:

- snake_case: `train_candidates`, `calculate_correlation`
- Private methods prefixed with underscore: `_prepare_candidate_input`

**Constructor Parameters**:

- Name-mangled style with class prefix: `_SpiralProblem__n_points`, `CandidateUnit__input_size`

### File Headers

All Python files include standardized headers:

```python
#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCascor
# Application:   juniper_cascor
# Purpose:       Juniper Project Cascade Correlation Neural Network
#
# Author:        Paul Calnon
# Version:       0.3.2 (0.7.3)
# File Name:     [File Name]
# File Path:     [Project]/[Sub-Project]/[Application]/src/

# Date Created:  [YYYY-MM-DD]
# Last Modified: [YYYY-MM-DD HH:MM:SS TZ]
#
# License:       MIT License
# Copyright:     Copyright (c) 2024,2025,2026 Paul Calnon
#
# Description:
#     [This is a placeholder for the actual description.]
#
#####################################################################################################################################################################################################
# Notes:
#
#####################################################################################################################################################################################################
# References:
#
#####################################################################################################################################################################################################
# TODO :
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#####################################################################################################################################################################################################
```

### Imports

Standard ordering:

1. Standard library imports
2. Third-party imports (numpy, torch, etc.)
3. Local application imports

Path manipulation for local imports:

```python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### Type Hints

The project uses Python type hints extensively:

```python
def forward(self, x: torch.Tensor = None) -> torch.Tensor:
def _calculate_correlation(
    self,
    output: torch.Tensor = None,
    residual_error: torch.Tensor = None,
) -> tuple([float, torch.Tensor, torch.Tensor, float, float]):
```

### Logging

Custom logging system with extended log levels:

- TRACE, VERBOSE, DEBUG, INFO, WARNING, ERROR, CRITICAL, FATAL

Logger usage pattern:

```python
from log_config.logger.logger import Logger
self.logger = Logger
self.logger.info("Message")
self.logger.trace("Detailed trace message")
self.logger.verbose("Verbose output")
```

### Documentation

- Docstrings follow structured format with Description, Args, Returns, Raises, Notes sections
- Extensive inline logging for debugging and tracing

---

## Testing Infrastructure

### Test Framework

- **Framework**: pytest
- **Location**: `src/tests/`
- **Configuration**: `src/tests/pytest.ini`
- **Fixtures**: `src/tests/conftest.py`

### Test Categories (Markers)

| Marker               | Description                          |
| -------------------- | ------------------------------------ |
| `unit`               | Unit tests for individual components |
| `integration`        | Integration tests for full workflows |
| `performance`        | Performance and benchmarking tests   |
| `slow`               | Tests that take a long time to run   |
| `gpu`                | Tests that require GPU/CUDA          |
| `multiprocessing`    | Tests using multiprocessing          |
| `spiral`             | Spiral problem tests                 |
| `correlation`        | Correlation calculation tests        |
| `network_growth`     | Network growth algorithm tests       |
| `candidate_training` | Candidate unit training tests        |
| `validation`         | Input validation tests               |
| `accuracy`           | Accuracy calculation tests           |
| `early_stopping`     | Early stopping logic tests           |

### Running Tests

```bash
# All unit tests with coverage
cd src/tests && bash scripts/run_tests.bash

# Specific test file
python -m pytest unit/test_forward_pass.py -v

# By marker
python -m pytest -m "unit and accuracy" -v

# With specific options
bash scripts/run_tests.bash -v -c     # Verbose with coverage
bash scripts/run_tests.bash -j        # Parallel execution
bash scripts/run_tests.bash -f        # Re-run failed only
```

### Test Output

- HTML Coverage: `src/tests/reports/htmlcov/index.html`
- XML Coverage: `src/tests/reports/coverage.xml`
- JUnit XML: `src/tests/reports/junit.xml`

---

## Key Dependencies

### Core Libraries

| Library      | Purpose                               |
| ------------ | ------------------------------------- |
| `torch`      | Neural network tensors and operations |
| `numpy`      | Numerical computations                |
| `matplotlib` | Plotting and visualization            |
| `h5py`       | HDF5 file serialization               |
| `PyYAML`     | YAML configuration parsing            |

### Testing Libraries

| Library      | Purpose            |
| ------------ | ------------------ |
| `pytest`     | Test framework     |
| `pytest-cov` | Coverage reporting |

### Optional Libraries

| Library    | Purpose                           |
| ---------- | --------------------------------- |
| `columnar` | Formatted table output (optional) |

---

## Serialization System

### HDF5 Snapshots

The project uses HDF5 for network serialization.

**Save Network**:

```python
network.save_to_hdf5(
    filepath="./snapshots/network.h5",
    include_training_state=True,
    include_training_data=False
)
```

**Load Network**:

```python
loaded_network = CascadeCorrelationNetwork.load_from_hdf5("./snapshots/network.h5")
```

**CLI Tools**:

```bash
python -m snapshots.snapshot_cli save network.pkl snapshot.h5
python -m snapshots.snapshot_cli load snapshot.h5
python -m snapshots.snapshot_cli verify snapshot.h5
python -m snapshots.snapshot_cli list ./snapshots/
```

### What's Serialized

- Network architecture (input/output sizes, hidden units)
- Trained weights and biases
- Activation functions
- Training history
- Random state (Python, NumPy, PyTorch) for deterministic resume
- UUID for tracking
- Checksums for data integrity

---

## Multiprocessing

### Candidate Training

Candidate units are trained in parallel using Python's multiprocessing:

```python
class CandidateTrainingManager(BaseManager):
    """Custom manager for handling candidate training queues."""
    pass

CandidateTrainingManager.register("get_task_queue", callable=_create_task_queue)
CandidateTrainingManager.register("get_result_queue", callable=_create_result_queue)
```

### Pickling Considerations

Classes implement `__getstate__` and `__setstate__` to handle non-picklable objects (loggers, closures):

```python
def __getstate__(self):
    state = self.__dict__.copy()
    state.pop('logger', None)
    state.pop('_candidate_display_progress', None)
    return state

def __setstate__(self, state):
    self.__dict__.update(state)
    self.logger = Logger
```

---

## Configuration

### Constants Organization

Constants are organized hierarchically in `src/constants/`:

- `constants_model/` - Model architecture defaults
- `constants_candidates/` - Candidate training parameters
- `constants_activation/` - Activation functions
- `constants_logging/` - Logging configuration
- `constants_problem/` - Problem-specific settings
- `constants_hdf5/` - Serialization paths

### Adjusting Log Levels

Log levels are controlled via constants. To change:

```python
# In constants/constants.py, uncomment the desired level:
# _CASCOR_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_DEBUG
_CASCOR_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_INFO
```

### Network Configuration

Use `CascadeCorrelationConfig` for network setup:

```python
config = CascadeCorrelationConfig(
    input_size=2,
    output_size=2,
    learning_rate=0.01,
    max_hidden_units=50,
    candidate_pool_size=16,
    correlation_threshold=0.001,
    random_seed=42
)
network = CascadeCorrelationNetwork(config=config)
```

---

## Common Patterns

### Creating a Network and Training

```python
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

config = CascadeCorrelationConfig(input_size=2, output_size=2)
network = CascadeCorrelationNetwork(config=config)
network.fit(x_train, y_train, epochs=100)
accuracy = network.get_accuracy(x_test, y_test)
```

### Using the Spiral Problem

```python
from spiral_problem.spiral_problem import SpiralProblem

sp = SpiralProblem(
    _SpiralProblem__n_points=100,
    _SpiralProblem__n_spirals=2,
    _SpiralProblem__noise=0.1
)
sp.evaluate(n_points=100, n_spirals=2, plot=True)
```

### Error Handling

Custom exceptions in `cascade_correlation_exceptions/`:

- `ConfigurationError` - Invalid configuration
- `TrainingError` - Training failures
- `ValidationError` - Input validation failures

---

## Known Issues and Workarounds

### Logger Pickling

Loggers cannot be pickled for multiprocessing. Classes exclude logger from `__getstate__`.

### GPU Support

Tests disable GPU by default. Use `--gpu` flag for GPU tests:

```bash
pytest --gpu
```

### Random Reproducibility

Set `random_seed` in config for deterministic training:

```python
config = CascadeCorrelationConfig(random_seed=42)
```

---

## Development Workflow

### Adding New Features

1. Create feature in appropriate module
2. Add constants to `src/constants/`
3. Add tests in `src/tests/unit/` or `src/tests/integration/`
4. Update documentation in `notes/`
5. Run tests: `bash scripts/run_tests.bash`

### Adding New Tests

1. Create test file following `test_<feature>.py` naming
2. Use appropriate markers (`@pytest.mark.unit`, etc.)
3. Use fixtures from `conftest.py`
4. Follow Arrange-Act-Assert pattern

---

## Documentation Files

| File                                   | Description                     |
| -------------------------------------- | ------------------------------- |
| `notes/FEATURES_GUIDE.md`              | Feature documentation and usage |
| `notes/CASCOR_ENHANCEMENTS_ROADMAP.md` | Enhancement roadmap             |
| `notes/IMPLEMENTATION_SUMMARY.md`      | Implementation status           |
| `notes/Current_Issues.md`              | Known issues                    |
| `src/tests/README.md`                  | Test suite documentation        |

---

## Performance Considerations

### Serialization Performance

- Save (100 units): < 2 seconds
- Load (100 units): < 3 seconds
- Checksum verification: < 200ms

### Training Tips

- Optimize `candidate_pool_size` for CPU core count
- Use N-best candidate selection for faster convergence
- Tune `patience` for speed vs. accuracy tradeoff

---

## Security Notes

- No secrets or API keys in codebase
- Sensitive files excluded via `.gitignore`
- Log files contain training data - handle appropriately

---

## Contact

For questions about this codebase, refer to:

- Project documentation in `notes/`
- Test examples in `src/tests/`
- Constants definitions in `src/constants/`
