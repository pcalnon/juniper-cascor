# Source Code Quick Start

This guide provides a rapid introduction to navigating and working with the Juniper Cascor codebase.

## Repository Tour

### Source Structure Overview

```text
src/
├── main.py                    # Application entry point
├── cascade_correlation/       # Core neural network implementation
│   ├── cascade_correlation.py           # CascadeCorrelationNetwork class
│   ├── cascade_correlation_config/      # Network configuration
│   └── cascade_correlation_exceptions/  # Custom exceptions
├── candidate_unit/            # Candidate hidden unit implementation
│   └── candidate_unit.py                # CandidateUnit class
├── spiral_problem/            # Two-spiral problem benchmark
│   └── spiral_problem.py                # SpiralProblem class
├── cascor_constants/          # Project-wide constants
│   ├── constants.py                     # Main constants aggregator
│   ├── constants_activation/            # Activation function constants
│   ├── constants_candidates/            # Candidate training constants
│   ├── constants_hdf5/                  # Serialization constants
│   ├── constants_logging/               # Logging constants
│   ├── constants_model/                 # Model architecture constants
│   └── constants_problem/               # Problem-specific constants
├── log_config/                # Logging infrastructure
│   ├── log_config.py                    # LogConfig class
│   └── logger/                          # Custom Logger class
├── profiling/                 # Performance profiling tools
├── snapshots/                 # HDF5 serialization system
│   ├── snapshot_serializer.py           # Main serialization logic
│   ├── snapshot_utils.py                # Utility functions
│   └── snapshot_cli.py                  # CLI tools
├── cascor_plotter/            # Visualization utilities
├── remote_client/             # Remote multiprocessing client
├── utils/                     # Helper utilities
└── tests/                     # Test suite
```

### Where to Start Reading Code

1. **Entry Point**: `src/main.py` - Application initialization, logging setup, and main execution flow
2. **Core Algorithm**: `src/cascade_correlation/cascade_correlation.py` - The `CascadeCorrelationNetwork` class
3. **Configuration**: `src/cascade_correlation/cascade_correlation_config/cascade_correlation_config.py`
4. **Constants**: `src/cascor_constants/constants.py` - All configurable parameters

### Key Entry Points

| File | Purpose |
|------|---------|
| `src/main.py` | Main application entry point |
| `src/cascade_correlation/cascade_correlation.py` | Core `CascadeCorrelationNetwork` class |
| `src/candidate_unit/candidate_unit.py` | `CandidateUnit` for network growth |
| `src/spiral_problem/spiral_problem.py` | Two-spiral benchmark problem |

## Common Dev Commands

### Run Application

```bash
cd src && python main.py
```

With profiling:

```bash
cd src && python main.py --profile                  # cProfile profiling
cd src && python main.py --profile-memory           # Memory profiling
cd src && python main.py --profile --profile-output ./my_profiles
```

### Run Tests

```bash
# All tests
cd src/tests && bash scripts/run_tests.bash

# Unit tests only
cd src/tests && bash scripts/run_tests.bash -u

# Integration tests
cd src/tests && bash scripts/run_tests.bash -i

# Tests with coverage
cd src/tests && bash scripts/run_tests.bash -v -c

# Specific test file
cd src/tests && python -m pytest unit/test_forward_pass.py -v

# Tests by marker
cd src/tests && python -m pytest -m "spiral" -v
```

### Type Checking

```bash
cd src && python -m mypy cascade_correlation/ candidate_unit/ --ignore-missing-imports
```

### Linting

```bash
cd src && python -m flake8 . --max-line-length=120 --extend-ignore=E203,E266,E501,W503
```

### Format Checking

```bash
cd src && python -m black --check --diff .
cd src && python -m isort --check-only --diff .
```

## Quick Code Navigation

### Main Network Class

**Location**: `src/cascade_correlation/cascade_correlation.py`

Key methods:

- `fit(x, y, epochs)` - Train the network
- `forward(x)` - Forward pass
- `train_output_layer(x, y, epochs)` - Output layer training
- `train_candidates(x, y, residual_error)` - Candidate pool training
- `get_accuracy(x, y)` - Calculate classification accuracy
- `save_to_hdf5(filepath)` / `load_from_hdf5(filepath)` - Serialization

### Configuration System

**Location**: `src/cascade_correlation/cascade_correlation_config/cascade_correlation_config.py`

The `CascadeCorrelationConfig` dataclass holds all network hyperparameters:

- `input_size`, `output_size` - Network dimensions
- `learning_rate` - Training learning rate
- `max_hidden_units` - Maximum hidden units to add
- `candidate_pool_size` - Number of candidates per round
- `correlation_threshold` - Stopping criterion
- `random_seed` - For reproducibility

### Serialization Code

**Location**: `src/snapshots/`

- `snapshot_serializer.py` - Main HDF5 save/load logic
- `snapshot_utils.py` - Utility functions
- `snapshot_cli.py` - Command-line interface

CLI usage:

```bash
python -m snapshots.snapshot_cli save network.pkl snapshot.h5
python -m snapshots.snapshot_cli load snapshot.h5
python -m snapshots.snapshot_cli verify snapshot.h5
```

### Test Examples

**Location**: `src/tests/unit/`

Example test files:

- `test_forward_pass.py` - Forward propagation tests
- `test_config_and_exceptions.py` - Configuration and error handling
- `test_candidate_unit_extended.py` - Candidate unit behavior
- `test_cascor_getters_setters.py` - Property accessors

## Running a Minimal Training Loop

### Python REPL Example

```python
import sys
sys.path.insert(0, 'src')

import torch
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

# Create configuration
config = CascadeCorrelationConfig(
    _CascadeCorrelationConfig__input_size=2,
    _CascadeCorrelationConfig__output_size=2,
    _CascadeCorrelationConfig__learning_rate=0.01,
    _CascadeCorrelationConfig__max_hidden_units=10,
    _CascadeCorrelationConfig__random_seed=42,
)

# Initialize network
network = CascadeCorrelationNetwork(
    _CascadeCorrelationNetwork__config=config
)

# Create sample data (XOR-like problem)
x_train = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y_train = torch.tensor([[1., 0.], [0., 1.], [0., 1.], [1., 0.]])

# Train
network.fit(x_train, y_train, epochs=100)

# Evaluate
accuracy = network.get_accuracy(x_train, y_train)
print(f"Training accuracy: {accuracy:.2%}")
```

### Quick Script Example

Create a file `train_example.py`:

```python
#!/usr/bin/env python
"""Minimal Cascade Correlation training example."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

def main():
    # Configuration
    config = CascadeCorrelationConfig(
        _CascadeCorrelationConfig__input_size=2,
        _CascadeCorrelationConfig__output_size=2,
        _CascadeCorrelationConfig__learning_rate=0.01,
        _CascadeCorrelationConfig__max_hidden_units=20,
        _CascadeCorrelationConfig__candidate_pool_size=8,
        _CascadeCorrelationConfig__patience=50,
        _CascadeCorrelationConfig__random_seed=42,
    )

    # Initialize network
    network = CascadeCorrelationNetwork(
        _CascadeCorrelationNetwork__config=config
    )

    # Generate simple 2-class dataset
    n_samples = 50
    torch.manual_seed(42)

    class0 = torch.randn(n_samples, 2) * 0.5 + torch.tensor([-1., -1.])
    class1 = torch.randn(n_samples, 2) * 0.5 + torch.tensor([1., 1.])

    x_train = torch.cat([class0, class1], dim=0)
    y_train = torch.cat([
        torch.tensor([[1., 0.]] * n_samples),
        torch.tensor([[0., 1.]] * n_samples)
    ], dim=0)

    # Shuffle
    perm = torch.randperm(x_train.size(0))
    x_train = x_train[perm]
    y_train = y_train[perm]

    # Train
    print("Training Cascade Correlation Network...")
    network.fit(x_train, y_train, epochs=500)

    # Evaluate
    accuracy = network.get_accuracy(x_train, y_train)
    print(f"Final training accuracy: {accuracy:.2%}")
    print(f"Hidden units added: {len(network.hidden_units)}")

    # Save trained network
    network.save_to_hdf5("./trained_network.h5")
    print("Network saved to trained_network.h5")

if __name__ == "__main__":
    main()
```

Run with:

```bash
python train_example.py
```

## Next Steps

- Review [API Documentation](../api/) for detailed class references
- See [Testing Guide](../testing/) for running and writing tests
- Check [Installation Guide](../install/) for environment setup
