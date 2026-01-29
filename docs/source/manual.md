# Juniper Cascor - Source Code Manual

**Version**: 0.3.21  
**Last Updated**: 2026-01-29  
**Purpose**: Comprehensive guide for understanding and modifying the source code

---

## Table of Contents

1. [Module-by-Module Overview](#module-by-module-overview)
2. [Code Architecture](#code-architecture)
3. [Extension Points](#extension-points)
4. [Coding Conventions](#coding-conventions)

---

## Module-by-Module Overview

### cascade_correlation/

**Location**: `src/cascade_correlation/`  
**Purpose**: Core neural network implementation

```
cascade_correlation/
├── cascade_correlation.py              # Main CascadeCorrelationNetwork class
├── cascade_correlation_config/
│   └── cascade_correlation_config.py   # CascadeCorrelationConfig, OptimizerConfig
└── cascade_correlation_exceptions/
    └── cascade_correlation_exceptions.py  # Custom exceptions
```

**Key Components**:

| Component | Description |
|-----------|-------------|
| `CascadeCorrelationNetwork` | Main network class with fit(), forward(), save/load |
| `CascadeCorrelationConfig` | Configuration dataclass |
| `OptimizerConfig` | Optimizer settings |
| `TrainingResults` | Training result dataclass |

**Core Logic**:

1. **Training Loop** (`fit()`):
   - Train output layer until convergence or patience exceeded
   - If target accuracy not reached, train candidate pool
   - Select best candidate(s) and add to network
   - Repeat until target accuracy or max hidden units

2. **Network Growth** (`_add_hidden_unit()`):
   - Freeze existing hidden units
   - Connect new unit to all inputs and existing hidden units
   - Expand output layer to include new unit output

3. **Validation** (`_validate_training_inputs()`):
   - Check tensor validity (not None, correct shape)
   - Validate NaN/Inf values
   - Check batch sizes match

### candidate_unit/

**Location**: `src/candidate_unit/`  
**Purpose**: Candidate hidden unit for network growth

```
candidate_unit/
└── candidate_unit.py   # CandidateUnit class
```

**Key Components**:

| Component | Description |
|-----------|-------------|
| `CandidateUnit` | Single candidate with weights/bias |
| `CandidateTrainingResult` | Training result dataclass |
| `CandidateCorrelationCalculation` | Correlation calculation result |

**Core Logic**:

1. **Correlation Optimization**:
   - Compute Pearson correlation between unit output and residual error
   - Use gradient descent to maximize absolute correlation
   - Track best correlation across all output dimensions

2. **Weight Updates**:

   ```python
   # Gradient ascent on correlation (sign determines direction)
   gradient = compute_correlation_gradient(output, residual_error)
   weights += learning_rate * sign(correlation) * gradient
   ```

### spiral_problem/

**Location**: `src/spiral_problem/`  
**Purpose**: Classic two-spiral classification benchmark

```
spiral_problem/
└── spiral_problem.py   # SpiralProblem class
```

**Key Components**:

| Component | Description |
|-----------|-------------|
| `SpiralProblem` | Problem generator and evaluator |
| `SpiralDataGenerator` | Data generation utilities |

**Usage Pattern**:

```python
sp = SpiralProblem()
results = sp.evaluate(
    n_points=100,    # Points per spiral
    n_spirals=2,     # Number of spirals
    noise=0.1,       # Noise level
    epochs=100,      # Training epochs
    plot=True        # Enable visualization
)
```

### snapshots/

**Location**: `src/snapshots/`  
**Purpose**: HDF5 serialization system

```
snapshots/
├── snapshot_serializer.py  # CascadeHDF5Serializer
├── snapshot_utils.py       # HDF5Utils helper functions
├── snapshot_cli.py         # Command-line interface
└── snapshot_common.py      # Common serialization helpers
```

**Key Components**:

| Component | Description |
|-----------|-------------|
| `CascadeHDF5Serializer` | Main serialization class |
| `HDF5Utils` | Utility functions (list, verify, cleanup) |
| CLI | Command-line tools for snapshot management |

**Serialization Flow**:

```
Network → save_to_hdf5()
    ↓
CascadeHDF5Serializer._save_network()
    ↓
Create HDF5 groups: /metadata, /architecture, /weights, /training_state, /random_state
    ↓
Compress and write to file
```

### cascor_constants/

**Location**: `src/cascor_constants/`  
**Purpose**: All project constants organized by domain

```
cascor_constants/
├── constants.py                  # Main constants aggregator
├── constants_activation/
│   └── constants_activation.py   # Activation function constants
├── constants_candidates/
│   └── constants_candidates.py   # Candidate training constants
├── constants_hdf5/
│   └── constants_hdf5.py         # Serialization constants
├── constants_logging/
│   └── constants_logging.py      # Logging constants
├── constants_model/
│   └── constants_model.py        # Model architecture constants
└── constants_problem/
    └── constants_problem.py      # Problem-specific constants
```

**Usage Pattern**:

```python
from cascor_constants.constants import (
    _CASCOR_LOG_LEVEL_NAME,
    _DEFAULT_LEARNING_RATE,
    _DEFAULT_CANDIDATE_POOL_SIZE,
)
```

**Important Note**: Constants are prefixed with underscore to indicate they are module-private. Use configuration objects to override defaults.

### log_config/

**Location**: `src/log_config/`  
**Purpose**: Custom logging infrastructure

```
log_config/
├── log_config.py     # LogConfig class
└── logger/
    └── logger.py     # Custom Logger class
```

**Key Features**:

- Extended log levels: TRACE (5), VERBOSE (7), FATAL (60)
- Configurable via YAML or code
- Environment variable override: `CASCOR_LOG_LEVEL`
- Performance logging helpers

**Usage**:

```python
from log_config.logger.logger import Logger

Logger.trace("Detailed trace")
Logger.verbose("Verbose output")
Logger.info("Information")
Logger.warning("Warning")
```

### profiling/

**Location**: `src/profiling/`  
**Purpose**: Profiling infrastructure (added in 0.3.20)

```
profiling/
├── __init__.py
├── profiling.py         # ProfileContext, MemoryTracker
└── logging_utils.py     # SampledLogger, BatchLogger
```

**Key Components**:

| Component | Description |
|-----------|-------------|
| `ProfileContext` | cProfile context manager |
| `MemoryTracker` | tracemalloc context manager |
| `profile_function` | Decorator for function profiling |
| `memory_profile` | Decorator for memory profiling |
| `SampledLogger` | Log sampling for hot paths |
| `BatchLogger` | Batch log messages |

**CLI Flags** (in main.py):

- `--profile`: Enable cProfile
- `--profile-memory`: Enable tracemalloc
- `--profile-output`: Custom output directory
- `--profile-top-n`: Number of top items to show

### remote_client/

**Location**: `src/remote_client/`  
**Purpose**: Remote multiprocessing client for distributed training

```
remote_client/
└── remote_worker_client.py   # RemoteWorkerClient class
```

**Current Status**: Implemented but experimental. Used for distributing candidate training across multiple machines.

**Key Methods**:

```python
client = RemoteWorkerClient()
client.connect(address, authkey)
client.start_workers(num_workers)
client.stop_workers(timeout)
client.disconnect()
```

### utils/

**Location**: `src/utils/`  
**Purpose**: General utility functions

```
utils/
└── utils.py   # Helper functions
```

**Key Functions**:

| Function | Description |
|----------|-------------|
| `display_progress(freq)` | Create progress display callback |
| `get_class_distribution(y)` | Get class counts from one-hot |
| `convert_to_numpy(x, y)` | Convert tensors to arrays |
| `convert_to_tensor(x, y)` | Convert arrays to tensors |

---

## Code Architecture

### Module Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                           main.py                                │
│                    (Entry point, CLI args)                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      spiral_problem/                             │
│              (Problem definition & evaluation)                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    cascade_correlation/                          │
│                (CascadeCorrelationNetwork)                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ fit() → train_output_layer() → train_candidates()        │  │
│  │         → _select_best_candidates() → _add_hidden_unit() │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ candidate_unit/ │  │   snapshots/    │  │  log_config/    │
│ (CandidateUnit) │  │ (Serialization) │  │   (Logging)     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     cascor_constants/                            │
│                 (Configuration defaults)                         │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow During Training

```
Input Data (x, y)
       │
       ▼
┌─────────────────────────────────────┐
│       Validate Inputs               │
│  (shape, type, NaN/Inf checks)      │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│    Train Output Layer               │
│  (gradient descent on MSE loss)     │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│   Check Stopping Criteria           │
│  (accuracy, patience, max epochs)   │
└─────────────────────────────────────┘
       │ (if not converged)
       ▼
┌─────────────────────────────────────┐
│  Calculate Residual Error           │
│  (target - output)                  │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│   Train Candidate Pool              │
│  (parallel or sequential)           │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ Candidate 1: maximize corr  │   │
│  │ Candidate 2: maximize corr  │   │
│  │ ...                         │   │
│  │ Candidate N: maximize corr  │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│   Select Best Candidate(s)          │
│  (by absolute correlation)          │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│   Add to Network                    │
│  (freeze weights, expand output)    │
└─────────────────────────────────────┘
       │
       ▼
    Repeat (back to Train Output Layer)
```

### Dependency Flow

```python
# Import order shows dependency hierarchy
from cascor_constants.constants import ...        # No dependencies
from log_config.logger.logger import Logger       # Depends on constants
from cascade_correlation_config import ...        # Depends on constants
from candidate_unit import CandidateUnit          # Depends on config, logger
from cascade_correlation import ...               # Depends on all above
from spiral_problem import SpiralProblem          # Depends on network
from snapshots import ...                         # Depends on network
```

---

## Extension Points

### Adding New Problems

Create a new problem class following the `SpiralProblem` pattern:

```python
# src/my_problem/my_problem.py

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

class MyProblem:
    """Custom classification problem."""

    def __init__(self, **kwargs):
        self.config_params = kwargs

    def generate_dataset(self, n_samples):
        """Generate training data."""
        # Return (x, y) tensors
        x = ...  # shape: (n_samples, input_features)
        y = ...  # shape: (n_samples, n_classes) one-hot
        return x, y

    def evaluate(self, n_samples=100, epochs=100, **kwargs):
        """Run complete evaluation."""
        x, y = self.generate_dataset(n_samples)

        config = CascadeCorrelationConfig(
            input_size=x.shape[1],
            output_size=y.shape[1],
            **kwargs
        )
        network = CascadeCorrelationNetwork(config=config)

        history = network.fit(x, y, epochs=epochs)
        accuracy = network.get_accuracy(x, y)

        return {
            'accuracy': accuracy,
            'history': history,
            'network': network,
        }
```

### Adding New Activation Functions

1. Define the activation function:

```python
# src/cascor_constants/constants_activation/constants_activation.py

import torch

def custom_activation(x):
    """Custom activation function."""
    return torch.sigmoid(x) * 2 - 1  # Example: scaled sigmoid

# Add to activation dictionary
ACTIVATION_FUNCTIONS = {
    'tanh': torch.tanh,
    'sigmoid': torch.sigmoid,
    'relu': torch.relu,
    'custom': custom_activation,
}
```

2. Use in configuration:

```python
from cascor_constants.constants_activation.constants_activation import ACTIVATION_FUNCTIONS

config = CascadeCorrelationConfig(
    activation_function=ACTIVATION_FUNCTIONS['custom']
)
```

### Adding New Serialization Formats

1. Create a new serializer class:

```python
# src/snapshots/json_serializer.py

import json
import torch

class CascadeJSONSerializer:
    """JSON serialization for network state."""

    @staticmethod
    def save(network, filepath):
        """Save network to JSON."""
        state = {
            'config': network.config.__dict__,
            'hidden_units': len(network.hidden_units),
            'weights': {
                'output': network.output_weights.tolist(),
                # ... more weights
            },
        }
        with open(filepath, 'w') as f:
            json.dump(state, f)

    @staticmethod
    def load(filepath):
        """Load network from JSON."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        # Reconstruct network
        ...
```

2. Add methods to network class:

```python
def save_to_json(self, filepath):
    CascadeJSONSerializer.save(self, filepath)

@classmethod
def load_from_json(cls, filepath):
    return CascadeJSONSerializer.load(filepath)
```

---

## Coding Conventions

### File Headers

All Python files include standardized headers:

```python
#!/usr/bin/env python
#####################################################################
# Project:       Juniper
# Sub-Project:   JuniperCascor
# Application:   juniper_cascor
# Purpose:       Juniper Project Cascade Correlation Neural Network
#
# Author:        Paul Calnon
# Version:       0.3.21 (0.7.3)
# File Name:     example_module.py
# File Path:     juniper_cascor/src/module/
#
# Date Created:  YYYY-MM-DD
# Last Modified: YYYY-MM-DD HH:MM:SS TZ
#
# License:       MIT License
# Copyright:     Copyright (c) 2024,2025,2026 Paul Calnon
#
# Description:
#     Brief description of the module.
#
#####################################################################
```

### Import Ordering

Follow the standard Python import order:

```python
# 1. Standard library imports
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# 2. Third-party imports
import numpy as np
import torch
import h5py

# 3. Local application imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cascor_constants.constants import _DEFAULT_LEARNING_RATE
from log_config.logger.logger import Logger
from cascade_correlation.cascade_correlation_config import CascadeCorrelationConfig
```

### Type Hints

Use type hints for all public methods:

```python
def fit(
    self,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = None,
    x_val: torch.Tensor = None,
    y_val: torch.Tensor = None,
) -> dict:
    """Train the network."""
    ...

def _calculate_correlation(
    self,
    output: torch.Tensor,
    residual_error: torch.Tensor,
) -> Tuple[float, torch.Tensor, torch.Tensor, float, float]:
    """Calculate Pearson correlation (internal method)."""
    ...
```

### Logging Patterns

Use the custom Logger class consistently:

```python
from log_config.logger.logger import Logger

class MyClass:
    def __init__(self):
        self.logger = Logger

    def my_method(self):
        self.logger.trace("Entering my_method")  # Most detailed
        self.logger.verbose("Processing data")    # Detailed
        self.logger.debug("Variable x = 5")       # Debug info
        self.logger.info("Operation complete")    # Normal info
        self.logger.warning("Potential issue")    # Warnings
        self.logger.error("Something failed")     # Errors
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Constants | UPPER_SNAKE_CASE with underscore prefix | `_CASCOR_LOG_LEVEL_NAME` |
| Classes | PascalCase | `CascadeCorrelationNetwork` |
| Methods/Functions | snake_case | `train_candidates` |
| Private methods | underscore prefix | `_prepare_candidate_input` |
| Constructor params | Name-mangled | `_ClassName__param_name` |

### Constructor Parameter Style

The project uses name-mangled parameters for explicit initialization:

```python
class CandidateUnit:
    def __init__(
        self,
        _CandidateUnit__input_size: int,
        _CandidateUnit__activation_function: callable = torch.tanh,
        _CandidateUnit__learning_rate: float = 0.01,
        _CandidateUnit__random_seed: int = None,
    ):
        self.input_size = _CandidateUnit__input_size
        self.activation_function = _CandidateUnit__activation_function
        # ...
```

This prevents accidental parameter passing and makes intent explicit.

---

**Document Version**: 0.3.21  
**Last Updated**: 2026-01-29
