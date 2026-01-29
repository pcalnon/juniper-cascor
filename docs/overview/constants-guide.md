# Juniper Cascor - Constants Guide

**Version**: 0.3.21  
**Last Updated**: 2026-01-29  
**Purpose**: Complete reference for project constants and configuration

---

## Table of Contents

1. [Overview](#overview)
2. [Constants Hierarchy](#constants-hierarchy)
3. [Logging Constants](#logging-constants)
4. [Model Constants](#model-constants)
5. [Candidate Training Constants](#candidate-training-constants)
6. [Serialization Constants](#serialization-constants)
7. [Problem Constants](#problem-constants)
8. [Activation Function Constants](#activation-function-constants)
9. [Overriding Constants](#overriding-constants)

---

## Overview

The Juniper Cascor project organizes all configuration defaults in `src/cascor_constants/`. Constants are organized by domain and use a consistent naming convention.

### Naming Convention

- All constants are prefixed with underscore: `_CONSTANT_NAME`
- Component prefix indicates scope: `_CASCOR_`, `_CANDIDATE_`, `_NETWORK_`
- UPPER_SNAKE_CASE for all constant names

### Best Practice

**DO NOT edit constants directly.** Instead, override via:

1. Configuration objects (`CascadeCorrelationConfig`)
2. Environment variables (`CASCOR_LOG_LEVEL`)
3. Runtime parameters

---

## Constants Hierarchy

```
src/cascor_constants/
├── constants.py                      # Main aggregator (imports all)
├── constants_activation/
│   └── constants_activation.py       # Activation functions
├── constants_candidates/
│   └── constants_candidates.py       # Candidate training
├── constants_hdf5/
│   └── constants_hdf5.py             # Serialization paths
├── constants_logging/
│   └── constants_logging.py          # Logging configuration
├── constants_model/
│   └── constants_model.py            # Model architecture
└── constants_problem/
    └── constants_problem.py          # Problem-specific defaults
```

### Importing Constants

```python
# Import specific constants
from cascor_constants.constants import (
    _CASCOR_LOG_LEVEL_NAME,
    _DEFAULT_LEARNING_RATE,
    _DEFAULT_CANDIDATE_POOL_SIZE,
)

# Import from specific modules
from cascor_constants.constants_logging.constants_logging import (
    _PROJECT_LOG_LEVEL_NAME_DEBUG,
    _PROJECT_LOG_LEVEL_NAME_INFO,
)
```

---

## Logging Constants

**Location**: `constants_logging/constants_logging.py`

### Log Level Names

| Constant | Value | Description |
|----------|-------|-------------|
| `_PROJECT_LOG_LEVEL_NAME_TRACE` | `"TRACE"` | Most detailed (level 5) |
| `_PROJECT_LOG_LEVEL_NAME_VERBOSE` | `"VERBOSE"` | Very detailed (level 7) |
| `_PROJECT_LOG_LEVEL_NAME_DEBUG` | `"DEBUG"` | Debug info (level 10) |
| `_PROJECT_LOG_LEVEL_NAME_INFO` | `"INFO"` | Normal info (level 20) |
| `_PROJECT_LOG_LEVEL_NAME_WARNING` | `"WARNING"` | Warnings (level 30) |
| `_PROJECT_LOG_LEVEL_NAME_ERROR` | `"ERROR"` | Errors (level 40) |
| `_PROJECT_LOG_LEVEL_NAME_CRITICAL` | `"CRITICAL"` | Critical (level 50) |
| `_PROJECT_LOG_LEVEL_NAME_FATAL` | `"FATAL"` | Fatal errors (level 60) |

### Active Log Level

```python
# Current active log level (change this to adjust logging)
_CASCOR_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_INFO

# To enable debug logging, change to:
# _CASCOR_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_DEBUG
```

### Runtime Override

Use the `CASCOR_LOG_LEVEL` environment variable:

```bash
# Quiet mode
export CASCOR_LOG_LEVEL=WARNING

# Debug mode
export CASCOR_LOG_LEVEL=DEBUG

# Trace mode (most verbose)
export CASCOR_LOG_LEVEL=TRACE
```

### Logging Format Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `_LOG_FORMAT` | `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"` | Log message format |
| `_LOG_DATE_FORMAT` | `"%Y-%m-%d %H:%M:%S"` | Timestamp format |

---

## Model Constants

**Location**: `constants_model/constants_model.py`

### Network Architecture Defaults

| Constant | Default | Description |
|----------|---------|-------------|
| `_DEFAULT_INPUT_SIZE` | `2` | Number of input features |
| `_DEFAULT_OUTPUT_SIZE` | `2` | Number of output classes |
| `_DEFAULT_MAX_HIDDEN_UNITS` | `50` | Maximum network growth |

### Training Defaults

| Constant | Default | Description |
|----------|---------|-------------|
| `_DEFAULT_LEARNING_RATE` | `0.01` | Output layer learning rate |
| `_DEFAULT_EPOCHS_MAX` | `1000` | Maximum training epochs |
| `_DEFAULT_OUTPUT_EPOCHS` | `100` | Epochs per output training phase |
| `_DEFAULT_TARGET_ACCURACY` | `0.95` | Stop training at this accuracy |
| `_DEFAULT_PATIENCE` | `10` | Early stopping patience |

### Display Frequencies

| Constant | Default | Description |
|----------|---------|-------------|
| `_DEFAULT_EPOCH_DISPLAY_FREQUENCY` | `10` | Log every N epochs |
| `_DEFAULT_STATUS_DISPLAY_FREQUENCY` | `10` | Status updates |

### Example Usage

```python
from cascor_constants.constants import (
    _DEFAULT_LEARNING_RATE,
    _DEFAULT_MAX_HIDDEN_UNITS,
)

# Use constants as defaults
config = CascadeCorrelationConfig(
    learning_rate=_DEFAULT_LEARNING_RATE,  # 0.01
    max_hidden_units=_DEFAULT_MAX_HIDDEN_UNITS,  # 50
)
```

---

## Candidate Training Constants

**Location**: `constants_candidates/constants_candidates.py`

### Pool Configuration

| Constant | Default | Description |
|----------|---------|-------------|
| `_DEFAULT_CANDIDATE_POOL_SIZE` | `16` | Candidates per training round |
| `_DEFAULT_CANDIDATES_PER_LAYER` | `1` | Candidates added at once |

### Training Parameters

| Constant | Default | Description |
|----------|---------|-------------|
| `_DEFAULT_CANDIDATE_EPOCHS` | `100` | Epochs per candidate |
| `_DEFAULT_CANDIDATE_LEARNING_RATE` | `0.01` | Candidate learning rate |
| `_DEFAULT_CORRELATION_THRESHOLD` | `0.001` | Minimum correlation for selection |

### Display Settings

| Constant | Default | Description |
|----------|---------|-------------|
| `_DEFAULT_CANDIDATE_DISPLAY_FREQUENCY` | `10` | Log candidate training progress |

### Multiprocessing Settings

| Constant | Default | Description |
|----------|---------|-------------|
| `_CANDIDATE_TRAINING_TIMEOUT` | `30` | Queue operation timeout (seconds) |
| `_MANAGER_START_TIMEOUT` | `10` | Manager startup timeout |
| `_WORKER_SHUTDOWN_TIMEOUT` | `5` | Worker shutdown timeout |

---

## Serialization Constants

**Location**: `constants_hdf5/constants_hdf5.py`

### Default Paths

| Constant | Default | Description |
|----------|---------|-------------|
| `_DEFAULT_SNAPSHOT_DIR` | `"./snapshots"` | Default snapshot directory |
| `_DEFAULT_SNAPSHOT_PREFIX` | `"cascor_"` | Snapshot filename prefix |
| `_DEFAULT_SNAPSHOT_EXTENSION` | `".h5"` | HDF5 file extension |

### Compression Settings

| Constant | Default | Description |
|----------|---------|-------------|
| `_DEFAULT_COMPRESSION` | `"gzip"` | Compression algorithm |
| `_DEFAULT_COMPRESSION_OPTS` | `4` | Compression level (1-9) |

### HDF5 Group Names

| Constant | Value | Description |
|----------|-------|-------------|
| `_HDF5_GROUP_METADATA` | `"metadata"` | Metadata group |
| `_HDF5_GROUP_ARCHITECTURE` | `"architecture"` | Architecture group |
| `_HDF5_GROUP_WEIGHTS` | `"weights"` | Weights group |
| `_HDF5_GROUP_TRAINING_STATE` | `"training_state"` | Training state group |
| `_HDF5_GROUP_RANDOM_STATE` | `"random_state"` | RNG state group |

### Example Usage

```python
from cascor_constants.constants import (
    _DEFAULT_SNAPSHOT_DIR,
    _DEFAULT_COMPRESSION,
)

# Use default snapshot directory
network.save_to_hdf5(
    filepath=f"{_DEFAULT_SNAPSHOT_DIR}/my_network.h5",
    compression=_DEFAULT_COMPRESSION,
)
```

---

## Problem Constants

**Location**: `constants_problem/constants_problem.py`

### Spiral Problem Defaults

| Constant | Default | Description |
|----------|---------|-------------|
| `_DEFAULT_N_POINTS` | `100` | Points per spiral |
| `_DEFAULT_N_SPIRALS` | `2` | Number of spirals |
| `_DEFAULT_NOISE` | `0.1` | Noise level |
| `_DEFAULT_SPIRAL_TURNS` | `1.5` | Number of turns |

### Random State

| Constant | Default | Description |
|----------|---------|-------------|
| `_DEFAULT_RANDOM_SEED` | `42` | Default random seed |

### Example Usage

```python
from cascor_constants.constants import (
    _DEFAULT_N_POINTS,
    _DEFAULT_N_SPIRALS,
    _DEFAULT_RANDOM_SEED,
)

sp = SpiralProblem()
sp.evaluate(
    n_points=_DEFAULT_N_POINTS,
    n_spirals=_DEFAULT_N_SPIRALS,
)
```

---

## Activation Function Constants

**Location**: `constants_activation/constants_activation.py`

### Available Activations

| Constant | Value | Description |
|----------|-------|-------------|
| `_DEFAULT_ACTIVATION_FUNCTION` | `torch.tanh` | Default activation |
| `_ACTIVATION_TANH` | `torch.tanh` | Hyperbolic tangent |
| `_ACTIVATION_SIGMOID` | `torch.sigmoid` | Sigmoid function |
| `_ACTIVATION_RELU` | `torch.relu` | Rectified linear unit |

### Activation Dictionary

```python
ACTIVATION_FUNCTIONS = {
    'tanh': torch.tanh,
    'sigmoid': torch.sigmoid,
    'relu': torch.relu,
}
```

### Example Usage

```python
from cascor_constants.constants_activation.constants_activation import (
    ACTIVATION_FUNCTIONS,
    _DEFAULT_ACTIVATION_FUNCTION,
)

# Use default
config = CascadeCorrelationConfig(
    activation_function=_DEFAULT_ACTIVATION_FUNCTION
)

# Use by name
config = CascadeCorrelationConfig(
    activation_function=ACTIVATION_FUNCTIONS['sigmoid']
)
```

---

## Overriding Constants

### Method 1: Configuration Objects (Recommended)

Use `CascadeCorrelationConfig` to override defaults:

```python
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import (
    CascadeCorrelationConfig,
    OptimizerConfig,
)

# Override specific defaults
config = CascadeCorrelationConfig(
    input_size=4,                    # Override _DEFAULT_INPUT_SIZE
    learning_rate=0.05,              # Override _DEFAULT_LEARNING_RATE
    candidate_pool_size=32,          # Override _DEFAULT_CANDIDATE_POOL_SIZE
    max_hidden_units=100,            # Override _DEFAULT_MAX_HIDDEN_UNITS
    random_seed=123,                 # Override _DEFAULT_RANDOM_SEED
)

network = CascadeCorrelationNetwork(config=config)
```

### Method 2: Environment Variables

For runtime configuration without code changes:

```bash
# Logging level
export CASCOR_LOG_LEVEL=DEBUG

# Then run application
cd src && python main.py
```

### Method 3: Constructor Parameters

Override at instantiation time:

```python
# Direct parameter override (name-mangled style)
network = CascadeCorrelationNetwork(
    _CascadeCorrelationNetwork__input_size=4,
    _CascadeCorrelationNetwork__learning_rate=0.05,
    _CascadeCorrelationNetwork__max_hidden_units=100,
)
```

### What NOT to Do

```python
# ❌ DON'T modify constants directly in source files
# This makes tracking changes difficult and breaks reproducibility

# In constants_model.py:
# _DEFAULT_LEARNING_RATE = 0.05  # Don't do this!

# ✅ DO use configuration objects or environment variables
config = CascadeCorrelationConfig(learning_rate=0.05)
```

---

## Quick Reference Table

| Domain | Key Constants | Override Method |
|--------|---------------|-----------------|
| Logging | `_CASCOR_LOG_LEVEL_NAME` | `CASCOR_LOG_LEVEL` env var |
| Learning Rate | `_DEFAULT_LEARNING_RATE` | Config `learning_rate` |
| Candidate Pool | `_DEFAULT_CANDIDATE_POOL_SIZE` | Config `candidate_pool_size` |
| Max Units | `_DEFAULT_MAX_HIDDEN_UNITS` | Config `max_hidden_units` |
| Correlation | `_DEFAULT_CORRELATION_THRESHOLD` | Config `correlation_threshold` |
| Patience | `_DEFAULT_PATIENCE` | Config `patience` |
| Random Seed | `_DEFAULT_RANDOM_SEED` | Config `random_seed` |
| Compression | `_DEFAULT_COMPRESSION` | `save_to_hdf5()` params |

---

## Related Documentation

- [API Reference](../api/api-reference.md) - API documentation
- [API Schemas](../api/api-schemas.md) - Data structure schemas
- [User Manual](../install/user-manual.md) - Usage guide
- [Source Manual](../source/manual.md) - Source code guide

---

**Document Version**: 0.3.21  
**Last Updated**: 2026-01-29
