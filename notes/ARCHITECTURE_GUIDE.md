# Juniper Cascor - Architecture Guide

**Generated**: 2025-01-12  
**Version**: 0.3.2 (0.7.3)  
**Purpose**: System architecture documentation for developers and AI agents

---

## System Overview

Juniper Cascor implements the Cascade Correlation Neural Network algorithm as a modular Python application. The architecture prioritizes research flexibility, transparency, and extensibility.

---

## High-Level Architecture

```bash
┌────────────────────────────────────────────────────────────────────┐
│                         Application Layer                          │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │    main.py      │  │  SpiralProblem   │  │  CLI Tools        │  │
│  │  Entry Point    │  │  Problem Domain  │  │  snapshot_cli     │  │
│  └────────┬────────┘  └────────┬─────────┘  └─────────┬─────────┘  │
└───────────┼────────────────────┼──────────────────────┼────────────┘
            │                    │                      │
┌───────────┼────────────────────┼──────────────────────┼────────────┐
│           ▼                    ▼                      ▼            │
│                          Core Layer                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              CascadeCorrelationNetwork                       │  │
│  │  ┌─────────────────┐  ┌────────────────┐  ┌───────────────┐  │  │
│  │  │ Output Layer    │  │ Hidden Units   │  │ Candidate Pool│  │  │
│  │  │ (trainable)     │  │ (frozen)       │  │ (training)    │  │  │
│  │  └─────────────────┘  └────────────────┘  └───────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │  CandidateUnit  │  │  CascorConfig    │  │  CascorPlotter    │  │
│  │  Hidden Nodes   │  │  Configuration   │  │  Visualization    │  │
│  └─────────────────┘  └──────────────────┘  └───────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
            │                    │                      │
┌───────────┼────────────────────┼──────────────────────┼────────────┐
│           ▼                    ▼                      ▼            │
│                        Infrastructure Layer                        │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │    Logger       │  │   Snapshots      │  │   Constants       │  │
│  │  Custom levels  │  │   HDF5 I/O       │  │   Configuration   │  │
│  └─────────────────┘  └──────────────────┘  └───────────────────┘  │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │    Utils        │  │  RemoteClient    │  │   Exceptions      │  │
│  │  Helpers        │  │  Multiprocessing │  │   Error types     │  │
│  └─────────────────┘  └──────────────────┘  └───────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
            │                    │                      │
┌───────────┼────────────────────┼──────────────────────┼────────────┐
│           ▼                    ▼                      ▼            │
│                        External Dependencies                       │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │     PyTorch     │  │      NumPy       │  │    Matplotlib     │  │
│  │  Tensor ops     │  │  Array ops       │  │    Plotting       │  │
│  └─────────────────┘  └──────────────────┘  └───────────────────┘  │
│  ┌─────────────────┐  ┌──────────────────┐                         │
│  │      h5py       │  │     PyYAML       │                         │
│  │  HDF5 files     │  │  Config files    │                         │
│  └─────────────────┘  └──────────────────┘                         │
└────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### Core Components

#### CascadeCorrelationNetwork

**Responsibility**: Implements the complete Cascade Correlation algorithm

**Key State**:

- `hidden_units`: List of installed hidden unit dictionaries
- `output_weights`: Trainable output layer weights
- `output_bias`: Trainable output layer bias
- `history`: Training history dictionary

**Lifecycle**:

1. Construction with config
2. Training via `fit()`
3. Inference via `forward()`
4. Persistence via `save_to_hdf5()` / `load_from_hdf5()`

#### CandidateUnit

**Responsibility**: Represents a potential hidden unit during training

**Key State**:

- `weights`: Connection weights from inputs
- `bias`: Unit bias
- `correlation`: Best achieved correlation
- `activation_fn`: Activation function

**Lifecycle**:

1. Pool creation during candidate training
2. Training to maximize correlation
3. Best candidate selected and installed
4. Weights frozen after installation

#### CascadeCorrelationConfig

**Responsibility**: Encapsulates all network hyperparameters

**Key Properties**:

- Network dimensions (input_size, output_size)
- Learning rates (network, candidate)
- Training limits (epochs, hidden units)
- Thresholds (correlation, patience)

### Support Components

#### Logger

**Responsibility**: Custom logging with extended levels

**Custom Levels**:

```bash
FATAL    = 60
CRITICAL = 50
ERROR    = 40
WARNING  = 30
INFO     = 20
DEBUG    = 10
VERBOSE  = 7
TRACE    = 5
```

**Usage Pattern**:

```python
from log_config.logger.logger import Logger
Logger.info("Message")
Logger.trace("Detailed trace")
```

#### Snapshot System

**Responsibility**: HDF5-based network serialization

**Components**:

- `snapshot_serializer.py`: Core save/load logic
- `snapshot_utils.py`: Utility functions
- `snapshot_cli.py`: Command-line interface
- `snapshot_common.py`: Shared helpers

**Features**:

- Checksum verification
- RNG state preservation
- Training history persistence
- UUID tracking

---

## Data Flow Diagrams

### Training Flow

```bash
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Input X    │────▶│   forward()  │────▶│  Prediction  │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                                                 ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Target Y   │────▶│    Loss      │◀────│  Prediction  │
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│                   Output Layer Training                  │
│  train_output_layer() - Gradient descent, output weights │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Check Error  │────▶│ If too high  │────▶│ Train        │
│              │     │              │     │ Candidates   │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                                                 ▼
┌──────────────────────────────────────────────────────────┐
│                   Candidate Selection                    │
│  Select candidate with highest correlation to residual   │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│                   Install Hidden Unit                    │
│  Add selected candidate as frozen hidden unit            │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   Repeat     │
                     └──────────────┘
```

### Inference Flow

```bash
┌──────────────┐
│   Input X    │
│  (batch, 2)  │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│                    Hidden Unit 1                         │
│  input = X                                               │
│  output = activation(sum(input * weights) + bias)        │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│                    Hidden Unit 2                         │
│  input = concat(X, hidden1_output)                       │
│  output = activation(sum(input * weights) + bias)        │
└──────────────────────────────────────────────────────────┘
       │
       ▼
      ...
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│                    Hidden Unit N                         │
│  input = concat(X, hidden1_output,..., hiddenN-1_output) │
│  output = activation(sum(input * weights) + bias)        │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│                    Output Layer                          │
│  input = concat(X, all_hidden_outputs)                   │
│  output = matmul(input, output_weights) + output_bias    │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│   Output Y   │
│ (batch, out) │
└──────────────┘
```

---

## Module Dependencies

```bash
main.py
├── constants.constants
├── log_config.log_config.LogConfig
├── log_config.logger.logger.Logger
└── spiral_problem.spiral_problem.SpiralProblem
    ├── cascade_correlation.cascade_correlation.CascadeCorrelationNetwork
    │   ├── cascade_correlation_config.CascadeCorrelationConfig
    │   ├── cascade_correlation_exceptions.*
    │   ├── candidate_unit.candidate_unit.CandidateUnit
    │   ├── cascor_plotter.cascor_plotter.CascadeCorrelationPlotter
    │   ├── snapshots.snapshot_serializer.CascadeHDF5Serializer
    │   ├── log_config.logger.logger.Logger
    │   └── utils.utils
    └── log_config.logger.logger.Logger
```

---

## Configuration Management

### Constants Hierarchy

```bash
constants/
├── constants.py              # Aggregator - imports and re-exports all
├── constants_model/          # Model architecture
│   └── constants_model.py
├── constants_candidates/     # Candidate training
│   └── constants_candidates.py
├── constants_activation/     # Activation functions
│   └── constants_activation.py
├── constants_logging/        # Logging settings
│   └── constants_logging.py
├── constants_problem/        # Problem-specific
│   └── constants_problem.py
└── constants_hdf5/           # Serialization
    └── constants_hdf5.py
```

### Configuration Flow

```bash
constants_model (base values)
    ↓
constants.py (aggregation + overrides)
    ↓
CascadeCorrelationConfig (runtime config object)
    ↓
CascadeCorrelationNetwork (instance configuration)
```

---

## Multiprocessing Architecture

### Candidate Training Pool

```bash
┌─────────────────────────────────────────────────────────────┐
│                     Main Process                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           CandidateTrainingManager                  │    │
│  │  ┌─────────────┐        ┌─────────────┐             │    │
│  │  │ Task Queue  │        │Result Queue │             │    │
│  │  └──────┬──────┘        └──────▲──────┘             │    │
│  └─────────┼──────────────────────┼────────────────────┘    │
└────────────┼──────────────────────┼─────────────────────────┘
             │                      │
             ▼                      │
┌────────────────────────────────────────────────────────────┐
│                     Worker Processes                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │  Worker 1  │  │  Worker 2  │  │  Worker N  │            │
│  │            │  │            │  │            │            │
│  │ Candidate  │  │ Candidate  │  │ Candidate  │            │
│  │  Training  │  │  Training  │  │  Training  │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└────────────────────────────────────────────────────────────┘
```

### Pickling Considerations

Classes implement `__getstate__` and `__setstate__` to handle non-picklable objects:

```python
def __getstate__(self):
    state = self.__dict__.copy()
    state.pop('logger', None)  # Remove non-picklable
    state.pop('_candidate_display_progress', None)
    return state

def __setstate__(self, state):
    self.__dict__.update(state)
    self.logger = Logger  # Reinitialize
```

---

## Serialization Architecture

### HDF5 Structure

```bash
network.h5
├── metadata/
│   ├── version
│   ├── uuid
│   ├── created_at
│   └── format_version
├── architecture/
│   ├── input_size
│   ├── output_size
│   └── num_hidden_units
├── output_layer/
│   ├── weights
│   ├── weights_checksum
│   ├── bias
│   └── bias_checksum
├── hidden_units/
│   ├── unit_0/
│   │   ├── weights
│   │   ├── weights_checksum
│   │   ├── bias
│   │   └── activation_function_name
│   ├── unit_1/
│   │   └── ...
│   └── unit_N/
│       └── ...
├── training_state/ (optional)
│   ├── history
│   ├── epochs_completed
│   └── current_loss
└── rng_state/
    ├── python_random
    ├── numpy_random
    └── torch_random
```

### Serialization Flow

```bash
CascadeCorrelationNetwork
        │
        ▼
save_to_hdf5()
        │
        ▼
CascadeHDF5Serializer.save_network()
        │
        ├── _serialize_metadata()
        ├── _serialize_architecture()
        ├── _serialize_output_layer()
        ├── _serialize_hidden_units()
        ├── _serialize_training_state()
        └── _serialize_rng_state()
        │
        ▼
    network.h5
```

---

## Extension Points

### Adding New Problem Types

1. Create new module in `src/` (e.g., `xor_problem/`)
2. Implement problem class with `evaluate()` method
3. Use `CascadeCorrelationNetwork` for training
4. Add constants to `constants_problem/`

### Adding New Activation Functions

1. Define function in `constants_activation/constants_activation.py`:

```python
_PROJECT_MODEL_ACTIVATION_FUNCTION_NN_NEW = torch.nn.NewActivation()
_PROJECT_MODEL_ACTIVATION_FUNCTION_NAME_NEW = "new"
```

2. Add to functions dictionary:

```python
_PROJECT_MODEL_ACTIVATION_FUNCTIONS_DICT["new"] = _PROJECT_MODEL_ACTIVATION_FUNCTION_NN_NEW
```

### Adding New Serialization Formats

1. Create new serializer in `snapshots/` (e.g., `snapshot_json.py`)
2. Implement `save_network()` and `load_network()` methods
3. Register in network class if needed

---

## Thread Safety

### Current State

- **Not thread-safe**: Network operations assume single-threaded access
- **Process-safe**: Multiprocessing used for candidate training
- **Logging**: Logger class uses singleton pattern (thread-safe)

### Recommendations for Concurrent Use

1. Create separate network instances per thread/process
2. Use multiprocessing for parallel training experiments
3. Serialize access to shared HDF5 files

---

## Memory Management

### Object Lifecycle

```bash
Network Creation
    │
    ├── Config object (small, static)
    ├── Logger reference (singleton)
    ├── Empty hidden_units list
    └── Output weights tensor
    │
Training Phase
    │
    ├── Candidate pool created (N × input_size tensors)
    ├── Training history accumulates
    └── Hidden units grow (1 per cycle)
    │
Post-Training
    │
    ├── Candidate pool released
    ├── History retained
    └── Hidden units retained
```

### Memory Optimization

1. Set `include_training_data=False` when saving
2. Consider history pruning for long training runs
3. Use smaller candidate pool sizes for memory-constrained systems

---

## Error Handling Strategy

### Exception Hierarchy

```bash
Exception
├── ConfigurationError    # Invalid configuration
├── TrainingError         # Training failures
└── ValidationError       # Input validation failures
```

### Error Propagation

1. **Validation layer**: Check inputs immediately
2. **Operation layer**: Wrap operations in try/except
3. **Logging layer**: Log errors before raising
4. **Caller layer**: Handle or propagate appropriately

### Example Pattern

```python
def method(self, x: torch.Tensor):
    # Validation
    if x is None:
        raise ValidationError("x cannot be None")
    
    # Operation with logging
    try:
        result = self._internal_operation(x)
    except Exception as e:
        self.logger.error(f"Operation failed: {e}")
        raise TrainingError(f"Operation failed: {e}") from e
    
    return result
```

---

## Versioning

### Version Scheme

Format: `MAJOR.MINOR.PATCH (INTERNAL.INTERNAL.INTERNAL)`

- **0.3.2**: Public version (pre-1.0, MVP phase)
- **(0.7.3)**: Internal development version

### Compatibility

- HDF5 format versioned in metadata
- Config format validated on load
- Backward compatibility maintained where possible

---

**Document Generated By**: AI Analysis Agent  
**Last Updated**: 2025-01-12
