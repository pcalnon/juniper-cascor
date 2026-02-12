# Juniper Cascor - API Reference

**Version**: 0.3.21  
**Last Updated**: 2026-01-29  
**Purpose**: Complete API documentation for developers and integrators

---

## Table of Contents

1. [CascadeCorrelationNetwork](#cascadecorrelationnetwork)
2. [CascadeCorrelationConfig](#cascadecorrelationconfig)
3. [CandidateUnit](#candidateunit)
4. [SpiralProblem](#spiralproblem)
5. [JuniperDataClient](#juniperdataclient)
6. [Serialization API](#serialization-api)
7. [Profiling API](#profiling-api)
8. [Logger API](#logger-api)
9. [Utility Functions](#utility-functions)
10. [Data Classes](#data-classes)
11. [Exceptions](#exceptions)

---

## API Stability

| Component                   | Stability    | Notes                               |
| --------------------------- | ------------ | ----------------------------------- |
| `CascadeCorrelationNetwork` | **Stable**   | Public API for training/inference   |
| `CascadeCorrelationConfig`  | **Stable**   | Configuration interface             |
| `CandidateUnit`             | Semi-stable  | Internal, but documented            |
| `SpiralProblem`             | **Stable**   | Example problem interface           |
| `JuniperDataClient`         | Semi-stable  | REST client for JuniperData service |
| Serialization API           | **Stable**   | HDF5 save/load                      |
| Profiling API               | Experimental | New in 0.3.20                       |
| Logger API                  | Semi-stable  | Subject to enhancement              |
| Data Classes                | Semi-stable  | Fields may be added                 |

---

## CascadeCorrelationNetwork

**Location**: `src/cascade_correlation/cascade_correlation.py`  
**Stability**: Stable

The main neural network class implementing the Cascade Correlation algorithm.

### Constructor

```python
CascadeCorrelationNetwork(
    config: CascadeCorrelationConfig = None,
    # Or individual parameters (name-mangled style):
    _CascadeCorrelationNetwork__activation_function: callable = torch.tanh,
    _CascadeCorrelationNetwork__candidate_display_frequency: int = 10,
    _CascadeCorrelationNetwork__candidate_epochs: int = 100,
    _CascadeCorrelationNetwork__candidate_learning_rate: float = 0.01,
    _CascadeCorrelationNetwork__candidate_pool_size: int = 16,
    _CascadeCorrelationNetwork__correlation_threshold: float = 0.001,
    _CascadeCorrelationNetwork__epoch_display_frequency: int = 10,
    _CascadeCorrelationNetwork__epochs_max: int = 1000,
    _CascadeCorrelationNetwork__generate_plots: bool = True,
    _CascadeCorrelationNetwork__input_size: int = 2,
    _CascadeCorrelationNetwork__learning_rate: float = 0.01,
    _CascadeCorrelationNetwork__max_hidden_units: int = 50,
    _CascadeCorrelationNetwork__output_epochs: int = 100,
    _CascadeCorrelationNetwork__output_size: int = 2,
    _CascadeCorrelationNetwork__patience: int = 10,
    _CascadeCorrelationNetwork__random_seed: int = 42,
    _CascadeCorrelationNetwork__status_display_frequency: int = 10,
    _CascadeCorrelationNetwork__target_accuracy: float = 0.95,
    **kwargs
)
```

**Parameters**:

| Parameter                 | Type                       | Default | Description                       |
| ------------------------- | -------------------------- | ------- | --------------------------------- |
| `config`                  | `CascadeCorrelationConfig` | `None`  | Configuration object (preferred)  |
| `input_size`              | `int`                      | `2`     | Number of input features          |
| `output_size`             | `int`                      | `2`     | Number of output classes          |
| `learning_rate`           | `float`                    | `0.01`  | Learning rate for output layer    |
| `candidate_learning_rate` | `float`                    | `0.01`  | Learning rate for candidates      |
| `candidate_pool_size`     | `int`                      | `16`    | Number of candidates per round    |
| `candidate_epochs`        | `int`                      | `100`   | Epochs to train each candidate    |
| `max_hidden_units`        | `int`                      | `50`    | Maximum network growth            |
| `correlation_threshold`   | `float`                    | `0.001` | Minimum correlation for selection |
| `patience`                | `int`                      | `10`    | Early stopping patience           |
| `target_accuracy`         | `float`                    | `0.95`  | Stop training at this accuracy    |
| `random_seed`             | `int`                      | `42`    | For reproducibility               |
| `generate_plots`          | `bool`                     | `True`  | Enable visualization              |

### Methods

#### fit

```python
def fit(
    self,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = None,
    x_val: torch.Tensor = None,
    y_val: torch.Tensor = None,
) -> dict
```

Train the network using the cascade correlation algorithm.

**Parameters**:

- `x`: Input tensor of shape `(batch_size, input_features)`
- `y`: Target tensor of shape `(batch_size, output_features)` (one-hot encoded)
- `epochs`: Maximum training epochs (uses config default if None)
- `x_val`: Validation input tensor (optional)
- `y_val`: Validation target tensor (optional)

**Returns**: Training history dictionary:

```python
{
    'train_loss': List[float],      # Loss per epoch
    'train_accuracy': List[float],  # Accuracy per epoch
    'val_loss': List[float],        # If validation provided
    'val_accuracy': List[float],    # If validation provided
    'hidden_units_added': List[dict]  # Unit info per addition
}
```

**Example**:

```python
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

config = CascadeCorrelationConfig(input_size=2, output_size=2)
network = CascadeCorrelationNetwork(config=config)
history = network.fit(x_train, y_train, epochs=100)
print(f"Final accuracy: {history['train_accuracy'][-1]:.2%}")
```

#### forward

```python
def forward(self, x: torch.Tensor = None) -> torch.Tensor
```

Perform forward pass through the network.

**Parameters**:

- `x`: Input tensor of shape `(batch_size, input_features)`

**Returns**: Output tensor of shape `(batch_size, output_features)`

**Raises**:

- `ValidationError`: If input tensor is None, wrong shape, or contains NaN/Inf

**Example**:

```python
predictions = network.forward(x_test)
predicted_classes = torch.argmax(predictions, dim=1)
```

#### get_accuracy

```python
def get_accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float
```

Calculate classification accuracy on given data.

**Parameters**:

- `x`: Input tensor
- `y`: Target tensor (one-hot encoded)

**Returns**: Accuracy as float between 0.0 and 1.0. Returns `NaN` for empty batches.

**Example**:

```python
accuracy = network.get_accuracy(x_test, y_test)
print(f"Test accuracy: {accuracy:.2%}")
```

#### train_output_layer

```python
def train_output_layer(
    self,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = None,
) -> float
```

Train only the output layer weights (used internally).

**Returns**: Final loss value

#### train_candidates

```python
def train_candidates(
    self,
    x: torch.Tensor,
    y: torch.Tensor,
    residual_error: torch.Tensor,
) -> TrainingResults
```

Train a pool of candidate units to maximize correlation with residual error.

**Returns**: `TrainingResults` dataclass with candidate training statistics

#### save_to_hdf5

```python
def save_to_hdf5(
    self,
    filepath: str,
    include_training_state: bool = True,
    include_training_data: bool = False,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> None
```

Save network to HDF5 file.

**Parameters**:

| Parameter                | Type   | Default  | Description                   |
| ------------------------ | ------ | -------- | ----------------------------- |
| `filepath`               | `str`  | -        | Path to save file             |
| `include_training_state` | `bool` | `True`   | Include training history      |
| `include_training_data`  | `bool` | `False`  | Include training data (large) |
| `compression`            | `str`  | `"gzip"` | Compression algorithm         |
| `compression_opts`       | `int`  | `4`      | Compression level (1-9)       |

**Example**:

```python
network.save_to_hdf5("./models/trained_network.h5")
```

#### load_from_hdf5 (classmethod)

```python
@classmethod
def load_from_hdf5(cls, filepath: str) -> CascadeCorrelationNetwork
```

Load network from HDF5 file.

**Returns**: Loaded `CascadeCorrelationNetwork` instance with full state restored

**Example**:

```python
network = CascadeCorrelationNetwork.load_from_hdf5("./models/trained_network.h5")
# Continue training
network.fit(x_train, y_train, epochs=50)
```

#### create_snapshot

```python
def create_snapshot(self) -> str
```

Create a timestamped snapshot of the network.

**Returns**: Path to created snapshot file

---

## CascadeCorrelationConfig

**Location**: `src/cascade_correlation/cascade_correlation_config/cascade_correlation_config.py`  
**Stability**: Stable

Configuration object for network parameters.

### Constructor: Cascade Correlation Config

```python
CascadeCorrelationConfig(
    input_size: int = 2,
    output_size: int = 2,
    learning_rate: float = 0.01,
    candidate_learning_rate: float = 0.01,
    candidate_pool_size: int = 16,
    candidate_epochs: int = 100,
    output_epochs: int = 100,
    epochs_max: int = 1000,
    max_hidden_units: int = 50,
    correlation_threshold: float = 0.001,
    patience: int = 10,
    target_accuracy: float = 0.95,
    random_seed: int = 42,
    generate_plots: bool = True,
    activation_function: callable = torch.tanh,
    optimizer_config: OptimizerConfig = None,
)
```

### Factory Methods

#### create_simple_config

```python
@classmethod
def create_simple_config(
    cls,
    input_size: int,
    output_size: int,
    learning_rate: float = 0.01,
    random_seed: int = 42,
) -> CascadeCorrelationConfig
```

Create a configuration with sensible defaults.

**Example**:

```python
config = CascadeCorrelationConfig.create_simple_config(
    input_size=4,
    output_size=3,
    learning_rate=0.01
)
```

### OptimizerConfig

Nested configuration for optimizer settings.

```python
@dataclass
class OptimizerConfig:
    optimizer_type: str = 'Adam'  # 'Adam', 'SGD', 'RMSprop', 'AdamW'
    learning_rate: float = 0.01
    momentum: float = 0.9         # For SGD, RMSprop
    beta1: float = 0.9            # For Adam, AdamW
    beta2: float = 0.999          # For Adam, AdamW
    weight_decay: float = 0.0
    epsilon: float = 1e-8
    amsgrad: bool = False         # For Adam, AdamW
```

**Example**:

```python
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import OptimizerConfig

sgd_config = OptimizerConfig(
    optimizer_type='SGD',
    learning_rate=0.1,
    momentum=0.9,
    weight_decay=1e-4
)
config = CascadeCorrelationConfig(optimizer_config=sgd_config)
```

---

## CandidateUnit

**Location**: `src/candidate_unit/candidate_unit.py`  
**Stability**: Semi-stable (internal API)

Represents a candidate hidden unit during network growth.

### Constructor: Candidate Unit

```python
CandidateUnit(
    _CandidateUnit__input_size: int,
    _CandidateUnit__activation_function: callable = torch.tanh,
    _CandidateUnit__learning_rate: float = 0.01,
    _CandidateUnit__random_seed: int = None,
)
```

### Methods: Candidate Unit

#### train

```python
def train(
    self,
    x: torch.Tensor,
    residual_error: torch.Tensor,
    epochs: int = 100,
) -> float
```

Train the candidate to maximize correlation with residual error.

**Returns**: Best correlation achieved (absolute value)

#### train_detailed

```python
def train_detailed(
    self,
    x: torch.Tensor,
    residual_error: torch.Tensor,
    epochs: int = 100,
) -> CandidateTrainingResult
```

Train with detailed result information.

**Returns**: `CandidateTrainingResult` dataclass with full statistics

---

## SpiralProblem

**Location**: `src/spiral_problem/spiral_problem.py`  
**Stability**: Stable

Classic two-spiral classification problem for testing.

> **Note**: Dataset generation now uses the JuniperData service via `JuniperDataClient`. See [JuniperDataClient](#juniperdataclient) for details.

### Constructor: Spiral Problem

```python
SpiralProblem(
    _SpiralProblem__n_points: int = 100,
    _SpiralProblem__n_spirals: int = 2,
    _SpiralProblem__noise: float = 0.1,
)
```

### Methods: Spiral Problem

#### evaluate

```python
def evaluate(
    self,
    n_points: int = 100,
    n_spirals: int = 2,
    noise: float = 0.1,
    epochs: int = 100,
    plot: bool = True,
) -> dict
```

Run complete evaluation pipeline.

**Returns**: Dictionary with training results and accuracy

**Example**:

```python
from spiral_problem.spiral_problem import SpiralProblem

sp = SpiralProblem()
results = sp.evaluate(n_points=100, n_spirals=2, plot=True)
print(f"Final accuracy: {results['accuracy']:.2%}")
```

#### generate_spiral_dataset

```python
def generate_spiral_dataset(
    self,
    n_points: int,
    n_spirals: int,
    noise: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]
```

Generate spiral classification data.

**Returns**: Tuple of (x, y) tensors

---

## JuniperDataClient

**Location**: `src/juniper_data_client/client.py`  
**Stability**: Semi-stable

REST API client for the JuniperData service, used for dataset generation and retrieval.

### Constructor: Juniper Data Client

```python
JuniperDataClient(
    base_url: str = "http://localhost:8100",
    timeout: int = 30
)
```

**Parameters**:

| Parameter  | Type  | Default                   | Description                      |
| ---------- | ----- | ------------------------- | -------------------------------- |
| `base_url` | `str` | `"http://localhost:8100"` | Base URL for JuniperData service |
| `timeout`  | `int` | `30`                      | Request timeout in seconds       |

### Methods: Juniper Data Client

#### create_dataset

```python
def create_dataset(
    self,
    generator: str,
    params: dict,
    persist: bool = True
) -> dict
```

Create a new dataset using the specified generator.

**Parameters**:

| Parameter   | Type   | Default | Description                                  |
| ----------- | ------ | ------- | -------------------------------------------- |
| `generator` | `str`  | -       | Generator type (e.g., `"spiral"`, `"xor"`)   |
| `params`    | `dict` | -       | Generator-specific parameters                |
| `persist`   | `bool` | `True`  | Whether to persist the dataset on the server |

**Returns**: Dictionary with dataset metadata including `dataset_id`

#### download_artifact_npz

```python
def download_artifact_npz(
    self,
    dataset_id: str
) -> Dict[str, np.ndarray]
```

Download dataset artifact as NumPy arrays.

**Parameters**:

| Parameter    | Type  | Description                   |
| ------------ | ----- | ----------------------------- |
| `dataset_id` | `str` | ID of the dataset to download |

**Returns**: Dictionary of NumPy arrays (typically `{"x": ..., "y": ...}`)

### Example Usage

```python
from juniper_data_client.client import JuniperDataClient
import numpy as np

# Create client
client = JuniperDataClient(base_url="http://localhost:8100")

# Create a spiral dataset
result = client.create_dataset(
    generator="spiral",
    params={"n_points": 100, "n_spirals": 2, "noise": 0.1},
    persist=True
)
dataset_id = result["dataset_id"]

# Download the dataset as NumPy arrays
data = client.download_artifact_npz(dataset_id)
x_train = data["x"]
y_train = data["y"]

print(f"Dataset shape: x={x_train.shape}, y={y_train.shape}")
```

---

## Serialization API

**Location**: `src/snapshots/`  
**Stability**: Stable

### CascadeHDF5Serializer

Main serialization class (used internally by network).

### HDF5Utils

Utility functions for HDF5 management.

```python
from snapshots.snapshot_utils import HDF5Utils

# List all snapshots in directory
networks = HDF5Utils.list_networks_in_directory("./snapshots")

# Verify snapshot integrity
info = HDF5Utils.verify_snapshot("snapshot.h5")

# Compare two snapshots
comparison = HDF5Utils.compare_networks("snap1.h5", "snap2.h5")

# Cleanup old files (keep 10 most recent)
deleted = HDF5Utils.cleanup_old_files("./snapshots", keep_count=10)
```

### CLI Tools

```bash
# Save network
python -m snapshots.snapshot_cli save network.pkl snapshot.h5

# Load and verify
python -m snapshots.snapshot_cli load snapshot.h5
python -m snapshots.snapshot_cli verify snapshot.h5

# List snapshots
python -m snapshots.snapshot_cli list ./snapshots/

# Cleanup
python -m snapshots.snapshot_cli cleanup ./snapshots/ --keep 5
```

---

## Profiling API

**Location**: `src/profiling/`  
**Stability**: Experimental (new in 0.3.20)

### ProfileContext

Context manager for cProfile profiling.

```python
from profiling.profiling import ProfileContext

with ProfileContext(output_dir="./profiles", top_n=20):
    network.fit(x_train, y_train)
```

### MemoryTracker

Context manager for memory profiling.

```python
from profiling.profiling import MemoryTracker

with MemoryTracker(top_n=10):
    network.fit(x_train, y_train)
```

### Decorators

```python
from profiling.profiling import profile_function, memory_profile

@profile_function
def my_training_function():
    ...

@memory_profile
def memory_intensive_function():
    ...
```

### Logging Utilities

```python
from profiling.logging_utils import SampledLogger, BatchLogger, log_if_enabled

# Sample 10% of log messages
sampled = SampledLogger(logger, sample_rate=0.1)
sampled.debug("This may or may not be logged")

# Batch log messages
batch = BatchLogger(logger, batch_size=100)
for i in range(1000):
    batch.add("info", f"Message {i}")
batch.flush()

# Conditional logging
log_if_enabled(logger, "debug", f"Expensive: {expensive_computation()}")
```

---

## Logger API

**Location**: `src/log_config/`  
**Stability**: Semi-stable

### Logger Class

Custom logger with extended log levels.

```python
from log_config.logger.logger import Logger

Logger.trace("Detailed trace message")
Logger.verbose("Verbose output")
Logger.debug("Debug message")
Logger.info("Information message")
Logger.warning("Warning message")
Logger.error("Error message")
Logger.critical("Critical error")
Logger.fatal("Fatal error")
```

### Log Levels

| Level    | Value | Description     |
| -------- | ----- | --------------- |
| TRACE    | 5     | Most detailed   |
| VERBOSE  | 7     | Detailed output |
| DEBUG    | 10    | Debugging info  |
| INFO     | 20    | General info    |
| WARNING  | 30    | Warnings        |
| ERROR    | 40    | Errors          |
| CRITICAL | 50    | Critical errors |
| FATAL    | 60    | Fatal errors    |

---

## Utility Functions

**Location**: `src/utils/utils.py`

### display_progress

```python
def display_progress(frequency: int) -> Callable[[int], bool]
```

Create a display progress callback function.

```python
should_display = display_progress(10)
for epoch in range(100):
    if should_display(epoch):
        print(f"Epoch {epoch}")
```

### get_class_distribution

```python
def get_class_distribution(y: torch.Tensor) -> Dict[int, int]
```

Get class distribution from one-hot targets.

### convert_to_numpy / convert_to_tensor

```python
def convert_to_numpy(x: torch.Tensor, y: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]
def convert_to_tensor(x: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]
```

Convert between tensor and array formats.

---

## Data Classes

### TrainingResults

```python
@dataclass
class TrainingResults:
    epochs_completed: int
    candidate_ids: List[int]
    candidate_uuids: List[str]
    correlations: List[float]
    candidate_objects: List[Any]
    best_candidate_id: int
    best_candidate_uuid: str
    best_correlation: float
    best_candidate: Optional[Any]
    success_count: int
    successful_candidates: int
    failed_count: int
    error_messages: List[str]
    max_correlation: float
    start_time: datetime.datetime
    end_time: datetime.datetime
```

### CandidateTrainingResult

```python
@dataclass
class CandidateTrainingResult:
    candidate_id: int = -1
    candidate_uuid: Optional[str] = None
    correlation: float = 0.0
    candidate: Optional[Any] = None
    best_corr_idx: int = -1
    all_correlations: List[float] = field(default_factory=list)
    norm_output: Optional[torch.Tensor] = None
    norm_error: Optional[torch.Tensor] = None
    numerator: float = 0.0
    denominator: float = 1.0
    success: bool = True
    epochs_completed: int = 0
    error_message: Optional[str] = None
```

### ValidateTrainingInputs / ValidateTrainingResults

```python
@dataclass
class ValidateTrainingInputs:
    epoch: int
    max_epochs: int
    patience_counter: int
    early_stopping: bool
    train_accuracy: float
    train_loss: float
    best_value_loss: float
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray

@dataclass
class ValidateTrainingResults:
    early_stop: bool
    patience_counter: int
    best_value_loss: float
    value_output: float
    value_loss: float
    value_accuracy: float
```

---

## Exceptions

**Location**: `src/cascade_correlation/cascade_correlation_exceptions/`

### ConfigurationError

```python
class ConfigurationError(Exception):
    """Raised when network configuration is invalid."""
```

**Common Causes**:

- Invalid input/output size
- Invalid learning rate (≤ 0)
- Incompatible parameter combinations

### TrainingError

```python
class TrainingError(Exception):
    """Raised when training fails."""
```

**Common Causes**:

- NaN values in tensors
- Convergence failure
- Resource exhaustion

### ValidationError

```python
class ValidationError(ValueError):
    """Raised when input validation fails."""
```

**Common Causes**:

- None tensors passed to methods
- Wrong tensor shape
- Invalid tensor values (NaN, Inf)

### Usage Example

```python
from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import (
    ConfigurationError,
    TrainingError,
    ValidationError,
)

try:
    network.fit(x_train, y_train)
except ValidationError as e:
    print(f"Invalid input: {e}")
except TrainingError as e:
    print(f"Training failed: {e}")
except ConfigurationError as e:
    print(f"Bad configuration: {e}")
```

---

## Quick Reference

### Common Operations

```python
# Create and train network
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

config = CascadeCorrelationConfig.create_simple_config(
    input_size=2, output_size=2, learning_rate=0.01
)
network = CascadeCorrelationNetwork(config=config)
history = network.fit(x_train, y_train, epochs=100)

# Evaluate
accuracy = network.get_accuracy(x_test, y_test)
predictions = network.forward(x_test)

# Save/Load
network.save_to_hdf5("model.h5")
loaded = CascadeCorrelationNetwork.load_from_hdf5("model.h5")

# Spiral problem
from spiral_problem.spiral_problem import SpiralProblem
sp = SpiralProblem()
results = sp.evaluate(n_points=100, n_spirals=2)
```

---

## Thread Safety Warning

The `CascadeCorrelationNetwork` class is **NOT thread-safe**. Do not share network instances between threads without proper synchronization. For concurrent training, create separate network instances per thread. The internal multiprocessing for candidate training is handled within the class.

---

**Document Version**: 0.3.21  
**Last Updated**: 2026-01-29
