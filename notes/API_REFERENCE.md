# Juniper Cascor - API Reference

**Generated**: 2025-01-12  
**Version**: 0.3.2 (0.7.3)  
**Purpose**: Complete API documentation for developers and AI agents

---

## Table of Contents

1. [CascadeCorrelationNetwork](#cascadecorrelationnetwork)
2. [CascadeCorrelationConfig](#cascadecorrelationconfig)
3. [CandidateUnit](#candidateunit)
4. [SpiralProblem](#spiralproblem)
5. [Serialization API](#serialization-api)
6. [Logger API](#logger-api)
7. [Utility Functions](#utility-functions)
8. [Data Classes](#data-classes)
9. [Exceptions](#exceptions)

---

## CascadeCorrelationNetwork

**Location**: `src/cascade_correlation/cascade_correlation.py`

### Constructor

```python
CascadeCorrelationNetwork(
    config: CascadeCorrelationConfig = None,
    # Or individual parameters:
    _CascadeCorrelationNetwork__activation_function: callable = torch.tanh,
    _CascadeCorrelationNetwork__candidate_display_frequency: int = 10:,
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

- `config`: Configuration object (preferred method)
- Individual parameters: Override specific settings

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

**Description**: Train the network using the cascade correlation algorithm.

**Parameters**:

- `x`: Input tensor of shape `(batch_size, input_features)`
- `y`: Target tensor of shape `(batch_size, output_features)`
- `epochs`: Maximum training epochs (uses config default if None)
- `x_val`: Validation input tensor (optional)
- `y_val`: Validation target tensor (optional)

**Returns**: Training history dictionary with keys:

- `train_loss`: List of training losses per epoch
- `train_accuracy`: List of training accuracies
- `val_loss`: List of validation losses (if validation data provided)
- `val_accuracy`: List of validation accuracies
- `hidden_units_added`: List of hidden unit information

**Example**:

```python
history = network.fit(x_train, y_train, epochs=100)
print(f"Final accuracy: {history['train_accuracy'][-1]:.2%}")
```

#### forward

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

**Description**: Perform forward pass through the network.

**Parameters**:

- `x`: Input tensor of shape `(batch_size, input_features)`

**Returns**: Output tensor of shape `(batch_size, output_features)`

**Raises**:

- `ValidationError`: If input tensor is invalid or has wrong shape

**Example**:

```python
predictions = network.forward(x_test)
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

**Description**: Train only the output layer weights.

**Parameters**:

- `x`: Input tensor
- `y`: Target tensor
- `epochs`: Number of training epochs

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

**Description**: Train a pool of candidate units to maximize correlation with residual error.

**Parameters**:

- `x`: Input tensor
- `y`: Target tensor
- `residual_error`: Current residual error tensor

**Returns**: `TrainingResults` dataclass with candidate training statistics

#### get_accuracy

```python
def get_accuracy(
    self,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float
```

**Description**: Calculate classification accuracy on given data.

**Parameters**:

- `x`: Input tensor
- `y`: Target tensor (one-hot encoded)

**Returns**: Accuracy as float between 0.0 and 1.0

**Example**:

```python
accuracy = network.get_accuracy(x_test, y_test)
print(f"Test accuracy: {accuracy:.2%}")
```

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

**Description**: Save network to HDF5 file.

**Parameters**:

- `filepath`: Path to save file
- `include_training_state`: Include training history
- `include_training_data`: Include training data (not recommended)
- `compression`: Compression algorithm
- `compression_opts`: Compression level (1-9)

**Example**:

```python
network.save_to_hdf5("./models/trained_network.h5")
```

#### load_from_hdf5 (classmethod)

```python
@classmethod
def load_from_hdf5(cls, filepath: str) -> CascadeCorrelationNetwork
```

**Description**: Load network from HDF5 file.

**Parameters**:

- `filepath`: Path to saved file

**Returns**: Loaded `CascadeCorrelationNetwork` instance

**Example**:

```python
network = CascadeCorrelationNetwork.load_from_hdf5("./models/trained_network.h5")
```

#### create_snapshot

```python
def create_snapshot(self) -> str
```

**Description**: Create a timestamped snapshot of the network.

**Returns**: Path to created snapshot file

### Properties

| Property         | Type         | Description                                |
| ---------------- | ------------ | ------------------------------------------ |
| `hidden_units`   | list         | List of installed hidden unit dictionaries |
| `output_weights` | torch.Tensor | Output layer weight matrix                 |
| `output_bias`    | torch.Tensor | Output layer bias vector                   |
| `input_size`     | int          | Number of input features                   |
| `output_size`    | int          | Number of output features                  |
| `history`        | dict         | Training history                           |

---

## CascadeCorrelationConfig

**Location**: `src/cascade_correlation/cascade_correlation_config/cascade_correlation_config.py`

### Constructor: CascadeCorrelationConfig

```python
CascadeCorrelationConfig(
    input_size: int = 2,
    output_size: int = 2,
    learning_rate: float = 0.01,
    candidate_learning_rate: float = 0.01,
    max_hidden_units: int = 50,
    candidate_pool_size: int = 16,
    correlation_threshold: float = 0.001,
    patience: int = 10,
    candidate_epochs: int = 100,
    output_epochs: int = 100,
    epochs_max: int = 1000,
    random_seed: int = 42,
    activation_function: callable = torch.tanh,
    target_accuracy: float = 0.95,
    generate_plots: bool = True,
    # ... additional parameters
)
```

### Class Methods

#### create_simple_config

```python
@classmethod
def create_simple_config(
    cls,
    input_size: int = 2,
    output_size: int = 2,
    learning_rate: float = 0.01,
    candidate_learning_rate: float = 0.01,
    max_hidden_units: int = 50,
    candidate_pool_size: int = 16,
    correlation_threshold: float = 0.001,
    patience: int = 10,
    candidate_epochs: int = 100,
    output_epochs: int = 100,
    epochs_max: int = 1000,
) -> CascadeCorrelationConfig
```

**Description**: Factory method to create configuration with common defaults.

**Example**:

```python
config = CascadeCorrelationConfig.create_simple_config(
    input_size=2,
    output_size=2,
    learning_rate=0.05
)
```

### Key Attributes

| Attribute                 | Type  | Default | Description                      |
| ------------------------- | ----- | ------- | -------------------------------- |
| `input_size`              | int   | 2       | Number of input features         |
| `output_size`             | int   | 2       | Number of output classes         |
| `learning_rate`           | float | 0.01    | Output layer learning rate       |
| `candidate_learning_rate` | float | 0.01    | Candidate training learning rate |
| `max_hidden_units`        | int   | 50      | Maximum hidden units to add      |
| `candidate_pool_size`     | int   | 16      | Number of candidates per cycle   |
| `correlation_threshold`   | float | 0.001   | Minimum correlation to add unit  |
| `patience`                | int   | 10      | Early stopping patience          |
| `candidate_epochs`        | int   | 100     | Epochs per candidate training    |
| `output_epochs`           | int   | 100     | Epochs for output training       |
| `random_seed`             | int   | 42      | Random seed for reproducibility  |

---

## CandidateUnit

**Location**: `src/candidate_unit/candidate_unit.py`

### Constructor: Candidate Unit

```python
CandidateUnit(
    CandidateUnit__input_size: int = 2,
    CandidateUnit__output_size: int = 2,
    CandidateUnit__learning_rate: float = 0.01,
    CandidateUnit__epochs: int = 100,
    CandidateUnit__epochs_max: int = 1000,
    CandidateUnit__patience: int = 10,
    CandidateUnit__early_stopping: bool = True,
    CandidateUnit__activation_function: callable = torch.tanh,
    CandidateUnit__random_seed: int = 42,
    CandidateUnit__candidate_index: int = 0,
    CandidateUnit__uuid: str = None,
    **kwargs
)
```

### Methods: Candidate Unit

#### train

```python
def train(
    self,
    x: torch.Tensor,
    residual_error: torch.Tensor,
    epochs: int = None,
) -> CandidateTrainingResult
```

**Description**: Train the candidate to maximize correlation with residual error.

**Parameters**:

- `x`: Input tensor (including any previous hidden outputs)
- `residual_error`: Target residual error tensor
- `epochs`: Number of training epochs

**Returns**: `CandidateTrainingResult` dataclass

#### forward: CandidateUnit

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

**Description**: Compute candidate output for given input.

**Parameters**:

- `x`: Input tensor

**Returns**: Output tensor

### Properties: Candidate Unit

| Property        | Type         | Description               |
| --------------- | ------------ | ------------------------- |
| `weights`       | torch.Tensor | Connection weights        |
| `bias`          | torch.Tensor | Unit bias                 |
| `correlation`   | float        | Best achieved correlation |
| `activation_fn` | callable     | Activation function       |
| `uuid`          | str          | Unique identifier         |

---

## SpiralProblem

**Location**: `src/spiral_problem/spiral_problem.py`

### Constructor: Spiral Problem

```python
SpiralProblem(
    _SpiralProblem__n_points: int = 100,
    _SpiralProblem__n_spirals: int = 2,
    _SpiralProblem__n_rotations: int = 3,
    _SpiralProblem__noise: float = 0.1,
    _SpiralProblem__clockwise: bool = True,
    _SpiralProblem__train_ratio: float = 0.8,
    _SpiralProblem__test_ratio: float = 0.2,
    _SpiralProblem__random_seed: int = 42,
    _SpiralProblem__log_config: LogConfig = None,
    # ... additional parameters
)
```

### Methods: Spiral Problem

#### evaluate

```python
def evaluate(
    self,
    n_points: int = None,
    n_spirals: int = None,
    n_rotations: int = None,
    clockwise: bool = None,
    distribution: float = None,
    random_value_scale: float = None,
    default_origin: float = None,
    default_radius: float = None,
    train_ratio: float = None,
    test_ratio: float = None,
    noise: float = None,
    plot: bool = True,
) -> dict
```

**Description**: Run complete spiral problem evaluation.

**Parameters**:

- `n_points`: Points per spiral
- `n_spirals`: Number of spirals
- `n_rotations`: Spiral rotations
- `clockwise`: Rotation direction
- `noise`: Noise level
- `plot`: Generate visualization

**Returns**: Evaluation results dictionary

**Example**:

```python
sp = SpiralProblem()
results = sp.evaluate(n_points=100, n_spirals=2, plot=True)
print(f"Accuracy: {results['accuracy']:.2%}")
```

#### generate_spiral_dataset

```python
def generate_spiral_dataset(
    self,
    n_points: int,
    n_spirals: int,
    n_rotations: int,
    noise: float,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**Description**: Generate spiral dataset.

**Returns**: Tuple of (x, y) tensors

---

## Serialization API

**Location**: `src/snapshots/`

### CascadeHDF5Serializer

```python
class CascadeHDF5Serializer:
    def save_network(
        self,
        network: CascadeCorrelationNetwork,
        filepath: str,
        include_training_state: bool = True,
        include_training_data: bool = False,
    ) -> None
    
    def load_network(self, filepath: str) -> CascadeCorrelationNetwork
    
    def verify_snapshot(self, filepath: str) -> dict
```

### HDF5Utils

```python
class HDF5Utils:
    @staticmethod
    def list_networks_in_directory(directory: str) -> list
    
    @staticmethod
    def compare_networks(path1: str, path2: str) -> dict
    
    @staticmethod
    def cleanup_old_files(directory: str, keep_count: int = 10) -> int
```

### CLI Commands

```bash
# Save network
python -m snapshots.snapshot_cli save <input.pkl> <output.h5>

# Load network
python -m snapshots.snapshot_cli load <input.h5>

# Verify snapshot
python -m snapshots.snapshot_cli verify <snapshot.h5>

# List snapshots
python -m snapshots.snapshot_cli list <directory>

# Compare snapshots
python -m snapshots.snapshot_cli compare <file1.h5> <file2.h5>

# Cleanup old files
python -m snapshots.snapshot_cli cleanup <directory> --keep 5
```

---

## Logger API

**Location**: `src/log_config/logger/logger.py`

### Logger Class

Singleton logger with custom log levels.

### Log Levels

| Level    | Value | Method                 |
| -------- | ----- | ---------------------- |
| FATAL    | 60    | `Logger.fatal(msg)`    |
| CRITICAL | 50    | `Logger.critical(msg)` |
| ERROR    | 40    | `Logger.error(msg)`    |
| WARNING  | 30    | `Logger.warning(msg)`  |
| INFO     | 20    | `Logger.info(msg)`     |
| DEBUG    | 10    | `Logger.debug(msg)`    |
| VERBOSE  | 7     | `Logger.verbose(msg)`  |
| TRACE    | 5     | `Logger.trace(msg)`    |

### Usage

```python
from log_config.logger.logger import Logger

# Class-level logging (no instance needed)
Logger.info("Information message")
Logger.debug("Debug message")
Logger.trace("Trace message")

# Set log level
Logger.set_level("DEBUG")
Logger.set_level("TRACE")
```

### LogConfig Class

```python
from log_config.log_config import LogConfig

config = LogConfig(
    _LogConfig__log_config=logging.config,
    _LogConfig__log_file_name="app.log",
    _LogConfig__log_file_path="./logs",
    _LogConfig__log_level_name="INFO",
)

logger = config.get_logger()
```

---

## Utility Functions

**Location**: `src/utils/utils.py`

### save_dataset

```python
def save_dataset(
    x: torch.Tensor,
    y: torch.Tensor,
    file_path: str,
) -> None
```

Save dataset to file.

### load_dataset

```python
def load_dataset(file_path: str) -> Tuple[torch.Tensor, torch.Tensor]
```

Load dataset from file.

### display_progress

```python
def display_progress(display_frequency: int) -> callable
```

Create progress check function.

**Example**:

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

### convert_to_numpy

```python
def convert_to_numpy(
    x: torch.Tensor,
    y: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]
```

Convert tensors to NumPy arrays.

### convert_to_tensor

```python
def convert_to_tensor(
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]
```

Convert NumPy arrays to tensors.

---

## Data Classes

**Location**: Various modules

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
    candidate: Optional[any] = None
    best_corr_idx: int = -1
    all_correlations: list[float] = field(default_factory=list)
    norm_output: Optional[torch.Tensor] = None
    norm_error: Optional[torch.Tensor] = None
    numerator: float = 0.0
    denominator: float = 1.0
    success: bool = True
    epochs_completed: int = 0
    error_message: Optional[str] = None
```

### CandidateParametersUpdate

```python
@dataclass
class CandidateParametersUpdate:
    x: torch.Tensor = None
    y: torch.Tensor = None
    residual_error: torch.Tensor = None
    learning_rate: float = 0.01
    norm_output: torch.Tensor = None
    norm_error: torch.Tensor = None
    best_corr_idx: int = -1
    numerator: float = 0.0
    denominator: float = 1.0
    success: bool = True
```

### CandidateCorrelationCalculation

```python
@dataclass
class CandidateCorrelationCalculation:
    correlation: float = 0.0
    best_corr_idx: int = -1
    best_norm_output: torch.Tensor = None
    best_norm_error: torch.Tensor = None
    numerator: float = 0.0
    denominator: float = 0.0
    output: torch.Tensor = None
    residual_error: torch.Tensor = None
```

### ValidateTrainingInputs

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
```

### ValidateTrainingResults

```python
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
    pass
```

**Common Causes**:

- Invalid input/output size
- Invalid learning rate
- Incompatible parameter combinations

### TrainingError

```python
class TrainingError(Exception):
    """Raised when training fails."""
    pass
```

**Common Causes**:

- NaN values in tensors
- Convergence failure
- Resource exhaustion

### ValidationError

```python
class ValidationError(Exception):
    """Raised when input validation fails."""
    pass
```

**Common Causes**:

- None tensors
- Wrong tensor shape
- Invalid tensor values (NaN, Inf)

### Usage Example

```python
from cascade_correlation_exceptions.cascade_correlation_exceptions import (
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
sp = SpiralProblem()
results = sp.evaluate(n_points=100, n_spirals=2)
```

---

**Document Generated By**: AI Analysis Agent  
**Last Updated**: 2025-01-12
