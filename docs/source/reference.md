# Source Code Reference

This document provides a technical reference for the internal architecture, data structures, conventions, and implementation details of the Juniper Cascor project.

---

## 1. Key Internal Invariants

### Hidden Unit Representation Format

Hidden units are stored as a list of dictionaries in `self.hidden_units`. Each dictionary contains:

```python
{
    "weights": torch.Tensor,     # Shape: (input_size + num_previous_hidden,)
    "bias": torch.Tensor,        # Scalar tensor
    "correlation": float         # Best correlation score when added
}
```

**Invariant**: The number of inputs to each hidden unit grows incrementally. The nth hidden unit receives `input_size + (n-1)` inputs (original inputs + all previous hidden unit outputs).

### Tensor Shapes Expected Throughout

| Component | Tensor | Shape |
|-----------|--------|-------|
| Input data | `x` | `(batch_size, input_features)` |
| Target data | `y` | `(batch_size, output_size)` |
| Output weights | `output_weights` | `(input_size + num_hidden_units, output_size)` |
| Output bias | `output_bias` | `(output_size,)` |
| Residual error | `residual_error` | `(batch_size, output_size)` |
| Hidden unit weights | `weights` | `(input_size + num_previous_hidden,)` |
| Candidate output | `output` | `(batch_size, 1)` |

**Key Invariant**: Output weight dimensions grow by 1 each time a hidden unit is added:

- Initial: `(input_size, output_size)`
- After adding N units: `(input_size + N, output_size)`

### Multiprocessing Manager Lifecycle

The `CandidateTrainingManager` follows a strict lifecycle:

1. **Initialization**: Manager attributes set to `None` in `_init_multiprocessing()`
2. **Start**: `_start_manager()` creates the manager server and obtains queue proxies
3. **Active**: Tasks distributed via `_task_queue`, results collected from `_result_queue`
4. **Stop**: `_stop_manager()` calls `shutdown()` and resets all references to `None`

```python
# Manager lifecycle states
self._manager = None      # Not started
self._task_queue = None   # No queue proxy
self._result_queue = None # No queue proxy
```

**Invariant**: Queue proxies are only valid while the manager is running. Always check `self._manager is not None` before queue operations.

---

## 2. Data Structures

### Network State Representation

The `CascadeCorrelationNetwork` maintains its state through several key attributes:

```python
# Architecture
self.config: CascadeCorrelationConfig  # Immutable configuration
self.hidden_units: List[Dict]          # List of added hidden units
self.output_weights: torch.Tensor      # Trainable output layer weights
self.output_bias: torch.Tensor         # Trainable output layer bias

# Training state
self.history: Dict[str, List]          # Training metrics over time
self.uuid: str                         # Unique network identifier
self.snapshot_counter: int             # HDF5 snapshot version

# Multiprocessing
self._manager: CandidateTrainingManager  # Manager server instance
self._task_queue: Queue                  # Task distribution queue
self._result_queue: Queue                # Result collection queue
self._mp_ctx: multiprocessing.Context    # Process context (forkserver)
```

### Candidate Pool Management

Candidate units are managed through the following dataclasses:

#### `CandidateTrainingResult`

```python
@dataclass
class CandidateTrainingResult:
    candidate_id: int = -1              # Unique ID within pool
    candidate_uuid: Optional[str] = None # UUID for tracking
    correlation: float = 0.0            # Final correlation score
    candidate: Optional[Any] = None     # Trained CandidateUnit object
    best_corr_idx: int = -1             # Index of best correlation output
    all_correlations: List[float]       # Correlations per output
    norm_output: Optional[torch.Tensor] # Mean-centered output
    norm_error: Optional[torch.Tensor]  # Mean-centered error
    numerator: float = 0.0              # Covariance component
    denominator: float = 1.0            # Normalization factor
    success: bool = True                # Training success flag
    epochs_completed: int = 0           # Number of epochs completed
    error_message: Optional[str] = None # Error description if failed
```

#### `TrainingResults` (Aggregated)

```python
@dataclass
class TrainingResults:
    epochs_completed: int               # Total epochs run
    candidate_ids: List[int]            # All candidate IDs
    candidate_uuids: List[str]          # All candidate UUIDs
    correlations: List[float]           # All correlation scores
    candidate_objects: List[Any]        # All trained candidates
    best_candidate_id: int              # Winner ID
    best_candidate_uuid: str            # Winner UUID
    best_correlation: float             # Highest correlation
    best_candidate: Optional[Any]       # Winner object
    success_count: int                  # Successful trainings
    successful_candidates: int          # Alias for success_count
    failed_count: int                   # Failed trainings
    error_messages: List[str]           # All error messages
    max_correlation: float              # Alias for best_correlation
    start_time: datetime.datetime       # Training start timestamp
    end_time: datetime.datetime         # Training end timestamp
```

### Training History Structure

The `history` dictionary tracks metrics throughout network lifetime:

```python
self.history = {
    "train_loss": [],          # List[float] - Training loss per epoch
    "value_loss": [],          # List[float] - Validation loss per epoch
    "train_accuracy": [],      # List[float] - Training accuracy per epoch
    "value_accuracy": [],      # List[float] - Validation accuracy per epoch
    "hidden_units_added": [],  # List[Dict] - Info about each added unit
}
```

**Note**: HDF5 serialization normalizes `val_*` keys to `value_*` for compatibility.

---

## 3. Error Handling Conventions

### Exception Hierarchy

All custom exceptions inherit from `CascadeCorrelationError`:

```python
class CascadeCorrelationError(Exception):
    """Base exception for Cascade Correlation Network errors."""

class NetworkInitializationError(CascadeCorrelationError):
    """Raised when network initialization fails."""

class TrainingError(CascadeCorrelationError):
    """Raised when training encounters a critical error."""

class ValidationError(CascadeCorrelationError, ValueError):
    """Raised when input validation fails.

    Dual inheritance for compatibility with code expecting ValueError.
    """

class ConfigurationError(CascadeCorrelationError):
    """Raised when configuration parameters are invalid."""
```

### Which Exceptions Raised Where

| Exception | Location | Trigger |
|-----------|----------|---------|
| `ValidationError` | `_validate_tensor_input()` | None tensors, wrong type, empty tensors, NaN/Inf values |
| `ValidationError` | `_validate_tensor_shapes()` | Non-2D tensors, mismatched batch sizes, wrong feature count |
| `ValidationError` | `_validate_numeric_parameter()` | Out-of-range values, wrong types |
| `TrainingError` | `grow_network()` | Validation failures during training loop |
| `TrainingError` | `_calculate_residual_error_safe()` | Errors during residual calculation |
| `ConfigurationError` | `set_uuid()` | Attempting to change UUID after initialization |
| `ConfigurationError` | Config setters | Invalid configuration values |

### Error Propagation Patterns

1. **Wrap and re-raise**: Low-level exceptions are caught and wrapped in domain-specific exceptions:

   ```python
   try:
       validate_training_results = self.validate_training(inputs)
   except Exception as e:
       self.logger.error(f"Exception: {e}")
       raise TrainingError from e
   ```

2. **Fallback with logging**: Multiprocessing errors trigger sequential fallback:

   ```python
   try:
       results = self._execute_parallel_training(tasks, process_count)
   except Exception as e:
       self.logger.error(f"Error: {e}")
       self.logger.warning("Falling back to sequential training")
       results = self._execute_sequential_training(tasks)
   ```

3. **Dummy results on total failure**: If both parallel and sequential fail:

   ```python
   except Exception as seq_error:
       self.logger.error(f"Sequential also failed: {seq_error}")
       results = self._get_dummy_results(len(tasks))
   ```

### Logging of Errors

All errors are logged before being raised or handled:

- `logger.error()` for exceptions with full traceback
- `logger.warning()` for fallback situations
- Tracebacks are formatted using `traceback.format_exc()`

---

## 4. Pickling and Serialization

### `__getstate__` / `__setstate__` Patterns

#### CascadeCorrelationNetwork

```python
def __getstate__(self):
    """Remove non-picklable items for multiprocessing."""
    state = self.__dict__.copy()
    # Non-picklable objects
    state.pop("logger", None)
    state.pop("plotter", None)
    state.pop("log_config", None)

    # Display functions (closures)
    state.pop("_network_display_progress", None)
    state.pop("_status_display_progress", None)
    state.pop("_candidate_display_progress", None)

    # Activation functions (local closures)
    state.pop("activation_fn", None)
    state.pop("activation_fn_no_diff", None)

    # Multiprocessing objects
    state.pop("_manager", None)
    state.pop("_task_queue", None)
    state.pop("_result_queue", None)
    state.pop("_mp_ctx", None)

    # Large training data
    state.pop("_training_data", None)
    state.pop("_validation_data", None)

    return state

def __setstate__(self, state):
    """Restore state and reinitialize non-picklable objects."""
    self.__dict__.update(state)

    # Reinitialize logger
    from log_config.logger.logger import Logger
    Logger.set_level(self.log_level_name if hasattr(self, "log_level_name") else "INFO")
    self.logger = Logger

    # Reinitialize activation function
    self._init_activation_function()

    # Reinitialize plotter
    if not hasattr(self, "plotter"):
        from cascor_plotter.cascor_plotter import CascadeCorrelationPlotter
        self.plotter = CascadeCorrelationPlotter(logger=self.logger)

    # Reinitialize display functions
    from utils.utils import display_progress
    # ... recreate display progress functions
```

#### CandidateUnit

```python
def __getstate__(self):
    """Remove non-picklable items for multiprocessing."""
    state = self.__dict__.copy()
    state.pop('logger', None)
    state.pop('_candidate_display_progress', None)
    state.pop('_candidate_display_status', None)
    return state

def __setstate__(self, state):
    """Restore instance from serialized state."""
    self.__dict__.update(state)
    self.logger = Logger
    self.logger.set_level(self.log_level_name)
    # Display functions recreated lazily in train()
```

### Non-Picklable Objects Handling

| Object Type | Handling Strategy |
|-------------|-------------------|
| Logger instances | Excluded, recreated from module-level `Logger` |
| Display functions | Excluded, recreated lazily when needed |
| Multiprocessing managers | Excluded, reinitialize on demand |
| Queue proxies | Excluded, obtained fresh from manager |
| Closures | Replaced with picklable wrapper classes |
| Plotters | Excluded, recreated from class |

### Activation Function Wrapper

The `ActivationWithDerivative` class solves the multiprocessing pickling issue for activation functions:

```python
class ActivationWithDerivative:
    """Picklable wrapper for activation functions with derivatives."""

    ACTIVATION_MAP = {
        'tanh': torch.tanh,
        'sigmoid': torch.sigmoid,
        'relu': torch.relu,
        'ReLU': torch.nn.ReLU(),
        'Tanh': torch.nn.Tanh(),
        # ... all standard PyTorch activations
    }

    def __init__(self, activation_fn):
        self.activation_fn = activation_fn
        self._activation_name = self._get_activation_name(activation_fn)

    def __call__(self, x, derivative: bool = False):
        if derivative:
            # Return analytical derivative for known functions
            # or numerical approximation for others
            ...
        return self.activation_fn(x)

    def __getstate__(self):
        """Store only the name for serialization."""
        return {'_activation_name': self._activation_name}

    def __setstate__(self, state):
        """Reconstruct function from name."""
        self._activation_name = state['_activation_name']
        self.activation_fn = self.ACTIVATION_MAP.get(
            self._activation_name,
            torch.nn.ReLU()  # Fallback
        )
```

**Key insight**: Only the string name is serialized; the actual function is reconstructed from the static `ACTIVATION_MAP` on unpickling.

---

## 5. Multiprocessing Details

### ForkServer Context Usage

The project uses `forkserver` as the default multiprocessing context:

```python
# From constants_model.py
_PROJECT_MODEL_CANDIDATE_TRAINING_CONTEXT = "forkserver"
```

**Initialization in `_init_multiprocessing()`:**

```python
context_type = self.config.candidate_training_context_type or "forkserver"
self._mp_ctx = mp.get_context(context_type)

if context_type == "forkserver":
    self._mp_ctx.set_forkserver_preload([
        "os", "uuid", "torch", "numpy",
        "random", "logging", "datetime"
    ])
```

**Why forkserver?**

- Clean process isolation without socket inheritance issues from `fork`
- Better than `spawn` for repeated process creation (performance)
- Custom Manager classes work properly in Python 3.14.2+

### Manager and Queue Setup

```python
class CandidateTrainingManager(BaseManager):
    """Custom manager for handling candidate training queues."""

    def start(self, method: str = None, initializer=None, initargs=()):
        if method is not None:
            # Validate method is supported
            valid_methods = {"fork", "spawn", "forkserver"}
            if method not in valid_methods:
                raise ValueError(f"Invalid start method: {method}")
            mp.get_context(method)  # Verify platform support
        return super().start(initializer=initializer, initargs=initargs)

# Register picklable factory functions (not lambdas)
CandidateTrainingManager.register("get_task_queue", callable=_create_task_queue)
CandidateTrainingManager.register("get_result_queue", callable=_create_result_queue)
```

**Queue Factory Functions** (module-level for pickling):

```python
_task_queue = None
_result_queue = None

def _create_task_queue():
    global _task_queue
    if _task_queue is None:
        _task_queue = Queue()
    return _task_queue

def _create_result_queue():
    global _result_queue
    if _result_queue is None:
        _result_queue = Queue()
    return _result_queue
```

### Timeout Handling

**Configuration constants:**

```python
_PROJECT_MODEL_WORKER_STANDBY_SLEEPYTIME = 2.0
_PROJECT_MODEL_TASK_QUEUE_TIMEOUT = 5.0
_PROJECT_MODEL_SHUTDOWN_TIMEOUT = 10.0  # 2x task queue timeout
```

**Result collection with timeouts:**

```python
def _collect_training_results(
    self,
    result_queue: Queue,
    num_tasks: int,
    queue_timeout: float = 60.0,     # Total timeout for all results
    request_timeout: float = 1.0,    # Per-get timeout
) -> list:
    results = []
    deadline = time.time() + queue_timeout

    while collected_results < num_tasks and time.time() < deadline:
        try:
            result = result_queue.get(timeout=request_timeout)
            results.append(result)
        except Empty:
            continue  # Retry until deadline
        except Exception as e:
            self.logger.error(f"Error collecting result: {e}")
            break

    return results
```

### Sequential Fallback

The system automatically falls back to sequential processing when parallel fails:

```python
def _execute_candidate_training(self, tasks: list, process_count: int) -> list:
    try:
        if process_count > 1:
            results = self._execute_parallel_training(tasks, process_count)
            if not results:
                raise RuntimeError("Parallel processing failed to return results")
        else:
            results = self._execute_sequential_training(tasks)
    except Exception as e:
        self.logger.warning("Parallel training failed, falling back to sequential")
        try:
            results = self._execute_sequential_training(tasks)
        except Exception as seq_error:
            # Last resort: create dummy failure results
            results = self._get_dummy_results(len(tasks))

    return results
```

**Fallback triggers:**

- Empty results from parallel processing
- Queue timeouts
- Worker process crashes
- Manager startup failures

---

## 6. Performance Considerations

### Hot Paths

The following code paths are executed frequently and should be optimized:

1. **Forward pass** (`forward(x)`)
   - Called every training iteration
   - Iterates through all hidden units
   - Tensor operations should use in-place when possible

2. **Correlation calculation** (`_calculate_correlation()`)
   - Called for each candidate, each epoch
   - Involves tensor flattening, mean/variance calculations
   - Uses epsilon (`1e-8`) to prevent division by zero

3. **Residual error calculation** (`calculate_residual_error()`)
   - Called every epoch during network growth
   - Difference between forward pass output and targets

4. **Output layer training loop**
   - Gradient computation and weight updates
   - Minimize Python overhead in inner loop

### Memory Management

**Tensor allocation patterns:**

- Hidden unit weights grow incrementally (no pre-allocation)
- Output weights are reallocated when hidden units are added
- Training data is excluded from pickling to reduce serialization overhead

**Large object handling:**

```python
# Excluded from __getstate__ to reduce memory during multiprocessing
state.pop("_training_data", None)
state.pop("_validation_data", None)
```

**Gradient management:**

```python
# Tensors created with requires_grad for training
self.output_weights = torch.randn(..., requires_grad=True)
self.output_bias = torch.randn(..., requires_grad=True)
```

### Logging Overhead

**Log level control via environment:**

```bash
export CASCOR_LOG_LEVEL=WARNING  # Production/benchmarking
export CASCOR_LOG_LEVEL=DEBUG    # Development
export CASCOR_LOG_LEVEL=TRACE    # Maximum verbosity
```

**Conditional logging in hot paths:**

```python
# Logger calls still execute string formatting even if not logged
# Consider guarding expensive formatting:
if self.logger.isEnabledFor(logging.DEBUG):
    self.logger.debug(f"Expensive: {large_tensor.tolist()}")
```

**Extended log levels** (from low to high):

- TRACE (5)
- VERBOSE (8)
- DEBUG (10)
- INFO (20)
- WARNING (30)
- ERROR (40)
- CRITICAL (50)
- FATAL (60)

**Recommendation**: Use `WARNING` or higher for production runs and benchmarking to minimize logging overhead.

---

## See Also

- [API Reference](../api/index.md) - Public API documentation
- [Testing Guide](../testing/index.md) - Test suite documentation
- [Installation Guide](../install/index.md) - Setup instructions
