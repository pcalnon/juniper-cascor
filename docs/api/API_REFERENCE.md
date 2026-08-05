# Juniper Cascor - API Reference

**Version**: 0.3.21
**Last Updated**: 2026-04-05
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
9. [Training Service API (FastAPI)](#training-service-api-fastapi)
10. [Utility Functions](#utility-functions)
11. [Data Classes](#data-classes)
12. [Exceptions](#exceptions)

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
| `init_output_weights`     | `str`                      | `"zero"`| New hidden-unit output weight init mode (`"zero"` or `"random"`) |
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
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor = None,
    y_val: torch.Tensor = None,
    max_epochs: int = None,
    epochs: int = None,
    max_iterations: int = None,
    early_stopping: bool = True,
) -> dict
```

Train the network using the cascade correlation algorithm.

**Parameters**:

- `x_train`: Input tensor of shape `(batch_size, input_features)`
- `y_train`: Target tensor of shape `(batch_size, output_features)` (one-hot encoded)
- `x_val`: Validation input tensor (optional)
- `y_val`: Validation target tensor (optional)
- `max_epochs`: Maximum training epochs (uses config defaults when `None`)
- `epochs`: Backward-compatible alias for `max_epochs`
- `max_iterations`: Maximum cascade growth iterations (uses network/config default when `None`)
- `early_stopping`: Enable/disable early stopping behavior

**Returns**: Training history dictionary:

```python
{
    'train_loss': List[float],      # Loss per epoch
    'train_accuracy': List[float],  # Accuracy per epoch
    'value_loss': List[float],      # If validation provided
    'value_accuracy': List[float],  # If validation provided
    'hidden_units_added': List[dict]  # Unit info per addition
}
```

`fit()` uses canonical internal history keys `value_loss` and `value_accuracy`.
Lifecycle REST snapshots (`GET /v1/metrics`) expose these as `val_loss` and
`val_accuracy` for API payload readability.

**Example**:

```python
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

config = CascadeCorrelationConfig(input_size=2, output_size=2)
network = CascadeCorrelationNetwork(config=config)
history = network.fit(x_train, y_train, max_epochs=100, max_iterations=250)
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
    on_epoch_callback: Optional[Callable[..., None]] = None,
) -> float
```

Train only the output layer weights (used internally).

**Returns**: Final loss value

**Optional callback**:

- `on_epoch_callback`: Receives throttled progress updates:

```python
on_epoch_callback(epoch: int, epochs: int, loss: float) -> None
```

**Emission cadence**:

- Callback is called on epoch `1`, every `25` epochs (`26`, `51`, ...), and the final epoch.
- Epoch value is 1-based.

**Example**:

```python
def on_output_epoch(epoch: int, epochs: int, loss: float) -> None:
    print(f"[output] {epoch}/{epochs} loss={loss:.6f}")

final_loss = network.train_output_layer(
    x_train,
    y_train,
    epochs=100,
    on_epoch_callback=on_output_epoch,
)
```

#### grow_network

```python
def grow_network(
    self,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    max_iterations: int = 1000,
    early_stopping: bool = True,
    patience_counter: int = 0,
    best_value_loss: float = float("inf"),
    x_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    on_grow_iteration_callback: Optional[Callable[..., None]] = None,
) -> ValidateTrainingResults
```

Grow the network by iteratively training candidate units and adding selected units.

**Returns**: `ValidateTrainingResults`

**Early-stopping behavior**:

- With validation tensors (`x_val` and `y_val`), stopping is driven by validation loss/patience plus max-hidden-units and target-accuracy checks.
- Without validation tensors, stopping is driven by training loss/patience (using `convergence_threshold`) plus max-hidden-units and target-accuracy checks.

**Optional callback**:

- `on_grow_iteration_callback`: Receives per-iteration grow progress:

```python
on_grow_iteration_callback(
    iteration: int,
    max_iterations: int,
    best_correlation: float,
    candidates_trained: int,
    candidates_total: int,
    phase_detail: str,
) -> None
```

**Callback semantics**:

- `iteration` is zero-based (first grow iteration is `0`).
- `phase_detail` is currently emitted as `"adding_candidate"` for grow-step updates.

**Example**:

```python
def on_grow_iter(
    iteration: int,
    max_iterations: int,
    best_correlation: float,
    candidates_trained: int,
    candidates_total: int,
    phase_detail: str,
) -> None:
    print(
        f"[grow] iter={iteration}/{max_iterations} "
        f"corr={best_correlation:.6f} "
        f"candidates={candidates_trained}/{candidates_total} "
        f"phase_detail={phase_detail}"
    )

result = network.grow_network(
    x_train=x_train,
    y_train=y_train,
    max_epochs=50,
    on_grow_iteration_callback=on_grow_iter,
)
```

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
    init_output_weights: str = "zero",
    output_epochs: int = 100,
    epochs_max: int = 1000,
    max_iterations: int = 1000,
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

`init_output_weights` controls how newly added hidden-unit connections into the output layer are initialized during network growth:

- `"zero"`: zero-initialize only the newly added rows, then copy existing output weights forward (default).
- `"random"`: initialize newly added rows from `torch.randn(...) * 0.1`, then copy existing output weights forward.

Constraint: this setting affects only growth events that add hidden units. It does not retroactively reinitialize existing output weights.

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

## Training Service API (FastAPI)

**Location**: `src/api/`
**Stability**: Stable (service interface), semi-stable (field additions possible)

### Service Startup Guardrails

The server binds to `JUNIPER_CASCOR_HOST` / `JUNIPER_CASCOR_PORT` (`127.0.0.1:8200` by default). During FastAPI lifespan startup, `enforce_bind_attestation_guard()` refuses to start when the host is non-loopback (for example `0.0.0.0`, `::`, or a non-local hostname) unless at least one bind attestation is set — `JUNIPER_CASCOR_LOOPBACK_PUBLISH_ATTESTED=true` (the port is reachable only via a loopback-only host publish) or `JUNIPER_CASCOR_AUTH_PROXY_ATTESTED=true` (a fronting authenticating reverse proxy terminates access).

Use loopback for local development:

```bash
cd src
JUNIPER_CASCOR_HOST=127.0.0.1 JUNIPER_CASCOR_PORT=8200 python server.py
```

Set `JUNIPER_CASCOR_LOOPBACK_PUBLISH_ATTESTED=true` when the port is published on loopback only, or `JUNIPER_CASCOR_AUTH_PROXY_ATTESTED=true` when an authenticating reverse proxy fronts the exposed port. Without at least one attestation, startup raises `NonLoopbackBindError` before background training or WebSocket services begin accepting traffic. There is no warning-only mode.

All REST responses use the standard response envelope:

```python
{
    "status": "success",
    "data": ...,
    "meta": {
        "timestamp": <unix_timestamp>,
        "version": "0.4.0",
    },
}
```

### Service Startup and WebSocket Admission

- REST and WebSocket authentication use the `X-API-Key` header when `JUNIPER_CASCOR_API_KEYS` is configured. Auth is disabled when no API keys are configured.
- `JUNIPER_CASCOR_HOST` defaults to `127.0.0.1`. If it is set to a non-loopback address such as `0.0.0.0`, startup fails with `NonLoopbackBindError` unless `JUNIPER_CASCOR_LOOPBACK_PUBLISH_ATTESTED=true` or `JUNIPER_CASCOR_AUTH_PROXY_ATTESTED=true`.
- Set `JUNIPER_CASCOR_LOOPBACK_PUBLISH_ATTESTED=true` when a loopback-only host-publish fronts the service, or `JUNIPER_CASCOR_AUTH_PROXY_ATTESTED=true` when an authenticating reverse proxy does. This guard runs before the server accepts connections.
- `SecurityHeadersMiddleware` adds always-on `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`, and a restrictive CSP to every HTTP response. HSTS is added only when `X-Forwarded-Proto: https` is present (TLS-terminator footgun if that header is omitted).
- WebSocket admission uses `JUNIPER_CASCOR_WS_MAX_CONNECTIONS_GLOBAL` (default 200) across `/ws/training`, `/ws/control`, and `/ws/v1/workers`. `/ws/control` also uses `JUNIPER_CASCOR_WS_MAX_CONNECTIONS_PER_IDENTITY` (default 5), keyed on a non-reversible digest of the `X-API-Key`.
- Over-cap WebSocket attempts close with `1013`. The peer-IP cap remains DoS-dampening only; behind Docker NAT, clients can share one bridge-gateway IP bucket.
- Worker `register` messages must present a `worker_id` matching `^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$`; failures close with `4008`. The string is stored as `client_name` only — the server assigns the registry id.
- Canopy staged configs use plural generator names (`spirals`/`moons`); `_translate_staged_config` aliases them to juniper-data keys (`spiral`/`moon`) at fetch time. See [JUNIPER_CASCOR_API_REFERENCE — Staged dataset dialect](JUNIPER_CASCOR_API_REFERENCE.md#staged-dataset-dialect-canopy--juniper-data).

### Training Lifecycle Endpoints

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/v1/training/start` | `POST` | Start async training (inline data or dataset source) |
| `/v1/training/stop` | `POST` | Request stop |
| `/v1/training/pause` | `POST` | Pause active training |
| `/v1/training/resume` | `POST` | Resume paused training |
| `/v1/training/reset` | `POST` | Reset lifecycle state and metric buffer |
| `/v1/training/status` | `GET` | Return state machine, monitor, and training-state snapshots |
| `/v1/training/params` | `GET` | Get runtime training params |
| `/v1/training/params` | `PATCH` | Update runtime-modifiable params |
| `/v1/training/dataset` | `POST` | Stage canopy-dialect dataset config for next start |
| `/v1/training/dataset` | `DELETE` | Cancel staged dataset config |
| `/v1/training/dataset/pending` | `GET` | Read staged dataset config (or null) |
| `/v1/training/dataset/live` | `POST` | In-flight live dataset swap (experimental gate) |

### Training Limit Semantics

`epochs_max` and `max_iterations` are separate controls:

- `epochs_max`: Epoch budget used for output-layer optimization cycles.
- `max_iterations`: Upper bound on cascade growth iterations in `grow_network()`.

The following interfaces accept `max_iterations`:

- `POST /v1/network` (`NetworkCreateRequest`)
- `PATCH /v1/training/params` (`TrainingParamUpdateRequest`)
- In-process Python API: `CascadeCorrelationConfig.max_iterations` and `CascadeCorrelationNetwork.fit(..., max_iterations=...)`

Current behavior note:

- `GET /v1/training/params` does not currently include `max_iterations` in its response payload even though PATCH accepts it and the network uses it.

### Metrics Endpoints

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/v1/metrics` | `GET` | Latest metrics snapshot |
| `/v1/metrics/history` | `GET` | Metrics history (`count` query optional) |

### `/v1/training/status` Data Shape

`data` includes:

- `state_machine`: FSM status/phase summary
- `monitor`: Monitor runtime summary
- `training_state`: Thread-safe lifecycle state
- `network_loaded`: Whether a network exists
- `training_active`: Whether lifecycle is currently started

`training_state` fields include:

- `status`, `phase`, `current_epoch`, `current_step`
- `learning_rate`, `max_hidden_units`, `max_epochs`, `max_iterations`
- `phase_detail`, `phase_started_at`
- `grow_iteration`, `grow_max`
- `best_correlation`, `candidates_trained`, `candidates_total`
- `candidate_epoch`, `candidate_total_epochs`

`phase_detail` currently uses these values during active training:

- `training_output`: output-layer training is running
- `training_candidates`: candidate workers are actively training
- `adding_candidate`: best candidate is being installed into the cascade
- `""` (empty string): no active phase detail

### `/v1/metrics/history` Entry Example

```python
{
    "epoch": 26,
    "timestamp": "2026-03-29T12:34:56.789012",
    "loss": 0.0842,
    "accuracy": None,
    "learning_rate": 0.01,
    "hidden_units": 3,
    "phase": "output",
    "validation_loss": None,
    "validation_accuracy": None,
}
```

### Accuracy Nullability

`accuracy` can be `null` (`None` in Python) for output-phase callback emissions where only loss is emitted.

### WebSocket Training Stream

`/ws/training` pushes real-time training updates. The maintained wire-protocol
details live in [JUNIPER_CASCOR_API_REFERENCE.md](JUNIPER_CASCOR_API_REFERENCE.md#ws-wstraining);
this section is a compact in-process API summary.

Typical fresh-connect sequence:

1. `connection_established`
2. `initial_status`
3. `state`
4. `initial_metrics`
5. ongoing broadcast messages (`metrics`, `cascade_add`, `candidate_progress`, `event`)

Message envelope:

```python
{
    "type": "<message_type>",
    "timestamp": <unix_timestamp>,
    "data": {...},
}
```

Broadcast messages include replay metadata (`seq`, `emitted_at_monotonic`).
Clients can send `pong`, `subscribe_metrics`, or a connect-time `resume`
request; training control commands belong on `/ws/control`.

### WebSocket Admission Caps

WebSocket connection caps are admission controls for availability and fairness; they are not authentication. Over-cap handshakes close with code `1013`.

| Setting | Default | Applies to | Notes |
|---------|---------|------------|-------|
| `JUNIPER_CASCOR_WS_MAX_CONNECTIONS_GLOBAL` | `200` | `/ws/training`, `/ws/control`, `/ws/v1/workers` combined | Stack-absolute cap that survives Docker NAT and should exceed expected clients plus worker fleet size. |
| `JUNIPER_CASCOR_WS_MAX_CONNECTIONS_PER_IDENTITY` | `5` | `/ws/control` | Keyed on a non-reversible per-process HMAC of the presented `X-API-Key`; anonymous callers rely on global and per-IP caps. |
| `JUNIPER_CASCOR_WS_MAX_CONNECTIONS_PER_IP` | `5` | Manager-routed sockets, including `/ws/training` | DoS dampening only. Behind Docker NAT all clients can share the bridge-gateway IP, so this can become one shared bucket. |

`/ws/v1/workers` is global-cap-only for this layer: worker fleets may share a machine token, and the server-assigned `worker_id` is not available until after the admission point. Worker capacity is still bounded by the global cap and by worker-registry limits.

### Lifecycle Failure Handling Path

Training exceptions are handled at the monitored `fit()` wrapper layer in the lifecycle manager.

- `TrainingLifecycleManager._run_training()` executes `network.fit(...)` and allows exceptions to propagate.
- `TrainingLifecycleManager._install_monitoring_hooks()` wraps `network.fit` with `monitored_fit(...)`, which owns:
- state transitions (`START`, `FAILED`, `COMPLETED`, `STOPPED`)
- `training_state` updates (`status`, `phase`)
- WebSocket/REST-visible broadcast updates

This keeps error handling centralized and avoids duplicate failure transitions in multiple call paths.

### `candidate_progress` Message Payload

Candidate worker progress updates are emitted with:

```python
{
    "type": "candidate_progress",
    "timestamp": <unix_timestamp>,
    "data": {
        "candidate_id": int,
        "candidate_uuid": str,
        "epoch": int,           # 1-based candidate epoch
        "total_epochs": int,
        "correlation": float,
    },
}
```

Emission behavior:

- Progress is throttled in candidate training: epoch `1`, every `50` epochs (`51`, `101`, ...), and final epoch.
- Delivery is best-effort when queues are saturated (updates may be dropped to keep workers non-blocking).

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
    iteration: int
    max_iterations: int
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
**Last Updated**: 2026-04-05
