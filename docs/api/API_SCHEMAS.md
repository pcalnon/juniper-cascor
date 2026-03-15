# Juniper Cascor - API Schemas

**Version**: 0.3.21  
**Last Updated**: 2026-01-29  
**Purpose**: Data schema documentation for serialization and data structures

---

## Table of Contents

1. [HDF5 Snapshot Schema](#hdf5-snapshot-schema)
2. [Training History Schema](#training-history-schema)
3. [Configuration Schema](#configuration-schema)
4. [Data Class Schemas](#data-class-schemas)
5. [JuniperData Artifact Schemas](#juniperdata-artifact-schemas)
6. [Backward Compatibility](#backward-compatibility)

---

## HDF5 Snapshot Schema

The HDF5 snapshot format stores the complete state of a `CascadeCorrelationNetwork` for persistence and reproducibility.

### Top-Level Structure

```
snapshot.h5
├── metadata/                    # Required
│   ├── uuid                     # Network unique identifier (string)
│   ├── version                  # Cascor version (string)
│   ├── creation_time            # ISO timestamp (string)
│   └── checksum                 # Data integrity hash (bytes)
├── architecture/                # Required
│   ├── input_size               # int
│   ├── output_size              # int
│   ├── hidden_unit_count        # int
│   └── activation_function      # string (function name)
├── config/                      # Required
│   └── json_config              # JSON-encoded config (string)
├── weights/                     # Required
│   ├── output_weights           # float32 array
│   ├── output_bias              # float32 array
│   └── hidden_units/            # Group for each unit
│       ├── unit_0/
│       │   ├── weights          # float32 array
│       │   ├── bias             # float32 array
│       │   └── checksum         # bytes
│       ├── unit_1/
│       └── ...
├── training_state/              # Optional (if include_training_state=True)
│   ├── history/                 # Training metrics
│   │   ├── train_loss           # float32 array
│   │   ├── train_accuracy       # float32 array
│   │   ├── val_loss             # float32 array (optional)
│   │   └── val_accuracy         # float32 array (optional)
│   ├── epochs_completed         # int
│   ├── patience_counter         # int
│   ├── best_loss                # float
│   └── snapshot_counter         # int
├── random_state/                # Required for deterministic resume
│   ├── python_random            # bytes (pickled state)
│   ├── numpy_random             # bytes (pickled state)
│   ├── torch_random             # bytes (torch state)
│   └── torch_cuda_random        # bytes (optional, CUDA state)
└── training_data/               # Optional (if include_training_data=True)
    ├── x_train                  # float32 array
    ├── y_train                  # float32 array
    ├── x_val                    # float32 array (optional)
    └── y_val                    # float32 array (optional)
```

### Metadata Group

| Dataset | Type | Description |
|---------|------|-------------|
| `uuid` | `string` | UUID4 identifying this network instance |
| `version` | `string` | Cascor version that created this file |
| `creation_time` | `string` | ISO 8601 timestamp |
| `checksum` | `bytes` | SHA-256 hash of critical data |

### Architecture Group

| Dataset | Type | Description |
|---------|------|-------------|
| `input_size` | `int64` | Number of input features |
| `output_size` | `int64` | Number of output classes |
| `hidden_unit_count` | `int64` | Current number of hidden units |
| `activation_function` | `string` | Name of activation function (e.g., "tanh") |

### Config Group

| Dataset | Type | Description |
|---------|------|-------------|
| `json_config` | `string` | JSON-encoded `CascadeCorrelationConfig` |

**Excluded from JSON** (non-serializable):

- `activation_functions_dict`
- `log_config`
- `logger`

### Weights Group

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `output_weights` | `float32` | `(hidden+input, output)` | Output layer weights |
| `output_bias` | `float32` | `(output,)` | Output layer bias |

**Hidden Unit Subgroups** (`hidden_units/unit_N/`):

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `weights` | `float32` | `(connections,)` | Unit input weights |
| `bias` | `float32` | `(1,)` | Unit bias |
| `checksum` | `bytes` | - | SHA-256 of weights+bias |

### Training State Group (Optional)

| Dataset | Type | Description |
|---------|------|-------------|
| `epochs_completed` | `int64` | Total epochs trained |
| `patience_counter` | `int64` | Current patience counter |
| `best_loss` | `float64` | Best validation loss seen |
| `snapshot_counter` | `int64` | Number of snapshots created |

**History Subgroup**:

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `train_loss` | `float32` | `(epochs,)` | Training loss per epoch |
| `train_accuracy` | `float32` | `(epochs,)` | Training accuracy per epoch |
| `val_loss` | `float32` | `(epochs,)` | Validation loss (if provided) |
| `val_accuracy` | `float32` | `(epochs,)` | Validation accuracy (if provided) |

### Random State Group

| Dataset | Type | Description |
|---------|------|-------------|
| `python_random` | `bytes` | `pickle.dumps(random.getstate())` |
| `numpy_random` | `bytes` | `pickle.dumps(numpy.random.get_state())` |
| `torch_random` | `bytes` | `torch.get_rng_state().numpy().tobytes()` |
| `torch_cuda_random` | `bytes` | CUDA RNG state (if CUDA available) |

### Compression Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `compression` | `"gzip"` | Compression algorithm |
| `compression_opts` | `4` | Compression level (1-9) |

**Trade-offs**:

- Level 1: Fastest writes, largest files
- Level 4: Balanced (default)
- Level 9: Smallest files, slowest writes

---

## Training History Schema

The training history is a dictionary returned by `fit()` and stored in snapshots.

### Schema

```python
{
    'train_loss': List[float],      # Required
    'train_accuracy': List[float],  # Required
    'val_loss': List[float],        # Optional (if validation data provided)
    'val_accuracy': List[float],    # Optional (if validation data provided)
    'hidden_units_added': List[dict]  # Required
}
```

### hidden_units_added Entry Schema

```python
{
    'epoch': int,               # Epoch when unit was added
    'correlation': float,       # Correlation with residual error
    'candidate_id': int,        # Index in candidate pool
    'candidate_uuid': str,      # UUID of the candidate
    'connections': int,         # Number of input connections
}
```

### Example

```python
history = network.fit(x_train, y_train, epochs=100, x_val=x_val, y_val=y_val)

# Access training metrics
print(f"Final loss: {history['train_loss'][-1]}")
print(f"Final accuracy: {history['train_accuracy'][-1]}")

# Access hidden unit info
for unit in history['hidden_units_added']:
    print(f"Added unit at epoch {unit['epoch']} with correlation {unit['correlation']:.4f}")
```

---

## Configuration Schema

### CascadeCorrelationConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input_size` | `int` | `2` | Number of input features |
| `output_size` | `int` | `2` | Number of output classes |
| `learning_rate` | `float` | `0.01` | Output layer learning rate |
| `candidate_learning_rate` | `float` | `0.01` | Candidate training learning rate |
| `candidate_pool_size` | `int` | `16` | Candidates per training round |
| `candidate_epochs` | `int` | `100` | Epochs per candidate |
| `output_epochs` | `int` | `100` | Epochs for output layer training |
| `epochs_max` | `int` | `1000` | Maximum total epochs |
| `max_hidden_units` | `int` | `50` | Maximum network growth |
| `correlation_threshold` | `float` | `0.001` | Minimum correlation for selection |
| `patience` | `int` | `10` | Early stopping patience |
| `target_accuracy` | `float` | `0.95` | Stop when accuracy reached |
| `random_seed` | `int` | `42` | For reproducibility |
| `generate_plots` | `bool` | `True` | Enable visualization |
| `activation_function` | `callable` | `torch.tanh` | Hidden unit activation |
| `optimizer_config` | `OptimizerConfig` | `None` | Optimizer settings |

### OptimizerConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `optimizer_type` | `str` | `'Adam'` | `'Adam'`, `'SGD'`, `'RMSprop'`, `'AdamW'` |
| `learning_rate` | `float` | `0.01` | Learning rate |
| `momentum` | `float` | `0.9` | Momentum (SGD, RMSprop) |
| `beta1` | `float` | `0.9` | Beta1 (Adam, AdamW) |
| `beta2` | `float` | `0.999` | Beta2 (Adam, AdamW) |
| `weight_decay` | `float` | `0.0` | Weight decay |
| `epsilon` | `float` | `1e-8` | Epsilon for stability |
| `amsgrad` | `bool` | `False` | Use AMSGrad variant |

### JSON Serialization

When saved to HDF5, the config is serialized as JSON with these exclusions:

```python
EXCLUDED_CONFIG_FIELDS = [
    'activation_functions_dict',
    'log_config',
    'logger',
]
```

After loading, these fields are re-initialized from defaults.

---

## Data Class Schemas

### TrainingResults

Returned by `train_candidates()` method.

| Field | Type | Description |
|-------|------|-------------|
| `epochs_completed` | `int` | Total epochs across all candidates |
| `candidate_ids` | `List[int]` | Pool indices of trained candidates |
| `candidate_uuids` | `List[str]` | UUIDs of trained candidates |
| `correlations` | `List[float]` | Final correlations achieved |
| `candidate_objects` | `List[CandidateUnit]` | Trained candidate objects |
| `best_candidate_id` | `int` | Index of best candidate |
| `best_candidate_uuid` | `str` | UUID of best candidate |
| `best_correlation` | `float` | Best correlation achieved |
| `best_candidate` | `CandidateUnit` | Best candidate object |
| `success_count` | `int` | Successfully trained candidates |
| `successful_candidates` | `int` | Alias for success_count |
| `failed_count` | `int` | Failed candidate trainings |
| `error_messages` | `List[str]` | Error messages from failures |
| `max_correlation` | `float` | Maximum correlation seen |
| `start_time` | `datetime` | Training start time |
| `end_time` | `datetime` | Training end time |

### CandidateTrainingResult

Returned by `CandidateUnit.train_detailed()`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `candidate_id` | `int` | `-1` | Pool index |
| `candidate_uuid` | `str` | `None` | Candidate UUID |
| `correlation` | `float` | `0.0` | Final correlation |
| `candidate` | `CandidateUnit` | `None` | Trained candidate object |
| `best_corr_idx` | `int` | `-1` | Output index with best correlation |
| `all_correlations` | `List[float]` | `[]` | Correlation per epoch |
| `norm_output` | `torch.Tensor` | `None` | Normalized output |
| `norm_error` | `torch.Tensor` | `None` | Normalized error |
| `numerator` | `float` | `0.0` | Correlation numerator |
| `denominator` | `float` | `1.0` | Correlation denominator |
| `success` | `bool` | `True` | Training succeeded |
| `epochs_completed` | `int` | `0` | Epochs actually run |
| `error_message` | `str` | `None` | Error if failed |

### CandidateParametersUpdate

Used internally for weight updates.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `x` | `torch.Tensor` | `None` | Input data |
| `y` | `torch.Tensor` | `None` | Target data |
| `residual_error` | `torch.Tensor` | `None` | Current residual |
| `learning_rate` | `float` | `0.01` | Update learning rate |
| `norm_output` | `torch.Tensor` | `None` | Normalized output |
| `norm_error` | `torch.Tensor` | `None` | Normalized error |
| `best_corr_idx` | `int` | `-1` | Best correlation index |
| `numerator` | `float` | `0.0` | Correlation numerator |
| `denominator` | `float` | `1.0` | Correlation denominator |
| `success` | `bool` | `True` | Update succeeded |

### CandidateCorrelationCalculation

Result of correlation calculation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `correlation` | `float` | `0.0` | Computed correlation |
| `best_corr_idx` | `int` | `-1` | Best output index |
| `best_norm_output` | `torch.Tensor` | `None` | Best normalized output |
| `best_norm_error` | `torch.Tensor` | `None` | Best normalized error |
| `numerator` | `float` | `0.0` | Correlation numerator |
| `denominator` | `float` | `0.0` | Correlation denominator |
| `output` | `torch.Tensor` | `None` | Raw output |
| `residual_error` | `torch.Tensor` | `None` | Residual error |

### ValidateTrainingInputs

Input validation for training.

| Field | Type | Description |
|-------|------|-------------|
| `epoch` | `int` | Current epoch |
| `max_epochs` | `int` | Maximum epochs |
| `patience_counter` | `int` | Current patience |
| `early_stopping` | `bool` | Early stopping enabled |
| `train_accuracy` | `float` | Current training accuracy |
| `train_loss` | `float` | Current training loss |
| `best_value_loss` | `float` | Best validation loss |
| `x_train` | `np.ndarray` | Training inputs |
| `y_train` | `np.ndarray` | Training targets |
| `x_val` | `np.ndarray` | Validation inputs |
| `y_val` | `np.ndarray` | Validation targets |

### ValidateTrainingResults

Result of validation.

| Field | Type | Description |
|-------|------|-------------|
| `early_stop` | `bool` | Should stop early |
| `patience_counter` | `int` | Updated patience |
| `best_value_loss` | `float` | Updated best loss |
| `value_output` | `float` | Validation output |
| `value_loss` | `float` | Validation loss |
| `value_accuracy` | `float` | Validation accuracy |

---

## JuniperData Artifact Schemas

JuniperData provides dataset generation and artifact storage. This section documents the schemas for dataset metadata and NPZ artifacts.

> **Note**: The project uses a clear separation between serialization formats:
>
> - **HDF5** (`.h5`): Network snapshots, model weights, and training state
> - **NPZ** (`.npz`): Dataset artifacts (input features and labels)

### Dataset Metadata Schema

Response from `POST /v1/datasets`:

```python
{
    "id": str,              # Unique dataset identifier
    "generator": str,       # Generator name (e.g., "SpiralGenerator")
    "params": dict,         # Parameters used for generation
    "created_at": str,      # ISO timestamp
    "artifact_path": str,   # Path to stored artifact (if persisted)
}
```

### NPZ Artifact Schema

Downloaded via `download_artifact_npz()`:

| Array Name | Type | Shape | Description |
|------------|------|-------|-------------|
| `x` | float32 | (n_samples, 2) | Input features (x, y coordinates) |
| `y` | float32 | (n_samples, n_classes) | One-hot encoded labels |

---

## Backward Compatibility

### Version Handling

The HDF5 file includes a version field in metadata:

```python
# Reading version
with h5py.File('snapshot.h5', 'r') as f:
    version = f['metadata/version'][()].decode()
```

### Compatibility Matrix

| File Version | Loader Version | Status |
|--------------|----------------|--------|
| 0.3.x | 0.3.21 | ✅ Compatible |
| 0.2.x | 0.3.21 | ⚠️ May require migration |
| 0.1.x | 0.3.21 | ❌ Not compatible |

### Breaking Changes by Version

**0.3.0 → 0.3.5**:

- Changed `CandidateTrainingResult` field names (`candidate_index` → `candidate_id`)
- Fixed history key names (`value_loss` → `val_loss`)

**0.3.5 → 0.3.19**:

- Renamed `constants/` → `cascor_constants/` (code change, not HDF5)

**0.3.19 → 0.3.20**:

- Added profiling module (no HDF5 changes)

### Migration Guidance

If loading older snapshots fails:

1. **Check version**: Read the metadata to identify file version
2. **Common issues**:
   - Missing `random_state/python_random` → Pre-0.1.1 file
   - Wrong history keys → Pre-0.3.5 file
3. **Migration script** (if needed):

```python
import h5py
import pickle

def migrate_old_snapshot(old_path, new_path):
    """Migrate pre-0.3.5 snapshot to current format."""
    with h5py.File(old_path, 'r') as old, h5py.File(new_path, 'w') as new:
        # Copy all groups
        for key in old.keys():
            old.copy(key, new)

        # Fix history keys if needed
        if 'training_state/history/value_loss' in new:
            new.move('training_state/history/value_loss',
                    'training_state/history/val_loss')

        # Update version
        del new['metadata/version']
        new['metadata/version'] = '0.3.21'
```

### Checksum Verification

Each hidden unit has a checksum for data integrity:

```python
import hashlib

def verify_unit_checksum(filepath, unit_index):
    with h5py.File(filepath, 'r') as f:
        unit = f[f'weights/hidden_units/unit_{unit_index}']
        weights = unit['weights'][()]
        bias = unit['bias'][()]
        stored_checksum = unit['checksum'][()]

        # Recompute
        data = weights.tobytes() + bias.tobytes()
        computed = hashlib.sha256(data).digest()

        return stored_checksum == computed
```

If verification fails:

- File may be corrupted
- Do not use for training (may produce incorrect results)
- Restore from backup or earlier snapshot

---

**Document Version**: 0.3.21  
**Last Updated**: 2026-01-29
