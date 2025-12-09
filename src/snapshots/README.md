# HDF5 Serialization for CascadeCorrelationNetwork

This package provides comprehensive HDF5 serialization capabilities for `CascadeCorrelationNetwork` objects, allowing complete state capture and restoration.

## Overview

The HDF5 serialization system captures the complete state and structure of a CascadeCorrelationNetwork including:

- **Network architecture**: Input/output sizes, hidden units, activation functions
- **Model parameters**: All weights, biases, and gradients
- **Hidden units**: Individual unit weights, biases, correlations, and activation functions
- **Configuration**: Complete network configuration parameters
- **Random state**: Random number generator states for reproducibility
- **Multiprocessing state**: Configuration for distributed training (restored deterministically)
- **Training history**: Loss curves, accuracy metrics, and unit addition history (optional)

## Key Features

✅ **Complete State Preservation**: Captures all network state for perfect restoration  
✅ **Robust String/Tensor I/O**: Safe handling of different data types and encodings  
✅ **Multiprocessing Support**: Deterministic recreation of multiprocessing configuration  
✅ **Format Versioning**: Forward-compatible file format with version tracking  
✅ **Compression**: Configurable HDF5 compression for smaller files  
✅ **Validation**: Built-in file verification and integrity checking  
✅ **CLI Tools**: Command-line interface for file management  

## Architecture

### Core Components

- **`snapshot_serializer.py`**: Main `CascadeHDF5Serializer` class
- **`snapshot_utils.py`**: Utility functions for file operations (`HDF5Utils`)
- **`snapshot_common.py`**: Robust I/O helpers for strings and tensors
- **`snapshot_cli.py`**: Command-line interface
- **`__init__.py`**: Package initialization

### File Format (Version 2.0)

```bash
network.h5
├── @format = "juniper.cascor"
├── @format_version = "2"
├── @serializer_version = "2.0.0"
├── @created = "2025-01-01T12:00:00"
│
├── meta/                          # Metadata
│   ├── @uuid = "network-uuid"
│   ├── @python_version = "3.x.x"
│   └── @torch_version = "x.x.x"
│
├── config/                        # Configuration
│   ├── config_json               # Full config as JSON
│   ├── @input_size = 2
│   └── @output_size = 1
│
├── arch/                          # Architecture
│   ├── @input_size = 2
│   ├── @output_size = 1
│   ├── @num_hidden_units = 3
│   └── connectivity/             # Connection info
│
├── params/                        # Model parameters
│   └── output_layer/
│       ├── weights               # Output weights tensor
│       └── bias                  # Output bias tensor
│
├── hidden_units/                  # Hidden units
│   ├── @num_units = 3
│   ├── unit_0/
│   │   ├── weights              # Unit weights
│   │   ├── bias                 # Unit bias
│   │   ├── @correlation = 0.85
│   │   └── @activation_function_name = "tanh"
│   └── ...
│
├── random/                        # Random state
│   ├── @seed = 42
│   ├── numpy_state/              # NumPy RNG state
│   ├── torch_state               # PyTorch RNG state
│   └── cuda_states/              # CUDA RNG states (if available)
│
├── mp/                            # Multiprocessing config
│   ├── @role = "server"
│   ├── @start_method = "spawn"
│   ├── @address_host = "127.0.0.1"
│   ├── @address_port = 8000
│   └── queues_to_create          # Queue configuration JSON
│
└── history/                       # Training history (optional)
    ├── train_loss                # Loss arrays
    ├── val_loss
    └── hidden_units_added/       # Unit addition history
```

## Usage

### Basic Save/Load

```python
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

# Create and train network
config = CascadeCorrelationConfig(input_size=2, output_size=1)
network = CascadeCorrelationNetwork(config=config)
# ... train network ...

# Save to HDF5
success = network.save_to_hdf5('network.h5')

# Load from HDF5
loaded_network = CascadeCorrelationNetwork.load_from_hdf5('network.h5')
```

### Advanced Save Options

```python
# Save with training history but exclude training data
network.save_to_hdf5(
    'network.h5',
    include_training_state=True,     # Include loss curves, etc.
    include_training_data=False,     # Exclude training datasets
    compression='gzip',              # Use compression
    compression_opts=6,              # Compression level
    create_backup=True               # Backup existing file
)
```

### Using the Serializer Directly

```python
from cascade_correlation.snapshots.snapshot_serializer import CascadeHDF5Serializer

serializer = CascadeHDF5Serializer()

# Save network
success = serializer.save_network(network, 'network.h5')

# Load network
loaded_network = serializer.load_network('network.h5')

# Verify file
verification = serializer.verify_saved_network('network.h5')
if verification['valid']:
    print(f"Valid network: {verification['input_size']}→{verification['output_size']}")
```

### File Management

```python
from cascade_correlation.hdf5 import HDF5Utils

# Get detailed file information
info = HDF5Utils.get_file_info('network.h5')
print(f"File size: {info['size_mb']:.2f} MB")
print(f"Groups: {len(info['groups'])}, Datasets: {len(info['datasets'])}")

# List networks in directory
networks = HDF5Utils.list_networks_in_directory('./snapshots/')
for net in networks:
    print(f"{net['filename']}: {net['input_size']}→{net['output_size']}")

# Compare two networks
comparison = HDF5Utils.compare_networks('net1.h5', 'net2.h5')
if comparison['comparable']:
    print(f"Same architecture: {comparison['same_architecture']}")
```

## Command-Line Interface

### Installation

The CLI script is located at `cascade_correlation/snapshots/snapshot_cli.py`:

```bash
cd /path/to/cascor/src
python -m cascade_correlation.snapshots.snapshot_cli --help
```

### Commands

```bash
# Save a network snapshot
python -m cascade_correlation.snapshots.snapshot_cli save network.pkl snapshot.h5

# Load and verify a snapshot
python -m cascade_correlation.snapshots.snapshot_cli load snapshot.h5

# List all snapshots in directory
python -m cascade_correlation.snapshots.snapshot_cli list ./snapshots/

# Verify a snapshot file
python -m cascade_correlation.snapshots.snapshot_cli verify snapshot.h5

# Compare two snapshots
python -m cascade_correlation.snapshots.snapshot_cli compare net1.h5 net2.h5

# Clean up old files (keep 5 most recent)
python -m cascade_correlation.snapshots.snapshot_cli cleanup ./snapshots/ --keep 5
```

## Multiprocessing Restoration

The system safely handles multiprocessing state by:

1. **Configuration Only**: Stores MP configuration (role, address, authkey), not live objects
2. **Deterministic Recreation**: Recreates managers/queues with same settings on restore
3. **Role-Based Restoration**:
   - `server`: Starts new manager server with saved settings
   - `client`: Sets up connection parameters for later use
   - `none`: No multiprocessing restoration

```python
# Load with multiprocessing restoration
network = CascadeCorrelationNetwork.load_from_hdf5(
    'network.h5', 
    restore_multiprocessing=True
)
```

## Error Handling and Validation

### File Validation

```python
verification = network.verify_hdf5_file('network.h5')
if not verification['valid']:
    print(f"Invalid file: {verification['error']}")
else:
    print(f"Valid network with {verification['num_hidden_units']} hidden units")
```

### Common Issues

- **Import Errors**: Ensure all cascade correlation modules are in Python path
- **String Encoding**: The system handles both old (bytes) and new (UTF-8) string formats
- **Version Compatibility**: Format version 2.0 can read some older formats
- **CUDA Availability**: CUDA states are saved/restored only if CUDA is available

## Testing

Run the test script to verify functionality:

```bash
cd /path/to/cascor/src
python cascade_correlation/hdf5/test_hdf5.py
```

Expected output:

```bash
Testing HDF5 serialization...
✓ Created network with UUID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
✓ Network saved successfully
✓ File verification passed
✓ Network loaded successfully
✓ All network properties match
✓ All tests passed!
```

## Migration from Old Implementation

The new implementation replaces multiple overlapping HDF5 classes:

### Old Files (Deprecated)

- `snapshots/hdf5_shapshots.py` → **Replaced by `snapshots/snapshot_serializer.py`**
- `snapshots/core_hdf5_manager.py` → **Functionality moved to `snapshot_serializer.py`**
- `snapshots/hdf5_utilities/hdf5_utilities.py` → **Replaced by `snapshots/snapshot_utils.py`**
- `snapshots/hdf5_integration.py` → **Replaced by `snapshots/snapshot_cli.py`**
- `snapshots/hdf5_manager.py` → **Replaced by `snapshots/snapshot_cli.py`**

### New Import Paths

```python
# Old imports (deprecated)
from cascade_correlation.hdf5_serializer.cascade_hdf5_serializer import CascadeHDF5Serializer

# New imports
from cascade_correlation.hdf5 import CascadeHDF5Serializer
from cascade_correlation.hdf5 import HDF5Utils
```

### Breaking Changes

- Default `include_training_state=False` (was `True`)
- Improved error handling with debug traces
- String attributes now properly handle UTF-8 encoding
- Multiprocessing restoration is deterministic, not live object restoration

## Performance and Storage

- **Compression**: Default `gzip` level 4 provides good balance of speed/size
- **File Sizes**: Typical networks: 1-10 MB depending on hidden units and precision
- **Load Speed**: Sub-second for most networks, scales with number of hidden units
- **Memory Usage**: Temporary 2x memory usage during save/load operations

## Security Considerations

- **No Code Execution**: Pure data serialization, no pickle/eval usage
- **Input Validation**: All loaded data is validated before network creation  
- **Path Safety**: File paths are validated and sanitized
- **No Secrets**: Multiprocessing authkeys are stored as hex strings (reversible)

## Contributing

When modifying the HDF5 implementation:

1. **Maintain Format Compatibility**: Don't break existing files
2. **Update Version**: Increment format version for breaking changes
3. **Test Thoroughly**: Run test suite and round-trip tests
4. **Document Changes**: Update this README and docstrings
5. **Handle Errors Gracefully**: Provide clear error messages

## Troubleshooting

### Common Error Messages

**"Missing required group: config":**

- File is corrupted or not a valid network snapshot
- Try with `verify` command to see full error details

**"Could not restore multiprocessing state":**

- Non-critical warning, network will work without multiprocessing
- Check if ports are available or processes are running

**"Input size mismatch: X != Y":**

- Network architecture changed between save/load
- Configuration may be incompatible

**"Format validation failed":**

- File format is too old or corrupted
- Try saving with current version first

For additional help, check the test script and CLI examples.
