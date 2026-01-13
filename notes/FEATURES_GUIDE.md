# Cascor Prototype Features Guide

**Version**: 0.3.2  
**Date**: 2025-10-28  
**Status**: MVP Complete

---

> **⚠️ Thread Safety Warning**: The `CascadeCorrelationNetwork` class is NOT thread-safe. Do not share network instances between threads without proper synchronization. For concurrent training, create separate network instances per thread.

---

## Table of Contents

1. [HDF5 Serialization](#hdf5-serialization)
2. [N-Best Candidate Selection](#n-best-candidate-selection)
3. [Flexible Optimizer Configuration](#flexible-optimizer-configuration)
4. [Deterministic Training](#deterministic-training)
5. [Multiprocessing Support](#multiprocessing-support)
6. [Data Integrity Validation](#data-integrity-validation)

---

## HDF5 Serialization

### Basic Save/Load

```python
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

# Create and train network
config = CascadeCorrelationConfig(input_size=2, output_size=1)
network = CascadeCorrelationNetwork(config=config)
network.fit(x_train, y_train, epochs=100)

# Save to HDF5
network.save_to_hdf5(
    filepath="./snapshots/my_network.h5",
    include_training_state=True,  # Include training history
    include_training_data=False   # Exclude data (recommended)
)

# Load from HDF5
loaded_network = CascadeCorrelationNetwork.load_from_hdf5(
    "./snapshots/my_network.h5"
)

# Continue training
loaded_network.fit(x_train, y_train, epochs=50)
```

### Resume Training Workflow

```python
# Train for some epochs
network = CascadeCorrelationNetwork(config=config)
network.fit(x_train, y_train, epochs=20)

# Save checkpoint
network.save_to_hdf5("./checkpoints/epoch_20.h5", include_training_state=True)

# Resume later (deterministic)
resumed_network = CascadeCorrelationNetwork.load_from_hdf5("./checkpoints/epoch_20.h5")
resumed_network.fit(x_train, y_train, epochs=20)  # Continues from epoch 20

# Result: Identical to training for 40 epochs continuously
```

### Snapshot Utilities

```python
from snapshots.snapshot_utils import HDF5Utils
from snapshots.snapshot_cli import verify_snapshot, list_snapshots

# List all snapshots in directory
networks = HDF5Utils.list_networks_in_directory("./snapshots")
for net in networks:
    print(f"{net['filename']}: {net['num_hidden_units']} units, {net['size_mb']:.2f} MB")

# Verify snapshot integrity
info = verify_snapshot("./snapshots/my_network.h5")
print(f"Valid: {info['valid']}, UUID: {info['network_uuid']}")

# Compare two snapshots
comparison = HDF5Utils.compare_networks("snapshot1.h5", "snapshot2.h5")
print(f"Same architecture: {comparison['same_architecture']}")

# Cleanup old files (keep 10 most recent)
deleted = HDF5Utils.cleanup_old_files("./snapshots", keep_count=10)
print(f"Deleted {deleted} old snapshots")
```

---

## N-Best Candidate Selection

### Enable Layer-Based Addition

```python
from cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

# Configure for N-best selection
config = CascadeCorrelationConfig(
    input_size=2,
    output_size=1,
    candidate_pool_size=20,        # Train 20 candidates
    candidates_per_layer=3,         # Add top 3 as layer
    correlation_threshold=0.001,    # Minimum correlation
)

network = CascadeCorrelationNetwork(config=config)
network.fit(x_train, y_train)

# Network will add 3 candidates at a time instead of 1
```

### Selection Strategies

The network uses `top_n` strategy by default:

```python
# In cascade_correlation.py:
def _select_best_candidates(self, results, num_candidates=1):
    # 1. Sort by absolute correlation (descending)
    sorted_results = sorted(
        results,
        key=lambda r: abs(r.correlation),
        reverse=True
    )
    
    # 2. Select top N
    selected = sorted_results[:num_candidates]
    
    # 3. Filter by correlation threshold
    selected = [r for r in selected 
                if abs(r.correlation) >= self.correlation_threshold]
    
    return selected
```

### Custom Selection Strategy

```python
# Future enhancement: Custom strategies
config = CascadeCorrelationConfig(
    candidates_per_layer=5,
    layer_selection_strategy='threshold',  # Add all above threshold
    # or
    layer_selection_strategy='adaptive',   # Adaptive selection
)
```

---

## Flexible Optimizer Configuration

### Using Different Optimizers

```python
from cascade_correlation_config.cascade_correlation_config import (
    CascadeCorrelationConfig,
    OptimizerConfig
)

# Configure Adam optimizer (default)
adam_config = OptimizerConfig(
    optimizer_type='Adam',
    learning_rate=0.01,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.0,
    epsilon=1e-8,
    amsgrad=False
)

# Configure SGD with momentum
sgd_config = OptimizerConfig(
    optimizer_type='SGD',
    learning_rate=0.1,
    momentum=0.9,
    weight_decay=1e-4
)

# Configure RMSprop
rmsprop_config = OptimizerConfig(
    optimizer_type='RMSprop',
    learning_rate=0.001,
    momentum=0.9,
    epsilon=1e-8
)

# Configure AdamW (Adam with weight decay)
adamw_config = OptimizerConfig(
    optimizer_type='AdamW',
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.01,
    amsgrad=True
)

# Use in network config
config = CascadeCorrelationConfig(
    input_size=2,
    output_size=1,
    learning_rate=0.01  # Used by OptimizerConfig if not overridden
)
config.optimizer_config = sgd_config  # Override default

network = CascadeCorrelationNetwork(config=config)
```

### Optimizer Factory Method

```python
# The network uses _create_optimizer() internally:
def _create_optimizer(self, parameters, optimizer_config=None):
    config = optimizer_config or self.config.optimizer_config
    
    # Automatically creates appropriate optimizer
    # Supports: Adam, SGD, RMSprop, AdamW
    return optimizer_map[config.optimizer_type]()

# Usage in training:
self.output_optimizer = self._create_optimizer(output_layer.parameters())
```

---

## Deterministic Training

### Full Reproducibility

```python
# Set random seed in config
config = CascadeCorrelationConfig(
    input_size=2,
    output_size=1,
    random_seed=42  # Ensures deterministic behavior
)

network = CascadeCorrelationNetwork(config=config)

# All RNG sources seeded:
# - Python random module
# - NumPy random
# - PyTorch random
# - PyTorch CUDA random (if available)

# Training is fully deterministic
network.fit(x_train, y_train)

# Save preserves RNG state
network.save_to_hdf5("checkpoint.h5")

# Load restores exact RNG state
loaded = CascadeCorrelationNetwork.load_from_hdf5("checkpoint.h5")

# Resumed training continues from exact state
loaded.fit(x_train, y_train)
```

### Verify Determinism

```python
import torch
import numpy as np

# Train two networks identically
config = CascadeCorrelationConfig(random_seed=42)

net1 = CascadeCorrelationNetwork(config=config)
net1.fit(x_train, y_train, epochs=50)

net2 = CascadeCorrelationNetwork(config=config)
net2.fit(x_train, y_train, epochs=50)

# Outputs should be identical
test_x = torch.randn(10, 2)
out1 = net1.forward(test_x)
out2 = net2.forward(test_x)

assert torch.allclose(out1, out2), "Training is not deterministic!"
```

---

## Multiprocessing Support

### Automatic Multiprocessing

```python
# Network automatically uses multiprocessing for candidate training
config = CascadeCorrelationConfig(
    candidate_pool_size=20,  # Trains 20 candidates in parallel
    candidate_epochs=100
)

network = CascadeCorrelationNetwork(config=config)

# Candidates trained in parallel using all CPU cores
network.fit(x_train, y_train)
```

### Multiprocessing Configuration

```python
config = CascadeCorrelationConfig(
    # Queue configuration
    candidate_training_queue_authkey="my_secret_key",
    candidate_training_queue_address=("127.0.0.1", 50000),
    
    # Timeout configuration
    candidate_training_task_queue_timeout=5.0,
    candidate_training_shutdown_timeout=10.0,
    
    # Context type
    candidate_training_context_type="forkserver"  # or "spawn", "fork"
)
```

### Remote Workers

```python
from remote_client.remote_client import RemoteWorkerClient

# On server machine
network = CascadeCorrelationNetwork(config=config)
network.start_candidate_training_server()  # Starts queue server

# On client machine(s)
client = RemoteWorkerClient(
    address=("server_ip", 50000),
    authkey="my_secret_key"
)
client.connect()
client.start_workers(num_workers=4)  # Add 4 remote workers

# Server distributes work to all workers (local + remote)
```

---

## Data Integrity Validation

### Automatic Checksums

All saved networks include SHA256 checksums for:

- Output layer weights
- Output layer bias
- Each hidden unit's weights
- Each hidden unit's bias

```python
# Save network (checksums calculated automatically)
network.save_to_hdf5("network.h5")

# Load network (checksums verified automatically)
loaded = CascadeCorrelationNetwork.load_from_hdf5("network.h5")

# Checksum failures logged as errors:
# ERROR: Hidden unit 5 weights checksum verification failed!
```

### Shape Validation

All loaded networks undergo shape validation:

```python
# Validates:
# - Output weights: (input_size + num_hidden, output_size)
# - Output bias: (output_size,)
# - Hidden unit i weights: (input_size + i,)
# - Hidden unit bias: scalar or (1,)

loaded = CascadeCorrelationNetwork.load_from_hdf5("network.h5")
# If shapes invalid, warnings logged but network still returned
```

### Format Validation

All HDF5 files validated on load:

```python
# Checks:
# - Format identifier: "juniper.cascor"
# - Format version compatibility
# - Required groups: meta, config, params, arch, random
# - Required datasets: output weights, output bias
# - Hidden units consistency

# Invalid files rejected with detailed error messages
```

---

## Advanced Features

### Custom Activation Functions

```python
import torch.nn as nn

config = CascadeCorrelationConfig(
    activation_function_name="ReLU",  # Tanh, Sigmoid, ReLU, etc.
)

network = CascadeCorrelationNetwork(config=config)

# Activation function name saved in snapshot
# Properly restored on load
```

### Early Stopping

```python
# Early stopping for output layer training
network.fit(
    x_train, y_train,
    x_val=x_val, y_val=y_val,  # Provide validation data
    epochs=1000,
    early_stopping=True,       # Enable early stopping
    patience=20                # Stop if no improvement for 20 epochs
)
```

### Training History

```python
# Access training history
history = network.history

print(f"Training loss: {history['train_loss']}")
print(f"Training accuracy: {history['train_accuracy']}")
print(f"Validation loss: {history['value_loss']}")
print(f"Validation accuracy: {history['value_accuracy']}")
print(f"Hidden units added: {len(history['hidden_units_added'])}")

# Plot training history
network.plotter.plot_training_history(history)
```

### Decision Boundary Plotting

```python
# Synchronous plotting (blocks)
network.plot_decision_boundary(x_test, y_test, async_plot=False)

# Asynchronous plotting (non-blocking, now works with BUG-002 fix!)
plot_process = network.plot_decision_boundary(x_test, y_test, async_plot=True)
# Training continues while plot renders
# plot_process.join() to wait for completion
```

---

## Configuration Examples

### Minimal Configuration

```python
# Simplest possible setup
config = CascadeCorrelationConfig(
    input_size=2,
    output_size=1
)
network = CascadeCorrelationNetwork(config=config)
```

### Recommended Configuration

```python
# Good defaults for most problems
config = CascadeCorrelationConfig(
    input_size=2,
    output_size=1,
    max_hidden_units=100,
    candidate_pool_size=10,
    candidate_epochs=50,
    output_epochs=50,
    learning_rate=0.01,
    candidate_learning_rate=0.01,
    correlation_threshold=0.001,
    patience=15,
    random_seed=42
)
```

### High-Performance Configuration

```python
# Optimized for speed
config = CascadeCorrelationConfig(
    input_size=2,
    output_size=1,
    candidate_pool_size=20,        # More parallel candidates
    candidates_per_layer=3,        # Add multiple units at once
    candidate_epochs=30,           # Fewer candidate epochs
    output_epochs=30,              # Fewer output epochs
    patience=10,                   # Less patience
    correlation_threshold=0.005,   # Higher threshold (earlier stopping)
)
```

### High-Accuracy Configuration

```python
# Optimized for accuracy
config = CascadeCorrelationConfig(
    input_size=2,
    output_size=1,
    max_hidden_units=200,          # Allow more units
    candidate_pool_size=50,        # Train more candidates
    candidates_per_layer=1,        # Add best only
    candidate_epochs=200,          # More training
    output_epochs=100,             # More output training
    patience=50,                   # More patience
    correlation_threshold=0.0001,  # Lower threshold (more units)
    learning_rate=0.005,           # Lower learning rate
)
```

---

## Best Practices

### Serialization

1. **Always set random_seed for reproducibility**:

   ```python
   config = CascadeCorrelationConfig(random_seed=42)
   ```

2. **Include training state for resume**:

   ```python
   network.save_to_hdf5(filepath, include_training_state=True)
   ```

3. **Exclude training data** (it's large and unnecessary):

   ```python
   network.save_to_hdf5(filepath, include_training_data=False)
   ```

4. **Verify snapshots after saving**:

   ```python
   from snapshots.snapshot_utils import HDF5Utils
   info = HDF5Utils.validate_network_file(filepath)
   assert info['valid'], f"Snapshot invalid: {info.get('error')}"
   ```

### Training

1. **Use validation data for early stopping**:

   ```python
   network.fit(x_train, y_train, x_val=x_val, y_val=y_val)
   ```

2. **Monitor training with history**:

   ```python
   history = network.history
   if history['train_accuracy'][-1] > 0.95:
       print("Target accuracy reached!")
   ```

3. **Save checkpoints periodically**:

   ```python
   for epoch in range(0, 1000, 100):
       network.fit(x_train, y_train, epochs=100)
       network.save_to_hdf5(f"checkpoint_epoch_{epoch+100}.h5")
   ```

### Multiprocessing

1. **Use forkserver context** (safest):

   ```python
   config = CascadeCorrelationConfig(
       candidate_training_context_type="forkserver"
   )
   ```

2. **Monitor worker shutdown**:
   - Check logs for graceful termination count
   - Look for SIGKILL warnings (indicates issues)

3. **Cleanup after training**:
   - Workers automatically cleaned up after each grow cycle
   - Manager shutdown handled automatically

---

## Troubleshooting

### Issue: Snapshot won't load

**Check**:

```python
from snapshots.snapshot_serializer import CascadeHDF5Serializer

serializer = CascadeHDF5Serializer()
verification = serializer.verify_saved_network("network.h5")

if not verification['valid']:
    print(f"Error: {verification['error']}")
else:
    print(f"File is valid: {verification}")
```

**Solutions**:

- Ensure file is not corrupted
- Check format version compatibility
- Verify all required groups present

### Issue: Training not deterministic

**Check**:

- Random seed set in config
- No external randomness (e.g., data shuffling without seed)
- CUDA randomness seeded (if using GPU)

**Verify**:

```python
# Train twice with same seed
config = CascadeCorrelationConfig(random_seed=42)

net1 = CascadeCorrelationNetwork(config=config)
net1.fit(x, y, epochs=50)

net2 = CascadeCorrelationNetwork(config=config)
net2.fit(x, y, epochs=50)

# Compare outputs
test_out1 = net1.forward(x_test)
test_out2 = net2.forward(x_test)

print(f"Difference: {torch.abs(test_out1 - test_out2).max().item()}")
# Should be ~0.0 or very small (< 1e-6)
```

### Issue: PicklingError during plotting

**If you see**:

```bash
pickle.PicklingError: logger cannot be pickled
```

**Solution**: Already fixed in BUG-002  

- Ensure you have latest cascade_correlation.py
- Ensure you have latest cascor_plotter.py
- Logger is now excluded from pickling

### Issue: Checksum verification failed

**If you see**:

```bash
ERROR: Hidden unit 5 weights checksum verification failed!
```

**Possible causes**:

- File corruption
- Manual editing of HDF5 file
- Incompatible load/save versions

**Solution**:

- Use most recent snapshot
- Check file integrity
- Verify not using damaged storage media

---

## CLI Tools

### Snapshot Management CLI

```bash
# Save network
python -m snapshots.snapshot_cli save network.pkl snapshot.h5 --include-training

# Load network
python -m snapshots.snapshot_cli load snapshot.h5 --output converted.h5

# List snapshots
python -m snapshots.snapshot_cli list ./snapshots/

# Verify snapshot
python -m snapshots.snapshot_cli verify snapshot.h5

# Compare snapshots
python -m snapshots.snapshot_cli compare snapshot1.h5 snapshot2.h5

# Cleanup old files
python -m snapshots.snapshot_cli cleanup ./snapshots/ --keep 5
```

---

## Performance Tips

### Serialization Performance

1. **Use compression** (default gzip level 4):

   ```python
   network.save_to_hdf5(filepath, compression="gzip", compression_opts=4)
   ```

2. **Adjust compression for speed vs size**:

   ```python
   # Faster saves (larger files)
   compression_opts=1
   
   # Smaller files (slower saves)
   compression_opts=9
   ```

3. **Expected performance**:
   - Save (100 units): < 2 seconds
   - Load (100 units): < 3 seconds
   - Checksum verify: < 200ms

### Training Performance

1. **Optimize candidate pool size**:

   ```python
   # Balance: More candidates = better selection, slower training
   candidate_pool_size=10  # Good for 4-8 CPU cores
   ```

2. **Use N-best for faster convergence**:

   ```python
   candidates_per_layer=3  # Add multiple units, fewer iterations
   ```

3. **Tune patience for speed**:

   ```python
   patience=10  # Less patience = faster (but may underfit)
   ```

---

## Example Workflows

### Workflow 1: Train and Save

```python
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
import torch

# Create data
x_train = torch.randn(1000, 2)
y_train = (x_train[:, 0] ** 2 + x_train[:, 1] ** 2 < 1).float().unsqueeze(1)

# Configure and train
config = CascadeCorrelationConfig(
    input_size=2, output_size=1, random_seed=42, max_hidden_units=50
)
network = CascadeCorrelationNetwork(config=config)
network.fit(x_train, y_train, epochs=100)

# Save
network.save_to_hdf5("./models/circle_classifier.h5", include_training_state=True)
print(f"Trained network with {len(network.hidden_units)} hidden units")
```

### Workflow 2: Load and Evaluate

```python
# Load saved network
network = CascadeCorrelationNetwork.load_from_hdf5("./models/circle_classifier.h5")

# Evaluate
x_test = torch.randn(200, 2)
y_test = (x_test[:, 0] ** 2 + x_test[:, 1] ** 2 < 1).float().unsqueeze(1)

accuracy = network.get_accuracy(x_test, y_test)
print(f"Test accuracy: {accuracy:.2%}")

# Visualize
network.plot_decision_boundary(x_test, y_test, "Circle Classifier")
```

### Workflow 3: Incremental Training

```python
# Initial training
config = CascadeCorrelationConfig(random_seed=42)
network = CascadeCorrelationNetwork(config=config)

for milestone in [20, 50, 100, 200]:
    # Train to milestone
    network.fit(x_train, y_train, epochs=milestone)
    
    # Save checkpoint
    network.save_to_hdf5(f"./checkpoints/epoch_{milestone}.h5")
    
    # Evaluate
    acc = network.get_accuracy(x_val, y_val)
    print(f"Epoch {milestone}: {acc:.2%} accuracy, {len(network.hidden_units)} units")
```

### Workflow 4: Hyperparameter Search

```python
# Test different configurations
configs = [
    {"learning_rate": 0.01, "correlation_threshold": 0.001},
    {"learning_rate": 0.05, "correlation_threshold": 0.005},
    {"learning_rate": 0.001, "correlation_threshold": 0.0001},
]

results = []
for params in configs:
    config = CascadeCorrelationConfig(random_seed=42, **params)
    network = CascadeCorrelationNetwork(config=config)
    network.fit(x_train, y_train, epochs=100)
    
    acc = network.get_accuracy(x_test, y_test)
    results.append({
        "params": params,
        "accuracy": acc,
        "hidden_units": len(network.hidden_units)
    })
    
    # Save best config
    network.save_to_hdf5(f"./experiments/lr_{params['learning_rate']}.h5")

# Find best
best = max(results, key=lambda x: x["accuracy"])
print(f"Best config: {best['params']}, accuracy: {best['accuracy']:.2%}")
```

---

## API Reference Quick Links

### Core Classes

- `CascadeCorrelationNetwork` - Main network class
- `CascadeCorrelationConfig` - Configuration object
- `OptimizerConfig` - Optimizer configuration
- `CandidateUnit` - Individual candidate node

### Serialization Classes

- `CascadeHDF5Serializer` - HDF5 save/load operations
- `HDF5Utils` - Utility functions for HDF5 management

### Support Classes

- `CascadeCorrelationPlotter` - Plotting utilities
- `RemoteWorkerClient` - Remote multiprocessing client

---

## Related Documentation

- `CASCOR_ENHANCEMENTS_ROADMAP.md` - Complete enhancement plan
- `IMPLEMENTATION_PROGRESS.md` - Real-time progress tracking
- `IMPLEMENTATION_SUMMARY.md` - Work summary
- `PHASE1_COMPLETE.md` - Phase 1 completion report
- `NEXT_STEPS.md` - Original MVP plan
- `P2_ENHANCEMENTS_PLAN.md` - P2 optimization details

---

**Last Updated**: 2025-10-28  
**Maintainer**: Development Team  
**Status**: Production Ready for MVP
