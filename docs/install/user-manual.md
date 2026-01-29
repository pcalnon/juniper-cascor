# Juniper Cascor User Manual

**Version**: 0.3.21 (0.7.3)  
**Last Updated**: 2026-01-29  
**License**: MIT License

---

## Table of Contents

1. [Overview of the Cascade Correlation Algorithm](#overview-of-the-cascade-correlation-algorithm)
2. [Basic Workflow](#basic-workflow)
3. [Configuration Options](#configuration-options)
4. [Working with Data](#working-with-data)
5. [Saving and Loading Networks](#saving-and-loading-networks)
6. [Visualization](#visualization)
7. [Deterministic Training](#deterministic-training)
8. [Example Workflows](#example-workflows)

---

## Overview of the Cascade Correlation Algorithm

The Cascade Correlation algorithm (Fahlman & Lebiere, 1990) is a **constructive neural network** learning algorithm that dynamically builds network architecture during training. Unlike traditional neural networks with fixed architectures, Cascade Correlation starts with a minimal network and incrementally adds hidden units as needed.

### Key Concepts

- **Constructive Learning**: The network starts with only input and output layers. Hidden units are added one at a time (or in layers) when the output layer cannot improve further.

- **Correlation-Based Training**: Candidate hidden units are trained to maximize correlation with the residual error. The candidate with the highest correlation is permanently added to the network.

- **Cascade Architecture**: Each new hidden unit receives input from all inputs and all previously added hidden units, creating a cascaded structure.

- **Frozen Weights**: Once a hidden unit is added, its input weights are frozen. Only output layer weights are retrained after each addition.

### Advantages

- **No architecture search**: Network size is determined automatically
- **Fast convergence**: Incremental building often trains faster than backpropagation on fixed networks
- **Interpretability**: Network complexity matches problem complexity

> **⚠️ Thread Safety Warning**: The `CascadeCorrelationNetwork` class is NOT thread-safe. Do not share network instances between threads without proper synchronization. For concurrent training, create separate network instances per thread.

---

## Basic Workflow

### Step 1: Creating a Network Configuration

The `CascadeCorrelationConfig` class holds all configuration parameters for the network:

```python
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import (
    CascadeCorrelationConfig
)

# Create configuration with required architecture parameters
config = CascadeCorrelationConfig(
    input_size=2,           # Number of input features
    output_size=2,          # Number of output classes (for classification)
    learning_rate=0.01,     # Learning rate for output layer
    max_hidden_units=50,    # Maximum hidden units to add
    random_seed=42          # For reproducibility
)
```

### Step 2: Creating a Network Instance

Instantiate the network with your configuration:

```python
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork

network = CascadeCorrelationNetwork(config=config)
```

### Step 3: Training with fit()

Train the network on your data:

```python
import torch

# Prepare training data (PyTorch tensors)
x_train = torch.randn(1000, 2)  # 1000 samples, 2 features
y_train = torch.tensor(...)     # One-hot encoded targets

# Train the network
network.fit(x_train, y_train, epochs=100)
```

The `fit()` method performs the complete Cascade Correlation training loop:

1. Train output layer weights
2. If error plateaus, train candidate pool
3. Select best candidate and add to network
4. Repeat until target accuracy reached or max hidden units added

### Step 4: Evaluating with get_accuracy()

Evaluate the trained network:

```python
# Evaluate on test data
accuracy = network.get_accuracy(x_test, y_test)
print(f"Test accuracy: {accuracy:.2%}")
```

### Step 5: Making Predictions with forward()

Use the trained network for inference:

```python
# Forward pass for predictions
predictions = network.forward(x_test)

# Get predicted class labels
predicted_classes = torch.argmax(predictions, dim=1)
```

---

## Configuration Options

### CascadeCorrelationConfig Parameters

#### Network Architecture

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_size` | int | 2 | Number of input features |
| `output_size` | int | 1 | Number of output units (classes for classification) |
| `max_hidden_units` | int | 100 | Maximum number of hidden units to add |

#### Activation Function

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `activation_function_name` | str | "Tanh" | Activation function: "Tanh", "Sigmoid", "ReLU", etc. |

#### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | 0.01 | Learning rate for output layer training |
| `candidate_learning_rate` | float | 0.01 | Learning rate for candidate training |
| `candidate_pool_size` | int | 16 | Number of candidates to train in parallel |
| `candidate_epochs` | int | 100 | Epochs to train each candidate |
| `output_epochs` | int | 100 | Epochs to train output layer |
| `epochs_max` | int | 1000 | Maximum total training epochs |
| `patience` | int | 15 | Early stopping patience (epochs without improvement) |

#### Thresholds

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `correlation_threshold` | float | 0.001 | Minimum correlation to accept a candidate |

#### N-Best Candidate Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `candidates_per_layer` | int | 1 | Number of top candidates to add per layer (N-best) |
| `layer_selection_strategy` | str | "top_n" | Selection strategy: "top_n", "threshold", "adaptive" |

#### Display and Visualization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `generate_plots` | bool | True | Enable/disable plot generation |
| `display_frequency` | int | 10 | How often to display training progress |

#### Random Number Generation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_seed` | int | None | Seed for deterministic training |

### OptimizerConfig for Different Optimizers

The `OptimizerConfig` dataclass allows fine-grained control over the optimizer used for training:

```python
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import (
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
    epsilon=1e-8
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

# Configure AdamW (Adam with decoupled weight decay)
adamw_config = OptimizerConfig(
    optimizer_type='AdamW',
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.01
)

# Apply to network configuration
config = CascadeCorrelationConfig(input_size=2, output_size=2)
config.optimizer_config = sgd_config  # Override default Adam
```

#### Supported Optimizers

| Optimizer | Key Parameters |
|-----------|---------------|
| `Adam` | learning_rate, beta1, beta2, epsilon, weight_decay |
| `AdamW` | learning_rate, beta1, beta2, epsilon, weight_decay |
| `SGD` | learning_rate, momentum, weight_decay |
| `RMSprop` | learning_rate, momentum, epsilon |

---

## Working with Data

### Expected Tensor Formats

All data must be PyTorch tensors with specific shapes:

#### Input Tensor (x)

```python
import torch

# Shape: (num_samples, num_features)
x_train = torch.randn(1000, 2)  # 1000 samples, 2 features
x_train = torch.randn(500, 10)  # 500 samples, 10 features
```

- Must be a 2D tensor
- First dimension: number of samples
- Second dimension: number of features (must match `input_size` in config)
- Data type: `torch.float32` (default) or `torch.float64`

#### Target Tensor (y)

For classification tasks, targets must be **one-hot encoded**:

```python
# Shape: (num_samples, num_classes)
y_train = torch.tensor([
    [1, 0],  # Class 0
    [0, 1],  # Class 1
    [1, 0],  # Class 0
    ...
])
```

### One-Hot Encoding for Classification

If you have class labels as integers, convert them to one-hot encoding:

```python
import torch
import torch.nn.functional as F

# Original labels (integers)
labels = torch.tensor([0, 1, 0, 1, 2])  # 5 samples, 3 classes

# Convert to one-hot encoding
num_classes = 3
y_one_hot = F.one_hot(labels, num_classes=num_classes).float()
# Result shape: (5, 3)
# [[1, 0, 0],
#  [0, 1, 0],
#  [1, 0, 0],
#  [0, 1, 0],
#  [0, 0, 1]]
```

### Data Preparation Example

```python
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# Generate or load your data
X = your_features  # numpy array (num_samples, num_features)
y = your_labels    # numpy array (num_samples,) with integer labels

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert to PyTorch tensors
x_train = torch.tensor(X_train, dtype=torch.float32)
x_test = torch.tensor(X_test, dtype=torch.float32)

# One-hot encode labels
num_classes = len(set(y))
y_train = F.one_hot(torch.tensor(y_train), num_classes=num_classes).float()
y_test = F.one_hot(torch.tensor(y_test), num_classes=num_classes).float()
```

---

## Saving and Loading Networks

Juniper Cascor uses HDF5 format for efficient, portable network serialization with built-in data integrity verification.

### save_to_hdf5() Usage

Save a trained network to disk:

```python
# Basic save
network.save_to_hdf5("./models/my_network.h5")

# Save with training state (recommended for resume)
network.save_to_hdf5(
    filepath="./models/my_network.h5",
    include_training_state=True,  # Includes training history
    include_training_data=False   # Excludes data (saves space)
)

# Control compression
network.save_to_hdf5(
    filepath="./models/my_network.h5",
    compression="gzip",
    compression_opts=4  # 1-9, higher = smaller files, slower
)
```

#### What Gets Saved

- Network architecture (input/output sizes, hidden units)
- All trained weights and biases
- Activation function configuration
- Training history (if `include_training_state=True`)
- Random state (Python, NumPy, PyTorch) for deterministic resume
- Unique network UUID
- SHA256 checksums for data integrity

### load_from_hdf5() Usage

Load a previously saved network:

```python
# Load a saved network
loaded_network = CascadeCorrelationNetwork.load_from_hdf5(
    "./models/my_network.h5"
)

# Use immediately for inference
predictions = loaded_network.forward(x_test)
accuracy = loaded_network.get_accuracy(x_test, y_test)
```

The loader automatically:

- Verifies file format and version
- Validates checksums for data integrity
- Restores complete network state
- Logs warnings for any validation issues

### Resume Training Workflow

One of the most powerful features is deterministic training resume:

```python
# Initial training session
config = CascadeCorrelationConfig(
    input_size=2,
    output_size=2,
    random_seed=42  # Important for determinism
)
network = CascadeCorrelationNetwork(config=config)
network.fit(x_train, y_train, epochs=50)

# Save checkpoint
network.save_to_hdf5(
    "./checkpoints/epoch_50.h5",
    include_training_state=True
)
print(f"Checkpoint: {len(network.hidden_units)} hidden units")

# Later: Resume training from exact state
resumed_network = CascadeCorrelationNetwork.load_from_hdf5(
    "./checkpoints/epoch_50.h5"
)

# Continue training - deterministically continues from saved state
resumed_network.fit(x_train, y_train, epochs=50)  # Trains epochs 51-100
```

### Snapshot Management CLI

Command-line tools for managing saved networks:

```bash
# List all snapshots in a directory
python -m snapshots.snapshot_cli list ./models/

# Verify snapshot integrity
python -m snapshots.snapshot_cli verify ./models/my_network.h5

# Compare two snapshots
python -m snapshots.snapshot_cli compare model_v1.h5 model_v2.h5

# Cleanup old files (keep 5 most recent)
python -m snapshots.snapshot_cli cleanup ./models/ --keep 5
```

### Programmatic Snapshot Utilities

```python
from snapshots.snapshot_utils import HDF5Utils

# List networks in directory
networks = HDF5Utils.list_networks_in_directory("./models")
for net in networks:
    print(f"{net['filename']}: {net['num_hidden_units']} units, {net['size_mb']:.2f} MB")

# Verify snapshot integrity
from snapshots.snapshot_cli import verify_snapshot
info = verify_snapshot("./models/my_network.h5")
print(f"Valid: {info['valid']}, UUID: {info['network_uuid']}")

# Compare two networks
comparison = HDF5Utils.compare_networks("model_v1.h5", "model_v2.h5")
print(f"Same architecture: {comparison['same_architecture']}")

# Cleanup old files
deleted = HDF5Utils.cleanup_old_files("./models", keep_count=10)
print(f"Deleted {deleted} old snapshots")
```

---

## Visualization

Juniper Cascor includes built-in visualization for datasets, decision boundaries, and training history.

### Enabling/Disabling Plots

Control plot generation via configuration:

```python
# Enable plotting (default)
config = CascadeCorrelationConfig(
    input_size=2,
    output_size=2,
    generate_plots=True
)

# Disable plotting (for batch training/benchmarks)
config = CascadeCorrelationConfig(
    input_size=2,
    output_size=2,
    generate_plots=False
)
```

### Available Visualization Methods

#### Plot Dataset

Visualize input data with class labels (requires 2D input):

```python
from cascor_plotter.cascor_plotter import CascadeCorrelationPlotter

# Static method - no network required
CascadeCorrelationPlotter.plot_dataset(
    x=x_train,
    y=y_train,
    title="Training Dataset"
)
```

#### Plot Decision Boundary

Visualize the learned decision boundary (requires 2D input):

```python
# Synchronous plotting (blocks until closed)
network.plot_decision_boundary(
    x_test,
    y_test,
    title="Decision Boundary",
    async_plot=False
)

# Asynchronous plotting (non-blocking)
plot_process = network.plot_decision_boundary(
    x_test,
    y_test,
    title="Decision Boundary",
    async_plot=True
)
# Training continues while plot renders
# plot_process.join()  # Wait for plot to complete if needed
```

#### Plot Training History

Visualize training progress over time:

```python
# Access training history
history = network.history

# Plot using the network's plotter
network.plotter.plot_training_history(history)

# History contains:
# - history['train_loss']: Loss per epoch
# - history['train_accuracy']: Accuracy per epoch
# - history['hidden_units_added']: When hidden units were added
```

---

## Deterministic Training

Juniper Cascor supports fully deterministic training for reproducibility in research and debugging.

### Setting random_seed for Reproducibility

Set the random seed in configuration:

```python
config = CascadeCorrelationConfig(
    input_size=2,
    output_size=2,
    random_seed=42  # Any integer for reproducibility
)

network = CascadeCorrelationNetwork(config=config)
network.fit(x_train, y_train, epochs=100)
```

When `random_seed` is set, the following are seeded:

- Python's `random` module
- NumPy's random generator
- PyTorch's CPU random generator
- PyTorch's CUDA random generator (if GPU available)

### Full RNG State Preservation

When saving a network with training state, complete RNG states are preserved:

```python
# Train partially
config = CascadeCorrelationConfig(random_seed=42)
network = CascadeCorrelationNetwork(config=config)
network.fit(x_train, y_train, epochs=25)

# Save - includes complete RNG state
network.save_to_hdf5("checkpoint.h5", include_training_state=True)

# Load - restores exact RNG state
resumed = CascadeCorrelationNetwork.load_from_hdf5("checkpoint.h5")
resumed.fit(x_train, y_train, epochs=25)

# Result: Identical to training 50 epochs continuously
```

### Verify Determinism

Test that training is reproducible:

```python
import torch

# Train two networks with same seed
config = CascadeCorrelationConfig(random_seed=42)

net1 = CascadeCorrelationNetwork(config=config)
net1.fit(x_train, y_train, epochs=50)

net2 = CascadeCorrelationNetwork(config=config)
net2.fit(x_train, y_train, epochs=50)

# Compare outputs - should be identical
test_x = torch.randn(10, 2)
out1 = net1.forward(test_x)
out2 = net2.forward(test_x)

difference = torch.abs(out1 - out2).max().item()
print(f"Maximum difference: {difference}")  # Should be ~0.0 or < 1e-6
assert torch.allclose(out1, out2), "Training is not deterministic!"
```

---

## Example Workflows

### Workflow 1: Train and Save

Complete workflow to train a network and save it:

```python
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import (
    CascadeCorrelationConfig
)
import torch

# Create synthetic data (circle classifier)
x_train = torch.randn(1000, 2)
y_labels = (x_train[:, 0] ** 2 + x_train[:, 1] ** 2 < 1).long()
y_train = torch.nn.functional.one_hot(y_labels, num_classes=2).float()

# Configure network
config = CascadeCorrelationConfig(
    input_size=2,
    output_size=2,
    max_hidden_units=50,
    random_seed=42,
    generate_plots=True
)

# Create and train
network = CascadeCorrelationNetwork(config=config)
network.fit(x_train, y_train, epochs=100)

# Save trained network
network.save_to_hdf5(
    "./models/circle_classifier.h5",
    include_training_state=True
)
print(f"Trained network with {len(network.hidden_units)} hidden units")
```

### Workflow 2: Load and Evaluate

Load a saved network and evaluate on test data:

```python
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
import torch

# Load saved network
network = CascadeCorrelationNetwork.load_from_hdf5(
    "./models/circle_classifier.h5"
)

# Generate test data
x_test = torch.randn(200, 2)
y_labels = (x_test[:, 0] ** 2 + x_test[:, 1] ** 2 < 1).long()
y_test = torch.nn.functional.one_hot(y_labels, num_classes=2).float()

# Evaluate
accuracy = network.get_accuracy(x_test, y_test)
print(f"Test accuracy: {accuracy:.2%}")

# Make predictions
predictions = network.forward(x_test)
predicted_classes = torch.argmax(predictions, dim=1)

# Visualize decision boundary
network.plot_decision_boundary(x_test, y_test, "Circle Classifier")
```

### Workflow 3: Incremental Training with Checkpoints

Save checkpoints during training for analysis or recovery:

```python
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import (
    CascadeCorrelationConfig
)

# Configure
config = CascadeCorrelationConfig(
    input_size=2,
    output_size=2,
    random_seed=42
)

network = CascadeCorrelationNetwork(config=config)

# Train with checkpoints
milestones = [20, 50, 100, 200]
for milestone in milestones:
    # Train to milestone
    network.fit(x_train, y_train, epochs=milestone)

    # Save checkpoint
    network.save_to_hdf5(
        f"./checkpoints/epoch_{milestone}.h5",
        include_training_state=True
    )

    # Evaluate
    acc = network.get_accuracy(x_val, y_val)
    print(f"Epoch {milestone}: {acc:.2%} accuracy, "
          f"{len(network.hidden_units)} hidden units")
```

### Workflow 4: Hyperparameter Search

Systematically test different configurations:

```python
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import (
    CascadeCorrelationConfig
)

# Define configurations to test
hyperparams = [
    {"learning_rate": 0.01, "correlation_threshold": 0.001},
    {"learning_rate": 0.05, "correlation_threshold": 0.005},
    {"learning_rate": 0.001, "correlation_threshold": 0.0001},
]

results = []
for params in hyperparams:
    # Create config with these hyperparameters
    config = CascadeCorrelationConfig(
        input_size=2,
        output_size=2,
        random_seed=42,  # Same seed for fair comparison
        generate_plots=False,  # Disable plots for speed
        **params
    )

    # Train
    network = CascadeCorrelationNetwork(config=config)
    network.fit(x_train, y_train, epochs=100)

    # Evaluate
    acc = network.get_accuracy(x_test, y_test)
    results.append({
        "params": params,
        "accuracy": acc,
        "hidden_units": len(network.hidden_units)
    })

    # Save this configuration
    network.save_to_hdf5(
        f"./experiments/lr_{params['learning_rate']}_ct_{params['correlation_threshold']}.h5"
    )

# Find best configuration
best = max(results, key=lambda x: x["accuracy"])
print(f"Best config: {best['params']}")
print(f"Accuracy: {best['accuracy']:.2%}")
print(f"Hidden units: {best['hidden_units']}")
```

### Workflow 5: Two-Spiral Problem

Use the built-in SpiralProblem class for the classic benchmark:

```python
from spiral_problem.spiral_problem import SpiralProblem

# Create and evaluate spiral problem
sp = SpiralProblem(
    _SpiralProblem__n_points=100,
    _SpiralProblem__n_spirals=2,
    _SpiralProblem__noise=0.1
)

# Run full evaluation pipeline
sp.evaluate(
    n_points=100,
    n_spirals=2,
    plot=True
)
```

---

## Troubleshooting

### PicklingError during multiprocessing

If you see `pickle.PicklingError: logger cannot be pickled`:

- This was fixed in BUG-002
- Ensure you have the latest version of the code
- Logger is now excluded from pickling

### Checksum verification failed

If you see `ERROR: Hidden unit X weights checksum verification failed`:

- Possible file corruption
- Try using a more recent snapshot
- Check storage media integrity

### Training not deterministic

If results differ between runs with same seed:

- Ensure `random_seed` is set in configuration
- Check that CUDA determinism is enabled (for GPU)
- Verify same training data is used

---

## Related Documentation

- [FEATURES_GUIDE.md](../../notes/FEATURES_GUIDE.md) - Detailed feature documentation
- [CASCOR_ENHANCEMENTS_ROADMAP.md](../../notes/CASCOR_ENHANCEMENTS_ROADMAP.md) - Enhancement roadmap
- [Test Suite README](../../src/tests/README.md) - Testing documentation

---

**Version**: 0.3.21 (0.7.3)  
**Maintainer**: Paul Calnon  
**License**: MIT License
