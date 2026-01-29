# Juniper Cascor - Testing Manual

**Version**: 0.3.21  
**Last Updated**: 2026-01-29  
**Purpose**: Comprehensive guide for writing and running tests

---

## Table of Contents

1. [Test Organization](#test-organization)
2. [Writing Tests](#writing-tests)
3. [Test Fixtures](#test-fixtures)
4. [Mock Objects](#mock-objects)
5. [Testing Specific Components](#testing-specific-components)
6. [Test Data](#test-data)
7. [Common Testing Patterns](#common-testing-patterns)

---

## Test Organization

### Directory Structure

```
src/tests/
├── conftest.py              # Shared fixtures and configuration
├── pytest.ini               # Pytest settings
├── unit/                    # Unit tests (isolated, fast)
│   ├── test_forward_pass.py
│   ├── test_accuracy.py
│   ├── test_candidate_unit.py
│   ├── test_cascade_correlation_coverage.py
│   ├── test_config_and_exceptions.py
│   └── ...
├── integration/             # Integration tests (end-to-end)
│   ├── test_spiral_problem.py
│   ├── test_serialization.py
│   ├── test_comprehensive_serialization.py
│   └── ...
├── helpers/                 # Test utility functions
│   └── test_helpers.py
├── mocks/                   # Mock objects
│   └── mock_network.py
├── scripts/                 # Test runner scripts
│   ├── run_tests.bash
│   └── run_benchmarks.bash
└── reports/                 # Generated reports
    ├── htmlcov/
    ├── coverage.xml
    └── junit.xml
```

### Unit vs Integration Tests

| Aspect | Unit Tests | Integration Tests |
|--------|-----------|-------------------|
| **Scope** | Single method/class | Multiple components |
| **Speed** | Fast (< 1s each) | Slower (may take seconds) |
| **Dependencies** | Mocked | Real implementations |
| **Marker** | `@pytest.mark.unit` | `@pytest.mark.integration` |
| **Location** | `tests/unit/` | `tests/integration/` |

### Test File Naming Conventions

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>` or `Test<Feature>`
- Test functions: `test_<what_is_being_tested>`

**Examples**:

- `test_forward_pass.py` → Tests for forward pass functionality
- `test_spiral_problem.py` → Tests for spiral problem module
- `TestCascadeCorrelationNetwork` → Tests for the network class

---

## Writing Tests

### Basic Test Structure

```python
import pytest
import torch

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig


@pytest.mark.unit
class TestForwardPass:
    """Tests for the forward pass functionality."""

    def test_forward_returns_correct_shape(self):
        """Forward pass should return tensor with correct shape."""
        # Arrange
        config = CascadeCorrelationConfig(input_size=2, output_size=3)
        network = CascadeCorrelationNetwork(config=config)
        x = torch.randn(10, 2)

        # Act
        output = network.forward(x)

        # Assert
        assert output.shape == (10, 3)

    def test_forward_with_none_input_raises_error(self):
        """Forward pass should raise ValidationError for None input."""
        # Arrange
        config = CascadeCorrelationConfig(input_size=2, output_size=2)
        network = CascadeCorrelationNetwork(config=config)

        # Act & Assert
        with pytest.raises(ValueError):
            network.forward(None)
```

### Using Markers

```python
import pytest


@pytest.mark.unit
def test_fast_operation():
    """This is a fast unit test."""
    assert 1 + 1 == 2


@pytest.mark.integration
def test_full_training_cycle():
    """This tests the complete training workflow."""
    # ... longer test
    pass


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_extensive_training():
    """This test takes a long time and needs extended timeout."""
    # ... training test
    pass


@pytest.mark.gpu
def test_cuda_operations():
    """This test requires GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # ... GPU test
    pass


@pytest.mark.multiprocessing
def test_parallel_candidate_training():
    """This test uses multiprocessing."""
    # ... multiprocessing test
    pass
```

### Arrange-Act-Assert Pattern

All tests should follow the AAA pattern:

```python
def test_accuracy_calculation():
    """Test that accuracy is calculated correctly."""
    # ============ Arrange ============
    # Set up test data and dependencies
    config = CascadeCorrelationConfig(
        input_size=2,
        output_size=2,
        random_seed=42
    )
    network = CascadeCorrelationNetwork(config=config)

    # Create known data where we know the expected accuracy
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    y = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]])  # One-hot

    # Train minimally
    network.fit(x, y, epochs=10)

    # ============ Act ============
    # Perform the action being tested
    accuracy = network.get_accuracy(x, y)

    # ============ Assert ============
    # Verify the expected outcome
    assert 0.0 <= accuracy <= 1.0
    assert isinstance(accuracy, float)
```

---

## Test Fixtures

### Available Fixtures (conftest.py)

```python
# Network fixtures
@pytest.fixture
def simple_config():
    """Simple network configuration for testing."""
    return CascadeCorrelationConfig(
        input_size=2,
        output_size=2,
        random_seed=42,
        max_hidden_units=5,
        candidate_pool_size=4,
    )

@pytest.fixture
def simple_network(simple_config):
    """Pre-configured network instance."""
    return CascadeCorrelationNetwork(config=simple_config)

# Data fixtures
@pytest.fixture
def sample_data():
    """Sample training data (XOR problem)."""
    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = torch.tensor([[1, 0], [0, 1], [0, 1], [1, 0]])  # XOR one-hot
    return x, y

@pytest.fixture
def spiral_data():
    """Spiral problem data for integration tests."""
    from spiral_problem.spiral_problem import SpiralProblem
    sp = SpiralProblem()
    return sp.generate_spiral_dataset(n_points=50, n_spirals=2)

# Trained network fixture
@pytest.fixture
def trained_network(simple_network, sample_data):
    """Network that has been trained."""
    x, y = sample_data
    simple_network.fit(x, y, epochs=20)
    return simple_network
```

### Using Fixtures

```python
@pytest.mark.unit
def test_forward_with_fixture(simple_network, sample_data):
    """Test using fixtures for setup."""
    x, y = sample_data
    output = simple_network.forward(x)
    assert output.shape[0] == x.shape[0]


@pytest.mark.unit
def test_trained_network_accuracy(trained_network, sample_data):
    """Test with pre-trained network."""
    x, y = sample_data
    accuracy = trained_network.get_accuracy(x, y)
    assert accuracy >= 0.0
```

### Creating New Fixtures

```python
# In conftest.py or test file

@pytest.fixture(scope="function")  # New instance per test
def candidate_unit():
    """Fresh candidate unit for testing."""
    from candidate_unit.candidate_unit import CandidateUnit
    return CandidateUnit(
        _CandidateUnit__input_size=2,
        _CandidateUnit__learning_rate=0.01,
        _CandidateUnit__random_seed=42,
    )


@pytest.fixture(scope="module")  # Shared across module
def expensive_data():
    """Data that's expensive to create."""
    # Only created once per module
    return generate_large_dataset()


@pytest.fixture(scope="session")  # Shared across session
def gpu_device():
    """GPU device if available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    pytest.skip("CUDA not available")
```

### Fixture Scopes

| Scope | Lifetime | Use Case |
|-------|----------|----------|
| `function` | One test | Default, isolated tests |
| `class` | One test class | Shared setup for class |
| `module` | One test file | Expensive setup shared in file |
| `session` | Entire test run | Very expensive, global setup |

---

## Mock Objects

### Available Mocks

```python
# src/tests/mocks/mock_network.py

class MockCascadeCorrelationNetwork:
    """Mock network for testing without actual training."""

    def __init__(self, *args, **kwargs):
        self.hidden_units = []
        self.config = kwargs.get('config')

    def forward(self, x):
        return torch.zeros(x.shape[0], 2)

    def fit(self, x, y, epochs=100):
        return {'train_loss': [0.5], 'train_accuracy': [0.5]}

    def get_accuracy(self, x, y):
        return 0.5
```

### Mocking Network Components

```python
from unittest.mock import Mock, patch, MagicMock

@pytest.mark.unit
def test_with_mocked_logger():
    """Test with mocked logger to avoid log output."""
    with patch('cascade_correlation.cascade_correlation.Logger') as mock_logger:
        network = CascadeCorrelationNetwork(config=config)
        network.fit(x, y, epochs=1)

        # Verify logging calls if needed
        assert mock_logger.info.called


@pytest.mark.unit
def test_with_mocked_candidate_training():
    """Test network behavior without actual candidate training."""
    config = CascadeCorrelationConfig(input_size=2, output_size=2)
    network = CascadeCorrelationNetwork(config=config)

    # Mock the expensive candidate training
    mock_result = Mock()
    mock_result.best_correlation = 0.8
    mock_result.best_candidate = Mock()

    with patch.object(network, 'train_candidates', return_value=mock_result):
        # Test logic that depends on candidate training
        pass
```

### Mocking Multiprocessing

```python
@pytest.mark.unit
def test_without_multiprocessing():
    """Test candidate training without spawning processes."""
    config = CascadeCorrelationConfig(
        input_size=2,
        output_size=2,
        candidate_pool_size=2,  # Small pool
    )
    network = CascadeCorrelationNetwork(config=config)

    # Force sequential execution
    with patch.object(network, '_execute_parallel_training') as mock_parallel:
        mock_parallel.side_effect = lambda *args: network._execute_sequential_training(*args)

        # Test runs sequentially
        network.fit(x, y, epochs=10)
```

---

## Testing Specific Components

### Testing CascadeCorrelationNetwork

```python
@pytest.mark.unit
class TestCascadeCorrelationNetwork:
    """Core network tests."""

    def test_initialization(self, simple_config):
        """Network initializes with correct state."""
        network = CascadeCorrelationNetwork(config=simple_config)

        assert network.input_size == simple_config.input_size
        assert network.output_size == simple_config.output_size
        assert len(network.hidden_units) == 0

    def test_forward_shape(self, simple_network, sample_data):
        """Forward pass produces correct output shape."""
        x, _ = sample_data
        output = simple_network.forward(x)

        assert output.shape == (x.shape[0], simple_network.output_size)

    def test_training_reduces_loss(self, simple_network, sample_data):
        """Training should reduce loss over time."""
        x, y = sample_data

        # Get initial loss
        initial_output = simple_network.forward(x)
        initial_loss = torch.nn.functional.mse_loss(initial_output, y.float())

        # Train
        simple_network.fit(x, y, epochs=50)

        # Get final loss
        final_output = simple_network.forward(x)
        final_loss = torch.nn.functional.mse_loss(final_output, y.float())

        assert final_loss < initial_loss
```

### Testing CandidateUnit

```python
@pytest.mark.unit
@pytest.mark.candidate_training
class TestCandidateUnit:
    """Candidate unit tests."""

    def test_initialization(self):
        """Candidate initializes with correct weights."""
        from candidate_unit.candidate_unit import CandidateUnit

        unit = CandidateUnit(
            _CandidateUnit__input_size=4,
            _CandidateUnit__random_seed=42,
        )

        assert unit.weights.shape[0] == 4

    def test_training_improves_correlation(self):
        """Training should improve correlation with error."""
        from candidate_unit.candidate_unit import CandidateUnit

        unit = CandidateUnit(
            _CandidateUnit__input_size=2,
            _CandidateUnit__random_seed=42,
        )

        x = torch.randn(100, 2)
        residual_error = torch.randn(100, 2)

        correlation = unit.train(x, residual_error, epochs=50)

        assert abs(correlation) > 0  # Some correlation achieved
```

### Testing Serialization

```python
@pytest.mark.integration
class TestSerialization:
    """Serialization round-trip tests."""

    def test_save_load_preserves_state(self, trained_network, tmp_path):
        """Saving and loading preserves network state."""
        save_path = tmp_path / "network.h5"
        x = torch.randn(10, 2)

        # Get output before save
        output_before = trained_network.forward(x)

        # Save
        trained_network.save_to_hdf5(str(save_path))

        # Load
        loaded = CascadeCorrelationNetwork.load_from_hdf5(str(save_path))

        # Get output after load
        output_after = loaded.forward(x)

        # Should be identical
        assert torch.allclose(output_before, output_after)

    def test_deterministic_resume(self, tmp_path):
        """Resumed training produces same results as continuous."""
        config = CascadeCorrelationConfig(random_seed=42)
        x = torch.randn(50, 2)
        y = torch.randint(0, 2, (50, 2)).float()

        # Train continuously for 40 epochs
        network1 = CascadeCorrelationNetwork(config=config)
        network1.fit(x, y, epochs=40)

        # Train 20, save, load, train 20 more
        network2 = CascadeCorrelationNetwork(config=config)
        network2.fit(x, y, epochs=20)
        network2.save_to_hdf5(str(tmp_path / "checkpoint.h5"))

        network2 = CascadeCorrelationNetwork.load_from_hdf5(str(tmp_path / "checkpoint.h5"))
        network2.fit(x, y, epochs=20)

        # Should produce same results
        test_x = torch.randn(10, 2)
        output1 = network1.forward(test_x)
        output2 = network2.forward(test_x)

        assert torch.allclose(output1, output2, atol=1e-5)
```

### Testing Multiprocessing Code

```python
@pytest.mark.multiprocessing
@pytest.mark.slow
@pytest.mark.timeout(120)
class TestMultiprocessing:
    """Multiprocessing-specific tests."""

    def test_parallel_training_completes(self):
        """Parallel candidate training completes without deadlock."""
        config = CascadeCorrelationConfig(
            input_size=2,
            output_size=2,
            candidate_pool_size=8,
            candidate_epochs=20,
            max_hidden_units=3,
        )
        network = CascadeCorrelationNetwork(config=config)

        x = torch.randn(100, 2)
        y = torch.randint(0, 2, (100, 2)).float()

        # Should complete without hanging
        network.fit(x, y, epochs=50)

        assert len(network.hidden_units) > 0
```

---

## Test Data

### Pre-generated Datasets

Located in `data/`:

- `spiral_2_100.npz` - 2-spiral, 100 points
- `spiral_4_200.npz` - 4-spiral, 200 points
- `spiral_5_250.npz` - 5-spiral, 250 points
- `spiral_8_400.npz` - 8-spiral, 400 points

### Loading Test Data

```python
import numpy as np
import torch

def load_spiral_data(n_spirals=2, n_points=100):
    """Load pre-generated spiral data."""
    data = np.load(f"data/spiral_{n_spirals}_{n_points}.npz")
    x = torch.tensor(data['x'], dtype=torch.float32)
    y = torch.tensor(data['y'], dtype=torch.float32)
    return x, y
```

### Creating Test Data

```python
from spiral_problem.spiral_problem import SpiralProblem

# Generate fresh spiral data
sp = SpiralProblem()
x, y = sp.generate_spiral_dataset(n_points=100, n_spirals=2, noise=0.1)

# Create simple XOR data
x_xor = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y_xor = torch.tensor([[1, 0], [0, 1], [0, 1], [1, 0]])  # One-hot

# Create random data for shape tests
x_random = torch.randn(50, 4)
y_random = torch.randint(0, 3, (50,))
y_onehot = torch.nn.functional.one_hot(y_random, num_classes=3).float()
```

---

## Common Testing Patterns

### Testing with Random Seeds

```python
@pytest.mark.unit
def test_reproducibility():
    """Same seed produces same results."""
    results = []

    for _ in range(2):
        config = CascadeCorrelationConfig(random_seed=42)
        network = CascadeCorrelationNetwork(config=config)

        x = torch.randn(20, 2)
        y = torch.randint(0, 2, (20, 2)).float()

        network.fit(x, y, epochs=10)
        results.append(network.forward(x))

    assert torch.allclose(results[0], results[1])
```

### Testing Exception Handling

```python
@pytest.mark.unit
class TestExceptionHandling:
    """Exception handling tests."""

    def test_none_input_raises_validation_error(self, simple_network):
        """None input should raise ValidationError."""
        with pytest.raises(ValueError):
            simple_network.forward(None)

    def test_wrong_shape_raises_error(self, simple_network):
        """Wrong input shape should raise error."""
        # Network expects 2 features, give 5
        x_wrong = torch.randn(10, 5)

        with pytest.raises((ValueError, RuntimeError)):
            simple_network.forward(x_wrong)

    def test_invalid_config_raises_error(self):
        """Invalid configuration should raise ConfigurationError."""
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import ConfigurationError

        with pytest.raises((ConfigurationError, ValueError)):
            CascadeCorrelationConfig(input_size=-1)
```

### Testing Async/Threading Code

```python
import threading
import time

@pytest.mark.unit
def test_thread_safety_warning():
    """Verify thread safety documentation applies."""
    config = CascadeCorrelationConfig(random_seed=42)
    network = CascadeCorrelationNetwork(config=config)

    # This is an example of what NOT to do
    # (demonstrates the thread safety issue)
    errors = []

    def train_in_thread(network, x, y):
        try:
            network.fit(x, y, epochs=5)
        except Exception as e:
            errors.append(e)

    x = torch.randn(50, 2)
    y = torch.randint(0, 2, (50, 2)).float()

    # Don't share networks between threads!
    # Create separate instances instead
    network1 = CascadeCorrelationNetwork(config=config)
    network2 = CascadeCorrelationNetwork(config=config)

    t1 = threading.Thread(target=train_in_thread, args=(network1, x, y))
    t2 = threading.Thread(target=train_in_thread, args=(network2, x, y))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Separate instances should work fine
    assert len(errors) == 0
```

### Parametrized Tests

```python
@pytest.mark.unit
@pytest.mark.parametrize("input_size,output_size", [
    (2, 2),
    (4, 3),
    (10, 5),
    (1, 2),
])
def test_various_network_sizes(input_size, output_size):
    """Network should work with various input/output sizes."""
    config = CascadeCorrelationConfig(
        input_size=input_size,
        output_size=output_size,
    )
    network = CascadeCorrelationNetwork(config=config)

    x = torch.randn(20, input_size)
    output = network.forward(x)

    assert output.shape == (20, output_size)


@pytest.mark.unit
@pytest.mark.parametrize("optimizer", ['Adam', 'SGD', 'RMSprop', 'AdamW'])
def test_different_optimizers(optimizer):
    """Network should work with different optimizers."""
    from cascade_correlation.cascade_correlation_config.cascade_correlation_config import OptimizerConfig

    opt_config = OptimizerConfig(optimizer_type=optimizer)
    config = CascadeCorrelationConfig(optimizer_config=opt_config)
    network = CascadeCorrelationNetwork(config=config)

    x = torch.randn(20, 2)
    y = torch.randint(0, 2, (20, 2)).float()

    # Should not raise
    history = network.fit(x, y, epochs=5)
    assert 'train_loss' in history
```

---

**Document Version**: 0.3.21  
**Last Updated**: 2026-01-29
