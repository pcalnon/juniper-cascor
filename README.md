# Juniper: Dynamic Neural Network Research Platform

[![CI/CD Pipeline](https://github.com/pcalnon/juniper_cascor/actions/workflows/ci.yml/badge.svg)](https://github.com/pcalnon/juniper_cascor/actions/workflows/ci.yml)

**Version**: 0.3.21 | [Changelog](CHANGELOG.md) | [Full Documentation](docs/index.md)

Juniper is an AI/ML research platform for investigating dynamic neural network architectures and novel learning paradigms. The project emphasizes ground-up implementations from primary literature, enabling a more transparent exploration of fundamental algorithms.

## Quick Start

### Prerequisites

- Python 3.11 or later (3.14 recommended)
- Conda package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/pcalnon/juniper_cascor.git
cd juniper_cascor

# Create and activate conda environment
conda env create -f conf/conda_environment.yaml
conda activate juniper_cascor

# Run the spiral problem evaluation
cd src && python main.py
```

### Run Tests

```bash
# Fast tests (recommended for development)
cd src/tests && bash scripts/run_tests.bash

# Or using pytest directly
pytest -m "not slow" -v
```

For detailed installation instructions, see the [Quick Start Guide](docs/install/quick-start.md).

## Active Research Components

**juniper_cascor**: Cascade Correlation Neural Network

- Reference implementation from foundational research (Fahlman & Lebiere, 1990)
- Designed for flexibility, modularity, and scalability
- Enables investigation of constructive learning algorithms

**juniper_canopy**: Interactive Research Interface

- Research-driven monitoring and visualization environment
- Delivers novel observations through real-time network introspection
- Transforms metrics into insights, accelerating experimental iteration

## Documentation

| Category | Description |
|----------|-------------|
| [Documentation Index](docs/index.md) | Complete documentation overview |
| [Quick Start](docs/install/quick-start.md) | Get up and running quickly |
| [User Manual](docs/install/user-manual.md) | Comprehensive usage guide |
| [API Reference](docs/api/api-reference.md) | Complete API documentation |
| [Testing Guide](docs/testing/quick-start.md) | Testing instructions |
| [CI/CD Guide](docs/ci/quick-start.md) | Continuous integration |
| [Source Code Guide](docs/source/quick-start.md) | Contributor documentation |
| [Constants Reference](docs/overview/constants-guide.md) | Configuration constants |

## Basic Usage

```python
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
import torch

# Create configuration
config = CascadeCorrelationConfig(
    input_size=2,
    output_size=2,
    random_seed=42
)

# Create and train network
network = CascadeCorrelationNetwork(config=config)
history = network.fit(x_train, y_train, epochs=100)

# Evaluate
accuracy = network.get_accuracy(x_test, y_test)
print(f"Test accuracy: {accuracy:.2%}")

# Save/Load
network.save_to_hdf5("model.h5")
loaded = CascadeCorrelationNetwork.load_from_hdf5("model.h5")
```

## Research Philosophy

Juniper prioritizes **transparency over convenience** and **understanding over abstraction**. By implementing algorithms from first principles, the platform provides researchers with increased visibility into network behavior, enabling a more rigorous and more controlled investigation of learning dynamics and architectural innovations.

## Important Notices

### Thread Safety Warning

**The `CascadeCorrelationNetwork` class is NOT thread-safe.** Do not share network instances between threads without proper synchronization. For concurrent training scenarios, create separate network instances per thread. The internal multiprocessing for candidate training is handled within the class and does not require external synchronization.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome. Please see the [Source Code Guide](docs/source/quick-start.md) for development setup and coding conventions.
