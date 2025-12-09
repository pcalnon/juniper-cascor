# Cascade Correlation Neural Network Test Suite

This directory contains a comprehensive testing framework for the Cascade Correlation Neural Network implementation.

## Overview

The test suite is designed with the following principles:
- **Clarity**: Easy to understand test structure and naming
- **Ease of Use**: Simple commands to run specific test categories
- **Scalability**: Modular structure that grows with the codebase
- **Ongoing Development**: Maintainable and extensible architecture

## Directory Structure

```
tests/
├── conftest.py                 # Pytest configuration and shared fixtures
├── pytest.ini                 # Pytest settings and markers
├── README.md                   # This file
│
├── fixtures/                   # Test fixtures and data setup
│   ├── network_fixtures.py     # Network instance fixtures
│   └── candidate_fixtures.py   # Candidate unit fixtures
│
├── helpers/                    # Testing utilities
│   ├── assertions.py           # Custom assertion functions
│   └── utilities.py            # Helper functions and utilities
│
├── mocks/                      # Mock objects for testing
│   ├── mock_candidate.py       # Mock candidate units
│   └── mock_logger.py          # Mock logger objects
│
├── test_data/                  # Test data generation
│   ├── generators.py           # Data generators for various problems
│   └── fixtures.py             # Data fixtures
│
├── unit/                       # Unit tests for individual components
│   ├── test_forward_pass.py    # Forward pass algorithm tests
│   ├── test_residual_error.py  # Residual error calculation tests
│   ├── test_accuracy.py        # Accuracy calculation tests
│   ├── test_candidate_training.py  # Candidate training tests
│   ├── test_network_growth.py  # Network growth algorithm tests
│   ├── test_unit_addition.py   # Unit addition tests
│   ├── test_output_training.py # Output layer training tests
│   ├── test_early_stopping.py  # Early stopping logic tests
│   ├── test_validation.py      # Input validation tests
│   └── test_activation_functions.py  # Activation function tests
│
├── integration/                # Integration tests
│   ├── test_spiral_problem.py  # Spiral problem solving tests
│   ├── test_full_training.py   # Complete training workflow tests
│   └── test_multiprocessing.py # Multiprocessing functionality tests
│
├── performance/                # Performance and benchmark tests
│   ├── test_benchmarks.py      # Performance benchmarks
│   └── test_memory_usage.py    # Memory usage tests
│
└── scripts/                    # Test runner scripts
    ├── run_tests.sh             # Main test runner script
    ├── run_coverage.sh          # Coverage analysis script
    └── setup_test_env.sh        # Environment setup script
```

## Quick Start

### Run All Unit Tests
```bash
cd src/tests
./scripts/run_tests.sh
```

### Run Specific Test Categories
```bash
# Unit tests only
./scripts/run_tests.sh -u

# Integration tests
./scripts/run_tests.sh -i

# Performance tests  
./scripts/run_tests.sh -p

# Spiral problem tests specifically
./scripts/run_tests.sh -m "spiral"
```

### Run Specific Test Files
```bash
./scripts/run_tests.sh -t unit/test_forward_pass.py
./scripts/run_tests.sh -t integration/test_spiral_problem.py
```

### Run with Different Options
```bash
# Verbose output with coverage
./scripts/run_tests.sh -v -c

# Parallel execution without coverage
./scripts/run_tests.sh -j --no-coverage

# Slow tests with GPU support
./scripts/run_tests.sh -s -g

# Re-run only failed tests
./scripts/run_tests.sh -f
```

## Test Categories and Markers

The test suite uses pytest markers to categorize tests:

- `unit`: Unit tests for individual components
- `integration`: Integration tests for full workflows
- `performance`: Performance and benchmarking tests
- `slow`: Tests that take a long time to run
- `gpu`: Tests that require GPU/CUDA
- `multiprocessing`: Tests that use multiprocessing
- `spiral`: Tests specifically for spiral problem solving
- `correlation`: Tests for correlation coefficient calculations
- `network_growth`: Tests for network growth algorithms
- `candidate_training`: Tests for candidate unit training
- `validation`: Tests for input validation functions
- `accuracy`: Tests for accuracy calculation methods
- `early_stopping`: Tests for early stopping logic

## Key Test Components

### Core Algorithm Tests (Unit)

1. **Forward Pass Tests** (`test_forward_pass.py`)
   - Network prediction with varying hidden unit counts
   - Shape validation and gradient flow
   - Cascading connection behavior

2. **Candidate Training Tests** (`test_candidate_training.py`)
   - Pool-based candidate generation
   - Correlation coefficient calculation
   - Multiprocessing coordination

3. **Network Growth Tests** (`test_network_growth.py`)
   - Iterative network expansion
   - Unit addition and integration
   - Dynamic weight matrix updates

4. **Accuracy Tests** (`test_accuracy.py`)
   - Classification accuracy calculation
   - One-hot encoded target handling
   - Edge cases and validation

5. **Residual Error Tests** (`test_residual_error.py`)
   - Error computation between predictions and targets
   - Shape compatibility and numerical stability

### Integration Tests

1. **Spiral Problem Tests** (`test_spiral_problem.py`)
   - Classic 2-spiral problem solving
   - N-spiral generalization capability
   - Noise robustness and scaling behavior

2. **Full Training Tests** (`test_full_training.py`)
   - Complete training workflows
   - Early stopping behavior
   - Training history tracking

### Utilities and Helpers

1. **Custom Assertions** (`helpers/assertions.py`)
   - Domain-specific validation functions
   - Network structure validation
   - Correlation and accuracy bounds checking

2. **Test Data Generators** (`test_data/generators.py`)
   - Spiral problem data generation
   - Classification and regression datasets
   - Correlation test data

3. **Mock Objects** (`mocks/`)
   - Controllable candidate units for testing
   - Mock loggers and configurations
   - Scenario-based testing support

## Adding New Tests

### For New Algorithms

1. Create a new test file in `unit/` following the naming convention `test_<algorithm_name>.py`
2. Use the existing test structure with test classes for different aspects
3. Add appropriate markers (e.g., `@pytest.mark.unit`, `@pytest.mark.<algorithm>`)
4. Include tests for:
   - Basic functionality
   - Shape validation
   - Edge cases
   - Error handling
   - Numerical stability

### For New Integration Scenarios

1. Create test file in `integration/` with descriptive name
2. Use `@pytest.mark.integration` marker
3. Test complete workflows rather than isolated components
4. Include performance expectations and assertions

### Example Test Structure

```python
class TestNewAlgorithm:
    """Test basic functionality of new algorithm."""
    
    @pytest.mark.unit
    @pytest.mark.new_algorithm
    def test_basic_functionality(self, simple_network):
        """Test that algorithm works with valid input."""
        # Arrange
        input_data = create_test_input()
        
        # Act
        result = simple_network.new_algorithm(input_data)
        
        # Assert
        assert_valid_output(result)
    
    @pytest.mark.unit
    @pytest.mark.new_algorithm
    @pytest.mark.parametrize("param", [1, 5, 10])
    def test_parameter_variations(self, simple_network, param):
        """Test algorithm with different parameter values."""
        # Test implementation
        pass
```

## Configuration Files

### pytest.ini
Contains pytest configuration including:
- Test discovery patterns
- Coverage settings
- Marker definitions
- Output formatting

### conftest.py
Provides shared fixtures including:
- Network instances with different configurations
- Test data generators
- Mock objects
- Utility functions

## Best Practices

1. **Test Naming**: Use descriptive names that clearly indicate what is being tested
2. **Test Structure**: Follow Arrange-Act-Assert pattern
3. **Fixtures**: Use fixtures for common setup to avoid code duplication  
4. **Markers**: Use appropriate markers to enable selective test running
5. **Assertions**: Use custom assertions for domain-specific validations
6. **Documentation**: Include docstrings explaining the purpose of complex tests
7. **Parametrization**: Use `@pytest.mark.parametrize` for testing multiple scenarios
8. **Mocking**: Use mocks to isolate components and control test conditions

## Coverage and Reporting

The test suite generates several types of reports:

- **HTML Coverage Report**: `tests/reports/htmlcov/index.html`
- **XML Coverage Report**: `tests/reports/coverage.xml`
- **JUnit XML Report**: `tests/reports/junit.xml`

These reports help track test coverage and can be integrated with CI/CD systems.

## Continuous Integration

The test structure is designed to work well with CI/CD systems:

1. Fast unit tests can run on every commit
2. Integration tests can run on pull requests
3. Performance tests can run on releases
4. Coverage reports can be uploaded to coverage tracking services

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure Python path includes the project root
2. **CUDA Errors**: Use `--gpu` flag only on systems with CUDA support
3. **Multiprocessing Issues**: Some tests may fail on systems with limited resources
4. **Random Test Failures**: Set deterministic behavior using provided utilities

### Debug Mode

Run tests with maximum verbosity for debugging:
```bash
./scripts/run_tests.sh -v -s --tb=long
```

### Running Individual Test Methods

```bash
pytest unit/test_forward_pass.py::TestForwardPassBasics::test_forward_pass_no_hidden_units -v
```

This comprehensive test suite ensures the robustness and correctness of the Cascade Correlation Neural Network implementation while providing a foundation for ongoing development and testing.
