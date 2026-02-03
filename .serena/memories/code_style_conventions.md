# Code Style and Conventions

## Naming Conventions

### Constants

- Uppercase with underscores, prefixed by component: `_CASCOR_LOG_LEVEL_NAME`
- Hierarchical naming: `_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION`

### Classes

- PascalCase: `CascadeCorrelationNetwork`, `CandidateUnit`

### Methods/Functions

- snake_case: `train_candidates`, `calculate_correlation`
- Private methods prefixed with underscore: `_prepare_candidate_input`

### Constructor Parameters

- Name-mangled style with class prefix: `_SpiralProblem__n_points`, `CandidateUnit__input_size`

## File Headers

All Python files include standardized headers:

```python
#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCascor
# Application:   juniper_cascor
# Purpose:       Juniper Project Cascade Correlation Neural Network
#
# Author:        Paul Calnon
# Version:       0.3.2 (0.7.3)
# File Name:     [File Name]
# File Path:     [Project]/[Sub-Project]/[Application]/src/

# Date Created:  [YYYY-MM-DD]
# Last Modified: [YYYY-MM-DD HH:MM:SS TZ]
#
# License:       MIT License
# Copyright:     Copyright (c) 2024,2025,2026 Paul Calnon
#
# Description:
#     [This is a placeholder for the actual description.]
#
#####################################################################################################################################################################################################
# Notes:
#
#####################################################################################################################################################################################################
# References:
#
#####################################################################################################################################################################################################
# TODO :
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#####################################################################################################################################################################################################
```

## Import Ordering

1. Standard library imports
2. Third-party imports (numpy, torch, etc.)
3. Local application imports

Path manipulation for local imports:

```python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

## Type Hints

The project uses Python type hints extensively:

```python
def forward(self, x: torch.Tensor = None) -> torch.Tensor:

def _calculate_correlation(
    self,
    output: torch.Tensor = None,
    residual_error: torch.Tensor = None,
) -> tuple([float, torch.Tensor, torch.Tensor, float, float]):
```

## Logging

Custom logging system with extended log levels:

- TRACE, VERBOSE, DEBUG, INFO, WARNING, ERROR, CRITICAL, FATAL

Logger usage pattern:

```python
from log_config.logger.logger import Logger
self.logger = Logger
self.logger.info("Message")
self.logger.trace("Detailed trace message")
```

## Formatting Configuration

- **Line length**: 512 characters (configured in Black, isort, flake8)
- **Black**: Python code formatting
- **isort**: Import sorting with black profile
- **Flake8**: Linting with extended ignores: E203, E265, E266, E501, W503, E722, E402, E226

## Documentation

- Docstrings follow structured format with Description, Args, Returns, Raises, Notes sections
- Extensive inline logging for debugging and tracing

## Multiprocessing Considerations

Classes implement `__getstate__` and `__setstate__` to handle non-picklable objects:

```python
def __getstate__(self):
    state = self.__dict__.copy()
    state.pop('logger', None)
    state.pop('_candidate_display_progress', None)
    return state

def __setstate__(self, state):
    self.__dict__.update(state)
    self.logger = Logger
```
