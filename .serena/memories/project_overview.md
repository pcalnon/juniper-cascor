# Juniper Cascor Project Overview

## Purpose

Juniper Cascor is an AI/ML research platform implementing the **Cascade Correlation Neural Network** algorithm from foundational research (Fahlman & Lebiere, 1990). The project emphasizes ground-up implementations from primary literature for transparent exploration of constructive learning algorithms.

### Research Philosophy

- **Transparency over convenience**: Algorithms implemented from first principles
- **Understanding over abstraction**: Full visibility into network behavior
- **Modularity and scalability**: Designed for research flexibility

## Tech Stack

- **Python**: 3.11+ (targets 3.11, 3.12, 3.13, 3.14)
- **PyTorch**: Neural network tensors and operations
- **NumPy**: Numerical computations
- **h5py**: HDF5 file serialization
- **matplotlib**: Plotting and visualization
- **PyYAML**: YAML configuration parsing
- **pytest**: Test framework with coverage

## Project Structure

```text
juniper_cascor/
├── src/                              # Source code root
│   ├── main.py                       # Application entry point
│   ├── cascade_correlation/          # Core Cascor network implementation
│   ├── candidate_unit/               # Candidate unit implementation
│   ├── spiral_problem/               # Two-spiral problem implementation
│   ├── cascor_constants/             # All project constants
│   ├── log_config/                   # Logging configuration
│   ├── cascor_plotter/               # Visualization utilities
│   ├── snapshots/                    # HDF5 serialization system
│   ├── profiling/                    # Profiling infrastructure
│   ├── remote_client/                # Remote multiprocessing client
│   ├── utils/                        # Utility functions
│   └── tests/                        # Test suite
├── conf/                             # Configuration files
├── util/                             # Shell utility scripts
├── notes/                            # Project documentation
├── data/                             # Data directory (gitignored)
├── logs/                             # Log files (gitignored)
└── pyproject.toml                    # Project configuration
```

## Key Entry Points

| File | Purpose |
|------|---------|
| `src/main.py` | Main application entry point |
| `src/cascade_correlation/cascade_correlation.py` | Core neural network implementation |
| `src/spiral_problem/spiral_problem.py` | Two-spiral problem solver |
| `src/candidate_unit/candidate_unit.py` | Candidate unit for network growth |

## Git Repository Note

This project is part of a single GitHub repo (`pcalnon/Juniper`) that contains multiple sub-projects (Juniper, JuniperData, JuniperCascor, JuniperCanopy). Each sub-project has its own local directory with a clone of the repo, working on separate branches.
