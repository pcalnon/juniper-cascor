# Environment Setup Guide

This guide covers setting up the development environment for the Juniper Cascor project.

## System Requirements

### Supported Operating Systems

| OS | Status | Notes |
|----|--------|-------|
| Linux (Ubuntu 22.04+) | ✅ Fully Supported | Ubuntu 25.10 tested in CI |
| macOS (12+) | ✅ Supported | Intel and Apple Silicon |
| Windows | ⚠️ Not Tested | May work with WSL2 |

### Python Version

- **Required**: Python 3.11 or higher
- **Supported**: 3.11, 3.12, 3.13, 3.14
- **CI Uses**: Python 3.14

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Any x64/ARM64 | Multi-core for parallel candidate training |
| RAM | 4 GB | 8+ GB for large networks |
| GPU | Not required | NVIDIA GPU with CUDA 12.x for acceleration |
| Storage | 500 MB | 2+ GB with conda environment |

---

## Conda Environment Setup (Recommended)

### Prerequisites

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download).

### Creating the Environment

```bash
# Navigate to project root
cd juniper_cascor

# Create environment from yaml file
conda env create -f conf/conda_environment.yaml --name juniper_cascor

# Alternatively, create with explicit name
conda env create -f conf/conda_environment.yaml -n cascor
```

### Activating the Environment

```bash
# Activate the environment
conda activate juniper_cascor

# Verify activation
which python
# Should show: /path/to/conda/envs/juniper_cascor/bin/python
```

### Verifying Installation

```bash
# Check Python version
python --version
# Expected: Python 3.14.x

# Verify core packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import h5py; print(f'h5py: {h5py.__version__}')"

# Run a quick test
cd src/tests && python -m pytest unit/test_forward_pass.py -v --collect-only
```

### Updating the Environment

```bash
# Update existing environment
conda env update -f conf/conda_environment.yaml --prune

# Or recreate from scratch
conda deactivate
conda env remove -n juniper_cascor
conda env create -f conf/conda_environment.yaml --name juniper_cascor
```

---

## Manual Dependency Installation (pip)

If not using Conda, install dependencies via pip.

### Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
```

### Install Core Dependencies

```bash
# Core packages
pip install torch numpy matplotlib h5py PyYAML

# Testing packages
pip install pytest pytest-cov

# System monitoring
pip install psutil

# Optional: columnar for formatted output
pip install columnar
```

### Key Packages List

| Package | Purpose | Version |
|---------|---------|---------|
| `torch` | Neural network tensors and autograd | 2.9+ |
| `numpy` | Numerical computations | 2.4+ |
| `matplotlib` | Plotting and visualization | Latest |
| `h5py` | HDF5 serialization | Latest |
| `PyYAML` | YAML configuration parsing | 6.0+ |
| `pytest` | Test framework | 9.0+ |
| `pytest-cov` | Coverage reporting | Latest |
| `psutil` | Process/system utilities | Latest |

---

## GPU Configuration

### PyTorch CUDA Setup

The conda environment includes CUDA 12.x libraries. For pip installations:

```bash
# Install PyTorch with CUDA 12.x support
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Or for CPU-only (smaller install)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Verifying GPU Availability

```bash
python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
"
```

### Running GPU Tests

Tests disable GPU by default. To run GPU-specific tests:

```bash
cd src/tests

# Run only GPU tests
python -m pytest -m "gpu" -v

# Run all tests including GPU
python -m pytest -v

# Exclude GPU tests explicitly
python -m pytest -m "not gpu" -v
```

---

## Development Tools Setup

### Code Formatting and Linting

The project uses Black, isort, flake8, and mypy (configured in `pyproject.toml`).

```bash
# Install development tools
pip install black isort flake8 mypy

# Format code with Black
cd src && python -m black .

# Sort imports with isort
cd src && python -m isort .

# Lint with flake8
cd src && python -m flake8 . --max-line-length=120 --extend-ignore=E203,E266,E501,W503

# Type check with mypy
cd src && python -m mypy cascade_correlation/ candidate_unit/ --ignore-missing-imports
```

### Format Checking (CI Mode)

```bash
# Check formatting without changes
cd src && python -m black --check --diff .
cd src && python -m isort --check-only --diff .
```

### Trunk (Optional)

If [Trunk](https://trunk.io) is installed:

```bash
# Run all configured linters
trunk check

# Auto-fix issues
trunk check --fix
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'torch'` | PyTorch not installed | `pip install torch` or recreate conda env |
| `ModuleNotFoundError: No module named 'h5py'` | h5py not installed | `pip install h5py` |
| `h5py installation fails on macOS` | Missing HDF5 libraries | `brew install hdf5` then `pip install h5py` |
| `torch.cuda.is_available()` returns False | No NVIDIA GPU or driver issue | Install NVIDIA drivers; verify with `nvidia-smi` |
| `RuntimeError: CUDA out of memory` | GPU memory exhausted | Reduce batch size or use CPU |
| `BrokenPipeError` in multiprocessing | macOS fork safety | Set `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` |
| `pytest: command not found` | pytest not in PATH | `pip install pytest` or activate venv |
| `_pickle.PicklingError` with logger | Logger in pickled object | Already handled in codebase; check custom classes |

### PyTorch Installation Issues

**Linux (missing CUDA)**:

```bash
# Install NVIDIA driver
sudo apt install nvidia-driver-535

# Verify
nvidia-smi
```

**macOS (Apple Silicon)**:

```bash
# PyTorch MPS backend (Metal)
pip install torch

# Verify MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"
```

### h5py Installation Issues

**Linux**:

```bash
sudo apt install libhdf5-dev
pip install --no-cache-dir h5py
```

**macOS**:

```bash
brew install hdf5
export HDF5_DIR=$(brew --prefix hdf5)
pip install --no-cache-dir h5py
```

### Multiprocessing Issues

**macOS spawn vs fork**:

The project uses multiprocessing for candidate training. On macOS, you may need:

```bash
# In your shell profile (.zshrc or .bashrc)
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

**Linux (too many open files)**:

```bash
# Increase file descriptor limit
ulimit -n 4096
```

### Environment Variable Configuration

```bash
# Set log level for quieter output
export CASCOR_LOG_LEVEL=WARNING

# Debug mode for verbose logging
export CASCOR_LOG_LEVEL=DEBUG

# Trace mode for maximum verbosity
export CASCOR_LOG_LEVEL=TRACE
```

---

## Verification Checklist

Run these commands to verify your environment:

```bash
# 1. Python version
python --version  # Should be 3.11+

# 2. Core imports
python -c "import torch, numpy, h5py, yaml, matplotlib; print('Core imports: OK')"

# 3. Test framework
python -m pytest --version

# 4. Run minimal test
cd src/tests && python -m pytest unit/test_forward_pass.py::test_forward_returns_tensor -v

# 5. Check GPU (optional)
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

If all checks pass, your environment is ready for development.

---

## Next Steps

- Run the full test suite: `cd src/tests && bash scripts/run_tests.bash`
- Try the spiral problem: `cd src && python main.py`
- Read [AGENTS.md](../../AGENTS.md) for project conventions
