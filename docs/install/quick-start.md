# Quick Start Guide

Get Juniper Cascor running in minutes.

## Prerequisites

- **Python**: 3.11 or higher (CI uses 3.14)
- **Conda**: Anaconda or Miniconda installed
- **Git**: For cloning the repository
- **JuniperData**: Service must be running for spiral problem evaluation

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/pcalnon/juniper_cascor.git
cd juniper_cascor
```

### 2. Create the Conda Environment

```bash
conda env create -f conf/conda_environment.yaml
```

This installs all dependencies including PyTorch, NumPy, pytest, and other required packages.

### 3. Activate the Environment

```bash
conda activate juniper_cascor
```

### 4. Start JuniperData Service

```bash
# Start JuniperData service (required for spiral datasets)
cd ../JuniperData/juniper_data
uvicorn juniper_data.api.main:app --port 8100 &
```

Verify JuniperData is running:

```bash
curl http://localhost:8100/health
```

## Running the Application

### Run the Spiral Problem Evaluation

```bash
cd src && python main.py
```

**Expected Output**: The application trains a Cascade Correlation neural network on the classic two-spiral classification problem. You'll see:

- Training progress with epoch counts and error metrics
- Candidate unit training and network growth
- Final accuracy on the spiral classification task
- Generated plots saved to the `images/` directory

## Running Tests

### Run All Tests

```bash
cd src/tests && bash scripts/run_tests.bash
```

### Run Fast Tests Only

Skip slow tests for quicker feedback:

```bash
cd src/tests
python -m pytest -m "not slow"
```

### Run Tests with Coverage

```bash
cd src/tests && bash scripts/run_tests.bash -v -c
```

Coverage reports are generated at `src/tests/reports/htmlcov/index.html`.

## Troubleshooting

If you see connection errors, ensure JuniperData is running on port 8100.

## Next Steps

- [AGENTS.md](../../AGENTS.md) — Full project guide with commands, conventions, and architecture
- [README.md](../../README.md) — Project overview and research background
- [notes/FEATURES_GUIDE.md](../../notes/FEATURES_GUIDE.md) — Feature documentation and usage
- [src/tests/README.md](../../src/tests/README.md) — Test suite documentation
