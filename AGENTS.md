# AGENTS.md - Juniper Cascor Project Guide

**Project**: Juniper Cascade Correlation Neural Network  
**Version**: 0.3.17
**License**: MIT License
**Author**: Paul Calnon
**Last Updated**: 2026-02-05

---

## Quick Reference

### Essential Commands

```bash
# Run the application
cd src && python main.py

# Run all unit tests
cd src/tests && bash scripts/run_tests.bash

# Run specific test categories
cd src/tests && bash scripts/run_tests.bash -u              # Unit tests only
cd src/tests && bash scripts/run_tests.bash -i              # Integration tests
cd src/tests && bash scripts/run_tests.bash -m "spiral"     # Spiral problem tests

# Run tests with coverage
cd src/tests && bash scripts/run_tests.bash -v -c

# Run a specific test file
cd src/tests && python -m pytest unit/test_forward_pass.py -v

# Run performance benchmarks (P3-004)
cd src/tests/scripts && bash run_benchmarks.bash           # All benchmarks
cd src/tests/scripts && bash run_benchmarks.bash -s        # Serialization only
cd src/tests/scripts && bash run_benchmarks.bash -q -n 10  # Quiet mode, 10 iterations

# Profiling commands (P3-NEW-001, P3-NEW-002)
cd src && python main.py --profile                         # cProfile deterministic profiling
cd src && python main.py --profile-memory                  # tracemalloc memory profiling
cd src && python main.py --profile --profile-output ./my_profiles  # Custom output dir
./util/profile_training.bash                               # py-spy sampling profiler
./util/profile_training.bash --svg                         # Generate SVG flame graph
./util/profile_training.bash --speedscope                  # Speedscope JSON format

# Type checking with mypy (configured in pyproject.toml)
cd src && python -m mypy cascade_correlation/ candidate_unit/ --ignore-missing-imports

# Linting with flake8
cd src && python -m flake8 . --max-line-length=512 --extend-ignore=E203,E266,E501,W503

# Format checking with black
cd src && python -m black --check --diff .

# Import sorting check with isort
cd src && python -m isort --check-only --diff .

# Linting via trunk (if available)
trunk check

# Pre-commit hooks (CI/CD local validation)
pip install pre-commit                    # Install pre-commit (one-time)
pre-commit install                        # Install git hooks (one-time)
pre-commit run --all-files                # Run all hooks on all files
pre-commit run black --all-files          # Run specific hook

# Security scanning
pip install bandit pip-audit              # Install security tools
bandit -r src/                            # Run Bandit SAST scan
pip-audit                                 # Check for dependency vulnerabilities
```

### Environment Variables

| Variable              | Description                                   | Example Values                      |
| --------------------- | --------------------------------------------- | ----------------------------------- |
| `CASCOR_LOG_LEVEL`    | Override log level at runtime (P2-003)        | `WARNING`, `INFO`, `DEBUG`, `ERROR` |
| `JUNIPER_DATA_URL`    | JuniperData service URL (REQUIRED for datasets) | `http://localhost:8100`             |
| `CASCOR_BACKEND_PATH` | Path to Cascor src (used by Canopy)           | `/path/to/juniper_cascor`           |
| `JUNIPER_DATA_API_KEY` | API key for JuniperData authentication       | `your-api-key-here`                 |

**Log Level Override Examples:**

```bash
# Quiet mode for production/benchmarking (less verbose)
export CASCOR_LOG_LEVEL=WARNING

# Debug mode for verbose output
export CASCOR_LOG_LEVEL=DEBUG

# Trace mode for maximum verbosity
export CASCOR_LOG_LEVEL=TRACE
```

### Key Entry Points

| File                                             | Purpose                              |
| ------------------------------------------------ | ------------------------------------ |
| `src/main.py`                                    | Main application entry point         |
| `src/cascade_correlation/cascade_correlation.py` | Core neural network implementation   |
| `src/spiral_problem/spiral_problem.py`           | Two-spiral problem solver            |
| `src/candidate_unit/candidate_unit.py`           | Candidate unit for network growth    |
| `src/profiling/`                                 | Profiling infrastructure (P3-NEW-001)|
| `src/tests/run_tests.bash`                       | Test runner script                   |
| `src/tests/conftest.py`                          | Test configuration and fixtures      |
| `util/profile_training.bash`                     | py-spy sampling profiler (P3-NEW-002)|

---

## Project Overview

Juniper Cascor is an AI/ML research platform implementing the **Cascade Correlation Neural Network** algorithm from foundational research (Fahlman & Lebiere, 1990). The project emphasizes ground-up implementations from primary literature for transparent exploration of constructive learning algorithms.

### Research Philosophy

- **Transparency over convenience**: Algorithms implemented from first principles
- **Understanding over abstraction**: Full visibility into network behavior
- **Modularity and scalability**: Designed for research flexibility

---

## Directory Structure

```bash
juniper_cascor/
├── src/                              # Source code root
│   ├── main.py                       # Application entry point
│   ├── cascade_correlation/          # Core Cascor network implementation
│   │   ├── cascade_correlation.py    # Main CascadeCorrelationNetwork class
│   │   ├── cascade_correlation_config/  # Network configuration
│   │   └── cascade_correlation_exceptions/  # Custom exceptions
│   ├── candidate_unit/               # Candidate unit implementation
│   │   └── candidate_unit.py         # CandidateUnit class
│   ├── spiral_problem/               # Two-spiral problem implementation
│   │   └── spiral_problem.py         # SpiralProblem class
│   ├── cascor_constants/             # All project constants (renamed from constants/)
│   │   ├── constants.py              # Main constants aggregator
│   │   ├── constants_activation/     # Activation function constants
│   │   ├── constants_candidates/     # Candidate training constants
│   │   ├── constants_hdf5/           # HDF5 serialization constants
│   │   ├── constants_logging/        # Logging constants
│   │   ├── constants_model/          # Model architecture constants
│   │   └── constants_problem/        # Problem-specific constants
│   ├── log_config/                   # Logging configuration
│   │   ├── log_config.py             # LogConfig class
│   │   └── logger/                   # Custom Logger class
│   ├── cascor_plotter/               # Visualization utilities
│   │   └── cascor_plotter.py         # CascadeCorrelationPlotter class
│   ├── snapshots/                    # HDF5 serialization system
│   │   ├── snapshot_serializer.py    # Main serialization logic
│   │   ├── snapshot_utils.py         # Utility functions
│   │   ├── snapshot_cli.py           # CLI tools
│   │   └── snapshot_common.py        # Common serialization helpers
│   ├── remote_client/                # Remote multiprocessing client
│   ├── utils/                        # Utility functions
│   │   └── utils.py                  # Helper functions
│   └── tests/                        # Test suite
│       ├── conftest.py               # Pytest configuration and fixtures
│       ├── pytest.ini                # Pytest settings
│       ├── unit/                     # Unit tests
│       ├── integration/              # Integration tests
│       ├── helpers/                  # Test utilities
│       ├── mocks/                    # Mock objects
│       └── scripts/                  # Test runner scripts
├── conf/                             # Configuration files
│   ├── conda_environment.yaml        # Conda environment
│   ├── logging_config.yaml           # Logging configuration
│   └── *.conf                        # Various shell config files
├── util/                             # Shell utility scripts
├── notes/                            # Project documentation
├── data/                             # Data directory (gitignored)
├── logs/                             # Log files (gitignored)
├── images/                           # Generated images
├── reports/                          # Generated reports
└── README.md                         # Project README
```

---

## Core Components

### CascadeCorrelationNetwork

The main neural network class implementing the Cascade Correlation algorithm.

**Location**: `src/cascade_correlation/cascade_correlation.py`

**Key Methods**:

- `fit(x, y, epochs)` - Train the network
- `forward(x)` - Forward pass through network
- `train_output_layer(x, y, epochs)` - Train output layer only
- `train_candidates(x, y, residual_error)` - Train candidate pool
- `get_accuracy(x, y)` - Calculate classification accuracy
- `save_to_hdf5(filepath)` - Save network to HDF5
- `load_from_hdf5(filepath)` - Load network from HDF5
- `create_snapshot()` - Create network snapshot

**Key Dataclasses**:

- `TrainingResults` - Aggregated candidate training results
- `ValidateTrainingInputs` - Inputs for training validation
- `ValidateTrainingResults` - Results from validation

### CandidateUnit

Represents a candidate hidden unit in the network.

**Location**: `src/candidate_unit/candidate_unit.py`

**Key Methods**:

- `train(x, residual_error, epochs)` - Train the candidate
- `_calculate_correlation(output, residual_error)` - Calculate Pearson correlation
- `_update_weights_and_bias(params)` - Update using autograd

**Key Dataclasses**:

- `CandidateTrainingResult` - Training result for single candidate
- `CandidateParametersUpdate` - Parameters for weight updates
- `CandidateCorrelationCalculation` - Correlation calculation results

### SpiralProblem

Implements the classic two-spiral classification problem.

**Location**: `src/spiral_problem/spiral_problem.py`

**Key Methods**:

- `evaluate(...)` - Run full evaluation pipeline
- `generate_spiral_dataset(...)` - Generate spiral data
- `_create_input_features(...)` - Create input tensors
- `_create_one_hot_targets(...)` - Create target tensors

---

## Programming Conventions

### Naming Conventions

**Constants**:

- Uppercase with underscores, prefixed by component: `_CASCOR_LOG_LEVEL_NAME`
- Hierarchical naming: `_CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION`

**Classes**:

- PascalCase: `CascadeCorrelationNetwork`, `CandidateUnit`

**Methods/Functions**:

- snake_case: `train_candidates`, `calculate_correlation`
- Private methods prefixed with underscore: `_prepare_candidate_input`

**Constructor Parameters**:

- Name-mangled style with class prefix: `_SpiralProblem__n_points`, `CandidateUnit__input_size`

### File Headers

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

### Imports

Standard ordering:

1. Standard library imports
2. Third-party imports (numpy, torch, etc.)
3. Local application imports

Path manipulation for local imports:

```python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### Type Hints

The project uses Python type hints extensively:

```python
def forward(self, x: torch.Tensor = None) -> torch.Tensor:
def _calculate_correlation(
    self,
    output: torch.Tensor = None,
    residual_error: torch.Tensor = None,
) -> tuple([float, torch.Tensor, torch.Tensor, float, float]):
```

### Logging

Custom logging system with extended log levels:

- TRACE, VERBOSE, DEBUG, INFO, WARNING, ERROR, CRITICAL, FATAL

Logger usage pattern:

```python
from log_config.logger.logger import Logger
self.logger = Logger
self.logger.info("Message")
self.logger.trace("Detailed trace message")
self.logger.verbose("Verbose output")
```

### Documentation

- Docstrings follow structured format with Description, Args, Returns, Raises, Notes sections
- Extensive inline logging for debugging and tracing

---

## Testing Infrastructure

### Test Framework

- **Framework**: pytest
- **Location**: `src/tests/`
- **Configuration**: `src/tests/pytest.ini`
- **Fixtures**: `src/tests/conftest.py`

### Test Categories (Markers)

| Marker               | Description                                     |
| -------------------- | ----------------------------------------------- |
| `unit`               | Unit tests for individual components            |
| `integration`        | Integration tests for full workflows            |
| `performance`        | Performance and benchmarking tests              |
| `slow`               | Tests that take a long time to run              |
| `long`               | Long-running correctness tests (use --run-long) |
| `gpu`                | Tests that require GPU/CUDA                     |
| `multiprocessing`    | Tests using multiprocessing                     |
| `spiral`             | Spiral problem tests                            |
| `correlation`        | Correlation calculation tests                   |
| `network_growth`     | Network growth algorithm tests                  |
| `candidate_training` | Candidate unit training tests                   |
| `validation`         | Input validation tests                          |
| `accuracy`           | Accuracy calculation tests                      |
| `early_stopping`     | Early stopping logic tests                      |

### Running Tests

```bash
# All unit tests with coverage
cd src/tests && bash scripts/run_tests.bash

# Specific test file
python -m pytest unit/test_forward_pass.py -v

# By marker
python -m pytest -m "unit and accuracy" -v

# With specific options
bash scripts/run_tests.bash -v -c     # Verbose with coverage
bash scripts/run_tests.bash -j        # Parallel execution
bash scripts/run_tests.bash -f        # Re-run failed only
```

### Test Output

- HTML Coverage: `src/tests/reports/htmlcov/index.html`
- XML Coverage: `src/tests/reports/coverage.xml`
- JUnit XML: `src/tests/reports/junit.xml`

---

## Key Dependencies

### Core Libraries

| Library      | Purpose                               |
| ------------ | ------------------------------------- |
| `torch`      | Neural network tensors and operations |
| `numpy`      | Numerical computations                |
| `matplotlib` | Plotting and visualization            |
| `h5py`       | HDF5 file serialization               |
| `PyYAML`     | YAML configuration parsing            |
| `requests`   | HTTP client for JuniperData REST API  |

### Testing Libraries

| Library      | Purpose            |
| ------------ | ------------------ |
| `pytest`     | Test framework     |
| `pytest-cov` | Coverage reporting |

### Optional Libraries

| Library    | Purpose                           |
| ---------- | --------------------------------- |
| `columnar` | Formatted table output (optional) |

---

## Serialization System

### HDF5 Snapshots

The project uses HDF5 for network serialization.

**Save Network**:

```python
network.save_to_hdf5(
    filepath="./snapshots/network.h5",
    include_training_state=True,
    include_training_data=False
)
```

**Load Network**:

```python
loaded_network = CascadeCorrelationNetwork.load_from_hdf5("./snapshots/network.h5")
```

**CLI Tools**:

```bash
python -m snapshots.snapshot_cli save network.pkl snapshot.h5
python -m snapshots.snapshot_cli load snapshot.h5
python -m snapshots.snapshot_cli verify snapshot.h5
python -m snapshots.snapshot_cli list ./snapshots/
```

### What's Serialized

- Network architecture (input/output sizes, hidden units)
- Trained weights and biases
- Activation functions
- Training history
- Random state (Python, NumPy, PyTorch) for deterministic resume
- UUID for tracking
- Checksums for data integrity

---

## Multiprocessing

### Candidate Training

Candidate units are trained in parallel using Python's multiprocessing:

```python
class CandidateTrainingManager(BaseManager):
    """Custom manager for handling candidate training queues."""
    pass

CandidateTrainingManager.register("get_task_queue", callable=_create_task_queue)
CandidateTrainingManager.register("get_result_queue", callable=_create_result_queue)
```

### Pickling Considerations

Classes implement `__getstate__` and `__setstate__` to handle non-picklable objects (loggers, closures):

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

---

## Configuration

### Constants Organization

Constants are organized hierarchically in `src/cascor_constants/`:

- `constants_model/` - Model architecture defaults
- `constants_candidates/` - Candidate training parameters
- `constants_activation/` - Activation functions
- `constants_logging/` - Logging configuration
- `constants_problem/` - Problem-specific settings
- `constants_hdf5/` - Serialization paths

### Adjusting Log Levels

Log levels are controlled via constants. To change:

```python
# In cascor_constants/constants.py, uncomment the desired level:
# _CASCOR_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_DEBUG
_CASCOR_LOG_LEVEL_NAME = _PROJECT_LOG_LEVEL_NAME_INFO
```

### Network Configuration

Use `CascadeCorrelationConfig` for network setup:

```python
config = CascadeCorrelationConfig(
    input_size=2,
    output_size=2,
    learning_rate=0.01,
    max_hidden_units=50,
    candidate_pool_size=16,
    correlation_threshold=0.001,
    random_seed=42
)
network = CascadeCorrelationNetwork(config=config)
```

---

## Common Patterns

### Creating a Network and Training

```python
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

config = CascadeCorrelationConfig(input_size=2, output_size=2)
network = CascadeCorrelationNetwork(config=config)
network.fit(x_train, y_train, epochs=100)
accuracy = network.get_accuracy(x_test, y_test)
```

### Using the Spiral Problem

```python
from spiral_problem.spiral_problem import SpiralProblem

sp = SpiralProblem(
    _SpiralProblem__n_points=100,
    _SpiralProblem__n_spirals=2,
    _SpiralProblem__noise=0.1
)
sp.evaluate(n_points=100, n_spirals=2, plot=True)
```

### Error Handling

Custom exceptions in `cascade_correlation_exceptions/`:

- `ConfigurationError` - Invalid configuration
- `TrainingError` - Training failures
- `ValidationError` - Input validation failures

---

## Known Issues and Workarounds

### Logger Pickling

Loggers cannot be pickled for multiprocessing. Classes exclude logger from `__getstate__`.

### GPU Support

Tests disable GPU by default. Use `--gpu` flag for GPU tests:

```bash
pytest --gpu
```

### Long-Running Tests

Critical long-running tests (like deterministic training resume) are skipped by default. Use `--run-long` to run them:

```bash
pytest --run-long                    # Run long-running correctness tests
pytest --slow --run-long             # Run both slow and long tests
```

### Random Reproducibility

Set `random_seed` in config for deterministic training:

```python
config = CascadeCorrelationConfig(random_seed=42)
```

---

## Development Workflow

### Adding New Features

1. Create feature in appropriate module
2. Add constants to `src/cascor_constants/`
3. Add tests in `src/tests/unit/` or `src/tests/integration/`
4. Update documentation in `notes/`
5. Run tests: `bash scripts/run_tests.bash`

### Adding New Tests

1. Create test file following `test_<feature>.py` naming
2. Use appropriate markers (`@pytest.mark.unit`, etc.)
3. Use fixtures from `conftest.py`
4. Follow Arrange-Act-Assert pattern

---

## Documentation Files

| File                                                           | Description                           |
| -------------------------------------------------------------- | ------------------------------------- |
| `notes/FEATURES_GUIDE.md`                                      | Feature documentation and usage       |
| `notes/PRE-DEPLOYMENT_ROADMAP-2.md`                            | Pre-deployment roadmap (consolidated) |
| `notes/JUNIPER-CASCOR_POST-RELEASE_DEVELOPMENT-ROADMAP.md`     | Post-release development roadmap      |
| `notes/ARCHITECTURE_GUIDE.md`                                  | Architecture overview                 |
| `notes/API_REFERENCE.md`                                       | API reference documentation           |
| `notes/INTEGRATION_ROADMAP-01.md`                              | Cascor-Canopy integration tracker     |
| `src/tests/README.md`                                          | Test suite documentation              |

Archived documentation (plans, roadmaps, implementation summaries from earlier development phases) is preserved in `notes/history/`.

---

## Performance Considerations

### Serialization Performance

- Save (100 units): < 2 seconds
- Load (100 units): < 3 seconds
- Checksum verification: < 200ms

### Training Tips

- Optimize `candidate_pool_size` for CPU core count
- Use N-best candidate selection for faster convergence
- Tune `patience` for speed vs. accuracy tradeoff

---

## Security Notes

- No secrets or API keys in codebase
- Sensitive files excluded via `.gitignore`
- Log files contain training data - handle appropriately

---

## Worktree Procedures (Mandatory — Task Isolation)

> **OPERATING INSTRUCTION**: All feature, bugfix, and task work SHOULD use git worktrees for isolation. Worktrees keep the main working directory on the default branch while task work proceeds in a separate checkout.

### What This Is

Git worktrees allow multiple branches of a repository to be checked out simultaneously in separate directories. For the Juniper ecosystem, all worktrees are centralized in **`/home/pcalnon/Development/python/Juniper/worktrees/`** using a standardized naming convention.

The full setup and cleanup procedures are defined in:

- **`notes/WORKTREE_SETUP_PROCEDURE.md`** — Creating a worktree for a new task
- **`notes/WORKTREE_CLEANUP_PROCEDURE.md`** — Merging, removing, and pushing after task completion

Read the appropriate file when starting or completing a task.

### Worktree Directory Naming

Format: `<repo-name>--<branch-name>--<YYYYMMDD-HHMM>--<short-hash>`

Example: `juniper-cascor--feature--add-validation--20260225-1430--fb530aa1`

- Slashes in branch names are replaced with `--`
- All worktrees reside in `/home/pcalnon/Development/python/Juniper/worktrees/`

### When to Use Worktrees

| Scenario | Use Worktree? |
| -------- | ------------- |
| Feature development (new feature branch) | **Yes** |
| Bug fix requiring a dedicated branch | **Yes** |
| Quick single-file documentation fix on main | No |
| Exploratory work that may be discarded | **Yes** |
| Hotfix requiring immediate merge | **Yes** |

### Quick Reference

**Setup** (full procedure in `notes/WORKTREE_SETUP_PROCEDURE.md`):

```bash
cd /home/pcalnon/Development/python/Juniper/juniper-cascor
git fetch origin && git checkout main && git pull origin main
BRANCH_NAME="feature/my-task"
git branch "$BRANCH_NAME" main
REPO_NAME=$(basename "$(pwd)")
SAFE_BRANCH=$(echo "$BRANCH_NAME" | sed 's|/|--|g')
WORKTREE_DIR="/home/pcalnon/Development/python/Juniper/worktrees/${REPO_NAME}--${SAFE_BRANCH}--$(date +%Y%m%d-%H%M)--$(git rev-parse --short=8 HEAD)"
git worktree add "$WORKTREE_DIR" "$BRANCH_NAME"
cd "$WORKTREE_DIR"
```

**Cleanup** (full procedure in `notes/WORKTREE_CLEANUP_PROCEDURE.md`):

```bash
cd "$WORKTREE_DIR" && git push origin "$BRANCH_NAME"
cd /home/pcalnon/Development/python/Juniper/juniper-cascor
git checkout main && git pull origin main
git merge "$BRANCH_NAME"
git push origin main
git worktree remove "$WORKTREE_DIR"
git branch -d "$BRANCH_NAME"
git push origin --delete "$BRANCH_NAME"
git worktree prune
```

### Rules

- **Centralized location**: All worktrees go in `/home/pcalnon/Development/python/Juniper/worktrees/`. Never create worktrees inside the repo directory.
- **Clean before you start**: Ensure the main working directory is clean before creating a worktree.
- **Push before you merge**: Always push the working branch to remote before merging (backup).
- **Prune after cleanup**: Run `git worktree prune` after removing a worktree to clean metadata.
- **Do not leave stale worktrees**: Clean up worktrees promptly after merging.

---

## Thread Handoff (Mandatory — Replaces Thread Compaction)

> **CRITICAL OPERATING INSTRUCTION**: Thread handoff MUST be performed instead of thread compaction. This section defines mandatory behavior that overrides Claude Code's default compaction process.

### What This Is

When a long-running conversation approaches context limits, Claude Code normally performs **thread compaction** — summarizing prior context to free token capacity. This introduces information loss. Instead, Claude Code instances working on this project MUST perform a **proactive thread handoff**: transferring a curated, high-signal summary to a fresh thread with full context capacity.

The full handoff protocol is defined in **`notes/THREAD_HANDOFF_PROCEDURE.md`**. Read that file when a handoff is triggered.

### When to Trigger a Handoff

**Automatic trigger (pre-compaction threshold):** Initiate a thread handoff when token utilization reaches **95% to 99%** of the level at which thread compaction would normally be triggered. This means the handoff fires when you are within **1% to 5%** of the compaction threshold, ensuring the handoff completes before compaction would occur.

Concretely:

- If compaction would trigger at N% context utilization, begin handoff at (N − 5)% to (N − 1)%.
- **Self-assessment rule**: At each turn where you are performing multi-step work, assess whether you are approaching the compaction threshold. If you estimate you are within 5% of it, begin the handoff protocol immediately.
- When the system compresses prior messages or you receive a context compression notification, treat this as a signal that handoff should have already occurred — immediately initiate one.

**Additional triggers** (from `notes/THREAD_HANDOFF_PROCEDURE.md`):

| Condition                   | Indicator                                                            |
| --------------------------- | -------------------------------------------------------------------- |
| **Context saturation**      | Thread has performed 15+ tool calls or edited 5+ files               |
| **Phase boundary**          | A logical phase of work is complete                                  |
| **Degraded recall**         | Re-reading a file already read, or re-asking a resolved question     |
| **Multi-module transition** | Moving between major components                                      |
| **User request**            | User says "hand off", "new thread", or similar                       |

**Do NOT handoff** when:

- The task is nearly complete (< 2 remaining steps)
- The current thread is still sharp and producing correct output
- The work is tightly coupled and splitting would lose critical in-flight state

### How to Execute a Handoff

1. **Checkpoint**: Inventory what was done, what remains, what was discovered, and what files are in play
2. **Compose the handoff goal**: Write a concise, actionable summary (see templates in `notes/THREAD_HANDOFF_PROCEDURE.md`)
3. **Present to user**: Output the handoff goal to the user and recommend starting a new thread with that goal as the initial prompt
4. **Include verification commands**: Always specify how the new thread should verify its starting state (test commands, file checks)
5. **State git status**: Mention branch, staged files, and any uncommitted work

### Rules

- **This is not optional.** Every Claude Code instance on this project must follow these rules.
- **Handoff early, not late.** A handoff at 70% context usage is better than compaction at 95%.
- **Do not duplicate CLAUDE.md content** in the handoff goal — the new thread reads CLAUDE.md automatically.
- **Be specific** in the handoff goal: include file paths, decisions made, and test status.

---

## Contact

For questions about this codebase, refer to:

- Project documentation in `notes/`
- Test examples in `src/tests/`
- Constants definitions in `src/cascor_constants/`
