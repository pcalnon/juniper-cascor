# Development Environment Setup

**Version 0.3.19** | Juniper Cascor Project

This guide covers setting up a complete development environment for contributing to Juniper Cascor, including code quality tools, IDE configuration, profiling, and debugging.

---

## Development Tools

### Code Formatter: Black

Black is the project's code formatter with the following configuration in `pyproject.toml`:

```toml
[tool.black]
line-length = 120
target-version = ["py311", "py312", "py313", "py314"]
include = '\.pyi?$'
```

**Installation:**

```bash
pip install black
```

**Usage:**

```bash
# Format all files
cd src && python -m black .

# Check without modifying
cd src && python -m black --check --diff .

# Format a specific file
python -m black path/to/file.py
```

### Import Sorter: isort

isort organizes imports using the Black-compatible profile:

```toml
[tool.isort]
profile = "black"
line_length = 120
known_first_party = ["cascade_correlation", "candidate_unit", "spiral_problem", "snapshots", "log_config", "constants", "utils"]
sections = ["FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"]
```

**Installation:**

```bash
pip install isort
```

**Usage:**

```bash
# Sort imports in all files
cd src && python -m isort .

# Check without modifying
cd src && python -m isort --check-only --diff .

# Sort a specific file
python -m isort path/to/file.py
```

### Linter: Flake8

Flake8 provides static code analysis:

**Installation:**

```bash
pip install flake8
```

**Usage:**

```bash
cd src && python -m flake8 . --max-line-length=120 --extend-ignore=E203,E266,E501,W503
```

**Ignored Rules:**

| Code | Description |
|------|-------------|
| E203 | Whitespace before ':' (conflicts with Black) |
| E266 | Too many leading '#' for block comment |
| E501 | Line too long (handled by Black) |
| W503 | Line break before binary operator |

### Type Checker: MyPy

MyPy is configured with permissive settings for gradual typing:

```toml
[tool.mypy]
python_version = "3.14"
warn_return_any = false
warn_unused_configs = true
ignore_missing_imports = true
no_strict_optional = true
```

**Installation:**

```bash
pip install mypy
```

**Usage:**

```bash
cd src && python -m mypy cascade_correlation/ candidate_unit/ --ignore-missing-imports
```

### Trunk Check (Meta-Linter)

The project includes Trunk for unified linting. Configuration is in `.trunk/trunk.yaml`:

```yaml
lint:
  enabled:
    - bandit@1.9.2
    - black@25.12.0
    - isort@7.0.0
    - markdownlint@0.47.0
    - ruff@0.14.11
    - shellcheck@0.11.0
    - shfmt@3.6.0
    - yamllint@1.38.0
```

**Installation:**

```bash
# macOS
brew install trunk-io

# Linux/Windows
curl https://get.trunk.io -fsSL | bash
```

**Usage:**

```bash
# Check all files
trunk check

# Check specific file
trunk check path/to/file.py

# Auto-fix issues
trunk fmt
```

---

## IDE Configuration

### VS Code

**Recommended Settings** (`.vscode/settings.json`):

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.conda/bin/python",
    "python.formatting.provider": "none",
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit"
        }
    },
    "black-formatter.args": ["--line-length", "120"],
    "isort.args": ["--profile", "black", "--line-length", "120"],
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Args": [
        "--max-line-length=120",
        "--extend-ignore=E203,E266,E501,W503"
    ],
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.diagnosticMode": "workspace",
    "editor.rulers": [120],
    "files.trimTrailingWhitespace": true,
    "files.insertFinalNewline": true
}
```

**Recommended Extensions:**

| Extension | ID | Purpose |
|-----------|-----|---------|
| Python | `ms-python.python` | Core Python support |
| Pylance | `ms-python.vscode-pylance` | Type checking & IntelliSense |
| Black Formatter | `ms-python.black-formatter` | Code formatting |
| isort | `ms-python.isort` | Import sorting |
| Python Debugger | `ms-python.debugpy` | Debugging support |
| Trunk Check | `trunk.io` | Unified linting |
| GitLens | `eamodio.gitlens` | Git integration |

**Launch Configuration** (`.vscode/launch.json`):

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Run Main",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/src/main.py",
            "cwd": "${workspaceFolder}/src",
            "console": "integratedTerminal",
            "justMyCode": false
        },
        {
            "name": "Run Main with Profiling",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/src/main.py",
            "args": ["--profile"],
            "cwd": "${workspaceFolder}/src",
            "console": "integratedTerminal"
        },
        {
            "name": "Run Tests",
            "type": "debugpy",
            "request": "launch",
            "module": "pytest",
            "args": ["-v", "tests/"],
            "cwd": "${workspaceFolder}/src",
            "console": "integratedTerminal"
        },
        {
            "name": "Debug Current Test File",
            "type": "debugpy",
            "request": "launch",
            "module": "pytest",
            "args": ["-v", "${file}"],
            "cwd": "${workspaceFolder}/src",
            "console": "integratedTerminal"
        }
    ]
}
```

### PyCharm

**Project Settings:**

1. **Python Interpreter**: Set to the conda environment
   - `Settings > Project > Python Interpreter`
   - Select `juniper_cascor` conda environment

2. **Code Style**:
   - `Settings > Editor > Code Style > Python`
   - Set line length to `120`
   - Enable "Optimize imports on the fly"

3. **Black Integration**:
   - `Settings > Tools > Black`
   - Enable "On save"
   - Args: `--line-length 120`

4. **isort Integration**:
   - `Settings > Tools > External Tools`
   - Create new tool:
     - Program: `isort`
     - Arguments: `--profile black $FilePath$`
     - Working directory: `$ProjectFileDir$`

5. **Type Checking**:
   - `Settings > Editor > Inspections > Python`
   - Enable "Type checker compatible with mypy"

**Run Configurations:**

| Name | Script | Parameters | Working Directory |
|------|--------|------------|-------------------|
| Main | `src/main.py` | | `src/` |
| Main (Profile) | `src/main.py` | `--profile` | `src/` |
| Tests | `pytest` | `-v` | `src/tests/` |

---

## Pre-commit Hooks

### Setup

**Installation:**

```bash
pip install pre-commit
```

**Configuration** (`.pre-commit-config.yaml`):

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.10.0
    hooks:
      - id: black
        args: [--line-length=120]

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: [--profile=black, --line-length=120]

  - repo: https://github.com/pycqa/flake8
    rev: 7.1.1
    hooks:
      - id: flake8
        args: [--max-line-length=120, --extend-ignore=E203,E266,E501,W503]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

**Install Hooks:**

```bash
pre-commit install
```

### Running Checks

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run

# Run specific hook
pre-commit run black --all-files

# Update hooks to latest versions
pre-commit autoupdate
```

---

## Profiling Tools

### py-spy (Sampling Profiler)

py-spy is a sampling profiler for Python that generates flame graphs.

**Installation:**

```bash
pip install py-spy

# System-wide (may require sudo for some features)
sudo pip install py-spy
```

**Usage:**

```bash
# Basic profiling
./util/profile_training.bash

# Generate SVG flame graph
./util/profile_training.bash --svg

# Generate Speedscope-compatible JSON
./util/profile_training.bash --speedscope

# Custom duration and rate
./util/profile_training.bash --duration 60 --rate 200

# Include native C/C++ frames
./util/profile_training.bash --native

# Profile with subprocesses
./util/profile_training.bash --subprocesses
```

**Output:**

- SVG files: Open in any web browser
- Speedscope JSON: Upload to [speedscope.app](https://www.speedscope.app/)
- Profiles saved to: `profiles/`

### cProfile (Deterministic Profiler)

Built-in deterministic profiling via the `--profile` flag.

**Usage:**

```bash
cd src && python main.py --profile

# Custom output directory
python main.py --profile --profile-output ./my_profiles

# Show more functions
python main.py --profile --profile-top-n 50
```

**Output:**

- Console output with top functions by cumulative time
- Profile files saved to specified output directory
- Compatible with `pstats` for further analysis

**Analyzing Profile Data:**

```python
import pstats

stats = pstats.Stats('profiles/main_training.prof')
stats.sort_stats('cumulative')
stats.print_stats(30)
```

### tracemalloc (Memory Profiler)

Memory allocation tracking via the `--profile-memory` flag.

**Usage:**

```bash
cd src && python main.py --profile-memory

# Show more allocations
python main.py --profile-memory --profile-top-n 50
```

**Output:**

- Memory summary (peak usage, current usage)
- Top memory allocations by size
- Memory diff between start and end

---

## Debugging Setup

### Python Debugger Configuration

**Using pdb (command-line):**

```python
import pdb; pdb.set_trace()
```

**Using breakpoint() (Python 3.7+):**

```python
breakpoint()
```

**Environment variable control:**

```bash
# Disable all breakpoints
export PYTHONBREAKPOINT=0

# Use a different debugger
export PYTHONBREAKPOINT=ipdb.set_trace
```

### VS Code Debugging

1. Set breakpoints by clicking in the gutter
2. Use the Debug panel (Ctrl+Shift+D)
3. Select a launch configuration
4. Press F5 to start debugging

**Debug Controls:**

| Key | Action |
|-----|--------|
| F5 | Continue |
| F10 | Step Over |
| F11 | Step Into |
| Shift+F11 | Step Out |
| Ctrl+Shift+F5 | Restart |
| Shift+F5 | Stop |

### Remote Debugging

**VS Code Remote Debugging:**

1. Install `debugpy` on the remote machine:

   ```bash
   pip install debugpy
   ```

2. Start the script with debugpy:

   ```bash
   python -m debugpy --listen 5678 --wait-for-client src/main.py
   ```

3. Add remote attach configuration to `.vscode/launch.json`:

   ```json
   {
       "name": "Attach to Remote",
       "type": "debugpy",
       "request": "attach",
       "connect": {
           "host": "remote-hostname",
           "port": 5678
       },
       "pathMappings": [
           {
               "localRoot": "${workspaceFolder}",
               "remoteRoot": "/path/on/remote"
           }
       ]
   }
   ```

### Logging for Debugging

The project uses a custom Logger with extended log levels.

**Log Levels:**

| Level | Value | Usage |
|-------|-------|-------|
| TRACE | 5 | Maximum verbosity, function entry/exit |
| VERBOSE | 8 | Detailed debugging |
| DEBUG | 10 | Standard debug information |
| INFO | 20 | General information |
| WARNING | 30 | Warning conditions |
| ERROR | 40 | Error conditions |
| CRITICAL | 50 | Critical errors |
| FATAL | 60 | Fatal errors |

**Runtime Log Level Override:**

```bash
# Set log level via environment variable
export CASCOR_LOG_LEVEL=DEBUG
python src/main.py

# Quiet mode for benchmarking
export CASCOR_LOG_LEVEL=WARNING
python src/main.py
```

**In-Code Logging:**

```python
from log_config.logger.logger import Logger

Logger.trace("Entering function with args: %s", args)
Logger.debug("Processing item: %s", item)
Logger.info("Training epoch %d complete", epoch)
Logger.warning("Convergence slow, consider adjusting learning rate")
Logger.error("Failed to load checkpoint: %s", e)
```

**Configuration File:**

Logging is configured in `conf/logging_config.yaml`. Adjust handler levels to control verbosity:

```yaml
handlers:
  handler_console:
    level: DEBUG  # Change to TRACE, INFO, WARNING as needed
  handler_file:
    level: DEBUG
```

---

## Quick Reference

### All-in-One Check

```bash
# Format, sort imports, and check linting
cd src
python -m black .
python -m isort .
python -m flake8 . --max-line-length=120 --extend-ignore=E203,E266,E501,W503
python -m mypy cascade_correlation/ candidate_unit/ --ignore-missing-imports
```

### Or Use Trunk

```bash
trunk check
trunk fmt
```

### Run Tests with Coverage

```bash
cd src/tests && bash scripts/run_tests.bash -v -c
```

### Profile Training

```bash
# CPU profiling
cd src && python main.py --profile

# Memory profiling
cd src && python main.py --profile-memory

# Sampling profiler
./util/profile_training.bash --svg
```
