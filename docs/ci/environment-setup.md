# CI Environment Setup Guide

This guide covers the GitHub Actions CI/CD environment for the Juniper Cascor project, including setup, configuration, and troubleshooting.

## GitHub Actions Environment

### Runner Configuration

| Setting | Value |
|---------|-------|
| Runner | `ubuntu-latest` |
| Python Version | 3.14 |
| Conda Distribution | Miniforge (via mamba) |
| Channels | `conda-forge`, `pytorch`, `nvidia` |

### Workflow Location

The CI/CD pipeline is defined in:

```
.github/workflows/ci.yml
```

### Pipeline Jobs

| Job | Description | Trigger |
|-----|-------------|---------|
| `lint` | Code quality checks (Black, isort, flake8, mypy) | All pushes and PRs |
| `test` | Unit tests with coverage | All pushes and PRs |
| `integration` | Integration tests | PRs only |
| `quality-gate` | Enforces quality thresholds | All pushes and PRs |
| `notify` | Build status notification | All pushes and PRs |

### Branch Triggers

```yaml
on:
  push:
    branches:
      - main
      - develop
      - feature/**
      - fix/**
  pull_request:
    branches:
      - main
      - develop
```

---

## Environment Creation in CI

### Conda Setup with Mamba

The CI uses [setup-miniconda](https://github.com/conda-incubator/setup-miniconda) with Mamba for fast environment creation:

```yaml
- name: Set up Conda
  uses: conda-incubator/setup-miniconda@v3
  with:
    python-version: ${{ matrix.python-version }}
    channels: conda-forge,pytorch,nvidia
    conda-remove-defaults: true
    miniforge-version: latest
    use-mamba: true
    channel-priority: flexible
    activate-environment: JuniperCascor
    environment-file: conf/conda_environment.yaml
    auto-activate-base: false
```

### Key Configuration Options

| Option | Value | Purpose |
|--------|-------|---------|
| `use-mamba: true` | Enables mamba | 5-10x faster dependency resolution |
| `miniforge-version: latest` | Latest Miniforge | Provides mamba by default |
| `channel-priority: flexible` | Flexible resolution | Allows cross-channel package matching |
| `conda-remove-defaults: true` | No defaults channel | Prevents conflicts with conda-forge |
| `environment-file` | `conf/conda_environment.yaml` | Reproducible environment spec |

### Shell Configuration

All steps that use conda must specify the login shell:

```yaml
shell: bash -el {0}
```

This ensures the conda environment is properly activated.

### Caching Considerations

The current workflow does **not** explicitly cache the conda environment due to:

1. **Environment complexity**: Large environments with CUDA packages may exceed cache limits
2. **Cross-platform issues**: Cached packages may not work across runner updates
3. **Mamba speed**: Mamba resolves dependencies quickly enough for most use cases

To add caching (if needed for performance):

```yaml
- name: Cache Conda Environment
  uses: actions/cache@v4
  with:
    path: |
      ~/miniconda3/envs/JuniperCascor
      ~/.cache/pip
    key: ${{ runner.os }}-conda-${{ hashFiles('conf/conda_environment.yaml') }}
    restore-keys: |
      ${{ runner.os }}-conda-
```

### Test Dependencies Installation

After conda environment creation, additional test packages are installed via pip:

```yaml
- name: Install Test Dependencies
  shell: bash -el {0}
  run: |
    pip install pytest pytest-cov pytest-timeout pytest-xdist
```

### Required Directories

The CI creates necessary directories for logs and reports:

```yaml
- name: Create Required Directories
  run: |
    mkdir -p logs src/logs reports/junit src/tests/reports
```

---

## Environment Variables in CI

### CASCOR_LOG_LEVEL

The project supports runtime log level override via `CASCOR_LOG_LEVEL`:

| Value | Use Case |
|-------|----------|
| `WARNING` | Quiet mode for benchmarks/production |
| `INFO` | Default CI logging |
| `DEBUG` | Verbose debugging output |
| `TRACE` | Maximum verbosity |

To set in CI:

```yaml
- name: Run Tests
  shell: bash -el {0}
  env:
    CASCOR_LOG_LEVEL: WARNING
  run: |
    cd src/tests
    python -m pytest unit/ -v
```

### GitHub Actions Context Variables

The workflow uses these GitHub context variables:

| Variable | Purpose | Example |
|----------|---------|---------|
| `${{ matrix.python-version }}` | Python version from matrix | `3.14` |
| `${{ github.ref_name }}` | Current branch name | `main`, `feature/xyz` |
| `${{ github.sha }}` | Commit SHA | `abc123...` |
| `${{ github.actor }}` | User who triggered | `username` |
| `${{ github.workflow }}` | Workflow name | `CI/CD Pipeline` |

### Secrets Configuration

Currently, no secrets are required for basic CI operation. If adding secrets:

```yaml
- name: Deploy Step
  env:
    API_KEY: ${{ secrets.API_KEY }}
  run: |
    # Use $API_KEY in commands
```

Required secrets would be configured in:
**Repository Settings → Secrets and variables → Actions**

---

## Local Reproduction

### Matching Local Environment to CI

To reproduce the CI environment locally:

```bash
# 1. Install Miniforge (provides mamba)
# Linux/macOS:
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh

# 2. Create environment matching CI
mamba env create -f conf/conda_environment.yaml --name JuniperCascor

# 3. Activate environment
conda activate JuniperCascor

# 4. Install test dependencies
pip install pytest pytest-cov pytest-timeout pytest-xdist

# 5. Create required directories
mkdir -p logs src/logs reports/junit src/tests/reports

# 6. Run tests exactly as CI does
cd src/tests
python -m pytest unit/ \
    --verbose \
    --timeout=60 \
    --cov=../cascade_correlation \
    --cov=../candidate_unit \
    --cov-report=term-missing \
    -m "unit and not slow"
```

### Running Linting Checks Locally

```bash
# Install linting tools
pip install black isort mypy flake8 flake8-bugbear flake8-comprehensions flake8-simplify

# Run checks matching CI
black --check --diff src/
isort --check-only --diff src/
flake8 src/ --count --select=B,C,E,F,W,T4,B9 --extend-ignore=E203,E266,E501,W503 --max-line-length=120 --max-complexity=15
mypy src/ --ignore-missing-imports --no-strict-optional
```

### Docker-Based Reproduction

For exact CI environment reproduction, use Docker:

```dockerfile
# Dockerfile.ci
FROM condaforge/miniforge3:latest

WORKDIR /app

# Copy environment file
COPY conf/conda_environment.yaml /app/conf/

# Create environment
RUN mamba env create -f conf/conda_environment.yaml --name JuniperCascor

# Activate environment in shell
SHELL ["conda", "run", "-n", "JuniperCascor", "/bin/bash", "-c"]

# Install test dependencies
RUN pip install pytest pytest-cov pytest-timeout pytest-xdist

# Copy project
COPY . /app/

# Create directories
RUN mkdir -p logs src/logs reports/junit src/tests/reports

# Default command
CMD ["conda", "run", "-n", "JuniperCascor", "bash", "-c", "cd src/tests && python -m pytest unit/ -v"]
```

Build and run:

```bash
# Build the image
docker build -f Dockerfile.ci -t juniper-cascor-ci .

# Run tests
docker run --rm juniper-cascor-ci

# Run with interactive shell
docker run --rm -it juniper-cascor-ci bash
```

### Using Act for Local GitHub Actions

[Act](https://github.com/nektos/act) can run GitHub Actions locally:

```bash
# Install act
brew install act  # macOS
# or: https://github.com/nektos/act#installation

# Run the CI workflow
act push

# Run specific job
act push -j test

# Run with verbose output
act push -v
```

---

## Troubleshooting CI Environment

### Dependency Version Mismatches

**Symptoms**:

- Tests pass locally but fail in CI
- Import errors for specific packages
- Version conflicts in conda resolution

**Solutions**:

1. **Pin exact versions** in `conf/conda_environment.yaml`:

   ```yaml
   dependencies:
     - python=3.14.2
     - torch=2.9.1
     - numpy=2.4.1
   ```

2. **Regenerate environment file** from working local env:

   ```bash
   conda list --explicit > conf/conda_environment.yaml
   ```

3. **Check conda-forge availability**:

   ```bash
   mamba search -c conda-forge package_name
   ```

### Platform-Specific Issues

| Issue | Platform | Solution |
|-------|----------|----------|
| CUDA not available | CI (ubuntu-latest) | CI runners have no GPU; tests use CPU |
| Different random results | CI vs local | Seed random state; set `random_seed` in config |
| Path separators | Windows local vs CI | Use `os.path.join()` or `pathlib.Path` |
| Line endings | Windows commits | Configure `.gitattributes` with `* text=auto` |

### Common CI Failures

**1. Timeout Errors**

```
E   pytest.PytestTimeoutError: Timeout after 60 seconds
```

Solution: Tests marked with `@pytest.mark.slow` are excluded by default. The CI uses:

```yaml
-m "unit and not slow"
```

**2. Coverage Below Threshold**

```
::warning::Coverage below 50% threshold
```

This is currently a soft warning. To make it strict, modify CI:

```yaml
python -m coverage report --fail-under=50
# Remove `|| true` to fail the job
```

**3. Environment Creation Failures**

```
ResolvePackageNotFound: ...
```

Solutions:

- Check package availability on conda-forge
- Use `channel-priority: flexible`
- Consider pip fallback for problematic packages

**4. Disk Space Issues**

```
No space left on device
```

The CI already includes disk cleanup:

```yaml
- name: Free Disk Space
  run: |
    conda clean --all -y
    sudo rm -rf /usr/share/dotnet
    sudo rm -rf /opt/ghc
```

### Debugging CI Runs

1. **Enable verbose output**:

   ```yaml
   python -m pytest unit/ -vvv
   ```

2. **Print environment info**:

   ```yaml
   - name: Debug Environment
     run: |
       conda list
       pip list
       python --version
       which python
       env | grep -E "PYTHON|CONDA|PATH"
   ```

3. **Download artifacts** from failed runs:
   - Go to Actions → Failed Run → Artifacts
   - Download `test-results-*` and `coverage-report-*`

4. **Use workflow_dispatch** for manual testing:

   ```yaml
   on:
     workflow_dispatch:
       inputs:
         debug_enabled:
           description: 'Run with debug'
           required: false
           default: 'false'
   ```

### Network and Rate Limiting

**Package download failures**:

```
CondaHTTPError: HTTP 429 TOO MANY REQUESTS
```

Solutions:

- Add retry logic in CI
- Cache conda packages
- Use GitHub-hosted mirrors if available

---

## CI Configuration Reference

### Test Timeouts

| Test Type | Timeout | Marker |
|-----------|---------|--------|
| Unit tests | 60s | `unit and not slow` |
| Integration tests | 120s | `integration and not slow` |
| Slow tests | 300s | `slow` (excluded by default) |

### Coverage Thresholds

| Metric | Current Threshold | Target |
|--------|-------------------|--------|
| Overall | 50% (soft) | 80% |
| Branch | Not enforced | 70% |

### Artifacts Retention

| Artifact | Retention |
|----------|-----------|
| Coverage reports | 30 days |
| Test results (JUnit XML) | 30 days |

---

## Related Documentation

- [Environment Setup Guide](../install/environment-setup.md) - Local development setup
- [Testing Guide](../testing/manual.md) - Running tests locally
- [AGENTS.md](../../AGENTS.md) - Project conventions and commands
