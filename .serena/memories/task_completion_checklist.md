# Task Completion Checklist

When completing a coding task in Juniper Cascor, verify the following:

## Before Committing

### 1. Code Quality Checks

Run pre-commit hooks to validate all code quality:

```bash
pre-commit run --all-files
```

This runs:

- Black (formatting)
- isort (import sorting)
- Flake8 (linting)
- MyPy (type checking)
- Bandit (security scanning)
- YAML/TOML/JSON syntax checks
- Markdown linting
- Shell script linting (shellcheck)

### 2. Run Tests

```bash
cd src/tests && bash scripts/run_tests.bash -v -c
```

Ensure:

- All tests pass
- Coverage meets minimum threshold (80%)
- No new test failures introduced

### 3. Add Tests for New Code

If adding new features:

1. Create test file following `test_<feature>.py` naming
2. Use appropriate markers (`@pytest.mark.unit`, etc.)
3. Use fixtures from `conftest.py`
4. Follow Arrange-Act-Assert pattern

### 4. Update Constants

If adding new configuration:

- Add constants to appropriate file in `src/cascor_constants/`

### 5. Documentation

If significant changes:

- Update relevant docs in `notes/`
- Add docstrings following project format

## Commit Conventions

- Use clear, descriptive commit messages
- Commits should be signed (GPG)
- Include Co-Authored-By for AI assistance

## Quick Validation Commands

```bash
# Format and lint
pre-commit run --all-files

# Run tests
cd src/tests && bash scripts/run_tests.bash

# Type check specific modules
cd src && python -m mypy cascade_correlation/ candidate_unit/ --ignore-missing-imports
```
