# Plan: Fix 4 Test Issues in juniper-cascor

**Date**: 2026-03-04
**Author**: Claude Code (Opus 4.6)

## Context

The juniper-cascor test suite has 4 issues producing spurious skips, misleading warnings, and potential runtime errors. All fixes are low-risk: two are dependency/configuration corrections, and two are log-level/message adjustments. No behavioral logic changes, no tests deleted or disabled.

---

## Issue 1: Misnamed Package (`test_juniper_data_e2e.py`)

**Root cause**: The optional dependency group `juniper-data` in `pyproject.toml` only contains `juniper-data-client`, not the `juniper-data` server package itself. The e2e test imports `from juniper_data.api.app import create_app` (the server), so these 10 tests always skip. Additionally, the `requires_juniper_data` marker is missing from `pyproject.toml` (only defined in `pytest.ini`), which can conflict with `--strict-markers`.

### Steps

1. **Identify**: Traced the skip to `test_juniper_data_e2e.py` line 78-81 where `from juniper_data.api.app import create_app` fails because the `juniper-data` service package is not installed.
2. **Investigate**: Confirmed `juniper_data.api.app` and `juniper_data.api.settings` exist in the juniper-data project at `/home/pcalnon/Development/python/Juniper/juniper-data/juniper_data/api/`. Import paths are correct.
3. **Verify**: The `juniper-data` optional dependency group in `pyproject.toml` (lines 55-57) only lists `juniper-data-client>=0.3.0`, not `juniper-data` itself.
4. **Analyze**: The `requires_juniper_data` marker is defined in `src/tests/pytest.ini` line 47 but absent from `pyproject.toml` markers (lines 134-149), risking `--strict-markers` conflicts.

### Changes

**`pyproject.toml`** (lines 55-57): Add `juniper-data` to the optional dependency group:
```toml
juniper-data = [
    "juniper-data-client>=0.3.0; python_version>='3.12'",
    "juniper-data>=0.4.0; python_version>='3.12'",
]
```

**`pyproject.toml`** (line 149): Add missing marker after `early_stopping`:
```toml
    "requires_juniper_data: Tests that require the juniper-data package to be installed",
```

**`test_juniper_data_e2e.py`** (line 81): Fix legacy PascalCase skip message:
```python
# Before:
pytest.skip("JuniperData package not installed")
# After:
pytest.skip("juniper-data package not installed")
```

### Validation

- Install updated deps: `pip install -e ".[juniper-data]"` in JuniperCascor env
- Confirm e2e tests no longer skip when `juniper-data` is installed
- Confirm the skip message reads "juniper-data package not installed" when it is absent

---

## Issue 2: Missing Generator for Random Seed (`candidate_unit.py`)

**Root cause**: `_initialize_randomness()` intentionally passes `generator=None` for the hash seeder (`self._seed_hash`) at lines 410-414, but `_seed_random_generator()` logs a WARNING at line 447 for this expected code path, producing noise in test output.

### Steps

1. **Identify**: Traced the warning to `candidate_unit.py` line 447 in `_seed_random_generator()`.
2. **Investigate**: Found that the 4th call in `_initialize_randomness()` (lines 410-414) explicitly passes `generator=None` because hash-based seeding does not use a random number generator for sequence rolling.
3. **Verify**: All other 3 calls to `_seed_random_generator()` pass valid generator functions (numpy, random, torch).
4. **Analyze**: The WARNING is inappropriate for intentional/expected behavior. The surrounding code uses `trace` level for operational flow messages.

### Changes

**`candidate_unit.py`** (line 447): Downgrade from `warning` to `trace`:
```python
# Before:
self.logger.warning("CandidateUnit: _seed_random_generator: No generator function provided, skipping random number generation and sequence rolling.")
# After:
self.logger.trace("CandidateUnit: _seed_random_generator: No generator function provided, skipping random number generation and sequence rolling.")
```

### Validation

- Run tests exercising `CandidateUnit._initialize_randomness()` with `CASCOR_LOG_LEVEL=WARNING`
- Confirm no WARNING appears for the hash seeder path
- Run with `CASCOR_LOG_LEVEL=TRACE` and confirm the message appears at TRACE level

---

## Issue 3: Process Training Results Mismatch (`cascade_correlation.py`)

**Root cause**: Lines 1805-1808 compare two semantically different metrics as if they should be equal:
- `success_count`: counts candidates where `success=True`, set as `success=(best_idx >= 0)` in `candidate_unit.py` line 843 — meaning "did training converge and find valid epochs"
- `successful_candidates`: counts candidates where `correlation >= self.correlation_threshold` — meaning "did the candidate achieve sufficient quality"

A candidate can successfully train but still have a correlation below the threshold. This is normal behavior (e.g., 8 candidates train successfully but 0 meet the quality bar), not an error.

### Steps

1. **Identify**: Traced the warning to `cascade_correlation.py` lines 1805-1808.
2. **Investigate**: Read `get_candidates_data_count()` at lines 1898-1911 — it counts results matching a field constraint.
3. **Verify**: Confirmed `success` field defaults to `True` in `CandidateTrainingResult` (line 92) and is set to `(best_idx >= 0)` in `_get_correlations()` (line 843).
4. **Analyze**: The metrics measure fundamentally different things. The comparison is informational, not an error condition.

### Changes

**`cascade_correlation.py`** (lines 1807-1808): Change to `info` with descriptive message:
```python
# Before:
self.logger.warning(f"CascadeCorrelationNetwork: _process_training_results: Mismatch in success counts: success_count: {success_count}, successful_candidates: {successful_candidates}")
# After:
self.logger.info(f"CascadeCorrelationNetwork: _process_training_results: Of {success_count} successfully trained candidates, {successful_candidates} met the correlation threshold ({self.correlation_threshold})")
```

### Validation

- Run candidate training tests and confirm the message appears at INFO level
- Confirm the message wording is clear and descriptive

---

## Issue 4: Grow Network Validation Fails (`cascade_correlation.py`)

**Root cause**: `validate_training_results` is initialized as `None` (line 2774). If the `for` loop breaks early (lines 2781, 2787, 2792) before `validate_training()` runs at line 2835, it stays `None`. The post-loop check at line 2850:
1. Logs "Validation failed" — misleading, since validation never ran
2. References `epoch + 1` — causes `UnboundLocalError` if `max_epochs=0`

### Steps

1. **Identify**: Traced the warning to `cascade_correlation.py` line 2850-2851.
2. **Investigate**: Read the full `grow_network()` method (lines 2740-2863). Found 3 `break` paths before validation.
3. **Verify**: `validate_training_results` remains `None` when loop exits early. `epoch` is undefined if loop never executes.
4. **Analyze**: The warning message is misleading and the `epoch` reference is unsafe.

### Changes

**`cascade_correlation.py`**: Add safe epoch counter after line 2774:
```python
        epochs_completed = 0
```

Add at end of loop body (after line 2849, inside the `for` block):
```python
            epochs_completed = epoch + 1
```

Fix post-loop warning (lines 2850-2851):
```python
# Before:
self.logger.warning(f"CascadeCorrelationNetwork: grow_network: Validation failed at epoch {epoch + 1}/{max_epochs}.")
# After:
self.logger.warning(f"CascadeCorrelationNetwork: grow_network: No validation was performed (training loop exited early or did not execute). Epochs completed: {epochs_completed}/{max_epochs}.")
```

Fix post-loop info message (line 2860):
```python
# Before:
self.logger.info(f"CascadeCorrelationNetwork: grow_network: Finished training after {epoch + 1} epochs. Total hidden units: {len(self.hidden_units)}")
# After:
self.logger.info(f"CascadeCorrelationNetwork: grow_network: Finished training after {epochs_completed} epochs. Total hidden units: {len(self.hidden_units)}")
```

### Validation

- Run network growth tests (`-m network_growth`)
- Confirm no `UnboundLocalError` on early loop exit
- Confirm warning message accurately says "No validation was performed" instead of "Validation failed"

---

## Summary of Files Modified

| File | Issues Addressed |
|------|-----------------|
| `pyproject.toml` | 1 (dependency + marker) |
| `src/tests/integration/test_juniper_data_e2e.py` | 1 (skip message) |
| `src/candidate_unit/candidate_unit.py` | 2 (log level) |
| `src/cascade_correlation/cascade_correlation.py` | 3 (log level + message), 4 (epoch safety + message) |

## Full Validation

After all changes:
```bash
conda activate JuniperCascor
cd /home/pcalnon/Development/python/Juniper/juniper-cascor
pip install -e ".[juniper-data]"
cd src/tests && bash scripts/run_tests.bash -v
```

Confirm:
1. E2e tests run (or skip with correct message if `juniper-data` not installed)
2. No spurious WARNING from candidate seeding
3. Training results message is INFO-level and descriptive
4. No `UnboundLocalError` and accurate warning on early loop exit
5. All tests pass with no remaining test issues
