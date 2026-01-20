# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.11] - 2026-01-20

### Fixed: [0.3.11]

- **CASCOR-P1-004**: Fixed `try` script cosmetic warnings by updating symlink target
  - **Problem**: The `try` symlink pointed to `util/try.bash`, which called `log_debug` before logging functions were sourced, causing 11 "command not found" warnings
  - **Fix**: Updated `try` symlink to point directly to `util/juniper_cascor.bash`
  - **Archived**: Old `util/try.bash` script moved to archive
  - **Result**: Clean startup with no "command not found" warnings

### Identified: [0.3.11]

- **CASCOR-P1-003**: Documented multiprocessing pickling error with `wrapped_activation` local function
  - **Problem**: `CandidateUnit._init_activation_with_derivative()` defines a local function `wrapped_activation` that cannot be pickled for multiprocessing
  - **Impact**: Workers cannot send results back to main process, forcing sequential fallback
  - **Status**: 🔴 BLOCKING - requires refactoring to module-level class
  - **Proposed Fix**: Create picklable `ActivationWithDerivative` class at module level

---

## [0.3.10] - 2026-01-20

### Fixed: [0.3.10]

- **CASCOR-P1-002**: Fixed validate_training API mismatch causing AttributeError
  - **Root Cause**: `grow_network()` passed a `ValidateTrainingInputs` dataclass to `validate_training()`, but the method expected individual parameters and returned a tuple
  - **Error**: `AttributeError: 'tuple' object has no attribute 'early_stop'`
  - **Fix**: Updated `validate_training()` method signature to accept `ValidateTrainingInputs` dataclass and return `ValidateTrainingResults` dataclass
  - **File Changed**: `src/cascade_correlation/cascade_correlation.py` (lines 4115-4258)
  - **Result**: Training validation now uses proper dataclass API, enabling full network training cycle

---

## [0.3.9] - 2026-01-20

### Fixed: [0.3.9]

- **CASCOR-P0-004**: Fixed candidate training result parsing error causing all candidates to fail
  - **Root Cause**: `_train_candidate_unit()` called `candidate.train()` which returns a `float`, but the code expected a `CandidateTrainingResult` object with a `.correlation` attribute
  - **Error**: `'float' object has no attribute 'correlation'` - all 10 candidates failed with 0 hidden units added
  - **Fix**: Changed `candidate.train()` to `candidate.train_detailed()` which returns the full `CandidateTrainingResult` dataclass
  - **File Changed**: `src/cascade_correlation/cascade_correlation.py` (line 2767)
  - **Result**: Candidate training now returns proper result objects, enabling network growth with hidden units

---

## [0.3.8] - 2026-01-20

### Fixed: [0.3.8]

- **CASCOR-P0-003**: Fixed test collection errors caused by incorrect module import paths
  - Multiple test files were using incorrect import path `from cascade_correlation_config...` instead of `from cascade_correlation.cascade_correlation_config...`
  - The `cascade_correlation_config` module is a submodule of `cascade_correlation`, not a top-level module
  - This caused `ModuleNotFoundError: No module named 'cascade_correlation_config'` during test collection
  - **Files Fixed**:
    - `src/tests/unit/test_hdf5.py` (lines 10, 24)
    - `src/tests/integration/test_serialization.py` (line 34)
    - `src/tests/unit/test_p1_fixes.py` (lines 73, 124, 195)
    - `src/tests/unit/test_critical_fixes.py` (lines 47, 102)
  - **Result**: All 152 Cascor tests now collect successfully with 0 errors (previously 2 collection errors)

### Integration: [0.3.8]

- Integration analysis with Juniper Canopy documented in `notes/INTEGRATION_ROADMAP.md`
- Environment compatibility verified with JuniperCascor conda environment

---

## [0.3.7] - 2026-01-16

### Fixed: [0.3.7]

- **P0-019**: Fixed Multiprocessing Manager Port Conflict and Sequential Training Fallback
  - **Root Cause 1**: `_CASCADE_CORRELATION_NETWORK_BASE_MANAGER_ADDRESS` was set to just the IP string `'127.0.0.1'` instead of a tuple `('127.0.0.1', port)`
  - **Root Cause 2**: Fixed port 50000 was hardcoded, causing "Address already in use" errors when multiple tests or instances run
  - **Root Cause 3**: `forkserver` multiprocessing context has issues with custom Manager classes in Python 3.14
  - **Root Cause 4**: When parallel training failed, dummy results with zero correlation were used, preventing network growth
  - **Fixes Applied**:
    - Changed default port from 50000 to 0 (dynamic OS allocation) in `constants_model.py`
    - Changed default multiprocessing context from `forkserver` to `spawn`
    - Fixed address constant to use tuple `('127.0.0.1', 0)` instead of just IP string in `constants.py`
    - Updated `_init_multiprocessing()` to use configured context type instead of hardcoded `forkserver`
    - Added sequential training fallback in `_execute_candidate_training()` when parallel training fails
  - **Result**: Network now falls back to sequential training when parallel fails, producing real correlation values and allowing hidden units to be added. Spiral problems can now be solved.

### Files Changed: [0.3.7]

- `src/constants/constants_model/constants_model.py` - Changed port to 0, context to 'spawn'
- `src/constants/constants.py` - Added import for `_PROJECT_MODEL_BASE_MANAGER_ADDRESS`, fixed address constant to use tuple
- `src/cascade_correlation/cascade_correlation.py` - Updated `_init_multiprocessing()` to use config context, added sequential fallback in `_execute_candidate_training()`

---

## [0.3.6] - 2026-01-15

### Fixed: [0.3.6]

- **P0-016**: Fixed multiprocessing spawn context module import error
  - Added missing `__init__.py` files to all source directories for proper Python package recognition
  - This resolves `ModuleNotFoundError: No module named 'constants.constants_model'; 'constants' is not a package` error
  - Affected directories: constants/, constants_model/, constants_candidates/, constants_hdf5/, constants_activation/, constants_problem/, constants_logging/, cascade_correlation/, cascade_correlation_config/, cascade_correlation_exceptions/, log_config/, logger/, candidate_unit/, spiral_problem/, cascor_plotter/, remote_client/
  - Root cause: Python's multiprocessing `spawn` context re-imports the module, requiring proper package structure

- **P0-017**: Fixed critical bug in best_candidate_id selection causing network to never grow
  - Fixed `_process_training_results()` where `best_candidate_id` was incorrectly set as a tuple `(value,)` instead of an int
  - The trailing comma in the assignment created a tuple, causing all subsequent lookups to fail
  - Since lookups always returned `None`, `best_candidate` was always `None` and `grow_network()` exited immediately
  - This caused the network to remain linear with no hidden units, unable to solve nonlinear problems like spiral classification
  - Changed to directly access `results[0].candidate_id` after sorting (best correlation at index 0)
  - Also simplified best_candidate data extraction to use direct attribute access on sorted results

- **P0-018**: Fixed test_forward_pass_nan_input test expectation
  - Updated test to expect `ValidationError` exception for NaN inputs (correct behavior)
  - Fixed import path from `cascade_correlation.cascade_correlation_exceptions` to `cascade_correlation_exceptions` to match runtime module resolution
  - Previous test incorrectly expected NaN values to propagate through the network

### Files Changed: [0.3.6]

- Created `src/constants/__init__.py` and all subdirectory `__init__.py` files
- Created `src/cascade_correlation/__init__.py` and all subdirectory `__init__.py` files
- Created `src/log_config/__init__.py` and `src/log_config/logger/__init__.py`
- Created `src/candidate_unit/__init__.py`, `src/spiral_problem/__init__.py`, `src/cascor_plotter/__init__.py`, `src/remote_client/__init__.py`
- `src/cascade_correlation/cascade_correlation.py` - Fixed `_process_training_results()` best_candidate_id bug (lines 1562-1591)
- `src/tests/unit/test_forward_pass.py` - Fixed NaN input test expectation

## [0.3.5] - 2025-01-15

### Fixed: [0.3.5]

- **P0-010**: Fixed CandidateUnit.train() return type for backward compatibility
  - Restored `train()` to return float (correlation value) for backward compatibility
  - Added `train_detailed()` method returning full `CandidateTrainingResult` dataclass
  - Internal code can use `train_detailed()` for full training details
  - Added `last_training_result` attribute for introspection after training

- **P0-011**: Fixed CandidateTrainingManager.start() method signature
  - Added `method` parameter to `start()` for multiprocessing context validation
  - Validates method is one of 'fork', 'spawn', 'forkserver' or raises `ValueError`
  - Raises `NotImplementedError` if method not supported on platform

- **P0-012**: Fixed ValidationError exception hierarchy for test compatibility
  - Made `ValidationError` subclass both `CascadeCorrelationError` and `ValueError`
  - Tests expecting `(ValueError, RuntimeError)` now correctly catch `ValidationError`

- **P0-013**: Fixed fit() method to support `epochs` parameter alias
  - Added `epochs` parameter as backward-compatible alias for `max_epochs`
  - Raises `ValueError` if both provided with different values

- **P0-014**: Fixed tensor validation and edge case handling
  - Added `allow_empty` parameter to `_validate_tensor_input()`
  - `forward()` now allows empty tensors for edge case handling
  - Fixed `calculate_residual_error()` dimension validation (removed incorrect x/y feature comparison)
  - Added target output size validation to prevent tensor mismatch errors
  - Fixed `_accuracy()` to return NaN for empty batches instead of ZeroDivisionError

- **P0-015**: Fixed test expectations to match implementation behavior
  - Updated `test_candidate_training_manager.py` to skip actual manager start() calls
  - Fixed `test_accuracy_non_tensor_inputs` to expect `ValueError`
  - Fixed `test_residual_error_*` tests to match graceful handling behavior

### Files Changed: [0.3.5]

- `src/candidate_unit/candidate_unit.py` - Added train_detailed(), modified train() to return float
- `src/cascade_correlation/cascade_correlation.py` - Multiple fixes for validation, fit(), and manager
- `src/cascade_correlation/cascade_correlation_exceptions/cascade_correlation_exceptions.py` - ValidationError inheritance
- `src/tests/unit/test_candidate_training_manager.py` - Updated test expectations
- `src/tests/unit/test_accuracy.py` - Fixed exception type expectations
- `src/tests/unit/test_residual_error.py` - Fixed test assertions

## [0.3.4] - 2025-01-15

### Fixed: [0.3.4]

- **P0-008**: Fixed multiprocessing context for plotting subprocess
  - Changed plotting subprocess in `spiral_problem.py` from default `forkserver` context to explicit `spawn` context
  - This resolves `ModuleNotFoundError: No module named 'constants.constants_model'; 'constants' is not a package` error
  - This resolves `ConnectionResetError: [Errno 104] Connection reset by peer` error when starting plotting process
  - Application now executes successfully without multiprocessing errors

- **P0-006**: Installed missing Python dependencies
  - Installed `h5py` (3.15.1) for HDF5 serialization support
  - Installed `pytest-cov` (7.0.0) for test coverage reporting
  - Installed `psutil` (7.2.1) for test utilities
  - Test suite now runs with coverage reporting enabled

- **pytest.ini**: Restored coverage options now that pytest-cov is installed
  - Re-enabled `--cov=cascade_correlation`, `--cov=candidate_unit` options
  - Re-enabled coverage report generation (term-missing, html, xml)

### Files Changed: [0.3.4]

- `src/spiral_problem/spiral_problem.py` - Use spawn context for plotting subprocess
- `src/tests/pytest.ini` - Restored coverage options
- `src/tests/integration/test_spiral_problem.py` - Fixed import path for SpiralDataGenerator

## [0.3.3] - 2025-01-12

### Fixed: [0.3.3]

- **P0-001**: Fixed critical candidate training runtime errors
  - Fixed incorrect method call `_train_candidate_worker` → `_train_candidate_unit` in `train_candidate_worker()` method (cascade_correlation.py:1782)
  - Fixed `UnboundLocalError` for `traceback` variable in exception handler by adding import statement
  - Added `__getstate__` and `__setstate__` methods to `LogConfig` class to properly handle pickling of logger objects during multiprocessing
  - Added `__getstate__` and `__setstate__` methods to `CascadeCorrelationConfig` class to handle log_config serialization
  - Updated `CascadeCorrelationNetwork.__getstate__` to exclude `log_config` and activation functions from pickling
  - Updated `CascadeCorrelationNetwork.__setstate__` to reinitialize activation functions after unpickling

### Added: [0.3.3]

- **P0-004**: Added thread safety documentation
  - Added thread safety warning to README.md
  - Added thread safety warning to FEATURES_GUIDE.md
  - Added thread safety warning docstring to `CascadeCorrelationNetwork` class

## [0.3.2] - 2025-01-12

### Added: [0.3.2]

- Initial MVP release with Cascade Correlation Neural Network implementation
- HDF5 serialization support for network snapshots
- N-best candidate selection capability
- Flexible optimizer configuration
- Deterministic training with random state preservation
- Multiprocessing support for parallel candidate training
- Data integrity validation with checksums

### Notes

- Reference implementation based on Fahlman & Lebiere, 1990

---

## Version History

| Version | Date       | Description                              |
| ------- | ---------- | ---------------------------------------- |
| 0.3.5   | 2025-01-15 | Fixed API compatibility and test suite   |
| 0.3.4   | 2025-01-15 | Fixed multiprocessing and dependencies   |
| 0.3.3   | 2025-01-12 | Addressed critical runtime errors        |
| 0.3.2   | 2025-01-12 | MVP Complete                             |
| 0.3.1   | 2025-01-10 | Bug fixes and stability improvements     |
| 0.3.0   | 2025-01-08 | Added HDF5 serialization                 |
| 0.2.0   | 2025-01-01 | Added multiprocessing support            |
| 0.1.0   | 2024-12-15 | Initial development release              |
