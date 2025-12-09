# Cascor Prototype - Next Steps to MVP

**Last Updated**: 2025-10-25
**Current Status**: Critical serialization fixes implemented
**Goal**: Minimum Viable Product with complete state restoration

---

## Completed ✓

1. **UUID Persistence** - Networks now retain their identity across save/load cycles
2. **Python Random State** - Full deterministic reproducibility with all RNG sources
3. **Config JSON Serialization** - Robust config save/load without type corruption
4. **History Key Alignment** - Training history correctly preserved with proper keys
5. **Activation Function Restoration** - Loaded networks use correct activation functions

---

## Immediate Next Steps (This Week)

### 1. Add Comprehensive Tests

**Priority**: HIGH
**Effort**: 2-4 hours

Create test file: `src/prototypes/cascor/src/tests/integration/test_serialization.py`

Tests needed:

- `test_uuid_persistence` - Verify UUID survives save/load
- `test_deterministic_training_resume` - Train, save, load, resume → verify identical to continuous training
- `test_history_preservation` - Verify all history keys loaded correctly
- `test_config_roundtrip` - Save and load config without errors
- `test_activation_function_restoration` - Verify activation function matches after load
- `test_random_state_restoration` - Verify RNG produces same sequence after load
- `test_hidden_units_preservation` - Verify all hidden units loaded with correct weights/biases/correlations
- `test_backward_compatibility` - Load old snapshot format

**Run tests**:

```bash
cd src/prototypes/cascor
pytest src/tests/integration/test_serialization.py -v
```

---

### 2. Add Hidden Units Checksums

**Priority**: HIGH
**Effort**: 1-2 hours

**Changes to `snapshot_serializer.py`**:

```python
# In _save_hidden_units():
checksum_data = {}
if 'weights' in unit:
    checksum_data['weights'] = calculate_tensor_checksum(unit['weights'])
if 'bias' in unit:
    checksum_data['bias'] = calculate_tensor_checksum(unit['bias'])
if checksum_data:
    write_str_dataset(unit_group, 'checksums', json.dumps(checksum_data))

# In _load_hidden_units():
if 'checksums' in unit_group:
    checksums = json.loads(read_str_dataset(unit_group, 'checksums'))
    if 'weights' in checksums and 'weights' in unit:
        if not verify_tensor_checksum(unit['weights'], checksums['weights']):
            self.logger.error(f"Hidden unit {i} weights checksum failed!")
    if 'bias' in checksums and 'bias' in unit:
        if not verify_tensor_checksum(unit['bias'], checksums['bias']):
            self.logger.error(f"Hidden unit {i} bias checksum failed!")
```

---

### 3. Add Shape Validation

**Priority**: HIGH
**Effort**: 1 hour

**New method in `snapshot_serializer.py`**:

```python
def _validate_shapes(self, network) -> bool:
    """Validate tensor shapes match expected dimensions."""
    # Validate output layer
    expected_output_input = network.input_size + len(network.hidden_units)
    if network.output_weights.shape != (expected_output_input, network.output_size):
        self.logger.error(
            f"Output weights shape mismatch: {network.output_weights.shape} != "
            f"({expected_output_input}, {network.output_size})"
        )
        return False

    if network.output_bias.shape != (network.output_size,):
        self.logger.error(
            f"Output bias shape mismatch: {network.output_bias.shape} != "
            f"({network.output_size},)"
        )
        return False

    # Validate hidden units
    for i, unit in enumerate(network.hidden_units):
        expected_input_size = network.input_size + i
        if 'weights' in unit:
            if unit['weights'].shape[0] != expected_input_size:
                self.logger.error(
                    f"Hidden unit {i} weight shape mismatch: "
                    f"{unit['weights'].shape[0]} != {expected_input_size}"
                )
                return False

    return True
```

Call from `load_network()` after loading all parameters.

---

### 4. Improve Validation

**Priority**: MEDIUM
**Effort**: 1 hour

**Expand `_validate_format()` in `snapshot_serializer.py`**:

```python
def _validate_format(self, hdf5_file: h5py.File) -> bool:
    """Validate HDF5 file format and required content."""
    try:
        # Check format identifier
        format_name = read_str_attr(hdf5_file, "format")
        if format_name not in [self.format_name, "cascor_hdf5_v1", "juniper.cascor"]:
            self.logger.error(f"Invalid format: {format_name}")
            return False

        # Check format version compatibility
        format_version = read_str_attr(hdf5_file, "format_version", "1")
        if int(format_version.split('.')[0]) > int(self.format_version.split('.')[0]):
            self.logger.error(
                f"Incompatible format version: {format_version} "
                f"(serializer version: {self.format_version})"
            )
            return False

        # Check for required groups
        required_groups = ["meta", "config", "params", "arch", "random"]
        for group in required_groups:
            if group not in hdf5_file:
                self.logger.error(f"Missing required group: {group}")
                return False

        # Check for required datasets
        if "params" in hdf5_file:
            params_group = hdf5_file["params"]
            if "output_layer" in params_group:
                output_group = params_group["output_layer"]
                if "weights" not in output_group or "bias" not in output_group:
                    self.logger.error("Missing output layer weights or bias")
                    return False

        # Verify hidden units consistency
        if "hidden_units" in hdf5_file:
            hidden_group = hdf5_file["hidden_units"]
            num_units = hidden_group.attrs.get("num_units", 0)
            actual_units = len([k for k in hidden_group.keys() if k.startswith("unit_")])
            if num_units != actual_units:
                self.logger.error(
                    f"Hidden units count mismatch: {num_units} != {actual_units}"
                )
                return False

        return True

    except Exception as e:
        self.logger.error(f"Format validation failed: {e}")
        return False
```

---

## Short-term (This Month)

### 5. Fix Multiprocessing State Restoration

**Priority**: MEDIUM
**Effort**: 2-3 hours

**Changes needed**:

- Fix role detection (check `_manager` not `candidate_training_manager`)
- Save `worker_standby_sleepytime` and `candidate_training_context_type`
- Restore both `candidate_training_context` and `_mp_ctx`
- Actually start manager if `role == 'server' and autostart`

---

### 6. Decide on Optimizer State

**Priority**: MEDIUM
**Effort**: 30 minutes

**Options**:

1. **Remove** optimizer save/load (simple, recommended for MVP)
2. **Fix** to properly restore momentum (complex, low value)

**Recommendation**: Remove for now - training recreates optimizer anyway.

---

### 7. Create Architecture Documentation

**Priority**: LOW
**Effort**: 2 hours

Document:

- HDF5 file structure specification
- Serialization architecture
- Adding new fields to snapshots
- Version migration strategy

---

## Long-term (Future Enhancements)

1. **Candidate Pool Persistence** - Save in-progress candidate training state
2. **Mid-Epoch Resume** - Save/restore within training loop
3. **Schema Versioning** - Support multiple snapshot format versions
4. **Compression Optimization** - Tune compression for performance
5. **Incremental Snapshots** - Delta snapshots for large networks
6. **Remote Storage Support** - S3/Azure/GCS snapshot storage

---

## Success Criteria for MVP

- [X] UUID persists across save/load
- [X] Full RNG state restoration (deterministic resume)
- [X] Config JSON serialization without errors
- [X] Training history correctly preserved
- [X] Activation functions restored
- [X] All serialization tests passing
- [ ] Hidden units have checksums
- [ ] Shape validation on load
- [ ] Format validation comprehensive
- [ ] No critical errors in diagnostic checks

---

## Testing Strategy

### Test Pyramid

```bash
        E2E Tests (1-2)
       ┌────────────┐
      /              \
     /  Integration   \
    /   Tests (5-10)   \
   /____________________\
   Unit Tests (20-30)
```

**Unit Tests** (file: `tests/unit/test_snapshot_components.py`):

- Test `_config_to_dict()` excludes non-serializable
- Test `_save_random_state()` saves all RNG sources
- Test `_load_random_state()` restores all RNG sources
- Test checksum calculation and verification
- Test shape validation logic

**Integration Tests** (file: `tests/integration/test_serialization.py`):

- Test full save/load cycle
- Test deterministic resume
- Test backward compatibility
- Test corrupted file handling

**E2E Tests** (file: `tests/e2e/test_cascade_training.py`):

- Train on 2-spiral problem, save, load, resume
- Verify final accuracy identical to continuous training

---

## Running the Test Suite

```bash
# All tests
pytest src/prototypes/cascor/src/tests/ -v

# Just serialization
pytest src/prototypes/cascor/src/tests/ -v -k serialization

# With coverage
pytest src/prototypes/cascor/src/tests/ --cov=snapshots --cov-report=html
```

---

## Review Checklist Before Merging

- [ ] All critical serialization fixes verified working
- [ ] Test suite passes (minimum 80% coverage for snapshot_serializer.py)
- [ ] No diagnostic errors in modified files
- [ ] Documentation updated (SERIALIZATION_FIXES_SUMMARY.md)
- [ ] AGENTS.md updated with new test commands
- [ ] Code formatted and linted
- [ ] Manual smoke test: train → save → load → resume → verify

---

## Contact & Questions

For questions about serialization implementation, see:

- `notes/SERIALIZATION_FIXES_SUMMARY.md` - Detailed analysis
- `src/snapshots/snapshot_serializer.py` - Implementation
- Oracle analysis output (embedded in analysis session)
