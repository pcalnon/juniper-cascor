#!/usr/bin/env python
"""
Test script to validate P1 (High Priority) fixes for Cascor prototype.
Run with: /opt/miniforge3/envs/JuniperPython/bin/python test_p1_fixes.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import tempfile
from pathlib import Path

print("="*70)
print("P1 High Priority Fixes Validation Tests")
print("="*70)

def test_1_early_stopping():
    """Test that early stopping works in candidate training."""
    print("\n[Test 1] Early stopping implementation...")
    try:
        return _validate_candidate_early_stopping_helper(candidate=None)
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def _validate_candidate_early_stopping_helper(candidate):
    from candidate_unit.candidate_unit import CandidateUnit

    # Create candidate with early stopping enabled
    candidate = CandidateUnit(
        CandidateUnit__input_size=2,
        CandidateUnit__candidate_index=0,
        CandidateUnit__early_stopping=True,
        CandidateUnit__patience=3,
        CandidateUnit__log_level_name="WARNING"
    )

    # Create data that won't improve (constant)
    x = torch.ones(10, 2)
    residual_error = torch.zeros(10)  # No error = no correlation

    result = candidate.train(
        x=x,
        epochs=100,  # Request many epochs
        residual_error=residual_error,
        learning_rate=0.01
    )

    # Should stop early, not run all 100 epochs
    assert result.epochs_completed < 100, f"Should stop early, but ran {result.epochs_completed} epochs" # trunk-ignore(bandit/B101)
    assert result.epochs_completed <= 10, f"Should stop within ~patience epochs, ran {result.epochs_completed}" # trunk-ignore(bandit/B101)
    print(f"✅ PASS: Early stopping triggered at epoch {result.epochs_completed} (patience=3)")
    return True

def test_2_optimizer_serialization():
    """Test that optimizer state is saved and loaded."""
    print("\n[Test 2] Optimizer state serialization...")
    try:
        return _verify_optimizer_saved_to_hdf5_helper()
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def _verify_optimizer_saved_to_hdf5_helper():
    from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
    # from cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
    from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

    # Create network and train to create optimizer
    config = CascadeCorrelationConfig(input_size=2, output_size=1)
    network = CascadeCorrelationNetwork(config=config)

    x_train = torch.randn(20, 2)
    y_train = torch.randn(20, 1)

    # Train output layer to create optimizer
    loss = network.train_output_layer(x_train, y_train, epochs=5)
    assert loss is not None, "Loss should be computed" # trunk-ignore(bandit/B101)

    # Verify optimizer exists
    assert hasattr(network, 'output_optimizer'), "Network should have output_optimizer after training" # trunk-ignore(bandit/B101)
    assert network.output_optimizer is not None, "Optimizer should not be None" # trunk-ignore(bandit/B101)

    # Save to HDF5
    with tempfile.TemporaryDirectory() as tmpdir:
        _save_and_reload_network_hdf5_helper(
            tmpdir, network, CascadeCorrelationNetwork
        )
    return True


def _save_and_reload_network_hdf5_helper(tmpdir, network, CascadeCorrelationNetwork):
    filepath = Path(tmpdir) / "test_optimizer.h5"
    success = network.save_to_hdf5(str(filepath), include_training_state=True)
    assert success, "Save should succeed" # trunk-ignore(bandit/B101)

    # Load and verify optimizer is restored
    loaded_network = CascadeCorrelationNetwork.load_from_hdf5(str(filepath))
    assert loaded_network is not None, "Network should load successfully" # trunk-ignore(bandit/B101)
    assert hasattr(loaded_network, 'output_optimizer'), "Loaded network should have optimizer" # trunk-ignore(bandit/B101)

    print("✅ PASS: Optimizer saved and restored successfully")

def test_3_training_counters_persistence():
    """Test that training counters are persisted in HDF5."""
    print("\n[Test 3] Training counter persistence...")
    try:
        return _save_cascor_network_to_hdf5_helper()
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def _save_cascor_network_to_hdf5_helper():
    from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
    # from cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
    from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

    # Create network
    config = CascadeCorrelationConfig(input_size=2, output_size=1)
    network = CascadeCorrelationNetwork(config=config)

    # Ensure counters are initialized (they should be from __init__)
    assert hasattr(network, 'snapshot_counter'), "Network should have snapshot_counter attribute" # trunk-ignore(bandit/B101)

    # Set training counters to specific test values
    network.snapshot_counter = 5
    network.current_epoch = 42
    network.patience_counter = 7
    network.best_value_loss = 0.123

    # Save to HDF5
    with tempfile.TemporaryDirectory() as tmpdir:
        _verify_saved_and_restored_cascor_network_helper(
            tmpdir, network, CascadeCorrelationNetwork
        )
    return True


# TODO Rename this here and in `test_3_training_counters_persistence`
def _verify_saved_and_restored_cascor_network_helper(tmpdir, network, CascadeCorrelationNetwork):
    filepath = Path(tmpdir) / "test_counters.h5"
    success = network.save_to_hdf5(str(filepath))
    assert success, "Save should succeed" # trunk-ignore(bandit/B101)

    # Load and verify counters
    loaded_network = CascadeCorrelationNetwork.load_from_hdf5(str(filepath))
    assert loaded_network is not None, "Network should load" # trunk-ignore(bandit/B101)
    assert loaded_network.snapshot_counter == 5, f"Expected snapshot_counter=5, got {loaded_network.snapshot_counter}" # trunk-ignore(bandit/B101)
    assert loaded_network.current_epoch == 42, f"Expected current_epoch=42, got {loaded_network.current_epoch}" # trunk-ignore(bandit/B101)
    assert loaded_network.patience_counter == 7, f"Expected patience_counter=7, got {loaded_network.patience_counter}" # trunk-ignore(bandit/B101)
    assert abs(loaded_network.best_value_loss - 0.123) < 0.001, f"Expected best_value_loss=0.123, got {loaded_network.best_value_loss}" # trunk-ignore(bandit/B101)

    print("✅ PASS: All training counters restored correctly")

def test_4_queue_timeout():
    """Test that queue operations have timeouts (structural test)."""
    print("\n[Test 4] Queue timeout implementation...")
    try:
        # Check that the code includes timeout parameter
        with open('/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src/cascade_correlation/cascade_correlation.py', 'r') as f:
            content = f.read()
            
        # Check for timeout in result_queue.put
        assert 'result_queue.put(result, timeout=' in content, "result_queue.put should have timeout parameter" # trunk-ignore(bandit/B101)
        assert 'from queue import Full' in content, "Should import Full exception from queue" # trunk-ignore(bandit/B101)
        
        print("✅ PASS: Queue timeouts implemented correctly")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False

def test_5_optimizer_initialization():
    """Test optimizer is created as instance variable."""
    print("\n[Test 5] Optimizer initialization...")
    try:
        return _validate_cascor_network_optimizer_helper()
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def _validate_cascor_network_optimizer_helper():
    from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
    # from cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
    from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

    config = CascadeCorrelationConfig(input_size=2, output_size=1)
    network = CascadeCorrelationNetwork(config=config)

    # Initially should not have optimizer
    assert not hasattr(network, 'output_optimizer') or network.output_optimizer is None, "New network should not have optimizer yet" # trunk-ignore(bandit/B101)

    # After training, should have optimizer
    x = torch.randn(10, 2)
    y = torch.randn(10, 1)
    network.train_output_layer(x, y, epochs=2)

    assert hasattr(network, 'output_optimizer'), "Should have optimizer after training" # trunk-ignore(bandit/B101)
    assert network.output_optimizer is not None, "Optimizer should not be None" # trunk-ignore(bandit/B101)
    assert hasattr(network.output_optimizer, 'state_dict'), "Optimizer should have state_dict method" # trunk-ignore(bandit/B101)

    print("✅ PASS: Optimizer created and accessible as instance variable")
    return True

def _display_test_results_helper():
    print("\n🎉 ALL P1 FIXES VALIDATED SUCCESSFULLY")
    print("\nNext steps:")
    print("  1. Run full spiral problem test: python cascor.py")
    print("  2. Test multiprocessing with candidate pool")
    print("  3. Verify HDF5 save/load with trained networks")
    return 0

def main():
    """Run all P1 validation tests."""
    results = [("Early Stopping", test_1_early_stopping())]
    results.append(("Optimizer Serialization", test_2_optimizer_serialization()))
    results.append(("Training Counter Persistence", test_3_training_counters_persistence()))
    results.append(("Queue Timeouts", test_4_queue_timeout()))
    results.append(("Optimizer Initialization", test_5_optimizer_initialization()))

    print("\n" + "="*70)
    print("P1 Test Results Summary")
    print("="*70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(bool(p) for _, p in results)

    print("="*70)
    print(f"Total: {passed}/{total} tests passed ({100*passed//total}%)")
    print("="*70)

    if passed == total:
        return _display_test_results_helper()
    print(f"\n⚠️  {total - passed} test(s) failed - review output above")
    return 1

if __name__ == "__main__":
    sys.exit(main())
