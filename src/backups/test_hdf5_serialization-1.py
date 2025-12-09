#!/usr/bin/env python
"""
Comprehensive test suite for HDF5 serialization functionality.
"""
import sys

# import os
from pathlib import Path

import torch

# import numpy as np

# Add source directory to path
sys.path.append("/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src")

from hdf5_storage.hdf5_manager import CascadeCorrelationHDF5Manager

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork


def create_test_network():
    """Create a test network with some training history."""
    print("Creating test network...")

    # Create network with small parameters
    network = CascadeCorrelationNetwork.create_simple_network(
        input_size=2, output_size=1, learning_rate=0.01, max_hidden_units=5
    )

    # Create some test data
    x = torch.randn(50, 2)
    y = torch.randint(0, 2, (50, 1)).float()

    # Train for a few epochs to create history
    network.fit(x, y, max_epochs=10)

    print(f"✓ Created network with {len(network.hidden_units)} hidden units")
    return network, x, y


def test_save_and_load():
    """Test basic save and load functionality."""
    print("\n=== Testing Save and Load Functionality ===")

    # Create test network
    original_network, x, y = create_test_network()
    original_uuid = original_network.get_uuid()

    # Save network
    test_file = Path("test_network.h5")
    success = original_network.save_to_hdf5(test_file)

    if not success:
        print("✗ Failed to save network")
        return False

    print("✓ Network saved successfully")

    # Load network
    loaded_network = CascadeCorrelationNetwork.load_from_hdf5(test_file)

    if loaded_network is None:
        print("✗ Failed to load network")
        return False

    print("✓ Network loaded successfully")

    # Verify UUID preservation
    if loaded_network.get_uuid() != original_uuid:
        print(f"✗ UUID mismatch: {loaded_network.get_uuid()} != {original_uuid}")
        return False

    print("✓ UUID preserved correctly")

    # Verify network architecture
    if (
        loaded_network.input_size != original_network.input_size
        or loaded_network.output_size != original_network.output_size
        or len(loaded_network.hidden_units) != len(original_network.hidden_units)
    ):
        print("✗ Network architecture not preserved")
        return False

    print("✓ Network architecture preserved")

    # Verify predictions are identical
    with torch.no_grad():
        original_output = original_network.forward(x)
        loaded_output = loaded_network.forward(x)

        if not torch.allclose(original_output, loaded_output, atol=1e-6):
            print("✗ Network predictions differ")
            return False

    print("✓ Network predictions identical")

    # Cleanup
    test_file.unlink()
    print("✓ Test file cleaned up")

    return True


def test_multiprocessing_restoration():
    """Test multiprocessing state restoration."""
    print("\n=== Testing Multiprocessing Restoration ===")

    # Create test network
    original_network, _, _ = create_test_network()

    # Save with multiprocessing state
    test_file = Path("test_network_mp.h5")
    success = original_network.save_to_hdf5(test_file)

    if not success:
        print("✗ Failed to save network")
        return False

    # Load with multiprocessing restoration
    loaded_network = CascadeCorrelationNetwork.load_from_hdf5(
        test_file, restore_multiprocessing=True
    )

    if loaded_network is None:
        print("✗ Failed to load network")
        return False

    # Verify multiprocessing attributes
    mp_attrs = [
        "candidate_training_queue_address",
        "candidate_training_queue_authkey",
        "candidate_training_tasks_queue_timeout",
        "candidate_training_shutdown_timeout",
    ]

    for attr in mp_attrs:
        if hasattr(original_network, attr) and hasattr(loaded_network, attr):
            orig_val = getattr(original_network, attr)
            loaded_val = getattr(loaded_network, attr)
            if orig_val != loaded_val:
                print(
                    f"✗ Multiprocessing attribute {attr} not preserved: {orig_val} != {loaded_val}"
                )
                return False

    print("✓ Multiprocessing configuration preserved")

    # Cleanup
    test_file.unlink()

    return True


def test_backup_functionality():
    """Test backup creation and listing."""
    print("\n=== Testing Backup Functionality ===")

    # Create test network
    network, _, _ = create_test_network()

    # Create backup directory
    backup_dir = Path("test_backups")
    backup_dir.mkdir(exist_ok=True)

    # Create HDF5 manager
    hdf5_manager = CascadeCorrelationHDF5Manager()

    # Create backup
    backup_path = hdf5_manager.create_backup(network, backup_dir)

    if backup_path is None or not backup_path.exists():
        print("✗ Failed to create backup")
        return False

    print(f"✓ Backup created at {backup_path}")

    # List backups
    backups = hdf5_manager.list_backups(backup_dir)

    if len(backups) == 0:
        print("✗ No backups found")
        return False

    print(f"✓ Found {len(backups)} backups")

    # Verify backup contains expected metadata
    backup_info = backups[0]
    expected_keys = ["filepath", "filename", "size_mb", "modified", "uuid"]

    for key in expected_keys:
        if key not in backup_info:
            print(f"✗ Missing backup metadata: {key}")
            return False

    print("✓ Backup metadata complete")

    # Cleanup
    import shutil

    shutil.rmtree(backup_dir)

    return True


def test_snapshot_functionality():
    """Test snapshot creation."""
    print("\n=== Testing Snapshot Functionality ===")

    # Create test network
    network, _, _ = create_test_network()

    # Create snapshot
    snapshot_dir = Path("test_snapshots")
    snapshot_path = network.create_snapshot(snapshot_dir)

    if snapshot_path is None or not snapshot_path.exists():
        print("✗ Failed to create snapshot")
        return False

    print(f"✓ Snapshot created at {snapshot_path}")

    # Verify snapshot can be loaded
    loaded_network = CascadeCorrelationNetwork.load_from_hdf5(snapshot_path)

    if loaded_network is None:
        print("✗ Failed to load snapshot")
        return False

    print("✓ Snapshot loaded successfully")

    # Cleanup
    import shutil

    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)

    return True


def run_all_tests():
    """Run all HDF5 serialization tests."""
    print("HDF5 Serialization Test Suite")
    print("=" * 50)

    tests = [
        test_save_and_load,
        test_multiprocessing_restoration,
        test_backup_functionality,
        test_snapshot_functionality,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ PASSED")
            else:
                print("❌ FAILED")
        except Exception as e:
            print(f"❌ FAILED with exception: {e}")

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("HDF5 serialization system is working correctly.")
    else:
        print("⚠️  SOME TESTS FAILED!")
        print("Please check the implementation.")


if __name__ == "__main__":
    run_all_tests()
