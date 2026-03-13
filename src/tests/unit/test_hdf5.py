#!/usr/bin/env python
"""
Simple test script to verify HDF5 serialization functionality.
"""

import os
import sys
import tempfile

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork

# from cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
from snapshots.snapshot_utils import HDF5Utils

# from pathlib import Path

# Add parent directories for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_hdf5_serialization():
    """Test basic HDF5 save/load functionality."""
    print("Testing HDF5 serialization...")

    # Import required modules
    from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork

    # from cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
    from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

    # from cascade_correlation.hdf5.serializer import CascadeHDF5Serializer
    # from cascade_correlation.hdf5.utils import HDF5Utils
    from snapshots.snapshot_utils import HDF5Utils

    # Create a simple network
    config = CascadeCorrelationConfig(input_size=2, output_size=1, max_hidden_units=5, learning_rate=0.1, activation_function_name="tanh")
    network = CascadeCorrelationNetwork(config=config)

    print(f"✓ Created network with UUID: {network.get_uuid()}")
    print(f"  Architecture: {network.input_size} → {len(network.hidden_units)} → {network.output_size}")

    # Create temporary file for testing
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        test_file = tmp_file.name

    try:
        _check_hdf5_snapshot(test_file, network)
    finally:
        # Cleanup
        if os.path.exists(test_file):  # sourcery skip: no-conditionals-in-tests
            os.remove(test_file)


def _check_hdf5_snapshot(test_file: str, network: CascadeCorrelationNetwork) -> None:
    # Test saving
    print(f"Saving network to {test_file}...")
    success = network.save_to_hdf5(test_file, include_training_state=False)

    assert success, "Failed to save network"

    print("✓ Network saved successfully")

    # Verify the file
    print("Verifying saved file...")
    verification = network.verify_hdf5_file(test_file)

    assert verification.get("valid", False), f"File verification failed: {verification.get('error', 'Unknown error')}"

    print("✓ File verification passed")
    print(f"  Format: {verification.get('format', 'unknown')} v{verification.get('format_version', 'unknown')}")

    # Test loading
    print("Loading network from file...")
    loaded_network = CascadeCorrelationNetwork.load_from_hdf5(test_file)

    assert loaded_network, "Failed to load network"

    print("✓ Network loaded successfully")
    print(f"  Loaded UUID: {loaded_network.get_uuid()}")

    # Verify loaded network matches original
    assert str(network.get_uuid()) == str(loaded_network.get_uuid()), "UUIDs don't match!"
    assert network.input_size == loaded_network.input_size, "Input sizes don't match!"
    assert network.output_size == loaded_network.output_size, "Output sizes don't match!"
    assert network.activation_function_name == loaded_network.activation_function_name, "Activation functions don't match!"

    print("✓ All network properties match")

    # Test utilities
    print("Testing HDF5 utilities...")
    file_info = HDF5Utils.get_file_info(test_file)

    assert file_info.get("exists", False), "File info failed"

    print(f"✓ File info: {file_info.get('size_mb', 0):.2f} MB, {len(file_info.get('groups', []))} groups")

    # Test network summary
    summary = HDF5Utils.get_network_summary(test_file)
    assert summary, "Failed to get network summary"
    print(f"✓ Network summary: {summary['input_size']}→{summary['output_size']}, {summary['num_hidden_units']} hidden units")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    try:
        test_hdf5_serialization()
        exit(0)
    except Exception as e:
        print(f"✗ Test failed: {e}")
        exit(1)
