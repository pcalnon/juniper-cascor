#!/usr/bin/env python
"""
Simple test script to verify HDF5 serialization functionality.
"""

import os
import sys
import tempfile
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

# from pathlib import Path

# Add parent directories for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_hdf5_serialization():
    """Test basic HDF5 save/load functionality."""
    try:
        print("Testing HDF5 serialization...")

        # Import required modules
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        from cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
        # from cascade_correlation.hdf5.serializer import CascadeHDF5Serializer
        # from cascade_correlation.hdf5.utils import HDF5Utils
        from cascade_correlation.snapshots.snapshot_utils import HDF5Utils

        # Create a simple network
        config = CascadeCorrelationConfig(
            input_size=2,
            output_size=1,
            max_hidden_units=5,
            learning_rate=0.1,
            activation_function_name='tanh'
        )
        network = CascadeCorrelationNetwork(config=config)

        print(f"✓ Created network with UUID: {network.get_uuid()}")
        print(f"  Architecture: {network.input_size} → {len(network.hidden_units)} → {network.output_size}")

        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            test_file = tmp_file.name

        try:
            return _check_hdf5_snapshot(test_file, network, CascadeCorrelationNetwork, HDF5Utils)
        finally:
            # Cleanup
            if os.path.exists(test_file):  # sourcery skip: no-conditionals-in-tests
                os.remove(test_file)

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


# TODO Rename this here and in `test_hdf5_serialization`
def _check_hdf5_snapshot(test_file: str, network: CascadeCorrelationNetwork, HDF5Utils: HDF5Utils) -> bool:
    # Test saving
    print(f"Saving network to {test_file}...")
    success = network.save_to_hdf5(test_file, include_training_state=False)

    if not success:  # sourcery skip: no-conditionals-in-tests
        print("✗ Failed to save network")
        return False

    print("✓ Network saved successfully")

    # Verify the file
    print("Verifying saved file...")
    verification = network.verify_hdf5_file(test_file)

    if not verification.get('valid', False):  # sourcery skip: no-conditionals-in-tests
        print(f"✗ File verification failed: {verification.get('error', 'Unknown error')}")
        return False

    print("✓ File verification passed")
    print(f"  Format: {verification.get('format', 'unknown')} v{verification.get('format_version', 'unknown')}")

    # Test loading
    print("Loading network from file...")
    loaded_network = CascadeCorrelationNetwork.load_from_hdf5(test_file)

    if not loaded_network:  # sourcery skip: no-conditionals-in-tests
        print("✗ Failed to load network")
        return False

    print("✓ Network loaded successfully")
    print(f"  Loaded UUID: {loaded_network.get_uuid()}")

    # Verify loaded network matches original
    if str(network.get_uuid()) != str(loaded_network.get_uuid()):  # sourcery skip: no-conditionals-in-tests
        print("✗ UUIDs don't match!")
        return False

    if network.input_size != loaded_network.input_size:  # sourcery skip: no-conditionals-in-tests
        print("✗ Input sizes don't match!")
        return False

    if network.output_size != loaded_network.output_size:  # sourcery skip: no-conditionals-in-tests
        print("✗ Output sizes don't match!")
        return False

    if network.activation_function_name != loaded_network.activation_function_name:  # sourcery skip: no-conditionals-in-tests
        print("✗ Activation functions don't match!")
        return False

    print("✓ All network properties match")

    # Test utilities
    print("Testing HDF5 utilities...")
    file_info = HDF5Utils.get_file_info(test_file)

    if not file_info.get('exists', False):  # sourcery skip: no-conditionals-in-tests
        print("✗ File info failed")
        return False

    print(f"✓ File info: {file_info.get('size_mb', 0):.2f} MB, {len(file_info.get('groups', []))} groups")

    # Test network summary
    if summary := HDF5Utils.get_network_summary(test_file):  # sourcery skip: no-conditionals-in-tests
        print(f"✓ Network summary: {summary['input_size']}→{summary['output_size']}, {summary['num_hidden_units']} hidden units")
    else:
        print("✗ Failed to get network summary")
        return False

    print("\n✓ All tests passed!")
    return True

if __name__ == '__main__':
    success = test_hdf5_serialization()
    exit(0 if success else 1)
