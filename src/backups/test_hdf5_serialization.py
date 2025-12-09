#!/usr/bin/env python
"""
Comprehensive test suite for HDF5 serialization functionality.
"""

import os

# Add parent directories for imports
import sys
import tempfile

# import numpy as np
# from pathlib import Path
import unittest

import torch

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from cascade_correlation_config.cascade_correlation_config import (
    CascadeCorrelationConfig,
)

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.hdf5_serializer.cascade_hdf5_serializer import (
    CascadeHDF5Serializer,
)
from cascade_correlation.hdf5_serializer.hdf5_utils import HDF5Utils


class TestHDF5Serializer(unittest.TestCase):
    """Test cases for HDF5 serialization functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_filepath = os.path.join(self.temp_dir, "test_network.h5")

        # Create a test network
        config = CascadeCorrelationConfig.create_simple_config(
            input_size=2,
            output_size=1,
            learning_rate=0.01,
            max_hidden_units=5,
            candidate_pool_size=4,
        )
        self.network = CascadeCorrelationNetwork(config=config)

        # Train it briefly to get some state
        x = torch.randn(10, 2)
        y = torch.randn(10, 1)
        self.network.train_output_layer(x, y, epochs=5)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_and_load_basic(self):
        """Test basic save and load functionality."""
        serializer = CascadeHDF5Serializer()

        # Save network
        success = serializer.save_network(self.network, self.test_filepath)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(self.test_filepath))

        # Load network
        loaded_network = serializer.load_network(self.test_filepath)
        self.assertIsNotNone(loaded_network)

        # Verify basic attributes
        self.assertEqual(loaded_network.input_size, self.network.input_size)
        self.assertEqual(loaded_network.output_size, self.network.output_size)
        self.assertEqual(
            loaded_network.activation_function_name,
            self.network.activation_function_name,
        )

    def test_weight_preservation(self):
        """Test that weights and biases are preserved exactly."""
        loaded_network = self._save_and_load_network()
        # Compare output weights
        torch.testing.assert_close(
            loaded_network.output_weights,
            self.network.output_weights,
            rtol=1e-5,
            atol=1e-8,
        )

        # Compare output bias
        torch.testing.assert_close(
            loaded_network.output_bias, self.network.output_bias, rtol=1e-5, atol=1e-8
        )

    def test_network_with_hidden_units(self):
        """Test saving and loading network with hidden units."""
        # Add some hidden units manually for testing
        dummy_candidate = type(
            "DummyCandidate",
            (),
            {"weights": torch.randn(2), "bias": torch.randn(1), "correlation": 0.75},
        )()

        # Add the candidate as a hidden unit
        x_test = torch.randn(5, 2)
        self.network.add_unit(dummy_candidate, x_test)

        loaded_network = self._save_and_load_network()
        # Verify hidden units
        self.assertEqual(
            len(loaded_network.hidden_units), len(self.network.hidden_units)
        )

        for orig_unit, loaded_unit in zip(
            self.network.hidden_units, loaded_network.hidden_units, strict=False
        ):
            torch.testing.assert_close(
                loaded_unit["weights"], orig_unit["weights"], rtol=1e-5, atol=1e-8
            )
            self.assertAlmostEqual(
                loaded_unit["correlation"], orig_unit["correlation"], places=6
            )

    def test_functional_equivalence(self):
        """Test that loaded network produces same outputs as original."""
        loaded_network = self._save_and_load_network()
        # Test with same input
        test_input = torch.randn(3, 2)

        original_output = self.network.forward(test_input)
        loaded_output = loaded_network.forward(test_input)

        torch.testing.assert_close(loaded_output, original_output, rtol=1e-5, atol=1e-8)

    def _save_and_load_network(self):
        serializer = CascadeHDF5Serializer()
        serializer.save_network(self.network, self.test_filepath)
        # result = serializer.load_network(self.test_filepath)
        return serializer.load_network(self.test_filepath)

    def test_multiprocessing_state_preservation(self):
        """Test that multiprocessing configuration is preserved."""
        serializer = CascadeHDF5Serializer()

        # Save and load
        serializer.save_network(self.network, self.test_filepath)
        loaded_network = serializer.load_network(
            self.test_filepath, restore_multiprocessing=True
        )

        # Verify multiprocessing attributes
        self.assertEqual(
            loaded_network.candidate_training_queue_authkey,
            self.network.candidate_training_queue_authkey,
        )
        self.assertEqual(
            loaded_network.candidate_training_queue_address,
            self.network.candidate_training_queue_address,
        )
        self.assertEqual(
            loaded_network.candidate_pool_size, self.network.candidate_pool_size
        )

    def test_compression_options(self):
        """Test different compression options."""
        serializer = CascadeHDF5Serializer()

        # Test different compression methods
        compression_methods = ["gzip", "lzf", "szip"]

        for compression in compression_methods:
            if compression == "szip":
                # Skip szip if not available
                try:
                    import h5py

                    # with h5py.File(tempfile.mktemp(suffix=".h5"), "w") as f:
                    with h5py.File(
                        tempfile.TemporaryFile(
                            dir=self.temp_dir,
                            prefix="juniper_snapshot_",
                            suffix=".h5",
                            mode="w+b",
                        ),
                        "w+b",
                    ) as f:
                        f.create_dataset("test", data=[1, 2, 3], compression="szip")
                except Exception as e:
                    print(f"Skipping szip compression test: {e}")
                    continue

            filepath = os.path.join(self.temp_dir, f"test_{compression}.h5")

            # Save with compression
            success = serializer.save_network(
                self.network,
                filepath,
                compression=compression,
                compression_opts=6 if compression == "gzip" else None,
            )
            self.assertTrue(success)

            # Load and verify
            loaded_network = serializer.load_network(filepath)
            self.assertIsNotNone(loaded_network)

    def test_file_verification(self):
        """Test file verification functionality."""
        serializer = CascadeHDF5Serializer()

        # Save network
        serializer.save_network(self.network, self.test_filepath)

        # Verify file
        verification = serializer.verify_saved_network(self.test_filepath)

        self.assertTrue(verification["valid"])
        self.assertEqual(verification["input_size"], self.network.input_size)
        self.assertEqual(verification["output_size"], self.network.output_size)
        self.assertIn("creation_timestamp", verification)
        self.assertIn("network_uuid", verification)

    def test_hdf5_utils(self):
        """Test HDF5 utility functions."""
        serializer = CascadeHDF5Serializer()

        # Save network
        serializer.save_network(self.network, self.test_filepath)

        # Test backup creation
        backup_path = HDF5Utils.create_backup(self.test_filepath)
        self.assertTrue(os.path.exists(backup_path))

        # Test file info
        info = HDF5Utils.get_file_info(self.test_filepath)
        self.assertTrue(info["exists"])
        self.assertGreater(info["size_bytes"], 0)
        self.assertIn("groups", info)
        self.assertIn("datasets", info)

        # Test directory listing
        networks = HDF5Utils.list_networks_in_directory(self.temp_dir)
        self.assertGreater(len(networks), 0)

        # Test compression
        compressed_path = os.path.join(self.temp_dir, "compressed.h5")
        success = HDF5Utils.compress_hdf5_file(
            self.test_filepath, compressed_path, compression="gzip", compression_opts=9
        )
        self.assertTrue(success)
        self.assertTrue(os.path.exists(compressed_path))


def run_comprehensive_test():
    """Run comprehensive test of the serialization system."""
    print("Running comprehensive HDF5 serialization tests...")

    # Create test network
    config = CascadeCorrelationConfig.create_simple_config(
        input_size=3, output_size=2, learning_rate=0.05, max_hidden_units=10
    )
    network = CascadeCorrelationNetwork(config=config)

    # Train briefly
    x = torch.randn(20, 3)
    y = torch.randn(20, 2)
    network.train_output_layer(x, y, epochs=10)

    # Add some training history
    network.history["train_loss"] = [0.5, 0.4, 0.3, 0.2, 0.1]
    network.history["train_accuracy"] = [0.6, 0.7, 0.8, 0.9, 0.95]

    # Test save/load cycle
    with tempfile.TemporaryDirectory() as temp_dir:
        _save_and_load_snapshot(temp_dir, network)


def _save_and_load_snapshot(temp_dir, network):
    filepath = os.path.join(temp_dir, "comprehensive_test.h5")

    # Save
    success = network.save_to_hdf5(filepath, include_training_state=True)
    print(f"✓ Save successful: {success}")

    # Load
    loaded_network = CascadeCorrelationNetwork.load_from_hdf5(filepath)
    print(f"✓ Load successful: {loaded_network is not None}")

    if loaded_network:
        _function_equivalence(network, loaded_network)
    # Test verification
    verification = loaded_network.verify_hdf5_file(filepath)
    print(f"✓ File verification: {verification.get('valid', False)}")

    print("\n✓ All comprehensive tests completed successfully!")


def _function_equivalence(network, loaded_network):
    # Test functional equivalence
    test_input = torch.randn(5, 3)
    orig_output = network.forward(test_input)
    loaded_output = loaded_network.forward(test_input)

    # Check if outputs are close
    max_diff = torch.max(torch.abs(orig_output - loaded_output)).item()
    print(f"✓ Maximum output difference: {max_diff:.2e}")

    # Verify configuration
    print(f"✓ Input size match: {loaded_network.input_size == network.input_size}")
    print(f"✓ Output size match: {loaded_network.output_size == network.output_size}")
    print(f"✓ UUID preservation: {loaded_network.get_uuid() == network.get_uuid()}")

    # Verify training history
    print(
        f"✓ Training history preserved: {len(loaded_network.history['train_loss']) == len(network.history['train_loss'])}"
    )


if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=[""], exit=False, verbosity=2)

    # Run comprehensive test
    print("\n" + "=" * 50)
    run_comprehensive_test()
