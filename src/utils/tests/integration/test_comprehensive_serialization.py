#!/usr/bin/env python
"""
Comprehensive serialization tests for CascadeCorrelationNetwork MVP completion.

Tests for:
- Deterministic training resume
- Hidden units preservation
- Config roundtrip
- Activation function restoration
- Torch random state restoration
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
from snapshots.snapshot_serializer import CascadeHDF5Serializer


class TestDeterministicTrainingResume(unittest.TestCase):
    """Test that training can be paused, saved, loaded, and resumed deterministically."""

    def test_deterministic_training_resume(self):
        """
        Critical test: Train → Save → Load → Resume should be identical to continuous training.
        This is the most important test for deterministic reproducibility.
        """
        # Setup
        config = CascadeCorrelationConfig(
            input_size=2,
            output_size=1,
            candidate_pool_size=3,
            candidate_epochs=10,
            output_epochs=20,
            max_hidden_units=2,
            random_seed=42,
        )

        # Create test data
        torch.manual_seed(42)
        x_train = torch.randn(50, 2)
        y_train = (x_train[:, 0] > x_train[:, 1]).float().unsqueeze(1)

        # Scenario A: Train for 20 epochs, save, train for 20 more
        network_a = CascadeCorrelationNetwork(config=config)
        network_a.fit(
            x_train=x_train,
            y_train=y_train,
            max_epochs=20
        )
        serializer = CascadeHDF5Serializer()
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_file = f.name
        try:
            success = serializer.save_network(network_a, temp_file)
            self.assertTrue(success, "Failed to save network")
            network_a_resumed = serializer.load_network(temp_file)
            self.assertIsNotNone(network_a_resumed, "Failed to load network")
            network_a_resumed.fit(x_train, y_train, epochs=20)

            # Scenario B: Train continuously for 40 epochs
            network_b = CascadeCorrelationNetwork(config=config)
            network_b.fit(x_train, y_train, epochs=40)

            # Verify outputs are identical
            torch.manual_seed(999)  # Different seed for test data
            test_x = torch.randn(10, 2)
            output_a = network_a_resumed.forward(test_x)
            output_b = network_b.forward(test_x)
            np.testing.assert_array_almost_equal(
                output_a.detach().numpy(),
                output_b.detach().numpy(),
                decimal=5,
                err_msg="Resumed training diverged from continuous training",
            )
            print("✓ Deterministic training resume test passed")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestHiddenUnitsPreservation(unittest.TestCase):
    """Test that all hidden units are correctly saved and loaded."""

    def test_hidden_units_preservation(self):
        """Test that all hidden units with weights, biases, and correlations are preserved."""
        # Create network and add hidden units
        config = CascadeCorrelationConfig(input_size=2, output_size=1, random_seed=42)
        network = CascadeCorrelationNetwork(config=config)

        # Manually add hidden units for testing
        for i in range(3):
            unit = {
                "weights": torch.randn(2 + i) * 0.1,
                "bias": torch.randn(1) * 0.1,
                "correlation": 0.5 + i * 0.1,
            }
            network.hidden_units.append(unit)

        # Save and load
        serializer = CascadeHDF5Serializer()
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_file = f.name
        try:
            self._node_preservation(serializer, network, temp_file)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def _node_preservation(self, serializer, network, temp_file):
        success = serializer.save_network(network, temp_file)
        self.assertTrue(success, "Failed to save network")
        loaded_network = serializer.load_network(temp_file)
        self.assertIsNotNone(loaded_network, "Failed to load network")

        # Verify unit count
        self.assertEqual(
            len(network.hidden_units),
            len(loaded_network.hidden_units),
            "Hidden unit count mismatch",
        )

        # Verify each unit's data
        for idx, (orig, loaded) in enumerate(
            zip(network.hidden_units, loaded_network.hidden_units, strict=False)
        ):
            np.testing.assert_array_almost_equal(
                orig["weights"].numpy(),
                loaded["weights"].numpy(),
                decimal=6,
                err_msg=f"Hidden unit {idx} weights mismatch",
            )
            np.testing.assert_array_almost_equal(
                orig["bias"].numpy(),
                loaded["bias"].numpy(),
                decimal=6,
                err_msg=f"Hidden unit {idx} bias mismatch",
            )
            self.assertAlmostEqual(
                orig["correlation"],
                loaded["correlation"],
                places=6,
                msg=f"Hidden unit {idx} correlation mismatch",
            )
        print(f"✓ Hidden units preservation test passed ({len(network.hidden_units)} units)")


class TestConfigRoundtrip(unittest.TestCase):
    """Test that configuration is correctly saved and loaded."""

    def test_config_roundtrip(self):
        """Test that all config parameters survive save/load cycle."""
        # Create network with non-default config
        config = CascadeCorrelationConfig(
            input_size=3,
            output_size=2,
            max_hidden_units=50,
            activation_function_name="Tanh",
            learning_rate=0.05,
            candidate_learning_rate=0.02,
            candidate_pool_size=15,
            candidate_epochs=100,
            epochs_max=5000,
            output_epochs=75,
            patience=20,
            correlation_threshold=0.002,
            random_seed=123,
            random_value_scale=0.2,
        )
        network = CascadeCorrelationNetwork(config=config)

        # Save and load
        serializer = CascadeHDF5Serializer()
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_file = f.name
        try:
            self._retrieved_nodes_match_original(serializer, network, temp_file)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def _retrieved_nodes_match_original(self, serializer, network, temp_file):
        success = serializer.save_network(network, temp_file)
        self.assertTrue(success, "Failed to save network")
        loaded_network = serializer.load_network(temp_file)
        self.assertIsNotNone(loaded_network, "Failed to load network")

        # Verify critical config parameters
        config_checks = {
            "input_size": (network.input_size, loaded_network.input_size),
            "output_size": (network.output_size, loaded_network.output_size),
            "max_hidden_units": (network.max_hidden_units, loaded_network.max_hidden_units),
            "activation_function_name": ( network.activation_function_name, loaded_network.activation_function_name,),
            "learning_rate": (network.learning_rate, loaded_network.learning_rate),
            "candidate_learning_rate": ( network.candidate_learning_rate, loaded_network.candidate_learning_rate,),
            "candidate_pool_size": ( network.candidate_pool_size, loaded_network.candidate_pool_size,),
            "correlation_threshold": ( network.correlation_threshold, loaded_network.correlation_threshold,),
            "random_seed": (network.random_seed, loaded_network.random_seed),
        }
        for param_name, (orig_val, loaded_val) in config_checks.items():
            self.assertEqual( orig_val, loaded_val, f"Config parameter '{param_name}' not preserved: {orig_val} != {loaded_val}",)
        print(f"✓ Config roundtrip test passed ({len(config_checks)} parameters verified)")


class TestActivationFunctionRestoration(unittest.TestCase):
    """Test that activation functions are correctly restored."""

    def test_activation_function_restoration(self):
        """Test that loaded networks use correct activation functions."""
        activation_functions = ["Tanh", "Sigmoid", "ReLU"]
        for af_name in activation_functions:
            with self.subTest(activation_function=af_name):

                # Create network with specific activation
                config = CascadeCorrelationConfig(
                    input_size=2,
                    output_size=1,
                    activation_function_name=af_name,
                    random_seed=42,
                )
                network = CascadeCorrelationNetwork(config=config)

                # Save and load
                serializer = CascadeHDF5Serializer()
                with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                    temp_file = f.name
                try:
                    serializer.save_network(network, temp_file)
                    loaded_network = serializer.load_network(temp_file)

                    # Verify activation function name
                    self.assertEqual( network.activation_function_name, loaded_network.activation_function_name, f"Activation function name not preserved for {af_name}",)

                    # Verify activation function produces same output
                    test_input = torch.tensor([1.0, -1.0, 0.0, 2.5])
                    orig_output = network.activation_fn(test_input)
                    loaded_output = loaded_network.activation_fn(test_input)
                    np.testing.assert_array_almost_equal(
                        orig_output.numpy(),
                        loaded_output.numpy(),
                        decimal=6,
                        err_msg=f"Activation function output mismatch for {af_name}",
                    )
                    print(f"✓ Activation function restoration test passed for {af_name}")
                finally:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)


class TestTorchRandomStateRestoration(unittest.TestCase):
    """Test that PyTorch RNG state is preserved across save/load."""

    def test_torch_random_state_restoration(self):
        """Test that PyTorch RNG state is preserved across save/load."""
        # Create network
        config = CascadeCorrelationConfig(input_size=2, output_size=1, random_seed=42)
        network = CascadeCorrelationNetwork(config=config)

        # Advance torch RNG
        for _ in range(10):
            torch.rand(1)

        # Save network
        serializer = CascadeHDF5Serializer()
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_file = f.name

        try:
            self._save_reload_and_validate_network( serializer, network, temp_file)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def _save_reload_and_validate_network(self, serializer, network, temp_file):
        success = serializer.save_network(network, temp_file)
        self.assertTrue(success, "Failed to save network")
        second_sequence = self._load_network_and_generate_sequence( serializer, temp_file, "Failed to load network")
        third_sequence = self._load_network_and_generate_sequence( serializer, temp_file, "Failed to load network second time")

        # Second and third sequences should match (proving deterministic restoration)
        np.testing.assert_array_almost_equal(
            second_sequence,
            third_sequence,
            decimal=6,
            err_msg="PyTorch random state restoration is not deterministic",
        )
        print("✓ PyTorch random state restoration test passed")

    def _load_network_and_generate_sequence(self, serializer, temp_file, arg2):
        # Load and generate second sequence
        loaded_network = serializer.load_network(temp_file)
        self.assertIsNotNone(loaded_network, arg2)
        return [torch.rand(1).item() for _ in range(5)]


class TestHistoryPreservation(unittest.TestCase):
    """Test that training history is correctly preserved."""

    def test_history_preservation(self):
        """Test that all history keys are loaded correctly."""
        # Create network and populate history
        config = CascadeCorrelationConfig(input_size=2, output_size=1)
        network = CascadeCorrelationNetwork(config=config)

        # Simulate training history
        network.history = {
            "train_loss": [0.5, 0.4, 0.3, 0.2],
            "train_accuracy": [0.6, 0.7, 0.8, 0.9],
            "value_loss": [0.55, 0.45, 0.35, 0.25],
            "value_accuracy": [0.55, 0.65, 0.75, 0.85],
            "hidden_units_added": [
                {
                    "correlation": 0.6,
                    "weights": np.array([0.1, 0.2]),
                    "bias": np.array([0.05]),
                },
                {
                    "correlation": 0.7,
                    "weights": np.array([0.3, 0.4, 0.5]),
                    "bias": np.array([0.06]),
                },
            ],
        }

        # Save and load with history
        serializer = CascadeHDF5Serializer()
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_file = f.name
        try:
            self._training_attribute_preservation(
                serializer, network, temp_file
            )
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def _training_attribute_preservation(self, serializer, network, temp_file):
        success = serializer.save_network( network, temp_file, include_training_state=True)
        self.assertTrue(success, "Failed to save network")
        loaded_network = serializer.load_network(temp_file)
        self.assertIsNotNone(loaded_network, "Failed to load network")

        # Verify history keys exist
        self.assertIn("train_loss", loaded_network.history)
        self.assertIn("train_accuracy", loaded_network.history)
        self.assertIn("value_loss", loaded_network.history)
        self.assertIn("value_accuracy", loaded_network.history)

        # Verify history values match
        np.testing.assert_array_almost_equal(
            network.history["train_loss"],
            loaded_network.history["train_loss"],
            decimal=6,
        )
        np.testing.assert_array_almost_equal(
            network.history["train_accuracy"],
            loaded_network.history["train_accuracy"],
            decimal=6,
        )
        print("✓ History preservation test passed")


if __name__ == "__main__":
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDeterministicTrainingResume))
    suite.addTests(loader.loadTestsFromTestCase(TestHiddenUnitsPreservation))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigRoundtrip))
    suite.addTests(loader.loadTestsFromTestCase(TestActivationFunctionRestoration))
    suite.addTests(loader.loadTestsFromTestCase(TestTorchRandomStateRestoration))
    suite.addTests(loader.loadTestsFromTestCase(TestHistoryPreservation))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    sys.exit(0 if result.wasSuccessful() else 1)
