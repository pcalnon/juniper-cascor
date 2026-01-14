#!/usr/bin/env python
"""
Integration tests for HDF5 serialization/deserialization.

Tests the complete save/load cycle for CascadeCorrelationNetwork including:
- UUID persistence
- Deterministic behavior after restore
- History preservation
- Config roundtrip
- Activation function restoration
- Random seed preservation
- Hidden units preservation
- Backward compatibility
"""

import os
import sys

# import pathlib
# import random
import tempfile

# import numpy as np
import pytest
import torch

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration
# Add parent directories for imports
# import sys
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from cascade_correlation_config.cascade_correlation_config import (  # trunk-ignore(ruff/E402)
    CascadeCorrelationConfig,
)

from cascade_correlation.cascade_correlation import (  # trunk-ignore(ruff/E402)
    CascadeCorrelationNetwork,
)
from snapshots.snapshot_serializer import (  # trunk-ignore(ruff/E402)
    CascadeHDF5Serializer,
)


@pytest.fixture
def temp_snapshot_file():
    """Create a temporary file for snapshot testing."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        filepath = f.name
    yield filepath
    # Cleanup
    if os.path.exists(filepath):
        os.unlink(filepath)


@pytest.fixture
def simple_network():
    """Create a simple network for testing."""
    config = CascadeCorrelationConfig.create_simple_config(
        input_size=2,
        output_size=1,
        learning_rate=0.1,
        max_hidden_units=3,
        random_seed=42,
    )
    return CascadeCorrelationNetwork(config=config)


@pytest.fixture
def network_with_hidden_units():
    """Create a network with some hidden units for testing."""
    config = CascadeCorrelationConfig.create_simple_config(
        input_size=2,
        output_size=1,
        learning_rate=0.1,
        max_hidden_units=5,
        random_seed=42,
    )
    network = CascadeCorrelationNetwork(config=config)

    # Manually add some hidden units for testing
    for i in range(3):
        unit = {
            "weights": torch.randn(2 + i) * 0.1,
            "bias": torch.randn(1) * 0.1,
            "correlation": 0.5 + i * 0.1,
            "activation_fn": network.activation_fn,
        }
        network.hidden_units.append(unit)

    # Update output weights to match new architecture
    network.output_weights = torch.randn(2 + 3, 1) * 0.1

    return network


class TestUUIDPersistence:
    """Test that network UUID survives save/load cycles."""

    def test_uuid_preserved_after_load(self, simple_network, temp_snapshot_file):
        """Test UUID is preserved across save/load."""
        original_uuid = simple_network.get_uuid()
        assert original_uuid is not None  # trunk-ignore(bandit/B101)

        # Save network
        serializer = CascadeHDF5Serializer()
        success = serializer.save_network(simple_network, temp_snapshot_file)
        assert success, "Failed to save network"  # trunk-ignore(bandit/B101)

        # Load network
        loaded_network = serializer.load_network(temp_snapshot_file)
        assert (
            loaded_network is not None
        ), "Failed to load network"  # trunk-ignore(bandit/B101)

        # Verify UUID matches
        loaded_uuid = loaded_network.get_uuid()
        assert (
            loaded_uuid == original_uuid
        ), f"UUID mismatch: {loaded_uuid} != {original_uuid}"  # trunk-ignore(bandit/B101)

    def test_multiple_save_load_cycles_preserve_uuid(
        self, simple_network, temp_snapshot_file
    ):
        """Test UUID remains consistent across multiple save/load cycles."""
        original_uuid = simple_network.get_uuid()
        serializer = CascadeHDF5Serializer()

        current_network = simple_network
        for cycle in range(3):
            # Save
            success = serializer.save_network(current_network, temp_snapshot_file)
            assert (
                success
            ), f"Failed to save network in cycle {cycle}"  # trunk-ignore(bandit/B101)

            # Load
            current_network = serializer.load_network(temp_snapshot_file)
            assert (
                current_network is not None
            ), f"Failed to load network in cycle {cycle}"  # trunk-ignore(bandit/B101)

            # Verify UUID
            assert (
                current_network.get_uuid() == original_uuid
            ), f"UUID mismatch in cycle {cycle}"  # trunk-ignore(bandit/B101)


class TestRandomSeedPreservation:
    """Test that random seed parameters are preserved for deterministic training."""

    def test_random_seed_parameters_preserved(self, simple_network, temp_snapshot_file):
        """Test that random seed configuration is preserved."""
        original_seed = simple_network.random_seed
        original_max = simple_network.random_max_value
        original_seq_max = simple_network.sequence_max_value
        original_scale = simple_network.random_value_scale
        serializer = CascadeHDF5Serializer()
        success = serializer.save_network(simple_network, temp_snapshot_file)
        assert success  # trunk-ignore(bandit/B101)
        loaded_network = serializer.load_network(temp_snapshot_file)
        assert loaded_network is not None  # trunk-ignore(bandit/B101)

        # Verify all random parameters match
        assert loaded_network.random_seed == original_seed  # trunk-ignore(bandit/B101)
        assert (
            loaded_network.random_max_value == original_max
        )  # trunk-ignore(bandit/B101)
        assert (
            loaded_network.sequence_max_value == original_seq_max
        )  # trunk-ignore(bandit/B101)
        assert (
            loaded_network.random_value_scale == original_scale
        )  # trunk-ignore(bandit/B101)

    def test_deterministic_forward_pass(self, simple_network, temp_snapshot_file):
        """Test that loaded network produces identical outputs."""
        # Create test input
        test_input = torch.randn(10, 2)

        # Get output from original network
        with torch.no_grad():
            original_output = simple_network.forward(test_input)

        # Save network
        serializer = CascadeHDF5Serializer()
        success = serializer.save_network(simple_network, temp_snapshot_file)
        assert success  # trunk-ignore(bandit/B101)

        # Load network
        loaded_network = serializer.load_network(temp_snapshot_file)
        assert loaded_network is not None  # trunk-ignore(bandit/B101)

        # Get output from loaded network
        with torch.no_grad():
            loaded_output = loaded_network.forward(test_input)

        # Outputs should be identical
        assert torch.allclose(
            original_output, loaded_output, atol=1e-6
        ), "Network outputs don't match after load"  # trunk-ignore(bandit/B101)

    def test_random_state_data_in_file(self, simple_network, temp_snapshot_file):
        """Test that HDF5 file contains random state datasets."""
        serializer = CascadeHDF5Serializer()
        success = serializer.save_network(simple_network, temp_snapshot_file)
        assert success  # trunk-ignore(bandit/B101)

        # Verify file structure
        import h5py

        with h5py.File(temp_snapshot_file, "r") as hdf5_file:
            assert "random" in hdf5_file  # trunk-ignore(bandit/B101)
            random_group = hdf5_file["random"]

            # Check all RNG states are present
            assert "python_state" in random_group  # trunk-ignore(bandit/B101)
            assert "numpy_state" in random_group  # trunk-ignore(bandit/B101)
            assert "torch_state" in random_group  # trunk-ignore(bandit/B101)


class TestHistoryPreservation:
    """Test that training history is correctly preserved."""

    def test_history_keys_preserved(self, simple_network, temp_snapshot_file):
        """Test that all history keys are preserved."""
        # Add some history
        simple_network.history["train_loss"] = [1.0, 0.8, 0.6]
        simple_network.history["value_loss"] = [1.1, 0.9, 0.7]
        simple_network.history["train_accuracy"] = [0.5, 0.6, 0.7]
        simple_network.history["value_accuracy"] = [0.4, 0.55, 0.65]

        # Save network with training state
        serializer = CascadeHDF5Serializer()
        success = serializer.save_network(
            simple_network, temp_snapshot_file, include_training_state=True
        )
        assert success  # trunk-ignore(bandit/B101)

        # Load network
        loaded_network = serializer.load_network(temp_snapshot_file)
        assert loaded_network is not None  # trunk-ignore(bandit/B101)

        # Verify all history keys exist
        assert "train_loss" in loaded_network.history  # trunk-ignore(bandit/B101)
        assert "value_loss" in loaded_network.history  # trunk-ignore(bandit/B101)
        assert "train_accuracy" in loaded_network.history  # trunk-ignore(bandit/B101)
        assert "value_accuracy" in loaded_network.history  # trunk-ignore(bandit/B101)

        # Verify values match
        assert loaded_network.history["train_loss"] == [
            1.0,
            0.8,
            0.6,
        ]  # trunk-ignore(bandit/B101)
        assert loaded_network.history["value_loss"] == [
            1.1,
            0.9,
            0.7,
        ]  # trunk-ignore(bandit/B101)
        assert loaded_network.history["train_accuracy"] == [
            0.5,
            0.6,
            0.7,
        ]  # trunk-ignore(bandit/B101)
        assert loaded_network.history["value_accuracy"] == [
            0.4,
            0.55,
            0.65,
        ]  # trunk-ignore(bandit/B101)

    def test_hidden_units_history_preserved(
        self, network_with_hidden_units, temp_snapshot_file
    ):
        """Test that hidden units added history is preserved."""
        # Add history of added units
        for unit in network_with_hidden_units.hidden_units:
            network_with_hidden_units.history["hidden_units_added"].append(
                {
                    "weights": unit["weights"].numpy(),
                    "bias": unit["bias"].numpy(),
                    "correlation": unit["correlation"],
                }
            )

        # Save with training state
        serializer = CascadeHDF5Serializer()
        success = serializer.save_network(
            network_with_hidden_units, temp_snapshot_file, include_training_state=True
        )
        assert success  # trunk-ignore(bandit/B101)

        # Load network
        loaded_network = serializer.load_network(temp_snapshot_file)
        assert loaded_network is not None  # trunk-ignore(bandit/B101)

        # Verify hidden units history
        assert (
            len(loaded_network.history["hidden_units_added"]) == 3
        )  # trunk-ignore(bandit/B101)
        for unit_data in loaded_network.history["hidden_units_added"]:
            assert "weights" in unit_data  # trunk-ignore(bandit/B101)
            assert "bias" in unit_data  # trunk-ignore(bandit/B101)
            assert "correlation" in unit_data  # trunk-ignore(bandit/B101)


class TestConfigRoundtrip:
    """Test that configuration survives save/load without errors."""

    def test_config_serialization_excludes_non_serializable(
        self, simple_network, temp_snapshot_file
    ):
        """Test that non-serializable config fields are excluded."""
        serializer = CascadeHDF5Serializer()
        success = serializer.save_network(simple_network, temp_snapshot_file)
        assert success  # trunk-ignore(bandit/B101)

        # Load network
        loaded_network = serializer.load_network(temp_snapshot_file)
        assert loaded_network is not None  # trunk-ignore(bandit/B101)

        # Verify network has reconstructed non-serializable objects
        assert hasattr(loaded_network, "activation_fn")  # trunk-ignore(bandit/B101)
        assert hasattr(
            loaded_network, "activation_functions_dict"
        )  # trunk-ignore(bandit/B101)
        assert loaded_network.activation_fn is not None  # trunk-ignore(bandit/B101)
        assert (
            loaded_network.activation_functions_dict is not None
        )  # trunk-ignore(bandit/B101)

    def test_config_values_preserved(self, simple_network, temp_snapshot_file):
        """Test that primitive config values are preserved."""
        original_config = simple_network.config
        serializer = CascadeHDF5Serializer()
        success = serializer.save_network(simple_network, temp_snapshot_file)
        assert success  # trunk-ignore(bandit/B101)
        loaded_network = serializer.load_network(temp_snapshot_file)
        assert loaded_network is not None  # trunk-ignore(bandit/B101)
        loaded_config = loaded_network.config

        # Verify key configuration values
        assert (
            loaded_config.input_size == original_config.input_size
        )  # trunk-ignore(bandit/B101)
        assert (
            loaded_config.output_size == original_config.output_size
        )  # trunk-ignore(bandit/B101)
        assert (
            loaded_config.learning_rate == original_config.learning_rate
        )  # trunk-ignore(bandit/B101)
        assert (
            loaded_config.max_hidden_units == original_config.max_hidden_units
        )  # trunk-ignore(bandit/B101)
        assert (
            loaded_config.random_seed == original_config.random_seed
        )  # trunk-ignore(bandit/B101)


class TestActivationFunctionRestoration:
    """Test that activation functions are properly restored."""

    def test_activation_function_name_restored(
        self, simple_network, temp_snapshot_file
    ):
        """Test activation function name is restored."""
        original_af_name = simple_network.activation_function_name
        serializer = CascadeHDF5Serializer()
        success = serializer.save_network(simple_network, temp_snapshot_file)
        assert success  # trunk-ignore(bandit/B101)
        loaded_network = serializer.load_network(temp_snapshot_file)
        assert loaded_network is not None  # trunk-ignore(bandit/B101)
        assert (
            loaded_network.activation_function_name == original_af_name
        )  # trunk-ignore(bandit/B101)

    def test_activation_function_callable_restored(
        self, simple_network, temp_snapshot_file
    ):
        """Test activation function callable is restored."""
        serializer = CascadeHDF5Serializer()
        success = serializer.save_network(simple_network, temp_snapshot_file)
        assert success  # trunk-ignore(bandit/B101)
        loaded_network = serializer.load_network(temp_snapshot_file)
        assert loaded_network is not None  # trunk-ignore(bandit/B101)

        # Test that activation function is callable and works
        test_input = torch.tensor([1.0, -1.0, 0.0])
        original_output = simple_network.activation_fn(test_input)
        loaded_output = loaded_network.activation_fn(test_input)
        assert torch.allclose(
            original_output, loaded_output
        ), "Activation function outputs don't match"  # trunk-ignore(bandit/B101)


class TestHiddenUnitsPreservation:
    """Test that hidden units are correctly preserved."""

    def test_hidden_units_count_preserved(
        self, network_with_hidden_units, temp_snapshot_file
    ):
        """Test number of hidden units is preserved."""
        original_count = len(network_with_hidden_units.hidden_units)
        assert original_count == 3  # trunk-ignore(bandit/B101)
        serializer = CascadeHDF5Serializer()
        success = serializer.save_network(network_with_hidden_units, temp_snapshot_file)
        assert success  # trunk-ignore(bandit/B101)
        loaded_network = serializer.load_network(temp_snapshot_file)
        assert loaded_network is not None  # trunk-ignore(bandit/B101)
        assert (
            len(loaded_network.hidden_units) == original_count
        )  # trunk-ignore(bandit/B101)

    def test_hidden_units_weights_preserved(
        self, network_with_hidden_units, temp_snapshot_file
    ):
        """Test hidden unit weights are preserved."""
        serializer = CascadeHDF5Serializer()
        success = serializer.save_network(network_with_hidden_units, temp_snapshot_file)
        assert success  # trunk-ignore(bandit/B101)
        loaded_network = serializer.load_network(temp_snapshot_file)
        assert loaded_network is not None  # trunk-ignore(bandit/B101)
        for i, (orig_unit, loaded_unit) in enumerate(
            zip(
                network_with_hidden_units.hidden_units,
                loaded_network.hidden_units,
                strict=False,
            )
        ):
            assert torch.allclose(
                orig_unit["weights"], loaded_unit["weights"]
            ), f"Hidden unit {i} weights don't match"  # trunk-ignore(bandit/B101)
            assert torch.allclose(
                orig_unit["bias"], loaded_unit["bias"]
            ), f"Hidden unit {i} bias doesn't match"  # trunk-ignore(bandit/B101)
            assert (
                orig_unit["correlation"] == loaded_unit["correlation"]
            ), f"Hidden unit {i} correlation doesn't match"  # trunk-ignore(bandit/B101)

    def test_hidden_units_checksums_verified(
        self, network_with_hidden_units, temp_snapshot_file
    ):
        """Test that hidden unit checksums are saved and verified."""
        serializer = CascadeHDF5Serializer()
        success = serializer.save_network(network_with_hidden_units, temp_snapshot_file)
        assert success  # trunk-ignore(bandit/B101)

        # Load and verify (checksum verification happens during load)
        loaded_network = serializer.load_network(temp_snapshot_file)
        assert loaded_network is not None  # trunk-ignore(bandit/B101)
        # If we get here, checksums were verified successfully


class TestShapeValidation:
    """Test that tensor shapes are validated on load."""

    def test_output_weights_shape_validated(self, simple_network, temp_snapshot_file):
        """Test output weights shape is validated."""
        serializer = CascadeHDF5Serializer()
        success = serializer.save_network(simple_network, temp_snapshot_file)
        assert success  # trunk-ignore(bandit/B101)

        loaded_network = serializer.load_network(temp_snapshot_file)
        assert loaded_network is not None  # trunk-ignore(bandit/B101)

        # Shape validation is logged, check that network loaded successfully
        expected_shape = (
            simple_network.input_size + len(simple_network.hidden_units),
            simple_network.output_size,
        )
        assert (
            loaded_network.output_weights.shape == expected_shape
        )  # trunk-ignore(bandit/B101)

    def test_hidden_units_shape_validated(
        self, network_with_hidden_units, temp_snapshot_file
    ):
        """Test hidden units shapes are validated."""
        serializer = CascadeHDF5Serializer()
        success = serializer.save_network(network_with_hidden_units, temp_snapshot_file)
        assert success  # trunk-ignore(bandit/B101)

        loaded_network = serializer.load_network(temp_snapshot_file)
        assert loaded_network is not None  # trunk-ignore(bandit/B101)

        # Verify shapes of hidden units
        for i, unit in enumerate(loaded_network.hidden_units):
            expected_input_size = loaded_network.input_size + i
            assert (
                unit["weights"].shape[0] == expected_input_size
            ), f"Hidden unit {i} has wrong weight shape"  # trunk-ignore(bandit/B101)


class TestFormatValidation:
    """Test that file format is properly validated."""

    def test_valid_format_accepted(self, simple_network, temp_snapshot_file):
        """Test that valid format is accepted."""
        serializer = CascadeHDF5Serializer()
        success = serializer.save_network(simple_network, temp_snapshot_file)
        assert success  # trunk-ignore(bandit/B101)

        loaded_network = serializer.load_network(temp_snapshot_file)
        assert loaded_network is not None  # trunk-ignore(bandit/B101)

    def test_format_verification_info(self, simple_network, temp_snapshot_file):
        """Test verify_saved_network provides correct information."""
        serializer = CascadeHDF5Serializer()
        success = serializer.save_network(simple_network, temp_snapshot_file)
        assert success  # trunk-ignore(bandit/B101)

        verification = serializer.verify_saved_network(temp_snapshot_file)

        assert verification["valid"] is True  # trunk-ignore(bandit/B101)
        assert "format" in verification  # trunk-ignore(bandit/B101)
        assert "input_size" in verification  # trunk-ignore(bandit/B101)
        assert "output_size" in verification  # trunk-ignore(bandit/B101)
        assert (
            verification["input_size"] == simple_network.input_size
        )  # trunk-ignore(bandit/B101)
        assert (
            verification["output_size"] == simple_network.output_size
        )  # trunk-ignore(bandit/B101)


class TestRandomStateRestoration:
    """Test that random state restoration enables deterministic training."""

    def test_python_random_state_restoration(self, simple_network, temp_snapshot_file):
        """Test that Python random state is restored for deterministic behavior."""
        import random

        # Set initial seed and advance RNG
        random.seed(42)
        for _ in range(10):  # sourcery skip: no-loop-in-tests
            random.random()  # trunk-ignore(bandit/B311)

        serializer = self._serialize_and_save_network(
            simple_network, temp_snapshot_file
        )
        # Generate sequence from current state
        first_sequence = [
            random.random() for _ in range(5)
        ]  # trunk-ignore(bandit/B311)
        assert first_sequence is not None  # trunk-ignore(bandit/B101)

        second_sequence = self._load_and_validate_network_helper(
            serializer, temp_snapshot_file, random
        )
        third_sequence = self._load_and_validate_network_helper(
            serializer, temp_snapshot_file, random
        )
        # Second and third sequences should match (proving deterministic restoration)
        assert (
            second_sequence == third_sequence
        ), "Python random state restoration is not deterministic"  # trunk-ignore(bandit/B101)

    # def _load_and_validate_network_helper(self, serializer, temp_snapshot_file, random):
    #     # Load network (should restore RNG to same state as when saved)
    #     # Note: Network initialization during load will affect global state,
    #     # but we verify restoration by loading again and comparing sequences
    #     loaded_network = serializer.load_network(temp_snapshot_file)
    #     assert loaded_network is not None
    #     return [random.random() for _ in range(5)]

    def test_numpy_random_state_restoration(self, simple_network, temp_snapshot_file):
        """Test that NumPy random state is restored for deterministic behavior."""
        import numpy as np

        # Set initial seed and advance RNG
        np.random.seed(42)
        for _ in range(10):
            np.random.rand()

        serializer = self._serialize_and_save_network(
            simple_network, temp_snapshot_file
        )
        # Generate sequence from current state
        first_sequence = [np.random.rand() for _ in range(5)]
        assert first_sequence is not None  # trunk-ignore(bandit/B101)

        second_sequence = self._load_and_validate_network_helper(
            serializer, temp_snapshot_file, np
        )
        third_sequence = self._load_and_validate_network_helper(
            serializer, temp_snapshot_file, np
        )
        # Second and third sequences should match (proving deterministic restoration)
        assert np.allclose(
            second_sequence, third_sequence
        ), "NumPy random state restoration is not deterministic"  # trunk-ignore(bandit/B101)

    # def _load_and_validate_network_helper(self, serializer, temp_snapshot_file, np):
    #     # Load network (should restore RNG to same state as when saved)
    #     loaded_network = serializer.load_network(temp_snapshot_file)
    #     assert loaded_network is not None
    #     return [np.random.rand() for _ in range(5)]

    def test_torch_random_state_restoration(self, simple_network, temp_snapshot_file):
        """Test that PyTorch random state is restored for deterministic behavior."""
        import torch

        # Set initial seed and advance RNG
        torch.manual_seed(42)
        for _ in range(10):  # sourcery skip: no-loop-in-tests
            torch.rand(1)

        serializer = self._serialize_and_save_network(
            simple_network, temp_snapshot_file
        )
        # Generate sequence from current state
        first_sequence = [torch.rand(1).item() for _ in range(5)]
        assert first_sequence is not None  # trunk-ignore(bandit/B101)

        second_sequence = self._load_and_validate_network_helper(
            serializer, temp_snapshot_file, torch
        )
        third_sequence = self._load_and_validate_network_helper(
            serializer, temp_snapshot_file, torch
        )

        # Second and third sequences should match (proving deterministic restoration)
        assert torch.allclose(
            torch.tensor(second_sequence), torch.tensor(third_sequence)
        ), "PyTorch random state restoration is not deterministic"  # trunk-ignore(bandit/B101)

    def _serialize_and_save_network(self, simple_network, temp_snapshot_file):
        result = CascadeHDF5Serializer()
        success = result.save_network(simple_network, temp_snapshot_file)
        assert success  # trunk-ignore(bandit/B101)
        return result

    def _load_and_validate_network_helper(
        self, serializer, temp_snapshot_file, rng_module
    ):
        """
        Helper to load network and generate random sequence using specified RNG module.

        Args:
            serializer: CascadeHDF5Serializer instance
            temp_snapshot_file: Path to snapshot file
            rng_module: Random number generator module (random, numpy, or torch)

        Returns:
            List of 5 random values from the specified module
        """
        # Load network (should restore RNG to same state as when saved)
        loaded_network = serializer.load_network(temp_snapshot_file)
        assert loaded_network is not None  # trunk-ignore(bandit/B101)

        # Generate sequence using the correct module
        if rng_module.__name__ == "random":
            return [rng_module.random() for _ in range(5)]
        elif rng_module.__name__ == "numpy":
            return [rng_module.random.rand() for _ in range(5)]
        elif rng_module.__name__ == "torch":
            return [rng_module.rand(1).item() for _ in range(5)]
        else:
            raise ValueError(f"Unsupported RNG module: {rng_module.__name__}")

    def test_deterministic_training_resume(self, simple_network, temp_snapshot_file):
        """Test that training can be paused, saved, loaded, and resumed deterministically."""
        import torch

        # Create simple XOR-like dataset for training
        X_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

        # Set seed for reproducibility
        torch.manual_seed(42)

        # Train network for N=5 epochs
        N = 5
        for _ in range(N):  # sourcery skip: no-loop-in-tests
            simple_network.train_output_layer(X_train, y_train, epochs=1)

        # Capture weights after N epochs
        W_N = simple_network.output_weights.clone()

        # Save network
        serializer = CascadeHDF5Serializer()
        success = serializer.save_network(
            simple_network, temp_snapshot_file, include_training_state=True
        )
        assert success  # trunk-ignore(bandit/B101)

        # Load network
        loaded_network = serializer.load_network(temp_snapshot_file)
        assert loaded_network is not None  # trunk-ignore(bandit/B101)

        # Verify loaded weights match
        W_N_loaded = loaded_network.output_weights.clone()
        assert torch.allclose(
            W_N, W_N_loaded
        ), "Loaded weights don't match saved weights"  # trunk-ignore(bandit/B101)

        # Continue training loaded network for M=5 more epochs
        M = 5
        for _ in range(M):  # sourcery skip: no-loop-in-tests
            loaded_network.train_output_layer(X_train, y_train, epochs=1)

        # Capture final weights from resumed training
        W_N_M_resumed = loaded_network.output_weights.clone()

        # Now train a fresh network continuously for N+M epochs
        torch.manual_seed(42)
        fresh_network = CascadeCorrelationNetwork(config=simple_network.config)

        for _ in range(N + M):  # sourcery skip: no-loop-in-tests
            fresh_network.train_output_layer(X_train, y_train, epochs=1)

        # Capture weights from continuous training
        W_N_M_fresh = fresh_network.output_weights.clone()

        # Verify that resumed training matches continuous training
        assert torch.allclose(
            W_N_M_resumed, W_N_M_fresh, atol=1e-5
        ), "Resumed training does not match continuous training - training is not deterministic"  # trunk-ignore(bandit/B101)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--integration"])
