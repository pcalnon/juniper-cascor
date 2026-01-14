#!/usr/bin/env python
"""
Project:       Juniper
Prototype:     Cascade Correlation Neural Network  
File Name:     test_spiral_problem.py
Author:        Paul Calnon
Version:       0.1.0

Date:          2025-09-26
Last Modified: 2025-09-26

License:       MIT License
Copyright:     Copyright (c) 2024-2025 Paul Calnon

Description:
    Integration tests for spiral problem solving with Cascade Correlation Network.
    Tests the network's ability to solve the classic N-spiral classification problem.
"""

import pytest
import torch
# import numpy as np
from helpers.assertions import (
    assert_network_structure_valid, assert_training_history_valid,
    assert_accuracy_valid, assert_network_learns
)
from helpers.utilities import set_deterministic_behavior, measure_training_time
from test_data.generators import SpiralDataGenerator   # sourcery skip: dont-import-test-modules


class TestSpiralProblemBasic:
    """Test basic spiral problem solving capability."""

    @pytest.mark.integration
    @pytest.mark.spiral
    @pytest.mark.slow
    def test_2_spiral_learning(self, spiral_network):
        """Test that network can learn 2-spiral problem."""
        set_deterministic_behavior(42)

        # Generate 2-spiral data
        x, y, info = SpiralDataGenerator.generate_2_spiral(
            n_per_spiral=50,
            noise=0.05,
            seed=42
        )

        # Train network
        initial_accuracy = spiral_network.calculate_accuracy(x, y)
        history, elapsed_time = measure_training_time(
            spiral_network, x, y, max_epochs=10, early_stopping=False
        )
        final_accuracy = spiral_network.calculate_accuracy(x, y)
        # Verify learning occurred
        assert_network_learns(
            initial_loss=history['train_loss'][0],
            final_loss=history['train_loss'][-1],
            min_improvement=0.01
        )

        # Verify accuracy improved
        assert final_accuracy > initial_accuracy  # trunk-ignore(bandit/B101)
        # Should achieve reasonable accuracy
        assert final_accuracy > 0.6               # trunk-ignore(bandit/B101)

        # Verify network structure
        assert_network_structure_valid(spiral_network)
        assert_training_history_valid(history)

        print(f"2-spiral results: accuracy {initial_accuracy:.3f} -> {final_accuracy:.3f} in {elapsed_time:.2f}s")

    @pytest.mark.integration
    @pytest.mark.spiral
    def test_3_spiral_learning(self, spiral_network):
        """Test that network can learn 3-spiral problem."""
        set_deterministic_behavior(42)

        # Generate 3-spiral data
        x, y, info = SpiralDataGenerator.generate_n_spiral(
            n_spirals=3,
            n_per_spiral=40,
            noise=0.03,
            seed=42
        )

        # Adjust network for 3-class problem
        spiral_network.output_size = 3
        spiral_network.output_weights = torch.randn(
            spiral_network.input_size, 3, requires_grad=True
        ) * spiral_network.random_value_scale
        spiral_network.output_bias = torch.randn(3, requires_grad=True) * spiral_network.random_value_scale

        # Train network
        initial_accuracy = spiral_network.calculate_accuracy(x, y)
        history = spiral_network.fit(x, y, max_epochs=8)
        final_accuracy = spiral_network.calculate_accuracy(x, y)

        # Verify learning
        assert final_accuracy > initial_accuracy  # trunk-ignore(bandit/B101)
        # 3-class problem is harder
        assert final_accuracy > 0.4               # trunk-ignore(bandit/B101)

        assert_network_structure_valid(spiral_network)
        assert_training_history_valid(history)

        print(f"3-spiral results: accuracy {initial_accuracy:.3f} -> {final_accuracy:.3f}, history length: {len(history['train_loss'])}")


class TestSpiralProblemProgressive:
    """Test progressive difficulty in spiral problems."""

    @pytest.mark.integration
    @pytest.mark.spiral
    @pytest.mark.slow
    @pytest.mark.parametrize("n_spirals", [2, 3, 4])
    def test_n_spiral_difficulty_progression(self, n_spirals):
        """Test that network can handle increasing spiral complexity."""
        set_deterministic_behavior(42)

        # Create network configured for n-spiral problem
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        network = CascadeCorrelationNetwork.create_simple_network(
            input_size=2,
            output_size=n_spirals,
            learning_rate=0.05,
            max_hidden_units=min(8, n_spirals * 2),
            candidate_pool_size=12,
            correlation_threshold=0.15
        )

        # Generate n-spiral data
        x, y, info = SpiralDataGenerator.generate_n_spiral(
            n_spirals=n_spirals,
            n_per_spiral=30,
            noise=0.02,
            seed=42
        )

        # Train network
        initial_accuracy = network.calculate_accuracy(x, y)
        history = network.fit(x, y, max_epochs=6)
        final_accuracy = network.calculate_accuracy(x, y)

        # Basic learning verification
        assert final_accuracy >= initial_accuracy # trunk-ignore(bandit/B101)

        # Expected accuracy should decrease with complexity but still be above random
        random_accuracy = 1.0 / n_spirals
        # At least 20% better than random
        assert final_accuracy > random_accuracy * 1.2  # trunk-ignore(bandit/B101)

        assert_network_structure_valid(network)

        print(f"{n_spirals}-spiral: {initial_accuracy:.3f} -> {final_accuracy:.3f} (random: {random_accuracy:.3f}), history length: {len(history['train_loss'])}")


class TestSpiralProblemRobustness:
    """Test robustness of spiral problem solving."""

    @pytest.mark.integration
    @pytest.mark.spiral
    @pytest.mark.parametrize("noise_level", [0.01, 0.05, 0.1])
    def test_spiral_noise_robustness(self, spiral_network, noise_level):
        """Test network performance with different noise levels."""
        set_deterministic_behavior(42)

        # Generate noisy spiral data
        x, y, info = SpiralDataGenerator.generate_2_spiral(
            n_per_spiral=40,
            noise=noise_level,
            seed=42
        )

        # Train network
        initial_accuracy = spiral_network.calculate_accuracy(x, y)
        history = spiral_network.fit(x, y, max_epochs=6)
        final_accuracy = spiral_network.calculate_accuracy(x, y)

        # Should learn even with noise, though performance may degrade
        assert final_accuracy >= initial_accuracy # trunk-ignore(bandit/B101)

        # Performance should still be better than random (0.5 for 2-class)
        if noise_level <= 0.05:  # sourcery skip: no-conditionals-in-tests
            # Should handle small noise well
            assert final_accuracy > 0.55  # trunk-ignore(bandit/B101)
        else:
            # Should at least beat random with high noise
            assert final_accuracy > 0.51  # trunk-ignore(bandit/B101)

        print(f"Noise {noise_level}: {initial_accuracy:.3f} -> {final_accuracy:.3f}, history length: {len(history['train_loss'])}")

    @pytest.mark.integration
    @pytest.mark.spiral
    @pytest.mark.parametrize("n_per_spiral", [20, 50, 100])
    def test_spiral_data_size_scaling(self, spiral_network, n_per_spiral):
        """Test network performance with different dataset sizes."""
        set_deterministic_behavior(42)

        # Generate spiral data of different sizes
        x, y, info = SpiralDataGenerator.generate_2_spiral(
            n_per_spiral=n_per_spiral,
            noise=0.03,
            seed=42
        )

        # Train network
        initial_accuracy = spiral_network.calculate_accuracy(x, y)
        history = spiral_network.fit(x, y, max_epochs=5)
        final_accuracy = spiral_network.calculate_accuracy(x, y)

        # Should learn regardless of data size
        assert final_accuracy > initial_accuracy # trunk-ignore(bandit/B101)
        assert final_accuracy > 0.5 # trunk-ignore(bandit/B101)

        # More data should generally lead to better or equal performance
        assert_network_structure_valid(spiral_network)

        print(f"Data size {n_per_spiral * 2}: {initial_accuracy:.3f} -> {final_accuracy:.3f}, history length: {len(history['train_loss'])}")


class TestSpiralProblemVisualization:
    """Test spiral problem with visualization and analysis."""

    @pytest.mark.integration
    @pytest.mark.spiral
    def test_spiral_training_progression(self, spiral_network):
        """Test and analyze training progression on spiral problem."""
        set_deterministic_behavior(42)

        # Generate spiral data
        x, y, info = SpiralDataGenerator.generate_2_spiral(
            n_per_spiral=60,
            noise=0.04,
            seed=42
        )

        # Split into train/validation
        n_train = int(0.8 * len(x))
        indices = torch.randperm(len(x))
        train_indices, val_indices = indices[:n_train], indices[n_train:]

        x_train, y_train = x[train_indices], y[train_indices]
        x_val, y_val = x[val_indices], y[val_indices]

        # Train with validation
        history = spiral_network.fit(
            x_train, y_train, 
            x_val, y_val,
            max_epochs=8,
            early_stopping=True
        )

        # Analyze training progression
        assert len(history['train_loss']) > 0 # trunk-ignore(bandit/B101)
        assert len(history['train_accuracy']) > 0 # trunk-ignore(bandit/B101)

        if 'value_loss' in history and history['value_loss']:  # sourcery skip: no-conditionals-in-tests
            assert len(history['value_loss']) > 0 # trunk-ignore(bandit/B101)
            assert len(history['value_accuracy']) > 0 # trunk-ignore(bandit/B101)

            # Validation accuracy should be reasonable
            final_val_accuracy = history['value_accuracy'][-1]
            assert final_val_accuracy > 0.5 # trunk-ignore(bandit/B101)

        # Final training accuracy should be good
        final_train_accuracy = history['train_accuracy'][-1]
        assert final_train_accuracy > 0.6 # trunk-ignore(bandit/B101)

        # Test generalization
        final_val_accuracy = spiral_network.calculate_accuracy(x_val, y_val)
        assert final_val_accuracy > 0.5 # trunk-ignore(bandit/B101)

        print(f"Spiral training: final train acc = {final_train_accuracy:.3f}, val acc = {final_val_accuracy:.3f}")
        print(f"Hidden units added: {len(spiral_network.hidden_units)}")


class TestSpiralProblemEdgeCases:
    """Test edge cases in spiral problem solving."""

    @pytest.mark.integration
    @pytest.mark.spiral
    def test_minimal_spiral_data(self, spiral_network):
        """Test network with minimal spiral data."""
        set_deterministic_behavior(42)

        # Very small dataset
        x, y, info = SpiralDataGenerator.generate_2_spiral(
            n_per_spiral=5,  # Very small
            noise=0.01,
            seed=42
        )

        # Should handle gracefully without crashing
        try:
            history = spiral_network.fit(x, y, max_epochs=3)
            final_accuracy = spiral_network.calculate_accuracy(x, y)

            # May not learn well with tiny dataset, but should not crash
            assert_accuracy_valid(final_accuracy)
            assert_network_structure_valid(spiral_network)
        except Exception as e:
            # If it fails, should be a reasonable error
            assert "empty" not in str(e).lower() or "size" in str(e).lower() # trunk-ignore(bandit/B101)
        print(f"Minimal data test completed, final accuracy: {final_accuracy:.3f}, history length: {len(history['train_loss'])}")

    @pytest.mark.integration 
    @pytest.mark.spiral
    def test_perfect_spiral_separation(self, spiral_network):
        """Test network with perfectly separated spiral data."""
        set_deterministic_behavior(42)

        # Generate data with no noise (perfect separation)
        x, y, info = SpiralDataGenerator.generate_2_spiral(
            n_per_spiral=30,
            noise=0.0,  # No noise - perfect separation
            seed=42
        )

        # Train network
        initial_accuracy = spiral_network.calculate_accuracy(x, y)
        history = spiral_network.fit(x, y, max_epochs=8)
        final_accuracy = spiral_network.calculate_accuracy(x, y)

        # Should achieve very high accuracy with perfect data
        assert final_accuracy > initial_accuracy  # trunk-ignore(bandit/B101)
        # Should achieve high accuracy
        assert final_accuracy > 0.8  # trunk-ignore(bandit/B101)

        # May achieve perfect accuracy
        if final_accuracy > 0.95:  # sourcery skip: no-conditionals-in-tests
            print("Achieved near-perfect accuracy on clean spiral data, impressive!")

        assert_network_structure_valid(spiral_network)
        print(f"Perfect spiral: {initial_accuracy:.3f} -> {final_accuracy:.3f}, history length: {len(history['train_loss'])}")


class TestSpiralProblemComparison:
    """Compare performance across different spiral configurations."""

    @pytest.mark.integration
    @pytest.mark.spiral
    @pytest.mark.slow
    def test_spiral_configuration_comparison(self):
        """Compare network performance across different spiral configurations."""
        set_deterministic_behavior(42)

        configurations = [
            {"n_spirals": 2, "noise": 0.02, "n_per_spiral": 40},
            {"n_spirals": 2, "noise": 0.08, "n_per_spiral": 40},
            {"n_spirals": 3, "noise": 0.02, "n_per_spiral": 30},
            {"n_spirals": 3, "noise": 0.05, "n_per_spiral": 30},
        ]

        results = []

        for config in configurations:  # sourcery skip: no-loop-in-tests
            # Create fresh network for each test
            from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
            network = CascadeCorrelationNetwork.create_simple_network(
                input_size=2,
                output_size=config["n_spirals"],
                learning_rate=0.04,
                max_hidden_units=6
            )

            # Generate data
            if config["n_spirals"] == 2:  # sourcery skip: no-conditionals-in-tests
                x, y, info = SpiralDataGenerator.generate_2_spiral(
                    n_per_spiral=config["n_per_spiral"],
                    noise=config["noise"],
                    seed=42
                )
            else:
                x, y, info = SpiralDataGenerator.generate_n_spiral(
                    n_spirals=config["n_spirals"],
                    n_per_spiral=config["n_per_spiral"],
                    noise=config["noise"],
                    seed=42
                )

            # Train and measure
            initial_accuracy = network.calculate_accuracy(x, y)
            history, elapsed_time = measure_training_time(
                network, x, y, max_epochs=6
            )
            final_accuracy = network.calculate_accuracy(x, y)

            result = {
                "config": config,
                "initial_accuracy": initial_accuracy,
                "final_accuracy": final_accuracy,
                "improvement": final_accuracy - initial_accuracy,
                "hidden_units": len(network.hidden_units),
                "time": elapsed_time
            }
            results.append(result)

            # Basic validation
            assert final_accuracy >= initial_accuracy # trunk-ignore(bandit/B101)
            assert_network_structure_valid(network)

        # Print comparison
        print("\nSpiral Problem Comparison:")
        print("Config\t\tInitial\tFinal\tImprovement\tUnits\tTime")
        for result in results:  # sourcery skip: no-loop-in-tests
            config = result["config"]
            print(
                f"{config['n_spirals']}-spiral, noise={config['noise']}\t"
                f"{result['initial_accuracy']:.3f}\t"
                f"{result['final_accuracy']:.3f}\t"
                f"{result['improvement']:.3f}\t\t"
                f"{result['hidden_units']}\t"
                f"{result['time']:.2f}s"
            )

        # All configurations should achieve some learning
        for result in results:  # sourcery skip: no-loop-in-tests
            # Allow small regression due to randomness
            assert result["improvement"] >= -0.05  # trunk-ignore(bandit/B101)
            # Better than random
            assert result["final_accuracy"] > 1.0 / result["config"]["n_spirals"]  # trunk-ignore(bandit/B101)
