#!/usr/bin/env python
"""
Unit tests for candidate seed diversity.

Tests focus on:
- Candidates in a pool have different random seeds (CASCOR-P0-005)
- Different candidates produce different initial weights
- Seed diversity enables exploration of weight space
"""

import os
import sys

import pytest
import torch

# Add parent directories for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from candidate_unit.candidate_unit import CandidateUnit
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit


class TestCandidateSeedDiversity:
    """Tests for verifying candidate seed diversity."""

    @pytest.fixture
    def basic_config(self):
        """Create a basic configuration for testing."""
        return CascadeCorrelationConfig(
            input_size=2,
            output_size=2,
            candidate_pool_size=8,
            random_seed=42,
        )

    @pytest.fixture
    def network(self, basic_config):
        """Create a network instance."""
        return CascadeCorrelationNetwork(config=basic_config)

    def test_candidates_have_different_seeds(self, network):
        """Test that candidates in pool are initialized with different seeds.

        Drives the real derivation instead of re-implementing it. This test used to
        reproduce `random.seed(...)` + `random.randint(...)` inline and assert on its own
        local copy, which said nothing about the network at all -- and said nothing at all
        once the derivation moved to a network-owned generator (juniper-cascor#532). A test
        that reproduces the code under test is how that regression returns silently.
        """
        # _generate_candidate_tasks reads only candidate_input.shape[1], so a 2-wide tensor
        # matches this fixture's input_size.
        candidate_input = torch.zeros((6, 2), dtype=torch.float32)
        try:
            tasks = network._generate_candidate_tasks(candidate_input, candidate_input.clone(), candidate_input.clone())
            # Task payload is (index, input_size, activation_name, random_value_scale, uuid,
            # SEED, random_max_value, sequence_max_value); the seed is element 5.
            candidate_seeds = [task[1][5] for task in tasks]
        finally:
            for blk in list(getattr(network, "_active_shm_blocks", [])):
                try:
                    blk.close_and_unlink()
                except Exception:  # nosec B110 -- test cleanup must not mask the assertion
                    pass
            if hasattr(network, "_active_shm_blocks"):
                network._active_shm_blocks.clear()

        assert len(candidate_seeds) == network.candidate_pool_size
        # All seeds should be unique (very high probability with large random_max_value)
        assert len(candidate_seeds) == len(set(candidate_seeds)), "All candidates should have unique seeds"

    def test_candidates_have_different_initial_weights(self):
        """Test that different candidates have different initial weights."""
        # Create 3 candidates with different seeds (keep it fast)
        candidates = []
        for i in range(3):
            candidate = CandidateUnit(
                CandidateUnit__input_size=2,
                CandidateUnit__random_seed=42 + i * 100,  # Different seeds
            )
            candidates.append(candidate)

        # Compare weights between candidates
        weights_list = [c.weights.detach().clone() for c in candidates]

        for i in range(len(weights_list)):
            for j in range(i + 1, len(weights_list)):
                # Weights should not be identical
                assert not torch.allclose(weights_list[i], weights_list[j], atol=1e-6), f"Candidates {i} and {j} have identical weights"

    def test_same_seed_produces_same_weights(self):
        """Test reproducibility: same seed produces same initial weights."""
        config1 = CascadeCorrelationConfig(
            input_size=2,
            output_size=2,
            random_seed=42,
        )
        config2 = CascadeCorrelationConfig(
            input_size=2,
            output_size=2,
            random_seed=42,
        )

        candidate1 = CandidateUnit(
            CandidateUnit__input_size=2,
            CandidateUnit__random_seed=42,
        )
        candidate2 = CandidateUnit(
            CandidateUnit__input_size=2,
            CandidateUnit__random_seed=42,
        )

        # Same seed should produce similar weights (within tolerance)
        assert torch.allclose(candidate1.weights, candidate2.weights, atol=1e-6), "Same seed should produce identical weights"

    def test_different_seeds_produce_different_weights(self):
        """Test that different seeds produce different initial weights."""
        candidate1 = CandidateUnit(
            CandidateUnit__input_size=2,
            CandidateUnit__random_seed=42,
        )
        candidate2 = CandidateUnit(
            CandidateUnit__input_size=2,
            CandidateUnit__random_seed=123,
        )

        # Different seeds should produce different weights
        assert not torch.allclose(candidate1.weights, candidate2.weights, atol=1e-6), "Different seeds should produce different weights"
