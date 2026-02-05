#!/usr/bin/env python
"""
Quick test to verify the cascor fixes work end-to-end.
Converted to proper pytest format per CRIT-002.
"""

import os
import sys

# Use relative path instead of hardcoded absolute path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import torch

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork


@pytest.mark.unit
@pytest.mark.slow
def test_cascor_candidate_training():
    """Test CascadeCorrelationNetwork candidate training produces different correlations."""
    # Create minimal test data
    torch.manual_seed(42)
    x = torch.randn(8, 2)  # Small batch
    y = torch.randint(0, 2, (8, 1)).float()

    # Create network with very simple config
    network = CascadeCorrelationNetwork(
        input_size=2,
        output_size=1,
        candidate_pool_size=2,  # Just 2 candidates
        candidate_epochs=2,  # Just 2 epochs
        candidate_learning_rate=0.01,
        learning_rate=0.01,
        log_level_name="ERROR",  # Minimal logging
    )

    # Force sequential processing to avoid multiprocessing issues
    original_cpu_count = os.cpu_count
    os.cpu_count = lambda: 1

    try:
        # Calculate residual error
        residual_error = network.calculate_residual_error(x, y)
        assert residual_error.shape == y.shape, f"Expected residual error shape {y.shape}, got {residual_error.shape}"

        # Train candidates with sequential processing
        results = network.train_candidates(x, y, residual_error)

        assert results is not None, "train_candidates should return results"

        # train_candidates returns a TrainingResults dataclass
        from cascade_correlation.cascade_correlation import TrainingResults

        assert isinstance(results, TrainingResults), f"Results should be TrainingResults, got {type(results)}"

        assert len(results.candidate_objects) > 0, "Should have at least one candidate"
        assert len(results.correlations) == len(results.candidate_objects), "Correlations should match candidates"
        assert len(results.candidate_ids) == len(results.candidate_objects), "Candidate IDs should match candidates"
        assert len(results.candidate_uuids) == len(results.candidate_objects), "Candidate UUIDs should match candidates"

        # Verify candidates have different correlations
        assert len(set(results.correlations)) > 1, "Candidates should have different correlations"

    finally:
        os.cpu_count = original_cpu_count
