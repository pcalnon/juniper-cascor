#!/usr/bin/env python
"""
Final test to verify the main issue is resolved.
Fixed per CRIT-004 to use proper assertions instead of return values.
"""

import os
import sys

# Use relative path instead of hardcoded absolute path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import torch

from candidate_unit.candidate_unit import CandidateUnit


def _get_test_epochs():
    """Get training epochs based on fast-slow mode."""
    if os.environ.get("JUNIPER_FAST_SLOW", "0") == "1":
        return 2  # Fast mode: minimal epochs
    return 5  # Normal mode


# CASCOR-TIMEOUT-001: Added slow marker and extended timeout
@pytest.mark.slow
@pytest.mark.timeout(300)
def test_candidate_units_simple(fast_training_params):
    """Test the core issue: different candidates should have different correlations."""
    # Use fast_training_params for epochs in fast-slow mode
    test_epochs = min(5, fast_training_params.get("candidate_epochs", 5))

    # Create test data
    torch.manual_seed(42)
    x = torch.randn(16, 2)
    residual_error = torch.randn(16, 1)

    # Create 3 candidates with different indices
    candidates = []
    correlations = []

    for i in range(3):  # sourcery skip: no-loop-in-tests
        candidate = CandidateUnit(
            CandidateUnit__input_size=2,
            CandidateUnit__candidate_index=i,
            CandidateUnit__epochs=test_epochs,
            CandidateUnit__learning_rate=0.01,
            CandidateUnit__log_level_name="ERROR",  # Minimal logging
            CandidateUnit__display_frequency=10000,  # No display to avoid errors
        )

        # Train the candidate
        final_correlation = candidate.train(
            x=x,
            epochs=test_epochs,
            residual_error=residual_error,
            learning_rate=0.01,
            display_frequency=10000,  # No display to avoid errors
        )

        # Verify the correlation is set properly (checked in consistent_correlations assertion)
        candidates.append(candidate)
        correlations.append(final_correlation)

    # Success criteria - using proper assertions per CRIT-004
    different_correlations = len(set(correlations)) > 1
    non_zero_correlations = all(c != 0.0 for c in correlations)
    consistent_correlations = all(abs(correlations[i] - candidates[i].get_correlation()) < 1e-6 for i in range(3))

    assert different_correlations, "Candidates should have different correlations"
    assert non_zero_correlations, "All correlations should be non-zero"
    assert consistent_correlations, "Returned correlations should match instance correlations"
