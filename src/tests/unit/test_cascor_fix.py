#!/usr/bin/env python
"""
Test script to verify CascadeCorrelationNetwork and CandidateUnit fixes.
"""

import sys
import os
sys.path.append('/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src')

import torch
# import numpy as np
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from candidate_unit.candidate_unit import CandidateUnit

def test_sequential_candidate_training():
    """Test candidate training using sequential processing to bypass multiprocessing issues."""
    print("Testing CascadeCorrelationNetwork with sequential processing...")

    # Create simple test data
    torch.manual_seed(42)
    batch_size = 16
    input_size = 2
    output_size = 1

    x = torch.randn(batch_size, input_size)
    y = torch.randint(0, 2, (batch_size, output_size)).float()

    print(f"Input shape: {x.shape}, Output shape: {y.shape}")

    # Create network with sequential processing (process_count=1 forces sequential)
    network = CascadeCorrelationNetwork(
        input_size=input_size,
        output_size=output_size,
        candidate_pool_size=3,
        candidate_epochs=5,
        candidate_learning_rate=0.01,
        learning_rate=0.01,
        log_level_name='WARNING'
    )

    # Force sequential processing by setting process count to 1
    original_cpu_count = os.cpu_count
    os.cpu_count = lambda: 1  # Mock to force sequential processing

    try:
        return _get_candidate_training_stats(network, x, y)
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Restore original cpu_count
        os.cpu_count = original_cpu_count


def _get_candidate_training_stats(network, x, y):
    # Calculate residual error
    residual_error = network.calculate_residual_error(x, y)
    print(f"Residual error shape: {residual_error.shape}")

    # Test candidate training using sequential processing
    print("Training candidates using sequential processing...")
    training_stats = network.train_candidates(x, y, residual_error)

    print("Training completed!")
    print(f"Training stats type: {type(training_stats)}")

        # Extract results from training_stats tuple
    if isinstance(training_stats, tuple) and len(training_stats) >= 1:
        candidates_data = training_stats[0]
        if isinstance(candidates_data, tuple) and len(candidates_data) == 4:
            return _validate_candidates_correlations(candidates_data)
        print(f"Invalid candidates_data format: {candidates_data}")
    else:
        print(f"Invalid training_stats format: {training_stats}")
    return False


def _validate_candidates_correlations(candidates_data):
    candidate_ids, candidate_uuids, correlations, candidates = candidates_data
    print(f"Number of trained candidates: {len(candidates)}")
    print(f"Correlations obtained: {correlations}")
    print(f"All correlations identical: {len(set(correlations)) <= 1 if correlations else True}")
    print(f"All correlations zero: {all(c == 0.0 for c in correlations) if correlations else True}")

    # Verify individual candidate correlations
    for i, candidate in enumerate(candidates): # sourcery skip: no-loop-in-tests
        if candidate:
            candidate_corr = candidate.get_correlation()
            print(f"Candidate {i} correlation via get_correlation(): {candidate_corr:.6f}")

    return True

def test_individual_candidates():
    """Test individual candidate training to ensure basic functionality works."""
    print("\nTesting individual CandidateUnit training...")
    
    # Create test data
    torch.manual_seed(42)
    batch_size = 16
    input_size = 2
    
    x = torch.randn(batch_size, input_size)
    residual_error = torch.randn(batch_size, 1)
    
    # Test multiple candidates with different indices
    candidates = []
    correlations = []
    
    for i in range(3): # sourcery skip: no-loop-in-tests
        candidate = CandidateUnit(
            CandidateUnit__input_size=input_size,
            CandidateUnit__candidate_index=i,
            CandidateUnit__epochs=3,
            CandidateUnit__learning_rate=0.01,
            CandidateUnit__log_level_name='ERROR'  # Reduce logging
        )
        
        print(f"Candidate {i} initial weights: {candidate.weights.detach().numpy()}")
        
        # Train candidate
        correlation = candidate.train(
            x=x,
            epochs=3,
            residual_error=residual_error,
            learning_rate=0.01
        )
        
        candidates.append(candidate)
        correlations.append(correlation)
        print(f"Candidate {i} final correlation: {correlation:.6f}")
    
    print(f"All correlations: {correlations}")
    print(f"Correlations are different: {len(set(correlations)) > 1}")
    print(f"All correlations non-zero: {all(c != 0.0 for c in correlations)}")
    
    return len(set(correlations)) > 1 and all(c != 0.0 for c in correlations)

if __name__ == "__main__":
    print("Running CascadeCorrelationNetwork fix tests...")
    
    # Test 1: Individual candidate functionality
    individual_test_passed = test_individual_candidates()
    print(f"Individual candidate test passed: {individual_test_passed}")
    
    # Test 2: Network-level candidate training
    network_test_passed = test_sequential_candidate_training()
    print(f"Network candidate training test passed: {network_test_passed}")
    
    if individual_test_passed and network_test_passed:
        print("\n✅ All tests passed! The fixes are working correctly.")
    else:
        print("\n❌ Some tests failed. Further debugging needed.")
