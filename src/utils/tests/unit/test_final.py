#!/usr/bin/env python
"""
Final test to verify the main issue is resolved.
"""

import sys
# import os
sys.path.append('/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src')

import torch
from candidate_unit.candidate_unit import CandidateUnit

def test_candidate_units_simple():
    """Test the core issue: different candidates should have different correlations."""
    print("Testing core CandidateUnit training issue...")
    
    # Create test data  
    torch.manual_seed(42)
    x = torch.randn(16, 2)
    residual_error = torch.randn(16, 1)
    
    print(f"Input shape: {x.shape}, Residual error shape: {residual_error.shape}")
    
    # Create 3 candidates with different indices
    candidates = []
    correlations = []
    
    for i in range(3): # sourcery skip: no-loop-in-tests
        print(f"\nTraining candidate {i}...")
        
        candidate = CandidateUnit(
            CandidateUnit__input_size=2,
            CandidateUnit__candidate_index=i,
            CandidateUnit__epochs=5,
            CandidateUnit__learning_rate=0.01,
            CandidateUnit__log_level_name='ERROR',  # Minimal logging
            CandidateUnit__display_frequency=10000  # No display to avoid errors
        )
        
        print(f"  Initial weights: {candidate.weights.detach().numpy()}")
        print(f"  Initial bias: {candidate.bias.item():.4f}")
        
        # Train the candidate
        final_correlation = candidate.train(
            x=x,
            epochs=5,
            residual_error=residual_error,
            learning_rate=0.01,
            display_frequency=10000  # No display to avoid errors
        )
        
        # Verify the correlation is set properly
        instance_correlation = candidate.get_correlation()
        
        candidates.append(candidate)
        correlations.append(final_correlation)
        
        print(f"  Final correlation (returned): {final_correlation:.6f}")
        print(f"  Final correlation (instance): {instance_correlation:.6f}")
        print(f"  Final weights: {candidate.weights.detach().numpy()}")
        print(f"  Final bias: {candidate.bias.item():.4f}")
    
    print("\n=== RESULTS ===")
    print(f"All correlations: {[f'{c:.6f}' for c in correlations]}")
    print(f"Correlations are different: {len(set(correlations)) > 1}")
    print(f"All correlations non-zero: {all(c != 0.0 for c in correlations)}")
    print(f"Returned == Instance correlations: {all(abs(correlations[i] - candidates[i].get_correlation()) < 1e-6 for i in range(3))}")
    
    # Success criteria
    different_correlations = len(set(correlations)) > 1
    non_zero_correlations = all(c != 0.0 for c in correlations)
    consistent_correlations = all(abs(correlations[i] - candidates[i].get_correlation()) < 1e-6 for i in range(3))
    
    success = different_correlations and non_zero_correlations and consistent_correlations
    
    print(f"\n✅ SUCCESS: {success}")
    print(f"   - Different correlations: {different_correlations}")
    print(f"   - Non-zero correlations: {non_zero_correlations}")
    print(f"   - Consistent correlations: {consistent_correlations}")
    
    return success

if __name__ == "__main__":
    success = test_candidate_units_simple()
    print(f"\nOVERALL RESULT: {'PASSED' if success else 'FAILED'}")
