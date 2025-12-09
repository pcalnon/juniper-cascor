#!/usr/bin/env python
"""
Test local multiprocessing functionality.
"""
import sys
# import os
import torch
import numpy as np

# Add the source directory to Python path
# sys.path.append('/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src')
sys.path.append('/Users/pcalnon/Development/python/Juniper/src/prototypes/cascor/src')

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork

from constants.constants import (
    _PROJECT_TESTING_PASSED_TEST,
    # _PROJECT_TESTING_SKIPPED_TEST,
    _PROJECT_TESTING_FAILED_TEST,
    # _PROJECT_TESTING_UNSTABLE_TEST,
    _PROJECT_TESTING_PARTIAL_TEST,
    # _PROJECT_TESTING_UNKNOWN_TEST
)

def create_test_data():
    """Create test data for spiral problem."""
    # Create simple 2D spiral data
    n_points = 100
    t = torch.linspace(0, 4*np.pi, n_points)
    
    # Create two spirals
    spiral1_x = t * torch.cos(t) * 0.1
    spiral1_y = t * torch.sin(t) * 0.1
    spiral2_x = -t * torch.cos(t) * 0.1  
    spiral2_y = -t * torch.sin(t) * 0.1
    
    # Combine spirals
    x = torch.cat([
        torch.stack([spiral1_x, spiral1_y], dim=1),
        torch.stack([spiral2_x, spiral2_y], dim=1)
    ])
    
    # Create labels (0 for first spiral, 1 for second)
    y = torch.cat([
        torch.zeros(n_points, 1),
        torch.ones(n_points, 1)
    ])
    
    return x, y

def test_multiprocessing():  # sourcery skip: extract-method
    """Test the multiprocessing functionality."""
    print("Testing multiprocessing candidate training...")
    
    # Create network
    network = CascadeCorrelationNetwork()
    
    # Create test data
    x, y = create_test_data()
    
    print(f"Test data shapes: x={x.shape}, y={y.shape}")
    
    # Calculate initial residual error
    residual_error = network.calculate_residual_error(x, y)
    print(f"Residual error shape: {residual_error.shape}")
    
    # Train candidates
    print("Starting candidate training...")
    try:
        results = network.train_candidates(x, y, residual_error)
        candidates_list, best_candidate, max_correlation = results
        candidate_ids, candidate_uuids, correlations, candidates = candidates_list
        print(f"\n{_PROJECT_TESTING_PARTIAL_TEST} Training completed!")
        print(f"\n{_PROJECT_TESTING_PARTIAL_TEST} Total candidates trained: {len(candidate_ids)}")
        print(f"\n{_PROJECT_TESTING_PARTIAL_TEST} Best correlation: {max_correlation[0]:.6f}")
        print(f"\n{_PROJECT_TESTING_PARTIAL_TEST} Successful candidates: {max_correlation[1]}")
        print(f"\n{_PROJECT_TESTING_PARTIAL_TEST} Failed candidates: {max_correlation[2]}")
        return True
        
    except Exception as e:
        print(f"\n{_PROJECT_TESTING_FAILED_TEST} Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if success := test_multiprocessing():
        print(f"\n{_PROJECT_TESTING_PASSED_TEST} Multiprocessing test passed!")
    else:
        print(f"\n{_PROJECT_TESTING_FAILED_TEST} Multiprocessing test failed!")
