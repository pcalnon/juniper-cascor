#!/usr/bin/env python
"""
Quick test to verify the cascor fixes work end-to-end.
"""

import sys
import os
sys.path.append('/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src')

import torch
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork

def main():
    print("Quick test of CascadeCorrelationNetwork fixes...")
    
    # Create minimal test data
    torch.manual_seed(42)
    x = torch.randn(8, 2)  # Small batch
    y = torch.randint(0, 2, (8, 1)).float()
    
    # Create network with very simple config
    network = CascadeCorrelationNetwork(
        input_size=2,
        output_size=1,
        candidate_pool_size=2,  # Just 2 candidates
        candidate_epochs=2,     # Just 2 epochs
        candidate_learning_rate=0.01,
        learning_rate=0.01,
        log_level_name='ERROR'  # Minimal logging
    )
    
    # Force sequential processing to avoid multiprocessing issues
    original_cpu_count = os.cpu_count
    os.cpu_count = lambda: 1
    
    try:
        print(f"Input: {x.shape}, Output: {y.shape}")
        
        # Calculate residual error
        residual_error = network.calculate_residual_error(x, y)
        print(f"Residual error: {residual_error.shape}")
        
        # Train candidates with sequential processing
        print("Training candidates...")
        results = network.train_candidates(x, y, residual_error)
        
        print(f"Results type: {type(results)}")
        print(f"Results length: {len(results) if hasattr(results, '__len__') else 'N/A'}")
        
        if isinstance(results, tuple) and len(results) >= 1:
            candidates_list = results[0]
            print(f"Candidates list type: {type(candidates_list)}")
            
            if isinstance(candidates_list, tuple) and len(candidates_list) == 4:
                candidate_ids, candidate_uuids, correlations, candidates = candidates_list
                print("✅ Successfully unpacked candidates_list!")
                print(f"Number of candidates: {len(candidates)}")
                print(f"Correlations: {correlations}")
                
                # Verify candidates have different correlations
                if len(set(correlations)) > 1:
                    print("✅ Candidates have different correlations!")
                    return True
                else:
                    print("❌ All candidates have identical correlations")
                    return False
            else:
                print(f"❌ Invalid candidates_list format: {candidates_list}")
                return False
        else:
            print(f"❌ Invalid results format: {results}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.cpu_count = original_cpu_count

if __name__ == "__main__":
    success = main()
    print(f"Test {'PASSED' if success else 'FAILED'}")
