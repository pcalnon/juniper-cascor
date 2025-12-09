#!/usr/bin/env python
"""
Test script to validate P0 critical fixes for Cascor prototype.
Run with: conda activate JuniperPython-ORIG && python test_critical_fixes.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch

print("="*70)
print("Critical Fixes Validation Tests")
print("="*70)

def test_1_dataclass_fields():
    """Test that CandidateTrainingResult has correct field names."""
    print("\n[Test 1] CandidateTrainingResult dataclass fields...")
    try:
        from candidate_unit.candidate_unit import CandidateTrainingResult
        result = CandidateTrainingResult(
            candidate_id=0,
            correlation=0.5,
            candidate=None
        )
        assert hasattr(result, 'candidate_id'), "Missing candidate_id field"
        assert hasattr(result, 'correlation'), "Missing correlation field"
        assert hasattr(result, 'candidate'), "Missing candidate field"
        assert not hasattr(result, 'candidate_index'), "Old candidate_index field still exists"
        assert not hasattr(result, 'best_correlation'), "Old best_correlation field still exists"
        print("✅ PASS: CandidateTrainingResult has correct fields")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False

def test_2_network_creation():
    """Test that network can be created with snapshot_counter."""
    print("\n[Test 2] Network creation and initialization...")
    try:
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        from cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
        
        config = CascadeCorrelationConfig(input_size=2, output_size=1, candidate_pool_size=2)
        network = CascadeCorrelationNetwork(config=config)
        
        assert hasattr(network, 'snapshot_counter'), "Missing snapshot_counter"
        assert network.snapshot_counter == 0, f"snapshot_counter should be 0, got {network.snapshot_counter}"
        print(f"✅ PASS: Network created, snapshot_counter={network.snapshot_counter}")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_3_candidate_training():
    """Test that candidate unit can train without crashes."""
    print("\n[Test 3] Candidate unit training...")
    try:
        from candidate_unit.candidate_unit import CandidateUnit, CandidateTrainingResult
        
        candidate = CandidateUnit(
            CandidateUnit__input_size=2,
            CandidateUnit__candidate_index=0,
            CandidateUnit__log_level_name="WARNING"  # Reduce log noise
        )
        
        x = torch.randn(10, 2)
        residual_error = torch.randn(10)
        
        result = candidate.train(
            x=x,
            epochs=5,
            residual_error=residual_error,
            learning_rate=0.01
        )
        
        assert isinstance(result, CandidateTrainingResult), "train() should return CandidateTrainingResult"
        assert hasattr(result, 'correlation'), "Result missing correlation field"
        assert hasattr(result, 'epochs_completed'), "Result missing epochs_completed"
        assert result.epochs_completed == 5, f"Expected 5 epochs, got {result.epochs_completed}"
        
        print(f"✅ PASS: Training completed - correlation: {result.correlation:.6f}, epochs: {result.epochs_completed}")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_4_get_single_candidate_data():
    """Test that get_single_candidate_data uses getattr correctly."""
    print("\n[Test 4] get_single_candidate_data() method...")
    try:
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        from cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
        from candidate_unit.candidate_unit import CandidateTrainingResult
        
        config = CascadeCorrelationConfig(input_size=2, output_size=1)
        network = CascadeCorrelationNetwork(config=config)
        
        results = [
            CandidateTrainingResult(candidate_id=0, correlation=0.3, candidate=None),
            CandidateTrainingResult(candidate_id=1, correlation=0.7, candidate=None),
            CandidateTrainingResult(candidate_id=2, correlation=0.5, candidate=None),
        ]
        
        correlation = network.get_single_candidate_data(results, 1, 'correlation', 0.0)
        assert correlation == 0.7, f"Expected 0.7, got {correlation}"
        
        candidate_id = network.get_single_candidate_data(results, 2, 'candidate_id', -1)
        assert candidate_id == 2, f"Expected 2, got {candidate_id}"
        
        default_val = network.get_single_candidate_data(results, 10, 'correlation', -999)
        assert default_val == -999, f"Expected default -999, got {default_val}"
        
        print("✅ PASS: get_single_candidate_data works correctly")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_5_training_results_dataclass():
    """Test that TrainingResults dataclass has max_correlation field."""
    print("\n[Test 5] TrainingResults dataclass...")
    try:
        from cascade_correlation.cascade_correlation import TrainingResults
        import datetime
        
        results = TrainingResults(
            epochs_completed=10,
            candidate_ids=[0, 1],
            candidate_uuids=["uuid1", "uuid2"],
            correlations=[0.3, 0.7],
            candidate_objects=[None, None],
            best_candidate_id=1,
            best_candidate_uuid="uuid2",
            best_correlation=0.7,
            best_candidate=None,
            success_count=2,
            successful_candidates=2,
            failed_count=0,
            error_messages=[],
            max_correlation=0.7,  # This field should exist
            start_time=datetime.datetime.now(),
            end_time=datetime.datetime.now()
        )
        
        assert hasattr(results, 'max_correlation'), "Missing max_correlation field"
        assert results.max_correlation == 0.7, f"max_correlation should be 0.7, got {results.max_correlation}"
        
        print("✅ PASS: TrainingResults has max_correlation field")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    results = []
    
    results.append(("Dataclass Fields", test_1_dataclass_fields()))
    results.append(("Network Creation", test_2_network_creation()))
    results.append(("Candidate Training", test_3_candidate_training()))
    results.append(("get_single_candidate_data", test_4_get_single_candidate_data()))
    results.append(("TrainingResults Dataclass", test_5_training_results_dataclass()))
    
    print("\n" + "="*70)
    print("Test Results Summary")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print("="*70)
    print(f"Total: {passed}/{total} tests passed ({100*passed//total}%)")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ALL CRITICAL FIXES VALIDATED SUCCESSFULLY")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - review output above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
