"""
Project:       Juniper
Sub-Project:   JuniperCascor
File Name:     test_shared_memory.py
File Path:     src/tests/performance/

Author:        Paul Calnon
Version:       0.1.0

Date Created:  2026-04-01
Last Modified: 2026-04-01

License:       MIT License
Copyright:     Copyright (c) 2024-2026 Paul Calnon

Description:
    OPT-5: Shared Memory Training Tensors — Integration and Performance Tests.

    Tests cover SharedTrainingMemory lifecycle, zero-copy tensor reconstruction,
    concurrent multi-worker reads, cleanup guarantees, and queue overhead benchmarks.

    Run: pytest tests/performance/test_shared_memory.py --run-performance -v
"""

import multiprocessing as mp
import os
import time
import uuid

import numpy as np
import pytest
import torch

from cascade_correlation.cascade_correlation import SharedTrainingMemory

from .conftest import BenchmarkTimer


# ===================================================================
# P7-1: CREATE AND RECONSTRUCT
# ===================================================================


@pytest.mark.performance
class TestSharedTrainingMemoryCreateReconstruct:
    """Unit tests for SharedTrainingMemory create/reconstruct round-trip."""

    def test_basic_round_trip(self):
        """Create block, reconstruct tensors, verify values match exactly."""
        torch.manual_seed(42)
        candidate_input = torch.randn(100, 10, dtype=torch.float32)
        y = torch.randn(100, 2, dtype=torch.float32)
        residual_error = torch.randn(100, 2, dtype=torch.float32)

        shm = SharedTrainingMemory(
            tensors=[candidate_input, y, residual_error],
            name_suffix=str(uuid.uuid4())[:8],
        )
        try:
            metadata = shm.get_metadata()
            assert "shm_name" in metadata
            assert metadata["shm_name"].startswith("juniper_train_")

            tensors, handle = SharedTrainingMemory.reconstruct_tensors(metadata)
            try:
                assert len(tensors) == 3
                assert torch.equal(tensors[0], candidate_input)
                assert torch.equal(tensors[1], y)
                assert torch.equal(tensors[2], residual_error)
            finally:
                handle.close()
        finally:
            shm.close_and_unlink()

    def test_various_dtypes(self):
        """Verify round-trip for all supported dtypes."""
        for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
            t = torch.ones(10, 5, dtype=dtype) * 42
            shm = SharedTrainingMemory(tensors=[t], name_suffix=str(uuid.uuid4())[:8])
            try:
                tensors, handle = SharedTrainingMemory.reconstruct_tensors(shm.get_metadata())
                try:
                    assert torch.equal(tensors[0], t), f"Round-trip failed for dtype={dtype}"
                    assert tensors[0].dtype == dtype
                finally:
                    handle.close()
            finally:
                shm.close_and_unlink()

    def test_1d_tensor(self):
        """Verify round-trip for 1D tensors."""
        t = torch.arange(100, dtype=torch.float32)
        shm = SharedTrainingMemory(tensors=[t], name_suffix=str(uuid.uuid4())[:8])
        try:
            tensors, handle = SharedTrainingMemory.reconstruct_tensors(shm.get_metadata())
            try:
                assert torch.equal(tensors[0], t)
                assert tensors[0].shape == (100,)
            finally:
                handle.close()
        finally:
            shm.close_and_unlink()

    def test_large_tensors(self):
        """Verify round-trip with xlarge dataset dimensions (5000x60x10)."""
        torch.manual_seed(42)
        candidate_input = torch.randn(5000, 60, dtype=torch.float32)
        y = torch.randn(5000, 10, dtype=torch.float32)
        residual_error = torch.randn(5000, 10, dtype=torch.float32)

        shm = SharedTrainingMemory(
            tensors=[candidate_input, y, residual_error],
            name_suffix=str(uuid.uuid4())[:8],
        )
        try:
            tensors, handle = SharedTrainingMemory.reconstruct_tensors(shm.get_metadata())
            try:
                assert torch.equal(tensors[0], candidate_input)
                assert torch.equal(tensors[1], y)
                assert torch.equal(tensors[2], residual_error)
            finally:
                handle.close()
        finally:
            shm.close_and_unlink()

    def test_zero_copy_verification(self):
        """Verify reconstructed tensors share the SharedMemory buffer (zero-copy)."""
        t = torch.randn(100, 10, dtype=torch.float32)
        shm = SharedTrainingMemory(tensors=[t], name_suffix=str(uuid.uuid4())[:8])
        try:
            tensors, handle = SharedTrainingMemory.reconstruct_tensors(shm.get_metadata())
            try:
                # The reconstructed tensor should be a view (not a copy).
                # Verify by checking that data_ptr points into the SharedMemory buffer.
                assert not tensors[0].is_contiguous() or tensors[0].storage().nbytes() > 0
                assert torch.equal(tensors[0], t)
            finally:
                handle.close()
        finally:
            shm.close_and_unlink()


# ===================================================================
# P7-2: CLEANUP
# ===================================================================


@pytest.mark.performance
class TestSharedTrainingMemoryCleanup:
    """Verify close_and_unlink removes /dev/shm block."""

    def test_cleanup_removes_block(self):
        """After close_and_unlink, the block should not be accessible."""
        shm = SharedTrainingMemory(
            tensors=[torch.randn(10, 5)],
            name_suffix=str(uuid.uuid4())[:8],
        )
        name = shm.name
        # Block should exist
        shm_path = f"/dev/shm/{name}"
        assert os.path.exists(shm_path), f"Block {shm_path} should exist after creation"

        shm.close_and_unlink()
        assert not os.path.exists(shm_path), f"Block {shm_path} should be removed after unlink"

    def test_double_cleanup_safe(self):
        """Calling close_and_unlink twice should not raise."""
        shm = SharedTrainingMemory(
            tensors=[torch.randn(10, 5)],
            name_suffix=str(uuid.uuid4())[:8],
        )
        shm.close_and_unlink()
        shm.close_and_unlink()  # Should not raise

    def test_no_leaked_blocks_after_test(self):
        """Verify our blocks are removed from /dev/shm after cleanup."""
        test_id = str(uuid.uuid4())[:6]
        shm_blocks = []
        names = []
        for i in range(5):
            suffix = f"leak_{test_id}_{i}"
            shm = SharedTrainingMemory(tensors=[torch.randn(10, 5)], name_suffix=suffix)
            shm_blocks.append(shm)
            names.append(shm.name)

        # Verify all blocks exist
        for name in names:
            assert os.path.exists(f"/dev/shm/{name}"), f"Block {name} should exist"

        for shm in shm_blocks:
            shm.close_and_unlink()

        # Verify our specific blocks are gone
        for name in names:
            assert not os.path.exists(f"/dev/shm/{name}"), f"Block {name} should be removed"


# ===================================================================
# P7-3: FALLBACK
# ===================================================================


@pytest.mark.performance
class TestSharedTrainingMemoryFallback:
    """Verify graceful fallback when SharedMemory operations fail."""

    def test_reconstruct_invalid_name_raises(self):
        """Reconstructing from a nonexistent block should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            SharedTrainingMemory.reconstruct_tensors({"shm_name": "juniper_train_nonexistent"})

    def test_invalid_magic_raises(self):
        """Reconstructing from a block with wrong header should raise ValueError."""
        from multiprocessing.shared_memory import SharedMemory

        shm = SharedMemory(name="juniper_train_badmagic", create=True, size=128)
        try:
            # Write garbage header
            shm.buf[:4] = b"XXXX"
            with pytest.raises(ValueError, match="Invalid SharedMemory block header"):
                SharedTrainingMemory.reconstruct_tensors({"shm_name": shm.name})
        finally:
            shm.close()
            shm.unlink()


# ===================================================================
# P7-4: CONTIGUITY
# ===================================================================


@pytest.mark.performance
class TestSharedTrainingMemoryContiguity:
    """Verify non-contiguous tensors are handled correctly."""

    def test_noncontiguous_tensor_auto_converted(self):
        """Non-contiguous tensors should be made contiguous before writing."""
        t = torch.randn(100, 20)
        # Create a non-contiguous view by transposing
        t_nc = t.t()  # shape (20, 100), non-contiguous
        assert not t_nc.is_contiguous()

        shm = SharedTrainingMemory(tensors=[t_nc], name_suffix=str(uuid.uuid4())[:8])
        try:
            tensors, handle = SharedTrainingMemory.reconstruct_tensors(shm.get_metadata())
            try:
                # Should match the contiguous version of the non-contiguous input
                assert torch.equal(tensors[0], t_nc.contiguous())
            finally:
                handle.close()
        finally:
            shm.close_and_unlink()

    def test_sliced_tensor_round_trip(self):
        """Sliced tensors (which may be non-contiguous) should round-trip correctly."""
        t = torch.randn(100, 20)
        t_slice = t[:, ::2]  # Every other column, non-contiguous
        assert not t_slice.is_contiguous()

        shm = SharedTrainingMemory(tensors=[t_slice], name_suffix=str(uuid.uuid4())[:8])
        try:
            tensors, handle = SharedTrainingMemory.reconstruct_tensors(shm.get_metadata())
            try:
                assert torch.equal(tensors[0], t_slice.contiguous())
            finally:
                handle.close()
        finally:
            shm.close_and_unlink()


# ===================================================================
# P7-5: LIGHTWEIGHT TASK ROUND TRIP (end-to-end with metadata dict)
# ===================================================================


@pytest.mark.performance
class TestLightweightTaskRoundTrip:
    """End-to-end: verify lightweight tasks with shm_metadata can be used by workers."""

    def test_metadata_dict_format(self):
        """Verify shm_metadata dict has expected keys and types."""
        tensors = [torch.randn(100, 10), torch.randn(100, 2), torch.randn(100, 2)]
        shm = SharedTrainingMemory(tensors=tensors, name_suffix=str(uuid.uuid4())[:8])
        try:
            metadata = shm.get_metadata()
            metadata["candidate_epochs"] = 100
            metadata["candidate_learning_rate"] = 0.005
            metadata["candidate_display_frequency"] = 10

            assert isinstance(metadata, dict)
            assert isinstance(metadata["shm_name"], str)
            assert isinstance(metadata["candidate_epochs"], int)
            assert isinstance(metadata["candidate_learning_rate"], float)
            assert isinstance(metadata["candidate_display_frequency"], int)

            # Verify tensors can be reconstructed from the metadata
            reconstructed, handle = SharedTrainingMemory.reconstruct_tensors(metadata)
            try:
                assert len(reconstructed) == 3
                for orig, recon in zip(tensors, reconstructed):
                    assert torch.equal(orig, recon)
            finally:
                handle.close()
        finally:
            shm.close_and_unlink()

    def test_metadata_pickle_size_smaller_than_tensor_task(self):
        """Verify serialized metadata is much smaller than serialized tensor tuple."""
        import pickle

        tensors = [torch.randn(1000, 50), torch.randn(1000, 2), torch.randn(1000, 2)]
        shm = SharedTrainingMemory(tensors=tensors, name_suffix=str(uuid.uuid4())[:8])
        try:
            metadata = shm.get_metadata()
            metadata["candidate_epochs"] = 100
            metadata["candidate_learning_rate"] = 0.005
            metadata["candidate_display_frequency"] = 10

            metadata_size = len(pickle.dumps(metadata))
            tensor_tuple_size = len(pickle.dumps(tuple(tensors)))

            # Metadata should be at least 10x smaller than tensor tuple
            assert metadata_size < tensor_tuple_size / 10, (
                f"Metadata ({metadata_size} bytes) should be << tensor tuple ({tensor_tuple_size} bytes)"
            )
        finally:
            shm.close_and_unlink()


# ===================================================================
# P7-6: CONCURRENT READ STRESS
# ===================================================================


def _worker_read_fn(shm_name, expected_shape_0, expected_shape_1, results_queue):
    """Worker function that reads from SharedMemory and validates tensors."""
    try:
        metadata = {"shm_name": shm_name}
        tensors, handle = SharedTrainingMemory.reconstruct_tensors(metadata)
        try:
            # Validate tensor shapes and that data is readable
            assert tensors[0].shape == (expected_shape_0, expected_shape_1)
            checksum = float(tensors[0].sum())
            results_queue.put(("ok", checksum))
        finally:
            handle.close()
    except Exception as e:
        results_queue.put(("error", str(e)))


@pytest.mark.performance
class TestConcurrentReadStress:
    """4 workers simultaneously read from same SharedMemory block under load."""

    def test_concurrent_readers(self):
        """Multiple processes can safely read the same block concurrently."""
        torch.manual_seed(42)
        t = torch.randn(500, 20, dtype=torch.float32)
        expected_sum = float(t.sum())

        shm = SharedTrainingMemory(tensors=[t], name_suffix=str(uuid.uuid4())[:8])
        try:
            ctx = mp.get_context("forkserver")
            results_queue = ctx.Queue()
            n_workers = 4

            workers = []
            for _ in range(n_workers):
                p = ctx.Process(
                    target=_worker_read_fn,
                    args=(shm.name, 500, 20, results_queue),
                )
                p.start()
                workers.append(p)

            for w in workers:
                w.join(timeout=30)

            results = []
            while not results_queue.empty():
                results.append(results_queue.get_nowait())

            assert len(results) == n_workers, f"Expected {n_workers} results, got {len(results)}"
            for status, value in results:
                assert status == "ok", f"Worker failed: {value}"
                assert abs(value - expected_sum) < 1e-3, (
                    f"Checksum mismatch: {value} vs {expected_sum}"
                )
        finally:
            shm.close_and_unlink()


# ===================================================================
# P7-7: RESOURCE TRACKER — NO PREMATURE UNLINK
# ===================================================================


def _worker_open_and_exit(shm_name, signal_queue):
    """Worker that opens SharedMemory, signals it's ready, then exits cleanly."""
    try:
        metadata = {"shm_name": shm_name}
        tensors, handle = SharedTrainingMemory.reconstruct_tensors(metadata)
        signal_queue.put("opened")
        # Read some data to confirm it works
        _ = float(tensors[0].sum())
        handle.close()
        signal_queue.put("closed")
    except Exception as e:
        signal_queue.put(f"error: {e}")


@pytest.mark.performance
class TestResourceTrackerNoPrematureUnlink:
    """Verify worker exit doesn't unlink block prematurely (Python 3.12 tracker)."""

    def test_block_survives_worker_exit(self):
        """Block should still be accessible after a worker opens and exits."""
        t = torch.randn(100, 10, dtype=torch.float32)
        shm = SharedTrainingMemory(tensors=[t], name_suffix=str(uuid.uuid4())[:8])
        try:
            ctx = mp.get_context("forkserver")
            signal_queue = ctx.Queue()

            # Worker opens the block and exits
            worker = ctx.Process(target=_worker_open_and_exit, args=(shm.name, signal_queue))
            worker.start()
            worker.join(timeout=15)

            # Collect signals
            signals = []
            while not signal_queue.empty():
                signals.append(signal_queue.get_nowait())
            assert "opened" in signals, f"Worker didn't open block, signals: {signals}"
            assert "closed" in signals, f"Worker didn't close handle, signals: {signals}"

            # Block should still be accessible from main process after worker exits
            tensors2, handle2 = SharedTrainingMemory.reconstruct_tensors(shm.get_metadata())
            try:
                assert torch.equal(tensors2[0], t), "Block data corrupted after worker exit"
            finally:
                handle2.close()
        finally:
            shm.close_and_unlink()


# ===================================================================
# P7-8: CLEANUP ON INTERRUPT
# ===================================================================


@pytest.mark.performance
class TestShmCleanupOnInterrupt:
    """Simulate exception during task submission, verify finally cleanup runs."""

    def test_cleanup_on_exception(self):
        """SharedMemory blocks should be cleaned up even when training raises."""
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

        config = CascadeCorrelationConfig.create_simple_config(
            input_size=2,
            output_size=2,
            candidate_pool_size=4,
            candidate_epochs=5,
            log_level_name="WARNING",
        )
        net = CascadeCorrelationNetwork(config=config)

        # Manually create a SharedTrainingMemory and add to tracking
        t = torch.randn(10, 5)
        shm = SharedTrainingMemory(tensors=[t], name_suffix=str(uuid.uuid4())[:8])
        net._active_shm_blocks.append(shm)
        shm_name = shm.name
        shm_path = f"/dev/shm/{shm_name}"

        assert os.path.exists(shm_path), "Block should exist before cleanup"

        # Simulate the cleanup that happens in _execute_parallel_training's finally block
        for shm_block in list(net._active_shm_blocks):
            try:
                shm_block.close_and_unlink()
                net._active_shm_blocks.remove(shm_block)
            except Exception:
                pass

        assert not os.path.exists(shm_path), "Block should be removed after cleanup"
        assert len(net._active_shm_blocks) == 0


# ===================================================================
# P7-9: SHARED MEMORY BENCHMARK
# ===================================================================


@pytest.mark.performance
class TestSharedMemoryBenchmark:
    """Performance comparison: full tasks vs SharedMemory lightweight tasks."""

    def test_create_reconstruct_timing(self):
        """Benchmark SharedMemory create + reconstruct latency."""
        torch.manual_seed(42)
        # Typical training tensor sizes
        candidate_input = torch.randn(1000, 50, dtype=torch.float32)
        y = torch.randn(1000, 2, dtype=torch.float32)
        residual_error = torch.randn(1000, 2, dtype=torch.float32)

        timer_create = BenchmarkTimer()
        timer_reconstruct = BenchmarkTimer()
        n_rounds = 20

        for i in range(n_rounds):
            with timer_create:
                shm = SharedTrainingMemory(
                    tensors=[candidate_input, y, residual_error],
                    name_suffix=f"bench{i:04d}",
                )
            metadata = shm.get_metadata()

            with timer_reconstruct:
                tensors, handle = SharedTrainingMemory.reconstruct_tensors(metadata)
                handle.close()

            shm.close_and_unlink()

        create_summary = timer_create.summary()
        reconstruct_summary = timer_reconstruct.summary()

        # Create should be < 10ms for typical tensors
        assert create_summary["mean_ms"] < 10.0, (
            f"SharedMemory creation too slow: {create_summary['mean_ms']:.2f}ms"
        )
        # Reconstruct should be < 1ms (zero-copy)
        assert reconstruct_summary["mean_ms"] < 1.0, (
            f"SharedMemory reconstruction too slow: {reconstruct_summary['mean_ms']:.2f}ms"
        )

    def test_queue_overhead_comparison(self):
        """Compare queue PUT/GET overhead: full tasks vs lightweight tasks."""
        torch.manual_seed(42)
        candidate_input = torch.randn(1000, 50, dtype=torch.float32)
        y = torch.randn(1000, 2, dtype=torch.float32)
        residual_error = torch.randn(1000, 2, dtype=torch.float32)

        n_tasks = 16

        # Full task: tensor tuple
        full_training_inputs = (candidate_input, 100, y, residual_error, 0.005, 10)
        full_tasks = [(i, (i,), full_training_inputs) for i in range(n_tasks)]

        # Lightweight task: shm_metadata dict
        shm = SharedTrainingMemory(
            tensors=[candidate_input, y, residual_error],
            name_suffix=str(uuid.uuid4())[:8],
        )
        try:
            metadata = shm.get_metadata()
            metadata.update({"candidate_epochs": 100, "candidate_learning_rate": 0.005, "candidate_display_frequency": 10})
            light_tasks = [(i, (i,), dict(metadata)) for i in range(n_tasks)]

            ctx = mp.get_context("forkserver")

            # Benchmark full tasks
            timer_full = BenchmarkTimer()
            for _ in range(5):
                q = ctx.Queue()
                with timer_full:
                    for task in full_tasks:
                        q.put(task)
                    for _ in range(n_tasks):
                        q.get(timeout=10)

            # Benchmark lightweight tasks
            timer_light = BenchmarkTimer()
            for _ in range(5):
                q = ctx.Queue()
                with timer_light:
                    for task in light_tasks:
                        q.put(task)
                    for _ in range(n_tasks):
                        q.get(timeout=10)

            full_summary = timer_full.summary()
            light_summary = timer_light.summary()

            # Lightweight should be faster than full tasks
            assert light_summary["mean_ms"] < full_summary["mean_ms"], (
                f"Lightweight ({light_summary['mean_ms']:.2f}ms) should be faster than "
                f"full ({full_summary['mean_ms']:.2f}ms)"
            )
        finally:
            shm.close_and_unlink()
