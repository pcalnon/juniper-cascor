"""
Project:       Juniper
Sub-Project:   JuniperCascor
File Name:     test_concurrency_scaling.py
File Path:     src/tests/performance/

Author:        Paul Calnon
Version:       0.1.0

Date Created:  2026-04-01
Last Modified: 2026-04-01

License:       MIT License
Copyright:     Copyright (c) 2024-2026 Paul Calnon

Description:
    Phase 3: Concurrency Scaling Performance Tests.

    Benchmarks for the multiprocessing subsystem of CascadeCorrelationNetwork,
    covering worker pool scaling, queue throughput, IPC serialization overhead,
    and worker lifecycle costs.

    Step 3.1 — Worker Pool Scaling Benchmarks (process_count 1/2/4)
    Step 3.2 — Queue Throughput Analysis (raw multiprocessing.Queue latency)
    Step 3.3 — IPC Serialization Overhead (pickle round-trip for torch tensors)
    Step 3.4 — Worker Startup and Pool Lifecycle (cold/warm pool, shutdown, subprocess torch)

    Run: pytest tests/performance/test_concurrency_scaling.py --run-performance -v

    Multiprocessing tests require: pytest tests/performance/test_concurrency_scaling.py --run-performance -v -k multiprocessing
"""

import multiprocessing
import os
import pickle
import time

import numpy as np
import pytest
import torch

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork

from .conftest import BenchmarkTimer, _make_benchmark_config


# Module-level functions required for forkserver pickling (cannot use local/lambda functions).

def _producer_fn(queue, n_items, producer_id):
    """Worker function that puts items into a shared queue (used by test_multi_producer_throughput)."""
    import numpy as np

    for i in range(n_items):
        # Use numpy arrays instead of torch tensors to avoid forkserver resource_sharer issues
        # with torch's FD-based tensor sharing across subprocess boundaries.
        queue.put((producer_id, i, np.random.randn(5, 2).astype(np.float32)))


def _subprocess_torch_timing(rq):
    """Run in a subprocess: measure first vs warm torch op latency (used by test_subprocess_first_torch_op_latency)."""
    import time

    import torch

    torch.set_num_threads(1)
    # First torch operation (triggers lazy init)
    start = time.perf_counter_ns()
    x = torch.randn(50, 2)
    _ = torch.nn.functional.linear(x, torch.randn(2, 2), torch.randn(2))
    elapsed_ns = time.perf_counter_ns() - start
    # Second torch operation (warm)
    start2 = time.perf_counter_ns()
    x2 = torch.randn(50, 2)
    _ = torch.nn.functional.linear(x2, torch.randn(2, 2), torch.randn(2))
    elapsed_warm_ns = time.perf_counter_ns() - start2
    rq.put((elapsed_ns, elapsed_warm_ns))


# ===================================================================
# FIXTURES: Override force_sequential_training for multiprocessing tests
# ===================================================================


@pytest.fixture
def real_process_count(monkeypatch):
    """Override force_sequential_training to respect CASCOR_NUM_PROCESSES env var.

    The autouse force_sequential_training fixture in tests/conftest.py patches
    _calculate_optimal_process_count to always return 1, preventing multiprocessing.
    This fixture re-patches the method with one that reads the CASCOR_NUM_PROCESSES
    environment variable, allowing multiprocessing tests to control process count.
    """

    def _env_aware_process_count(self) -> int:
        env_override = os.environ.get("CASCOR_NUM_PROCESSES")
        if env_override is not None:
            return max(1, int(env_override))
        return max(1, min((os.cpu_count() or 2) - 1, 4))

    monkeypatch.setattr(
        CascadeCorrelationNetwork,
        "_calculate_optimal_process_count",
        _env_aware_process_count,
    )


@pytest.fixture
def forkserver_context():
    """Provide a forkserver multiprocessing context matching production configuration."""
    return multiprocessing.get_context("forkserver")


def _make_scaling_network(candidate_pool_size=8, candidate_epochs=30, process_count_override=None):
    """Create a CascadeCorrelationNetwork configured for concurrency scaling tests.

    Uses small training parameters to keep tests CI-friendly while still exercising
    the multiprocessing subsystem.
    """
    config = _make_benchmark_config(
        input_size=2,
        output_size=2,
        candidate_pool_size=candidate_pool_size,
        candidate_epochs=candidate_epochs,
        output_epochs=10,
        max_hidden_units=3,
        patience=2,
        learning_rate=0.01,
        candidate_learning_rate=0.005,
    )
    net = CascadeCorrelationNetwork(config=config)
    return net


# ===================================================================
# Step 3.1: Worker Pool Scaling Benchmarks
# ===================================================================


@pytest.mark.performance
@pytest.mark.multiprocessing
class TestWorkerPoolScaling:
    """Benchmark candidate training with varying process counts.

    Measures total round time, speedup factor, and parallel efficiency
    for process_count = 1, 2, 4 with small datasets (100-200 samples).
    """

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("process_count", [1, 2, 4])
    def test_scaling_small_dataset(self, process_count, small_spiral_data, real_process_count, monkeypatch):
        """Measure training round time with 100 samples, pool_size=8, epochs=30."""
        monkeypatch.setenv("CASCOR_NUM_PROCESSES", str(process_count))

        x_train, y_train = small_spiral_data
        net = _make_scaling_network(candidate_pool_size=8, candidate_epochs=30)

        timer = BenchmarkTimer()
        with timer:
            net.fit(x_train, y_train, max_epochs=1)

        summary = timer.summary()
        print(f"\n[Step 3.1] process_count={process_count}, samples=100, pool_size=8, epochs=30")
        print(f"  Total round time: {summary['mean_ms']:.1f} ms")
        assert summary["mean_ms"] > 0, "Training should take measurable time"

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("process_count", [1, 2, 4])
    def test_scaling_medium_dataset(self, process_count, medium_spiral_data, real_process_count, monkeypatch):
        """Measure training round time with 400 samples, pool_size=8, epochs=20."""
        monkeypatch.setenv("CASCOR_NUM_PROCESSES", str(process_count))

        x_train, y_train = medium_spiral_data
        net = _make_scaling_network(candidate_pool_size=8, candidate_epochs=20)

        timer = BenchmarkTimer()
        with timer:
            net.fit(x_train, y_train, max_epochs=1)

        summary = timer.summary()
        print(f"\n[Step 3.1] process_count={process_count}, samples=400, pool_size=8, epochs=20")
        print(f"  Total round time: {summary['mean_ms']:.1f} ms")
        assert summary["mean_ms"] > 0

    @pytest.mark.timeout(60)
    def test_speedup_comparison(self, small_spiral_data, real_process_count, monkeypatch):
        """Compare sequential vs parallel training and compute speedup/efficiency.

        Runs process_count=1, then process_count=2, then process_count=4,
        and reports speedup factors relative to sequential baseline.
        """
        x_train, y_train = small_spiral_data
        results = {}

        for pc in [1, 2, 4]:
            monkeypatch.setenv("CASCOR_NUM_PROCESSES", str(pc))
            net = _make_scaling_network(candidate_pool_size=8, candidate_epochs=30)

            timer = BenchmarkTimer()
            with timer:
                net.fit(x_train, y_train, max_epochs=1)

            # Clean up worker pool between runs
            net._shutdown_worker_pool()
            results[pc] = timer.summary()["mean_ms"]

        baseline = results[1]
        print("\n[Step 3.1] Speedup comparison (100 samples, pool=8, epochs=30):")
        print(f"  Sequential (1 proc): {baseline:.1f} ms")
        for pc in [2, 4]:
            speedup = baseline / results[pc] if results[pc] > 0 else 0
            efficiency = speedup / pc * 100
            print(f"  {pc} procs: {results[pc]:.1f} ms | speedup={speedup:.2f}x | efficiency={efficiency:.1f}%")

        # Verify all runs completed successfully (times are positive)
        for pc, ms in results.items():
            assert ms > 0, f"Training with {pc} processes should complete in measurable time"


# ===================================================================
# Step 3.2: Queue Throughput Analysis
# ===================================================================


@pytest.mark.performance
class TestQueueThroughput:
    """Benchmark raw multiprocessing.Queue put/get latency.

    Uses forkserver context to match production configuration.
    Tests uncontested throughput, multi-producer scenarios, and drain overhead.
    """

    def test_uncontested_put_get_latency(self, benchmark, forkserver_context):
        """Measure single put/get round-trip latency on an empty queue."""
        q = forkserver_context.Queue(maxsize=1024)
        payload = (42, "hello", torch.randn(10, 2))

        def put_get():
            q.put(payload)
            return q.get()

        result = benchmark.pedantic(put_get, rounds=50, warmup_rounds=5)
        assert result is not None

    def test_burst_throughput(self, forkserver_context):
        """Measure throughput for bursting N items into a queue and draining them.

        Simulates the task submission pattern: all tasks enqueued before workers consume.
        """
        q = forkserver_context.Queue(maxsize=1024)
        payloads = [(i, torch.randn(10, 2)) for i in range(64)]

        timer = BenchmarkTimer()

        for _ in range(5):
            # Burst put
            with timer:
                for p in payloads:
                    q.put(p)

        put_summary = timer.summary()

        drain_timer = BenchmarkTimer()
        # Items are still in the queue from the last burst iteration
        # Drain all remaining items
        from queue import Empty

        while True:
            try:
                q.get_nowait()
            except Empty:
                break

        # Re-burst and drain for clean measurement
        for p in payloads:
            q.put(p)

        with drain_timer:
            for _ in range(len(payloads)):
                q.get(timeout=5.0)

        drain_summary = drain_timer.summary()

        print(f"\n[Step 3.2] Burst throughput (64 items, tensor payload):")
        print(f"  Put burst (5 rounds): mean={put_summary['mean_ms']:.2f} ms, min={put_summary['min_ms']:.2f} ms")
        print(f"  Drain 64 items: {drain_summary['mean_ms']:.2f} ms")
        print(f"  Per-item put: {put_summary['mean_ms'] / len(payloads):.4f} ms")
        print(f"  Per-item get: {drain_summary['mean_ms'] / len(payloads):.4f} ms")

        assert put_summary["mean_ms"] > 0
        assert drain_summary["mean_ms"] > 0

    def test_multi_producer_throughput(self, forkserver_context):
        """Measure queue throughput with multiple producer processes writing concurrently.

        Spawns N producers that each put M items, then drains from the main process.
        """
        q = forkserver_context.Queue(maxsize=1024)
        n_producers = 3
        items_per_producer = 20
        total_items = n_producers * items_per_producer

        # Spawn producers
        timer = BenchmarkTimer()
        with timer:
            producers = []
            for pid in range(n_producers):
                p = forkserver_context.Process(
                    target=_producer_fn,
                    args=(q, items_per_producer, pid),
                    daemon=True,
                )
                p.start()
                producers.append(p)

            # Wait for all producers to finish
            for p in producers:
                p.join(timeout=15.0)

            # Drain all results
            collected = 0
            from queue import Empty

            deadline = time.time() + 10.0
            while collected < total_items and time.time() < deadline:
                try:
                    q.get(timeout=1.0)
                    collected += 1
                except Empty:
                    break

        summary = timer.summary()
        print(f"\n[Step 3.2] Multi-producer throughput ({n_producers} producers x {items_per_producer} items):")
        print(f"  Total time: {summary['mean_ms']:.2f} ms")
        print(f"  Items collected: {collected}/{total_items}")
        print(f"  Per-item throughput: {summary['mean_ms'] / max(collected, 1):.4f} ms/item")

        assert collected == total_items, f"Expected {total_items} items, got {collected}"

    def test_drain_overhead_empty_queue(self, benchmark, forkserver_context):
        """Measure the cost of attempting to drain an already-empty queue.

        This pattern occurs at the start of _execute_parallel_training (RC-5 stale drain).
        """
        q = forkserver_context.Queue(maxsize=1024)
        from queue import Empty

        def drain_empty():
            count = 0
            while True:
                try:
                    q.get_nowait()
                    count += 1
                except Empty:
                    break
            return count

        result = benchmark.pedantic(drain_empty, rounds=100, warmup_rounds=10)
        assert result == 0, "Empty queue should drain zero items"


# ===================================================================
# Step 3.3: IPC Serialization Overhead
# ===================================================================


@pytest.mark.performance
class TestIPCSerialization:
    """Benchmark pickle serialization/deserialization for task tuples containing torch tensors.

    These measurements isolate the IPC cost of sending tasks through multiprocessing.Queue,
    which internally uses pickle. No multiprocessing is needed -- just raw pickle timing.
    """

    @pytest.mark.parametrize("sample_count", [50, 200, 1000])
    @pytest.mark.parametrize("input_size", [2, 10, 50])
    def test_task_tuple_pickle_roundtrip(self, benchmark, sample_count, input_size):
        """Measure pickle.dumps + pickle.loads time for a task tuple.

        Task tuple structure matches CascadeCorrelationNetwork task format:
        (candidate_index, candidate_data_dict, training_inputs_tuple)
        """
        torch.manual_seed(42)
        output_size = 2

        # Build a representative task tuple
        candidate_data = {
            "activation_fn": torch.nn.Tanh(),
            "candidate_epochs": 30,
            "candidate_learning_rate": 0.005,
            "input_size": input_size,
            "candidate_seed": 42,
            "random_max_value": 1.0,
            "random_value_scale": 0.1,
            "sequence_max_value": 100,
            "candidate_uuid": "test-uuid-1234",
            "candidate_display_frequency": 0,
            "candidate_index": 0,
        }
        training_inputs = (
            torch.randn(sample_count, input_size),   # candidate_input (x)
            30,                                        # epochs
            torch.randn(sample_count, output_size),   # y targets
            torch.randn(sample_count, output_size),   # residual_error
            0.005,                                     # learning_rate
            0,                                         # display_frequency
        )
        task_tuple = (0, candidate_data, training_inputs)

        def roundtrip():
            data = pickle.dumps(task_tuple)
            return pickle.loads(data)

        result = benchmark.pedantic(roundtrip, rounds=10, warmup_rounds=2)
        assert result is not None

    @pytest.mark.parametrize("sample_count", [50, 200, 1000])
    @pytest.mark.parametrize("input_size", [2, 10, 50])
    def test_tensor_pickle_dumps_size(self, sample_count, input_size):
        """Measure serialized byte size for tensors of varying dimensions.

        Reports the pickle payload size that flows through the OS pipe backing
        multiprocessing.Queue. Larger payloads increase IPC latency.
        """
        torch.manual_seed(42)
        output_size = 2

        x = torch.randn(sample_count, input_size)
        y = torch.randn(sample_count, output_size)
        residual = torch.randn(sample_count, output_size)

        training_inputs = (x, 30, y, residual, 0.005, 0)

        timer = BenchmarkTimer()
        sizes = []

        for _ in range(10):
            with timer:
                data = pickle.dumps(training_inputs)
            sizes.append(len(data))

        summary = timer.summary()
        avg_size = np.mean(sizes)
        print(f"\n[Step 3.3] Tensor pickle size: samples={sample_count}, input_size={input_size}")
        print(f"  Serialized size: {avg_size / 1024:.1f} KB")
        print(f"  Dumps time: mean={summary['mean_ms']:.3f} ms, min={summary['min_ms']:.3f} ms")
        print(f"  Throughput: {avg_size / 1024 / 1024 / (summary['mean_ms'] / 1000):.1f} MB/s")

        assert avg_size > 0, "Serialized payload must have nonzero size"
        assert summary["mean_ms"] > 0, "Serialization must take measurable time"

    def test_pickle_dumps_vs_loads_asymmetry(self, benchmark):
        """Compare dumps vs loads cost for a medium-sized task tuple.

        Deserialization (loads) is often slower than serialization (dumps) for
        tensors because it involves memory allocation and copy.
        """
        torch.manual_seed(42)
        x = torch.randn(200, 10)
        y = torch.randn(200, 2)
        residual = torch.randn(200, 2)
        training_inputs = (x, 30, y, residual, 0.005, 0)
        task_tuple = (0, {"activation_fn": torch.nn.Tanh(), "input_size": 10}, training_inputs)

        # Pre-serialize for loads benchmark
        serialized = pickle.dumps(task_tuple)

        dumps_timer = BenchmarkTimer()
        loads_timer = BenchmarkTimer()

        for _ in range(20):
            with dumps_timer:
                pickle.dumps(task_tuple)
            with loads_timer:
                pickle.loads(serialized)

        dumps_summary = dumps_timer.summary()
        loads_summary = loads_timer.summary()

        print(f"\n[Step 3.3] Pickle asymmetry (200 samples, input_size=10):")
        print(f"  dumps: mean={dumps_summary['mean_ms']:.3f} ms")
        print(f"  loads: mean={loads_summary['mean_ms']:.3f} ms")
        print(f"  loads/dumps ratio: {loads_summary['mean_ms'] / max(dumps_summary['mean_ms'], 1e-6):.2f}x")
        print(f"  Payload size: {len(serialized) / 1024:.1f} KB")

        assert dumps_summary["mean_ms"] > 0
        assert loads_summary["mean_ms"] > 0


# ===================================================================
# Step 3.4: Worker Startup and Pool Lifecycle
# ===================================================================


@pytest.mark.performance
@pytest.mark.multiprocessing
class TestWorkerLifecycle:
    """Benchmark worker pool creation, reuse, and shutdown costs.

    Tests _ensure_worker_pool cold start, warm reuse, _shutdown_worker_pool,
    and first-op torch latency in a subprocess.
    """

    @pytest.mark.timeout(30)
    def test_cold_start_time(self, real_process_count, monkeypatch):
        """Measure time for initial _ensure_worker_pool call (cold start).

        Cold start includes forkserver process creation, Python interpreter
        initialization in each subprocess, and PyTorch lazy init.
        """
        monkeypatch.setenv("CASCOR_NUM_PROCESSES", "2")
        net = _make_scaling_network(candidate_pool_size=4, candidate_epochs=10)

        timer = BenchmarkTimer()
        try:
            with timer:
                task_queue, result_queue = net._ensure_worker_pool(2)

            summary = timer.summary()
            print(f"\n[Step 3.4] Cold start (2 workers): {summary['mean_ms']:.1f} ms")

            assert task_queue is not None
            assert result_queue is not None
            assert len(net._persistent_workers) == 2
            assert all(w.is_alive() for w in net._persistent_workers)
        finally:
            net._shutdown_worker_pool()

    @pytest.mark.timeout(30)
    def test_warm_reuse_time(self, real_process_count, monkeypatch):
        """Measure time for subsequent _ensure_worker_pool call (warm reuse).

        When the pool already exists with the correct size and all workers are
        alive, _ensure_worker_pool should return almost instantly.
        """
        monkeypatch.setenv("CASCOR_NUM_PROCESSES", "2")
        net = _make_scaling_network(candidate_pool_size=4, candidate_epochs=10)

        try:
            # Cold start
            net._ensure_worker_pool(2)

            # Warm reuse (multiple iterations to get stable timing)
            timer = BenchmarkTimer()
            for _ in range(20):
                with timer:
                    task_queue, result_queue = net._ensure_worker_pool(2)

            summary = timer.summary()
            print(f"\n[Step 3.4] Warm reuse (2 workers, 20 iterations):")
            print(f"  mean={summary['mean_ms']:.3f} ms, median={summary['median_ms']:.3f} ms, min={summary['min_ms']:.3f} ms")

            # Warm reuse should be significantly faster than cold start
            assert summary["median_ms"] < 10.0, "Warm reuse should be under 10ms"
            assert task_queue is not None
            assert result_queue is not None
        finally:
            net._shutdown_worker_pool()

    @pytest.mark.timeout(30)
    def test_shutdown_time(self, real_process_count, monkeypatch):
        """Measure _shutdown_worker_pool time (sentinel send + join + cleanup)."""
        monkeypatch.setenv("CASCOR_NUM_PROCESSES", "2")

        results = []

        for trial in range(3):
            net = _make_scaling_network(candidate_pool_size=4, candidate_epochs=10)
            net._ensure_worker_pool(2)

            timer = BenchmarkTimer()
            with timer:
                net._shutdown_worker_pool()

            results.append(timer.summary()["mean_ms"])

        mean_ms = np.mean(results)
        min_ms = np.min(results)
        max_ms = np.max(results)

        print(f"\n[Step 3.4] Shutdown (2 workers, 3 trials):")
        print(f"  mean={mean_ms:.1f} ms, min={min_ms:.1f} ms, max={max_ms:.1f} ms")

        # Shutdown should complete within the join timeout (5s per worker + overhead)
        assert mean_ms < 15000, "Shutdown should complete within 15 seconds"

    @pytest.mark.timeout(30)
    @pytest.mark.parametrize("num_workers", [1, 2, 4])
    def test_pool_size_cold_start_scaling(self, num_workers, real_process_count, monkeypatch):
        """Measure how cold start time scales with worker count."""
        monkeypatch.setenv("CASCOR_NUM_PROCESSES", str(num_workers))
        net = _make_scaling_network(candidate_pool_size=max(num_workers, 4), candidate_epochs=10)

        timer = BenchmarkTimer()
        try:
            with timer:
                net._ensure_worker_pool(num_workers)

            summary = timer.summary()
            print(f"\n[Step 3.4] Cold start scaling: {num_workers} workers = {summary['mean_ms']:.1f} ms")

            assert len(net._persistent_workers) == num_workers
            assert all(w.is_alive() for w in net._persistent_workers)
        finally:
            net._shutdown_worker_pool()

    @pytest.mark.timeout(30)
    def test_subprocess_first_torch_op_latency(self, forkserver_context):
        """Measure the latency of the first torch operation in a subprocess.

        This captures the one-time PyTorch lazy initialization cost that each
        worker pays on its first training iteration. The forkserver context
        pre-loads modules but torch internals still have lazy init paths.
        """
        result_queue = forkserver_context.Queue(maxsize=16)

        timer = BenchmarkTimer()
        with timer:
            p = forkserver_context.Process(
                target=_subprocess_torch_timing,
                args=(result_queue,),
                daemon=True,
            )
            p.start()
            p.join(timeout=20.0)

        total_ms = timer.summary()["mean_ms"]

        # Get timing from the subprocess
        from queue import Empty

        try:
            first_ns, warm_ns = result_queue.get(timeout=5.0)
            first_ms = first_ns / 1_000_000
            warm_ms = warm_ns / 1_000_000

            print(f"\n[Step 3.4] Subprocess torch op latency:")
            print(f"  Process spawn + join: {total_ms:.1f} ms")
            print(f"  First torch op: {first_ms:.3f} ms")
            print(f"  Warm torch op: {warm_ms:.3f} ms")
            print(f"  Cold/warm ratio: {first_ms / max(warm_ms, 1e-6):.1f}x")
        except Empty:
            pytest.fail("Subprocess did not report timing results within timeout")

    @pytest.mark.timeout(30)
    def test_pool_recreation_after_shutdown(self, real_process_count, monkeypatch):
        """Measure cost of destroying and recreating the worker pool.

        This exercises the full lifecycle: create -> use -> shutdown -> recreate,
        which occurs when the pool size changes between training rounds.
        """
        monkeypatch.setenv("CASCOR_NUM_PROCESSES", "2")
        net = _make_scaling_network(candidate_pool_size=4, candidate_epochs=10)

        create_times = []
        shutdown_times = []

        try:
            for cycle in range(3):
                ct = BenchmarkTimer()
                with ct:
                    net._ensure_worker_pool(2)
                create_times.append(ct.summary()["mean_ms"])

                st = BenchmarkTimer()
                with st:
                    net._shutdown_worker_pool()
                shutdown_times.append(st.summary()["mean_ms"])

            print(f"\n[Step 3.4] Pool lifecycle (3 create/destroy cycles, 2 workers):")
            for i in range(3):
                print(f"  Cycle {i + 1}: create={create_times[i]:.1f} ms, shutdown={shutdown_times[i]:.1f} ms")
            print(f"  Avg create: {np.mean(create_times):.1f} ms")
            print(f"  Avg shutdown: {np.mean(shutdown_times):.1f} ms")
            print(f"  Avg full cycle: {np.mean(create_times) + np.mean(shutdown_times):.1f} ms")
        finally:
            # Ensure cleanup even if assertions fail
            net._shutdown_worker_pool()

        assert len(create_times) == 3, "All 3 cycles should complete"
