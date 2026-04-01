# Juniper-CasCor Performance Testing Plan

**Date**: 2026-03-31
**Version**: 1.1.0
**Status**: Draft (validated 2026-03-31)
**Scope**: juniper-cascor application performance testing, profiling, and optimization

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Architecture Context](#architecture-context)
- [Constraints and Guardrails](#constraints-and-guardrails)
- [Performance Testing Tool Options](#performance-testing-tool-options)
- [Phase 1: Baseline Profiling Infrastructure](#phase-1-baseline-profiling-infrastructure)
- [Phase 2: Component Micro-benchmarks](#phase-2-component-micro-benchmarks)
- [Phase 3: Concurrency and Scaling Analysis](#phase-3-concurrency-and-scaling-analysis)
- [Phase 4: End-to-End Training Profiling](#phase-4-end-to-end-training-profiling)
- [Phase 5: Optimization and Reporting](#phase-5-optimization-and-reporting)
- [Performance Report Data](#performance-report-data)
- [Key Optimization Opportunities](#key-optimization-opportunities)
- [Priority Summary](#priority-summary)
- [Validation Notes](#validation-notes)

---

## Executive Summary

This plan defines a systematic approach to performance testing the juniper-cascor Cascade Correlation Neural Network training engine. The application uses a multiprocessing-based parallelization model with persistent forkserver workers, queue-based task distribution, and optional remote WebSocket workers. Five rounds of parallel-fix optimizations (RC-1 through RC-5) have already been applied.

The plan is organized into 5 phases spanning infrastructure setup, micro-benchmarking of hot paths, concurrency scaling analysis, end-to-end profiling, and targeted optimization implementation.

### Goals

1. Establish reproducible performance baselines for all critical code paths
2. Identify remaining bottlenecks in the training pipeline
3. Quantify scaling efficiency of the multiprocessing worker pool
4. Produce actionable optimization recommendations with measurable impact predictions
5. Create a regression-safe performance testing infrastructure integrated with CI/CD

---

## Architecture Context

### Computational Pipeline

```text
fit()
 └─ grow_network()                          # Main growth loop (per hidden unit)
     ├─ calculate_residual_error()           # Forward pass + subtraction
     ├─ train_candidates()                   # Parallel candidate evaluation
     │   ├─ _prepare_candidate_input()       # Concatenate input + hidden outputs
     │   ├─ _generate_candidate_tasks()      # Create pool_size task tuples
     │   ├─ _calculate_optimal_process_count()
     │   └─ _execute_candidate_training()    # Dispatch to workers
     │       ├─ Sequential path              # process_count <= 1
     │       ├─ Parallel path (local)        # Persistent forkserver pool
     │       │   └─ _worker_loop()           # Queue-based task/result
     │       │       └─ train_candidate_worker()
     │       │           └─ CandidateUnit.train_detailed()
     │       │               ├─ forward()            # HOT PATH #1
     │       │               ├─ _calculate_correlation()  # HOT PATH #2
     │       │               └─ _update_weights_and_bias() # HOT PATH #3
     │       └─ Remote path (WebSocket)      # TaskDistributor overflow
     ├─ _process_training_results()          # Rank candidates by correlation
     ├─ add_unit()                           # Install best candidate
     └─ train_output_layer()                 # HOT PATH #4: Retrain outputs
```

### Key Performance Constants

| Constant                    | Default | Location                           |
|-----------------------------|---------|------------------------------------|
| `candidate_pool_size`       | 16      | `constants_candidates.py`          |
| `candidate_epochs`          | 100     | `constants_candidates.py`          |
| `worker_thread_count`       | 1       | `constants_model.py:75` (RC-1 pin) |
| `task_queue_timeout`        | 5.0s    | `constants_model.py:67`            |
| `shutdown_timeout`          | 10.0s   | `constants_model.py:68`            |
| `worker_standby_sleepytime` | 2.0s    | `constants_model.py:66`            |
| `_QUEUE_MAXSIZE`            | 1024    | `cascade_correlation.py:187`       |

### Existing Profiling Infrastructure

| Tool                               | Location                                | Status      |
|------------------------------------|-----------------------------------------|-------------|
| cProfile decorator/context manager | `src/profiling/deterministic.py`        | Implemented |
| py-spy sampling profiler wrapper   | `util/profile_training.bash`            | Implemented |
| Benchmark harness (bash)           | `src/tests/scripts/run_benchmarks.bash` | Implemented |
| `measure_training_time()`          | `src/tests/helpers/utilities.py`        | Implemented |
| `monitor_memory()` context manager | `src/tests/helpers/utilities.py`        | Implemented |
| `--profile` CLI flag               | `src/main.py`                           | Implemented |
| `--profile-memory` CLI flag        | `src/main.py`                           | Implemented |

---

## Constraints and Guardrails

### Immutable Architecture (Do Not Modify)

The following core design elements are requirements and must not be changed:

1. **Python `multiprocessing` library** for candidate training concurrency
2. **Persistent forkserver** context for worker process spawning
3. **Queue-based task/result management** (`multiprocessing.Queue`)
4. **Remote worker processing** via WebSocket protocol
5. **Process manager pattern** (`CandidateTrainingManager`)
6. **TaskDistributor** local-first scheduling with remote overflow

### Testing Guardrails

- Performance tests must be **deterministic** (seeded randomness) for reproducible baselines
- Tests must not **starve CI runners** -- use timeout limits and resource caps
- Multiprocessing tests must use the **existing `force_sequential_training` fixture** or explicitly opt out with the `@pytest.mark.multiprocessing` marker
- Performance tests must **not modify production code** -- use external profiling, fixtures, and configuration overrides
- All benchmarks must report **statistical summaries** (mean, stddev, min, max, iterations) not single-run values
- Benchmarks must specify **log level**: use `CASCOR_LOG_LEVEL=WARNING` to suppress hot-path logging overhead (the codebase has 15+ logger calls per correlation computation; at TRACE/DEBUG these involve string formatting with tensor values that can add 5-20% overhead)
- Benchmarks must run both **with and without GC** (`--benchmark-disable-gc`) to separate computation time from GC pressure -- GC pressure is itself an optimization target (see OPT-1)
- Micro-benchmarks must report **wall time**; for multiprocessing benchmarks, explicitly state whether speedup is computed from wall time (includes IPC wait) or CPU time
- Cross-process timing with `time.perf_counter_ns()` is **per-process monotonic only** -- measure put latency from the parent side, get latency from the worker side; do not compute cross-process durations
- Collect baselines at **both `OMP_NUM_THREADS=1` (isolation) and `OMP_NUM_THREADS=2` (production default)** as `main.py` sets `OMP_NUM_THREADS=2` at startup
- For stress/long benchmarks: insert **cooldown periods** between runs and discard the first stress run as thermal stabilization; optionally monitor CPU frequency via `/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq`

---

## Performance Testing Tool Options

### Option A: pytest-benchmark (Recommended for Micro-benchmarks)

**Description**: pytest plugin that provides statistical benchmarking with calibration, warmup, and comparison between runs.

| Aspect         | Assessment                                                                                                                                                                                         |
|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Strengths**  | Integrate existing pytest infra; auto stats analysis (mean, stddev, rounds, iterations); comparison between runs `--benchmark-compare`; JSON for CI/CD regression track; pedantic calibration mode |
| **Weaknesses** | Overhead per benchmark iteration; not suitable for long-running (>30s) benchmarks; limited multiprocessing support                                                                                 |
| **Risks**      | Benchmark timer resolution may hide sub-microsecond differences; noisy CI environments can produce unstable results                                                                                |
| **Guardrails** | Use `--benchmark-min-rounds=5`; run on dedicated hardware or use `--benchmark-warmup=on`; pin `--benchmark-disable-gc` for consistency                                                             |
| **Best For**   | Component-level micro-benchmarks: forward pass, correlation calculation, weight update, serialization                                                                                              |

**Alternative**: The existing `run_benchmarks.bash` harness provides similar functionality with manual timing. pytest-benchmark adds statistical rigor and CI integration but requires an additional dependency.

**Recommendation**: **Use pytest-benchmark for new micro-benchmarks** while keeping `run_benchmarks.bash` for quick ad-hoc measurements. The two complement each other.

### Option B: py-spy (Recommended for System Profiling)

**Description**: Sampling profiler that attaches to running Python processes with minimal overhead. Already partially integrated via `util/profile_training.bash`.

| Aspect         | Assessment                                                                                                                                                              |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Strengths**  | Near-zero overhead (~2-5%); supports multiprocessing via `--subprocesses`; flame graph and speedscope output; no code changes required; works with forkserver processes |
| **Weaknesses** | Sampling can miss short-lived functions; requires `sudo` or `SYS_PTRACE` capability; line-level attribution can be noisy                                                |
| **Risks**      | May not capture queue contention or IPC delays (these are kernel-level); security restrictions on CI runners may prevent attachment                                     |
| **Guardrails** | Use `--rate 200` minimum for fine-grained profiling; always profile with `--subprocesses` for multiprocessing; compare wall time vs CPU time                            |
| **Best For**   | Full training run profiling, identifying which functions consume the most wall-clock time, visualizing call stacks across workers                                       |

**Alternative**: cProfile (already implemented in `src/profiling/deterministic.py`) provides per-function granularity but with 10-30% overhead and no multiprocessing support.

**Recommendation**: **Use py-spy for system-level profiling**, cProfile `ProfileContext` for targeted function-level profiling where overhead is acceptable.

### Option C: scalene (Consider for CPU + Memory Combined)

**Description**: High-performance CPU, GPU, and memory profiler with line-level granularity.

| Aspect         | Assessment                                                                                                                                                                                     |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Strengths**  | Line-level CPU and memory profiling simultaneously; separates Python vs C time; low overhead (< 10%); identifies memory leaks                                                                  |
| **Weaknesses** | May not work reliably with forkserver multiprocessing; requires Python 3.8+; less mature than py-spy for process attachment                                                                    |
| **Risks**      | Forkserver compat **likely broken** -- scalene's proc attach model conflicts w/ forkserver's server process arch & preload module set (torch, numpy, etc.); interfere torch thread mgmt (RC-1) |
| **Guardrails** | **Do not use for multiprocessing profiling** -- limit to single-process mode only; test compatibility with forkserver before any use; validate against py-spy results                          |
| **Best For**   | Identifying combined CPU + memory bottlenecks in single-process mode; memory leak detection during long training                                                                               |

**Alternative**: tracemalloc (built-in) for memory-only profiling with `--profile-memory` flag already implemented.

**Recommendation**: **Evaluate scalene as a secondary tool** after establishing py-spy baselines. Use tracemalloc for memory-specific investigations.

### Option D: Custom Timing Instrumentation (Recommended for IPC/Queue Analysis)

**Description**: Targeted timing decorators and context managers placed around specific code paths, particularly IPC boundaries that sampling profilers cannot observe.

| Aspect         | Assessment                                                                                                                                                 |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Strengths**  | Works across multiprocessing boundaries; measures queue put/get latency directly; zero-dependency; can be enabled/disabled via config flag                 |
| **Weaknesses** | Manual instrumentation effort; must be careful not to introduce overhead that distorts results; risk of accidentally committing debug timing code          |
| **Risks**      | Timing code that touches multiprocessing.Queue internals could introduce subtle bugs; clock synchronization across processes                               |
| **Guardrails** | Use `time.perf_counter_ns()` for precision; gate behind environment variable (`CASCOR_PERF_TIMING=1`); never commit timing code to hot paths without guard |
| **Best For**   | Queue throughput measurement; IPC latency; worker startup/teardown timing; task serialization overhead                                                     |

**Alternative**: None -- sampling profilers cannot observe queue wait times or IPC latency directly.

**Recommendation**: **Implement targeted timing for IPC/queue analysis** as part of Phase 3. This fills a gap that no external tool covers.

### Tool Selection Summary

| Use Case                                      | Primary Tool                     | Secondary Tool            |
|-----------------------------------------------|----------------------------------|---------------------------|
| Micro-benchmarks (forward, correlation, etc.) | pytest-benchmark                 | `run_benchmarks.bash`     |
| Full training profiling                       | py-spy (`--subprocesses`)        | cProfile `ProfileContext` |
| Memory profiling                              | tracemalloc (`--profile-memory`) | scalene (evaluate)        |
| IPC / Queue analysis                          | Custom timing instrumentation    | py-spy flame graphs       |
| CI regression detection                       | pytest-benchmark JSON output     | None                      |

---

## Phase 1: Baseline Profiling Infrastructure

**Goal**: Establish reproducible performance baselines and enhance the profiling toolkit.

### Step 1.1: Configure Deterministic Benchmark Environment

**Tasks**:

1. **Create benchmark fixture set** in `src/tests/conftest.py` or a dedicated `src/tests/performance/conftest.py`:
   - Fixed random seeds (torch, numpy, python random)
   - Deterministic torch mode (`torch.use_deterministic_algorithms(True)`)
   - Controlled dataset sizes: small (50 samples), medium (200 samples), large (1000 samples)
   - Standard network configurations: untrained (0 hidden), small (5 hidden), medium (15 hidden)
   - PyTorch thread count pinned via `torch.set_num_threads(1)` for single-thread baselines

2. **Create standardized datasets** for benchmarking:
   - 2-spiral classification (default CasCor benchmark problem)
   - Linear regression (minimal computation, isolates overhead)
   - High-dimensional input (input_size=20, tests scaling with feature count)

3. **Environment variable gating**:
   - `CASCOR_BENCHMARK_MODE=1` to enable performance test collection
   - Prevents accidental inclusion in standard test runs

### Step 1.2: Establish Sequential Baselines

**Tasks**:

1. **Collect baseline timings** for each hot path in single-threaded mode:
   - `CandidateUnit.train_detailed()` (1 candidate, 100 epochs, spiral data)
   - `CascadeCorrelationNetwork.forward()` (varying hidden unit counts: 0, 5, 10, 20, 50)
   - `CascadeCorrelationNetwork.train_output_layer()` (varying hidden unit counts)
   - `CandidateUnit._calculate_correlation()` (varying sample counts)
   - `CascadeCorrelationNetwork.calculate_residual_error()` (varying network sizes)

2. **Record baseline memory footprint**:
   - Base network memory (0 hidden units)
   - Per-hidden-unit memory increment
   - Peak memory during candidate training round (all pool_size candidates in-flight)
   - Worker process memory footprint

3. **Save baselines to JSON** for regression comparison:
   - File: `src/tests/performance/baselines/baseline_YYYYMMDD.json`
   - Schema: `{test_name, mean_ms, stddev_ms, min_ms, max_ms, iterations, environment}`

### Step 1.3: Enhance Profiling Toolkit

**Tasks**:

1. **Extend `ProfileContext`** with JSON export:
   - Add `to_dict()` method returning structured profile data
   - Add `save_json()` for machine-readable output
   - Add elapsed wall time alongside cProfile CPU time

2. **Create performance test marker and collection**:
   - Add `@pytest.mark.performance` marker (already defined in pytest.ini)
   - Create `src/tests/performance/` directory for dedicated performance tests
   - Configure pytest to skip performance tests by default (require `--run-performance` flag)

---

## Phase 2: Component Micro-benchmarks

**Goal**: Quantify the performance of each computational hot path in isolation.

### Step 2.1: Candidate Training Micro-benchmarks

**Benchmark**: `CandidateUnit.train_detailed()`

**Test Matrix**:

| Parameter      | Values              | Purpose                             |
|----------------|---------------------|-------------------------------------|
| `epochs`       | 10, 50, 100, 200    | Measure per-epoch cost              |
| `input_size`   | 2, 10, 50           | Scaling with feature dimensionality |
| `sample_count` | 50, 200, 1000       | Scaling with dataset size           |
| `activation`   | tanh, sigmoid, relu | Activation function overhead        |

**Metrics Captured**:

- Total training time (ms)
- Per-epoch time (ms)
- Final correlation value (validates correctness)
- Peak memory during training

**Expected Output**: O(samples x input_size x epochs) scaling confirmation.

### Step 2.2: Forward Pass Scaling Benchmarks

**Benchmark**: `CascadeCorrelationNetwork.forward()`

**Test Matrix**:

| Parameter      | Values           | Purpose                |
|----------------|------------------|------------------------|
| `hidden_units` | 0, 5, 10, 20, 50 | Cascade depth scaling  |
| `sample_count` | 50, 200, 1000    | Batch size impact      |
| `input_size`   | 2, 10, 50        | Feature dimensionality |

**Key Question**: Does forward pass scale linearly with hidden_units, or is there quadratic behavior from the cascading concatenation pattern?

The CasCor forward pass builds incrementally: each hidden unit receives input + all previous hidden outputs. This means hidden unit N receives (input_size + N-1) features, creating O(N^2) total feature processing across all units.

**Metrics Captured**:

- Forward pass time (us)
- Time per hidden unit (us) -- expect linear growth due to cascade
- Memory allocation per pass

### Step 2.3: Correlation Calculation Benchmarks

**Benchmark**: `CandidateUnit._calculate_correlation()`

**Test Matrix**:

| Parameter      | Values              | Purpose                           |
|----------------|---------------------|-----------------------------------|
| `sample_count` | 50, 200, 1000, 5000 | Scaling with data size            |
| `output_size`  | 1, 2, 5, 10         | Multi-output correlation overhead |

**Metrics Captured**:

- Correlation computation time (us)
- Time breakdown: mean-centering, dot product, normalization
- Numerical precision at large sample counts

### Step 2.4: Output Layer Training Benchmarks

**Benchmark**: `CascadeCorrelationNetwork.train_output_layer()`

**Test Matrix**:

| Parameter       | Values           | Purpose             |
|-----------------|------------------|---------------------|
| `hidden_units`  | 0, 5, 10, 20, 50 | Network size impact |
| `output_epochs` | 10, 50, 100      | Epoch count scaling |
| `sample_count`  | 50, 200, 1000    | Dataset size impact |

**Key Question**: How does output training time scale as the network grows? The output layer width grows with each hidden unit addition.

### Step 2.5: Weight Update and Autograd Overhead

**Benchmark**: `CandidateUnit._update_weights_and_bias()`

This method is called `candidate_pool_size * candidate_epochs` times per growth cycle (e.g., 16 x 100 = 1,600 times). It performs `.clone().detach().requires_grad_(True)` followed by a forward pass, `.backward()`, and gradient application -- building and destroying a PyTorch autograd graph **every single iteration**.

**Test Matrix**:

| Parameter    | Values         | Purpose                          |
|--------------|----------------|----------------------------------|
| `input_size` | 2, 10, 50      | Gradient computation scaling     |
| `iterations` | 100, 500, 1000 | Amortized overhead per iteration |

**Metrics Captured**:

- Per-iteration autograd graph construction time (us)
- Per-iteration `.backward()` time (us)
- Memory growth over iterations (detect graph fragment leaks)
- Comparison: autograd overhead vs forward pass + correlation time

**Key Question**: Does the per-iteration clone+detach+requires_grad+backward pattern introduce significant overhead compared to the numerical computation itself? Could graph reuse or `torch.no_grad()` partitioning reduce this cost?

### Step 2.6: Serialization Benchmarks (Existing + Enhanced)

**Benchmark**: HDF5 save/load via `run_benchmarks.bash` + new pytest-benchmark equivalents

**Test Matrix** (extends existing harness):

| Parameter               | Values          | Purpose                 |
|-------------------------|-----------------|-------------------------|
| `hidden_units`          | 0, 10, 50, 100  | Model size impact       |
| `compression`           | None, gzip, lzf | Compression overhead    |
| `include_training_data` | True, False     | Data inclusion overhead |

**Metrics Captured**:

- Save time (ms), Load time (ms), Verify time (ms)
- File size (bytes)
- Compression ratio

---

## Phase 3: Concurrency and Scaling Analysis

**Goal**: Quantify the parallel efficiency of the multiprocessing worker pool and identify scaling bottlenecks.

### Step 3.1: Worker Pool Scaling Benchmarks

**Benchmark**: Complete candidate training round with varying worker counts.

**Test Matrix**:

| Parameter             | Values              | Purpose                   |
|-----------------------|---------------------|---------------------------|
| `process_count`       | 1, 2, 4, 8, N_cores | Scaling efficiency        |
| `candidate_pool_size` | 4, 8, 16, 32        | Work saturation point     |
| `candidate_epochs`    | 50, 100, 200        | Per-task work granularity |

**Metrics Captured**:

- Total round time (ms)
- Speedup factor vs sequential (S = T_sequential / T_parallel)
- Efficiency (E = S / N_workers)
- Per-worker utilization (active time / total time)

**Key Questions**:

1. At what worker count does scaling plateau?
2. What is the minimum task granularity for parallelism to be beneficial?
3. Does queue contention appear at high worker counts?

**Implementation Note**: This step requires real multiprocessing processes (not mocked). Tests must explicitly use `@pytest.mark.multiprocessing` and skip the `force_sequential_training` fixture. Use `CASCOR_NUM_PROCESSES` environment variable to control worker count.

### Step 3.2: Queue Throughput Analysis

**Benchmark**: Measure queue put/get latency under varying loads.

**Approach**: Custom timing instrumentation (Option D from tool selection).

**Test Scenarios**:

| Scenario       | Description                              | Metric                       |
|----------------|------------------------------------------|------------------------------|
| Uncontested    | Single producer, single consumer         | Put/Get latency (us)         |
| N-producer     | N workers writing results simultaneously | Get latency under contention |
| Saturation     | Queue near MAXSIZE (1024)                | Put blocking time            |
| Drain overhead | Stale result drain (RC-5)                | Drain time vs queue depth    |

**Implementation**:

- Wrap `task_queue.put()` and `result_queue.get()` with `time.perf_counter_ns()` guards
- Gate behind `CASCOR_PERF_TIMING=1` environment variable
- Log timing data to structured JSON for post-processing

### Step 3.3: IPC Serialization Overhead

**Benchmark**: Measure the cost of serializing task tuples through multiprocessing.Queue.

**What Gets Serialized Per Task**:

| Component                | Approximate Size                  | Serialization Method  |
|--------------------------|-----------------------------------|-----------------------|
| `candidate_data` tuple   | ~200 bytes                        | pickle                |
| `candidate_input` tensor | 4 x N_samples x input_size bytes  | pickle (torch.Tensor) |
| `y` tensor               | 4 x N_samples x output_size bytes | pickle (torch.Tensor) |
| `residual_error` tensor  | 4 x N_samples x output_size bytes | pickle (torch.Tensor) |

**Test Matrix**:

| Parameter       | Values              | Purpose                  |
|-----------------|---------------------|--------------------------|
| `sample_count`  | 50, 200, 1000, 5000 | Payload size scaling     |
| `input_size`    | 2, 10, 50           | Feature dimensionality   |
| `shared_inputs` | True, False         | RC-3 optimization impact |

**Key Question**: At what dataset size does IPC serialization overhead dominate per-candidate training time?

### Step 3.4: Worker Startup and Pool Lifecycle

**Benchmark**: Measure persistent pool (RC-4) vs per-round pool creation.

**Metrics**:

- Cold start time: first `_ensure_worker_pool()` call (process creation + forkserver init)
- Warm reuse time: subsequent calls (no new processes)
- Pool resize overhead (adding/removing workers)
- Shutdown time: `_shutdown_worker_pool()` with varying worker counts
- **First-op latency per worker**: The first `torch` operation in a forkserver-spawned process triggers BLAS library re-initialization (MKL/OpenBLAS), which can take 50-200ms. Additionally, `torch.set_num_threads()` is called in `_worker_loop()` (line 2945), which triggers BLAS re-initialization. Measure first-op vs subsequent-op latency per worker.

**CPU Affinity Note**: By default, the OS scheduler may migrate worker processes between CPU cores, invalidating L1/L2 caches. For reproducible benchmarks on multi-core systems, consider pinning workers via `os.sched_setaffinity()` or `taskset`. On NUMA systems, cross-node memory access adds latency. Document whether affinity pinning is used in benchmark results.

**GIL Contention Note**: If the FastAPI server is running during benchmarks (e.g., end-to-end API tests), the GIL contention between the event loop thread, coordinator monitor thread, and the main process's queue operations can introduce measurement noise. Phase 3 benchmarks should specify whether they run with or without the API server active.

---

## Phase 4: End-to-End Training Profiling

**Goal**: Profile complete training runs to identify system-level bottlenecks and validate micro-benchmark findings.

### Step 4.1: Full Training Run Profiling

**Approach**: Use py-spy with `--subprocesses` to capture flame graphs across all workers during a complete spiral problem training run.

**Test Configurations**:

| Configuration  | Parameters                                   | Purpose                 |
|----------------|----------------------------------------------|-------------------------|
| Quick baseline | 2-spiral, pool=8, epochs=50, max_hidden=5    | Minimal viable training |
| Standard       | 2-spiral, pool=16, epochs=100, max_hidden=15 | Typical usage           |
| Stress         | 2-spiral, pool=32, epochs=200, max_hidden=30 | Scaling limits          |
| Large data     | 5000 samples, pool=16, epochs=100            | Data-heavy workload     |

**Execution**:

```bash
# Using existing py-spy wrapper
./util/profile_training.bash --subprocesses --rate 200 --duration 120
```

**Output**:

- Flame graph SVG for visual analysis
- Speedscope JSON for interactive exploration
- Top-N functions by cumulative time (extract from py-spy output)

### Step 4.2: Training Phase Time Distribution

**Approach**: Instrument `grow_network()` to capture time spent in each phase.

**Phases to Measure**:

| Phase                         | Method                                                       | Expected % |
|-------------------------------|--------------------------------------------------------------|------------|
| Residual error calculation    | `calculate_residual_error()`                                 | 5-15%      |
| Candidate task preparation    | `_prepare_candidate_input()` + `_generate_candidate_tasks()` | 1-5%       |
| Candidate training (parallel) | `_execute_candidate_training()`                              | 60-80%     |
| Result processing             | `_process_training_results()`                                | 1-3%       |
| Unit addition                 | `add_unit()`                                                 | <1%        |
| Output layer retraining       | `train_output_layer()`                                       | 10-25%     |
| Validation                    | `validate_training()`                                        | 2-5%       |

**Implementation**: Use `ProfileContext` around each phase within `grow_network()`, aggregate across iterations, report as percentage of total training time.

### Step 4.3: Memory Growth Profiling

**Approach**: Track memory over the lifetime of a full training run.

**Metrics**:

- RSS memory at each growth epoch (after each hidden unit addition)
- Peak memory during candidate training rounds
- Memory delta per hidden unit addition
- Worker process memory footprint (via `psutil.Process(pid).memory_info().rss`)

**Implementation**: Use existing `monitor_memory()` context manager from test utilities, extended with per-epoch sampling.

### Step 4.4: Training Convergence vs Performance Trade-offs

**Approach**: Measure how hyperparameter choices affect both convergence quality and training speed.

**Test Matrix**:

| Parameter             | Values           | Measures                      |
|-----------------------|------------------|-------------------------------|
| `candidate_pool_size` | 4, 8, 16, 32, 64 | Speed vs convergence quality  |
| `candidate_epochs`    | 25, 50, 100, 200 | Training depth vs time        |
| `patience`            | 3, 5, 10, 20     | Early stopping aggressiveness |

**Metrics**:

- Time to target accuracy (e.g., 90% on spiral)
- Total training wall time
- Number of hidden units added
- Final accuracy achieved

---

## Phase 5: Optimization and Reporting

**Goal**: Translate profiling data into specific, measurable code optimizations.

### Step 5.1: Bottleneck Identification Framework

For each identified bottleneck, document:

1. **What**: Function/operation and its measured cost
2. **Where**: File path, line numbers, call frequency
3. **Why**: Root cause of the inefficiency
4. **How much**: Percentage of total training time, absolute time
5. **Fix**: Specific code change with predicted impact
6. **Risk**: What could break, regression test strategy

### Step 5.2: Performance Report Generation

**Report Format**: Structured markdown with embedded metrics tables.

**Report Sections**:

1. Environment (hardware, Python version, torch version, dataset)
2. Baseline summary (sequential hot path timings)
3. Scaling analysis (speedup curves, efficiency plots)
4. Memory profile (growth curve, peak analysis)
5. Bottleneck ranking (sorted by impact)
6. Optimization recommendations (prioritized by effect / effort ratio)

### Step 5.3: CI/CD Integration

**Approach**: Add performance regression detection to scheduled CI pipeline.

**Options**:

| Option                           | Description                                                                       | Pros                                                | Cons                                                |
|----------------------------------|-----------------------------------------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|
| pytest-benchmark in scheduled CI | Run benchmarks nightly, compare against stored baselines                          | Automatic regression alerts; statistical comparison | Noisy CI environments; requires baseline management |
| Manual baseline comparison       | Run benchmarks locally, compare against committed baselines                       | Stable results; developer-controlled                | No automatic detection; relies on discipline        |
| Hybrid                           | Nightly CI runs benchmarks and posts results; manual review for regressions > 10% | Best of both; reduces false positives               | More complex setup                                  |

**Recommendation**: Start with **manual baseline comparison** (Option 2), migrate to **hybrid** (Option 3) once baselines stabilize. The existing `scheduled-tests.yml` already runs performance benchmarks nightly -- extend it to save JSON results as artifacts.

---

## Performance Report Data

**Collected**: 2026-03-31
**Environment**: Python 3.14.3, PyTorch (CPU), Linux, OMP_NUM_THREADS=1
**Mode**: Single-threaded sequential, deterministic (seed=42), CASCOR_LOG_LEVEL=WARNING

### Forward Pass Scaling (test_micro_forward_pass.py)

| Hidden Units | Mean (us) | StdDev (us) | Relative | Notes                          |
|--------------|-----------|-------------|----------|--------------------------------|
| 0            | 3,252     | 695         | 1.00x    | Baseline: input-to-output only |
| 5            | 6,979     | -           | 2.15x    | ~745 us per hidden unit        |
| 10           | 2,518     | 2,249       | 0.77x    | Faster than 0 (warmup/caching) |
| 20           | 3,037     | 695         | 0.93x    | Minimal growth from 10         |
| 50           | 6,043     | 1,104       | 1.86x    | Sub-linear scaling confirmed   |

**Key Finding**: Forward pass does **NOT** show quadratic scaling. The 0-hidden baseline includes more overhead proportionally. At 50 hidden units, cost is only ~1.9x the 0-hidden case, confirming that `torch.cat()` concatenation is not the dominant cost. **OPT-1 priority should be reduced.**

| Sample Count | Mean (us) | Notes                                 |
|--------------|-----------|---------------------------------------|
| 50           | 6,138     | 10 hidden units                       |
| 200          | 4,977     | Faster due to better vectorization    |
| 1000         | 1,676     | Even faster -- torch batch efficiency |

**Finding**: Forward pass is **faster** with larger batch sizes (better vectorization amortization).

| Input Size | Mean (us) | Notes                        |
|------------|-----------|------------------------------|
| 2          | 2,587     | 10 hidden units, 100 samples |
| 10         | 3,599     | 1.39x                        |
| 50         | 2,968     | 1.15x (sub-linear)           |

### Autograd Overhead (test_micro_autograd.py)

| Operation                             | Mean (us) | Relative | Notes                                       |
|---------------------------------------|-----------|----------|---------------------------------------------|
| Forward only (no_grad, input_size=2)  | 42        | 1.0x     | Pure computation baseline                   |
| Forward only (no_grad, input_size=10) | 47        | 1.1x     |                                             |
| Autograd cycle (input_size=2)         | 269       | 6.4x     | clone+detach+requires_grad+forward+backward |
| Autograd cycle (input_size=10)        | 306       | 7.3x     |                                             |
| Autograd cycle (input_size=50)        | 356       | 8.5x     |                                             |

**Key Finding**: Autograd overhead is **6-8.5x** the cost of the pure forward computation. For candidate training (100 epochs x pool_size=16 = 1,600 autograd cycles per growth epoch), this is a significant multiplier. **OPT-2 (fused correlation) matters less than the autograd pattern itself.** Memory growth tests confirm no leaks (< 10MB over 1000 iterations).

### Correlation Calculation (test_micro_correlation.py)

| Samples | Output Size | Mean (us) | Notes                                  |
|---------|-------------|-----------|----------------------------------------|
| 50      | 2           | 1,556     |                                        |
| 200     | 2           | 2,181     | 1.4x for 4x data                       |
| 1000    | 2           | 884       | Sub-linear (vectorization)             |
| 5000    | 2           | 1,667     | 1.9x for 5x data                       |
| 1000    | 1           | 19,257    | **Anomalous: 22x slower for 1 output** |
| 1000    | 2           | 709       |                                        |
| 1000    | 5           | 2,529     | 3.6x for 2.5x outputs                  |
| 1000    | 10          | 2,016     | 2.8x for 5x outputs                    |

**Key Finding**: output_size=1 is anomalously slow (19ms vs 0.7ms for output_size=2). This suggests a code path branch for single-output that is significantly less optimized. **Investigate `_calculate_correlation` single-output path.**

### Candidate Training (test_micro_candidate.py)

| Epochs | Input | Samples | Activation | Mean (ms) | Notes                         |
|--------|-------|---------|------------|-----------|-------------------------------|
| 10     | 2     | 100     | tanh       | 95        | Baseline                      |
| 50     | 2     | 100     | tanh       | 417       | 4.4x for 5x epochs (linear)   |
| 100    | 2     | 100     | tanh       | 821       | 8.6x for 10x epochs (linear)  |
| 200    | 2     | 100     | tanh       | 1,706     | 18.0x for 20x epochs (linear) |
| 50     | 2     | 100     | tanh       | 6,072     | Full parametrized version     |
| 50     | 2     | 100     | sigmoid    | 6,682     | 1.10x tanh (10% slower)       |
| 50     | 2     | 100     | relu       | 5,978     | 0.98x tanh (negligible diff)  |
| 50     | 2     | 50      | tanh       | 7,554     |                               |
| 50     | 2     | 200     | tanh       | 12,239    |                               |
| 50     | 2     | 1000    | tanh       | 31,372    |                               |
| 50     | 10    | 100     | tanh       | 10,458    | 1.7x for 5x input features    |
| 50     | 50    | 100     | tanh       | 8,500     | 1.4x for 25x input features   |

**Key Findings**:

- **Epoch scaling is linear** as expected: 10->200 epochs = 18x time.
- **Sample scaling is sub-linear**: 50->1000 samples = 4.2x for 20x data.
- **Input size scaling is sub-linear**: input_size 2->50 = 1.4x for 25x features.
- **Activation function choice has minimal impact** (< 10% difference between tanh/sigmoid/relu).

### Output Layer Training (test_micro_output_training.py)

| Hidden Units | Epochs | Samples | Mean (ms) | Notes                        |
|--------------|--------|---------|-----------|------------------------------|
| 0            | 25     | 100     | 183       | Baseline                     |
| 5            | 25     | 100     | 193       | 1.05x                        |
| 10           | 25     | 100     | 203       | 1.11x                        |
| 20           | 25     | 100     | 215       | 1.18x                        |
| 50           | 25     | 100     | 293       | 1.60x                        |
| 10           | 10     | 100     | 87        | Epoch baseline               |
| 10           | 50     | 100     | 409       | 4.7x for 5x epochs (linear)  |
| 10           | 100    | 100     | 804       | 9.2x for 10x epochs (linear) |
| 10           | 25     | 50      | 137       |                              |
| 10           | 25     | 200     | 347       |                              |
| 10           | 25     | 1000    | 88        | Faster (vectorization)       |

**Key Findings**:

- **Hidden unit scaling is sub-linear**: 0->50 units = only 1.60x. The output layer width grows from 2 to 52, but PyTorch matrix ops scale efficiently. **OPT-3 priority should be reduced.**
- **Epoch scaling is linear** as expected.

### Scaling Analysis

| Workers        | Pool Size         | Round Time (ms) | Speedup | Efficiency | Notes |
|----------------|-------------------|-----------------|---------|------------|-------|
| 1 (sequential) | *pending Phase 3* | -               | 1.00x   | 100%       | -     |

### Memory Profile

| Network State       | RSS (MB) | Delta (MB) | Notes |
|---------------------|----------|------------|-------|
| *pending Phase 3/4* | -        | -          | -     |

---

## Key Optimization Opportunities

Based on architectural analysis, these are the most likely optimization targets. Actual priority will be determined by profiling data from Phases 1-4.

### OPT-1: Forward Pass Cascade Concatenation — **IMPLEMENTED**

**Location**: `cascade_correlation.py` `forward()` and `_prepare_candidate_input()` (fallback path)
**Issue**: Each hidden unit call involved `torch.cat()` concatenation, creating N+1 intermediate tensors.
**Fix Applied**: Pre-allocate a single buffer tensor `[batch_size, input_size + N_hidden]`, copy input features once, then fill hidden unit columns incrementally via `buffer[:, col] = activation(...)`. Eliminates all `torch.cat()` calls. Applied to both `forward()` and `_prepare_candidate_input()` fallback.
**Risk**: Low -- forward pass is a pure computation with no side effects.
**Estimated Improvement**: < 10% forward pass speedup (Phase 2 confirmed sub-linear scaling). Eliminates GC pressure from N intermediate tensor allocations.

### OPT-2: Batch Correlation Computation — **IMPLEMENTED**

**Location**: `candidate_unit.py` `_calculate_correlation()` and `_update_weights_and_bias()`
**Issue**: Correlation used separate `torch.sum(a * b)` and `torch.sqrt(torch.sum(x**2))` — multiple kernel launches and intermediate tensor allocations.
**Fix Applied**: Replaced with `torch.dot()` (single BLAS call, no intermediate tensor) and `torch.linalg.norm()` (single BLAS call, avoids square+sum+sqrt). Also applied to the autograd correlation path in `_update_weights_and_bias()`. Reduced hot-path logging.
**Risk**: Low -- pure numerical computation, same mathematical result.
**Estimated Improvement**: 5-10% per correlation computation (fewer kernel launches, less allocation).

### OPT-3: Output Layer Weight Transfer Overhead

**Location**: `cascade_correlation.py` `train_output_layer()` lines 1449-1554
**Issue**: Creates a temporary `nn.Linear` layer each time, copies weights in and out via `.weight.t()` transposition. This repeated allocation and transposition adds overhead per growth epoch.
**Potential Fix**: Maintain a persistent `nn.Linear` layer that grows in-place when hidden units are added. Avoid weight transposition by storing weights in the expected orientation.
**Risk**: Medium -- touches weight management logic; requires careful validation of gradient flow.
**Estimated Improvement**: ~~10-20% output training speedup~~ **REVISED**: Phase 2 shows output training scales sub-linearly (0->50 hidden = 1.6x). The nn.Linear creation and weight transposition overhead is small relative to the actual optimization loop. Likely < 5% improvement. Deprioritized.

### OPT-4: Candidate Input Preparation Redundancy — **IMPLEMENTED**

**Location**: `cascade_correlation.py` `forward()` line 1445, `_prepare_candidate_input()` lines 1625-1634
**Issue**: Runs a full forward pass to collect hidden unit outputs, then concatenates them. This forward pass is separate from the residual error calculation forward pass -- potential for reuse.
**Fix Applied**: `forward()` caches `output_input` (identical to `candidate_input`) keyed by `x.data_ptr()`. `_prepare_candidate_input()` consumes the cache if valid, clears it after use, and falls back to recomputation if cache is stale or absent. 3 methods modified, no public API changes.
**Risk**: Low -- instance cache with data pointer validation, safe fallback to recomputation.
**Measured Improvement**: `_prepare_candidate_input()` reduced from O(N_hidden × N_samples) to O(1). Micro-benchmark: 22x–1607x speedup on the isolated call depending on network depth (5–50 hidden units) and sample count (200–1000). Total grow_network impact is 5-15% as predicted, scaling with network depth.

### OPT-5: Queue Serialization for Large Datasets

**Location**: `cascade_correlation.py` `_execute_parallel_training()` lines 1791-1929
**Issue**: When RC-3 shared inputs are not used, each task includes full copies of training tensors serialized through the queue. For large datasets, this serialization dominates per-task overhead.
**Potential Fix**: Ensure RC-3 shared training inputs path is always used (pass tensors at pool creation, not per-task). Alternatively, use shared memory (`multiprocessing.shared_memory`) for training tensors.
**Risk**: **High** for `multiprocessing.shared_memory` approach -- forkserver context does not automatically inherit shared memory handles (unlike fork); shared memory block names must be explicitly communicated to workers via queue or startup args; lifecycle management (cleanup, unlinking) is platform-dependent and error-prone. **A proof-of-concept must validate forkserver + shared_memory compatibility before committing to this approach.** Low risk for ensuring the existing RC-3 path is always used.
**Estimated Improvement**: Proportional to dataset size -- negligible for small datasets, potentially 30-50% round time reduction for 5000+ sample datasets.
**Note**: The RC-3 `shared_training_inputs` parameter belongs to `_ensure_worker_pool()` (line 2791) and `_worker_loop()` (line 2907), not `_execute_parallel_training()` directly. The persistent pool path (RC-4) currently passes `shared_training_inputs=None`.

---

## Priority Summary

### Phase Prioritization

| Phase                      | Priority           | Effort            | Prerequisite                                    | Deliverable                                   |
|----------------------------|--------------------|-------------------|-------------------------------------------------|-----------------------------------------------|
| Phase 1 (Infrastructure)   | **P0 -- Critical** | Medium (3-5 days) | None                                            | Reproducible baselines, benchmark fixtures    |
| Phase 2 (Micro-benchmarks) | **P1 -- High**     | Medium (3-5 days) | Phase 1                                         | Hot path timing data, scaling characteristics |
| Phase 3 (Concurrency)      | **P1 -- High**     | High (5-7 days)   | Phase 1                                         | Scaling curves, IPC overhead data             |
| Phase 4 (End-to-End)       | **P2 -- Medium**   | Medium (3-5 days) | Phase 1 (Steps 4.1-4.3); Phases 2, 3 (Step 4.4) | Full training profiles, bottleneck ranking    |
| Phase 5 (Optimization)     | **P2 -- Medium**   | Varies per OPT    | Phase 4                                         | Code changes with measured impact             |

### Optimization Prioritization

| Optimization                           | Priority      | Effort      | Measured/Predicted Impact                            | Risk   |
|----------------------------------------|---------------|-------------|------------------------------------------------------|--------|
| OPT-6 (Correlation single-output path) | **P0 — DONE** | Low         | **37x speedup** (18.24ms → 0.49ms for output_size=1) | Low    |
| OPT-4 (Cached forward pass)            | **P0 — DONE** | Low         | 22x–1607x on isolated call; 5-15% total time         | Low    |
| OPT-5 (Shared memory tensors)          | **P1**        | Medium-High | 30-50% for large data (Phase 3 will quantify)        | High   |
| OPT-2 (Fused correlation)              | **P2 — DONE** | Low         | 5-10% (torch.dot + linalg.norm fusion)               | Low    |
| OPT-1 (Pre-allocated forward)          | **P3 — DONE** | Low         | Eliminates N+1 torch.cat() per forward pass          | Low    |
| OPT-3 (Persistent output layer)        | **P3**        | Medium      | < 5% (**REVISED**: 0->50 hidden = only 1.6x)         | Medium |

### NEW: OPT-6: Correlation Single-Output Path Anomaly — **IMPLEMENTED**

**Location**: `candidate_unit.py` — 15 log calls across `forward()`, `train()`, `_get_correlations()`, `_multi_output_correlation()`, `_update_weights_and_bias()`
**Issue**: Phase 2 benchmarks revealed `output_size=1` took **19.3ms** while `output_size=2` took **0.7ms** — a **27x slowdown**. Root cause: f-string tensor formatting in hot-path log calls triggered PyTorch's 1000-element print threshold (`torch._tensor_str`), which performs expensive formatting even at suppressed log levels because Python evaluates f-string arguments eagerly.
**Fix Applied**: Removed `\n{tensor_value}` patterns from 15 hot-path log calls, replacing with shape/dtype metadata only. Commit `4463217`.
**Risk**: Low — logging-only changes, no computation changes.
**Measured Improvement**: output_size=1 went from 18.24ms to 0.49ms (**37x speedup**). Anomaly eliminated — output_size=1 now comparable to output_size=2.

*Note*: Priorities revised using Phase 2 benchmark data (collected 2026-03-31). The user must grant explicit positive permission before implementing each optimization.

---

## Validation Notes

This plan was validated on 2026-03-31 against the juniper-cascor codebase. Key findings:

### Technical Accuracy Validation

| Claim                          | Status    | Notes                                                                                                          |
|--------------------------------|-----------|----------------------------------------------------------------------------------------------------------------|
| Forward pass cascade (OPT-1)   | **FIXED** | Pre-allocated buffer eliminates `torch.cat()` per hidden unit                                                  |
| Redundant forward pass (OPT-4) | **FIXED** | Cache in `forward()` consumed by `_prepare_candidate_input()` — eliminates redundant pass                      |
| Temporary nn.Linear (OPT-3)    | Verified  | Created at line 1492, weight transposition at lines 1494 and 1540                                              |
| RC-3 shared_training_inputs    | Corrected | Parameter belongs to `_ensure_worker_pool()` and `_worker_loop()`, not `_execute_parallel_training()` directly |
| All performance constants      | Verified  | All values and line numbers confirmed exact                                                                    |
| Test markers and fixtures      | Verified  | `@pytest.mark.performance` in pytest.ini, `force_sequential_training` is autouse                               |
| Profiling infrastructure       | Verified  | All 7 tools/utilities exist as described                                                                       |

### Completeness Validation

| Aspect                                         | Status       | Action Taken                                               |
|------------------------------------------------|--------------|------------------------------------------------------------|
| `_update_weights_and_bias()` autograd overhead | Added        | Step 2.5 added with test matrix and metrics                |
| Logging overhead in hot paths                  | Added        | Guardrail added requiring `CASCOR_LOG_LEVEL=WARNING`       |
| Thermal throttling                             | Added        | Guardrail added for cooldown periods and first-run discard |
| PyTorch lazy init in workers                   | Added        | Step 3.4 first-op latency measurement added                |
| CPU affinity / pinning                         | Added        | Note added to Step 3.4                                     |
| GIL + mixed threading                          | Added        | Note added to Step 3.4                                     |
| GC measurement policy                          | Added        | Guardrail: benchmark both with and without GC              |
| Cross-process clock sync                       | Added        | Guardrail: measure only within-process durations           |
| OMP_NUM_THREADS baseline gap                   | Added        | Guardrail: baseline at both 1 and 2 threads                |
| Scalene forkserver risk                        | Strengthened | Changed to "Do not use for multiprocessing profiling"      |
| OPT-5 shared_memory risk                       | Strengthened | Risk raised to High, PoC required before committing        |
| Phase 4 dependency                             | Relaxed      | Steps 4.1-4.3 depend only on Phase 1                       |

---

## Appendix: File Reference

| File                                                      | Purpose                                             |
|-----------------------------------------------------------|-----------------------------------------------------|
| `src/cascade_correlation/cascade_correlation.py`          | Core CasCor network, worker pool, parallel training |
| `src/candidate_unit/candidate_unit.py`                    | Candidate training, correlation computation         |
| `src/parallelism/task_distributor.py`                     | Local/remote task scheduling                        |
| `src/api/workers/coordinator.py`                          | Remote worker coordination                          |
| `src/profiling/deterministic.py`                          | cProfile decorator and context manager              |
| `src/tests/helpers/utilities.py`                          | Test timing and memory utilities                    |
| `src/tests/scripts/run_benchmarks.bash`                   | Existing benchmark harness                          |
| `src/tests/conftest.py`                                   | Test fixtures, `force_sequential_training`          |
| `src/cascor_constants/constants_model/constants_model.py` | Performance-relevant constants                      |
| `util/profile_training.bash`                              | py-spy profiling wrapper                            |
