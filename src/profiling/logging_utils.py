#!/usr/bin/env python
"""
Logging Utilities for Hot Path Optimization

P4-NEW-004: Reduce Debug Logging in Hot Paths

Provides utilities to reduce logging overhead in performance-critical code:
- Sampled logging (log every Nth call)
- Batch logging (aggregate logs and emit periodically)
- Conditional trace logging with level checks
- Log frequency tracking

Usage:
    # Sample logging every 100 calls
    sampled_logger = SampledLogger(logger, sample_rate=100)
    for epoch in range(1000):
        sampled_logger.debug(f"Epoch {epoch}")  # Only logs every 100th

    # Batch logging
    with BatchLogger(logger, "training", flush_interval=10) as bl:
        for i in range(100):
            bl.add(f"Step {i}: loss={loss}")
    # Logs all 100 messages at once at the end

    # Check log level before expensive formatting
    if logger.isEnabledFor(TRACE_LEVEL):
        logger.trace(f"Expensive: {compute_debug_info()}")
"""

import logging
import time
from contextlib import contextmanager

# from functools import wraps  # TODO: F401 - unused import, may be needed for future use
# from typing import Any, Callable, List, Optional  # TODO: F401 - Any, Optional unused
from typing import Callable, List


class SampledLogger:
    """
    Logger wrapper that samples log messages at a configurable rate.

    Useful for hot paths where logging every iteration is too expensive.

    Example:
        sampled = SampledLogger(logger, sample_rate=100)
        for i in range(10000):
            sampled.debug(f"Iteration {i}")  # Only logs every 100th
    """

    def __init__(self, logger, sample_rate: int = 100, include_first: bool = True):
        """
        Initialize sampled logger.

        Args:
            logger: Underlying logger instance
            sample_rate: Log every Nth message
            include_first: Always log the first message
        """
        self.logger = logger
        self.sample_rate = sample_rate
        self.include_first = include_first
        self._counts = {}

    def _should_log(self, key: str) -> bool:
        """Determine if this message should be logged."""
        if key not in self._counts:
            self._counts[key] = 0
            if self.include_first:
                return True

        self._counts[key] += 1
        if self._counts[key] >= self.sample_rate:
            self._counts[key] = 0
            return True
        return False

    def _log_with_sample(self, level: int, msg: str, key: str = None):
        """Log if sampling criteria met."""
        sample_key = key or msg[:50]  # Use first 50 chars as key if not provided
        if self._should_log(sample_key):
            self.logger.log(level, f"[sampled] {msg}")

    def debug(self, msg: str, key: str = None):
        """Debug log with sampling."""
        self._log_with_sample(logging.DEBUG, msg, key)

    def trace(self, msg: str, key: str = None):
        """Trace log with sampling (level 5)."""
        self._log_with_sample(5, msg, key)

    def verbose(self, msg: str, key: str = None):
        """Verbose log with sampling (level 15)."""
        self._log_with_sample(15, msg, key)

    def reset(self):
        """Reset all sample counters."""
        self._counts.clear()


class BatchLogger:
    """
    Logger that batches messages and emits them periodically or on flush.

    Reduces I/O overhead by collecting multiple log messages and writing
    them together.

    Example:
        with BatchLogger(logger, "epoch_stats") as bl:
            for epoch in range(100):
                bl.add(f"Epoch {epoch}: loss={loss}")
        # All 100 messages logged at once
    """

    def __init__(self, logger, prefix: str = "", flush_interval: int = 0, max_buffer: int = 1000, level: int = logging.DEBUG):
        """
        Initialize batch logger.

        Args:
            logger: Underlying logger instance
            prefix: Prefix for batch output
            flush_interval: Auto-flush every N messages (0 = no auto-flush)
            max_buffer: Maximum messages to buffer before forced flush
            level: Log level for output
        """
        self.logger = logger
        self.prefix = prefix
        self.flush_interval = flush_interval
        self.max_buffer = max_buffer
        self.level = level
        self._buffer: List[str] = []
        self._start_time = None

    def __enter__(self) -> "BatchLogger":
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()

    def add(self, msg: str):
        """Add a message to the batch."""
        self._buffer.append(msg)

        if len(self._buffer) >= self.max_buffer:
            self.flush()
        elif self.flush_interval > 0 and len(self._buffer) >= self.flush_interval:
            self.flush()

    def flush(self):
        """Flush all buffered messages."""
        if not self._buffer:
            return

        elapsed = time.time() - self._start_time if self._start_time else 0

        header = f"[Batch: {self.prefix}] ({len(self._buffer)} messages, {elapsed:.2f}s)"
        self.logger.log(self.level, header)

        for msg in self._buffer:
            self.logger.log(self.level, f"  {msg}")

        self._buffer.clear()


def log_if_enabled(logger, level: int, msg_func: Callable[[], str]):
    """
    Only evaluate and log message if the level is enabled.

    Avoids expensive string formatting when log level is disabled.

    Example:
        log_if_enabled(logger, TRACE, lambda: f"Expensive: {compute_debug()}")
    """
    if logger.isEnabledFor(level):
        logger.log(level, msg_func())


@contextmanager
def log_timing(logger, operation: str, level: int = logging.DEBUG):
    """
    Context manager to log operation timing.

    Example:
        with log_timing(logger, "forward_pass"):
            result = network.forward(x)
        # Logs: "[Timing] forward_pass: 0.123s"
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.log(level, f"[Timing] {operation}: {elapsed:.4f}s")


class LogFrequencyTracker:
    """
    Track how often log statements are called for profiling.

    Helps identify which log statements are called most frequently
    so they can be optimized or sampled.

    Example:
        tracker = LogFrequencyTracker()
        for i in range(1000):
            tracker.record("epoch_log")
        tracker.print_stats()
    """

    def __init__(self):
        self._counts = {}
        self._total_time = {}

    def record(self, key: str, elapsed: float = 0.0):
        """Record a log call."""
        if key not in self._counts:
            self._counts[key] = 0
            self._total_time[key] = 0.0
        self._counts[key] += 1
        self._total_time[key] += elapsed

    def get_stats(self) -> dict:
        """Get frequency statistics."""
        stats = {}
        for key in self._counts:
            stats[key] = {"count": self._counts[key], "total_time": self._total_time[key], "avg_time": self._total_time[key] / self._counts[key] if self._counts[key] > 0 else 0}
        return stats

    def print_stats(self, top_n: int = 20):
        """Print frequency statistics."""
        print("\n" + "=" * 60)
        print("Log Frequency Statistics")
        print("=" * 60)

        sorted_keys = sorted(self._counts.keys(), key=lambda k: self._counts[k], reverse=True)

        for key in sorted_keys[:top_n]:
            count = self._counts[key]
            total_time = self._total_time[key]
            print(f"  {key}: {count} calls, {total_time:.4f}s total")

    def reset(self):
        """Reset all statistics."""
        self._counts.clear()
        self._total_time.clear()


# Pre-defined log levels (match custom levels from log_config)
TRACE = 5
VERBOSE = 15
