#!/usr/bin/env python
"""
Memory Profiling Module (tracemalloc wrappers)

P3-NEW-001: Development Profiling Infrastructure

Provides memory profiling utilities:
- Function decorator for memory tracking
- Context manager for block memory profiling
- Memory snapshot comparison
- Peak memory tracking

Usage:
    # Decorator usage
    @memory_profile
    def load_large_dataset():
        pass

    # Context manager usage
    with MemoryTracker("training") as tracker:
        network.fit(x, y)
    tracker.print_summary()

    # Snapshot comparison
    snapshot1 = get_memory_snapshot()
    # ... code that allocates memory ...
    snapshot2 = get_memory_snapshot()
    compare_memory_snapshots(snapshot1, snapshot2)
"""

import functools
import linecache
import tracemalloc
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple


@dataclass
class MemoryStats:
    """Container for memory statistics."""

    current_bytes: int
    peak_bytes: int
    traced_blocks: int

    @property
    def current_mb(self) -> float:
        return self.current_bytes / (1024 * 1024)

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024 * 1024)

    def __str__(self) -> str:
        return f"Memory: current={self.current_mb:.2f}MB, " f"peak={self.peak_mb:.2f}MB, " f"blocks={self.traced_blocks}"


def memory_profile(func: Callable = None, *, top_n: int = 10):
    """
    Decorator to profile memory usage of a function.

    Args:
        func: The function to profile
        top_n: Number of top allocations to display

    Returns:
        Decorated function that profiles memory.

    Example:
        @memory_profile
        def process_data():
            pass

        @memory_profile(top_n=20)
        def load_model():
            pass
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            tracemalloc.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                snapshot = tracemalloc.take_snapshot()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                print(f"\n{'='*60}")
                print(f"Memory Profile: {fn.__name__}")
                print(f"{'='*60}")
                print(f"Current: {current / 1024 / 1024:.2f} MB")
                print(f"Peak: {peak / 1024 / 1024:.2f} MB")
                print(f"\nTop {top_n} allocations:")

                top_stats = snapshot.statistics("lineno")[:top_n]
                for stat in top_stats:
                    print(f"  {stat}")

            return result

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


class MemoryTracker:
    """
    Context manager for tracking memory usage in code blocks.

    Example:
        with MemoryTracker("model_loading") as tracker:
            model = load_model()
        tracker.print_summary()
        tracker.print_top_allocations(20)
    """

    def __init__(self, name: str = "memory_profile", frames: int = 25):
        """
        Initialize memory tracker.

        Args:
            name: Name for this tracking session
            frames: Number of frames to trace (higher = more detail)
        """
        self.name = name
        self.frames = frames
        self.snapshot_start = None
        self.snapshot_end = None
        self.current_bytes = 0
        self.peak_bytes = 0
        self._start_time = None
        self._end_time = None

    def __enter__(self) -> "MemoryTracker":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def start(self) -> None:
        """Start memory tracking."""
        self._start_time = datetime.now()
        tracemalloc.start(self.frames)
        self.snapshot_start = tracemalloc.take_snapshot()

    def stop(self) -> None:
        """Stop memory tracking and capture final state."""
        self.snapshot_end = tracemalloc.take_snapshot()
        self.current_bytes, self.peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._end_time = datetime.now()

    def get_stats(self) -> MemoryStats:
        """Get memory statistics."""
        traced = tracemalloc.get_tracemalloc_memory() if tracemalloc.is_tracing() else 0
        return MemoryStats(current_bytes=self.current_bytes, peak_bytes=self.peak_bytes, traced_blocks=traced)

    def print_summary(self) -> None:
        """Print memory usage summary."""
        duration = (self._end_time - self._start_time).total_seconds()

        print(f"\n{'='*60}")
        print(f"Memory Profile: {self.name}")
        print(f"{'='*60}")
        print(f"Duration: {duration:.3f}s")
        print(f"Current: {self.current_bytes / 1024 / 1024:.2f} MB")
        print(f"Peak: {self.peak_bytes / 1024 / 1024:.2f} MB")

    def print_top_allocations(self, top_n: int = 10, key_type: str = "lineno") -> None:
        """
        Print top memory allocations.

        Args:
            top_n: Number of top allocations to show
            key_type: Group by "lineno", "filename", or "traceback"
        """
        if self.snapshot_end is None:
            print("No snapshot available. Call stop() first.")
            return

        print(f"\nTop {top_n} allocations (by {key_type}):")
        top_stats = self.snapshot_end.statistics(key_type)[:top_n]

        for i, stat in enumerate(top_stats, 1):
            print(f"  {i}. {stat}")

    def get_diff(self) -> List:
        """
        Get memory difference between start and end.

        Returns:
            List of StatisticDiff objects
        """
        if self.snapshot_start is None or self.snapshot_end is None:
            return []
        return self.snapshot_end.compare_to(self.snapshot_start, "lineno")

    def print_diff(self, top_n: int = 10) -> None:
        """Print memory difference between start and end."""
        diff = self.get_diff()

        print(f"\nMemory changes (top {top_n}):")
        for stat in diff[:top_n]:
            print(f"  {stat}")


def get_memory_snapshot() -> Any:
    """
    Take a memory snapshot.

    Note: tracemalloc must be started before calling this.

    Returns:
        tracemalloc.Snapshot object

    Example:
        tracemalloc.start()
        snapshot1 = get_memory_snapshot()
        # ... allocate memory ...
        snapshot2 = get_memory_snapshot()
        compare_memory_snapshots(snapshot1, snapshot2)
        tracemalloc.stop()
    """
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    return tracemalloc.take_snapshot()


def compare_memory_snapshots(snapshot1, snapshot2, top_n: int = 10, key_type: str = "lineno") -> List:
    """
    Compare two memory snapshots and print the difference.

    Args:
        snapshot1: First snapshot (baseline)
        snapshot2: Second snapshot (comparison)
        top_n: Number of top differences to show
        key_type: Group by "lineno", "filename", or "traceback"

    Returns:
        List of StatisticDiff objects
    """
    diff = snapshot2.compare_to(snapshot1, key_type)

    print(f"\n{'='*60}")
    print(f"Memory Comparison (top {top_n} changes)")
    print(f"{'='*60}")

    for stat in diff[:top_n]:
        print(f"  {stat}")

    total_diff = sum(stat.size_diff for stat in diff)
    print(f"\nTotal change: {total_diff / 1024 / 1024:+.2f} MB")

    return diff


def display_top_allocations(snapshot, top_n: int = 10, key_type: str = "traceback") -> None:
    """
    Display top memory allocations from a snapshot with source code context.

    Args:
        snapshot: tracemalloc.Snapshot object
        top_n: Number of top allocations to show
        key_type: How to group allocations
    """
    top_stats = snapshot.statistics(key_type)[:top_n]

    print(f"\nTop {top_n} memory allocations:")
    print("=" * 60)

    for index, stat in enumerate(top_stats, 1):
        print(f"\n#{index}: {stat.size / 1024:.1f} KiB")

        if key_type == "traceback":
            for line in stat.traceback.format():
                print(f"  {line}")
        else:
            print(f"  {stat.traceback}")
