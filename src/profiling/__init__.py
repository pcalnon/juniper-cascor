#!/usr/bin/env python
"""
Profiling Infrastructure for Juniper Cascor

This module provides profiling utilities for performance analysis:
- Deterministic profiling via cProfile
- Memory profiling via tracemalloc
- Function decorators for targeted profiling
- CLI integration for profiling training runs

P3-NEW-001: Development Profiling Infrastructure
"""

from profiling.deterministic import (
    profile_function,
    ProfileContext,
    save_profile_stats,
    print_profile_stats,
)
from profiling.memory import (
    memory_profile,
    MemoryTracker,
    get_memory_snapshot,
    compare_memory_snapshots,
)

__all__ = [
    "profile_function",
    "ProfileContext",
    "save_profile_stats",
    "print_profile_stats",
    "memory_profile",
    "MemoryTracker",
    "get_memory_snapshot",
    "compare_memory_snapshots",
]
