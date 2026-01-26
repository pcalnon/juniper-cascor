#!/usr/bin/env python
"""
Deterministic Profiling Module (cProfile wrappers)

P3-NEW-001: Development Profiling Infrastructure

Provides cProfile-based deterministic profiling for:
- Function-level profiling via decorator
- Context manager for block profiling
- Profile statistics saving and printing

Usage:
    # Decorator usage
    @profile_function
    def my_expensive_function():
        pass

    # Context manager usage
    with ProfileContext("training_phase") as p:
        network.fit(x, y)
    p.print_stats(top_n=20)

    # Manual profiling
    profiler = ProfileContext("custom")
    profiler.start()
    # ... code to profile ...
    profiler.stop()
    profiler.save("profile_output.prof")
"""

import cProfile
import pstats
import io
import functools
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Any


def profile_function(func: Callable = None, *, output_dir: str = None, top_n: int = 30):
    """
    Decorator to profile a function using cProfile.
    
    Args:
        func: The function to profile (used when decorator has no arguments)
        output_dir: Directory to save profile output (default: ./profiles/)
        top_n: Number of top functions to display in stats
        
    Returns:
        Decorated function that profiles execution.
        
    Example:
        @profile_function
        def train_network():
            pass
            
        @profile_function(output_dir="./my_profiles", top_n=50)
        def expensive_computation():
            pass
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            profiler = cProfile.Profile()
            profiler.enable()
            try:
                result = fn(*args, **kwargs)
            finally:
                profiler.disable()
                
                stats = pstats.Stats(profiler)
                stats.sort_stats(pstats.SortKey.CUMULATIVE)
                
                print(f"\n{'='*60}")
                print(f"Profile: {fn.__name__}")
                print(f"{'='*60}")
                stats.print_stats(top_n)
                
                if output_dir:
                    save_dir = Path(output_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = save_dir / f"{fn.__name__}_{timestamp}.prof"
                    profiler.dump_stats(str(filename))
                    print(f"Profile saved to: {filename}")
                    
            return result
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator


class ProfileContext:
    """
    Context manager for profiling code blocks.
    
    Example:
        with ProfileContext("candidate_training") as p:
            results = train_candidates(pool)
        p.print_stats(top_n=20)
        p.save("candidate_training.prof")
    """
    
    def __init__(self, name: str = "profile", output_dir: str = None):
        """
        Initialize profile context.
        
        Args:
            name: Name for this profiling session
            output_dir: Directory for saving profile output
        """
        self.name = name
        self.output_dir = Path(output_dir) if output_dir else Path("./profiles")
        self.profiler = cProfile.Profile()
        self.stats = None
        self._start_time = None
        self._end_time = None
        
    def __enter__(self) -> "ProfileContext":
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
        
    def start(self) -> None:
        """Start profiling."""
        self._start_time = datetime.now()
        self.profiler.enable()
        
    def stop(self) -> None:
        """Stop profiling and compute stats."""
        self.profiler.disable()
        self._end_time = datetime.now()
        self.stats = pstats.Stats(self.profiler)
        self.stats.sort_stats(pstats.SortKey.CUMULATIVE)
        
    def print_stats(self, top_n: int = 30, sort_by: str = "cumulative") -> None:
        """
        Print profile statistics.
        
        Args:
            top_n: Number of top functions to display
            sort_by: Sort key (cumulative, time, calls)
        """
        if self.stats is None:
            print("No profile data. Call stop() first.")
            return
            
        duration = (self._end_time - self._start_time).total_seconds()
        
        print(f"\n{'='*60}")
        print(f"Profile: {self.name}")
        print(f"Duration: {duration:.3f}s")
        print(f"{'='*60}")
        
        sort_key = {
            "cumulative": pstats.SortKey.CUMULATIVE,
            "time": pstats.SortKey.TIME,
            "calls": pstats.SortKey.CALLS,
        }.get(sort_by, pstats.SortKey.CUMULATIVE)
        
        self.stats.sort_stats(sort_key)
        self.stats.print_stats(top_n)
        
    def get_stats_string(self, top_n: int = 30) -> str:
        """Get profile statistics as a string."""
        if self.stats is None:
            return "No profile data."
            
        stream = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stream)
        stats.sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats(top_n)
        return stream.getvalue()
        
    def save(self, filename: str = None) -> Path:
        """
        Save profile data to file.
        
        Args:
            filename: Output filename (auto-generated if not provided)
            
        Returns:
            Path to saved file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}.prof"
            
        filepath = self.output_dir / filename
        self.profiler.dump_stats(str(filepath))
        print(f"Profile saved to: {filepath}")
        return filepath


def save_profile_stats(profiler: cProfile.Profile, filepath: str) -> None:
    """
    Save cProfile stats to a file.
    
    Args:
        profiler: cProfile.Profile instance
        filepath: Output file path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(filepath)
    print(f"Profile saved to: {filepath}")


def print_profile_stats(
    profiler: cProfile.Profile,
    top_n: int = 30,
    sort_by: str = "cumulative"
) -> None:
    """
    Print profile statistics from a cProfile.Profile instance.
    
    Args:
        profiler: cProfile.Profile instance
        top_n: Number of top functions to display
        sort_by: Sort key (cumulative, time, calls)
    """
    stats = pstats.Stats(profiler)
    
    sort_key = {
        "cumulative": pstats.SortKey.CUMULATIVE,
        "time": pstats.SortKey.TIME,
        "calls": pstats.SortKey.CALLS,
    }.get(sort_by, pstats.SortKey.CUMULATIVE)
    
    stats.sort_stats(sort_key)
    stats.print_stats(top_n)
