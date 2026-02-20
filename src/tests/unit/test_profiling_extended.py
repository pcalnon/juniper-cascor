#!/usr/bin/env python
"""
Extended unit tests for profiling modules to improve code coverage.

Tests cover:
- ProfileContext: __init__, __enter__, __exit__, get_stats, print_stats, save methods
- MemoryTracker: __init__, __enter__, __exit__, print_summary, print_top_allocations, print_diff methods
- Edge cases and error handling
"""

import io
import sys
import tempfile
import tracemalloc
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestProfileContextExtended:
    """Extended tests for ProfileContext class."""

    @pytest.mark.unit
    def test_profile_context_init_with_defaults(self):
        """Test ProfileContext initialization with default values."""
        from profiling.deterministic import ProfileContext

        profiler = ProfileContext()
        assert profiler.name == "profile"
        assert profiler.output_dir == Path("./profiles")
        assert profiler.stats is None
        assert profiler._start_time is None
        assert profiler._end_time is None

    @pytest.mark.unit
    def test_profile_context_init_with_custom_values(self):
        """Test ProfileContext initialization with custom values."""
        from profiling.deterministic import ProfileContext

        profiler = ProfileContext(name="custom_profile", output_dir="/tmp/my_profiles")
        assert profiler.name == "custom_profile"
        assert profiler.output_dir == Path("/tmp/my_profiles")

    @pytest.mark.unit
    def test_profile_context_enter_starts_profiler(self):
        """Test that __enter__ calls start()."""
        from profiling.deterministic import ProfileContext

        profiler = ProfileContext("test")
        result = profiler.__enter__()

        assert result is profiler
        assert profiler._start_time is not None

        profiler.stop()

    @pytest.mark.unit
    def test_profile_context_exit_stops_profiler(self):
        """Test that __exit__ calls stop()."""
        from profiling.deterministic import ProfileContext

        profiler = ProfileContext("test")
        profiler.start()
        profiler.__exit__(None, None, None)

        assert profiler.stats is not None
        assert profiler._end_time is not None

    @pytest.mark.unit
    def test_profile_context_print_stats_no_data(self, capsys):
        """Test print_stats when no profiling data exists."""
        from profiling.deterministic import ProfileContext

        profiler = ProfileContext("test")
        profiler.print_stats()

        captured = capsys.readouterr()
        assert "No profile data" in captured.out

    @pytest.mark.unit
    def test_profile_context_print_stats_with_sort_by_time(self, capsys):
        """Test print_stats with sort_by='time'."""
        from profiling.deterministic import ProfileContext

        with ProfileContext("test") as p:
            result = sum(range(100))

        p.print_stats(top_n=5, sort_by="time")
        captured = capsys.readouterr()
        assert "Profile: test" in captured.out
        assert "Duration:" in captured.out

    @pytest.mark.unit
    def test_profile_context_print_stats_with_sort_by_calls(self, capsys):
        """Test print_stats with sort_by='calls'."""
        from profiling.deterministic import ProfileContext

        with ProfileContext("test") as p:
            result = sum(range(100))

        p.print_stats(top_n=5, sort_by="calls")
        captured = capsys.readouterr()
        assert "Profile: test" in captured.out

    @pytest.mark.unit
    def test_profile_context_print_stats_with_invalid_sort_key(self, capsys):
        """Test print_stats with invalid sort_by falls back to cumulative."""
        from profiling.deterministic import ProfileContext

        with ProfileContext("test") as p:
            result = sum(range(100))

        p.print_stats(top_n=5, sort_by="invalid_key")
        captured = capsys.readouterr()
        assert "Profile: test" in captured.out

    @pytest.mark.unit
    def test_profile_context_get_stats_string_no_data(self):
        """Test get_stats_string when no profiling data exists."""
        from profiling.deterministic import ProfileContext

        profiler = ProfileContext("test")
        result = profiler.get_stats_string()

        assert result == "No profile data."

    @pytest.mark.unit
    def test_profile_context_get_stats_string_with_data(self):
        """Test get_stats_string returns stats string."""
        from profiling.deterministic import ProfileContext

        with ProfileContext("test") as p:
            result = sum(range(100))

        stats_str = p.get_stats_string(top_n=10)
        assert isinstance(stats_str, str)
        assert len(stats_str) > 0

    @pytest.mark.unit
    def test_profile_context_save_auto_filename(self, tmp_path, capsys):
        """Test save with auto-generated filename."""
        from profiling.deterministic import ProfileContext

        profiler = ProfileContext("test_save", output_dir=str(tmp_path))
        with profiler:
            result = sum(range(100))

        saved_path = profiler.save()

        assert saved_path.exists()
        assert "test_save_" in saved_path.name
        assert saved_path.suffix == ".prof"

        captured = capsys.readouterr()
        assert "Profile saved to:" in captured.out

    @pytest.mark.unit
    def test_profile_context_save_custom_filename(self, tmp_path, capsys):
        """Test save with custom filename."""
        from profiling.deterministic import ProfileContext

        profiler = ProfileContext("test_save", output_dir=str(tmp_path))
        with profiler:
            result = sum(range(100))

        saved_path = profiler.save(filename="custom_output.prof")

        assert saved_path.exists()
        assert saved_path.name == "custom_output.prof"

    @pytest.mark.unit
    def test_profile_context_save_creates_directory(self, tmp_path, capsys):
        """Test save creates output directory if it doesn't exist."""
        from profiling.deterministic import ProfileContext

        nested_dir = tmp_path / "nested" / "profiles"
        profiler = ProfileContext("test", output_dir=str(nested_dir))
        with profiler:
            result = sum(range(100))

        saved_path = profiler.save()

        assert nested_dir.exists()
        assert saved_path.exists()


class TestProfileFunctionDecorator:
    """Extended tests for profile_function decorator."""

    @pytest.mark.unit
    def test_profile_function_with_arguments(self, tmp_path, capsys):
        """Test profile_function decorator with arguments."""
        from profiling.deterministic import profile_function

        @profile_function(output_dir=str(tmp_path), top_n=5)
        def compute():
            return sum(range(1000))

        result = compute()

        assert result == 499500
        captured = capsys.readouterr()
        assert "Profile: compute" in captured.out
        assert "Profile saved to:" in captured.out

        saved_files = list(tmp_path.glob("*.prof"))
        assert len(saved_files) == 1

    @pytest.mark.unit
    def test_profile_function_preserves_function_metadata(self):
        """Test that profile_function preserves function metadata."""
        from profiling.deterministic import profile_function

        @profile_function
        def documented_function():
            """This is a documented function."""
            return 42

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a documented function."


class TestSaveProfileStats:
    """Tests for save_profile_stats function."""

    @pytest.mark.unit
    def test_save_profile_stats(self, tmp_path, capsys):
        """Test save_profile_stats function."""
        import cProfile

        from profiling.deterministic import save_profile_stats

        profiler = cProfile.Profile()
        profiler.enable()
        result = sum(range(100))
        profiler.disable()

        filepath = tmp_path / "test_stats.prof"
        save_profile_stats(profiler, str(filepath))

        assert filepath.exists()
        captured = capsys.readouterr()
        assert "Profile saved to:" in captured.out

    @pytest.mark.unit
    def test_save_profile_stats_creates_parent_dirs(self, tmp_path, capsys):
        """Test save_profile_stats creates parent directories."""
        import cProfile

        from profiling.deterministic import save_profile_stats

        profiler = cProfile.Profile()
        profiler.enable()
        result = sum(range(100))
        profiler.disable()

        filepath = tmp_path / "nested" / "dir" / "test_stats.prof"
        save_profile_stats(profiler, str(filepath))

        assert filepath.exists()


class TestPrintProfileStats:
    """Tests for print_profile_stats function."""

    @pytest.mark.unit
    def test_print_profile_stats_cumulative(self, capsys):
        """Test print_profile_stats with cumulative sort."""
        import cProfile

        from profiling.deterministic import print_profile_stats

        profiler = cProfile.Profile()
        profiler.enable()
        result = sum(range(100))
        profiler.disable()

        print_profile_stats(profiler, top_n=5, sort_by="cumulative")

        captured = capsys.readouterr()
        assert len(captured.out) > 0

    @pytest.mark.unit
    def test_print_profile_stats_time(self, capsys):
        """Test print_profile_stats with time sort."""
        import cProfile

        from profiling.deterministic import print_profile_stats

        profiler = cProfile.Profile()
        profiler.enable()
        result = sum(range(100))
        profiler.disable()

        print_profile_stats(profiler, top_n=5, sort_by="time")

        captured = capsys.readouterr()
        assert len(captured.out) > 0

    @pytest.mark.unit
    def test_print_profile_stats_calls(self, capsys):
        """Test print_profile_stats with calls sort."""
        import cProfile

        from profiling.deterministic import print_profile_stats

        profiler = cProfile.Profile()
        profiler.enable()
        result = sum(range(100))
        profiler.disable()

        print_profile_stats(profiler, top_n=5, sort_by="calls")

        captured = capsys.readouterr()
        assert len(captured.out) > 0

    @pytest.mark.unit
    def test_print_profile_stats_invalid_sort(self, capsys):
        """Test print_profile_stats with invalid sort falls back to cumulative."""
        import cProfile

        from profiling.deterministic import print_profile_stats

        profiler = cProfile.Profile()
        profiler.enable()
        result = sum(range(100))
        profiler.disable()

        print_profile_stats(profiler, top_n=5, sort_by="invalid")

        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestMemoryTrackerExtended:
    """Extended tests for MemoryTracker class."""

    @pytest.mark.unit
    def test_memory_tracker_init_with_defaults(self):
        """Test MemoryTracker initialization with defaults."""
        from profiling.memory import MemoryTracker

        tracker = MemoryTracker()
        assert tracker.name == "memory_profile"
        assert tracker.frames == 25
        assert tracker.snapshot_start is None
        assert tracker.snapshot_end is None
        assert tracker.current_bytes == 0
        assert tracker.peak_bytes == 0

    @pytest.mark.unit
    def test_memory_tracker_init_with_custom_values(self):
        """Test MemoryTracker initialization with custom values."""
        from profiling.memory import MemoryTracker

        tracker = MemoryTracker(name="custom_tracker", frames=50)
        assert tracker.name == "custom_tracker"
        assert tracker.frames == 50

    @pytest.mark.unit
    def test_memory_tracker_enter_returns_self(self):
        """Test that __enter__ returns self."""
        from profiling.memory import MemoryTracker

        tracker = MemoryTracker("test")
        result = tracker.__enter__()

        assert result is tracker
        tracker.stop()

    @pytest.mark.unit
    def test_memory_tracker_exit_stops_tracking(self):
        """Test that __exit__ stops tracking."""
        from profiling.memory import MemoryTracker

        tracker = MemoryTracker("test")
        tracker.start()
        tracker.__exit__(None, None, None)

        assert tracker.snapshot_end is not None
        assert tracker._end_time is not None

    @pytest.mark.unit
    def test_memory_tracker_print_summary(self, capsys):
        """Test print_summary outputs expected information."""
        from profiling.memory import MemoryTracker

        with MemoryTracker("test_summary") as m:
            data = list(range(10000))

        m.print_summary()
        captured = capsys.readouterr()

        assert "Memory Profile: test_summary" in captured.out
        assert "Duration:" in captured.out
        assert "Current:" in captured.out
        assert "Peak:" in captured.out
        assert "MB" in captured.out

    @pytest.mark.unit
    def test_memory_tracker_print_top_allocations_no_snapshot(self, capsys):
        """Test print_top_allocations with no snapshot available."""
        from profiling.memory import MemoryTracker

        tracker = MemoryTracker("test")
        tracker.print_top_allocations()

        captured = capsys.readouterr()
        assert "No snapshot available" in captured.out

    @pytest.mark.unit
    def test_memory_tracker_print_top_allocations_lineno(self, capsys):
        """Test print_top_allocations with lineno key_type."""
        from profiling.memory import MemoryTracker

        with MemoryTracker("test") as m:
            data = list(range(10000))

        m.print_top_allocations(top_n=5, key_type="lineno")
        captured = capsys.readouterr()

        assert "Top 5 allocations (by lineno):" in captured.out

    @pytest.mark.unit
    def test_memory_tracker_print_top_allocations_filename(self, capsys):
        """Test print_top_allocations with filename key_type."""
        from profiling.memory import MemoryTracker

        with MemoryTracker("test") as m:
            data = list(range(10000))

        m.print_top_allocations(top_n=5, key_type="filename")
        captured = capsys.readouterr()

        assert "Top 5 allocations (by filename):" in captured.out

    @pytest.mark.unit
    def test_memory_tracker_get_diff_no_snapshots(self):
        """Test get_diff returns empty list when no snapshots exist."""
        from profiling.memory import MemoryTracker

        tracker = MemoryTracker("test")
        result = tracker.get_diff()

        assert result == []

    @pytest.mark.unit
    def test_memory_tracker_get_diff_with_snapshots(self):
        """Test get_diff returns diff when snapshots exist."""
        from profiling.memory import MemoryTracker

        with MemoryTracker("test") as m:
            data = list(range(10000))

        diff = m.get_diff()
        assert isinstance(diff, list)

    @pytest.mark.unit
    def test_memory_tracker_print_diff(self, capsys):
        """Test print_diff outputs memory changes."""
        from profiling.memory import MemoryTracker

        with MemoryTracker("test") as m:
            data = list(range(10000))

        m.print_diff(top_n=5)
        captured = capsys.readouterr()

        assert "Memory changes (top 5):" in captured.out

    @pytest.mark.unit
    def test_memory_tracker_get_stats_while_tracing(self):
        """Test get_stats returns correct stats."""
        from profiling.memory import MemoryTracker

        with MemoryTracker("test") as m:
            data = list(range(10000))

        stats = m.get_stats()
        assert stats.current_bytes >= 0
        assert stats.peak_bytes >= 0


class TestMemoryProfileDecorator:
    """Extended tests for memory_profile decorator."""

    @pytest.mark.unit
    def test_memory_profile_with_top_n_argument(self, capsys):
        """Test memory_profile decorator with top_n argument."""
        from profiling.memory import memory_profile

        @memory_profile(top_n=3)
        def allocate_memory():
            return list(range(10000))

        result = allocate_memory()

        assert len(result) == 10000
        captured = capsys.readouterr()
        assert "Memory Profile: allocate_memory" in captured.out
        assert "Top 3 allocations:" in captured.out

    @pytest.mark.unit
    def test_memory_profile_preserves_function_metadata(self):
        """Test that memory_profile preserves function metadata."""
        from profiling.memory import memory_profile

        @memory_profile
        def documented_function():
            """This is a documented function."""
            return []

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a documented function."


class TestGetMemorySnapshot:
    """Tests for get_memory_snapshot function."""

    @pytest.mark.unit
    def test_get_memory_snapshot_starts_tracing_if_not_started(self):
        """Test get_memory_snapshot starts tracemalloc if not already running."""
        from profiling.memory import get_memory_snapshot

        if tracemalloc.is_tracing():
            tracemalloc.stop()

        snapshot = get_memory_snapshot()

        assert snapshot is not None
        assert tracemalloc.is_tracing()

        tracemalloc.stop()

    @pytest.mark.unit
    def test_get_memory_snapshot_when_already_tracing(self):
        """Test get_memory_snapshot when tracemalloc is already running."""
        from profiling.memory import get_memory_snapshot

        tracemalloc.start()

        snapshot = get_memory_snapshot()

        assert snapshot is not None
        tracemalloc.stop()


class TestCompareMemorySnapshots:
    """Tests for compare_memory_snapshots function."""

    @pytest.mark.unit
    def test_compare_memory_snapshots(self, capsys):
        """Test compare_memory_snapshots outputs comparison."""
        from profiling.memory import compare_memory_snapshots, get_memory_snapshot

        tracemalloc.start()
        snapshot1 = get_memory_snapshot()
        data = list(range(10000))
        snapshot2 = get_memory_snapshot()
        tracemalloc.stop()

        diff = compare_memory_snapshots(snapshot1, snapshot2, top_n=5)

        captured = capsys.readouterr()
        assert "Memory Comparison (top 5 changes)" in captured.out
        assert "Total change:" in captured.out
        assert isinstance(diff, list)

    @pytest.mark.unit
    def test_compare_memory_snapshots_with_filename_key(self, capsys):
        """Test compare_memory_snapshots with filename key_type."""
        from profiling.memory import compare_memory_snapshots, get_memory_snapshot

        tracemalloc.start()
        snapshot1 = get_memory_snapshot()
        data = list(range(10000))
        snapshot2 = get_memory_snapshot()
        tracemalloc.stop()

        diff = compare_memory_snapshots(snapshot1, snapshot2, top_n=5, key_type="filename")

        assert isinstance(diff, list)


class TestDisplayTopAllocations:
    """Tests for display_top_allocations function."""

    @pytest.mark.unit
    def test_display_top_allocations_lineno(self, capsys):
        """Test display_top_allocations with lineno key_type."""
        from profiling.memory import display_top_allocations

        tracemalloc.start()
        data = list(range(10000))
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        display_top_allocations(snapshot, top_n=5, key_type="lineno")

        captured = capsys.readouterr()
        assert "Top 5 memory allocations:" in captured.out

    @pytest.mark.unit
    def test_display_top_allocations_traceback(self, capsys):
        """Test display_top_allocations with traceback key_type."""
        from profiling.memory import display_top_allocations

        tracemalloc.start(1)
        data = list(range(10000))
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        display_top_allocations(snapshot, top_n=3, key_type="traceback")

        captured = capsys.readouterr()
        assert "Top 3 memory allocations:" in captured.out

    @pytest.mark.unit
    def test_display_top_allocations_filename(self, capsys):
        """Test display_top_allocations with filename key_type."""
        from profiling.memory import display_top_allocations

        tracemalloc.start()
        data = list(range(10000))
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        display_top_allocations(snapshot, top_n=3, key_type="filename")

        captured = capsys.readouterr()
        assert "Top 3 memory allocations:" in captured.out


class TestMemoryStatsExtended:
    """Extended tests for MemoryStats dataclass."""

    @pytest.mark.unit
    def test_memory_stats_properties(self):
        """Test MemoryStats MB conversion properties."""
        from profiling.memory import MemoryStats

        stats = MemoryStats(
            current_bytes=5 * 1024 * 1024,
            peak_bytes=10 * 1024 * 1024,
            traced_blocks=500,
        )

        assert stats.current_mb == 5.0
        assert stats.peak_mb == 10.0
        assert stats.traced_blocks == 500

    @pytest.mark.unit
    def test_memory_stats_str_format(self):
        """Test MemoryStats string representation format."""
        from profiling.memory import MemoryStats

        stats = MemoryStats(
            current_bytes=3 * 1024 * 1024,
            peak_bytes=7 * 1024 * 1024,
            traced_blocks=250,
        )

        str_repr = str(stats)
        assert "current=3.00MB" in str_repr
        assert "peak=7.00MB" in str_repr
        assert "blocks=250" in str_repr
