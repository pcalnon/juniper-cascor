#!/usr/bin/env python
"""
Unit tests for the profiling module.

P3-NEW-001: Development Profiling Infrastructure tests.

Tests cover:
- ProfileContext
- MemoryTracker
- SampledLogger
- BatchLogger
- Decorators
"""

import logging
import time
from unittest.mock import MagicMock

import pytest


class TestProfileContext:
    """Tests for ProfileContext class."""

    @pytest.mark.unit
    def test_profile_context_creation(self):
        """Test ProfileContext can be created."""
        from profiling.deterministic import ProfileContext

        profiler = ProfileContext("test")
        assert profiler.name == "test"

    @pytest.mark.unit
    def test_profile_context_as_context_manager(self):
        """Test ProfileContext as context manager."""
        from profiling.deterministic import ProfileContext

        with ProfileContext("test") as p:
            result = sum(range(100))

        assert p.stats is not None

    @pytest.mark.unit
    def test_profile_context_start_stop(self):
        """Test ProfileContext start/stop methods."""
        from profiling.deterministic import ProfileContext

        profiler = ProfileContext("test")
        profiler.start()
        result = sum(range(100))
        profiler.stop()

        assert profiler.stats is not None

    @pytest.mark.unit
    def test_profile_context_get_stats_string(self):
        """Test get_stats_string method."""
        from profiling.deterministic import ProfileContext

        with ProfileContext("test") as p:
            result = sum(range(100))

        stats_str = p.get_stats_string()
        assert isinstance(stats_str, str)


class TestMemoryTracker:
    """Tests for MemoryTracker class."""

    @pytest.mark.unit
    def test_memory_tracker_creation(self):
        """Test MemoryTracker can be created."""
        from profiling.memory import MemoryTracker

        tracker = MemoryTracker("test")
        assert tracker.name == "test"

    @pytest.mark.unit
    def test_memory_tracker_as_context_manager(self):
        """Test MemoryTracker as context manager."""
        from profiling.memory import MemoryTracker

        with MemoryTracker("test") as m:
            data = list(range(1000))

        assert m.current_bytes >= 0

    @pytest.mark.unit
    def test_memory_tracker_get_stats(self):
        """Test get_stats method."""
        from profiling.memory import MemoryTracker

        with MemoryTracker("test") as m:
            data = list(range(1000))

        stats = m.get_stats()
        assert stats.current_bytes >= 0


class TestSampledLogger:
    """Tests for SampledLogger class."""

    @pytest.mark.unit
    def test_sampled_logger_creation(self):
        """Test SampledLogger can be created."""
        from profiling.logging_utils import SampledLogger

        logger = MagicMock()
        sampled = SampledLogger(logger, sample_rate=10)

        assert sampled.sample_rate == 10

    @pytest.mark.unit
    def test_sampled_logger_samples_correctly(self):
        """Test that SampledLogger samples messages."""
        from profiling.logging_utils import SampledLogger

        logger = MagicMock()
        sampled = SampledLogger(logger, sample_rate=5, include_first=False)

        for i in range(10):
            sampled.debug(f"Message {i}", key="test")

        assert logger.log.call_count == 2

    @pytest.mark.unit
    def test_sampled_logger_includes_first(self):
        """Test that SampledLogger includes first message."""
        from profiling.logging_utils import SampledLogger

        logger = MagicMock()
        sampled = SampledLogger(logger, sample_rate=100, include_first=True)

        sampled.debug("First message", key="test_first")

        assert logger.log.call_count == 1

    @pytest.mark.unit
    def test_sampled_logger_reset(self):
        """Test SampledLogger reset."""
        from profiling.logging_utils import SampledLogger

        logger = MagicMock()
        sampled = SampledLogger(logger, sample_rate=10)

        sampled.debug("Test", key="reset_test")
        sampled.reset()

        assert len(sampled._counts) == 0


class TestBatchLogger:
    """Tests for BatchLogger class."""

    @pytest.mark.unit
    def test_batch_logger_creation(self):
        """Test BatchLogger can be created."""
        from profiling.logging_utils import BatchLogger

        logger = MagicMock()
        batch = BatchLogger(logger, "test")

        assert batch.prefix == "test"

    @pytest.mark.unit
    def test_batch_logger_buffers_messages(self):
        """Test BatchLogger buffers messages."""
        from profiling.logging_utils import BatchLogger

        logger = MagicMock()
        batch = BatchLogger(logger, "test")

        batch.add("Message 1")
        batch.add("Message 2")

        assert len(batch._buffer) == 2

    @pytest.mark.unit
    def test_batch_logger_flush(self):
        """Test BatchLogger flush."""
        from profiling.logging_utils import BatchLogger

        logger = MagicMock()

        with BatchLogger(logger, "test") as batch:
            batch.add("Message 1")
            batch.add("Message 2")

        assert logger.log.called

    @pytest.mark.unit
    def test_batch_logger_auto_flush_on_max_buffer(self):
        """Test BatchLogger auto-flushes on max buffer."""
        from profiling.logging_utils import BatchLogger

        logger = MagicMock()
        batch = BatchLogger(logger, "test", max_buffer=3)
        batch._start_time = time.time()

        batch.add("1")
        batch.add("2")
        batch.add("3")

        assert len(batch._buffer) == 0


class TestLogFrequencyTracker:
    """Tests for LogFrequencyTracker class."""

    @pytest.mark.unit
    def test_tracker_creation(self):
        """Test LogFrequencyTracker can be created."""
        from profiling.logging_utils import LogFrequencyTracker

        tracker = LogFrequencyTracker()
        assert tracker is not None

    @pytest.mark.unit
    def test_tracker_records_calls(self):
        """Test tracker records calls."""
        from profiling.logging_utils import LogFrequencyTracker

        tracker = LogFrequencyTracker()
        tracker.record("test_key")
        tracker.record("test_key")
        tracker.record("test_key")

        stats = tracker.get_stats()
        assert stats["test_key"]["count"] == 3

    @pytest.mark.unit
    def test_tracker_reset(self):
        """Test tracker reset."""
        from profiling.logging_utils import LogFrequencyTracker

        tracker = LogFrequencyTracker()
        tracker.record("test")
        tracker.reset()

        assert len(tracker._counts) == 0


class TestProfilingDecorators:
    """Tests for profiling decorators."""

    @pytest.mark.unit
    def test_profile_function_decorator(self):
        """Test profile_function decorator."""
        from profiling.deterministic import profile_function

        @profile_function
        def test_func():
            return sum(range(100))

        result = test_func()
        assert result == 4950

    @pytest.mark.unit
    def test_memory_profile_decorator(self):
        """Test memory_profile decorator."""
        from profiling.memory import memory_profile

        @memory_profile
        def test_func():
            return list(range(100))

        result = test_func()
        assert len(result) == 100


class TestMemoryStats:
    """Tests for MemoryStats dataclass."""

    @pytest.mark.unit
    def test_memory_stats_creation(self):
        """Test MemoryStats creation."""
        from profiling.memory import MemoryStats

        stats = MemoryStats(current_bytes=1024 * 1024, peak_bytes=2 * 1024 * 1024, traced_blocks=100)

        assert stats.current_mb == 1.0
        assert stats.peak_mb == 2.0

    @pytest.mark.unit
    def test_memory_stats_str(self):
        """Test MemoryStats string representation."""
        from profiling.memory import MemoryStats

        stats = MemoryStats(current_bytes=1024 * 1024, peak_bytes=2 * 1024 * 1024, traced_blocks=100)

        str_repr = str(stats)
        assert "1.00" in str_repr
        assert "2.00" in str_repr


class TestLogIfEnabled:
    """Tests for log_if_enabled function."""

    @pytest.mark.unit
    def test_log_if_enabled_logs_when_enabled(self):
        """Test log_if_enabled logs when level is enabled."""
        from profiling.logging_utils import log_if_enabled

        logger = MagicMock()
        logger.isEnabledFor.return_value = True

        log_if_enabled(logger, logging.DEBUG, lambda: "Test message")

        assert logger.log.called

    @pytest.mark.unit
    def test_log_if_enabled_skips_when_disabled(self):
        """Test log_if_enabled skips when level is disabled."""
        from profiling.logging_utils import log_if_enabled

        logger = MagicMock()
        logger.isEnabledFor.return_value = False

        expensive_called = []

        def expensive_msg():
            expensive_called.append(True)
            return "Expensive"

        log_if_enabled(logger, logging.DEBUG, expensive_msg)

        assert len(expensive_called) == 0


class TestLogTiming:
    """Tests for log_timing context manager."""

    @pytest.mark.unit
    def test_log_timing_logs_duration(self):
        """Test log_timing logs operation duration."""
        from profiling.logging_utils import log_timing

        logger = MagicMock()

        with log_timing(logger, "test_operation"):
            time.sleep(0.01)

        assert logger.log.called
        call_args = str(logger.log.call_args)
        assert "test_operation" in call_args
