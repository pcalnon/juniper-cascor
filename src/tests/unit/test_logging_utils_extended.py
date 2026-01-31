#!/usr/bin/env python
"""
Extended unit tests for profiling/logging_utils.py to improve test coverage.

Tests cover:
- SampledLogger.trace() method (line 90)
- SampledLogger.verbose() method (line 94)
- BatchLogger.add() with flush triggered by flush_interval (line 148)
- BatchLogger.flush() with empty buffer (line 153)
- LogFrequencyTracker.print_stats() output format (lines 232-241)
"""

import io
import logging
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profiling.logging_utils import BatchLogger, LogFrequencyTracker, SampledLogger

pytestmark = pytest.mark.unit


class TestSampledLoggerTrace:
    """Tests for SampledLogger.trace() method (line 90)."""

    def test_trace_logs_at_level_5(self):
        """Test that trace() logs at level 5 with sampling."""
        mock_logger = MagicMock()
        sampled = SampledLogger(mock_logger, sample_rate=1, include_first=True)

        sampled.trace("trace message")

        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert call_args[0][0] == 5
        assert "[sampled]" in call_args[0][1]
        assert "trace message" in call_args[0][1]

    def test_trace_respects_sample_rate(self):
        """Test that trace() respects the sample rate."""
        mock_logger = MagicMock()
        sampled = SampledLogger(mock_logger, sample_rate=3, include_first=True)

        for i in range(7):
            sampled.trace(f"message {i}", key="trace_key")

        assert mock_logger.log.call_count == 3

    def test_trace_with_custom_key(self):
        """Test trace() with a custom sampling key."""
        mock_logger = MagicMock()
        sampled = SampledLogger(mock_logger, sample_rate=2, include_first=True)

        sampled.trace("msg1", key="key_a")
        sampled.trace("msg2", key="key_b")
        sampled.trace("msg3", key="key_a")
        sampled.trace("msg4", key="key_a")

        assert mock_logger.log.call_count == 3


class TestSampledLoggerVerbose:
    """Tests for SampledLogger.verbose() method (line 94)."""

    def test_verbose_logs_at_level_15(self):
        """Test that verbose() logs at level 15 with sampling."""
        mock_logger = MagicMock()
        sampled = SampledLogger(mock_logger, sample_rate=1, include_first=True)

        sampled.verbose("verbose message")

        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert call_args[0][0] == 15
        assert "[sampled]" in call_args[0][1]
        assert "verbose message" in call_args[0][1]

    def test_verbose_respects_sample_rate(self):
        """Test that verbose() respects the sample rate."""
        mock_logger = MagicMock()
        sampled = SampledLogger(mock_logger, sample_rate=5, include_first=True)

        for i in range(12):
            sampled.verbose(f"msg {i}", key="verbose_key")

        assert mock_logger.log.call_count == 3

    def test_verbose_without_include_first(self):
        """Test verbose() without logging the first message."""
        mock_logger = MagicMock()
        sampled = SampledLogger(mock_logger, sample_rate=3, include_first=False)

        for i in range(6):
            sampled.verbose(f"msg {i}", key="v_key")

        assert mock_logger.log.call_count == 2


class TestBatchLoggerFlushInterval:
    """Tests for BatchLogger.add() with flush triggered by flush_interval (line 148)."""

    def test_add_triggers_flush_at_interval(self):
        """Test that add() triggers flush when flush_interval is reached."""
        mock_logger = MagicMock()
        batch = BatchLogger(mock_logger, prefix="test", flush_interval=3)
        batch._start_time = 0

        batch.add("msg1")
        batch.add("msg2")
        assert mock_logger.log.call_count == 0

        batch.add("msg3")
        assert mock_logger.log.call_count > 0
        assert len(batch._buffer) == 0

    def test_add_multiple_interval_flushes(self):
        """Test multiple flushes triggered by flush_interval."""
        mock_logger = MagicMock()
        batch = BatchLogger(mock_logger, prefix="test", flush_interval=2)
        batch._start_time = 0

        for i in range(6):
            batch.add(f"msg {i}")

        assert mock_logger.log.call_count == 9

    def test_add_flush_interval_zero_no_auto_flush(self):
        """Test that flush_interval=0 disables auto-flush on interval."""
        mock_logger = MagicMock()
        batch = BatchLogger(mock_logger, prefix="test", flush_interval=0, max_buffer=100)
        batch._start_time = 0

        for i in range(10):
            batch.add(f"msg {i}")

        assert mock_logger.log.call_count == 0
        assert len(batch._buffer) == 10


class TestBatchLoggerEmptyFlush:
    """Tests for BatchLogger.flush() with empty buffer (line 153)."""

    def test_flush_empty_buffer_returns_early(self):
        """Test that flush() with empty buffer returns early without logging."""
        mock_logger = MagicMock()
        batch = BatchLogger(mock_logger, prefix="test")
        batch._start_time = 0

        batch.flush()

        mock_logger.log.assert_not_called()

    def test_flush_after_previous_flush(self):
        """Test that flush() after a previous flush does nothing."""
        mock_logger = MagicMock()
        batch = BatchLogger(mock_logger, prefix="test")
        batch._start_time = 0

        batch.add("message")
        batch.flush()
        call_count_after_first = mock_logger.log.call_count

        batch.flush()
        assert mock_logger.log.call_count == call_count_after_first

    def test_flush_empty_buffer_no_start_time(self):
        """Test flush() with empty buffer and no start time."""
        mock_logger = MagicMock()
        batch = BatchLogger(mock_logger, prefix="test")

        batch.flush()

        mock_logger.log.assert_not_called()


class TestLogFrequencyTrackerPrintStats:
    """Tests for LogFrequencyTracker.print_stats() output (lines 232-241)."""

    def test_print_stats_output_format(self, capsys):
        """Test that print_stats outputs correct format."""
        tracker = LogFrequencyTracker()
        tracker.record("key1", 0.1)
        tracker.record("key1", 0.2)
        tracker.record("key2", 0.05)

        tracker.print_stats()

        captured = capsys.readouterr()
        assert "=" * 60 in captured.out
        assert "Log Frequency Statistics" in captured.out
        assert "key1: 2 calls" in captured.out
        assert "0.3000s total" in captured.out
        assert "key2: 1 calls" in captured.out
        assert "0.0500s total" in captured.out

    def test_print_stats_sorted_by_count(self, capsys):
        """Test that print_stats sorts by count descending."""
        tracker = LogFrequencyTracker()
        tracker.record("low", 0.1)
        for _ in range(5):
            tracker.record("high", 0.01)
        for _ in range(3):
            tracker.record("medium", 0.02)

        tracker.print_stats()

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")

        high_idx = next(i for i, line in enumerate(lines) if "high:" in line)
        medium_idx = next(i for i, line in enumerate(lines) if "medium:" in line)
        low_idx = next(i for i, line in enumerate(lines) if "low:" in line)

        assert high_idx < medium_idx < low_idx

    def test_print_stats_respects_top_n(self, capsys):
        """Test that print_stats respects top_n parameter."""
        tracker = LogFrequencyTracker()
        for i in range(10):
            for _ in range(10 - i):
                tracker.record(f"key{i}", 0.01)

        tracker.print_stats(top_n=3)

        captured = capsys.readouterr()
        assert "key0:" in captured.out
        assert "key1:" in captured.out
        assert "key2:" in captured.out
        assert "key9:" not in captured.out

    def test_print_stats_empty_tracker(self, capsys):
        """Test print_stats with no recorded data."""
        tracker = LogFrequencyTracker()

        tracker.print_stats()

        captured = capsys.readouterr()
        assert "Log Frequency Statistics" in captured.out
        assert "=" * 60 in captured.out

    def test_print_stats_single_entry(self, capsys):
        """Test print_stats with single entry."""
        tracker = LogFrequencyTracker()
        tracker.record("only_key", 0.5)

        tracker.print_stats(top_n=20)

        captured = capsys.readouterr()
        assert "only_key: 1 calls, 0.5000s total" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
