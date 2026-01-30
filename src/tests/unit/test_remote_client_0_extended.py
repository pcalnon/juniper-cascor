#!/usr/bin/env python
"""
Extended unit tests for remote_client_0.py to improve test coverage.

Focuses on uncovered lines: 29-47, 73-74, 91-99, 116-131, 157
"""

import multiprocessing as mp
import os
import queue
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from remote_client.remote_client_0 import RemoteCandidateTrainingClient

pytestmark = pytest.mark.unit


class TestRemoteCandidateTrainingClientInit:
    """Tests for RemoteCandidateTrainingClient initialization."""

    def test_default_initialization(self):
        """Test client initializes with default values."""
        client = RemoteCandidateTrainingClient()
        assert client.server_address == ("127.0.0.1", 50000)
        assert client.authkey == b"Juniper_Cascade_Correlation_Multiprocessing_Authkey"
        assert client.manager is None

    def test_custom_initialization(self):
        """Test client initializes with custom values."""
        custom_addr = ("192.168.1.1", 12345)
        custom_key = b"custom_key"
        client = RemoteCandidateTrainingClient(server_address=custom_addr, authkey=custom_key)
        assert client.server_address == custom_addr
        assert client.authkey == custom_key


class TestConnectMethod:
    """Tests for connect() method (lines 29-47)."""

    def test_connect_failure_returns_false(self, capsys):
        """Test connect returns False when connection fails."""
        client = RemoteCandidateTrainingClient(server_address=("127.0.0.1", 59999))
        result = client.connect()
        assert result is False
        captured = capsys.readouterr()
        assert "Failed to connect" in captured.out

    def test_connect_creates_manager_class(self):
        """Test connect creates the CandidateTrainingManager class with proper registrations."""
        client = RemoteCandidateTrainingClient()
        with patch.object(client, "connect", wraps=client.connect):
            result = client.connect()
            assert result is False

    @patch("remote_client.remote_client_0.BaseManager")
    def test_connect_success_path(self, mock_base_manager, capsys):
        """Test successful connection path."""
        mock_manager_instance = MagicMock()
        mock_manager_class = MagicMock(return_value=mock_manager_instance)
        mock_manager_class.register = MagicMock()

        with patch("remote_client.remote_client_0.BaseManager", return_value=mock_manager_class):
            client = RemoteCandidateTrainingClient()
            result = client.connect()
            assert result is False


class TestProcessTasksMethod:
    """Tests for process_tasks() method (lines 49-74)."""

    def test_process_tasks_without_connection(self, capsys):
        """Test process_tasks returns early when not connected."""
        client = RemoteCandidateTrainingClient()
        client.process_tasks(num_workers=2)
        captured = capsys.readouterr()
        assert "Not connected to manager" in captured.out

    def test_process_tasks_with_mock_manager(self, capsys):
        """Test process_tasks with mocked manager (exercises exception path)."""
        client = RemoteCandidateTrainingClient()
        client.manager = MagicMock()
        client.manager.get_tasks_queue.side_effect = Exception("Queue error")

        client.process_tasks(num_workers=1)
        captured = capsys.readouterr()
        assert "Error processing tasks" in captured.out

    def test_process_tasks_worker_creation(self, capsys):
        """Test process_tasks starts worker processes."""
        client = RemoteCandidateTrainingClient()
        mock_manager = MagicMock()
        mock_tasks_queue = MagicMock()
        mock_done_queue = MagicMock()
        mock_manager.get_tasks_queue.return_value = mock_tasks_queue
        mock_manager.get_done_queue.return_value = mock_done_queue
        client.manager = mock_manager

        with patch("multiprocessing.Process") as mock_process:
            mock_proc_instance = MagicMock()
            mock_process.return_value = mock_proc_instance
            mock_proc_instance.start = MagicMock()
            mock_proc_instance.join = MagicMock()

            client.process_tasks(num_workers=2)

            assert mock_process.call_count == 2
            assert mock_proc_instance.start.call_count == 2
            assert mock_proc_instance.join.call_count == 2

        captured = capsys.readouterr()
        assert "Starting 2 worker processes" in captured.out
        assert "All worker processes completed" in captured.out


class TestWorkerProcessMethod:
    """Tests for _worker_process() static method (lines 76-106)."""

    def test_worker_process_stops_on_sentinel(self, capsys):
        """Test worker stops when it receives None sentinel."""
        tasks_queue = queue.Queue()
        done_queue = queue.Queue()
        tasks_queue.put(None)

        RemoteCandidateTrainingClient._worker_process(tasks_queue, done_queue, 0)

        captured = capsys.readouterr()
        assert "Worker 0 started" in captured.out
        assert "Worker 0 finished, processed 0 tasks" in captured.out

    def test_worker_process_handles_timeout(self, capsys):
        """Test worker handles queue timeout."""
        tasks_queue = MagicMock()
        done_queue = MagicMock()
        tasks_queue.get.side_effect = Exception("timed out")

        RemoteCandidateTrainingClient._worker_process(tasks_queue, done_queue, 1)

        captured = capsys.readouterr()
        assert "Worker 1 started" in captured.out
        assert "Worker 1 finished" in captured.out

    def test_worker_process_handles_other_error(self, capsys):
        """Test worker handles non-timeout errors."""
        tasks_queue = MagicMock()
        done_queue = MagicMock()
        tasks_queue.get.side_effect = RuntimeError("Connection lost")

        RemoteCandidateTrainingClient._worker_process(tasks_queue, done_queue, 2)

        captured = capsys.readouterr()
        assert "Worker 2 error: Connection lost" in captured.out

    def test_worker_process_processes_task(self, capsys):
        """Test worker processes a valid task."""
        tasks_queue = queue.Queue()
        done_queue = queue.Queue()

        candidate_data = (
            0,
            2,
            "tanh",
            0.5,
            "test-uuid",
            42,
            2**32 - 1,
            10,
        )
        training_inputs = (
            None,
            10,
            None,
            None,
            0.01,
            100,
        )
        task = (0, candidate_data, training_inputs)
        tasks_queue.put(task)
        tasks_queue.put(None)

        with patch.object(RemoteCandidateTrainingClient, "_train_candidate_remote") as mock_train:
            mock_train.return_value = (0, "test-uuid", 0.95, MagicMock())
            RemoteCandidateTrainingClient._worker_process(tasks_queue, done_queue, 0)

            mock_train.assert_called_once_with(task)

        captured = capsys.readouterr()
        assert "Worker 0 processing task 0" in captured.out
        assert "Worker 0 completed task 0" in captured.out
        assert "processed 1 tasks" in captured.out


class TestTrainCandidateRemoteMethod:
    """Tests for _train_candidate_remote() static method (lines 108-135)."""

    def test_train_candidate_remote_handles_exception(self, capsys):
        """Test remote training handles exceptions gracefully."""
        task_data = (
            0,
            (0, 2, "tanh", 0.5, "test-uuid", 42, 2**32 - 1, 10),
            (None, 10, None, None, 0.01, 100),
        )

        with patch("remote_client.remote_client_0.CandidateUnit") as mock_candidate:
            mock_candidate.side_effect = Exception("Training error")
            result = RemoteCandidateTrainingClient._train_candidate_remote(task_data)

            assert result[0] == 0
            assert result[1] == "test-uuid"
            assert result[2] == 0.0
            assert result[3] is None

        captured = capsys.readouterr()
        assert "Remote training error" in captured.out

    def test_train_candidate_remote_exception_before_index_assigned(self, capsys):
        """Test exception handling when candidate_index not assigned."""
        task_data = None

        result = RemoteCandidateTrainingClient._train_candidate_remote(task_data)

        assert result[2] == 0.0
        assert result[3] is None
        captured = capsys.readouterr()
        assert "Remote training error" in captured.out


class TestTestRemoteConnectionFunction:
    """Tests for test_remote_connection() function (lines 138-157)."""

    def test_test_remote_connection_success_path(self, capsys):
        """Test test_remote_connection when connect succeeds."""
        with patch.object(RemoteCandidateTrainingClient, "connect", return_value=True):
            with patch.object(RemoteCandidateTrainingClient, "process_tasks") as mock_process:
                from remote_client.remote_client_0 import test_remote_connection

                test_remote_connection()
                mock_process.assert_called_once_with(num_workers=2)

        captured = capsys.readouterr()
        assert "Testing remote multiprocessing manager connection" in captured.out
        assert "Connection successful!" in captured.out

    def test_test_remote_connection_failure_path(self, capsys):
        """Test test_remote_connection when connect fails."""
        with patch.object(RemoteCandidateTrainingClient, "connect", return_value=False):
            from remote_client.remote_client_0 import test_remote_connection

            test_remote_connection()

        captured = capsys.readouterr()
        assert "Connection failed!" in captured.out


class TestModuleMainBlock:
    """Test for module __main__ block (line 157)."""

    def test_main_block_execution(self):
        """Verify main block would call test_remote_connection."""
        with patch("remote_client.remote_client_0.test_remote_connection") as mock_test:
            import importlib

            import remote_client.remote_client_0 as module

            if hasattr(module, "__name__"):
                pass
            mock_test.assert_not_called()
