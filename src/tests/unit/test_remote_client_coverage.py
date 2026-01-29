#!/usr/bin/env python
"""
Tests for remote_client module to increase code coverage.
Tests RemoteWorkerClient and RemoteCandidateTrainingClient classes.
"""
import os
import sys
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from remote_client.remote_client import RemoteWorkerClient
from remote_client.remote_client_0 import RemoteCandidateTrainingClient


@pytest.mark.unit
class TestRemoteWorkerClientInit:
    """Tests for RemoteWorkerClient initialization."""

    def test_init_with_string_authkey(self):
        """Test initialization with string authkey converts to bytes."""
        address = ("127.0.0.1", 50000)
        authkey = "test_authkey"

        with patch("remote_client.remote_client.mp.get_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            client = RemoteWorkerClient(address, authkey)

        assert client.authkey == b"test_authkey"
        assert client.address == address

    def test_init_with_bytes_authkey(self):
        """Test initialization with bytes authkey keeps as bytes."""
        address = ("127.0.0.1", 50000)
        authkey = b"test_authkey_bytes"

        with patch("remote_client.remote_client.mp.get_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            client = RemoteWorkerClient(address, authkey)

        assert client.authkey == b"test_authkey_bytes"

    def test_init_with_custom_context(self):
        """Test initialization with custom multiprocessing context."""
        address = ("127.0.0.1", 50000)
        authkey = b"test"
        mock_ctx = MagicMock()

        client = RemoteWorkerClient(address, authkey, ctx=mock_ctx)

        assert client.ctx is mock_ctx

    def test_init_with_custom_logger(self):
        """Test initialization with custom logger."""
        address = ("127.0.0.1", 50000)
        authkey = b"test"
        mock_logger = MagicMock()

        with patch("remote_client.remote_client.mp.get_context"):
            client = RemoteWorkerClient(address, authkey, logger=mock_logger)

        assert client.logger is mock_logger

    def test_init_defaults(self):
        """Test initialization sets correct defaults."""
        address = ("localhost", 12345)
        authkey = b"key"

        with patch("remote_client.remote_client.mp.get_context") as mock_ctx:
            mock_ctx.return_value = MagicMock()
            client = RemoteWorkerClient(address, authkey)

        assert client.manager is None
        assert client.task_queue is None
        assert client.result_queue is None
        assert client.workers == []


@pytest.mark.unit
class TestRemoteWorkerClientConnect:
    """Tests for RemoteWorkerClient.connect method."""

    def test_connect_success(self):
        """Test successful connection to remote manager."""
        address = ("127.0.0.1", 50000)
        authkey = b"test"

        with patch("remote_client.remote_client.mp.get_context"):
            client = RemoteWorkerClient(address, authkey)

        mock_manager = MagicMock()
        mock_task_queue = MagicMock()
        mock_result_queue = MagicMock()
        mock_manager.get_task_queue.return_value = mock_task_queue
        mock_manager.get_result_queue.return_value = mock_result_queue

        with patch.dict("sys.modules", {"cascade_correlation.cascade_correlation": MagicMock(CandidateTrainingManager=MagicMock(return_value=mock_manager))}):
            client.connect()

        assert client.manager is mock_manager
        assert client.task_queue is mock_task_queue
        assert client.result_queue is mock_result_queue
        mock_manager.connect.assert_called_once()

    def test_connect_failure_raises_exception(self):
        """Test connection failure raises exception."""
        address = ("127.0.0.1", 50000)
        authkey = b"test"

        with patch("remote_client.remote_client.mp.get_context"):
            client = RemoteWorkerClient(address, authkey)

        mock_module = MagicMock()
        mock_module.CandidateTrainingManager.side_effect = ConnectionRefusedError("Connection refused")

        with patch.dict("sys.modules", {"cascade_correlation.cascade_correlation": mock_module}):
            with pytest.raises(ConnectionRefusedError):
                client.connect()

    def test_connect_manager_connect_failure(self):
        """Test manager.connect() failure is handled."""
        address = ("127.0.0.1", 50000)
        authkey = b"test"

        with patch("remote_client.remote_client.mp.get_context"):
            client = RemoteWorkerClient(address, authkey)

        mock_manager = MagicMock()
        mock_manager.connect.side_effect = OSError("Network unreachable")
        mock_module = MagicMock()
        mock_module.CandidateTrainingManager.return_value = mock_manager

        with patch.dict("sys.modules", {"cascade_correlation.cascade_correlation": mock_module}):
            with pytest.raises(OSError):
                client.connect()


@pytest.mark.unit
class TestRemoteWorkerClientStartWorkers:
    """Tests for RemoteWorkerClient.start_workers method."""

    def test_start_workers_without_connect_raises(self):
        """Test starting workers without connecting raises RuntimeError."""
        address = ("127.0.0.1", 50000)
        authkey = b"test"

        with patch("remote_client.remote_client.mp.get_context"):
            client = RemoteWorkerClient(address, authkey)

        with pytest.raises(RuntimeError, match="Must call connect"):
            client.start_workers(num_workers=2)

    def test_start_workers_creates_processes(self):
        """Test start_workers creates the correct number of worker processes."""
        address = ("127.0.0.1", 50000)
        authkey = b"test"

        mock_ctx = MagicMock()
        mock_process = MagicMock()
        mock_ctx.Process.return_value = mock_process

        with patch("remote_client.remote_client.mp.get_context", return_value=mock_ctx):
            client = RemoteWorkerClient(address, authkey)

        client.manager = MagicMock()
        client.task_queue = MagicMock()
        client.result_queue = MagicMock()

        mock_module = MagicMock()
        with patch.dict("sys.modules", {"cascade_correlation.cascade_correlation": mock_module}):
            client.start_workers(num_workers=3)

        assert len(client.workers) == 3
        assert mock_ctx.Process.call_count == 3
        assert mock_process.start.call_count == 3

    def test_start_workers_default_count(self):
        """Test start_workers with default worker count."""
        address = ("127.0.0.1", 50000)
        authkey = b"test"

        mock_ctx = MagicMock()
        mock_process = MagicMock()
        mock_ctx.Process.return_value = mock_process

        with patch("remote_client.remote_client.mp.get_context", return_value=mock_ctx):
            client = RemoteWorkerClient(address, authkey)

        client.manager = MagicMock()
        client.task_queue = MagicMock()
        client.result_queue = MagicMock()

        mock_module = MagicMock()
        with patch.dict("sys.modules", {"cascade_correlation.cascade_correlation": mock_module}):
            client.start_workers()

        assert len(client.workers) == 1


@pytest.mark.unit
class TestRemoteWorkerClientStopWorkers:
    """Tests for RemoteWorkerClient.stop_workers method."""

    def test_stop_workers_no_workers(self):
        """Test stop_workers with no active workers."""
        address = ("127.0.0.1", 50000)
        authkey = b"test"

        with patch("remote_client.remote_client.mp.get_context"):
            client = RemoteWorkerClient(address, authkey)

        client.stop_workers()
        assert client.workers == []

    def test_stop_workers_sends_sentinels(self):
        """Test stop_workers sends sentinel values to task queue."""
        address = ("127.0.0.1", 50000)
        authkey = b"test"

        with patch("remote_client.remote_client.mp.get_context"):
            client = RemoteWorkerClient(address, authkey)

        mock_queue = MagicMock()
        client.task_queue = mock_queue

        mock_worker1 = MagicMock()
        mock_worker1.is_alive.return_value = False
        mock_worker2 = MagicMock()
        mock_worker2.is_alive.return_value = False

        client.workers = [mock_worker1, mock_worker2]

        client.stop_workers(timeout=5)

        assert mock_queue.put.call_count == 2
        mock_queue.put.assert_called_with(None)

    def test_stop_workers_terminates_unresponsive(self):
        """Test stop_workers terminates workers that don't stop gracefully."""
        address = ("127.0.0.1", 50000)
        authkey = b"test"

        with patch("remote_client.remote_client.mp.get_context"):
            client = RemoteWorkerClient(address, authkey)

        mock_queue = MagicMock()
        client.task_queue = mock_queue

        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = True
        mock_worker.name = "TestWorker"

        client.workers = [mock_worker]

        client.stop_workers(timeout=1)

        mock_worker.terminate.assert_called_once()
        assert client.workers == []

    def test_stop_workers_handles_queue_error(self):
        """Test stop_workers handles errors when putting sentinels."""
        address = ("127.0.0.1", 50000)
        authkey = b"test"

        with patch("remote_client.remote_client.mp.get_context"):
            client = RemoteWorkerClient(address, authkey)

        mock_queue = MagicMock()
        mock_queue.put.side_effect = Exception("Queue error")
        client.task_queue = mock_queue

        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = False

        client.workers = [mock_worker]

        client.stop_workers()
        assert client.workers == []


@pytest.mark.unit
class TestRemoteWorkerClientDisconnect:
    """Tests for RemoteWorkerClient.disconnect method."""

    def test_disconnect_stops_workers_first(self):
        """Test disconnect stops workers before disconnecting."""
        address = ("127.0.0.1", 50000)
        authkey = b"test"

        with patch("remote_client.remote_client.mp.get_context"):
            client = RemoteWorkerClient(address, authkey)

        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = False
        client.workers = [mock_worker]
        client.task_queue = MagicMock()
        client.manager = MagicMock()

        client.disconnect()

        assert client.manager is None
        assert client.task_queue is None
        assert client.result_queue is None

    def test_disconnect_without_manager(self):
        """Test disconnect when not connected."""
        address = ("127.0.0.1", 50000)
        authkey = b"test"

        with patch("remote_client.remote_client.mp.get_context"):
            client = RemoteWorkerClient(address, authkey)

        client.disconnect()
        assert client.manager is None


@pytest.mark.unit
class TestRemoteWorkerClientContextManager:
    """Tests for RemoteWorkerClient context manager protocol."""

    def test_context_manager_enter(self):
        """Test context manager __enter__ calls connect."""
        address = ("127.0.0.1", 50000)
        authkey = b"test"

        with patch("remote_client.remote_client.mp.get_context"):
            client = RemoteWorkerClient(address, authkey)

        mock_manager = MagicMock()
        mock_manager.get_task_queue.return_value = MagicMock()
        mock_manager.get_result_queue.return_value = MagicMock()

        with patch.dict("sys.modules", {"cascade_correlation.cascade_correlation": MagicMock(CandidateTrainingManager=MagicMock(return_value=mock_manager))}):
            result = client.__enter__()

        assert result is client
        mock_manager.connect.assert_called_once()

    def test_context_manager_exit(self):
        """Test context manager __exit__ calls disconnect."""
        address = ("127.0.0.1", 50000)
        authkey = b"test"

        with patch("remote_client.remote_client.mp.get_context"):
            client = RemoteWorkerClient(address, authkey)

        client.manager = MagicMock()

        client.__exit__(None, None, None)

        assert client.manager is None

    def test_context_manager_exit_with_exception(self):
        """Test context manager __exit__ with exception still disconnects."""
        address = ("127.0.0.1", 50000)
        authkey = b"test"

        with patch("remote_client.remote_client.mp.get_context"):
            client = RemoteWorkerClient(address, authkey)

        client.manager = MagicMock()

        client.__exit__(ValueError, ValueError("test"), None)

        assert client.manager is None


@pytest.mark.unit
class TestRemoteCandidateTrainingClientInit:
    """Tests for RemoteCandidateTrainingClient initialization."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        client = RemoteCandidateTrainingClient()

        assert client.server_address == ("127.0.0.1", 50000)
        assert client.authkey == b"Juniper_Cascade_Correlation_Multiprocessing_Authkey"
        assert client.manager is None

    def test_init_custom_address(self):
        """Test initialization with custom address."""
        address = ("192.168.1.100", 60000)
        client = RemoteCandidateTrainingClient(server_address=address)

        assert client.server_address == address

    def test_init_custom_authkey(self):
        """Test initialization with custom authkey."""
        authkey = b"custom_key"
        client = RemoteCandidateTrainingClient(authkey=authkey)

        assert client.authkey == authkey


@pytest.mark.unit
class TestRemoteCandidateTrainingClientConnect:
    """Tests for RemoteCandidateTrainingClient.connect method."""

    def test_connect_success(self):
        """Test successful connection returns True."""
        client = RemoteCandidateTrainingClient()

        with patch.object(client, "manager", create=True) as mock_manager:
            with patch("remote_client.remote_client_0.BaseManager") as mock_base_manager:
                mock_instance = MagicMock()
                mock_base_manager.return_value = mock_instance

                with patch.object(RemoteCandidateTrainingClient, "connect", return_value=True):
                    result = client.connect()

        assert result is True

    def test_connect_failure_returns_false(self):
        """Test connection failure returns False."""
        client = RemoteCandidateTrainingClient()

        with patch("remote_client.remote_client_0.RemoteCandidateTrainingClient.connect") as mock_connect:
            mock_connect.return_value = False
            result = mock_connect()

        assert result is False


@pytest.mark.unit
class TestRemoteCandidateTrainingClientProcessTasks:
    """Tests for RemoteCandidateTrainingClient.process_tasks method."""

    def test_process_tasks_without_manager(self, capsys):
        """Test process_tasks without connection prints error."""
        client = RemoteCandidateTrainingClient()

        client.process_tasks(num_workers=2)

        captured = capsys.readouterr()
        assert "Not connected to manager" in captured.out

    def test_process_tasks_with_manager(self):
        """Test process_tasks starts worker processes."""
        client = RemoteCandidateTrainingClient()
        client.manager = MagicMock()

        mock_tasks_queue = MagicMock()
        mock_done_queue = MagicMock()
        client.manager.get_tasks_queue.return_value = mock_tasks_queue
        client.manager.get_done_queue.return_value = mock_done_queue

        with patch("remote_client.remote_client_0.mp.Process") as mock_process:
            mock_proc = MagicMock()
            mock_process.return_value = mock_proc

            client.process_tasks(num_workers=2)

            assert mock_process.call_count == 2
            assert mock_proc.start.call_count == 2
            assert mock_proc.join.call_count == 2


@pytest.mark.unit
class TestRemoteCandidateTrainingClientWorkerProcess:
    """Tests for RemoteCandidateTrainingClient._worker_process static method."""

    def test_worker_process_exits_on_sentinel(self, capsys):
        """Test worker process exits when receiving None sentinel."""
        mock_tasks_queue = MagicMock()
        mock_tasks_queue.get.return_value = None
        mock_done_queue = MagicMock()

        RemoteCandidateTrainingClient._worker_process(mock_tasks_queue, mock_done_queue, 0)

        captured = capsys.readouterr()
        assert "Worker 0 started" in captured.out
        assert "Worker 0 finished" in captured.out

    def test_worker_process_handles_timeout(self, capsys):
        """Test worker process handles queue timeout."""
        mock_tasks_queue = MagicMock()
        mock_tasks_queue.get.side_effect = Exception("timed out")
        mock_done_queue = MagicMock()

        RemoteCandidateTrainingClient._worker_process(mock_tasks_queue, mock_done_queue, 1)

        captured = capsys.readouterr()
        assert "Worker 1 started" in captured.out
        assert "Worker 1 finished" in captured.out

    def test_worker_process_handles_error(self, capsys):
        """Test worker process handles other exceptions."""
        mock_tasks_queue = MagicMock()
        mock_tasks_queue.get.side_effect = Exception("Some other error")
        mock_done_queue = MagicMock()

        RemoteCandidateTrainingClient._worker_process(mock_tasks_queue, mock_done_queue, 2)

        captured = capsys.readouterr()
        assert "Worker 2 error" in captured.out


@pytest.mark.unit
class TestRemoteCandidateTrainingClientTrainRemote:
    """Tests for RemoteCandidateTrainingClient._train_candidate_remote static method."""

    def test_train_candidate_remote_error_handling(self, capsys):
        """Test _train_candidate_remote handles errors gracefully."""
        invalid_task = (0, "invalid_data", "more_invalid")

        result = RemoteCandidateTrainingClient._train_candidate_remote(invalid_task)

        captured = capsys.readouterr()
        assert "Remote training error" in captured.out
        assert result[2] == 0.0
        assert result[3] is None


@pytest.mark.unit
class TestTestRemoteConnection:
    """Tests for test_remote_connection function."""

    def test_test_remote_connection_success(self, capsys):
        """Test test_remote_connection with successful connection."""
        with patch.object(RemoteCandidateTrainingClient, "connect", return_value=True):
            with patch.object(RemoteCandidateTrainingClient, "process_tasks"):
                from remote_client.remote_client_0 import test_remote_connection

                test_remote_connection()

        captured = capsys.readouterr()
        assert "Testing remote multiprocessing manager connection" in captured.out

    def test_test_remote_connection_failure(self, capsys):
        """Test test_remote_connection with failed connection."""
        with patch.object(RemoteCandidateTrainingClient, "connect", return_value=False):
            from remote_client.remote_client_0 import test_remote_connection

            test_remote_connection()

        captured = capsys.readouterr()
        assert "Connection failed!" in captured.out
