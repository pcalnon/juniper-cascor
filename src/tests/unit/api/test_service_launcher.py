"""Unit tests for api.service_launcher module.

Tests cover ManagedService lifecycle, cleanup helpers, log directory
resolution, health-check polling, and the start_service orchestrator.
"""

import asyncio
import subprocess
import urllib.error
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, call, mock_open, patch

import pytest


# ---------------------------------------------------------------------------
# ManagedService
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestManagedService:
    """Tests for the ManagedService wrapper class."""

    def _make_service(self, poll_return=None, pid=1234, returncode=0, log_handle=None):
        from api.service_launcher import ManagedService

        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = poll_return
        proc.pid = pid
        proc.returncode = returncode
        return ManagedService("test-svc", proc, log_handle)

    # -- is_running ---------------------------------------------------------

    def test_is_running_true_when_poll_none(self):
        svc = self._make_service(poll_return=None)
        assert svc.is_running() is True

    def test_is_running_false_when_poll_returns_code(self):
        svc = self._make_service(poll_return=0)
        assert svc.is_running() is False

    # -- terminate ----------------------------------------------------------

    def test_terminate_already_stopped(self):
        """Terminate on a stopped process should just close the log."""
        handle = MagicMock()
        svc = self._make_service(poll_return=0, returncode=1, log_handle=handle)
        svc.terminate()
        svc.process.terminate.assert_not_called()
        handle.close.assert_called_once()

    def test_terminate_graceful(self):
        """Process stops within timeout."""
        svc = self._make_service(poll_return=None)
        # After terminate() is called, wait() succeeds
        svc.process.wait.return_value = None
        # is_running must return True initially
        svc.process.poll.return_value = None
        svc.terminate(timeout=5.0)
        svc.process.terminate.assert_called_once()
        svc.process.wait.assert_called_once_with(timeout=5.0)
        svc.process.kill.assert_not_called()

    def test_terminate_force_kill_on_timeout(self):
        """Process does not stop gracefully, so SIGKILL is sent."""
        svc = self._make_service(poll_return=None)
        svc.process.wait.side_effect = [subprocess.TimeoutExpired("cmd", 10), None]
        svc.terminate(timeout=10.0)
        svc.process.terminate.assert_called_once()
        svc.process.kill.assert_called_once()
        assert svc.process.wait.call_count == 2

    # -- _close_log ---------------------------------------------------------

    def test_close_log_with_handle(self):
        handle = MagicMock()
        svc = self._make_service(log_handle=handle)
        svc._close_log()
        handle.close.assert_called_once()
        assert svc._log_handle is None

    def test_close_log_without_handle(self):
        svc = self._make_service(log_handle=None)
        svc._close_log()  # must not raise

    def test_close_log_exception_suppressed(self):
        handle = MagicMock()
        handle.close.side_effect = OSError("disk error")
        svc = self._make_service(log_handle=handle)
        svc._close_log()  # should NOT raise
        assert svc._log_handle is None

    def test_close_log_called_after_terminate(self):
        """Terminate should close the log handle after stopping."""
        handle = MagicMock()
        svc = self._make_service(poll_return=None, log_handle=handle)
        svc.process.wait.return_value = None
        svc.terminate()
        handle.close.assert_called_once()
        assert svc._log_handle is None


# ---------------------------------------------------------------------------
# _cleanup_at_exit
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestCleanupAtExit:
    """Tests for the module-level atexit callback."""

    def test_cleanup_terminates_all_services(self):
        from api import service_launcher
        from api.service_launcher import ManagedService

        proc1 = MagicMock(spec=subprocess.Popen)
        proc1.poll.return_value = None
        proc1.wait.return_value = None
        proc1.pid = 1001
        svc1 = ManagedService("svc1", proc1)

        proc2 = MagicMock(spec=subprocess.Popen)
        proc2.poll.return_value = None
        proc2.wait.return_value = None
        proc2.pid = 1002
        svc2 = ManagedService("svc2", proc2)

        original = service_launcher._active_services[:]
        service_launcher._active_services.clear()
        service_launcher._active_services.extend([svc1, svc2])
        try:
            service_launcher._cleanup_at_exit()
            proc1.terminate.assert_called_once()
            proc2.terminate.assert_called_once()
        finally:
            service_launcher._active_services.clear()
            service_launcher._active_services.extend(original)

    def test_cleanup_suppresses_exceptions(self):
        from api import service_launcher
        from api.service_launcher import ManagedService

        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = None
        proc.pid = 1003
        proc.terminate.side_effect = RuntimeError("boom")
        svc = ManagedService("bad-svc", proc)

        original = service_launcher._active_services[:]
        service_launcher._active_services.clear()
        service_launcher._active_services.append(svc)
        try:
            service_launcher._cleanup_at_exit()  # must NOT raise
        finally:
            service_launcher._active_services.clear()
            service_launcher._active_services.extend(original)


# ---------------------------------------------------------------------------
# _resolve_log_dir
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestResolveLogDir:
    """Tests for log directory resolution."""

    def test_resolve_from_constants(self):
        with patch.dict("sys.modules", {"cascor_constants": MagicMock(), "cascor_constants.constants": MagicMock(_PROJECT_LOG_DIR_DEFAULT="/tmp/cascor_logs")}):
            from api.service_launcher import _resolve_log_dir

            result = _resolve_log_dir()
            assert result == Path("/tmp/cascor_logs")

    def test_resolve_fallback_when_import_fails(self):
        """When cascor_constants is unavailable, fall back to <project>/logs."""
        from api import service_launcher

        with patch.dict("sys.modules", {"cascor_constants": None, "cascor_constants.constants": None}):
            # Force reimport of the function with the patched sys.modules
            # by calling the function directly — it catches ImportError internally
            result = service_launcher._resolve_log_dir()
            # Fallback is <source_root>/logs  (3 parents up from service_launcher.py)
            expected = Path(service_launcher.__file__).resolve().parent.parent.parent / "logs"
            assert result == expected


# ---------------------------------------------------------------------------
# wait_for_health
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestWaitForHealth:
    """Tests for the async health-check poller."""

    @pytest.mark.asyncio
    async def test_healthy_on_first_try(self):
        from api.service_launcher import wait_for_health

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("api.service_launcher.urllib.request.urlopen", return_value=mock_resp):
            result = await wait_for_health("http://localhost:8100/v1/health", timeout=5.0, interval=0.01)

        assert result is True

    @pytest.mark.asyncio
    async def test_healthy_after_retries(self):
        from api.service_launcher import wait_for_health

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        call_count = 0

        def fake_urlopen(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("not ready")
            return mock_resp

        with patch("api.service_launcher.urllib.request.urlopen", side_effect=fake_urlopen):
            result = await wait_for_health("http://localhost:8100/v1/health", timeout=10.0, interval=0.01)

        assert result is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_timeout_returns_false(self):
        from api.service_launcher import wait_for_health

        with patch("api.service_launcher.urllib.request.urlopen", side_effect=ConnectionError("nope")):
            result = await wait_for_health("http://localhost:8100/v1/health", timeout=0.05, interval=0.01)

        assert result is False

    @pytest.mark.asyncio
    async def test_non_200_keeps_retrying(self):
        from api.service_launcher import wait_for_health

        mock_resp = MagicMock()
        mock_resp.status = 503
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        call_count = 0

        def fake_urlopen(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return mock_resp
            # Third call returns 200
            good = MagicMock()
            good.status = 200
            good.__enter__ = MagicMock(return_value=good)
            good.__exit__ = MagicMock(return_value=False)
            return good

        with patch("api.service_launcher.urllib.request.urlopen", side_effect=fake_urlopen):
            result = await wait_for_health("http://localhost:8100/v1/health", timeout=10.0, interval=0.01)

        assert result is True


# ---------------------------------------------------------------------------
# start_service
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestStartService:
    """Tests for the start_service orchestrator."""

    @pytest.mark.asyncio
    async def test_start_service_success(self):
        from api import service_launcher
        from api.service_launcher import start_service

        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        mock_proc.pid = 5678

        mock_log_handle = MagicMock()

        original = service_launcher._active_services[:]
        service_launcher._active_services.clear()

        try:
            with (
                patch("api.service_launcher.subprocess.Popen", return_value=mock_proc),
                patch("api.service_launcher._resolve_log_dir", return_value=Path("/tmp/test_logs")),
                patch("builtins.open", return_value=mock_log_handle),
                patch("api.service_launcher.wait_for_health", return_value=True),
                patch.object(Path, "mkdir"),
            ):
                result = await start_service(
                    name="test-data",
                    command="python -m uvicorn app:app",
                    health_url="http://localhost:8100/v1/health",
                    env_overrides={"FOO": "bar"},
                    health_timeout=30.0,
                )

            assert result is not None
            assert result.name == "test-data"
            assert result.process is mock_proc
            assert result in service_launcher._active_services
        finally:
            service_launcher._active_services.clear()
            service_launcher._active_services.extend(original)

    @pytest.mark.asyncio
    async def test_start_service_popen_fails(self):
        from api import service_launcher
        from api.service_launcher import start_service

        mock_log_handle = MagicMock()

        original = service_launcher._active_services[:]
        service_launcher._active_services.clear()

        try:
            with (
                patch("api.service_launcher.subprocess.Popen", side_effect=FileNotFoundError("not found")),
                patch("api.service_launcher._resolve_log_dir", return_value=Path("/tmp/test_logs")),
                patch("builtins.open", return_value=mock_log_handle),
                patch.object(Path, "mkdir"),
            ):
                result = await start_service(
                    name="bad-cmd",
                    command="nonexistent-binary --start",
                    health_url="http://localhost:8100/v1/health",
                )

            assert result is None
            mock_log_handle.close.assert_called_once()
        finally:
            service_launcher._active_services.clear()
            service_launcher._active_services.extend(original)

    @pytest.mark.asyncio
    async def test_start_service_health_check_fails_while_running(self):
        """Service starts but health check times out (still running)."""
        from api import service_launcher
        from api.service_launcher import start_service

        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None  # process still running
        mock_proc.pid = 9999
        mock_proc.wait.return_value = None

        mock_log_handle = MagicMock()

        original = service_launcher._active_services[:]
        service_launcher._active_services.clear()

        try:
            with (
                patch("api.service_launcher.subprocess.Popen", return_value=mock_proc),
                patch("api.service_launcher._resolve_log_dir", return_value=Path("/tmp/test_logs")),
                patch("builtins.open", return_value=mock_log_handle),
                patch("api.service_launcher.wait_for_health", return_value=False),
                patch.object(Path, "mkdir"),
            ):
                result = await start_service(
                    name="unhealthy-svc",
                    command="python app.py",
                    health_url="http://localhost:8100/v1/health",
                    health_timeout=1.0,
                )

            assert result is None
            # Service should have been terminated and removed
            assert len(service_launcher._active_services) == 0
            mock_proc.terminate.assert_called_once()
        finally:
            service_launcher._active_services.clear()
            service_launcher._active_services.extend(original)

    @pytest.mark.asyncio
    async def test_start_service_premature_exit(self):
        """Service process exits before health check passes."""
        from api import service_launcher
        from api.service_launcher import start_service

        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 7777
        mock_proc.returncode = 1
        # After start, the process has already exited
        mock_proc.poll.return_value = 1
        mock_proc.wait.return_value = None

        mock_log_handle = MagicMock()

        original = service_launcher._active_services[:]
        service_launcher._active_services.clear()

        try:
            with (
                patch("api.service_launcher.subprocess.Popen", return_value=mock_proc),
                patch("api.service_launcher._resolve_log_dir", return_value=Path("/tmp/test_logs")),
                patch("builtins.open", return_value=mock_log_handle),
                patch("api.service_launcher.wait_for_health", return_value=False),
                patch.object(Path, "mkdir"),
            ):
                result = await start_service(
                    name="crasher",
                    command="python crash.py",
                    health_url="http://localhost:8100/v1/health",
                )

            assert result is None
            assert len(service_launcher._active_services) == 0
        finally:
            service_launcher._active_services.clear()
            service_launcher._active_services.extend(original)

    @pytest.mark.asyncio
    async def test_start_service_log_file_open_fails(self):
        """When log file cannot be opened, service still starts with /dev/null."""
        from api import service_launcher
        from api.service_launcher import start_service

        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        mock_proc.pid = 4321

        original = service_launcher._active_services[:]
        service_launcher._active_services.clear()

        try:
            with (
                patch("api.service_launcher.subprocess.Popen", return_value=mock_proc) as mock_popen,
                patch("api.service_launcher._resolve_log_dir", return_value=Path("/tmp/test_logs")),
                patch("builtins.open", side_effect=OSError("permission denied")),
                patch("api.service_launcher.wait_for_health", return_value=True),
                patch.object(Path, "mkdir"),
            ):
                result = await start_service(
                    name="log-fail",
                    command="python app.py",
                    health_url="http://localhost:8100/v1/health",
                )

            assert result is not None
            assert result.name == "log-fail"
            # When log open fails, stdout should be DEVNULL
            popen_kwargs = mock_popen.call_args
            assert popen_kwargs.kwargs["stdout"] == subprocess.DEVNULL
            assert popen_kwargs.kwargs["stderr"] == subprocess.DEVNULL
        finally:
            service_launcher._active_services.clear()
            service_launcher._active_services.extend(original)

    @pytest.mark.asyncio
    async def test_start_service_popen_fails_and_log_open_fails(self):
        """When both log file open and Popen fail, the falsy log_handle branch is taken."""
        from api import service_launcher
        from api.service_launcher import start_service

        original = service_launcher._active_services[:]
        service_launcher._active_services.clear()

        try:
            with (
                patch("api.service_launcher.subprocess.Popen", side_effect=FileNotFoundError("not found")),
                patch("api.service_launcher._resolve_log_dir", return_value=Path("/tmp/test_logs")),
                patch("builtins.open", side_effect=OSError("permission denied")),
                patch.object(Path, "mkdir"),
            ):
                result = await start_service(
                    name="double-fail",
                    command="nonexistent --start",
                    health_url="http://localhost:8100/v1/health",
                )

            assert result is None
        finally:
            service_launcher._active_services.clear()
            service_launcher._active_services.extend(original)

    @pytest.mark.asyncio
    async def test_start_service_no_env_overrides(self):
        """start_service works without env_overrides (default None)."""
        from api import service_launcher
        from api.service_launcher import start_service

        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        mock_proc.pid = 1111

        mock_log_handle = MagicMock()

        original = service_launcher._active_services[:]
        service_launcher._active_services.clear()

        try:
            with (
                patch("api.service_launcher.subprocess.Popen", return_value=mock_proc),
                patch("api.service_launcher._resolve_log_dir", return_value=Path("/tmp/test_logs")),
                patch("builtins.open", return_value=mock_log_handle),
                patch("api.service_launcher.wait_for_health", return_value=True),
                patch.object(Path, "mkdir"),
            ):
                result = await start_service(
                    name="no-env",
                    command="python app.py",
                    health_url="http://localhost:8100/v1/health",
                )

            assert result is not None
        finally:
            service_launcher._active_services.clear()
            service_launcher._active_services.extend(original)

    @pytest.mark.asyncio
    async def test_start_service_shlex_parses_command(self):
        """Verify that the command string is properly split by shlex."""
        from api import service_launcher
        from api.service_launcher import start_service

        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        mock_proc.pid = 2222

        mock_log_handle = MagicMock()

        original = service_launcher._active_services[:]
        service_launcher._active_services.clear()

        try:
            with (
                patch("api.service_launcher.subprocess.Popen", return_value=mock_proc) as mock_popen,
                patch("api.service_launcher._resolve_log_dir", return_value=Path("/tmp/test_logs")),
                patch("builtins.open", return_value=mock_log_handle),
                patch("api.service_launcher.wait_for_health", return_value=True),
                patch.object(Path, "mkdir"),
            ):
                result = await start_service(
                    name="shlex-test",
                    command='python -m uvicorn "app:main" --host 0.0.0.0',
                    health_url="http://localhost:8100/v1/health",
                )

            assert result is not None
            cmd_parts = mock_popen.call_args.args[0]
            assert cmd_parts == ["python", "-m", "uvicorn", "app:main", "--host", "0.0.0.0"]
        finally:
            service_launcher._active_services.clear()
            service_launcher._active_services.extend(original)
