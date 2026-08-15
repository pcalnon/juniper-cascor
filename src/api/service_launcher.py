"""Subprocess launcher for companion Juniper services.

Starts juniper-data and juniper-canopy as managed subprocesses when
auto-start is enabled via JUNIPER_CASCOR_AUTO_START_DATA_SERVICE and
JUNIPER_CASCOR_AUTO_START_CANOPY environment variables.

Primarily intended for non-containerized (local development) environments
where Docker Compose is not managing service orchestration.  In Docker
deployments, use ``depends_on`` with ``condition: service_healthy`` instead.
"""

import asyncio
import atexit
import logging
import os
import shlex
import subprocess  # nosec B404 — subprocess is the core purpose of this module
import urllib.request
from pathlib import Path

from cascor_constants.constants_api import _PROJECT_API_HEALTH_CHECK_HTTP_TIMEOUT, _PROJECT_API_PROCESS_TERMINATION_TIMEOUT, _PROJECT_API_SERVICE_DEFAULT_TERMINATE_TIMEOUT, _PROJECT_API_SERVICE_HEALTH_POLL_INTERVAL, _PROJECT_API_SERVICE_HEALTH_POLL_TIMEOUT, _PROJECT_API_SERVICE_TERMINATION_TIMEOUT

logger = logging.getLogger("juniper_cascor.service_launcher")

_active_services: list["ManagedService"] = []


class ManagedService:
    """A subprocess-managed companion service with lifecycle support."""

    def __init__(
        self,
        name: str,
        process: subprocess.Popen,
        log_handle: object | None = None,
    ):
        self.name = name
        self.process = process
        self._log_handle = log_handle

    def is_running(self) -> bool:
        return self.process.poll() is None

    def terminate(self, timeout: float = _PROJECT_API_SERVICE_DEFAULT_TERMINATE_TIMEOUT) -> None:
        # Always close the log handle — even when wait/kill raises — so a
        # failed companion startup cannot leak open file descriptors.
        try:
            if not self.is_running():
                logger.debug(f"{self.name} already stopped (rc={self.process.returncode})")
                return
            logger.info(f"Terminating {self.name} (pid={self.process.pid})")
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
                logger.info(f"{self.name} stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning(f"{self.name} did not stop in {timeout}s, sending SIGKILL")
                self.process.kill()
                self.process.wait(timeout=_PROJECT_API_PROCESS_TERMINATION_TIMEOUT)
                logger.info(f"{self.name} killed")
        finally:
            self._close_log()

    def _close_log(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:  # nosec B110 — cleanup must not propagate exceptions
                pass
            self._log_handle = None


def _cleanup_at_exit() -> None:
    """Terminate all managed services on interpreter exit."""
    for svc in _active_services:
        try:
            svc.terminate(timeout=_PROJECT_API_SERVICE_TERMINATION_TIMEOUT)
        except Exception:  # nosec B110 — cleanup must not propagate exceptions
            pass


atexit.register(_cleanup_at_exit)


def _resolve_log_dir() -> Path:
    """Resolve the canonical log directory for subprocess output.

    Honours the ``JUNIPER_CASCOR_LOG_DIR`` override first (Q-6 / H-7) so a per-run
    launcher keeps subprocess output inside its own ``RUN_DIR/logs`` rather than the
    checkout-shared ``<repo>/logs``. Read at call time, not import time, so the
    override also holds on the ``ImportError`` fallback below, which never consults
    the constants. A set-but-blank value is treated as unset (the blank-env guard class).
    """
    override = os.environ.get("JUNIPER_CASCOR_LOG_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    try:
        from cascor_constants.constants import _PROJECT_LOG_DIR_DEFAULT

        return Path(_PROJECT_LOG_DIR_DEFAULT)
    except ImportError:
        return Path(__file__).resolve().parent.parent.parent / "logs"


async def wait_for_health(
    url: str,
    timeout: float = _PROJECT_API_SERVICE_HEALTH_POLL_TIMEOUT,
    interval: float = _PROJECT_API_SERVICE_HEALTH_POLL_INTERVAL,
) -> bool:
    """Poll a health endpoint until it responds HTTP 200 or timeout expires."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=_PROJECT_API_HEALTH_CHECK_HTTP_TIMEOUT) as resp:  # nosec B310 — internal health check URL from configuration
                if resp.status == 200:
                    return True
        except Exception:  # nosec B110 — health poll retries on any exception
            pass
        await asyncio.sleep(interval)
    return False


async def start_service(
    name: str,
    command: str,
    health_url: str,
    env_overrides: dict[str, str] | None = None,
    health_timeout: float = _PROJECT_API_SERVICE_HEALTH_POLL_TIMEOUT,
) -> ManagedService | None:
    """Start a service as a subprocess and wait for it to become healthy.

    Args:
        name: Human-readable service name for logging.
        command: Shell command string to start the service (parsed with shlex).
        health_url: URL to poll for health status (expects HTTP 200).
        env_overrides: Additional environment variables for the subprocess.
        health_timeout: Seconds to wait for the health check to pass.

    Returns:
        ManagedService instance if started successfully, None otherwise.
    """
    cmd_parts = shlex.split(command)
    logger.info(f"Starting {name}: {command}")

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    # Redirect subprocess output to a log file for diagnostics
    log_dir = _resolve_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.lower().replace(" ", "_").replace("-", "_")
    log_file = log_dir / f"subprocess_{safe_name}.log"

    log_handle = None
    stdout_target = subprocess.DEVNULL
    stderr_target = subprocess.DEVNULL
    try:
        log_handle = open(log_file, "a", encoding="utf-8")
        stdout_target = log_handle
        stderr_target = subprocess.STDOUT
        logger.info(f"{name} output -> {log_file}")
    except OSError:
        logger.warning(f"Could not open log file {log_file}, using /dev/null")

    try:
        process = subprocess.Popen(  # nosec B603 — command is from settings, not user input
            cmd_parts,
            env=env,
            stdout=stdout_target,
            stderr=stderr_target,
            start_new_session=True,
        )
    except Exception:
        logger.exception(f"Failed to start {name}")
        if log_handle:
            log_handle.close()
        return None

    service = ManagedService(name, process, log_handle)
    _active_services.append(service)

    logger.info(f"Waiting for {name} health at {health_url} (timeout={health_timeout}s)")
    try:
        healthy = await wait_for_health(health_url, timeout=health_timeout)
    except Exception:
        # Health probe failures (unexpected exceptions, cancellation) must
        # still tear down the subprocess and drop the active-service entry —
        # otherwise atexit / shutdown can chase an orphaned companion.
        logger.exception(f"{name} health probe raised unexpectedly")
        healthy = False

    if not healthy:
        if service.is_running():
            logger.error(f"{name} started but health check failed after {health_timeout}s")
        else:
            logger.error(f"{name} exited prematurely (rc={process.returncode})")
        try:
            service.terminate()
        except Exception:  # nosec B110 — cleanup must not leave a stale registry entry
            logger.exception(f"Failed to terminate unhealthy {name}")
        finally:
            if service in _active_services:
                _active_services.remove(service)
        return None

    logger.info(f"{name} is healthy (pid={process.pid})")
    return service
