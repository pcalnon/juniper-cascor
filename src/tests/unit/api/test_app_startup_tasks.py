"""Phase 3C regression tests for fire-and-forget startup tasks.

CONC-09 — `lifespan()` previously created auto-start tasks with
`asyncio.create_task(...)` and immediately discarded the return value, so:

  1. The task held no strong reference and could be garbage-collected mid-flight
     — see https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
  2. Any exception raised inside the task was swallowed by the loop and only
     surfaced (if at all) as the cryptic "Task exception was never retrieved"
     warning when the task was GC'd.

The fix stores every startup task on `app.state.startup_tasks` and attaches
`_log_startup_task_exception` as a done-callback so failures are logged with
full traceback. Cancellation during shutdown clears the list cleanly.

These tests exercise just the helper + lifespan wiring; they don't spin up
a full FastAPI server (which would require torch + the rest of the cascor
import chain).
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_APP_SRC = Path(__file__).resolve().parents[3] / "api" / "app.py"


@pytest.mark.unit
class TestStartupTaskWiringSourceLevel:
    """Source-level guards that don't require importing api.app (and therefore torch).

    The behavioural tests below import `_log_startup_task_exception` from
    api.app, which transitively imports torch. In environments where torch's
    C-extension is broken (`torch._C` directory shadowing `torch._C.so`
    under Python 3.14 free-threading; documented in auto-memory) those tests
    will skip via `importorskip`. These source-level checks always run and
    guard against accidental removal of the helpers or regression of the
    lifespan task tracking.
    """

    @classmethod
    def setup_class(cls) -> None:
        cls.source = _APP_SRC.read_text(encoding="utf-8")

    def test_log_startup_task_exception_helper_defined(self):
        assert re.search(r"^def\s+_log_startup_task_exception\s*\(\s*task:\s*asyncio\.Task\s*\)\s*->\s*None\s*:", self.source, re.MULTILINE), "_log_startup_task_exception helper missing"

    def test_lifespan_stores_startup_tasks(self):
        assert "app.state.startup_tasks = startup_tasks" in self.source, "lifespan no longer stores startup_tasks on app.state"

    def test_create_task_calls_register_done_callback(self):
        # Both auto-start tasks must (a) get a name and (b) attach the
        # exception-logger as a done-callback before being appended to
        # startup_tasks. We assert the callback line appears at least twice.
        callbacks = re.findall(r"task\.add_done_callback\(\s*_log_startup_task_exception\s*\)", self.source)
        assert len(callbacks) >= 2, f"expected ≥2 add_done_callback hookups for startup tasks, found {len(callbacks)}"

    def test_shutdown_cancels_in_flight_startup_tasks(self):
        # The shutdown stanza must collect un-finished tasks, cancel each,
        # and gather them with return_exceptions=True so a cancelled task
        # does not propagate to the caller.
        assert "task.cancel()" in self.source
        assert "asyncio.gather" in self.source and "return_exceptions=True" in self.source, "shutdown must await startup tasks with return_exceptions=True"


# Defer the import until the behavioural fixture asks for it.


@pytest.fixture(scope="module")
def _app_module():
    pytest.importorskip("torch", exc_type=ImportError)
    from api import app as app_module

    return app_module


@pytest.fixture
def _log_startup_task_exception(_app_module):
    return _app_module._log_startup_task_exception


@pytest.mark.unit
class TestLogStartupTaskException:
    """`_log_startup_task_exception` callback contract."""

    def test_silent_on_clean_completion(self, _log_startup_task_exception, caplog):
        """A task that returns normally produces no log output."""
        loop = asyncio.new_event_loop()
        try:

            async def _noop() -> str:
                return "done"

            task = loop.create_task(_noop(), name="auto_start_clean")
            loop.run_until_complete(task)

            with caplog.at_level(logging.ERROR, logger="juniper_cascor.api"):
                _log_startup_task_exception(task)

            assert not caplog.records
        finally:
            loop.close()

    def test_silent_on_cancellation(self, _log_startup_task_exception, caplog):
        """A cancelled task is not treated as a failure."""
        loop = asyncio.new_event_loop()
        try:

            async def _sleeper() -> None:
                await asyncio.sleep(60)

            task = loop.create_task(_sleeper(), name="auto_start_cancelled")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                loop.run_until_complete(task)

            with caplog.at_level(logging.ERROR, logger="juniper_cascor.api"):
                _log_startup_task_exception(task)

            assert not caplog.records
        finally:
            loop.close()

    def test_logs_with_traceback_on_exception(self, _log_startup_task_exception, caplog):
        """A task that raised must log at error level with the original exception."""
        loop = asyncio.new_event_loop()
        try:

            async def _bad() -> None:
                raise RuntimeError("auto-start blew up")

            task = loop.create_task(_bad(), name="auto_start_failing")
            with pytest.raises(RuntimeError):
                loop.run_until_complete(task)

            # ``caplog.at_level(level, logger=...)`` only elevates the level on
            # the named logger; caplog's handler stays attached to root, so we
            # rely on propagation. Some test environments (cascor's logging
            # setup loaded via dictConfig in other tests, dictConfig with
            # ``disable_existing_loggers``) can suppress that propagation,
            # producing the misleading "logger.error never fired" failure mode
            # observed in V35b. Attach caplog's handler directly to the
            # ``juniper_cascor.api`` logger to make the test independent of
            # propagation state. Pin the logger level so the record passes the
            # handler filter regardless of the inherited effective level.
            target_logger = logging.getLogger("juniper_cascor.api")
            previous_level = target_logger.level
            target_logger.setLevel(logging.ERROR)
            target_logger.addHandler(caplog.handler)
            try:
                _log_startup_task_exception(task)
            finally:
                target_logger.removeHandler(caplog.handler)
                target_logger.setLevel(previous_level)

            assert len(caplog.records) == 1
            record = caplog.records[0]
            assert record.levelno == logging.ERROR
            assert "auto_start_failing" in record.getMessage()
            assert "auto-start blew up" in record.getMessage()
            # exc_info should be attached for the traceback.
            assert record.exc_info is not None
            assert isinstance(record.exc_info[1], RuntimeError)
        finally:
            loop.close()


@pytest.mark.unit
class TestLifespanStartupTaskTracking:
    """Lifespan must store task references and cancel them on shutdown."""

    def test_lifespan_stores_and_cancels_startup_tasks(self, _log_startup_task_exception):
        """Imitate the lifespan task-tracking shape and assert the cleanup contract."""
        # We don't drive lifespan() directly here — that would require booting
        # the entire cascor app, which depends on torch and the full import
        # chain. Instead, we assert the contract that lifespan now relies on:
        # (a) startup_tasks is a list of asyncio.Task on app.state, and
        # (b) the shutdown phase cancels any task that hasn't completed and
        #     awaits them with return_exceptions=True.

        async def _slow() -> None:
            await asyncio.sleep(60)

        async def _drive() -> None:
            app = MagicMock()
            startup_tasks: list[asyncio.Task] = []
            app.state.startup_tasks = startup_tasks

            t1 = asyncio.create_task(_slow(), name="auto_start_training")
            t1.add_done_callback(_log_startup_task_exception)
            startup_tasks.append(t1)

            t2 = asyncio.create_task(_slow(), name="auto_start_canopy")
            t2.add_done_callback(_log_startup_task_exception)
            startup_tasks.append(t2)

            # Sanity: tasks are live and tracked.
            assert all(isinstance(t, asyncio.Task) for t in app.state.startup_tasks)
            assert all(not t.done() for t in app.state.startup_tasks)

            # Mirror the new shutdown stanza in lifespan().
            in_flight = [t for t in app.state.startup_tasks if not t.done()]
            for task in in_flight:
                task.cancel()
            results = await asyncio.gather(*in_flight, return_exceptions=True)

            # Every cancelled task surfaces a CancelledError to gather() but
            # the gather itself does not raise, and every task is now done.
            assert len(results) == 2
            assert all(isinstance(r, asyncio.CancelledError) for r in results)
            assert all(t.done() for t in app.state.startup_tasks)

        asyncio.run(_drive())
