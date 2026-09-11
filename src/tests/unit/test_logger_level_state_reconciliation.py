"""Pin the logger's configured level to ONE state, read by both the guard and the emit filter.

Project:     juniper-cascor
Sub-Project: logging
Author:      Paul Calnon
License:     MIT License

WHY THIS FILE EXISTS
``Logger.set_level()`` used to be a **no-op for emission**. Two disjoint pieces of class state
carried "the configured level":

* ``_log_level`` -- written by ``set_level``, read by ``get_level`` and therefore by
  ``isEnabledFor``. **The guard path.**
* ``_level_logger_name`` / ``_level_logger_config`` -- read by ``_log_at_level`` to build the
  threshold it hands ``_filter_by_level``. ``_level_logger_name`` was assigned once in the class
  body and **never written again anywhere in the repository**. **The emit path.**

So ``Logger.set_level("TRACE")`` -- which ``candidate_unit.py`` calls on the construction of every
candidate -- opened every guard while the records behind them were still discarded. The guarded
work (argument evaluation, ``.shape`` reads, frame and timestamp capture) was paid, and nothing
came out.

**This defect is invisible to log inspection**: a correct guard and a broken guard produce the
same log, because the thing that differs is work done for records that are then thrown away. It
can only be caught by comparing the two paths, which is what this file does.

WHY THE EMIT TEST RE-IMPORTS THE MODULE
``src/tests/conftest.py`` installs a session-scoped autouse fixture that replaces
``Logger._log_at_level`` with a no-op. Every filter decision, every ``print``, every ``open`` --
the whole emit path -- is therefore unreachable from the suite as normally imported, and a test
that drove ``Logger.trace(...)`` here would pass no matter what the emit path read. That is the
vacuous-pass shape this arc exists to remove, so ``TestEmitPathReadsLogLevel`` loads a PRIVATE
copy of the module under a different name, which the fixture has not patched, and drives the real
code.

WHAT WOULD BREAK THIS
Re-introducing a second configured-level state and wiring it into the filter. That is what
``test_mutating_the_retired_state_changes_nothing`` guards: ``_level_logger_name`` and
``_level_logger_config`` are retained for the arc's instrument but are NOT level state any more,
and mutating them must not move a single emission decision.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path

import pytest

from log_config.logger.logger import Logger

#: REQUIRED, not decorative. CI's unit lane runs ``pytest -m "unit and not slow" src/tests/unit``
#: (ci.yml:312-317), so an UNMARKED file in this directory is collected and then silently
#: deselected -- it never runs, and nothing reports that it did not. Measured 2026-09-11: 12 files
#: / 156 tests in src/tests/unit are deselected for exactly this reason, among them
#: ``test_logger_frame_resolution.py``, which the roadmap names as P2.1's only detector.
#: Without this line, the guard below would be a guard in name only.
pytestmark = pytest.mark.unit

LOGGER_SOURCE = Path(__file__).resolve().parents[2] / "log_config" / "logger" / "logger.py"

#: (name, numeric value, the public method that emits at that level)
LEVELS = (("TRACE", 1, "trace"), ("VERBOSE", 5, "verbose"), ("DEBUG", 10, "debug"), ("INFO", 20, "info"))


def _load_private_logger():
    """Import logger.py under a private module name, escaping conftest's class-level stub.

    The fixture patches ``_log_at_level`` on the imported class OBJECT. A fresh module load
    creates a new class object, so the real emit path is reachable here and nowhere else in the
    suite.
    """
    spec = importlib.util.spec_from_file_location("_p11_private_logger", LOGGER_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_p11_private_logger"] = module
    spec.loader.exec_module(module)
    return module.Logger


class _RestoreLevel:
    """Put the class-wide level back, so these tests cannot leak into the rest of the session."""

    def __init__(self, logger_cls):
        self.logger_cls = logger_cls

    def __enter__(self):
        self._saved = self.logger_cls._log_level
        return self.logger_cls

    def __exit__(self, *exc):
        self.logger_cls._log_level = self._saved
        return False


class TestSetLevelMovesBothPaths(unittest.TestCase):
    """``set_level(X)`` must move the guard AND the emit filter. It used to move only the guard.

    **This class is a contract test, NOT the detector.** It hands ``_filter_by_level`` the
    threshold explicitly, which is the post-P1.1 expression, so it passes against the unpatched
    logger too -- verified: 4 of this file's tests fail on the pre-P1.1 code and these are not
    among them. ``TestEmitPathReadsLogLevel`` is the detector, because it drives the real
    ``_log_at_level`` and never restates what that method should read.
    """

    def test_guard_and_filter_agree_at_every_level(self):
        with _RestoreLevel(Logger):
            for configured, _num, _method in LEVELS:
                Logger.set_level(configured)
                for name, num, _m in LEVELS:
                    guard = Logger.isEnabledFor(level=num)
                    emit = Logger._filter_by_level(level=name, log_level=Logger._log_level)
                    self.assertEqual(
                        guard, emit,
                        f"configured={configured}: guard says {guard} for {name} but the emit "
                        f"filter says {emit}. The two configured-level states have diverged again.",
                    )

    def test_set_level_actually_changes_the_decision(self):
        """Guard against a vacuous pass: the levels must not agree merely by never changing."""
        with _RestoreLevel(Logger):
            Logger.set_level("INFO")
            quiet = Logger.isEnabledFor(level=1)
            Logger.set_level("TRACE")
            loud = Logger.isEnabledFor(level=1)
            self.assertFalse(quiet, "TRACE should be disabled at INFO")
            self.assertTrue(loud, "TRACE should be enabled at TRACE")

    def test_get_level_reflects_the_written_value(self):
        with _RestoreLevel(Logger):
            for configured, _num, _m in LEVELS:
                Logger.set_level(configured)
                self.assertEqual(Logger.get_level(), configured)


class TestEmitPathReadsLogLevel(unittest.TestCase):
    """Drive the REAL ``_log_at_level`` and observe emission -- not a reconstruction of it.

    Reconstructing the threshold expression is how an earlier version of this check stopped
    measuring the defect: written as ``_filter_by_level(level, _log_level)`` it asserts the
    PATCHED behaviour and passes against the unpatched code too.
    """

    @classmethod
    def setUpClass(cls):
        cls.logger = _load_private_logger()

    def _emits(self, method, marker):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            getattr(self.logger, method)(marker)
        return marker in buf.getvalue()

    def test_emission_follows_set_level(self):
        with _RestoreLevel(self.logger) as log:
            for configured, _num, _m in LEVELS:
                log.set_level(configured)
                for name, num, method in LEVELS:
                    emitted = self._emits(method, f"__p11_{configured}_{name}__")
                    guard = log.isEnabledFor(level=num)
                    self.assertEqual(
                        guard, emitted,
                        f"configured={configured}: isEnabledFor({name})={guard} but the real emit "
                        f"path {'emitted' if emitted else 'discarded'} the record. set_level is a "
                        f"no-op for emission again.",
                    )

    def test_the_private_module_is_not_stubbed(self):
        """If conftest ever stubs this copy too, the emit test above becomes vacuous. Say so loudly."""
        buf = io.StringIO()
        with _RestoreLevel(self.logger) as log:
            log.set_level("TRACE")
            with contextlib.redirect_stdout(buf):
                log.trace("__p11_liveness__")
        self.assertIn(
            "__p11_liveness__", buf.getvalue(),
            "The privately-imported logger emitted nothing at TRACE. The emit path is stubbed or "
            "broken, so TestEmitPathReadsLogLevel proves nothing.",
        )


class TestRetiredLevelStateIsInert(unittest.TestCase):
    """``_level_logger_name`` / ``_level_logger_config`` are no longer level state."""

    def test_mutating_the_retired_state_changes_nothing(self):
        logger = _load_private_logger()
        with _RestoreLevel(logger) as log:
            log.set_level("INFO")
            before = [(n, log._filter_by_level(level=n, log_level=log._log_level)) for n, _u, _m in LEVELS]

            saved_name, saved_config = log._level_logger_name, log._level_logger_config
            try:
                log._level_logger_name = "TRACE"
                log._level_logger_config = 1
                after = [(n, log._filter_by_level(level=n, log_level=log._log_level)) for n, _u, _m in LEVELS]
                self.assertEqual(
                    before, after,
                    "Mutating _level_logger_name / _level_logger_config moved an emission decision. "
                    "They have been wired back into the filter -- that is the two-state split P1.1 "
                    "removed.",
                )

                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    log.trace("__p11_retired__")
                self.assertNotIn(
                    "__p11_retired__", buf.getvalue(),
                    "A TRACE record emitted while _log_level is INFO, because the retired state was "
                    "consulted. The emit path must read _log_level and nothing else.",
                )
            finally:
                log._level_logger_name, log._level_logger_config = saved_name, saved_config


class TestIsValidLevel(unittest.TestCase):
    """``is_valid_level`` returned True for EVERY input -- ``_is_valid_level_number(level == level)``."""

    def test_rejects_invalid(self):
        for bad in ("BANANA", None, "", 8, 999, -1, object()):
            self.assertFalse(
                Logger.is_valid_level(bad),
                f"is_valid_level({bad!r}) should be False; the repo's only validity predicate "
                f"cannot say no, and P4-G4 (fail loudly on an unknown level) needs it to.",
            )

    def test_accepts_valid(self):
        for name, num, _m in LEVELS:
            self.assertTrue(Logger.is_valid_level(name), f"is_valid_level({name!r}) should be True")
            self.assertTrue(Logger.is_valid_level(num), f"is_valid_level({num!r}) should be True")

    def test_resolution_is_unchanged_for_the_odd_inputs(self):
        """The fix must not change what ``_resolve_level_number`` returns.

        Before the fix ``is_valid_level`` was always True, so the pair fell through to
        ``getLevelNumber``, which returned None for an invalid level anyway. The corrected
        predicate short-circuits instead -- same answer, fewer steps.
        """
        for bad in ("BANANA", None, 8, 999):
            self.assertIsNone(
                Logger._resolve_level_number(bad),
                f"_resolve_level_number({bad!r}) must still be None",
            )
        for name, num, _m in LEVELS:
            self.assertEqual(Logger._resolve_level_number(name), num)


if __name__ == "__main__":
    unittest.main()
