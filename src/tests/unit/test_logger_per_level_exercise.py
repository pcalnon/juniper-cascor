"""Exercise the converted candidate_unit call sites once per level, and pin what each emits.

Project:     juniper-cascor
Sub-Project: logging
Author:      Paul Calnon
License:     MIT License

WHY THIS FILE EXISTS
P1.5 of cascor#573. The logger's own suites test the logger; nothing ran ``candidate_unit.py``'s
guard sites, which is where the level system is actually consumed. Two separate stubs in
``src/tests/conftest.py`` guarantee that:

* ``CandidateUnit.__init__`` and ``__setstate__`` are patched to set ``self.logger = _noop_logger``
  (``conftest.py:893-913``), and ``_NoOpLogger.isEnabledFor`` returns ``level >= 30``. With
  ``Logger.DEBUG`` 10, ``VERBOSE`` 5 and ``TRACE`` 1, **all eight guards are False for the entire
  suite** -- every guarded block is unreachable, in every test.
* ``Logger._log_at_level`` is replaced by a no-op (``conftest.py:921-927``), so even a reachable
  block would emit nothing.

So "the suite is green" has never said anything about these sites. This file is the first thing to
execute them. It defeats both stubs the same way
``test_logger_level_state_reconciliation.py`` does -- by importing a PRIVATE copy of the logger
module, which the fixture has not patched -- and binds that copy onto the unit under test.

WHAT IT PINS
For each of TRACE / VERBOSE / DEBUG / INFO, with that level configured:

1. the converted methods run without raising (P1.5's "no exception");
2. **no record below the configured level is emitted** -- the leak direction, which would mean a
   guard opened a block whose call then emitted when it should not have; and
3. at TRACE, a TRACE record *does* appear -- without which (1) and (2) would be satisfied by a
   logger that simply emits nothing, which is exactly the vacuous pass this arc keeps finding.

WHY IT DEPENDS ON P1.1
Until P1.1 made ``set_level`` drive the emit filter, there was no supported runtime way to move the
emission level at all: the filter read state ``set_level`` never wrote. This file could not have
been written as specified before it.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import sys
import unittest
from pathlib import Path

import pytest
import torch

from candidate_unit.candidate_unit import CandidateUnit

#: Required: CI's unit lane selects by marker (ci.yml:312-317); an unmarked file here is collected
#: and then silently deselected.
pytestmark = pytest.mark.unit

LOGGER_SOURCE = Path(__file__).resolve().parents[2] / "log_config" / "logger" / "logger.py"

#: configured level -> the levels that may legitimately appear in output
LEVELS = ("TRACE", "VERBOSE", "DEBUG", "INFO")
NUMBERS = {"TRACE": 1, "VERBOSE": 5, "DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50, "FATAL": 60}

RECORD_LEVEL = re.compile(r"\[(TRACE|VERBOSE|DEBUG|INFO|WARNING|ERROR|CRITICAL|FATAL)\]")


def _private_logger(tmp_log):
    """Load logger.py under a private name, escaping conftest's class-level stubs.

    The fixture patches ``_log_at_level`` on the imported class OBJECT; a fresh module load makes a
    new class, so the real emit path is reachable here and nowhere else in the suite.
    """
    spec = importlib.util.spec_from_file_location("_p15_private_logger", LOGGER_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_p15_private_logger"] = module
    spec.loader.exec_module(module)
    logger = module.Logger
    # Never write into the shared repo log file: two cascor writers on one path interleave and
    # rotate away each other's records (constants.py:422-427, roadmap trap 1).
    logger._logging_file = str(tmp_log)
    return logger


class PerLevelExercise(unittest.TestCase):
    """Run the converted call sites at each level and pin what comes out."""

    def setUp(self):
        self.tmp_log = Path(__file__).resolve().parent / "_p15_scratch.log"
        self.logger = _private_logger(self.tmp_log)
        self.addCleanup(lambda: self.tmp_log.exists() and self.tmp_log.unlink())

        self.unit = CandidateUnit(
            CandidateUnit__input_size=2,
            CandidateUnit__epochs=1,
            CandidateUnit__learning_rate=0.01,
            CandidateUnit__random_seed=42,
            CandidateUnit__candidate_index=0,
        )
        # conftest's patched __init__ has just set this to _noop_logger, whose isEnabledFor is
        # `level >= 30` -- which makes all eight guards False. Bind the real (private) logger so
        # the guarded blocks are reachable at all.
        self.unit.logger = self.logger

    def _run_converted(self):
        """Drive the two methods holding 5 of the 8 guards, across all three guard levels."""
        residual = torch.tensor([[0.5], [-0.25], [0.75], [-0.1]], dtype=torch.float32)
        output = torch.tensor([[0.2], [0.4], [-0.3], [0.1]], dtype=torch.float32)
        self.unit._multi_output_correlation(residual_error=residual, output=output)
        self.unit._get_correlations(output=output, residual_error=residual)

    def _emit_at(self, configured):
        buf = io.StringIO()
        self.logger.set_level(configured)
        with contextlib.redirect_stdout(buf):
            self._run_converted()
        return buf.getvalue()

    def test_runs_without_raising_at_every_level(self):
        for configured in LEVELS:
            with self.subTest(configured=configured):
                try:
                    self._emit_at(configured)
                except Exception as exc:  # noqa: BLE001 -- the assertion IS "nothing raises"
                    self.fail(f"configured={configured}: converted call sites raised {exc!r}")

    def test_no_record_below_the_configured_level_is_emitted(self):
        for configured in LEVELS:
            with self.subTest(configured=configured):
                out = self._emit_at(configured)
                floor = NUMBERS[configured]
                leaked = sorted({lv for lv in RECORD_LEVEL.findall(out) if NUMBERS[lv] < floor})
                self.assertEqual(
                    leaked,
                    [],
                    f"configured={configured}: records emitted BELOW the configured level: {leaked}. " f"A guard opened a block whose call then emitted when the filter should have " f"discarded it -- the guard and the emit filter have diverged again.",
                )

    def test_trace_actually_emits_at_trace(self):
        """Anti-vacuous: without this, a logger that emits nothing satisfies the other two."""
        out = self._emit_at("TRACE")
        seen = set(RECORD_LEVEL.findall(out))
        self.assertIn(
            "TRACE",
            seen,
            "configured=TRACE emitted no TRACE record. Either the guard is closed when it should be " "open, or the emit path is stubbed -- in which case the other tests in this file prove " f"nothing. Levels seen: {sorted(seen)}",
        )

    def test_raising_the_level_strictly_narrows_what_is_emitted(self):
        """TRACE must be a superset of INFO: a monotonicity check the set comparisons alone miss."""
        at_trace = set(RECORD_LEVEL.findall(self._emit_at("TRACE")))
        at_info = set(RECORD_LEVEL.findall(self._emit_at("INFO")))
        self.assertTrue(
            at_info <= at_trace,
            f"raising the configured level ADDED levels: at INFO {sorted(at_info)} is not a subset " f"of at TRACE {sorted(at_trace)}",
        )
        self.assertNotEqual(
            at_trace,
            at_info,
            "TRACE and INFO emitted the same set of levels, so the configured level is not moving " "the emit decision at all -- the P1.1 defect has returned.",
        )


if __name__ == "__main__":
    unittest.main()
