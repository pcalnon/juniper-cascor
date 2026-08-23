"""Pin the logger's frame resolution: same values as before, without the stack walk.

Project:     juniper-cascor
Sub-Project: logging
Author:      Paul Calnon
License:     MIT License

WHY THIS FILE EXISTS
``Logger._frame_info`` used to be::

    return lambda name: getattr(getouterframes(frame)[1], name)

which is the single most expensive line in the trainer. ``getouterframes`` builds a ``FrameInfo``
for every frame on the stack -- each one calling ``inspect.getmodule``, whose cache-miss path
copies and scans the whole of ``sys.modules`` -- and then this used index ``[1]`` and threw the
rest away. The returned closure re-walked the stack on every field access, and a single log record
reads five fields across ``_console_dict`` and ``_file_dict``.

Measured at a realistic stack depth with ~1,300 modules loaded: **20,711 us** per resolution
against **1.0 us** for direct frame attributes, and native profiling attributed ~78% of
candidate-worker CPU to those ``inspect`` frames.

Two things therefore need pinning, and they are different in kind:

* an EQUIVALENCE test -- the cheap path must return exactly what the expensive path returned, or
  every log line in the system silently changes; and
* a REGRESSION GUARD -- ``getouterframes`` / ``inspect.stack`` must not come back. That is the one
  that matters in a year: the fix is a one-line change and so is undoing it, and the symptom is
  "training got slower", which nobody attributes to a logger.
"""

from __future__ import annotations

import os
import re
import unittest
from inspect import currentframe, getouterframes
from pathlib import Path

from log_config.logger.logger import Logger

LOGGER_SOURCE = Path(__file__).resolve().parents[2] / "log_config" / "logger" / "logger.py"


def _legacy_frame_info(frame):
    """Exactly the pre-fix implementation, kept here as the oracle."""
    return lambda name: getattr(getouterframes(frame)[1], name)


def _read(getter):
    return (
        os.path.basename(getter(Logger._frame_file)),
        getter(Logger._frame_line),
        getter(Logger._frame_func),
    )


class TestFrameResolutionEquivalence(unittest.TestCase):
    """The cheap path must agree with the expensive one it replaced."""

    def _probe(self, depth):
        """Resolve ONE frame with both implementations.

        Both getters must be applied to the SAME frame. Calling them from two different helper
        functions compares two different call sites and reports a mismatch that is an artefact of
        the test rather than of the code -- which is exactly what a first draft of this did.
        """
        if depth:
            return self._probe(depth - 1)
        frame = currentframe()
        return _read(_legacy_frame_info(frame)), _read(Logger._frame_info(frame=frame))

    def test_matches_legacy_at_several_depths(self):
        for depth in (0, 1, 5, 12):
            with self.subTest(depth=depth):
                legacy, current = self._probe(depth)
                self.assertEqual(legacy, current)

    def test_reports_the_caller_not_the_logger(self):
        """The value must describe the frame that logged, not the logging machinery.

        This models the real contract rather than calling `_frame_info` directly. `Logger.info`
        passes `currentframe()` -- ITS OWN frame -- and `_frame_info` reports `f_back`, i.e. whoever
        called `Logger.info`. So the stand-in below must also be a function whose caller is the code
        that should be named. Handing it this test method's own frame instead reports unittest's
        `_callTestMethod`, which is right behaviour and a wrong test.
        """

        def stands_in_for_logger_info():
            return Logger._frame_info(frame=currentframe())

        getter = stands_in_for_logger_info()
        self.assertEqual(getter(Logger._frame_func), "test_reports_the_caller_not_the_logger")
        self.assertEqual(os.path.basename(getter(Logger._frame_file)), os.path.basename(__file__))


class TestFrameResolutionEdgeCases(unittest.TestCase):
    def test_no_caller_frame_renders_placeholder_rather_than_raising(self):
        """A depth-1 stack used to raise IndexError from inside a logging call.

        A log line must never be the thing that fails a training run, so the missing frame renders
        a placeholder instead. This is a deliberate behaviour change, not an accident.
        """
        getter = Logger._frame_info(frame=None)
        self.assertEqual(getter(Logger._frame_file), Logger._frame_unknown)
        self.assertEqual(getter(Logger._frame_line), Logger._frame_unknown)

    def test_unknown_field_name_renders_placeholder(self):
        getter = Logger._frame_info(frame=currentframe())
        self.assertEqual(getter("not_a_frame_field"), Logger._frame_unknown)

    def test_console_and_file_dicts_still_build(self):
        """The two real consumers must keep working, since they read the fields by name."""
        import datetime

        frame = currentframe()
        tsp = datetime.datetime.now()
        console = Logger._console_dict(frame=frame, tsp=tsp, level="INFO", message="probe")
        file_d = Logger._file_dict(frame=frame, tsp=tsp, level="INFO", message="probe")
        self.assertIsInstance(console, dict)
        self.assertIsInstance(file_d, dict)
        self.assertNotIn(Logger._frame_unknown, console.values())
        self.assertNotIn(Logger._frame_unknown, file_d.values())


class TestNoStackWalkRegression(unittest.TestCase):
    """The guard that actually matters in a year.

    Re-introducing ``getouterframes`` or ``inspect.stack`` restores an O(len(sys.modules)) scan on
    every log record. The symptom is "training got slower", which nobody traces back to a logger,
    so the protection has to be a source assertion rather than a benchmark.
    """

    def test_logger_does_not_walk_the_stack(self):
        source = LOGGER_SOURCE.read_text(encoding="utf-8")
        # Strip comments and docstring prose: this file's own rationale names the banned calls.
        code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
        code = re.sub(r'""".*?"""', "", code, flags=re.DOTALL)
        for banned in ("getouterframes", "inspect.stack", "getframeinfo"):
            with self.subTest(banned=banned):
                self.assertNotIn(
                    banned,
                    code,
                    msg=(f"{banned} is back in logger.py. It walks the whole stack and calls " "inspect.getmodule per frame, which copies and scans sys.modules -- " "~20,700x slower per resolution and ~78% of candidate-worker CPU when " "this was last measured. Use frame.f_back attributes instead."),
                )


if __name__ == "__main__":
    unittest.main()
