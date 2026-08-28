#!/usr/bin/env python
"""F-CANOPY-026: ``phase_started_at`` must be tz-aware UTC, never naive local.

Ledger: juniper-ml notes/JUNIPER_2026-08-09_JUNIPER-CANOPY_E2E-VALIDATION-EVIDENCE.md

The lifecycle manager stamped both phase transitions with ``datetime.now().isoformat()``
— **naive local time**. canopy's phase-duration readout
(``metrics_panel._update_phase_duration_handler``) treats a naive value as UTC
(``started.replace(tzinfo=timezone.utc)``) and subtracts it from ``now(timezone.utc)``,
so the displayed elapsed time is inflated by exactly the host's UTC offset. Measured
live: **"Phase Duration: 300m 37s" on a run 37 seconds old** — a CDT box, delta exactly
18000 s. The counter ticked correctly at 1 s/s, so it was a pure constant offset.

**Why fourteen segments of dashboard testing never surfaced it, and why this test is
written the way it is:** the bug is *invisible* wherever the host is already UTC-0 —
CI runners and most containers. A test that merely round-trips the timestamp on a CI
box would pass against the naive version too. So these tests force a non-UTC local zone
via ``TZ`` and assert on the property that actually distinguishes the two: the emitted
string carries an offset, and parsing it yields an instant that is *actually now*.

**Which test is the regression pin.** Driving a real phase transition needs a full
training run, so the two ``TestPhaseStartedAtIsAwareUtc`` tests construct the timestamp
the way each version of the call site does; they *document and keep exercising the
mechanism* but do not import the fix, and they pass on the parent. The discriminating
test is ``test_every_phase_started_at_passes_a_timezone``, which reads the real call
sites out of ``manager.py``'s AST — **that one fails on the parent commit (d68e654)**,
and it is the one that stops the naive form coming back.
"""

import ast
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

_MANAGER = Path(__file__).resolve().parents[3] / "api" / "lifecycle" / "manager.py"


@pytest.fixture
def cdt_timezone():
    """Run the body on a box whose local time is NOT UTC (the observed condition)."""
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "America/Chicago"
    time.tzset()
    yield
    if previous is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = previous
    time.tzset()


@pytest.mark.unit
class TestPhaseStartedAtIsAwareUtc:
    def test_emitted_stamp_is_aware_and_is_actually_now(self, cdt_timezone):
        """The emitted value must parse to an instant within seconds of the real now.

        This is the assertion the naive version fails: ``datetime.now().isoformat()``
        on a CDT box parses to a naive value that, once stamped UTC by the consumer,
        sits 5 hours in the past.
        """
        emitted = datetime.now(UTC).isoformat()

        parsed = datetime.fromisoformat(emitted)
        assert parsed.tzinfo is not None, "phase_started_at is naive; the consumer will stamp it UTC and inflate the phase duration by the host's UTC offset"

        skew = abs((datetime.now(UTC) - parsed).total_seconds())
        assert skew < 60, f"phase_started_at resolves {skew:.0f}s away from now — that is the F-CANOPY-026 offset inflation"

    def test_a_naive_stamp_would_be_wrong_by_the_host_offset(self, cdt_timezone):
        """Pins the mechanism itself, so the test cannot quietly stop discriminating.

        If this ever stops failing, the environment is UTC-0 and the sibling test above
        has become vacuous there — which is exactly how this bug survived fourteen
        segments of dashboard testing.
        """
        naive = datetime.now().isoformat()
        as_consumer_reads_it = datetime.fromisoformat(naive).replace(tzinfo=UTC)
        skew = abs((datetime.now(UTC) - as_consumer_reads_it).total_seconds())
        assert skew > 3600, f"expected a naive local stamp to be >=1h off under TZ=America/Chicago, got {skew:.0f}s — the fixture did not take effect"


@pytest.mark.unit
class TestNoNaiveStampsOnThePhaseTransitions:
    """Source pin: neither transition may go back to a naive ``datetime.now()``.

    Exercising the real transitions needs a full training run, so the invariant is
    pinned at the call site instead — the same place the defect lived.
    """

    @staticmethod
    def _phase_started_at_calls():
        tree = ast.parse(_MANAGER.read_text(encoding="utf-8"))
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg == "phase_started_at":
                    found.append(kw.value)
        return found

    def test_both_transitions_are_still_present(self):
        """Guard against the pin going vacuous if the call sites are renamed away."""
        calls = self._phase_started_at_calls()
        assert len(calls) == 2, f"expected 2 phase_started_at call sites (Candidate + Output), found {len(calls)}"

    def test_every_phase_started_at_passes_a_timezone(self):
        for value in self._phase_started_at_calls():
            src = ast.unparse(value)
            assert "datetime.now()" not in src, f"naive local timestamp back on a phase transition: {src} — F-CANOPY-026"
            assert "UTC" in src or "timezone.utc" in src, f"phase_started_at is not tz-aware UTC: {src}"
