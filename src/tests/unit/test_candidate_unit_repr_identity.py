#!/usr/bin/env python
"""
Unit tests for CandidateUnit log identity (juniper-cascor#532).

`CascadeCorrelationNetwork._add_best_candidate` interpolates the winning CandidateUnit
directly into its "Adding best candidate {best_candidate}" record. With no __repr__ on
the class, that renders as a bare `<candidate_unit.candidate_unit.CandidateUnit object
at 0x7f...>` -- an address that names nothing and differs on every run. WHICH candidate
was installed is the single fact that separates "a near-tie flipped" from "the
arithmetic jittered", and it was unrecoverable from any shipped log; the #532 campaign
had to run a patched build to get it.

Tests focus on:
- repr() carries the unit's identity, not a memory address (the regression guard)
- the f-string interpolation path -- the one _add_best_candidate actually uses
- repr() never raises on a part-formed instance (it runs on a logging path, and
  instances arrive part-formed via __setstate__ on the forked-worker path)
- correlation is rendered at full precision, not rounded to 6 dp
"""

import os
import re
import sys

import pytest

# Add parent directories for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from candidate_unit.candidate_unit import CandidateUnit

# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit

# The rendering this guard exists to keep out of the logs: CPython's default object repr.
DEFAULT_OBJECT_REPR = re.compile(r"^<.*CandidateUnit object at 0x[0-9a-fA-F]+>$")


class TestCandidateUnitReprIdentity:
    """CandidateUnit must identify itself in logs."""

    @pytest.fixture
    def candidate(self):
        """A real, minimally-constructed candidate unit."""
        return CandidateUnit(CandidateUnit__input_size=2, CandidateUnit__random_seed=42)

    def test_repr_is_not_a_memory_address(self, candidate):
        """The regression guard: repr must not fall back to CPython's default."""
        rendered = repr(candidate)
        assert not DEFAULT_OBJECT_REPR.match(rendered), f"CandidateUnit repr fell back to a memory address: {rendered}"
        assert "object at 0x" not in rendered, f"CandidateUnit repr leaks an address: {rendered}"

    def test_repr_carries_identity_fields(self, candidate):
        """repr names the class and the fields that identify which candidate this is."""
        candidate.candidate_index = 7
        candidate.correlation = 0.5
        rendered = repr(candidate)

        assert rendered.startswith("CandidateUnit("), rendered
        assert "candidate_index=7" in rendered, rendered
        assert "correlation=0.5" in rendered, rendered
        assert "uuid=" in rendered, rendered

    def test_interpolation_path_uses_repr(self, candidate):
        """_add_best_candidate interpolates the unit itself -- pin that exact path.

        An f-string goes __format__ -> __str__ -> __repr__, so the fix has to reach the
        log line without _add_best_candidate naming any attribute explicitly.
        """
        candidate.candidate_index = 3
        rendered = f"Adding best candidate {candidate} at iteration 0"

        assert "candidate_index=3" in rendered, rendered
        assert "object at 0x" not in rendered, rendered

    def test_repr_does_not_raise_on_part_formed_instance(self):
        """repr runs on a logging path, so it must never raise.

        Instances reach __repr__ part-formed on the forked-worker path (__setstate__
        rebuilds __dict__), and a __repr__ that raises breaks the very log record that
        called it -- turning an observability gap into a crash.
        """
        bare = object.__new__(CandidateUnit)  # __init__ deliberately not run

        rendered = repr(bare)  # must not raise

        assert rendered.startswith("CandidateUnit("), rendered
        assert "candidate_index=None" in rendered, rendered

    def test_correlation_is_not_rounded(self, candidate):
        """Full precision, because the near-tie question lives below 6 dp.

        The adjacent "Final Correlation" record formats with :.6f, and in the cap-4 cell
        the top two round-0 correlations are 0.091185 / 0.091184 -- adjacent at that
        precision. A repr that rounded the same way would answer nothing.
        """
        candidate.correlation = 0.09118512345
        rendered = repr(candidate)

        assert "0.09118512345" in rendered, f"correlation was rounded away: {rendered}"

    def test_repr_does_not_format_tensors(self, candidate):
        """O(1) by construction -- CR-062 warns against expensive reprs on hot paths."""
        rendered = repr(candidate)

        assert "tensor(" not in rendered, rendered
        assert len(rendered) < 500, f"repr is unexpectedly large ({len(rendered)} chars): {rendered}"
