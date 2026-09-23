#!/usr/bin/env python
"""
Project:       Juniper
Sub-Project:   JuniperCascor
Application:   juniper_cascor
File Name:     test_candidate_epochs_completed_advisory.py
File Path:     src/tests/unit/

Author:        Paul Calnon

Date Created:  2026-09-23
Last Modified: 2026-09-23

License:       MIT License
Copyright:     Copyright (c) 2024-2026 Paul Calnon

Description:
    D6 ADVISORY: an exact-match check on the candidate micro-benchmark's ``epochs_completed`` that
    WARNS on a mismatch and never fails the build. Owner decision D6, ruled 2026-09-22 (juniper-ml
    ``notes/JUNIPER_2026-09-11_JUNIPER-ECOSYSTEM_PERF-LANE-SIX-OWNER-DECISIONS-RULED.md`` §5.4):
    "Build it non-blocking, count how often it fires on real cascor PRs, then decide whether it
    blocks."

    WHAT IT CHECKS
    The construction of
    ``src/tests/performance/test_micro_candidate.py::TestCandidateEpochScaling::test_epoch_scaling``,
    copied VERBATIM below, at budgets 100 and 200 only. At 10 and 50 the count equals the request
    (those cells are budget-bound), so a check there would assert a tautology. The measurement
    behind the reference is juniper-ml
    ``notes/JUNIPER_2026-09-22_JUNIPER-ECOSYSTEM_PERF-LANE-D6-EPOCHS-COMPLETED-SPREAD.md``: 10 / 50 /
    68 / 68 at budgets 10 / 50 / 100 / 200, with zero spread over 100 observations across torch
    thread widths 1-16, and the same values at 1-minute host loads of 19.50, 32.26 and 54.49.

    WHY IT IS A UNIT TEST
    ``performance``-marked tests never run in CI. The unit lane runs
    ``pytest -m "unit and not slow" src/tests/unit`` (``.github/workflows/ci.yml``, step "Run Unit
    Tests"), so this is the only tier in which a firing can be counted at all: the same check in
    the performance file would count zero by construction. The module-level ``pytestmark`` is what
    selects it -- a test here without the ``unit`` marker is collected and then silently deselected.

    ADVISORY SEMANTICS
    A mismatch with ``EPOCHS_COMPLETED_REFERENCE`` never fails the build. It is reported three ways:

      1. an ``EpochsCompletedDriftWarning`` in pytest's warnings summary, issued under a local
         ``"always"`` filter, so a later ``-W error`` or ``filterwarnings = ["error"]`` cannot turn
         the advisory into a gate by accident;
      2. one markdown line appended to ``$GITHUB_STEP_SUMMARY``, when that is set;
      3. a ``::warning title=D6 advisory - epochs_completed drift::...`` workflow command, when
         ``GITHUB_ACTIONS == "true"``: a check-run annotation, which is the thing to count.

    Genuine errors still FAIL: an exception from training, a count that is not an ``int``, or a
    count outside ``1..budget``. Only equality with the reference is advisory.

    THE ANNOTATION MUST ESCAPE PYTEST'S CAPTURE, AND WRITING TO ``sys.__stdout__`` DOES NOT DO THAT
    Under pytest's default ``--capture=fd``, file descriptor 1 is redirected to a temporary file
    while each test runs, and ``sys.__stdout__`` writes to fd 1: a flushed write is captured and,
    for a passing test, discarded. So the emitter suspends capture for the write (the real check
    passes ``capsys.disabled``), flushes inside the suspension, and writes a newline first. The
    runner reads a workflow command only from a line that STARTS with ``::``, and under
    ``--verbose`` pytest has already written ``<nodeid> `` with no newline. Measured 2026-09-23 with
    pytest 9.0.3 in ``--verbose`` mode: a flushed ``sys.__stdout__`` write vanished; an unflushed
    one, one under ``capsys.disabled()`` and one under the capture manager each landed mid-line
    after the node id; only suspension plus a leading newline gave a line that starts
    ``::warning``. (Under pytest-xdist a worker's stdout is not relayed at all; CI's unit lane does
    not use ``-n``.)

    HOW TO COUNT FIRINGS
    Each leg of the ``Unit Tests + Coverage (Python <version> on <os>)`` matrix is its own check
    run (four legs on 2026-09-23), so one cause can fire up to two annotations on every leg. Count
    per PR head SHA, and record which legs fired::

        gh api "repos/pcalnon/juniper-cascor/commits/<sha>/check-runs?per_page=100" \\
            --jq '.check_runs[] | select(.name | startswith("Unit Tests")) | "\\(.id) \\(.name)"'
        gh api "repos/pcalnon/juniper-cascor/check-runs/<check-run-id>/annotations" \\
            --jq '.[] | select(.title == "D6 advisory - epochs_completed drift") | .message'

    Or search a run's log. The runner rewrites the command as ``##[warning]<message>`` and drops the
    title, which is why every message begins with the title string::

        gh run view <run-id> --log | grep -F '##[warning]D6 advisory - epochs_completed drift'

    WHAT A FIRING DOES AND DOES NOT MEAN
    The count is a function of (code, seed, budget), measured invariant to host load and thread
    width on one host. It is NOT established across platforms or torch builds: the reference was
    taken on Linux x86_64 with a CUDA build of torch, while CI installs CPU-only wheels on Linux and
    native wheels on macOS. A firing on some legs and not on others is a platform divergence until
    shown otherwise; the message names the platform so that the two can be told apart. Do not call
    the count "deterministic": invariance to things nobody tested has not been measured.

    WHEN IT FIRES ON AN INTENDED CHANGE
    Re-measure on the new tree under the same pin (``observe_epochs_completed``), then update
    ``EPOCHS_COMPLETED_REFERENCE`` and its provenance (``REFERENCE_CASCOR_SHA``, ``REFERENCE_DATE``,
    ``REFERENCE_PLATFORM``) together, in the same change.
"""

import contextlib
import io
import os
import platform
import sys
import types
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch

from candidate_unit.candidate_unit import CandidateTrainingResult, CandidateUnit  # noqa: E402

# CI's unit lane runs ``pytest -m "unit and not slow" src/tests/unit``, so a test here with no
# ``unit`` marker is collected and then silently DESELECTED -- it never runs and the job still
# reports success. Module-level, so it covers every test in the file, including ones added later.
pytestmark = pytest.mark.unit

# ===================================================================
# THE REFERENCE
# ===================================================================

#: Budget -> expected ``epochs_completed`` for the construction below. Budgets 10 and 50 are left
#: out on purpose: there the count equals the request, and a check would assert a tautology.
EPOCHS_COMPLETED_REFERENCE = {100: 68, 200: 68}

# Provenance. The ruling requires the cascor SHA and the thread pin: without them an intended
# numeric change reads as a flake, which is how work gates get switched off. Verified by running
# ``observe_epochs_completed`` at budgets 10 / 50 / 100 / 200 on this tree -> 10 / 50 / 68 / 68.
REFERENCE_CASCOR_SHA = "6276c459fd83c1a33d8b14535243b15becdfe45f"  # juniper-cascor origin/main
REFERENCE_DATE = "2026-09-23"
REFERENCE_THREAD_PIN = 1  # == BENCHMARK_THREAD_PIN in src/tests/performance/timing_reference.py
REFERENCE_PLATFORM = "Linux x86_64, Python 3.14.7, torch 2.11.0+cu130"

#: The annotation title, and the first words of every message (the job log drops the title).
ANNOTATION_TITLE = "D6 advisory - epochs_completed drift"

_THIS_FILE = Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[3]).as_posix()


class EpochsCompletedDriftWarning(UserWarning):
    """``epochs_completed`` differs from ``EPOCHS_COMPLETED_REFERENCE``. Advisory: never an error."""


# ===================================================================
# THE CONSTRUCTION -- VERBATIM from src/tests/performance/test_micro_candidate.py
# ===================================================================
# ``_make_candidate`` and ``_make_data`` are that file's helpers, and the ``run()`` closure is
# ``TestCandidateEpochScaling.test_epoch_scaling``'s. If that test changes, change this copy with it
# and re-verify the reference: the two are compared by eye, because the performance tier never runs
# in CI and so cannot keep this copy honest.


def _make_candidate(input_size, activation_fn):
    """Create a CandidateUnit with correct constructor parameters."""
    return CandidateUnit(
        CandidateUnit__input_size=input_size,
        CandidateUnit__activation_function=activation_fn,
    )


def _make_data(input_size, n_samples, output_size=2):
    """Create synthetic x and residual_error tensors."""
    torch.manual_seed(42)
    x = torch.randn(n_samples, input_size)
    residual = torch.randn(n_samples, output_size)
    return x, residual


@contextlib.contextmanager
def _torch_thread_pin(width):
    """Run the block under ``torch.set_num_threads(width)``, then restore the prior width.

    ``torch.get_num_threads()`` is NOT a passive read: it runs torch's per-thread lazy init and
    re-pins the CALLING thread's OpenMP width to torch's global value (juniper-ml D6 note §2.2).
    That is acceptable here -- a save/restore on the pytest thread around a unit test, not a
    measurement of thread width, and ``set_num_threads`` re-pins the same thread anyway -- but it
    would destroy an instrument that measures width.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(width)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def observe_epochs_completed(epochs):
    """``epochs_completed`` of ``test_epoch_scaling``'s construction at budget ``epochs``, under the reference pin.

    The whole body runs under the pin, as the performance tier's autouse fixture pins the whole
    test. That test runs ``run()`` through ``benchmark.pedantic(rounds=3, warmup_rounds=1)`` and
    keeps the last result; every call re-seeds and builds a fresh candidate, and the D6 measurement
    found zero spread across repeats, so a single call reproduces it.
    """
    with _torch_thread_pin(REFERENCE_THREAD_PIN):
        x, residual = _make_data(input_size=2, n_samples=100)

        def run():
            torch.manual_seed(42)
            c = _make_candidate(2, torch.nn.Tanh())
            return c.train_detailed(x=x, epochs=epochs, residual_error=residual, learning_rate=0.005, display_frequency=0)

        result = run()
    return result.epochs_completed


# ===================================================================
# THE ADVISORY
# ===================================================================


def drift_message(budget, expected, observed):
    """The one-line message used by the warning, the step-summary line and the annotation."""
    reference = f"Reference: cascor {REFERENCE_CASCOR_SHA[:7]}, {REFERENCE_DATE}, {REFERENCE_PLATFORM}."
    this_run = f"This run: {platform.system()} {platform.machine()}, Python {platform.python_version()}, torch {torch.__version__}"
    instruction = f"advisory only (owner decision D6, 2026-09-22); if a change to candidate numerics is intended, update the reference in {_THIS_FILE}"
    return f"{ANNOTATION_TITLE} at budget {budget}: expected {expected}, observed {observed} (torch thread pin {REFERENCE_THREAD_PIN}). {reference} {this_run} -- {instruction}"


def _escape_workflow_data(value):
    """Escape a workflow-command message, as the Actions runner expects."""
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_workflow_property(value):
    """Escape a workflow-command property (``title=``), which additionally reserves ``:`` and ``,``."""
    return _escape_workflow_data(value).replace(":", "%3A").replace(",", "%2C")


def _report_drift(message, *, suspend_capture):
    """Report a reference mismatch three ways (module docstring). Never raises because of the mismatch."""
    with warnings.catch_warnings():
        # Shown, never raised, whatever filters are installed around it: advisory by construction.
        warnings.simplefilter("always", EpochsCompletedDriftWarning)
        warnings.warn(message, EpochsCompletedDriftWarning, stacklevel=3)
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path:
            try:
                # Append: both budgets, and anything else in this step, write to the same file.
                with open(summary_path, "a", encoding="utf-8") as handle:
                    handle.write(f"- {message}\n")
            except OSError as exc:
                warnings.warn(f"{ANNOTATION_TITLE}: could not append to GITHUB_STEP_SUMMARY ({exc})", EpochsCompletedDriftWarning, stacklevel=3)
    if os.environ.get("GITHUB_ACTIONS") == "true" and sys.__stdout__ is not None:
        with suspend_capture():
            sys.__stdout__.write(f"\n::warning title={_escape_workflow_property(ANNOTATION_TITLE)}::{_escape_workflow_data(message)}\n")
            sys.__stdout__.flush()


def check_epochs_completed(budget, observed, *, suspend_capture):
    """Fail on a genuine error; REPORT -- never raise on -- a mismatch with the reference.

    ``suspend_capture`` is a zero-argument callable returning a context manager inside which
    pytest's output capture is suspended: ``capsys.disabled`` in the real check. It has no default,
    so no caller can silently drop the annotation into pytest's capture buffer.

    Returns the advisory message on a mismatch, and ``None`` on a match.
    """
    # Genuine errors -- NOT advisory. The type check comes first: ``68.0 == 68``, so an equality
    # test alone would accept a float as a match.
    assert isinstance(observed, int) and not isinstance(observed, bool), f"epochs_completed must be an int, got {type(observed).__name__} {observed!r} at budget {budget}"
    assert 1 <= observed <= budget, f"epochs_completed={observed} is outside 1..{budget}"
    expected = EPOCHS_COMPLETED_REFERENCE[budget]
    if observed == expected:
        return None
    message = drift_message(budget, expected, observed)
    _report_drift(message, suspend_capture=suspend_capture)
    return message


# ===================================================================
# THE CHECK -- runs in CI's unit lane
# ===================================================================


class TestEpochsCompletedAdvisory:
    """The D6 advisory itself: warns on a reference mismatch and passes; fails only on a genuine error."""

    @pytest.mark.parametrize("budget", sorted(EPOCHS_COMPLETED_REFERENCE))
    def test_epochs_completed_against_reference(self, budget, capsys):
        observed = observe_epochs_completed(budget)
        check_epochs_completed(budget, observed, suspend_capture=capsys.disabled)


# ===================================================================
# THE MECHANISM -- mutation-style tests of the advisory machinery
# ===================================================================


def _forced_mismatch(budget):
    """An observed count that is in range but is not the reference, whatever the reference is."""
    return EPOCHS_COMPLETED_REFERENCE[budget] - 1


class TestAdvisoryMechanism:
    """Force a mismatch and prove it is reported and never raised; prove a match reports nothing."""

    @pytest.fixture(autouse=True)
    def _outside_ci(self, monkeypatch):
        """Every mechanism test starts OUTSIDE GitHub Actions, whatever the real environment is.

        These tests force mismatches on purpose. In CI, ``GITHUB_ACTIONS`` and ``GITHUB_STEP_SUMMARY``
        are set for real, so without this every CI run would write forced firings into the real job
        summary and annotation stream, and the count D6 exists to take would be noise.
        """
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        monkeypatch.setattr(sys, "__stdout__", io.StringIO())

    @pytest.fixture
    def ci_env(self, monkeypatch, tmp_path):
        """A simulated GitHub Actions step with a private annotation stream and step-summary file."""
        stream = io.StringIO()
        summary = tmp_path / "step_summary.md"
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        monkeypatch.setattr(sys, "__stdout__", stream)
        return types.SimpleNamespace(stream=stream, summary=summary)

    def test_mismatch_does_not_raise_and_warns(self):
        observed = _forced_mismatch(100)
        with pytest.warns(EpochsCompletedDriftWarning) as record:
            message = check_epochs_completed(100, observed, suspend_capture=contextlib.nullcontext)
        assert message is not None
        assert [str(w.message) for w in record] == [message]

    def test_mismatch_does_not_raise_even_when_warnings_are_errors(self):
        # What ``-W error`` or ``filterwarnings = ["error"]`` would install. Without the local
        # "always" filter in ``_report_drift`` the advisory would become a hard failure here.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("error")
            check_epochs_completed(100, _forced_mismatch(100), suspend_capture=contextlib.nullcontext)
        assert [w.category for w in caught] == [EpochsCompletedDriftWarning]

    def test_mismatch_appends_exactly_one_line_to_the_step_summary(self, ci_env):
        ci_env.summary.write_text("line written earlier in this step\n", encoding="utf-8")
        with pytest.warns(EpochsCompletedDriftWarning):
            message = check_epochs_completed(200, _forced_mismatch(200), suspend_capture=contextlib.nullcontext)
        assert ci_env.summary.read_text(encoding="utf-8") == f"line written earlier in this step\n- {message}\n"

    def test_mismatch_writes_one_annotation_on_a_line_of_its_own(self, ci_env):
        with pytest.warns(EpochsCompletedDriftWarning):
            message = check_epochs_completed(100, _forced_mismatch(100), suspend_capture=contextlib.nullcontext)
        # The leading newline matters: under --verbose pytest has already written "<nodeid> " with no
        # newline, and the runner reads a workflow command only from the START of a line.
        assert ci_env.stream.getvalue() == f"\n::warning title={ANNOTATION_TITLE}::{message}\n"

    def test_annotation_is_written_and_flushed_while_capture_is_suspended(self, ci_env, monkeypatch):
        events = []

        class _Stream(io.StringIO):
            def write(self, text):
                events.append("write")
                return super().write(text)

            def flush(self):
                events.append("flush")
                super().flush()

        @contextlib.contextmanager
        def _suspend():
            events.append("suspend")
            yield
            events.append("resume")

        monkeypatch.setattr(sys, "__stdout__", _Stream())
        with pytest.warns(EpochsCompletedDriftWarning):
            check_epochs_completed(100, _forced_mismatch(100), suspend_capture=_suspend)
        # A write outside the suspension lands in pytest's capture buffer and is discarded with it;
        # an unflushed one leaks only if something else happens to flush it, mid-line.
        assert events[0] == "suspend" and events[-1] == "resume", events
        assert set(events[1:-1]) == {"write", "flush"} and events[-2] == "flush", events

    @pytest.mark.parametrize("github_actions", [None, "false", ""], ids=["unset", "false", "empty"])
    def test_no_annotation_outside_github_actions(self, monkeypatch, github_actions):
        stream = io.StringIO()
        monkeypatch.setattr(sys, "__stdout__", stream)
        if github_actions is not None:
            monkeypatch.setenv("GITHUB_ACTIONS", github_actions)
        with pytest.warns(EpochsCompletedDriftWarning):
            check_epochs_completed(100, _forced_mismatch(100), suspend_capture=contextlib.nullcontext)
        assert stream.getvalue() == ""

    def test_a_match_emits_nothing_at_all(self, ci_env):
        suspended = []

        @contextlib.contextmanager
        def _suspend():
            suspended.append(True)
            yield

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = check_epochs_completed(100, EPOCHS_COMPLETED_REFERENCE[100], suspend_capture=_suspend)
        assert result is None
        assert caught == []
        assert suspended == []
        assert ci_env.stream.getvalue() == ""
        assert not ci_env.summary.exists()

    def test_message_carries_budget_values_pin_provenance_and_instruction(self):
        message = drift_message(200, 68, 71)
        assert message.startswith(f"{ANNOTATION_TITLE} at budget 200: expected 68, observed 71")
        assert f"torch thread pin {REFERENCE_THREAD_PIN}" in message
        assert f"cascor {REFERENCE_CASCOR_SHA[:7]}, {REFERENCE_DATE}" in message
        assert f"torch {torch.__version__}" in message
        assert message.endswith("advisory only (owner decision D6, 2026-09-22); if a change to candidate numerics is intended, update the reference in src/tests/unit/test_candidate_epochs_completed_advisory.py")
        assert "\n" not in message and "\r" not in message

    @pytest.mark.parametrize(
        "budget,observed",
        [(100, 0), (100, -1), (100, 101), (200, 201), (100, 68.0), (100, "68"), (100, True), (100, None)],
        ids=["zero", "negative", "over-budget-100", "over-budget-200", "float-equal-to-reference", "str", "bool", "none"],
    )
    def test_genuine_errors_still_fail_and_report_nothing(self, ci_env, budget, observed):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(AssertionError):
                check_epochs_completed(budget, observed, suspend_capture=contextlib.nullcontext)
        assert caught == []
        assert ci_env.stream.getvalue() == ""
        assert not ci_env.summary.exists()

    def test_an_exception_from_training_still_fails_and_the_width_is_restored(self, monkeypatch):
        def _boom(self, **kwargs):
            raise RuntimeError("candidate training blew up")

        monkeypatch.setattr(CandidateUnit, "train_detailed", _boom)
        original = torch.get_num_threads()
        other = REFERENCE_THREAD_PIN + 1
        torch.set_num_threads(other)
        try:
            with pytest.raises(RuntimeError, match="candidate training blew up"):
                observe_epochs_completed(100)
            assert torch.get_num_threads() == other
        finally:
            torch.set_num_threads(original)

    def test_training_runs_under_the_pin_and_the_prior_width_is_restored(self, monkeypatch):
        seen = []

        def _spy(self, **kwargs):
            seen.append(torch.get_num_threads())
            return CandidateTrainingResult(epochs_completed=kwargs["epochs"] // 2)

        monkeypatch.setattr(CandidateUnit, "train_detailed", _spy)
        original = torch.get_num_threads()
        other = REFERENCE_THREAD_PIN + 1
        torch.set_num_threads(other)
        try:
            assert observe_epochs_completed(100) == 50
            assert seen == [REFERENCE_THREAD_PIN]
            assert torch.get_num_threads() == other
        finally:
            torch.set_num_threads(original)

    def test_forced_mismatch_on_the_real_construction_is_reported_and_passes(self, ci_env, monkeypatch):
        observed = observe_epochs_completed(100)
        # Derived from the observation, so this stays a mismatch whatever cascor's numerics do.
        monkeypatch.setitem(EPOCHS_COMPLETED_REFERENCE, 100, observed + 1)
        with pytest.warns(EpochsCompletedDriftWarning) as record:
            message = check_epochs_completed(100, observed, suspend_capture=contextlib.nullcontext)
        assert f"expected {observed + 1}, observed {observed}" in message
        assert [str(w.message) for w in record] == [message]
        assert ci_env.summary.read_text(encoding="utf-8") == f"- {message}\n"
        assert ci_env.stream.getvalue() == f"\n::warning title={ANNOTATION_TITLE}::{message}\n"

    def test_an_unwritable_step_summary_does_not_fail_the_build(self, ci_env, monkeypatch, tmp_path):
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path))  # a directory: open() raises
        with pytest.warns(EpochsCompletedDriftWarning) as record:
            message = check_epochs_completed(100, _forced_mismatch(100), suspend_capture=contextlib.nullcontext)
        assert len(record) == 2
        assert "could not append to GITHUB_STEP_SUMMARY" in str(record[1].message)
        assert ci_env.stream.getvalue() == f"\n::warning title={ANNOTATION_TITLE}::{message}\n"

    def test_workflow_command_escaping(self):
        assert _escape_workflow_data("100%\r\nnext") == "100%25%0D%0Anext"
        assert _escape_workflow_property("a:b,c%") == "a%3Ab%2Cc%25"
        # The title needs no escaping, so a search for it matches the rendered annotation verbatim.
        assert _escape_workflow_property(ANNOTATION_TITLE) == ANNOTATION_TITLE
