#!/usr/bin/env python
"""Unit tests for the grow_network completion-reason diagnostic (Issue #3 follow-up).

``grow_network`` records *which* exit fired on the most recent growth run in
``self._completion_reason`` (surfaced by ``manager.get_status()`` as
``completion_reason``), so canopy can distinguish a genuine convergence from a
0-unit stall instead of both showing a bare "Completed". These tests pin each
break path to its reason string. Producer side only — the canopy consumer is a
separate follow-up.
"""

import datetime
from unittest.mock import MagicMock, patch

import pytest
import torch

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork, TrainingResults
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import CandidateTrainingError


def _make_network():
    config = CascadeCorrelationConfig(input_size=2, output_size=2, candidate_pool_size=2, candidate_epochs=1)
    return CascadeCorrelationNetwork(config=config)


def _xy():
    torch.manual_seed(0)
    return torch.randn(8, 2), torch.randn(8, 2)


def _all_failed_results(count=3, success_count=0, best_candidate=None, error_messages=None):
    """A ``TrainingResults`` shaped exactly like the all-candidates-errored return.

    Issue #509: the per-candidate handlers catch a training error and *return* a
    ``CandidateTrainingResult(success=False, candidate=None)``, so the result list
    is full-length while ``success_count`` is 0 and ``best_candidate`` is ``None``.
    A real dataclass (not a MagicMock) is used so the predicate is exercised
    honestly — a MagicMock would make every attribute truthy.
    """
    now = datetime.datetime.now()
    if error_messages is None:
        error_messages = {index: f"Candidate ID {index}: CUDA error: out of memory" for index in range(count)}
    return TrainingResults(
        epochs_completed=[0] * count,
        candidate_ids=list(range(count)),
        candidate_uuids=[],
        correlations=[],
        candidate_objects=[],
        best_candidate_id=-1,
        best_candidate_uuid=None,
        best_correlation=0.0,
        best_candidate=best_candidate,
        success_count=success_count,
        successful_candidates=0,
        failed_count=count,
        error_messages=error_messages,
        max_correlation=0.0,
        start_time=now,
        end_time=now,
    )


def _grow_with(net, results, **kwargs):
    """Run grow_network with the residual error and candidate results pinned."""
    x, y = _xy()
    with patch.object(net, "_calculate_residual_error_safe", return_value=torch.ones(8, 2)), patch.object(net, "_get_training_results", return_value=results):
        return net.grow_network(x_train=x, y_train=y, max_iterations=3, early_stopping=False, **kwargs)


@pytest.mark.unit
class TestCompletionReason:
    """Each grow_network exit sets the expected completion_reason."""

    def test_default_is_none_before_training(self):
        """A freshly constructed network has no completion reason yet."""
        assert _make_network()._completion_reason is None

    def test_max_iterations(self):
        """max_iterations=0 → the for-loop body never runs → for/else fires.

        Exercises the cap path with zero training work (range(0) is empty, so
        the ``else`` clause runs immediately).
        """
        net = _make_network()
        x, y = _xy()
        net.grow_network(x_train=x, y_train=y, max_iterations=0, early_stopping=False)
        assert net._completion_reason == "max_iterations"

    def test_residual_collapsed(self):
        """Residual error None on the first iteration → residual_collapsed."""
        net = _make_network()
        x, y = _xy()
        with patch.object(net, "_calculate_residual_error_safe", return_value=None):
            net.grow_network(x_train=x, y_train=y, max_iterations=3, early_stopping=False)
        assert net._completion_reason == "residual_collapsed"

    def test_no_candidate(self):
        """No training results / no best candidate → no_candidate (the stall signature)."""
        net = _make_network()
        x, y = _xy()
        with patch.object(net, "_calculate_residual_error_safe", return_value=torch.ones(8, 2)), patch.object(net, "_get_training_results", return_value=None):
            net.grow_network(x_train=x, y_train=y, max_iterations=3, early_stopping=False)
        assert net._completion_reason == "no_candidate"

    def test_below_threshold(self):
        """Best candidate correlation below the adaptive threshold → below_threshold."""
        net = _make_network()
        x, y = _xy()
        results = MagicMock()
        results.best_candidate.get_correlation.return_value = 1e-9  # far below any adaptive threshold
        with patch.object(net, "_calculate_residual_error_safe", return_value=torch.ones(8, 2)), patch.object(net, "_get_training_results", return_value=results):
            net.grow_network(x_train=x, y_train=y, max_iterations=3, early_stopping=False)
        assert net._completion_reason == "below_threshold"

    def test_early_stopped(self):
        """Validation early-stop on an otherwise healthy iteration → early_stopped."""
        net = _make_network()
        x, y = _xy()
        results = MagicMock()
        results.best_candidate.get_correlation.return_value = 10.0  # well above threshold
        validation = MagicMock(early_stop=True, patience_counter=0, best_value_loss=0.0)
        with patch.object(net, "_calculate_residual_error_safe", return_value=torch.ones(8, 2)), patch.object(net, "_get_training_results", return_value=results), patch.object(net, "_effective_candidate_count", return_value=1), patch.object(net, "_add_best_candidate", return_value=(0.1, 0.9)), patch.object(net, "validate_training", return_value=validation):
            net.grow_network(x_train=x, y_train=y, max_iterations=3, early_stopping=True)
        assert net._completion_reason == "early_stopped"


@pytest.mark.unit
class TestCandidateTrainingFailed:
    """Issue #509: an all-candidates-errored round must not masquerade as ``no_candidate``.

    ``no_candidate`` means "no candidate was good enough" — a real algorithmic
    outcome. When *every* candidate errors (the observed case is a full GPU), the
    run trained nothing, and reporting that as a normal completion silently
    corrupts experiment campaigns. These tests pin the separation.
    """

    def test_all_candidates_failed_raises(self):
        """0 of N trained successfully → raise rather than break with a benign reason."""
        net = _make_network()
        with pytest.raises(CandidateTrainingError, match="failed to train"):
            _grow_with(net, _all_failed_results())

    def test_all_candidates_failed_sets_its_own_reason(self):
        """The reason is set *before* the raise and survives it (get_status reads it off the network)."""
        net = _make_network()
        with pytest.raises(CandidateTrainingError):
            _grow_with(net, _all_failed_results())
        assert net._completion_reason == "candidate_training_failed"

    def test_underlying_error_is_surfaced(self):
        """The candidate error text reaches the message — that is what makes it diagnosable."""
        net = _make_network()
        with pytest.raises(CandidateTrainingError, match="out of memory"):
            _grow_with(net, _all_failed_results())

    def test_partial_success_still_reports_no_candidate(self):
        """At least one candidate trained → genuine algorithmic outcome, not infrastructure.

        Guards against the fix over-firing: a degraded-but-honest round must keep
        its benign exit.
        """
        net = _make_network()
        _grow_with(net, _all_failed_results(success_count=1))
        assert net._completion_reason == "no_candidate"

    def test_no_candidates_attempted_still_reports_no_candidate(self):
        """Nothing attempted → nothing to call an infrastructure failure."""
        net = _make_network()
        _grow_with(net, _all_failed_results(count=0))
        assert net._completion_reason == "no_candidate"

    def test_none_results_still_reports_no_candidate(self):
        """The pre-existing None path is untouched by the guard."""
        net = _make_network()
        _grow_with(net, None)
        assert net._completion_reason == "no_candidate"

    def test_error_messages_as_list_is_handled(self):
        """``error_messages`` is annotated ``List[str]`` but produced as a dict — accept both."""
        net = _make_network()
        with pytest.raises(CandidateTrainingError, match="listed failure"):
            _grow_with(net, _all_failed_results(error_messages=["listed failure"]))

    def test_missing_error_messages_still_raises(self):
        """An empty error map must not suppress the failure or crash the guard."""
        net = _make_network()
        with pytest.raises(CandidateTrainingError, match="no error message recorded"):
            _grow_with(net, _all_failed_results(error_messages={}))
