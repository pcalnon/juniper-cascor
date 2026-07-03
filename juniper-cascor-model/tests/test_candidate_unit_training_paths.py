#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     test_candidate_unit_training_paths.py
# Author:        Paul Calnon
#
# Date Created:  2026-07-03
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#     Behavioral coverage for the CandidateUnit training hot-path branches that the
#     existing contract/coverage suites leave unexercised: the payload-coercion guards,
#     the log-level-gated diagnostic blocks (run once at TRACE so every ``_log_debug`` /
#     ``_log_trace`` / ``_log_verbose`` guard evaluates truthy), the sequence-roll cap
#     warning, the empty-correlation fallback, and display-function re-initialization.
#     Part of the juniper-cascor-model per-file coverage rollout (C-5).
#
#####################################################################################################################################################################################################

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from candidate_unit.candidate_unit import CandidateParametersUpdate, CandidateUnit  # noqa: E402
from cascor_constants.constants_candidates.constants_candidates import _PROJECT_MODEL_CANDIDATE_MAX_ROLL_COUNT  # noqa: E402
from log_config.logger.logger import Logger  # noqa: E402


def _make_candidate(**overrides):
    params = {
        "CandidateUnit__input_size": 2,
        "CandidateUnit__output_size": 1,
        "CandidateUnit__random_seed": 13,
        "CandidateUnit__candidate_index": 1,
        "CandidateUnit__display_frequency": 0,
        "CandidateUnit__status_frequency": 0,
        "CandidateUnit__random_value_scale": 0.01,
        "CandidateUnit__log_level_name": "CRITICAL",
    }
    params.update(overrides)
    return CandidateUnit(**params)


class TestCoerceIntLike:
    """The JSON-payload numeric coercion guard used by the seed/range APIs."""

    def test_bool_is_rejected(self):
        with pytest.raises(TypeError, match="not bool"):
            CandidateUnit._coerce_int_like(True, "CandidateUnit__candidate_index")

    def test_non_integer_float_is_rejected(self):
        with pytest.raises(ValueError, match="integer-valued number"):
            CandidateUnit._coerce_int_like(2.5, "CandidateUnit__random_seed")

    def test_integer_valued_float_is_normalized(self):
        assert CandidateUnit._coerce_int_like(4.0, "CandidateUnit__candidate_index") == 4


class TestTrainDiagnosticPaths:
    """Run a real training loop at TRACE so every level-gated diagnostic block executes."""

    def test_train_at_trace_level_exercises_diagnostic_guards(self):
        saved_level = Logger._log_level
        try:
            candidate = _make_candidate(
                CandidateUnit__output_size=1,
                CandidateUnit__log_level_name="TRACE",
                CandidateUnit__early_stopping=True,
                CandidateUnit__patience=1,
                CandidateUnit__convergence_threshold=10.0,
            )
            # Multi-output residual so the _multi_output_correlation loop runs more than once.
            x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
            residual_error = torch.tensor(
                [[-0.5, 0.1], [0.25, -0.2], [0.25, 0.3], [0.5, -0.4]],
                dtype=torch.float32,
            )

            correlation = candidate.train(x=x, residual_error=residual_error, epochs=3, learning_rate=0.05)

            assert isinstance(correlation, float)
            assert 0.0 <= correlation <= 1.0
            assert candidate.last_training_result.epochs_completed >= 1
        finally:
            Logger._log_level = saved_level


class TestSequenceRollCap:
    """The OOM-guard cap in _roll_sequence_number warns when the sequence overshoots."""

    def test_roll_sequence_number_caps_and_warns(self):
        candidate = _make_candidate()
        calls = {"count": 0}

        def _counting_generator(_low, _high):
            calls["count"] += 1
            return 0

        # A sequence beyond the cap must clamp the roll count and log the cap warning,
        # never iterate the full (potentially 2**32) sequence.
        candidate._roll_sequence_number(
            sequence=_PROJECT_MODEL_CANDIDATE_MAX_ROLL_COUNT + 5,
            max_value=10,
            generator=_counting_generator,
        )

        assert calls["count"] == _PROJECT_MODEL_CANDIDATE_MAX_ROLL_COUNT


class TestEmptyCorrelationFallback:
    """_get_correlations must degrade to an unsuccessful result when no outputs correlate."""

    def test_zero_width_residual_yields_unsuccessful_result(self):
        candidate = _make_candidate()
        output = torch.zeros(4)
        # A zero-feature residual makes the correlation loop empty -> best_idx < 0 branch.
        residual_error = torch.zeros(4, 0)

        result = candidate._get_correlations(output=output, residual_error=residual_error)

        assert result.success is False
        assert result.best_corr_idx == -1
        assert result.correlation == 0.0
        assert result.all_correlations == []


class TestDisplayProgressReinit:
    """_display_training_progress rebuilds its frequency checker if it was cleared."""

    def test_display_progress_reinitializes_when_none(self):
        candidate = _make_candidate(CandidateUnit__display_frequency=0)
        candidate.clear_display_progress()
        assert candidate._candidate_display_progress is None

        update = CandidateParametersUpdate(
            norm_output=torch.zeros(4),
            norm_error=torch.zeros(4),
        )
        residual_error = torch.zeros(4, 1)

        candidate._display_training_progress(0, update, residual_error)

        # The checker must have been rebuilt (no longer None) by the re-init branch.
        assert candidate._candidate_display_progress is not None
