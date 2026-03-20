"""Unit tests for Phase 1a security hardening of CasCor concurrency.

Tests cover:
- RestrictedUnpickler allowlist enforcement
- _validate_training_result bounds and type checking
- Queue maxsize limits
- Random authkey generation when no default is configured
"""

import io
import pickle
from unittest.mock import MagicMock, patch

import pytest
import torch

from candidate_unit.candidate_unit import CandidateTrainingResult, CandidateUnit
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork, RestrictedUnpickler
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

# ===================================================================
# RestrictedUnpickler Tests
# ===================================================================


@pytest.mark.unit
class TestRestrictedUnpickler:
    """Tests for RestrictedUnpickler allowlist enforcement."""

    def test_allows_builtins(self) -> None:
        """RestrictedUnpickler should allow standard builtins."""
        data = pickle.dumps([1, 2.0, "hello", True, (3, 4)])
        result = RestrictedUnpickler.loads(data)
        assert result == [1, 2.0, "hello", True, (3, 4)]

    def test_allows_dict(self) -> None:
        """RestrictedUnpickler should allow dict."""
        data = pickle.dumps({"key": "value", "num": 42})
        result = RestrictedUnpickler.loads(data)
        assert result == {"key": "value", "num": 42}

    def test_blocks_os_system(self) -> None:
        """RestrictedUnpickler should block os.system and similar dangerous calls."""
        import os

        # Use pickle's __reduce__ mechanism to create a payload that would call os.system
        class MaliciousOS:
            def __reduce__(self):
                return (os.system, ("echo pwned",))

        payload = pickle.dumps(MaliciousOS())
        with pytest.raises(pickle.UnpicklingError, match="Blocked unpickling of"):
            RestrictedUnpickler.loads(payload)

    def test_blocks_subprocess(self) -> None:
        """RestrictedUnpickler should block subprocess module."""
        import subprocess

        class MaliciousSubprocess:
            def __reduce__(self):
                return (subprocess.call, (["echo", "hi"],))

        payload = pickle.dumps(MaliciousSubprocess())
        with pytest.raises(pickle.UnpicklingError, match="Blocked unpickling of"):
            RestrictedUnpickler.loads(payload)

    def test_allows_candidate_training_result(self) -> None:
        """RestrictedUnpickler should allow CandidateTrainingResult."""
        result = CandidateTrainingResult(candidate_id=0, correlation=0.5, success=True)
        data = pickle.dumps(result)
        restored = RestrictedUnpickler.loads(data)
        assert isinstance(restored, CandidateTrainingResult)
        assert restored.candidate_id == 0
        assert restored.correlation == 0.5

    def test_allows_torch_tensor(self) -> None:
        """RestrictedUnpickler should allow torch.Tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        data = pickle.dumps(tensor)
        restored = RestrictedUnpickler.loads(data)
        assert isinstance(restored, torch.Tensor)
        assert torch.equal(restored, tensor)

    def test_allowlist_completeness_for_training_result_with_candidate(self) -> None:
        """RestrictedUnpickler should handle a full CandidateTrainingResult with a CandidateUnit."""
        candidate = CandidateUnit(
            _CandidateUnit__input_size=2,
            _CandidateUnit__learning_rate=0.01,
            _CandidateUnit__epochs=10,
            _CandidateUnit__log_level_name="ERROR",
        )
        result = CandidateTrainingResult(
            candidate_id=0,
            correlation=0.75,
            candidate=candidate,
            norm_output=torch.randn(10),
            norm_error=torch.randn(10),
            success=True,
            epochs_completed=10,
        )
        data = pickle.dumps(result)
        restored = RestrictedUnpickler.loads(data)
        assert isinstance(restored, CandidateTrainingResult)
        assert isinstance(restored.candidate, CandidateUnit)
        assert restored.correlation == 0.75


# ===================================================================
# _validate_training_result Tests
# ===================================================================


@pytest.mark.unit
class TestValidateTrainingResult:
    """Tests for CascadeCorrelationNetwork._validate_training_result."""

    @pytest.fixture
    def network(self) -> CascadeCorrelationNetwork:
        """Create a minimal network for testing validation."""
        config = CascadeCorrelationConfig.create_simple_config(
            input_size=2,
            output_size=1,
            max_hidden_units=2,
            candidate_pool_size=2,
            candidate_epochs=3,
            output_epochs=3,
            epochs_max=5,
        )
        return CascadeCorrelationNetwork(config=config)

    def test_valid_result(self, network) -> None:
        """Valid CandidateTrainingResult should pass validation."""
        result = CandidateTrainingResult(
            candidate_id=0,
            correlation=0.85,
            success=True,
            epochs_completed=100,
            norm_output=torch.randn(10),
            norm_error=torch.randn(10),
        )
        assert network._validate_training_result(result) is True

    def test_valid_result_with_candidate(self, network) -> None:
        """Valid result with CandidateUnit should pass validation."""
        candidate = CandidateUnit(
            _CandidateUnit__input_size=2,
            _CandidateUnit__learning_rate=0.01,
            _CandidateUnit__epochs=10,
            _CandidateUnit__log_level_name="ERROR",
        )
        result = CandidateTrainingResult(
            candidate_id=0,
            correlation=0.5,
            candidate=candidate,
            success=True,
        )
        assert network._validate_training_result(result) is True

    def test_wrong_type(self, network) -> None:
        """Non-CandidateTrainingResult should fail validation."""
        assert network._validate_training_result("not a result") is False
        assert network._validate_training_result(42) is False
        assert network._validate_training_result(None) is False

    def test_correlation_out_of_bounds_high(self, network) -> None:
        """Correlation > 1.0 should fail validation."""
        result = CandidateTrainingResult(correlation=1.5)
        assert network._validate_training_result(result) is False

    def test_correlation_out_of_bounds_negative(self, network) -> None:
        """Negative correlation should fail validation."""
        result = CandidateTrainingResult(correlation=-0.1)
        assert network._validate_training_result(result) is False

    def test_correlation_boundary_zero(self, network) -> None:
        """Correlation of exactly 0.0 should pass."""
        result = CandidateTrainingResult(correlation=0.0)
        assert network._validate_training_result(result) is True

    def test_correlation_boundary_one(self, network) -> None:
        """Correlation of exactly 1.0 should pass."""
        result = CandidateTrainingResult(correlation=1.0)
        assert network._validate_training_result(result) is True

    def test_invalid_candidate_type(self, network) -> None:
        """Non-CandidateUnit candidate should fail validation."""
        result = CandidateTrainingResult(
            correlation=0.5,
            candidate="not a candidate unit",
        )
        assert network._validate_training_result(result) is False

    def test_norm_output_nan(self, network) -> None:
        """norm_output with NaN should fail validation."""
        result = CandidateTrainingResult(
            correlation=0.5,
            norm_output=torch.tensor([1.0, float("nan"), 3.0]),
        )
        assert network._validate_training_result(result) is False

    def test_norm_output_inf(self, network) -> None:
        """norm_output with Inf should fail validation."""
        result = CandidateTrainingResult(
            correlation=0.5,
            norm_output=torch.tensor([1.0, float("inf"), 3.0]),
        )
        assert network._validate_training_result(result) is False

    def test_norm_error_nan(self, network) -> None:
        """norm_error with NaN should fail validation."""
        result = CandidateTrainingResult(
            correlation=0.5,
            norm_error=torch.tensor([1.0, float("nan")]),
        )
        assert network._validate_training_result(result) is False

    def test_norm_error_inf(self, network) -> None:
        """norm_error with Inf should fail validation."""
        result = CandidateTrainingResult(
            correlation=0.5,
            norm_error=torch.tensor([float("-inf"), 1.0]),
        )
        assert network._validate_training_result(result) is False

    def test_none_tensors_pass(self, network) -> None:
        """None norm_output and norm_error should pass validation."""
        result = CandidateTrainingResult(
            correlation=0.5,
            norm_output=None,
            norm_error=None,
        )
        assert network._validate_training_result(result) is True

    def test_none_candidate_passes(self, network) -> None:
        """None candidate should pass validation (failure results have candidate=None)."""
        result = CandidateTrainingResult(
            correlation=0.0,
            candidate=None,
            success=False,
            error_message="Training failed",
        )
        assert network._validate_training_result(result) is True


# ===================================================================
# Queue Size Limit Tests
# ===================================================================


@pytest.mark.unit
class TestQueueSizeLimits:
    """Tests for queue maxsize enforcement."""

    def test_manager_task_queue_has_maxsize(self) -> None:
        """Manager-hosted task queue should have maxsize=_QUEUE_MAXSIZE (1024)."""
        import cascade_correlation.cascade_correlation as cc_module
        from cascade_correlation.cascade_correlation import _QUEUE_MAXSIZE, _create_task_queue

        # Reset module-level queue to force recreation
        original = cc_module._task_queue
        cc_module._task_queue = None
        try:
            q = _create_task_queue()
            assert q.maxsize == _QUEUE_MAXSIZE
        finally:
            cc_module._task_queue = original

    def test_manager_result_queue_has_maxsize(self) -> None:
        """Manager-hosted result queue should have maxsize=_QUEUE_MAXSIZE (1024)."""
        import cascade_correlation.cascade_correlation as cc_module
        from cascade_correlation.cascade_correlation import _QUEUE_MAXSIZE, _create_result_queue

        original = cc_module._result_queue
        cc_module._result_queue = None
        try:
            q = _create_result_queue()
            assert q.maxsize == _QUEUE_MAXSIZE
        finally:
            cc_module._result_queue = original


# ===================================================================
# Authkey Generation Tests
# ===================================================================


@pytest.mark.unit
class TestAuthkeyGeneration:
    """Tests for random authkey generation when no default is configured."""

    def test_none_authkey_generates_random(self) -> None:
        """Config with None authkey should auto-generate a random hex string."""
        config = CascadeCorrelationConfig.create_simple_config(
            input_size=2,
            output_size=1,
        )
        assert config.candidate_training_queue_authkey is not None
        assert isinstance(config.candidate_training_queue_authkey, str)
        assert len(config.candidate_training_queue_authkey) == 64  # 32 bytes = 64 hex chars

    def test_two_configs_get_different_authkeys(self) -> None:
        """Two configs should get different random authkeys."""
        config1 = CascadeCorrelationConfig.create_simple_config(
            input_size=2,
            output_size=1,
        )
        config2 = CascadeCorrelationConfig.create_simple_config(
            input_size=2,
            output_size=1,
        )
        assert config1.candidate_training_queue_authkey != config2.candidate_training_queue_authkey

    def test_explicit_authkey_preserved(self) -> None:
        """Explicitly provided authkey should not be overridden."""
        config = CascadeCorrelationConfig.create_simple_config(
            input_size=2,
            output_size=1,
            candidate_training_queue_authkey="my-explicit-key",
        )
        assert config.candidate_training_queue_authkey == "my-explicit-key"

    def test_default_constant_is_none(self) -> None:
        """The default authkey constant should be None (no hardcoded value)."""
        from cascor_constants.constants_model.constants_model import _PROJECT_MODEL_AUTHKEY

        assert _PROJECT_MODEL_AUTHKEY is None
