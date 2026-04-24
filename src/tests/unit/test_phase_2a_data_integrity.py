#!/usr/bin/env python
"""
Regression tests for Phase 2A (Track 2) data-integrity bug fixes in juniper-cascor.

Covers:
- BUG-CC-11: walrus operator precedence in utils._object_attributes_to_table
- BUG-CC-03: falsy-safe parameter fallbacks in SpiralProblem parameter init methods
- BUG-CC-18 / ROBUST-01: CandidateTrainingError raised on double candidate-training failure
  instead of silently installing dummy zero-correlation candidates.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

# Ensure `src/` is on sys.path so the top-level packages (utils, spiral_problem, ...) resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# BUG-CC-11: walrus operator precedence in utils._object_attributes_to_table
# ---------------------------------------------------------------------------
class TestBugCC11WalrusPrecedence:
    """`content := _init_content_list(...)` must receive the list, not a bool.

    Before the fix, `if content := _init_content_list(...) is not None:` bound
    ``content`` to the result of `is not None` (True/False), then attempted to
    call `.append` on a boolean inside the loop body.
    """

    def test_attributes_table_returns_string_for_valid_input(self):
        """With the walrus fix, a dict with public keys produces a non-None table."""
        from utils.utils import _object_attributes_to_table

        obj_dict = {"alpha": 1, "beta": 2}
        keys = list(obj_dict.keys())

        result = _object_attributes_to_table(
            obj_dict=obj_dict,
            keys=keys,
            private_attrs=False,
        )

        # Before fix: `content` was a bool, calling .append() would AttributeError.
        # After fix: we get a rendered string (columnar or fallback) containing both keys.
        assert result is not None
        assert isinstance(result, str)
        assert "alpha" in result
        assert "beta" in result

    def test_attributes_table_handles_only_private_attrs(self):
        """Private-only input with ``private_attrs=False`` still renders (empty) without crashing."""
        from utils.utils import _object_attributes_to_table

        obj_dict = {"_hidden": 1}
        keys = list(obj_dict.keys())

        # Would AttributeError pre-fix; post-fix returns either None or an empty-rendered string.
        result = _object_attributes_to_table(
            obj_dict=obj_dict,
            keys=keys,
            private_attrs=False,
        )
        # No crash is the assertion of interest; accept either empty-string/None.
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# BUG-CC-03: falsy-safe parameter fallbacks in SpiralProblem
# ---------------------------------------------------------------------------
def _make_spiral_problem_shell():
    """Create an un-__init__'d SpiralProblem shell with a no-op logger.

    Avoids the heavy constructor (Logger/LogConfig/network wiring) while still
    allowing direct exercise of private parameter-initialization methods.
    """
    from spiral_problem.spiral_problem import SpiralProblem

    sp = SpiralProblem.__new__(SpiralProblem)
    sp.logger = MagicMock()
    return sp


class TestBugCC03FalsyFallbacks:
    """`param if param is not None else ...` must preserve valid falsy values.

    Before the fix, `param or self.param or DEFAULT` silently replaced ``False``,
    ``0``, and ``0.0`` with class attributes or module defaults.
    """

    def test_initialize_preserves_boolean_false_clockwise(self):
        from spiral_problem import spiral_problem as sp_mod

        sp = _make_spiral_problem_shell()
        # Seed attributes so the fallback chain can be exercised.
        sp.min_new = sp.max_new = 0
        sp.min_orig = sp.max_orig = 0
        sp.orig_points = None
        sp.train_ratio = sp.test_ratio = 0.5
        sp.clockwise = True  # class default truthy — explicit False must override.
        sp.n_spirals = sp.n_rotations = sp.n_points = 1
        sp.default_origin = sp.default_radius = 0
        sp.noise = sp.distribution = 0.0

        sp._initialize_spiral_problem_params(clockwise=False)

        # Pre-fix this would silently become True (because `False or True or DEFAULT == True`).
        assert sp.clockwise is False

    def test_initialize_preserves_zero_noise(self):
        sp = _make_spiral_problem_shell()
        sp.min_new = sp.max_new = sp.min_orig = sp.max_orig = 0
        sp.orig_points = None
        sp.train_ratio = sp.test_ratio = 0.5
        sp.clockwise = False
        sp.n_spirals = sp.n_rotations = sp.n_points = 1
        sp.default_origin = sp.default_radius = 0
        sp.noise = 0.25  # non-zero prior value
        sp.distribution = 1.0

        sp._initialize_spiral_problem_params(noise_level=0.0)

        # Pre-fix: `0.0 or 0.25 or DEFAULT` -> 0.25. Post-fix: caller's 0.0 wins.
        assert sp.noise == 0.0

    def test_initialize_preserves_zero_integer_params(self):
        sp = _make_spiral_problem_shell()
        sp.min_new = sp.max_new = sp.min_orig = sp.max_orig = 0
        sp.orig_points = 10  # non-zero prior value
        sp.train_ratio = sp.test_ratio = 0.5
        sp.clockwise = False
        sp.n_spirals = sp.n_rotations = sp.n_points = 1
        sp.default_origin = sp.default_radius = 0
        sp.noise = sp.distribution = 0.0

        sp._initialize_spiral_problem_params(orig_points=0)

        # Pre-fix: `0 or 10 or DEFAULT` -> 10. Post-fix: caller's 0 wins.
        assert sp.orig_points == 0

    def test_none_still_falls_back_to_class_attribute(self):
        """Regression for the non-None fallback path: None should fall through to self.X."""
        sp = _make_spiral_problem_shell()
        sp.min_new = sp.max_new = sp.min_orig = sp.max_orig = 0
        sp.orig_points = None
        sp.train_ratio = 0.8
        sp.test_ratio = 0.2
        sp.clockwise = True
        sp.n_spirals = 3
        sp.n_rotations = sp.n_points = 1
        sp.default_origin = sp.default_radius = 0
        sp.noise = 0.05
        sp.distribution = 1.0

        sp._initialize_spiral_problem_params(n_spirals=None, noise_level=None)

        # None caller -> class attribute (not module default).
        assert sp.n_spirals == 3
        assert sp.noise == 0.05


# ---------------------------------------------------------------------------
# BUG-CC-18 / ROBUST-01: CandidateTrainingError on double failure
# ---------------------------------------------------------------------------
class TestBugCC18CandidateTrainingError:
    """Double failure in _execute_candidate_training must raise, not install dummies.

    Pre-fix: on sequential-fallback failure OR empty-result return, the code
    silently installed zero-correlation dummy candidates via
    ``_get_dummy_results``, corrupting the network with meaningless data.
    Post-fix: an explicit ``CandidateTrainingError`` propagates to the caller.
    """

    def test_candidate_training_error_is_subclass_of_training_error(self):
        """CandidateTrainingError must be a TrainingError so existing catch blocks still work."""
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import CandidateTrainingError, TrainingError

        assert issubclass(CandidateTrainingError, TrainingError)

    def test_execute_raises_when_sequential_fallback_fails(self):
        """Both-paths-failed path must raise CandidateTrainingError instead of returning dummies."""
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import CandidateTrainingError

        net = CascadeCorrelationNetwork.__new__(CascadeCorrelationNetwork)
        net.logger = MagicMock()
        # Steer _execute_candidate_training down the sequential-only path.
        task_distributor = MagicMock()
        task_distributor.remote_worker_count = 0
        net._task_distributor = task_distributor
        net._execute_sequential_training = MagicMock(side_effect=RuntimeError("sequential boom"))
        # Non-dict, length-1 task list: sequential is attempted first, then the except
        # branch retries sequential, which also raises -> CandidateTrainingError.
        tasks = [(0, "uuid-0", [None, None, None, None])]

        with pytest.raises(CandidateTrainingError) as excinfo:
            net._execute_candidate_training(tasks, process_count=1)

        assert "sequential" in str(excinfo.value).lower() or "parallel" in str(excinfo.value).lower()

    def test_execute_raises_when_sequential_returns_empty(self):
        """Empty-results branch must also raise rather than synthesize dummies."""
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        from cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions import CandidateTrainingError

        net = CascadeCorrelationNetwork.__new__(CascadeCorrelationNetwork)
        net.logger = MagicMock()
        task_distributor = MagicMock()
        task_distributor.remote_worker_count = 0
        net._task_distributor = task_distributor
        # Single-process sequential path returns no results — this is the empty-results branch.
        net._execute_sequential_training = MagicMock(return_value=[])
        tasks = [(0, "uuid-0", [None, None, None, None])]

        with pytest.raises(CandidateTrainingError):
            net._execute_candidate_training(tasks, process_count=1)
