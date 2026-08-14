"""Tests for L-1: the direct CLI must honour the configured output-epoch budget.

``solve_n_spiral_problem`` used to call ``self.network.fit(max_epochs=_SPIRAL_PROBLEM_OUTPUT_EPOCHS)``
-- the module *constant* -- rather than ``self.output_epochs``, the instance attribute
main.py's W-11 mapping populates from the experiment YAML. ``fit`` keeps a non-None
``max_epochs`` as-is (``cascade_correlation.py``: ``max_epochs = (max_epochs, self.output_epochs)[max_epochs is None]``)
and spends it on the initial output-layer pass, so the configured budget was silently
discarded for that pass while every per-round pass inside ``grow_network`` -- which reads
``self.output_epochs`` -- correctly honoured it.

Measured on the preserved R-5 arm C run (YAML ``training.params.max_epochs: 100``): the
initial pass ran the constant's full 10000 epochs, taking ~18 s of a ~40 s run, with the
loss flat at 0.203637 from roughly epoch 150 onward; the two later per-round passes in the
same log each stopped at exactly epoch 100.

The defect also contradicted the project's own cost model:
``TrainingLifecycleManager.derive_epochs_cap`` documents the initial pass as costing
``output_epochs``, and ``_W11_TRAINING_KEY_MAP`` maps the YAML's ``max_epochs`` onto
``output_epochs`` precisely so it can bound that pass.

These tests pin the fix and the wiring it depends on.
"""

import ast
import inspect
import textwrap

import pytest

pytestmark = pytest.mark.unit


def _fit_call_keywords():
    """Return the keyword nodes of the ``fit`` call inside ``solve_n_spiral_problem``."""
    import spiral_problem.spiral_problem as sp

    source = textwrap.dedent(inspect.getsource(sp.SpiralProblem.solve_n_spiral_problem))
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "fit"]
    assert len(calls) == 1, f"expected exactly one fit() call in solve_n_spiral_problem, found {len(calls)}"
    return {kw.arg: kw.value for kw in calls[0].keywords}


class TestSolveUsesInstanceBudget:
    def test_fit_receives_self_output_epochs(self):
        """The configured budget, not a module constant, bounds the initial output pass."""
        max_epochs = _fit_call_keywords().get("max_epochs")
        assert max_epochs is not None, "solve_n_spiral_problem must pass max_epochs explicitly"
        assert isinstance(max_epochs, ast.Attribute), f"max_epochs must be an attribute lookup, got {type(max_epochs).__name__}"
        assert isinstance(max_epochs.value, ast.Name) and max_epochs.value.id == "self"
        assert max_epochs.attr == "output_epochs"

    def test_fit_does_not_receive_a_module_constant(self):
        """Anti-resurrection: a bare _SPIRAL_PROBLEM_* name here re-breaks the budget."""
        max_epochs = _fit_call_keywords()["max_epochs"]
        assert not isinstance(max_epochs, ast.Name), f"max_epochs must not be a bare module constant ({getattr(max_epochs, 'id', '')})"


class TestOutputEpochsWiring:
    """The fix reads ``self.output_epochs``; these pin what fills it."""

    def test_constructor_kwarg_populates_the_attribute(self):
        from spiral_problem.spiral_problem import SpiralProblem

        assert SpiralProblem(_SpiralProblem__output_epochs=137).output_epochs == 137

    def test_w11_maps_yaml_max_epochs_onto_output_epochs(self):
        """C2b semantics: the YAML's max_epochs is the initial-output-pass budget."""
        from main import _W11_TRAINING_KEY_MAP

        assert _W11_TRAINING_KEY_MAP["max_epochs"] == "output_epochs"
        assert _W11_TRAINING_KEY_MAP["output_epochs"] == "output_epochs"

    def test_main_threads_the_w11_value_into_the_constructor(self):
        """Without this hop the YAML budget never reaches the attribute the fix reads."""
        import main

        source = inspect.getsource(main)
        assert '_SpiralProblem__output_epochs=_w11.get("output_epochs"' in source
