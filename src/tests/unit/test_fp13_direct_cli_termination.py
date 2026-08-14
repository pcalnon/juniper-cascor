"""Tests for F-P1-3: the direct CLI must terminate after training finishes.

The direct CLI used to hang forever once the cascade stopped growing.
``solve_n_spiral_problem`` ends with ``plt.show()`` followed by
``self.plotter.join()``; under an interactive backend the first parks the process
in the GUI event loop and the second waits on a (non-daemon) child parked in its
own ``plt.show()``. Plotting is on by default and, before this change, had no
CLI or experiment-YAML knob, so an automated run could only be killed at whatever
bound the caller set -- which read from outside as a compute-bound run.

Measured on the P1/R-5 smoke arm (2 hidden units, pool 4): training finished in
~39 s in every arm; unfixed + interactive backend hung past a 240 s bound, while
``--no-plots`` terminated cleanly in ~38-40 s.

These tests pin the two halves of the fix:
  * ``--no-plots`` exists and reaches ``main(generate_plots=...)``;
  * the blocking pair is guarded by ``_backend_is_interactive()``, which is
    False for the non-interactive backends a headless run resolves to.
"""

import inspect

import pytest

pytestmark = pytest.mark.unit


class TestNoPlotsFlag:
    def test_no_plots_defaults_to_false(self, monkeypatch):
        import sys

        from main import parse_args

        monkeypatch.setattr(sys, "argv", ["main.py"])
        assert parse_args().no_plots is False

    def test_no_plots_sets_true(self, monkeypatch):
        import sys

        from main import parse_args

        monkeypatch.setattr(sys, "argv", ["main.py", "--no-plots"])
        assert parse_args().no_plots is True

    def test_main_accepts_generate_plots(self):
        """--no-plots is inert unless main() actually takes the flag through."""
        import main

        assert "generate_plots" in inspect.signature(main.main).parameters

    def test_entrypoint_threads_flag_into_every_main_call(self):
        """cProfile / tracemalloc runs are automated too -- none may call a bare main()."""
        import main

        source = inspect.getsource(main)
        entrypoint = source.split('if __name__ == "__main__":', 1)[1]
        assert "not args.no_plots" in entrypoint
        # Every main() call on the entrypoint path must pass the resolved flag.
        assert "main()" not in entrypoint
        assert entrypoint.count("main(generate_plots=generate_plots)") == 3


class TestBackendIsInteractive:
    @pytest.mark.parametrize("backend", ["agg", "Agg", "pdf", "svg", "template", "ps"])
    def test_non_interactive_backends(self, backend, monkeypatch):
        import spiral_problem.spiral_problem as sp

        monkeypatch.setattr(sp.matplotlib, "get_backend", lambda: backend)
        assert sp._backend_is_interactive() is False

    @pytest.mark.parametrize("backend", ["tkagg", "TkAgg", "qtagg"])
    def test_interactive_backends(self, backend, monkeypatch):
        import spiral_problem.spiral_problem as sp

        monkeypatch.setattr(sp.matplotlib, "get_backend", lambda: backend)
        assert sp._backend_is_interactive() is True

    def test_falls_back_when_registry_unavailable(self, monkeypatch):
        """matplotlib < 3.9 (or a renamed registry) must still classify Agg correctly."""
        import builtins

        import spiral_problem.spiral_problem as sp

        real_import = builtins.__import__

        def _no_registry(name, *args, **kwargs):
            if name == "matplotlib.backends":
                raise ImportError("simulated: no backend_registry")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_registry)
        monkeypatch.setattr(sp.matplotlib, "get_backend", lambda: "agg")
        assert sp._backend_is_interactive() is False
        monkeypatch.setattr(sp.matplotlib, "get_backend", lambda: "tkagg")
        assert sp._backend_is_interactive() is True


class TestTerminalBlockIsGuarded:
    def test_show_and_join_sit_behind_the_backend_guard(self):
        """The anti-regression pin: neither blocking call may run unguarded again.

        Matched on whole code lines rather than substrings -- the guarded block's own
        comment and log message mention ``plt.show()`` in prose, and a substring count
        would score those as call sites.
        """
        from spiral_problem.spiral_problem import SpiralProblem

        lines = [line.strip() for line in inspect.getsource(SpiralProblem.solve_n_spiral_problem).splitlines()]

        guard_idx = [i for i, line in enumerate(lines) if line == "if _backend_is_interactive():"]
        show_idx = [i for i, line in enumerate(lines) if line == "plt.show()"]
        join_idx = [i for i, line in enumerate(lines) if line == "self.plotter.join()"]

        assert len(guard_idx) == 1, "expected exactly one backend guard"
        assert len(show_idx) == 1, "expected exactly one plt.show() call site"
        assert len(join_idx) == 1, "expected exactly one plotter.join() call site"
        assert show_idx[0] > guard_idx[0], "plt.show() must sit inside the backend guard"
        assert join_idx[0] > guard_idx[0], "plotter.join() must sit inside the backend guard"
