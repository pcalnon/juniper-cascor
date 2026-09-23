"""BLAS thread policy: one policy, applied identically by BOTH entry points (#531).

The defect these tests lock down was not a wrong value -- it was a policy that existed in only one
of two entry points. ``main.py`` capped OMP/MKL/OPENBLAS to 2 before importing torch; the service
enters through ``uvicorn api.app:create_app``, never ran that code, and loaded BLAS uncapped. Since
the variables are read once at library load and candidate workers are forkserver children, every
worker inherited its entry point's pool permanently. Measured: the capped path's candidate phase ran
1.52x the uncapped path's on identical data and initialisation, 1.30x of it attributable to the cap.

So the behavioural tests below matter less than the two structural ones at the bottom: they are what
stop the asymmetry growing back.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from parallelism import blas_threads  # noqa: E402

# CI's unit lane runs ``pytest -m "unit and not slow" src/tests/unit`` (ci.yml:312-317),
# so a test here with no ``unit`` marker is collected and then silently DESELECTED -- it
# never runs and the job still reports success. Module-level, so it covers every test in
# the file including ones added later.
pytestmark = pytest.mark.unit

SRC = Path(__file__).resolve().parents[2]


class _EnvIsolated(unittest.TestCase):
    """Each test gets a pristine copy of the three variables plus the override."""

    def setUp(self) -> None:
        self._saved = {k: os.environ.get(k) for k in (*blas_threads.BLAS_THREAD_VARS, blas_threads.BLAS_THREADS_ENV)}
        for key in self._saved:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class ConfigureBlasThreadsTest(_EnvIsolated):
    def test_default_caps_all_three_at_two(self) -> None:
        """Unset override caps at 2 (owner decision D1, 2026-09-23).

        Until 2026-09-23 this was a no-op, for #531's reason. The flip was gated on showing that the
        cap does not move the epoch count, which wall time cannot see; see the module docstring.
        """
        self.assertEqual(blas_threads.DEFAULT_BLAS_THREADS, 2)
        self.assertEqual(blas_threads.configure_blas_threads(), "2")
        for var in blas_threads.BLAS_THREAD_VARS:
            self.assertEqual(os.environ[var], "2", f"{var} must carry the default cap")

    def test_blank_override_uses_the_default(self) -> None:
        os.environ[blas_threads.BLAS_THREADS_ENV] = "   "
        self.assertEqual(blas_threads.configure_blas_threads(), "2")
        self.assertEqual(os.environ["OMP_NUM_THREADS"], "2")

    def test_opt_out_leaves_the_runtime_default(self) -> None:
        """``0`` / ``off`` / ``none`` restore the pre-D1 behaviour: set nothing at all."""
        for opt_out in ("0", "off", "OFF", "none", " None "):
            with self.subTest(value=opt_out):
                os.environ[blas_threads.BLAS_THREADS_ENV] = opt_out
                for var in blas_threads.BLAS_THREAD_VARS:
                    os.environ.pop(var, None)
                self.assertIsNone(blas_threads.configure_blas_threads())
                for var in blas_threads.BLAS_THREAD_VARS:
                    self.assertNotIn(var, os.environ, f"{var} must stay unset when the operator opts out")

    def test_default_never_overrides_an_operator_value(self) -> None:
        """setdefault holds for the default too: a width the launcher or operator exported wins.

        juniper-ml's experiment launcher delivers `runtime.blas_threads` by exporting these variables,
        so an unconditional default here would silently override every experiment's named width.
        """
        os.environ["OMP_NUM_THREADS"] = "9"
        self.assertEqual(blas_threads.configure_blas_threads(), "2")
        self.assertEqual(os.environ["OMP_NUM_THREADS"], "9")
        self.assertEqual(os.environ["MKL_NUM_THREADS"], "2")

    def test_override_sets_all_three(self) -> None:
        os.environ[blas_threads.BLAS_THREADS_ENV] = "3"
        self.assertEqual(blas_threads.configure_blas_threads(), "3")
        for var in blas_threads.BLAS_THREAD_VARS:
            self.assertEqual(os.environ[var], "3")

    def test_explicit_operator_value_wins(self) -> None:
        """setdefault semantics: a deployment that already decided is never overridden."""
        os.environ[blas_threads.BLAS_THREADS_ENV] = "3"
        os.environ["OMP_NUM_THREADS"] = "9"
        blas_threads.configure_blas_threads()
        self.assertEqual(os.environ["OMP_NUM_THREADS"], "9")
        self.assertEqual(os.environ["MKL_NUM_THREADS"], "3")

    def test_malformed_override_is_ignored_not_raised(self) -> None:
        """A mistyped tuning knob must not abort a training run before logging even exists.

        It falls back to the documented default. ``0`` is NOT malformed any more: it is the opt-out.
        """
        for bad in ("abc", "-4", "2.5"):
            with self.subTest(value=bad):
                os.environ[blas_threads.BLAS_THREADS_ENV] = bad
                for var in blas_threads.BLAS_THREAD_VARS:
                    os.environ.pop(var, None)
                self.assertEqual(blas_threads.configure_blas_threads(), "2")
                self.assertEqual(os.environ["OMP_NUM_THREADS"], "2")

    def test_module_is_import_cheap(self) -> None:
        """It must import without pulling in a BLAS-linked library.

        The helper's entire job is to run BEFORE numpy/torch load. If importing it dragged one of
        them in, the policy would be applied after the pool was already fixed -- inert, and silently
        so. Checked in a subprocess because this test process has already imported torch.
        """
        code = "import sys; import parallelism.blas_threads; print('torch' in sys.modules, 'numpy' in sys.modules)"
        # nosec B603 - argv is a fixed literal (this interpreter, -c, a constant string); no shell,
        # no user input, no PATH lookup. The subprocess is the point of the test: this process has
        # already imported torch, so import-cheapness can only be observed in a fresh interpreter.
        out = subprocess.run([sys.executable, "-c", code], cwd=SRC, capture_output=True, text=True, timeout=120)  # nosec B603
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "False False", "blas_threads must not import torch or numpy")


class EntryPointParityTest(unittest.TestCase):
    """Structural: BOTH entry points apply the policy, and NEITHER hard-codes a cap (#531)."""

    def test_both_entry_points_call_the_shared_helper(self) -> None:
        for rel in ("main.py", "api/__init__.py"):
            with self.subTest(entry_point=rel):
                text = (SRC / rel).read_text(encoding="utf-8")
                self.assertIn("configure_blas_threads", text, f"{rel} must apply the shared BLAS thread policy")

    def test_no_entry_point_hardcodes_blas_thread_vars(self) -> None:
        """Anti-resurrection: the whole defect was one entry point setting these on its own.

        ``parallelism/blas_threads.py`` is the sole legitimate site. Anything else assigning these
        recreates the asymmetry -- and, in a worker, would be inert as well as wrong.
        """
        offenders = []
        for path in SRC.rglob("*.py"):
            rel = path.relative_to(SRC).as_posix()
            if rel.startswith(("tests/", "backups/")) or rel == "parallelism/blas_threads.py":
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for var in blas_threads.BLAS_THREAD_VARS:
                # Assignment through either os.environ form; a bare mention in a comment is fine.
                if f'os.environ["{var}"]' in text or f'os.environ.setdefault("{var}"' in text:
                    offenders.append(f"{rel}: assigns {var}")
        self.assertEqual(offenders, [], "BLAS thread vars must be set only by parallelism.blas_threads:\n  " + "\n  ".join(offenders))

    def test_worker_still_pins_torch_threads(self) -> None:
        """RC-1's real oversubscription fix must survive this change untouched."""
        text = (SRC / "cascade_correlation/cascade_correlation.py").read_text(encoding="utf-8")
        self.assertIn("_torch.set_num_threads(max(1, worker_thread_count))", text)
        self.assertIn("torch.set_num_threads(parent_thread_count)", text)


if __name__ == "__main__":
    unittest.main()
