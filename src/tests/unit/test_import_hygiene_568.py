#!/usr/bin/env python
"""Issue #568/#570: import-hygiene guards for the two worker-bloat routes.

A forkserver candidate worker's module table is built from exactly two import events:
multiprocessing's spawn-preparation re-importing the launcher's ``__main__`` (as
``__mp_main__``), and unpickling the Process target (``import
cascade_correlation.cascade_correlation``). Measured on the pre-fix tree those cost
1,867 and 1,334 modules respectively, carrying matplotlib/PIL, fastapi/pydantic/
starlette/httpx (via the bootstrap Sentry integrations) and a per-worker
``sentry_sdk.init`` into every one of the 7-8 pool workers (cascor#570 evidence
comment, 2026-08-25).

GUARDS (fail on the pre-fix tree): the __mp_main__ probe and the trainer-import probe.
NOT a guard: the normal-import probe — it protects the OTHER direction (the guard must
not fire for ``import main`` / ``python main.py``), and passes on the pre-fix tree too.

Each probe runs in a fresh subprocess so this test's own interpreter state cannot
contaminate the measurement.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SRC = Path(__file__).resolve().parents[2]

_HEAVY = ("matplotlib", "PIL", "fastapi", "pydantic", "starlette", "httpx", "sentry_sdk", "spiral_problem", "cascor_plotter")


def _probe(code: str) -> dict:
    proc = subprocess.run(  # nosec B603 -- fixed interpreter, generated code, no user input
        [sys.executable, "-c", code],
        cwd=_SRC,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"probe failed: rc={proc.returncode}\nstderr tail: {proc.stderr[-2000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_mp_main_reimport_is_lightweight_and_sentryless():
    """The forkserver child's ``__mp_main__`` re-import must not drag the world (GUARD)."""
    report = _probe(
        r"""
import importlib.util, json, sys
sys.argv = ["main.py"]
spec = importlib.util.spec_from_file_location("__mp_main__", "main.py")
m = importlib.util.module_from_spec(spec)
sys.modules["__mp_main__"] = m
spec.loader.exec_module(m)
heavy = [x for x in %r if x in sys.modules]
print(json.dumps({"n": len(sys.modules), "heavy": heavy, "spiral_is_none": m.SpiralProblem is None}))
"""
        % (_HEAVY,)
    )
    assert report["heavy"] == [], f"__mp_main__ re-import dragged heavy packages back in: {report['heavy']}"
    assert report["spiral_is_none"] is True
    # 1,867 pre-fix, 1,142 post-fix; the bound leaves headroom without letting a
    # wholesale regression (any single heavy chain costs 100+) slip through.
    assert report["n"] < 1400, f"__mp_main__ table grew to {report['n']} modules"


def test_trainer_import_is_matplotlib_free():
    """Unpickling the worker target must not import the plotter chain (GUARD)."""
    report = _probe(
        r"""
import json, sys
import cascade_correlation.cascade_correlation  # noqa: F401
heavy = [x for x in %r if x in sys.modules]
print(json.dumps({"n": len(sys.modules), "heavy": heavy}))
"""
        % (_HEAVY,)
    )
    assert report["heavy"] == [], f"trainer import dragged heavy packages: {report['heavy']}"
    assert report["n"] < 1400, f"trainer import table grew to {report['n']} modules"


def test_normal_main_import_keeps_full_surface():
    """``import main`` (tests, tooling) and ``python main.py`` keep the real class.

    NOT a guard for the #568 fix — it fails only if the ``__mp_main__`` gate misfires
    on a normal import, which is the regression in the other direction.
    """
    report = _probe(
        r"""
import json, sys
import main
print(json.dumps({"spiral_real": main.SpiralProblem is not None and main.SpiralProblem.__name__ == "SpiralProblem"}))
"""
    )
    assert report["spiral_real"] is True
