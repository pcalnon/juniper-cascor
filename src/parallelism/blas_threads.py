"""Single source of truth for the process-wide BLAS thread policy.

WHY THIS MODULE EXISTS
----------------------
The BLAS thread-count environment variables (``OMP_NUM_THREADS`` and friends) are read **once**,
when the BLAS library is first loaded; setting them afterwards has no effect. Candidate workers are
created from a ``forkserver`` context, so each worker inherits the pool of the process it descends
from and can never resize it. That makes these variables an entry-point-time decision with
permanent, process-tree-wide consequences.

Until this module existed the decision was made in only ONE of the two entry points. ``main.py``
(direct CLI) capped all three to ``2``; ``uvicorn api.app:create_app`` never executed that code and
``src/api/`` set nothing, so the service loaded BLAS with the runtime's own default. Two entry
points into the same trainer therefore ran with different thread pools -- not by configuration, but
by which file the process happened to start in.

The cost was measured, not assumed (juniper-cascor#531): on identical data, identical network
initialisation and an identical config, the capped path's candidate phase ran **1.52x** the
uncapped path's, and the cap accounted for **1.30x** of that. It acted through two channels --
throughput (1.26x -> 1.14x as the budget rose) and, less obviously, **epoch count** (1.21x ->
1.03x), because thread count changes BLAS reduction order, hence floating-point results, hence
where a patience-based candidate early-stopping loop terminates. A tighter cap therefore did not
merely slow each epoch down, it caused *more epochs to be run*.

WHAT THE POLICY IS
------------------
Default: **do nothing**, leaving the runtime's own choice. That is exactly what the service tier has
always done, it is the faster of the two behaviours as measured, and it means every service-tier
result recorded to date remains valid. The direct CLI changes to match it.

Opt in with ``JUNIPER_CASCOR_BLAS_THREADS=<n>`` to cap all three variables. The capability RC-1
(commit ``aa46ad5``) introduced is retained -- it simply stops being an accident of entry point.

WHAT THIS IS *NOT*
------------------
This is not the oversubscription guard. That is RC-1's real fix and it is untouched: each candidate
worker calls ``torch.set_num_threads(max(1, worker_thread_count))`` (``cascade_correlation.py:3873``,
default ``worker_thread_count = 1``) and the parent calls
``torch.set_num_threads(max(2, worker_thread_count * 2))`` (``:1126``). Both run on both paths and
neither depends on these environment variables. The service is the live proof: it has never set them
and its candidate pool runs fully parallel.
"""

from __future__ import annotations

import os
import sys

#: The variables every common BLAS backend reads at load time.
BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")

#: Opt-in override. Unset (or blank) means "leave the runtime's default alone".
BLAS_THREADS_ENV = "JUNIPER_CASCOR_BLAS_THREADS"


def configure_blas_threads() -> str | None:
    """Apply the BLAS thread policy. Call BEFORE importing numpy / torch / scipy.

    Returns the value applied, or ``None`` when the policy is a no-op (the default).

    ``setdefault`` semantics are deliberate: an operator who exports ``OMP_NUM_THREADS`` directly
    still wins, so this never overrides a deployment that has already made the decision.

    A malformed override is reported on stderr and ignored rather than raised. This runs before
    logging is configured and before the application exists; aborting a training run over a
    mistyped tuning knob would be a worse failure than proceeding on the documented default.
    """
    raw = os.environ.get(BLAS_THREADS_ENV, "").strip()
    if not raw:
        return None

    try:
        count = int(raw)
    except ValueError:
        print(f"[cascor] {BLAS_THREADS_ENV}={raw!r} is not an integer -- ignoring, using the BLAS default", file=sys.stderr)
        return None
    if count < 1:
        print(f"[cascor] {BLAS_THREADS_ENV}={raw!r} must be >= 1 -- ignoring, using the BLAS default", file=sys.stderr)
        return None

    value = str(count)
    for var in BLAS_THREAD_VARS:
        os.environ.setdefault(var, value)
    return value
