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
Default: **cap all three at 2**, and only where the variable is unset (owner decision D1, ruled
2026-09-23). Opt out with ``JUNIPER_CASCOR_BLAS_THREADS=0`` (or ``off`` / ``none``) to leave the
runtime's own choice, or set it to ``<n>`` to cap at ``n``. An operator who exports
``OMP_NUM_THREADS`` or its siblings directly still wins, so a deployment that has already decided is
never overridden -- and neither is a width the juniper-ml experiment launcher exports from an
experiment's ``runtime.blas_threads``. Both entry points apply it, so it is still never an accident of
entry point (RC-1, commit ``aa46ad5``).

**This reverses the previous default ("do nothing"), which was chosen for #531's reason above.** The
reversal rests on three measurements from the juniper-ml perf lane:

* **The old default was not one width.** The constructor's ``torch.set_num_threads`` pin binds only
  the thread that constructs the network. The service trains on a different thread, which ran its
  INITIAL output pass at the runtime default (16 on the 16-core dev host) until it was re-pinned to
  2 during the first candidate-result collection. Capping that pass cut it by 49.2%
  (``JUNIPER_2026-09-16_JUNIPER-ECOSYSTEM_PERF-LANE-THREAD-WIDTH-SWEEP.md``, §2 correction).
* **#531's wall-time penalty does not reproduce on current code** (same sweep: the environment route
  at width 16 cost about 1%, and widths 2-8 were indistinguishable).
* **The cap does not move the epoch count.** #531's second channel was the COUNT, which wall time
  cannot see, so this flip was gated on measuring it
  (``JUNIPER_2026-09-23_JUNIPER-ECOSYSTEM_PERF-LANE-D1-EPOCH-COUNT-DEBT.md``). Against the old
  default, a cap of 2 reproduced every per-candidate ``epochs_completed``, the winning candidate of
  every phase and the final loss exactly; a seed change did move the counts, so the instrument could
  see a difference. A training thread held at 16 for the whole run DID move a count late in the
  run: the channel is real, and this default steers away from it.

WHAT THIS IS *NOT*
------------------
This is not the oversubscription guard. That is RC-1's real fix and it is untouched: each candidate
worker calls ``torch.set_num_threads(max(1, worker_thread_count))`` (``cascade_correlation.py:4153``,
default ``worker_thread_count = 1``) and the parent calls
``torch.set_num_threads(max(2, worker_thread_count * 2))`` (``:1180``). Both run on both paths and
neither depends on these environment variables.
"""

from __future__ import annotations

import os
import sys

#: The variables every common BLAS backend reads at load time.
BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")

#: Override. Unset (or blank) means DEFAULT_BLAS_THREADS; an OPT_OUT_VALUES entry means "do nothing".
BLAS_THREADS_ENV = "JUNIPER_CASCOR_BLAS_THREADS"

#: The width applied when the override is unset (owner decision D1, 2026-09-23).
DEFAULT_BLAS_THREADS = 2

#: Override values that leave the runtime's own choice alone -- the pre-2026-09-23 default.
OPT_OUT_VALUES = frozenset({"0", "off", "none"})


def configure_blas_threads() -> str | None:
    """Apply the BLAS thread policy. Call BEFORE importing numpy / torch / scipy.

    Returns the value applied to any variable that was unset, or ``None`` when the operator opted
    out with ``JUNIPER_CASCOR_BLAS_THREADS=0`` / ``off`` / ``none``.

    ``setdefault`` semantics are deliberate: an operator who exports ``OMP_NUM_THREADS`` directly
    still wins, so this never overrides a deployment that has already made the decision.

    A malformed override is reported on stderr and ignored rather than raised, and the documented
    default applies. This runs before logging is configured and before the application exists;
    aborting a training run over a mistyped tuning knob would be a worse failure than proceeding on
    the documented default.
    """
    raw = os.environ.get(BLAS_THREADS_ENV, "").strip()
    if raw.lower() in OPT_OUT_VALUES:
        return None

    count = DEFAULT_BLAS_THREADS
    if raw:
        try:
            count = int(raw)
        except ValueError:
            print(f"[cascor] {BLAS_THREADS_ENV}={raw!r} is not an integer -- ignoring, using the default of {DEFAULT_BLAS_THREADS}", file=sys.stderr)
            count = DEFAULT_BLAS_THREADS
        if count < 1:
            print(f"[cascor] {BLAS_THREADS_ENV}={raw!r} must be >= 1 (or 0/off/none to opt out) -- ignoring, using the default of {DEFAULT_BLAS_THREADS}", file=sys.stderr)
            count = DEFAULT_BLAS_THREADS

    value = str(count)
    for var in BLAS_THREAD_VARS:
        os.environ.setdefault(var, value)
    return value
