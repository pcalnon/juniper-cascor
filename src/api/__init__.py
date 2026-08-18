"""JuniperCascor Service API.

The BLAS thread policy is applied HERE, and it has to be here rather than in ``api.app``: the
variables are read once when BLAS first loads, ``api.app`` imports ``torch`` at module level, and
this package's ``__init__`` is what runs before it. Placing the call any later would be silently
inert -- the same failure mode that made the setting entry-point-dependent in the first place
(juniper-cascor#531).

The direct CLI applies the identical policy from the identical helper at the top of ``main.py``.
Default is a no-op, which is what this tier has always done; ``JUNIPER_CASCOR_BLAS_THREADS`` opts
into a cap on both paths at once.
"""

from parallelism.blas_threads import configure_blas_threads

configure_blas_threads()
