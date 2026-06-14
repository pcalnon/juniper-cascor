"""``juniper-cascor-model`` -- the shared CasCor candidate-training core.

Single source of truth for the importable model code a distributed CasCor worker
(``juniper-cascor-worker``) needs to execute a candidate unit: :class:`CandidateUnit`,
the activation registry (:class:`utils.activation.ActivationWithDerivative`), the shared
``utils``, the resilient logging, and the candidate-relevant ``cascor_constants`` -- all
decoupled from the cascor server / training stack (no FastAPI, no ``cascade_correlation``,
no ``api``).

Extracted from ``juniper-cascor`` src under CW-05 (juniper-cascor-worker#97) so the worker
can depend on a single PyPI package instead of needing the cascor source tree on
``sys.path`` via ``--cascor-path``.

Per the migration plan (§3.1 option (i)) the candidate core is shipped under the SAME
top-level package names cascor uses, so the canonical import works verbatim::

    from candidate_unit.candidate_unit import CandidateUnit
    from utils.activation import ActivationWithDerivative

This module deliberately stays lightweight (no torch import at ``import
juniper_cascor_model`` time); import the candidate code from the top-level packages above.
"""

from juniper_cascor_model._version import __version__

__all__ = ["__version__"]
