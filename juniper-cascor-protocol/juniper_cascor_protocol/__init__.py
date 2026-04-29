"""Public surface for juniper-cascor-protocol.

Two subpackages:

- :mod:`juniper_cascor_protocol.envelope` — Pydantic v2 schemas for
  ``/ws/training`` and ``/ws/control``. Loads Pydantic on import.
- :mod:`juniper_cascor_protocol.worker` — :class:`WorkerMessageType`
  StrEnum + numpy-only :class:`BinaryFrame` codec for
  ``/ws/v1/workers``. Does **not** load Pydantic.

Worker consumers should import only from
:mod:`juniper_cascor_protocol.worker` to keep Pydantic out of their
runtime — see the METRICS-MON R2 exit-gate decision in juniper-ml.

The top-level ``juniper_cascor_protocol`` namespace re-exports the
**worker** symbols only by default to preserve the lazy-Pydantic
guarantee. Envelope symbols are reachable via
``from juniper_cascor_protocol.envelope import …`` so the import edge
is explicit at the call site.
"""

from juniper_cascor_protocol._version import __version__
from juniper_cascor_protocol.worker import BinaryFrame, WorkerMessageType

__all__ = ["__version__", "WorkerMessageType", "BinaryFrame"]
