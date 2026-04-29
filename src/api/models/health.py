"""Health check response models — re-exported from ``juniper-observability``.

METRICS-MON R2.1.4 / seed-06: the previous in-repo definitions of
:class:`DependencyStatus`, :class:`ReadinessResponse`, and
:func:`probe_dependency` have been promoted into the shared
:mod:`juniper_observability` package so all three Juniper servers
consume one source of truth. This module is preserved as a thin
re-export shim for backwards compatibility — any existing code that
imports ``from api.models.health import DependencyStatus,
ReadinessResponse, probe_dependency`` continues to work unchanged.

The migration **closes BUG-JD-06-equivalent naive-tz drift**: cascor's
former ``timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())``
used local time, while juniper-data's was already tz-aware UTC. The
shared model uses ``datetime.now(UTC).timestamp()`` so all services emit
the same epoch-seconds value regardless of the host timezone.

New code should prefer ``from juniper_observability import …`` to make
the dependency on the shared lib explicit.

See: notes/code-review/METRICS_MONITORING_R2.1_SHARED_OBSERVABILITY_DESIGN_2026-04-28.md
in juniper-ml.
"""

from juniper_observability import DependencyStatus, ReadinessResponse, probe_dependency

__all__ = ["DependencyStatus", "ReadinessResponse", "probe_dependency"]
