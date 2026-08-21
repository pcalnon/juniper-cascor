#!/usr/bin/env python
"""
Shared vocabulary for *why* a snapshot load failed (D-B).

Before this module the loader collapsed every failure into a bare ``None``, and the
API collapsed that into ``404 "not found or failed to load"`` — fusing two opposite
operator situations: *pick a different snapshot* and *investigate data loss*.

The distinction can only be drawn where the cause is still known — at the load site —
so the taxonomy lives in one place that the serializer, the lifecycle manager, and the
API routes can all name. Design of record:
``notes/JUNIPER_2026-08-20_JUNIPER-CASCOR_SNAPSHOT-ERROR-TAXONOMY-DESIGN.md``
(juniper-ml#1193).
"""

from dataclasses import dataclass
from typing import Any, Optional

#: The load succeeded.
SNAPSHOT_OK = "ok"

#: No snapshot with that id exists. The operator should pick a different one.
#: Maps to HTTP 404.
SNAPSHOT_ABSENT = "snapshot_absent"

#: A snapshot exists but cannot be read: bad format, missing groups, an unreadable
#: file, or a config group that yields no network. The operator should investigate
#: data loss. Maps to HTTP 422 — the request is well formed, the entity is not
#: processable. Deliberately NOT 404 (a lie for a file that exists), NOT 500 (implies
#: a retry might help), and NOT 409 (already used by these routes for FSM conflicts).
SNAPSHOT_CORRUPT = "snapshot_corrupt"

#: A snapshot is internally readable but describes a DIFFERENT network than it
#: declares — its ``arch`` group and its parameter tensors agree with each other while
#: disagreeing with the ``config`` group the network is actually built from (D-E §3).
#: Distinguished from ``SNAPSHOT_CORRUPT`` because it points at a different
#: investigation: nothing is damaged, the file is self-inconsistent about its own
#: dimensions. Also maps to HTTP 422.
SNAPSHOT_ARCH_MISMATCH = "snapshot_arch_mismatch"

#: Statuses that mean "the file is there, but we will not load it" — i.e. every
#: failure that is NOT simply absent. Route-level status mapping keys off this.
SNAPSHOT_UNUSABLE = (SNAPSHOT_CORRUPT, SNAPSHOT_ARCH_MISMATCH)


@dataclass(frozen=True)
class SnapshotLoadResult:
    """Outcome of a snapshot load, carrying the reason when it failed.

    ``__bool__`` follows ``ok`` so the pre-existing ``if not loaded:`` call sites read
    unchanged after being handed one of these instead of a bare ``bool``.
    """

    network: Optional[Any] = None
    status: str = SNAPSHOT_OK
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == SNAPSHOT_OK and self.network is not None

    def __bool__(self) -> bool:
        return self.ok


def absent(detail: str) -> SnapshotLoadResult:
    """A snapshot with that id does not exist."""
    return SnapshotLoadResult(network=None, status=SNAPSHOT_ABSENT, detail=detail)


def corrupt(detail: str) -> SnapshotLoadResult:
    """A snapshot exists but cannot be read."""
    return SnapshotLoadResult(network=None, status=SNAPSHOT_CORRUPT, detail=detail)


def arch_mismatch(detail: str) -> SnapshotLoadResult:
    """A snapshot's declared architecture disagrees with the network it builds."""
    return SnapshotLoadResult(network=None, status=SNAPSHOT_ARCH_MISMATCH, detail=detail)


def loaded(network: Any) -> SnapshotLoadResult:
    """The load succeeded."""
    return SnapshotLoadResult(network=network, status=SNAPSHOT_OK, detail="")
