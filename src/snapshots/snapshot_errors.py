#!/usr/bin/env python
"""
Typed exceptions for the snapshot subsystem.

Kept in a dedicated stdlib-only module so API route modules can import the
exception types at module scope without pulling the heavy serializer stack
(h5py / numpy / torch) into their import graph — the serializer itself is
deliberately imported lazily by the lifecycle manager.
"""


class SnapshotSaveError(RuntimeError):
    """A snapshot write failed after the save was attempted.

    C1 (I-3 upstream half — see juniper-ml
    ``notes/JUNIPER_2026-07-11_JUNIPER-CANOPY_TRAINING-RUNTIME-DEFECTS-PLAN.md``):
    ``CascadeHDF5Serializer.save_network`` used to swallow every exception
    into ``False``, which the lifecycle collapsed to ``None`` and the API
    route mapped to a 404 "No network available to snapshot" — a failed
    save masquerading as a missing network. The serializer now raises this
    exception (chaining the underlying cause) so the lifecycle and route
    can propagate the real reason with a correct 500 status, while the
    no-network case keeps its 404.
    """
