#!/usr/bin/env python
"""
Run provenance for snapshots (D-C) — which run, experiment, and dataset made this model.

A snapshot has always recorded *what* it is (architecture, parameters, library versions)
and never *where it came from*. ``meta`` carries a ``uuid``, but that identifies the
file, not the work. So a question as ordinary as "find the model from the E-I cap-128
cell" was unanswerable across an archive of ~27.9k files, and **identity has to precede
retention** — a deletion rule over anonymous artifacts is guesswork.

The identity itself is not invented here. The experiment layer already computes every
field (``run_id``, ``experiment``, ``cell_id``, ``dataset_id``) and git supplies the
SHA; D-C is a *propagation* problem, not a modelling one.

Transport is process env, mirroring ``JUNIPER_CASCOR_SNAPSHOTS_DIR`` — the launcher
already exports per-run configuration that way and both the direct CLI and the service
honour it, so provenance needs no new channel and no API change.

Read at CALL time, never import time: a long-lived process must see a value exported
after it started, and tests must be able to set one per-case. A set-but-blank value is
treated as unset (the ecosystem's blank-env guard class).

**Absence is meaningful and must stay legal.** Every snapshot written before this
existed has no ``provenance`` group, and the entire archive would fail to load if the
group were required. ``_validate_format`` deliberately does not list it.
"""

import os
from typing import Any, Dict, Optional

from .snapshot_common import read_str_attr, write_str_attr

#: Bumped when the field set changes, so a reader can tell which fields to expect
#: rather than inferring from what happens to be present.
PROVENANCE_SCHEMA_VERSION = "1"

#: HDF5 group name. Top-level and separate from ``meta`` so identity does not mix with
#: training bookkeeping (patience counters, library versions) and can version itself.
PROVENANCE_GROUP = "provenance"

#: Field -> environment variable. The field names are the experiment layer's own.
PROVENANCE_ENV = {
    "run_id": "JUNIPER_CASCOR_RUN_ID",
    "experiment": "JUNIPER_CASCOR_EXPERIMENT",
    "cell_id": "JUNIPER_CASCOR_CELL_ID",
    "dataset_id": "JUNIPER_CASCOR_DATASET_ID",
    "git_sha": "JUNIPER_CASCOR_GIT_SHA",
}

PROVENANCE_FIELDS = tuple(PROVENANCE_ENV)


def capture_from_env(environ: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Collect whatever run identity this process was launched with.

    Args:
        environ: Mapping to read; defaults to ``os.environ``. Injectable so tests do
            not have to mutate global state.

    Returns:
        Only the fields that are actually set, so a partially-identified run records
        what it knows instead of a row of empty strings. An unidentified run yields
        ``{}``, and no group is written at all.
    """
    source = os.environ if environ is None else environ
    captured = {}
    for field, env_var in PROVENANCE_ENV.items():
        value = (source.get(env_var) or "").strip()
        if value:
            captured[field] = value
    return captured


def write_provenance(hdf5_file: Any, provenance: Optional[Dict[str, str]] = None) -> bool:
    """Write the ``provenance`` group, if there is anything to record.

    Writing nothing when nothing is known is deliberate: an empty group would make an
    unidentified snapshot indistinguishable from one whose provenance failed to write.

    Returns:
        True if a group was written.
    """
    captured = capture_from_env() if provenance is None else dict(provenance)
    if not captured:
        return False
    group = hdf5_file.require_group(PROVENANCE_GROUP)
    write_str_attr(group, "schema_version", PROVENANCE_SCHEMA_VERSION)
    for field, value in captured.items():
        write_str_attr(group, field, value)
    return True


def read_provenance(hdf5_file: Any) -> Optional[Dict[str, str]]:
    """Read the ``provenance`` group back.

    Returns:
        The recorded fields (including ``schema_version``), or ``None`` when the
        snapshot predates D-C or was written by an unidentified run. ``None`` means
        *unknown*, which is a real answer — not a failure.
    """
    if PROVENANCE_GROUP not in hdf5_file:
        return None
    group = hdf5_file[PROVENANCE_GROUP]
    recovered = {}
    for key in ("schema_version", *PROVENANCE_FIELDS):
        value = read_str_attr(group, key)
        if value:
            recovered[key] = value
    return recovered or None
