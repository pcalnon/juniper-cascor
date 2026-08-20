#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     constants_hdf5.py
# Author:        Paul Calnon
#
# Date Created:  2025-09-14
# Last Modified: 2026-01-12
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
#
# Description:
#    This file contains hdf5 constants used in the Cascade Correlation Neural Network implementation.
#
#####################################################################################################################################################################################################
# Notes:
#
#####################################################################################################################################################################################################
# References:
#
#
#####################################################################################################################################################################################################
# TODO :
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#
#####################################################################################################################################################################################################
import os
import pathlib
import sysconfig
import warnings

# import torch
# import hd5py
# import numpy as np


#####################################################################################################################################################################################################
_HDF5_PROJECT_HDF5_CONSTANTS_DIR = pathlib.Path(__file__).parent.resolve()
_HDF5_PROJECT_CONSTANTS_DIR = _HDF5_PROJECT_HDF5_CONSTANTS_DIR.parent.resolve()
_HDF5_PROJECT_SOURCE_DIR = _HDF5_PROJECT_CONSTANTS_DIR.parent.resolve()
_HDF5_PROJECT_DIR = _HDF5_PROJECT_SOURCE_DIR.parent.resolve()

_HDF5_PROJECT_SNAPSHOTS_DIR_NAME = "cascor-snapshots"


def _hdf5_module_is_installed() -> bool:
    """True when this module was imported from a Python INSTALL tree.

    ``cascor_constants`` is vendored verbatim into the published
    ``juniper-cascor-model`` wheel, so this same file also runs from
    ``site-packages`` inside the distributed worker. There, every path derived
    from ``__file__`` points into the interpreter's own library tree:

        source-relative default   ->  <site-packages>/cascor_snapshots
        repo-root-relative default -> <python-lib>/cascor-snapshots

    Both are wrong, and the second is worse -- it escapes ``site-packages``
    entirely and writes into a directory that may be root-owned and is never
    captured by the project's whole-tree offline backup. The package already had
    to solve this exact class once for the log directory (see
    ``juniper-cascor-model/tests/test_drift.py`` ``_PACKAGE_LOG_DIR_OVERRIDE``).

    ``sysconfig``'s ``purelib`` / ``platlib`` containment is the precise test:
    stdlib-only, no subprocess, and it does not guess from a neighbouring file
    the way a ``pyproject.toml``-adjacency probe would -- that probe answers
    wrongly inside the container, whose runtime stage copies only ``src/``.
    """
    here = pathlib.Path(__file__).resolve()
    for key in ("purelib", "platlib"):
        root = sysconfig.get_paths().get(key)
        if not root:
            continue
        try:
            if here.is_relative_to(pathlib.Path(root).resolve()):
                return True
        except (OSError, ValueError):  # pragma: no cover - defensive
            continue
    return False


# W-6 (CLI experimentation plan §11 / H-5): JUNIPER_CASCOR_SNAPSHOTS_DIR overrides everything below,
# so a per-run launcher can point the direct CLI at its own RUN_DIR/snapshots. Constants resolve at
# import time, so the override must be in the process env before the first cascor_constants import
# (the launcher exports it before exec). A set-but-blank value is treated as unset (the blank-env
# guard class). Shares one env var with the service tier (api/lifecycle/manager.py
# _get_snapshots_dir), which reads it at CALL time.
#
# Unset, in a CHECKOUT, the default is <repo>/cascor-snapshots -- the repo root, and the ONE
# snapshot root shared by every stack origin on that host: this direct CLI, the FastAPI service, and
# the container, which bind-mounts the same host directory. Snapshots are project ASSETS, not
# per-origin scratch; a model saved by one origin is restored and resumed by another. Supersedes
# <repo>/src/cascor_snapshots (this tier's historical root) and <repo>/snapshots (the service's
# short-lived one). A linked git WORKTREE gets its own root, deliberately -- see the design's
# "worktrees are a developer context, not a stack origin".
#
# Unset, from an INSTALLED copy, there is no project root to speak of, so we fall back to the
# working directory and WARN. Every deployed path declares the variable explicitly (compose, the
# systemd unit, the image's own ENV), so reaching this branch means something is unconfigured, and a
# warning that names the variable beats silently writing into the interpreter's library tree.
#
# On the hyphen: it is defence in depth, NOT a structural guarantee. An earlier draft of this comment
# claimed setuptools "can never discover" a hyphenated directory as a package. That is FALSE --
# pyproject's [tool.setuptools.packages.find] defaults to namespaces=True (PEP 420), and
# find_namespace_packages returns "cascor-snapshots" quite happily; a built wheel carries it in
# top_level.txt. The structural fix is `namespaces = false` in pyproject.toml, which is where it now
# lives. The hyphen still buys something real -- it cannot be written as `import cascor_snapshots`,
# and plain find_packages skips it -- so it stays; it is just not the load-bearing part.
_HDF5_PROJECT_SNAPSHOTS_DIR_OVERRIDE = os.environ.get("JUNIPER_CASCOR_SNAPSHOTS_DIR", "").strip()
if _HDF5_PROJECT_SNAPSHOTS_DIR_OVERRIDE:
    _HDF5_PROJECT_SNAPSHOTS_DIR = pathlib.Path(_HDF5_PROJECT_SNAPSHOTS_DIR_OVERRIDE).expanduser()
elif _hdf5_module_is_installed():
    warnings.warn(
        "JUNIPER_CASCOR_SNAPSHOTS_DIR is not set and cascor_constants was imported from an " "installed package, so there is no project root to derive a snapshot directory from. " f"Falling back to {pathlib.Path.cwd() / _HDF5_PROJECT_SNAPSHOTS_DIR_NAME}. Set " "JUNIPER_CASCOR_SNAPSHOTS_DIR to the shared snapshot root for this host.",
        RuntimeWarning,
        stacklevel=2,
    )
    _HDF5_PROJECT_SNAPSHOTS_DIR = pathlib.Path.cwd().joinpath(_HDF5_PROJECT_SNAPSHOTS_DIR_NAME)
else:
    _HDF5_PROJECT_SNAPSHOTS_DIR = pathlib.Path(_HDF5_PROJECT_DIR).joinpath(_HDF5_PROJECT_SNAPSHOTS_DIR_NAME)


# Define HDF5 Storage class Constants to provide reasonable defaults

# Define HDF5Storage Constants for hdf5 file and dataset structure
# _HDF5_STORAGE_HOME_DIR = _GENERATED_DATASETS_HOME_DIR
# _HDF5_STORAGE_ROOT_DIR = _GENERATED_DATASETS_ROOT_DIR
# _HDF5_STORAGE_PARENT_DIR_NAME = _GENERATED_DATASETS_PARENT_DIR_NAME
# _HDF5_STORAGE_PARENT_DIR = _GENERATED_DATASETS_PARENT_DIR
# _HDF5_STORAGE_APPLICATION_DIR_NAME = _GENERATED_DATASETS_APPLICATION_DIR_NAME
# _HDF5_STORAGE_APPLICATION_DIR = _GENERATED_DATASETS_APPLICATION_DIR
# _HDF5_STORAGE_PROJ_DIR_NAME = _GENERATED_DATASETS_PROJ_DIR_NAME
# _HDF5_STORAGE_PROJ_DIR = _GENERATED_DATASETS_PROJ_DIR
# _HDF5_STORAGE_DATA_DIR_NAME = _GENERATED_DATASETS_DATA_DIR_NAME
# _HDF5_STORAGE_DATA_DIR = _GENERATED_DATASETS_DATA_DIR
# _HDF5_STORAGE_IMAGE_DIR_NAME = _GENERATED_DATASETS_IMAGE_DIR_NAME
# _HDF5_STORAGE_IMAGE_DIR = _GENERATED_DATASETS_IMAGE_DIR


# Define HDF5Storage Constants for Logging
# _HDF5_STORAGE_LOG_NAME = _CASCOR_SPIRAL_DATASET_LOG_NAME
# _HDF5_STORAGE_LOG_DATE_FORMAT = _CASCOR_SPIRAL_DATASET_LOG_DATE_FORMAT
# _HDF5_STORAGE_LOG_FORMATTER_STRING = _CASCOR_SPIRAL_DATASET_LOG_FORMATTER_STRING
# _HDF5_STORAGE_LOG_DIR = _CASCOR_SPIRAL_DATASET_LOG_DIR_DEFAULT
# _HDF5_STORAGE_LOG_FILE_PATH = _CASCOR_SPIRAL_DATASET_LOG_FILE_PATH_DEFAULT
# _HDF5_STORAGE_LOG_LEVEL = _CASCOR_SPIRAL_DATASET_LOG_LEVEL_DEFAULT


#####################################################################################################################################################################################################
# HDF5 Snapshot Format Identifiers (snapshots/snapshot_serializer.py)
_HDF5_FORMAT_NAME_CURRENT: str = "juniper.cascor"
_HDF5_FORMAT_NAME_LEGACY: str = "cascor_hdf5_v1"
