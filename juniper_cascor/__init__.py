"""Juniper Cascor - Cascade Correlation Neural Network implementation."""

import importlib.metadata

# juniper-cascor#668: resolve the version from the installed distribution's metadata -- the same
# source as ``api.app._API_VERSION`` and ``/v1/health`` (BUG-CC-04) -- instead of restating it.
# The literal this replaces read "0.6.0" while pyproject.toml read 0.11.0, and publish.yml's
# TestPyPI check (``from juniper_cascor import __version__``) printed that stale value on every
# release. The fallback is deliberately NOT a release number: a literal that has to be bumped by
# hand is how the drift happened, and a never-installed source checkout has no release version.
try:
    __version__ = importlib.metadata.version("juniper-cascor")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover - source checkout only
    __version__ = "0.0.0-dev"
__author__ = "Paul Calnon"
