"""Regression tests for Docker defaults and the SEC-F22 bind guard.

The runtime image must not bake in a non-loopback bind without an explicit
operator attestation. Otherwise the application lifespan raises
NonLoopbackBindError immediately and the image crash-loops on a bare
``docker run``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = REPO_ROOT / "Dockerfile"


def _dockerfile_env() -> dict[str, str]:
    """Return simple one-line ``ENV NAME=value`` declarations from Dockerfile."""
    env: dict[str, str] = {}
    for raw_line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("ENV "):
            continue
        _, assignment = line.split(" ", 1)
        if "=" not in assignment:
            continue
        name, value = assignment.split("=", 1)
        env[name.strip()] = value.strip()
    return env


def test_dockerfile_default_host_does_not_trip_bind_guard() -> None:
    """The image default must start without requiring baked-in attestation."""
    env = _dockerfile_env()

    assert env.get("JUNIPER_CASCOR_HOST") == "127.0.0.1"
    assert env.get("JUNIPER_CASCOR_FRONTING_AUTH_ATTESTED", "").lower() not in {"1", "true", "yes", "on"}
