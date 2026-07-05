"""Tests for the SEC-F22 / D2 startup bind-guard.

Covers ``api.app._is_loopback_host`` and ``enforce_fronting_auth_bind_guard``
(the symmetric counterpart to the canopy bind-guard), plus the lifespan wiring
that makes a mis-configured non-loopback bring-up refuse to start.

Design of record: juniper-ml
``notes/JUNIPER_CANOPY_CONTROL_SURFACE_AUTH_AND_NAT_DESIGN_2026-07-03.md``
§4 Option A / §8 D2.
"""

import asyncio
from unittest import mock

import pytest

from api.app import NonLoopbackBindError, _is_loopback_host, create_app, enforce_fronting_auth_bind_guard, lifespan
from api.settings import Settings

pytestmark = pytest.mark.unit


class TestIsLoopbackHost:
    """Robust loopback detection over IPv4/IPv6/hostname forms."""

    @pytest.mark.parametrize(
        "host",
        [
            "127.0.0.1",
            "127.0.0.5",  # anywhere in 127.0.0.0/8
            "::1",
            "localhost",
            "LOCALHOST",  # case-insensitive
            "[::1]",  # bracketed IPv6
            "::ffff:127.0.0.1",  # IPv4-mapped IPv6 loopback
            "127.0.0.1%eth0",  # zone-id stripped
        ],
    )
    def test_loopback_hosts(self, host):
        assert _is_loopback_host(host) is True

    @pytest.mark.parametrize(
        "host",
        [
            "0.0.0.0",  # bind-all is NOT loopback
            "::",  # unspecified IPv6 is NOT loopback
            "192.168.1.100",
            "10.0.0.5",
            "172.23.0.1",  # the HO-3 bridge gateway
            "juniper-cascor",  # a routable container hostname
            "example.com",
            "",  # empty host -> bind-all -> non-loopback (fail-closed)
        ],
    )
    def test_non_loopback_hosts(self, host):
        assert _is_loopback_host(host) is False


class TestBindGuardFunction:
    """``enforce_fronting_auth_bind_guard`` refuse/allow matrix (D2)."""

    @pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
    def test_loopback_always_starts(self, host):
        # Loopback binds always start, regardless of the attestation flag.
        enforce_fronting_auth_bind_guard(Settings(host=host, fronting_auth_attested=False))
        enforce_fronting_auth_bind_guard(Settings(host=host, fronting_auth_attested=True))

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.100"])
    def test_non_loopback_without_attest_refuses(self, host):
        with pytest.raises(NonLoopbackBindError):
            enforce_fronting_auth_bind_guard(Settings(host=host, fronting_auth_attested=False))

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.100"])
    def test_non_loopback_with_attest_starts(self, host):
        # Attestation is the operator's assertion that a fronting auth layer
        # fronts the port; the guard then allows the non-loopback bind.
        enforce_fronting_auth_bind_guard(Settings(host=host, fronting_auth_attested=True))

    def test_direct_uvicorn_host_override_is_guarded(self, monkeypatch):
        # Direct uvicorn startup can bind a public socket even when Settings.host
        # remains at its loopback default. The guard must inspect the explicit
        # CLI --host override so the documented production command cannot bypass
        # SEC-F22 by accident.
        monkeypatch.setattr(
            "sys.argv",
            [
                "uvicorn",
                "api.app:create_app",
                "--factory",
                "--host",
                "0.0.0.0",
                "--port",
                "8200",
            ],
        )

        with pytest.raises(NonLoopbackBindError):
            enforce_fronting_auth_bind_guard(Settings(host="127.0.0.1", fronting_auth_attested=False))

        enforce_fronting_auth_bind_guard(Settings(host="127.0.0.1", fronting_auth_attested=True))

    def test_default_settings_start(self):
        # The class default host (127.0.0.1) must never trip the guard.
        enforce_fronting_auth_bind_guard(Settings())

    def test_refusal_logs_critical(self):
        # Assert the guard is LOUD (D2 fail-closed + loud) by patching the
        # module logger directly — immune to whatever global logging.disable /
        # configure_logging state an earlier test in the session may have left.
        import api.app as app_module

        with mock.patch.object(app_module.logger, "critical") as mock_critical:
            with pytest.raises(NonLoopbackBindError):
                enforce_fronting_auth_bind_guard(Settings(host="0.0.0.0", fronting_auth_attested=False))

        assert mock_critical.called
        assert "REFUSING TO START" in mock_critical.call_args[0][0]


class TestBindGuardLifespanWiring:
    """The guard is actually invoked at application startup (before bind)."""

    def test_lifespan_refuses_non_loopback_without_attest(self):
        # Building the app is fine; STARTING it (entering the lifespan) must
        # raise before any socket bind or background thread is created.
        app = create_app(Settings(host="0.0.0.0", auto_start=False, fronting_auth_attested=False))

        async def _enter() -> None:
            async with lifespan(app):
                pass  # pragma: no cover — guard raises before yield

        with pytest.raises(NonLoopbackBindError):
            asyncio.run(_enter())

    def test_create_app_itself_does_not_raise(self):
        # create_app must not raise on a non-loopback host — the guard fires at
        # startup (lifespan), not at construction, so the object is still built.
        app = create_app(Settings(host="0.0.0.0", auto_start=False, fronting_auth_attested=False))
        assert app.state.settings.host == "0.0.0.0"
