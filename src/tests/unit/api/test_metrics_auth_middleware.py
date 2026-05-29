"""Regression tests for cascor's ``MetricsAuthMiddleware`` (POC §3.1).

Mirrors the juniper-data ``TestMetricsAuthMiddlewareCIDR`` suite so the
two services share a contract: bare IP literals, CIDR ranges, IPv6 zone
strip, IPv4-mapped IPv6 unwrap, fail-loud invalid config, and the
``/metrics in EXEMPT_PATHS`` invariant. Promotion of the helper to
``juniper-observability`` will collapse both suites into a single
shared one (roadmap §R5).

Tests exercise ``MetricsAuthMiddleware`` directly via ASGI scopes
(instead of through ``create_app`` + ``TestClient``) because cascor's
full app lifespan does not tear down cleanly under ``TestClient`` in
a unit-test context (worker registry monitor, lifecycle manager, etc).
The middleware contract is the same either way: 200 from the wrapped
app when the client IP matches, 403 from the middleware when it
doesn't, no fallthrough either way.
"""

from __future__ import annotations

from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers — minimal ASGI driver for ``MetricsAuthMiddleware``
# ---------------------------------------------------------------------------


async def _stub_app(scope, receive, send) -> None:
    """ASGI stub that responds 200 with body ``b"metrics-body"``."""
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain; charset=utf-8")],
        }
    )
    await send({"type": "http.response.body", "body": b"metrics-body"})


class _Captured:
    """Collect the ASGI ``send`` messages so tests can introspect them."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def send(self, message: dict[str, Any]) -> None:
        self.messages.append(message)


async def _empty_receive() -> dict[str, Any]:  # pragma: no cover — never called by /metrics
    return {"type": "http.disconnect"}


def _scope(client_ip: str | None = "127.0.0.1") -> dict[str, Any]:
    """Minimal HTTP scope for the metrics-auth middleware."""
    return {
        "type": "http",
        "method": "GET",
        "path": "/metrics",
        "headers": [],
        "client": (client_ip, 12345) if client_ip is not None else None,
    }


async def _drive(middleware, scope) -> _Captured:
    """Run a single request through the middleware, return captured messages."""
    captured = _Captured()
    await middleware(scope, _empty_receive, captured.send)
    return captured


def _status_of(captured: _Captured) -> int:
    """Pull the HTTP status out of the captured ``http.response.start``."""
    start = next(m for m in captured.messages if m["type"] == "http.response.start")
    return int(start["status"])


# ---------------------------------------------------------------------------
# EXEMPT_PATHS invariant
# ---------------------------------------------------------------------------


def test_metrics_in_exempt_paths() -> None:
    """``/metrics`` must be in ``EXEMPT_PATHS`` so SecurityMiddleware does
    not 401 prometheus before ``MetricsAuthMiddleware`` ever sees the
    request. Without this exempt, the IP allowlist is dead code on any
    deployment that has ``api_keys`` set."""
    from api.middleware import EXEMPT_PATHS

    assert "/metrics" in EXEMPT_PATHS
    assert "/metrics/" in EXEMPT_PATHS


# ---------------------------------------------------------------------------
# CIDR + IPv6 normalization in ``MetricsAuthMiddleware``
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMetricsAuthMiddlewareCIDR:
    """``MetricsAuthMiddleware`` accepts CIDR ranges, IPv6 zone-ids, and
    IPv4-mapped IPv6. Pins the contract documented in
    ``notes/poc/POC_REMEDIATION_PLAN_2026-05-27.md`` §3.1.
    """

    async def test_cidr_v4_match_allows_request(self) -> None:
        from api.observability import MetricsAuthMiddleware

        middleware = MetricsAuthMiddleware(_stub_app, trusted_ips=["172.18.0.0/16"])
        captured = await _drive(middleware, _scope(client_ip="172.18.0.5"))
        assert _status_of(captured) == 200

    async def test_cidr_v4_miss_rejects_request(self) -> None:
        from api.observability import MetricsAuthMiddleware

        middleware = MetricsAuthMiddleware(_stub_app, trusted_ips=["172.18.0.0/16"])
        captured = await _drive(middleware, _scope(client_ip="10.0.0.5"))
        assert _status_of(captured) == 403

    async def test_mixed_cidr_and_literal_entries_allowed(self) -> None:
        from api.observability import MetricsAuthMiddleware

        # Literal address (no CIDR suffix) widens to /32; exact match wins.
        middleware = MetricsAuthMiddleware(_stub_app, trusted_ips=["172.18.0.0/16", "10.0.0.99"])
        captured = await _drive(middleware, _scope(client_ip="10.0.0.99"))
        assert _status_of(captured) == 200

    async def test_cidr_v6_match_allows_request(self) -> None:
        from api.observability import MetricsAuthMiddleware

        middleware = MetricsAuthMiddleware(_stub_app, trusted_ips=["fd00::/8"])
        captured = await _drive(middleware, _scope(client_ip="fd12::1"))
        assert _status_of(captured) == 200

    async def test_ipv4_mapped_ipv6_against_ipv4_cidr(self) -> None:
        """Regression for the docker-bridge IPv4-mapped IPv6 case:
        ``::ffff:172.18.0.5`` must be unwrapped to ``172.18.0.5`` before
        CIDR membership."""
        from api.observability import MetricsAuthMiddleware

        middleware = MetricsAuthMiddleware(_stub_app, trusted_ips=["172.18.0.0/16"])
        captured = await _drive(middleware, _scope(client_ip="::ffff:172.18.0.5"))
        assert _status_of(captured) == 200

    async def test_ipv6_zone_id_is_stripped(self) -> None:
        """``fe80::1%eth0`` would be rejected by ``ip_address`` without the
        zone-id strip; with it, the address parses and membership works."""
        from api.observability import MetricsAuthMiddleware

        middleware = MetricsAuthMiddleware(_stub_app, trusted_ips=["fe80::/10"])
        captured = await _drive(middleware, _scope(client_ip="fe80::1%eth0"))
        assert _status_of(captured) == 200

    async def test_default_loopback_allowlist_still_works(self) -> None:
        """Backward-compat: ``None`` defaults to ``("127.0.0.1", "::1")``
        and 127.0.0.1 resolves."""
        from api.observability import MetricsAuthMiddleware

        middleware = MetricsAuthMiddleware(_stub_app, trusted_ips=None)
        captured = await _drive(middleware, _scope(client_ip="127.0.0.1"))
        assert _status_of(captured) == 200

    async def test_malformed_client_address_falls_through_to_403(self) -> None:
        """``"not-an-ip"`` (TestClient's default ``"testclient"``-style host)
        must never accidentally match a CIDR; the ``except ValueError``
        path in ``__call__`` keeps ``allowed = False``."""
        from api.observability import MetricsAuthMiddleware

        middleware = MetricsAuthMiddleware(_stub_app, trusted_ips=["0.0.0.0/0"])
        captured = await _drive(middleware, _scope(client_ip="not-an-ip"))
        assert _status_of(captured) == 403

    async def test_missing_client_address_falls_through_to_403(self) -> None:
        """Scope without a ``client`` entry — defensive coverage."""
        from api.observability import MetricsAuthMiddleware

        middleware = MetricsAuthMiddleware(_stub_app, trusted_ips=["127.0.0.1"])
        captured = await _drive(middleware, _scope(client_ip=None))
        assert _status_of(captured) == 403


# ---------------------------------------------------------------------------
# Settings-side fail-loud validation
# ---------------------------------------------------------------------------


def test_invalid_cidr_raises_at_settings_construction() -> None:
    """Fail-loud: a typo like ``172.18.0.0/164`` must surface at
    ``Settings()`` time, not on the first scrape (when it would silently
    403). Mirrors juniper-data's validator."""
    import pydantic_core

    from api.settings import Settings

    with pytest.raises((ValueError, pydantic_core.ValidationError)):
        Settings(
            metrics_enabled=True,
            metrics_trusted_ips=["172.18.0.0/164"],
        )


def test_middleware_raises_on_invalid_cidr_at_init() -> None:
    """If the Settings validator were ever bypassed, the middleware itself
    fails loud on construction too — defense in depth."""
    from api.observability import MetricsAuthMiddleware

    with pytest.raises(ValueError):
        MetricsAuthMiddleware(_stub_app, trusted_ips=["172.18.0.0/164"])
