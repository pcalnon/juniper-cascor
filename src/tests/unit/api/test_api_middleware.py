"""Unit tests for SecurityMiddleware, SecurityHeadersMiddleware, and RequestBodyLimitMiddleware."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware import EXEMPT_PATHS, RequestBodyLimitMiddleware, SecurityHeadersMiddleware, SecurityMiddleware
from api.security import APIKeyAuth, FailedAuthThrottle, RateLimiter, build_failed_auth_throttle


@pytest.fixture
def app_with_middleware():
    """Create a FastAPI app with security middleware."""

    def _create(api_keys=None, rate_limit_enabled=False, rpm=60, throttle=None):
        app = FastAPI()
        auth = APIKeyAuth(api_keys)
        limiter = RateLimiter(requests_per_minute=rpm, enabled=rate_limit_enabled)
        # ``throttle=None`` deliberately exercises the production default (an enabled
        # FailedAuthThrottle at the library budget), so the pre-existing arms below prove the
        # default is transparent to well-behaved traffic.
        app.add_middleware(SecurityMiddleware, api_key_auth=auth, rate_limiter=limiter, failed_auth_throttle=throttle)

        @app.get("/v1/health")
        async def health():
            return {"status": "ok"}

        @app.get("/v1/network")
        async def network():
            return {"data": []}

        return app

    return _create


@pytest.fixture
def headers_app():
    """Minimal app with SecurityHeadersMiddleware only."""

    def _create(content_security_policy=None):
        app = FastAPI()
        if content_security_policy is None:
            app.add_middleware(SecurityHeadersMiddleware)
        else:
            app.add_middleware(SecurityHeadersMiddleware, content_security_policy=content_security_policy)

        @app.get("/v1/health")
        async def health():
            return {"status": "ok"}

        return app

    return _create


@pytest.mark.unit
class TestSecurityHeadersMiddleware:
    """Always-on response header contract (clickjacking / MIME / CSP / HSTS)."""

    def test_baseline_security_headers_present(self, headers_app):
        client = TestClient(headers_app())
        response = client.get("/v1/health")
        assert response.status_code == 200
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert response.headers["Permissions-Policy"] == "camera=(), microphone=(), geolocation=()"
        assert response.headers["Content-Security-Policy"] == "default-src 'none'; frame-ancestors 'none'"

    def test_no_hsts_without_forwarded_proto(self, headers_app):
        client = TestClient(headers_app())
        response = client.get("/v1/health")
        assert "Strict-Transport-Security" not in response.headers

    def test_hsts_when_forwarded_proto_https(self, headers_app):
        client = TestClient(headers_app())
        response = client.get("/v1/health", headers={"X-Forwarded-Proto": "https"})
        assert response.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"

    def test_no_hsts_when_forwarded_proto_http(self, headers_app):
        client = TestClient(headers_app())
        response = client.get("/v1/health", headers={"X-Forwarded-Proto": "http"})
        assert "Strict-Transport-Security" not in response.headers

    def test_custom_content_security_policy(self, headers_app):
        custom = "default-src 'self'; frame-ancestors 'none'"
        client = TestClient(headers_app(content_security_policy=custom))
        response = client.get("/v1/health")
        assert response.headers["Content-Security-Policy"] == custom


@pytest.mark.unit
class TestSecurityMiddleware:
    def test_exempt_path_bypasses_security(self, app_with_middleware):
        app = app_with_middleware(api_keys=["secret"])
        client = TestClient(app)
        response = client.get("/v1/health")
        assert response.status_code == 200

    def test_auth_required_returns_401(self, app_with_middleware):
        app = app_with_middleware(api_keys=["secret"])
        client = TestClient(app)
        response = client.get("/v1/network")
        assert response.status_code == 401

    def test_invalid_key_returns_401(self, app_with_middleware):
        app = app_with_middleware(api_keys=["secret"])
        client = TestClient(app)
        response = client.get("/v1/network", headers={"X-API-Key": "wrong"})
        assert response.status_code == 401

    def test_valid_key_passes(self, app_with_middleware):
        app = app_with_middleware(api_keys=["secret"])
        client = TestClient(app)
        response = client.get("/v1/network", headers={"X-API-Key": "secret"})
        assert response.status_code == 200

    def test_rate_limit_exceeded_returns_429(self, app_with_middleware):
        app = app_with_middleware(rate_limit_enabled=True, rpm=2)
        client = TestClient(app)
        for _ in range(2):
            client.get("/v1/network")
        response = client.get("/v1/network")
        assert response.status_code == 429

    def test_rate_limit_headers_included(self, app_with_middleware):
        app = app_with_middleware(rate_limit_enabled=True, rpm=10)
        client = TestClient(app)
        response = client.get("/v1/network")
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers

    def test_exempt_paths_set(self):
        assert "/v1/health" in EXEMPT_PATHS
        assert "/v1/health/live" in EXEMPT_PATHS
        assert "/v1/health/ready" in EXEMPT_PATHS
        assert "/docs" in EXEMPT_PATHS
        assert "/openapi.json" in EXEMPT_PATHS
        assert "/redoc" in EXEMPT_PATHS
        assert "/v1/network" not in EXEMPT_PATHS


@pytest.mark.unit
class TestSecurityMiddlewareAuthRateLimitInterplay:
    """Auth-first + rate-limit keying contracts on SecurityMiddleware.

    These pin the interplay that the happy-path middleware tests do not:
    failed auth must not burn rate-limit budget, 429 must preserve Retry-After
    headers through the middleware JSONResponse path, distinct API keys get
    independent budgets, and exempt paths stay free under a saturated limiter.
    """

    def test_failed_auth_does_not_increment_rate_limit(self, app_with_middleware):
        """Missing/invalid keys return 401 before RateLimiter.check runs.

        Swapping auth/rate-limit order would let an attacker exhaust a shared
        IP budget with forged keys, then lock out a legitimate principal.

        The failed-auth throttle is given a deliberately generous budget so that this test keeps
        measuring what it was written to measure -- the *identity-keyed limiter's* counters --
        rather than tripping the pre-auth throttle. Mirrors the inverse move in
        ``juniper-service-core``'s own harness, which uses a generous ``requests_per_minute`` so
        any 429 it sees must have come from the throttle. The 10 failed attempts below are
        exactly the throttle's default budget, so without this the final valid request is
        throttled at the door; APD-CASCOR-004 coverage lives in
        ``TestFailedAuthThrottleIntegration``.
        """
        app = app_with_middleware(
            api_keys=["secret"],
            rate_limit_enabled=True,
            rpm=2,
            throttle=build_failed_auth_throttle(max_failures=100, window_seconds=60),
        )
        # Reach into the middleware instance to inspect counters after failed auth.
        middleware = next(m for m in app.user_middleware if m.cls is SecurityMiddleware)
        limiter: RateLimiter = middleware.kwargs["rate_limiter"]
        client = TestClient(app)

        for _ in range(5):
            assert client.get("/v1/network").status_code == 401
            assert client.get("/v1/network", headers={"X-API-Key": "wrong"}).status_code == 401

        assert limiter._counters == {}, "failed auth must not create rate-limit counters"

        # Legitimate key still has a full budget after the auth failures.
        assert client.get("/v1/network", headers={"X-API-Key": "secret"}).status_code == 200
        assert client.get("/v1/network", headers={"X-API-Key": "secret"}).status_code == 200
        assert client.get("/v1/network", headers={"X-API-Key": "secret"}).status_code == 429

    def test_valid_key_429_preserves_retry_after_headers(self, app_with_middleware):
        """429 from RateLimiter must keep Retry-After / X-RateLimit-* on the wire.

        SecurityMiddleware catches HTTPException and rebuilds JSONResponse —
        dropping ``exc.headers`` would silently strip Retry-After for clients.
        """
        app = app_with_middleware(api_keys=["secret"], rate_limit_enabled=True, rpm=1)
        client = TestClient(app)
        headers = {"X-API-Key": "secret"}
        assert client.get("/v1/network", headers=headers).status_code == 200
        response = client.get("/v1/network", headers=headers)
        assert response.status_code == 429
        assert response.headers["X-RateLimit-Limit"] == "1"
        assert response.headers["X-RateLimit-Remaining"] == "0"
        assert "X-RateLimit-Reset" in response.headers
        assert "Retry-After" in response.headers
        assert "Rate limit exceeded" in response.json()["detail"]

    def test_distinct_api_keys_have_independent_budgets(self, app_with_middleware):
        """Rate limiting keys on the authenticated API key, not a shared IP bucket."""
        app = app_with_middleware(api_keys=["key-a", "key-b"], rate_limit_enabled=True, rpm=1)
        client = TestClient(app)

        assert client.get("/v1/network", headers={"X-API-Key": "key-a"}).status_code == 200
        assert client.get("/v1/network", headers={"X-API-Key": "key-a"}).status_code == 429
        # key-b must still be admitted on the same client IP.
        assert client.get("/v1/network", headers={"X-API-Key": "key-b"}).status_code == 200
        assert client.get("/v1/network", headers={"X-API-Key": "key-b"}).status_code == 429

    def test_auth_disabled_rate_limit_keys_by_ip(self, app_with_middleware):
        """Open-auth + rate-limit-on falls back to ip:… keys (dev posture)."""
        app = app_with_middleware(api_keys=None, rate_limit_enabled=True, rpm=2)
        middleware = next(m for m in app.user_middleware if m.cls is SecurityMiddleware)
        limiter: RateLimiter = middleware.kwargs["rate_limiter"]
        client = TestClient(app)

        assert client.get("/v1/network").status_code == 200
        assert client.get("/v1/network").status_code == 200
        assert client.get("/v1/network").status_code == 429
        assert any(key.startswith("ip:") for key in limiter._counters), limiter._counters

    def test_exempt_path_ignores_saturated_rate_limit(self, app_with_middleware):
        """Health probes must stay reachable even after a non-exempt 429."""
        app = app_with_middleware(api_keys=["secret"], rate_limit_enabled=True, rpm=1)
        client = TestClient(app)
        headers = {"X-API-Key": "secret"}

        assert client.get("/v1/network", headers=headers).status_code == 200
        assert client.get("/v1/network", headers=headers).status_code == 429
        # Exempt path bypasses both auth and rate-limit entirely.
        assert client.get("/v1/health").status_code == 200
        assert client.get("/v1/health").status_code == 200

    def test_empty_api_keys_list_disables_auth_like_none(self, app_with_middleware):
        """api_keys=[] is open-access (same as None) — rate-limit still applies by IP."""
        app = app_with_middleware(api_keys=[], rate_limit_enabled=True, rpm=1)
        client = TestClient(app)
        assert client.get("/v1/network").status_code == 200
        assert client.get("/v1/network").status_code == 429


@pytest.fixture
def body_limit_app():
    """Create a FastAPI app with a tiny body limit for testing (CR-024)."""
    app = FastAPI()
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=1024)

    @app.post("/echo")
    async def echo(payload: dict):
        return {"received_bytes": len(str(payload))}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


@pytest.mark.unit
class TestRequestBodyLimitMiddleware:
    """Regression tests for CR-024: body limit enforcement including
    chunked transfer encoding and lying Content-Length headers."""

    def test_small_body_accepted(self, body_limit_app):
        client = TestClient(body_limit_app)
        response = client.post("/echo", json={"a": "b"})
        assert response.status_code == 200

    def test_get_request_not_affected(self, body_limit_app):
        client = TestClient(body_limit_app)
        response = client.get("/health")
        assert response.status_code == 200

    def test_declared_content_length_over_limit_rejected_early(self, body_limit_app):
        """Fast-path: Content-Length header exceeds limit → immediate 413."""
        client = TestClient(body_limit_app)
        # Send a body that fits; lie about Content-Length being huge.
        # httpx computes Content-Length from the actual body, so to exercise
        # the fast path we hand-craft it.
        response = client.post(
            "/echo",
            content=b'{"a":"b"}',
            headers={"Content-Length": str(10 * 1024)},
        )
        assert response.status_code == 413

    def test_invalid_content_length_rejected(self, body_limit_app):
        client = TestClient(body_limit_app)
        response = client.post(
            "/echo",
            content=b'{"a":"b"}',
            headers={"Content-Length": "not-a-number"},
        )
        assert response.status_code == 400

    def test_actual_body_over_limit_rejected(self, body_limit_app):
        """Stream-read enforcement: body exceeds limit even when the client
        writes it in one shot (Content-Length will be set by httpx to the
        real size)."""
        client = TestClient(body_limit_app)
        oversized = '{"x":"' + ("A" * 2000) + '"}'  # > 1024 byte limit
        response = client.post("/echo", content=oversized)
        assert response.status_code == 413

    def test_chunked_body_under_limit_accepted(self, body_limit_app):
        """Chunked transfer with small body still works end-to-end."""
        client = TestClient(body_limit_app)

        # httpx sends as chunked when given a generator
        def body_gen():
            yield b'{"a":'
            yield b'"b"}'

        # ``content=`` doesn't auto-set Content-Type; FastAPI/Pydantic body
        # parsing for the ``payload: dict`` annotation only invokes JSON
        # parsing when Content-Type is application/json (since FastAPI 0.100+
        # / Starlette 1.x). Without this header the raw JSON string would be
        # passed to Pydantic and rejected with 422 ("Input should be a valid
        # dictionary") regardless of what the body-limit middleware does.
        response = client.post("/echo", content=body_gen(), headers={"Content-Type": "application/json"})
        assert response.status_code == 200

    def test_chunked_body_over_limit_rejected(self, body_limit_app):
        """CR-024 regression: a streaming/chunked client sending more than
        the limit must be rejected by the middleware's per-chunk
        accumulation, NOT by running out of memory. No Content-Length
        header is emitted when the body is a generator."""
        client = TestClient(body_limit_app)

        def body_gen():
            # Emit several 512-byte chunks → total > 1024 byte limit.
            for _ in range(5):
                yield b"A" * 512

        response = client.post("/echo", content=body_gen())
        assert response.status_code == 413

    @pytest.mark.asyncio
    async def test_bug_cc_15_chunked_body_aborts_before_full_buffer(self):
        """BUG-CC-15: middleware must abort streaming read before consuming the full body.

        Verified at the dispatch level by counting how many chunks ``request.stream()``
        yields before the middleware aborts. With max_bytes=1024 and 20 x 512-byte
        chunks (total 10240 bytes), the middleware must return 413 after reading
        at most ~1024 + 1 chunk, not 10240 bytes.
        """
        from unittest.mock import MagicMock

        bytes_yielded = 0

        async def stream_gen():
            nonlocal bytes_yielded
            for _ in range(20):
                chunk = b"A" * 512
                bytes_yielded += len(chunk)
                yield chunk

        request = MagicMock()
        request.headers = {}
        request.method = "POST"
        request.stream = stream_gen

        async def call_next(_req):  # pragma: no cover — should not be reached.
            raise AssertionError("call_next should not run after 413 abort")

        middleware = RequestBodyLimitMiddleware(app=MagicMock(), max_bytes=1024)
        response = await middleware.dispatch(request, call_next)

        assert response.status_code == 413
        # Streaming early-abort: should abort within a single chunk past the limit.
        assert bytes_yielded <= 1024 + 512, f"Streaming read consumed {bytes_yielded} bytes — should abort near the 1024 byte limit"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["POST", "PUT", "PATCH"])
    async def test_cr024_underdeclared_content_length_still_enforces_cap(self, method):
        """CR-024: under-declared Content-Length must not bypass the stream cap.

        A client that claims ``Content-Length`` under the limit and then
        streams more than ``max_bytes`` must still receive 413. Gating the
        stream-read on ``content_length is None`` would admit this bypass.
        """
        from unittest.mock import MagicMock

        bytes_yielded = 0

        async def stream_gen():
            nonlocal bytes_yielded
            for _ in range(20):
                chunk = b"A" * 512
                bytes_yielded += len(chunk)
                yield chunk

        request = MagicMock()
        # Declared length is under the 1024-byte cap; actual stream is 10 KiB.
        request.headers = {"content-length": "100"}
        request.method = method
        request.stream = stream_gen

        async def call_next(_req):  # pragma: no cover — should not be reached.
            raise AssertionError("call_next should not run after under-declared 413 abort")

        middleware = RequestBodyLimitMiddleware(app=MagicMock(), max_bytes=1024)
        response = await middleware.dispatch(request, call_next)

        assert response.status_code == 413
        assert bytes_yielded <= 1024 + 512, f"Under-declared stream consumed {bytes_yielded} bytes — should abort near the 1024 byte limit"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["POST", "PUT", "PATCH"])
    async def test_cr024_declared_content_length_under_limit_caches_body(self, method):
        """Mutating requests with a truthful under-limit Content-Length still
        stream-read and cache ``request._body`` for downstream handlers."""
        from unittest.mock import MagicMock

        payload = b'{"a":"b"}'

        async def stream_gen():
            yield payload

        request = MagicMock()
        request.headers = {"content-length": str(len(payload))}
        request.method = method
        request.stream = stream_gen
        # MagicMock would invent ``_body``; start unset so the cache write is real.
        del request._body

        call_next_seen = []

        async def call_next(req):
            call_next_seen.append(req._body)
            return MagicMock(status_code=200)

        middleware = RequestBodyLimitMiddleware(app=MagicMock(), max_bytes=1024)
        response = await middleware.dispatch(request, call_next)

        assert response.status_code == 200
        assert call_next_seen == [payload]
        assert request._body == payload

    def test_bug_cc_15_body_cached_for_downstream_handlers(self, body_limit_app):
        """BUG-CC-15: after streaming read, body must remain readable by downstream handlers."""
        client = TestClient(body_limit_app)

        # Send a chunked body under the limit; handler must still be able to parse JSON.
        def body_gen():
            yield b'{"a":"b'
            yield b'","c":"d"}'

        # See note in test_chunked_body_under_limit_accepted: FastAPI's
        # ``payload: dict`` parsing requires Content-Type: application/json.
        response = client.post("/echo", content=body_gen(), headers={"Content-Type": "application/json"})
        assert response.status_code == 200
        # Handler echoes received_bytes; presence + 200 confirm body was readable downstream.
        assert "received_bytes" in response.json()


@pytest.mark.unit
class TestFailedAuthThrottleIntegration:
    """APD-CASCOR-004: the 401 path must consume budget.

    Mirrors the corpus in ``juniper-service-core/tests/test_middleware.py`` so the fork and the
    shared package cannot drift apart silently again.
    """

    def test_failed_auth_attempts_are_throttled(self, app_with_middleware):
        """The arm that catches a half-port.

        Wiring only the pre-auth ``check()`` and omitting ``record_failure()`` yields a throttle
        that never accumulates -- a silent no-op -- and every request below would stay 401
        forever instead of turning 429.
        """
        app = app_with_middleware(api_keys=["secret"], throttle=build_failed_auth_throttle(max_failures=3, window_seconds=60))
        client = TestClient(app)

        for _ in range(3):
            assert client.get("/v1/network", headers={"X-API-Key": "wrong"}).status_code == 401

        response = client.get("/v1/network", headers={"X-API-Key": "wrong"})
        assert response.status_code == 429
        assert int(response.headers["Retry-After"]) >= 1

    def test_valid_credentials_never_consume_the_throttle_budget(self, app_with_middleware):
        """Well-behaved traffic sees no behaviour change, which is why the default is enabled."""
        app = app_with_middleware(api_keys=["secret"], throttle=build_failed_auth_throttle(max_failures=2, window_seconds=60))
        client = TestClient(app)

        for _ in range(25):
            assert client.get("/v1/network", headers={"X-API-Key": "secret"}).status_code == 200

    def test_throttle_is_enabled_by_default(self, app_with_middleware):
        """No throttle passed: the production ``add_middleware`` call site must still be covered.

        ``api/app.py`` constructs SecurityMiddleware without a throttle argument, so a default of
        ``None`` that meant "disabled" would leave the running service exactly as unprotected as
        before the fix.
        """
        app = app_with_middleware(api_keys=["secret"])
        client = TestClient(app)

        for _ in range(10):
            assert client.get("/v1/network", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/v1/network", headers={"X-API-Key": "wrong"}).status_code == 429

    def test_throttle_can_be_opted_out(self, app_with_middleware):
        app = app_with_middleware(api_keys=["secret"], throttle=build_failed_auth_throttle(enabled=False))
        client = TestClient(app)

        for _ in range(25):
            assert client.get("/v1/network", headers={"X-API-Key": "wrong"}).status_code == 401

    def test_quota_429_is_not_counted_as_an_authentication_failure(self, app_with_middleware):
        """Only a 401 feeds the throttle.

        A 429 from the identity-keyed limiter is a quota outcome, not a credential guess.
        Counting it would let an authenticated caller throttle *itself* out of the auth path
        merely by exceeding its own quota.
        """
        throttle = build_failed_auth_throttle(max_failures=2, window_seconds=60)
        app = app_with_middleware(api_keys=["secret"], rate_limit_enabled=True, rpm=1, throttle=throttle)
        client = TestClient(app)

        assert client.get("/v1/network", headers={"X-API-Key": "secret"}).status_code == 200
        for _ in range(5):
            assert client.get("/v1/network", headers={"X-API-Key": "secret"}).status_code == 429

        # None of those quota 429s were recorded, so the failure budget is still intact.
        assert throttle.check("testclient")[0] is False

    def test_exempt_paths_bypass_the_throttle(self, app_with_middleware):
        """Health checks stay reachable even from an IP that is currently throttled."""
        app = app_with_middleware(api_keys=["secret"], throttle=build_failed_auth_throttle(max_failures=1, window_seconds=60))
        client = TestClient(app)

        assert client.get("/v1/network", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/v1/network", headers={"X-API-Key": "wrong"}).status_code == 429
        assert client.get("/v1/health").status_code == 200


@pytest.mark.unit
class TestFailedAuthThrottle:
    """Unit behaviour of the throttle itself (APD-CASCOR-004)."""

    def test_check_does_not_consume_budget(self):
        throttle = FailedAuthThrottle(max_failures=1, window_seconds=60)
        for _ in range(10):
            assert throttle.check("1.2.3.4") == (False, 0)
        throttle.record_failure("1.2.3.4")
        blocked, retry_after = throttle.check("1.2.3.4")
        assert blocked is True
        assert retry_after >= 1

    def test_is_keyed_per_source_ip(self):
        throttle = FailedAuthThrottle(max_failures=1, window_seconds=60)
        throttle.record_failure("1.2.3.4")
        assert throttle.check("1.2.3.4")[0] is True
        assert throttle.check("5.6.7.8")[0] is False

    def test_window_rolls_over(self):
        throttle = FailedAuthThrottle(max_failures=1, window_seconds=0)  # every check starts a new window
        throttle.record_failure("1.2.3.4")
        assert throttle.check("1.2.3.4")[0] is False

    def test_disabled_never_blocks(self):
        throttle = FailedAuthThrottle(max_failures=1, enabled=False)
        for _ in range(10):
            throttle.record_failure("1.2.3.4")
        assert throttle.check("1.2.3.4") == (False, 0)

    def test_reset_clears_state(self):
        throttle = FailedAuthThrottle(max_failures=1, window_seconds=60)
        throttle.record_failure("1.2.3.4")
        assert throttle.check("1.2.3.4")[0] is True
        throttle.reset()
        assert throttle.check("1.2.3.4")[0] is False

    def test_prunes_expired_entries(self):
        """Bounded memory: a dict keyed by attacker-supplied source IPs is itself a DoS vector."""
        throttle = FailedAuthThrottle(max_failures=100, window_seconds=0)
        for i in range(FailedAuthThrottle._CLEANUP_INTERVAL + 10):
            throttle.record_failure(f"10.0.0.{i % 255}")
        assert len(throttle._failures) <= FailedAuthThrottle._MAX_ENTRIES

    def test_build_factory_defaults_match_the_documented_budget(self):
        throttle = build_failed_auth_throttle()
        assert throttle.enabled is True
        assert throttle.max_failures == 10
