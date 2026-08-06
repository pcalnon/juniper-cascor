"""Unit tests for SecurityMiddleware, SecurityHeadersMiddleware, and RequestBodyLimitMiddleware."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware import EXEMPT_PATHS, RequestBodyLimitMiddleware, SecurityHeadersMiddleware, SecurityMiddleware
from api.security import APIKeyAuth, RateLimiter


@pytest.fixture
def app_with_middleware():
    """Create a FastAPI app with security middleware."""

    def _create(api_keys=None, rate_limit_enabled=False, rpm=60):
        app = FastAPI()
        auth = APIKeyAuth(api_keys)
        limiter = RateLimiter(requests_per_minute=rpm, enabled=rate_limit_enabled)
        app.add_middleware(SecurityMiddleware, api_key_auth=auth, rate_limiter=limiter)

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
