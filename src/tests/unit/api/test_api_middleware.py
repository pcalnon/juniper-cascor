"""Unit tests for SecurityMiddleware and RequestBodyLimitMiddleware."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware import EXEMPT_PATHS, RequestBodyLimitMiddleware, SecurityMiddleware
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

        response = client.post("/echo", content=body_gen())
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

    def test_bug_cc_15_body_cached_for_downstream_handlers(self, body_limit_app):
        """BUG-CC-15: after streaming read, body must remain readable by downstream handlers."""
        client = TestClient(body_limit_app)

        # Send a chunked body under the limit; handler must still be able to parse JSON.
        def body_gen():
            yield b'{"a":"b'
            yield b'","c":"d"}'

        response = client.post("/echo", content=body_gen())
        assert response.status_code == 200
        # Handler echoes received_bytes; presence + 200 confirm body was readable downstream.
        assert "received_bytes" in response.json()
