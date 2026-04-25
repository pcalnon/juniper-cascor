"""FastAPI middleware for security and request processing."""

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from cascor_constants.constants_api import _PROJECT_API_HTTP_400_BAD_REQUEST, _PROJECT_API_HTTP_413_PAYLOAD_TOO_LARGE, _PROJECT_API_MAX_REQUEST_BODY_BYTES

from .security import APIKeyAuth, RateLimiter

EXEMPT_PATHS = {
    "/v1/health",
    "/v1/health/live",
    "/v1/health/ready",
    "/docs",
    "/openapi.json",
    "/redoc",
}

# Default Content-Security-Policy for API-only services.
_DEFAULT_CSP = "default-src 'none'; frame-ancestors 'none'"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses.

    Injects standard security headers (X-Content-Type-Options, X-Frame-Options,
    Referrer-Policy, Permissions-Policy, CSP, and conditional HSTS) into every
    HTTP response.
    """

    def __init__(self, app: ASGIApp, content_security_policy: str = _DEFAULT_CSP) -> None:
        super().__init__(app)
        self._csp = content_security_policy

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = self._csp

        if request.headers.get("X-Forwarded-Proto") == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


# Module-level alias preserved for tests that import this name directly.
# The canonical source of truth is
# :data:`cascor_constants.constants_api._PROJECT_API_MAX_REQUEST_BODY_BYTES`.
_MAX_REQUEST_BODY_BYTES = _PROJECT_API_MAX_REQUEST_BODY_BYTES


class RequestBodyLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose body exceeds a configurable limit.

    The ``Content-Length`` header is used as an **early reject** fast path but
    is not trusted as the sole size check (CR-024): a malicious client can
    under-declare or omit the header and send an unbounded chunked stream.
    For POST/PUT/PATCH requests we always stream-read the body with a
    cumulative byte cap, aborting with HTTP 413 as soon as the cap is
    exceeded. This prevents the classic chunked-encoding memory-exhaustion
    bypass in which ``await request.body()`` would allocate the entire body
    before any size check runs.

    The fully-read body is cached on ``request._body`` so downstream FastAPI
    route handlers can consume it via ``request.body()`` / ``request.json()``
    / pydantic body parsing without triggering a second read.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = _MAX_REQUEST_BODY_BYTES) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Fast-path early reject on declared Content-Length. Still untrusted
        # as a floor, so the stream-read below enforces the real limit.
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                return JSONResponse(status_code=_PROJECT_API_HTTP_400_BAD_REQUEST, content={"detail": "Invalid Content-Length header"})
            if declared_length > self._max_bytes:
                return JSONResponse(status_code=_PROJECT_API_HTTP_413_PAYLOAD_TOO_LARGE, content={"detail": "Request body too large"})
        if content_length is None and request.method in ("POST", "PUT", "PATCH"):
            # BUG-CC-15: stream-read with early abort to avoid buffering full body before size check.
            chunks: list[bytes] = []
            size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > self._max_bytes:
                    return JSONResponse(status_code=_PROJECT_API_HTTP_413_PAYLOAD_TOO_LARGE, content={"detail": "Request body too large"})
                chunks.append(chunk)
            # Cache body for downstream handlers (Starlette convention).
            request._body = b"".join(chunks)
        return await call_next(request)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication and rate limiting.

    Applies authentication and rate limiting to all requests except
    explicitly exempt paths (health checks, docs). WebSocket upgrade
    requests are not intercepted by BaseHTTPMiddleware, so /ws/* paths
    are inherently exempt.
    """

    def __init__(
        self,
        app: ASGIApp,
        api_key_auth: APIKeyAuth,
        rate_limiter: RateLimiter,
    ) -> None:
        """Initialize the security middleware.

        Args:
            app: The ASGI application.
            api_key_auth: API key authentication handler.
            rate_limiter: Rate limiter instance.
        """
        super().__init__(app)
        self._api_key_auth = api_key_auth
        self._rate_limiter = rate_limiter

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process the request through security checks.

        Args:
            request: The incoming request.
            call_next: The next middleware/handler in the chain.

        Returns:
            The response from the application.
        """
        path = request.url.path

        if self._is_exempt(path):
            return await call_next(request)

        api_key = None
        try:
            if self._api_key_auth.enabled:
                api_key = await self._api_key_auth(request)

            if self._rate_limiter.enabled:
                await self._rate_limiter(request, api_key)
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=exc.headers,
            )

        response = await call_next(request)

        if self._rate_limiter.enabled and hasattr(request.state, "rate_limit_remaining"):
            response.headers["X-RateLimit-Limit"] = str(self._rate_limiter.limit)
            response.headers["X-RateLimit-Remaining"] = str(request.state.rate_limit_remaining)
            response.headers["X-RateLimit-Reset"] = str(request.state.rate_limit_reset)

        return response

    def _is_exempt(self, path: str) -> bool:
        """Check if a path is exempt from security checks.

        Args:
            path: The request path.

        Returns:
            True if the path is exempt, False otherwise.
        """
        return path in EXEMPT_PATHS
