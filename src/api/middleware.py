"""FastAPI middleware for security and request processing."""

import logging

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from cascor_constants.constants_api import _PROJECT_API_HTTP_400_BAD_REQUEST, _PROJECT_API_HTTP_413_PAYLOAD_TOO_LARGE, _PROJECT_API_MAX_REQUEST_BODY_BYTES

from .security import APIKeyAuth, FailedAuthThrottle, RateLimiter, build_failed_auth_throttle

logger = logging.getLogger(__name__)

EXEMPT_PATHS = {
    "/v1/health",
    "/v1/health/live",
    "/v1/health/ready",
    # APD-DATA-024 / service-core 0.6.0: the documentation surface is deliberately
    # NOT exempt. ``_is_exempt()`` ignores whether a key is configured, so listing
    # /docs, /openapi.json and /redoc here made "docs enabled" and "docs public"
    # the same switch: re-enabling ``docs_url`` in app.py (currently gated on
    # ``docs_enabled = not settings.api_keys``) would serve the schema to everyone
    # while looking like it sat behind the key. Removing them decouples the two --
    # an authenticated deployment can now mount the docs AND require a key for them.
    # SEC-16 / POC §3.1: ``/metrics`` is gated by the parallel
    # ``MetricsAuthMiddleware`` IP allowlist (cascor mirror of
    # ``juniper-data``'s middleware) instead of SecurityMiddleware's
    # API-key check. Both literal forms cover the FastAPI auto-redirect
    # from missing/extra trailing slash.
    "/metrics",
    "/metrics/",
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
        # as a floor, so the stream-read below enforces the real limit —
        # including the under-declared case (CR-024): a client that claims
        # ``Content-Length: N`` (N <= max) and then streams more than
        # ``max_bytes`` must still be aborted by the cumulative cap.
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                return JSONResponse(status_code=_PROJECT_API_HTTP_400_BAD_REQUEST, content={"detail": "Invalid Content-Length header"})
            if declared_length > self._max_bytes:
                return JSONResponse(status_code=_PROJECT_API_HTTP_413_PAYLOAD_TOO_LARGE, content={"detail": "Request body too large"})
        if request.method in ("POST", "PUT", "PATCH"):
            # BUG-CC-15 / CR-024: always stream-read mutating methods with
            # early abort. Do not gate on ``content_length is None`` — that
            # would let an under-declared Content-Length bypass the cap.
            chunks: list[bytes] = []
            size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > self._max_bytes:
                    return JSONResponse(status_code=_PROJECT_API_HTTP_413_PAYLOAD_TOO_LARGE, content={"detail": "Request body too large"})
                chunks.append(chunk)
            # Cache body for downstream handlers. Starlette's
            # ``BaseHTTPMiddleware._CachedRequest.wrapped_receive`` short-
            # circuits to a synthetic ``http.request`` message constructed
            # from ``self._body`` when that attribute is set, so subsequent
            # ``await request.body()`` / ``request.json()`` / Pydantic body
            # parsing in downstream handlers all see the cached payload.
            request._body = b"".join(chunks)
        return await call_next(request)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication and rate limiting.

    Applies authentication and rate limiting to all requests except
    explicitly exempt paths (health checks, docs). WebSocket upgrade
    requests are not intercepted by BaseHTTPMiddleware, so /ws/* paths
    are inherently exempt.

    The identity-keyed :class:`~api.security.RateLimiter` runs *after* authentication and is
    therefore never reached when auth raises, so before APD-CASCOR-004 the 401 path consumed no
    budget at all and credential guessing was unthrottled. The fix is not to reorder -- that
    trades a real protection for a worse one, collapsing every authenticated caller behind one
    NAT into a single ``ip:`` bucket -- but to add a coarse, IP-keyed
    :class:`~api.security.FailedAuthThrottle` *ahead* of authentication. It only consumes budget
    on a failed attempt, so a caller with a valid key is never counted and well-behaved traffic
    is unaffected.
    """

    def __init__(
        self,
        app: ASGIApp,
        api_key_auth: APIKeyAuth,
        rate_limiter: RateLimiter,
        failed_auth_throttle: FailedAuthThrottle | None = None,
    ) -> None:
        """Initialize the security middleware.

        Args:
            app: The ASGI application.
            api_key_auth: API key authentication handler.
            rate_limiter: Rate limiter instance.
            failed_auth_throttle: Pre-authentication, IP-keyed throttle for failed attempts.
                Defaults to an enabled throttle at the library default budget. Pass
                ``build_failed_auth_throttle(enabled=False)`` to opt out. Defaulting to enabled
                is safe because budget is consumed only on a *failed* attempt, so a caller with
                valid credentials never sees a behaviour change.
        """
        super().__init__(app)
        self._api_key_auth = api_key_auth
        self._rate_limiter = rate_limiter
        self._failed_auth_throttle = failed_auth_throttle if failed_auth_throttle is not None else build_failed_auth_throttle()

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

        client_ip = request.client.host if request.client else "unknown"

        # Pre-authentication throttle. Checked first so an IP already over its failed-attempt
        # budget is rejected without burning an auth comparison, and -- crucially -- so the
        # rejection happens on a path that auth failure cannot skip past.
        if self._failed_auth_throttle.enabled:
            blocked, retry_after = self._failed_auth_throttle.check(client_ip)
            if blocked:
                logger.warning("Too many failed authentication attempts from %s; throttled for %ss", client_ip, retry_after)
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": f"Too many failed authentication attempts. Try again in {retry_after} seconds."},
                    headers={"Retry-After": str(retry_after)},
                )

        api_key = None
        try:
            if self._api_key_auth.enabled:
                api_key = await self._api_key_auth(request)

            if self._rate_limiter.enabled:
                await self._rate_limiter(request, api_key)
        except HTTPException as exc:
            # Record the attempt only for authentication failures. A 429 from the identity-keyed
            # limiter is a quota outcome, not a credential guess, and counting it here would let
            # a caller throttle itself out of the auth path by exceeding its own quota.
            if exc.status_code == status.HTTP_401_UNAUTHORIZED and self._failed_auth_throttle.enabled:
                self._failed_auth_throttle.record_failure(client_ip)
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
