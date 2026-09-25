"""Unit tests for API security: authentication and rate limiting."""

import time
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.websockets import WebSocketDisconnect

from api.security import APIKeyAuth, RateLimiter

# ``hmac.compare_digest`` raises ``TypeError`` on a ``str`` holding any non-ASCII character.
# Starlette decodes header bytes as latin-1, so every byte above 0x7f reaches ``validate`` as
# one: an anonymous ``X-API-Key: \xa0`` was a 500 whose Sentry event carried the comparison
# loop's ``candidate`` -- the real configured key (found by the validation of
# juniper-canopy#683, 2026-09-24).

#: Header values as ``validate`` receives them. U+00A0 and U+0085 are the two the validation
#: drove through both uvicorn parsers; the rest vary position and byte.
_NON_ASCII_PRESENTED = ["\xa0", "\x85", "\xff", "v\xe1lid-key", "valid-key\xa0"]

#: Raw header bytes, as a client sends them.
_NON_ASCII_RAW = [b"\xa0", b"\x85", b"valid-key\xa0"]

#: Strings chosen to separate the candidate encodings. "\ud800" is a lone HIGH surrogate --
#: ``strict`` and ``surrogateescape`` both raise on it; "\udcc3\udca9" is what
#: ``surrogateescape`` encodes to the same bytes as "\xe9". Only an encoding that is both
#: total and injective gives ``validate(x) == (x in keys)`` for every pair below.
_ENCODING_PROBES = ["valid-key", "cl\xe9", "\xe9", "\udcc3\udca9", "\ud800-key", "\udcff", "\U0001f511"]


@pytest.mark.unit
class TestAPIKeyAuth:
    """Tests for APIKeyAuth class."""

    def test_disabled_when_no_keys(self) -> None:
        """Auth should be disabled when no keys are configured."""
        auth = APIKeyAuth(None)
        assert not auth.enabled

        auth = APIKeyAuth([])
        assert not auth.enabled

    def test_enabled_when_keys_configured(self) -> None:
        """Auth should be enabled when keys are configured."""
        auth = APIKeyAuth(["key1", "key2"])
        assert auth.enabled

    def test_blank_key_does_not_enable_auth(self) -> None:
        """APD-CASCOR-006: a blank key must not enable authentication.

        ``JUNIPER_CASCOR_API_KEYS='[""]'`` parses to ``['']``. Without the filter
        that set ``_enabled = True`` and then validated an empty ``X-API-Key`` --
        strictly worse than auth being off, because the deployment believes it is
        protected.
        """
        assert not APIKeyAuth([""]).enabled

    def test_whitespace_only_key_does_not_enable_auth(self) -> None:
        for blank in (" ", "\t", "\n", "   \t\n "):
            assert not APIKeyAuth([blank]).enabled, f"{blank!r} must not enable auth"

    def test_blank_key_never_validates(self) -> None:
        """The impact, not just the flag: an empty key must never be accepted.

        With ONLY a blank key auth is disabled, so ``validate`` is vacuously True --
        which is why the meaningful assertion pairs a blank with a real key and
        checks that the blank still does not admit an empty ``X-API-Key``.
        """
        auth = APIKeyAuth(["", "real-key"])
        assert auth.enabled
        assert auth.validate("real-key")
        assert not auth.validate("")
        assert not auth.validate(" ")

    def test_blank_keys_are_dropped_but_real_ones_kept(self) -> None:
        auth = APIKeyAuth(["", "  ", "real-key", "\t"])
        assert auth.enabled
        assert auth.validate("real-key")

    def test_non_string_entries_are_dropped(self) -> None:
        """A malformed JSON list must not crash or half-enable auth.

        The filter runs before the element is hashed into the set, so an
        unhashable entry cannot raise TypeError here.
        """
        assert not APIKeyAuth([None, 0, {}]).enabled  # type: ignore[list-item]
        auth = APIKeyAuth([None, "real-key"])  # type: ignore[list-item]
        assert auth.enabled
        assert auth.validate("real-key")

    def test_validate_returns_true_when_disabled(self) -> None:
        """Validate should return True when auth is disabled."""
        auth = APIKeyAuth(None)
        assert auth.validate(None)
        assert auth.validate("any-key")

    def test_validate_valid_key(self) -> None:
        """Validate should return True for valid key."""
        auth = APIKeyAuth(["valid-key"])
        assert auth.validate("valid-key")

    def test_validate_invalid_key(self) -> None:
        """Validate should return False for invalid key."""
        auth = APIKeyAuth(["valid-key"])
        assert not auth.validate("invalid-key")
        assert not auth.validate(None)

    def test_validate_empty_string_key_is_invalid(self) -> None:
        """Empty-string X-API-Key is present but must not authenticate.

        Distinguishes blank credentials from a missing header (None): both
        fail closed when auth is enabled, but only None is "Missing".
        """
        auth = APIKeyAuth(["valid-key"])
        assert not auth.validate("")
        assert not auth.validate("   ")

    def test_validate_unequal_length_key_is_invalid(self) -> None:
        """Unequal-length keys exercise hmac.compare_digest safely and fail."""
        auth = APIKeyAuth(["valid-key"])
        assert not auth.validate("short")
        assert not auth.validate("valid-key-with-extra-suffix")

    def test_validate_uses_timing_safe_comparison(self) -> None:
        """Validate should use hmac.compare_digest for timing-safe comparison."""
        import hmac
        from unittest.mock import patch

        auth = APIKeyAuth(["valid-key"])
        with patch.object(hmac, "compare_digest", wraps=hmac.compare_digest) as mock_compare:
            auth.validate("valid-key")
            mock_compare.assert_called()

    def test_validate_with_multiple_keys(self) -> None:
        """Validate should work correctly with multiple configured keys."""
        auth = APIKeyAuth(["key1", "key2", "key3"])
        assert auth.validate("key1")
        assert auth.validate("key2")
        assert auth.validate("key3")
        assert not auth.validate("key4")

    @pytest.mark.asyncio
    async def test_call_returns_none_when_disabled(self) -> None:
        """Dependency should return None when auth is disabled."""
        auth = APIKeyAuth(None)
        request = MagicMock()
        request.headers.get.return_value = None

        result = await auth(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_call_raises_401_when_missing_key(self) -> None:
        """Dependency should raise 401 when key is missing."""
        auth = APIKeyAuth(["valid-key"])
        request = MagicMock()
        request.headers.get.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await auth(request)
        assert exc_info.value.status_code == 401
        assert "Missing API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_call_raises_401_when_invalid_key(self) -> None:
        """Dependency should raise 401 when key is invalid."""
        auth = APIKeyAuth(["valid-key"])
        request = MagicMock()
        request.headers.get.return_value = "invalid-key"

        with pytest.raises(HTTPException) as exc_info:
            await auth(request)
        assert exc_info.value.status_code == 401
        assert "Invalid API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_call_raises_401_invalid_not_missing_for_empty_string(self) -> None:
        """Present-but-blank X-API-Key must surface as Invalid, not Missing."""
        auth = APIKeyAuth(["valid-key"])
        request = MagicMock()
        request.headers.get.return_value = ""

        with pytest.raises(HTTPException) as exc_info:
            await auth(request)
        assert exc_info.value.status_code == 401
        assert "Invalid API key" in exc_info.value.detail
        assert "Missing" not in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_call_returns_key_when_valid(self) -> None:
        """Dependency should return the key when valid."""
        auth = APIKeyAuth(["valid-key"])
        request = MagicMock()
        request.headers.get.return_value = "valid-key"

        result = await auth(request)
        assert result == "valid-key"

    @pytest.mark.parametrize("presented", _NON_ASCII_PRESENTED)
    def test_non_ascii_presented_key_is_a_mismatch_not_an_exception(self, presented) -> None:
        assert APIKeyAuth(["valid-key"]).validate(presented) is False

    @pytest.mark.parametrize("configured", _ENCODING_PROBES)
    @pytest.mark.parametrize("presented", _ENCODING_PROBES)
    def test_validate_matches_exactly_when_the_strings_are_equal(self, configured, presented) -> None:
        """The bytes compare must accept what ``==`` accepts: no raise, no collision."""
        assert APIKeyAuth([configured]).validate(presented) is (presented == configured)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw", _NON_ASCII_RAW)
    async def test_call_raises_401_on_a_non_ascii_header(self, raw) -> None:
        """A real ``Request`` (not a mock) so the header takes Starlette's latin-1 decode."""
        auth = APIKeyAuth(["valid-key"])
        request = Request({"type": "http", "method": "GET", "path": "/", "headers": [(b"x-api-key", raw)], "client": ("testclient", 50000)})

        with pytest.raises(HTTPException) as exc_info:
            await auth(request)
        assert exc_info.value.status_code == 401
        assert "Invalid API key" in exc_info.value.detail


@pytest.mark.unit
class TestNonAsciiApiKeyThroughTheApp:
    """The whole stack, from ``create_app`` with auth on, for a raw non-ASCII ``X-API-Key``.

    ``raise_server_exceptions=False`` so that a regression reads as the 500 a caller would
    see, not as the exception. Header values go in as ``bytes``, which httpx sends unencoded.
    """

    @pytest.fixture
    def client(self):
        from api.app import create_app
        from api.settings import Settings

        with TestClient(create_app(Settings(api_keys=["valid-key"], auto_start=False)), raise_server_exceptions=False) as test_client:
            yield test_client

    @pytest.mark.parametrize("raw", _NON_ASCII_RAW)
    def test_http_request_is_401_not_500(self, client, raw) -> None:
        response = client.get("/v1/network", headers={"X-API-Key": raw})
        assert response.status_code == 401
        assert response.json() == {"detail": "Invalid API key."}

    def test_failed_attempts_reach_the_throttle(self, client) -> None:
        """Only a 401 records a failure, so while this was a 500 a flood of it was never throttled.

        The app's throttle runs at the library default budget: 10 failures per 60 s.
        """
        for _ in range(10):
            assert client.get("/v1/network", headers={"X-API-Key": b"\xa0"}).status_code == 401
        assert client.get("/v1/network", headers={"X-API-Key": b"\xa0"}).status_code == 429

    def test_websocket_handshake_closes_4001(self, client) -> None:
        """``ws_authenticate`` calls the same ``validate``, so the handshake raised with it.

        ``/ws/training`` rather than ``/ws/control``: the control channel's rejection cooldown
        is process-global, and a rejection recorded here would leak into other tests.
        """
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws/training", headers={"X-API-Key": b"\xa0"}) as ws:
                ws.receive_json()
        assert exc_info.value.code == 4001


@pytest.mark.unit
class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_disabled_allows_all(self) -> None:
        """Disabled limiter should allow all requests."""
        limiter = RateLimiter(requests_per_minute=5, enabled=False)

        for _ in range(100):
            allowed, remaining, _ = limiter.check("key")
            assert allowed

    def test_allows_within_limit(self) -> None:
        """Limiter should allow requests within limit."""
        limiter = RateLimiter(requests_per_minute=5, enabled=True)

        for i in range(5):
            allowed, remaining, _ = limiter.check("key")
            assert allowed
            assert remaining == 5 - i - 1

    def test_blocks_over_limit(self) -> None:
        """Limiter should block requests over limit."""
        limiter = RateLimiter(requests_per_minute=5, enabled=True)

        for _ in range(5):
            limiter.check("key")

        allowed, remaining, reset_in = limiter.check("key")
        assert not allowed
        assert remaining == 0
        assert reset_in > 0

    def test_different_keys_tracked_separately(self) -> None:
        """Different keys should have separate limits."""
        limiter = RateLimiter(requests_per_minute=5, enabled=True)

        for _ in range(5):
            limiter.check("key1")

        allowed1, _, _ = limiter.check("key1")
        allowed2, _, _ = limiter.check("key2")

        assert not allowed1
        assert allowed2

    def test_window_reset(self) -> None:
        """Window should reset after time expires."""
        limiter = RateLimiter(requests_per_minute=5, window_seconds=1, enabled=True)

        for _ in range(5):
            limiter.check("key")

        allowed, _, _ = limiter.check("key")
        assert not allowed

        time.sleep(1.1)

        allowed, remaining, _ = limiter.check("key")
        assert allowed
        assert remaining == 4

    def test_reset_clears_counters(self) -> None:
        """Reset should clear all counters."""
        limiter = RateLimiter(requests_per_minute=5, enabled=True)

        for _ in range(5):
            limiter.check("key")

        allowed, _, _ = limiter.check("key")
        assert not allowed

        limiter.reset()

        allowed, _, _ = limiter.check("key")
        assert allowed

    @pytest.mark.asyncio
    async def test_call_allows_when_within_limit(self) -> None:
        """Dependency should allow requests within limit."""
        limiter = RateLimiter(requests_per_minute=5, enabled=True)
        request = MagicMock()
        request.client.host = "127.0.0.1"
        request.state = MagicMock()

        for _ in range(5):
            await limiter(request, api_key=None)

    @pytest.mark.asyncio
    async def test_call_raises_429_when_over_limit(self) -> None:
        """Dependency should raise 429 when over limit."""
        limiter = RateLimiter(requests_per_minute=5, enabled=True)
        request = MagicMock()
        request.client.host = "127.0.0.1"
        request.state = MagicMock()

        for _ in range(5):
            await limiter(request, api_key=None)

        with pytest.raises(HTTPException) as exc_info:
            await limiter(request, api_key=None)
        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_call_429_includes_rate_limit_headers(self) -> None:
        """429 responses must carry X-RateLimit-* and Retry-After for client backoff."""
        limiter = RateLimiter(requests_per_minute=2, window_seconds=60, enabled=True)
        request = MagicMock()
        request.client.host = "127.0.0.1"
        request.state = MagicMock()

        await limiter(request, api_key=None)
        await limiter(request, api_key=None)

        with pytest.raises(HTTPException) as exc_info:
            await limiter(request, api_key=None)

        headers = exc_info.value.headers or {}
        assert headers["X-RateLimit-Limit"] == "2"
        assert headers["X-RateLimit-Remaining"] == "0"
        assert "X-RateLimit-Reset" in headers
        assert headers["Retry-After"] == headers["X-RateLimit-Reset"]
        assert int(headers["Retry-After"]) >= 0
        assert int(headers["Retry-After"]) <= 60

    @pytest.mark.asyncio
    async def test_call_populates_request_state_on_allowed(self) -> None:
        """Allowed calls must attach remaining/reset onto request.state for middleware."""
        limiter = RateLimiter(requests_per_minute=5, window_seconds=60, enabled=True)
        request = MagicMock()
        request.client.host = "127.0.0.1"
        request.state = MagicMock()

        await limiter(request, api_key=None)

        assert request.state.rate_limit_remaining == 4
        assert isinstance(request.state.rate_limit_reset, int)
        assert 0 < request.state.rate_limit_reset <= 60

    @pytest.mark.asyncio
    async def test_call_uses_api_key_for_limiting(self) -> None:
        """Dependency should use API key for rate limiting when provided."""
        limiter = RateLimiter(requests_per_minute=5, enabled=True)
        request = MagicMock()
        request.client.host = "127.0.0.1"
        request.state = MagicMock()

        for _ in range(5):
            await limiter(request, api_key="key1")

        with pytest.raises(HTTPException):
            await limiter(request, api_key="key1")

        await limiter(request, api_key="key2")

    def test_rate_limiter_window_property(self) -> None:
        """Window property should return configured window seconds."""
        limiter = RateLimiter(requests_per_minute=10, window_seconds=30)
        assert limiter.window == 30

    def test_get_key_with_no_client(self) -> None:
        """_get_key should return 'ip:unknown' when request has no client."""
        limiter = RateLimiter()
        request = MagicMock()
        request.client = None
        key = limiter._get_key(request, None)
        assert key == "ip:unknown"

    @pytest.mark.asyncio
    async def test_call_noop_when_disabled(self) -> None:
        """Dependency should do nothing when disabled."""
        limiter = RateLimiter(requests_per_minute=5, enabled=False)
        request = MagicMock()
        request.client.host = "127.0.0.1"

        for _ in range(100):
            await limiter(request, api_key=None)


@pytest.mark.unit
class TestRateLimiterBugCC13Cleanup:
    """BUG-CC-13: RateLimiter._counters must be pruned to prevent unbounded growth."""

    def test_expired_entries_pruned_after_cleanup_interval(self) -> None:
        """After CLEANUP_INTERVAL calls, expired entries (older than 2*window) are removed."""
        limiter = RateLimiter(requests_per_minute=1000, window_seconds=1, enabled=True)
        # Seed many distinct keys; each will create a counter entry.
        for i in range(50):
            limiter.check(f"key-{i}")
        # Force timestamps into the past so they are eligible for eviction.
        with limiter._lock:
            for k in list(limiter._counters):
                count, _ts = limiter._counters[k]
                limiter._counters[k] = (count, time.time() - (2 * limiter._window) - 1)
        # Drive cleanup by reaching CLEANUP_INTERVAL across one fresh key.
        for _ in range(limiter._CLEANUP_INTERVAL):
            limiter.check("driver-key")
        # Only the driver key should remain; expired keys pruned.
        with limiter._lock:
            assert "driver-key" in limiter._counters
            for i in range(50):
                assert f"key-{i}" not in limiter._counters

    def test_unbounded_growth_prevented_by_max_entries_cap(self) -> None:
        """Hard cap _MAX_ENTRIES prevents unbounded counter growth even with fresh keys."""
        limiter = RateLimiter(requests_per_minute=10, window_seconds=60, enabled=True)
        # Override cap and interval to keep test fast.
        limiter._MAX_ENTRIES = 100
        limiter._CLEANUP_INTERVAL = 50
        # Insert many more keys than the cap; each is fresh so not "expired".
        for i in range(500):
            limiter.check(f"key-{i}")
        # After cleanup runs, dict size must be bounded near _MAX_ENTRIES
        # (allow +1 for the current request's freshly-inserted key after the
        # cleanup pass that ran at the start of this same check() call).
        assert len(limiter._counters) <= limiter._MAX_ENTRIES + 1
        # And critically: it must not approach the unbounded-growth scale (500).
        assert len(limiter._counters) < 200

    def test_cleanup_does_not_evict_fresh_entries_under_cap(self) -> None:
        """Entries within the active window must not be pruned by periodic cleanup."""
        limiter = RateLimiter(requests_per_minute=10, window_seconds=60, enabled=True)
        limiter._CLEANUP_INTERVAL = 5
        for i in range(20):
            limiter.check(f"fresh-{i}")
        # All 20 fresh keys should still be present.
        for i in range(20):
            assert f"fresh-{i}" in limiter._counters

    def test_reset_clears_cleanup_counter(self) -> None:
        """reset() should also reset the cleanup tick counter."""
        limiter = RateLimiter(requests_per_minute=5, enabled=True)
        for _ in range(5):
            limiter.check("key")
        assert limiter._request_count_since_cleanup > 0
        limiter.reset()
        assert limiter._request_count_since_cleanup == 0


@pytest.mark.unit
class TestSecurityModuleFunctions:
    """Tests for module-level security functions."""

    def test_get_api_key_auth_returns_instance(self) -> None:
        """get_api_key_auth should return an APIKeyAuth instance."""
        from api.security import get_api_key_auth, reset_security_state

        reset_security_state()
        auth = get_api_key_auth()
        assert isinstance(auth, APIKeyAuth)

    def test_get_api_key_auth_returns_same_instance(self) -> None:
        """get_api_key_auth should return same instance on second call."""
        from api.security import get_api_key_auth, reset_security_state

        reset_security_state()
        auth1 = get_api_key_auth()
        auth2 = get_api_key_auth()
        assert auth1 is auth2

    def test_get_rate_limiter_returns_instance(self) -> None:
        """get_rate_limiter should return a RateLimiter instance."""
        from api.security import get_rate_limiter, reset_security_state

        reset_security_state()
        limiter = get_rate_limiter()
        assert isinstance(limiter, RateLimiter)

    def test_get_rate_limiter_returns_same_instance(self) -> None:
        """get_rate_limiter should return same instance on second call."""
        from api.security import get_rate_limiter, reset_security_state

        reset_security_state()
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        assert limiter1 is limiter2

    def test_reset_security_state(self) -> None:
        """reset_security_state should clear cached instances."""
        from api.security import get_api_key_auth, get_rate_limiter, reset_security_state

        reset_security_state()
        auth1 = get_api_key_auth()
        limiter1 = get_rate_limiter()
        reset_security_state()
        auth2 = get_api_key_auth()
        limiter2 = get_rate_limiter()
        assert auth1 is not auth2
        assert limiter1 is not limiter2


@pytest.mark.unit
class TestApiKeysSettingsValidator:
    """APD-CASCOR-006: both branches of ``_parse_api_keys`` must filter blanks.

    The comma-separated-string branch always did; the list branch returned ``v``
    untouched, which is why the JSON form was the reachable shape of this defect.
    Mirrors the juniper-data sibling test of the same name.
    """

    @staticmethod
    def _parse(value):
        from api.settings import Settings

        return Settings._parse_api_keys(value)

    def test_json_list_of_one_blank_becomes_none(self) -> None:
        """The exact reachable shape: JUNIPER_CASCOR_API_KEYS='[""]'."""
        assert self._parse([""]) is None

    def test_json_list_of_whitespace_becomes_none(self) -> None:
        assert self._parse(["  ", "\t"]) is None

    def test_json_list_keeps_real_keys_and_strips_them(self) -> None:
        assert self._parse(["  real  ", ""]) == ["real"]

    def test_comma_string_branch_still_filters(self) -> None:
        """Regression guard on the branch that was already correct."""
        assert self._parse("a, ,b") == ["a", "b"]

    def test_none_and_empty_string_are_unchanged(self) -> None:
        assert self._parse(None) is None
        assert self._parse("") is None

    def test_tuple_is_handled_like_a_list(self) -> None:
        assert self._parse(("", "real")) == ["real"]
