"""``Retry-After`` and ``X-RateLimit-Reset`` round UP, in both limiters (APD-ECO-016).

``RateLimiter.check`` reported ``int(window - elapsed)``, which truncates toward zero. A rejection in
a window's last second therefore said ``Retry-After: 0``, an instruction to retry immediately into a
limiter that rejects again, and every other fractional remainder was under-reported (9.4s left said
9). ``FailedAuthThrottle.check`` reported ``max(1, int(window - elapsed))``: never 0, but still short
by the fraction. juniper-service-core's copies round up (``APD-SVCCORE-004``, and 0.7.2 for the
throttle); these tests pin the same direction here.

The clock is frozen, so every remainder is exact.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware import SecurityMiddleware
from api.security import APIKeyAuth, FailedAuthThrottle, RateLimiter

# CI's unit lane selects ``-m "unit and not slow"``: an unmarked module is collected, green locally,
# and never run there.
pytestmark = pytest.mark.unit

_START = 1_000.0

# Seconds into a 10 s window -> the whole seconds left, rounded up. 9.4 s left is the plain case. The other three sit
# just above a whole number, where a rounding that agrees with ``ceil`` only near the half (add 0.7 and truncate,
# round to nearest) answers a second short.
_FRACTIONAL = [(0.6, 10), (0.99, 10), (0.999, 10), (8.999, 2)]
# Exact remainders are not bumped. 7 and 5 are ODD on purpose: ``round(x + 0.5)`` rounds a half to even, so it bumps
# an odd remainder and leaves an even one (6) alone.
_EXACT = [(4.0, 6), (3.0, 7), (5.0, 5)]


class _Clock:
    """A stand-in for ``time.time`` that a test moves by assignment."""

    def __init__(self, now: float) -> None:
        self.now = now

    def time(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """Freeze the clock the security module reads, at ``_START``."""
    frozen = _Clock(_START)
    monkeypatch.setattr("api.security.time.time", frozen.time)
    return frozen


def _app(limiter: RateLimiter, throttle: FailedAuthThrottle, api_keys: list[str] | None = None) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/network")
    async def network():
        return {"data": []}

    app.add_middleware(SecurityMiddleware, api_key_auth=APIKeyAuth(api_keys), rate_limiter=limiter, failed_auth_throttle=throttle)
    return app


class TestRateLimiterResetRoundsUp:
    """``RateLimiter.check``'s third value, on the rejection path and the allowed path."""

    @pytest.mark.parametrize("elapsed", [9.01, 9.30, 9.60, 9.90, 9.99])
    def test_a_rejection_in_the_last_second_says_one_never_zero(self, clock: _Clock, elapsed: float) -> None:
        """The defect: ``int()`` truncated each of these remainders to 0."""
        limiter = RateLimiter(requests_per_minute=1, window_seconds=10)
        assert limiter.check("k") == (True, 0, 10)

        clock.now = _START + elapsed

        assert limiter.check("k") == (False, 0, 1)

    @pytest.mark.parametrize(("elapsed", "expected"), _FRACTIONAL)
    def test_a_rejection_rounds_any_fraction_up(self, clock: _Clock, elapsed: float, expected: int) -> None:
        """9.4s left is 10. Truncation says 9, and so does a floor of 1 laid over truncation."""
        limiter = RateLimiter(requests_per_minute=1, window_seconds=10)
        limiter.check("k")

        clock.now = _START + elapsed

        assert limiter.check("k") == (False, 0, expected)

    @pytest.mark.parametrize(("elapsed", "expected"), _EXACT)
    def test_an_exact_remainder_is_not_bumped(self, clock: _Clock, elapsed: float, expected: int) -> None:
        """6.0s left stays 6. A repair that always adds one second fails here; one that rounds a half to even fails on 7 and 5."""
        limiter = RateLimiter(requests_per_minute=1, window_seconds=10)
        limiter.check("k")

        clock.now = _START + elapsed

        assert limiter.check("k") == (False, 0, expected)

    def test_a_fresh_rejection_reports_the_whole_window(self, clock: _Clock) -> None:
        """The floor of 1 must not flatten every answer to 1."""
        limiter = RateLimiter(requests_per_minute=1, window_seconds=60)
        limiter.check("k")

        assert limiter.check("k") == (False, 0, 60)

    def test_the_allowed_path_rounds_up_too(self, clock: _Clock) -> None:
        """The allowed path feeds ``X-RateLimit-Reset`` and carried the same truncation."""
        limiter = RateLimiter(requests_per_minute=5, window_seconds=10)
        assert limiter.check("k") == (True, 4, 10)

        clock.now = _START + 0.6
        assert limiter.check("k") == (True, 3, 10)

        clock.now = _START + 9.7
        assert limiter.check("k") == (True, 2, 1)

    @pytest.mark.parametrize(("elapsed", "expected"), _FRACTIONAL + _EXACT)
    def test_the_allowed_path_rounds_the_same_way(self, clock: _Clock, elapsed: float, expected: int) -> None:
        """Both ``check`` returns go through one helper. This holds the allowed one to every point the rejection meets."""
        limiter = RateLimiter(requests_per_minute=5, window_seconds=10)
        limiter.check("k")

        clock.now = _START + elapsed

        assert limiter.check("k") == (True, 3, expected)

    def test_a_window_longer_than_a_minute_is_not_capped(self, clock: _Clock) -> None:
        """Every other test uses a 10 s or a 60 s window, so a wait capped at 60 passes them all."""
        limiter = RateLimiter(requests_per_minute=1, window_seconds=120)
        limiter.check("k")

        clock.now = _START + 0.5

        assert limiter.check("k") == (False, 0, 120)

    def test_the_request_that_spends_the_last_slot_rounds_up_too(self, clock: _Clock) -> None:
        """The allowed path with nothing left: 6.6s left says 7, not the whole window."""
        limiter = RateLimiter(requests_per_minute=2, window_seconds=10)
        limiter.check("k")

        clock.now = _START + 3.4

        assert limiter.check("k") == (True, 0, 7)

    def test_the_window_still_rolls_over_at_the_boundary(self, clock: _Clock) -> None:
        """One millisecond before the boundary is still a 1-second wait; the boundary itself is a new window."""
        limiter = RateLimiter(requests_per_minute=1, window_seconds=10)
        limiter.check("k")

        clock.now = _START + 9.999
        assert limiter.check("k") == (False, 0, 1)

        clock.now = _START + 10.0
        assert limiter.check("k") == (True, 0, 10)

    def test_the_floor_holds_at_the_exact_boundary(self) -> None:
        """``ceil`` returns 0 at the boundary. ``check`` rolls the window over there, so only a direct call reaches the floor."""
        limiter = RateLimiter(requests_per_minute=1, window_seconds=10)

        assert limiter._seconds_until_reset(_START + 10.0, _START) == 1

    def test_the_429_carries_the_rounded_wait_in_both_headers(self, clock: _Clock) -> None:
        """End to end: the headers and the message a caller receives, not ``check``'s tuple."""
        limiter = RateLimiter(requests_per_minute=1, window_seconds=10, enabled=True)
        client = TestClient(_app(limiter, FailedAuthThrottle(enabled=False)))
        first = client.get("/v1/network")
        assert first.status_code == 200
        assert first.headers["X-RateLimit-Reset"] == "10"

        clock.now = _START + 9.7
        response = client.get("/v1/network")

        assert response.status_code == 429
        assert response.headers["Retry-After"] == "1"
        assert response.headers["X-RateLimit-Reset"] == "1"
        assert response.json()["detail"] == "Rate limit exceeded. Try again in 1 seconds."

    def test_the_429_names_the_rounded_wait_not_just_a_nonzero_one(self, clock: _Clock) -> None:
        """9.4s left: both headers and the message say 10. Truncation says 9, and a constant 1 passes the test above."""
        limiter = RateLimiter(requests_per_minute=1, window_seconds=10, enabled=True)
        client = TestClient(_app(limiter, FailedAuthThrottle(enabled=False)))
        client.get("/v1/network")

        clock.now = _START + 0.6
        response = client.get("/v1/network")

        assert response.status_code == 429
        assert response.headers["Retry-After"] == "10"
        assert response.headers["X-RateLimit-Reset"] == "10"
        assert response.json()["detail"] == "Rate limit exceeded. Try again in 10 seconds."

    def test_an_allowed_response_carries_the_rounded_reset(self, clock: _Clock) -> None:
        """The allowed path's value reaches ``X-RateLimit-Reset`` through the middleware: 10 with 9.4s left, and 1, never 0, in the last second."""
        limiter = RateLimiter(requests_per_minute=5, window_seconds=10, enabled=True)
        client = TestClient(_app(limiter, FailedAuthThrottle(enabled=False)))
        client.get("/v1/network")

        clock.now = _START + 0.6
        response = client.get("/v1/network")

        assert response.status_code == 200
        assert response.headers["X-RateLimit-Reset"] == "10"

        clock.now = _START + 9.7
        last_second = client.get("/v1/network")

        assert last_second.status_code == 200
        assert last_second.headers["X-RateLimit-Reset"] == "1"

    def test_the_wire_carries_what_is_left_of_the_budget(self, clock: _Clock) -> None:
        """``X-RateLimit-Remaining`` on an allowed response and on the 429."""
        limiter = RateLimiter(requests_per_minute=2, window_seconds=10, enabled=True)
        client = TestClient(_app(limiter, FailedAuthThrottle(enabled=False)))

        assert client.get("/v1/network").headers["X-RateLimit-Remaining"] == "1"
        assert client.get("/v1/network").headers["X-RateLimit-Remaining"] == "0"
        rejected = client.get("/v1/network")
        assert rejected.status_code == 429
        assert rejected.headers["X-RateLimit-Remaining"] == "0"


class TestFailedAuthThrottleRetryAfterRoundsUp:
    """``FailedAuthThrottle.check``'s wait, which becomes the pre-auth 429's ``Retry-After``."""

    @pytest.mark.parametrize(("elapsed", "expected"), _FRACTIONAL)
    def test_a_blocked_wait_rounds_any_fraction_up(self, clock: _Clock, elapsed: float, expected: int) -> None:
        """9.4s left is 10, not 9. Truncation and round-half-up both say 9."""
        throttle = FailedAuthThrottle(max_failures=1, window_seconds=10)
        throttle.record_failure("1.2.3.4")

        clock.now = _START + elapsed

        assert throttle.check("1.2.3.4") == (True, expected)

    @pytest.mark.parametrize(("elapsed", "expected"), _EXACT)
    def test_an_exact_remainder_is_not_bumped(self, clock: _Clock, elapsed: float, expected: int) -> None:
        """6.0s left stays 6. A repair that always adds one second fails here; one that rounds a half to even fails on 7 and 5."""
        throttle = FailedAuthThrottle(max_failures=1, window_seconds=10)
        throttle.record_failure("1.2.3.4")

        clock.now = _START + elapsed

        assert throttle.check("1.2.3.4") == (True, expected)

    def test_a_subsecond_remainder_is_one_second(self, clock: _Clock) -> None:
        """A sub-second remainder is one second, never 0. The old floor held this too: it pins the floor under the new rounding."""
        throttle = FailedAuthThrottle(max_failures=1, window_seconds=1)
        throttle.record_failure("1.2.3.4")

        for elapsed in (0.30, 0.60, 0.90, 0.99):
            clock.now = _START + elapsed
            assert throttle.check("1.2.3.4") == (True, 1)

    def test_a_fresh_block_reports_the_whole_window(self, clock: _Clock) -> None:
        """The floor of 1 must not flatten every answer to 1."""
        throttle = FailedAuthThrottle(max_failures=1, window_seconds=60)
        throttle.record_failure("1.2.3.4")

        assert throttle.check("1.2.3.4") == (True, 60)

    def test_under_budget_and_a_rolled_window_report_zero(self, clock: _Clock) -> None:
        """The wait is 0 unless the source is blocked inside the current window, the boundary included."""
        throttle = FailedAuthThrottle(max_failures=2, window_seconds=10)
        throttle.record_failure("1.2.3.4")
        assert throttle.check("1.2.3.4") == (False, 0)

        throttle.record_failure("1.2.3.4")
        assert throttle.check("1.2.3.4")[0] is True

        clock.now = _START + 10.0
        assert throttle.check("1.2.3.4") == (False, 0)

    def test_the_pre_auth_429_carries_the_rounded_wait(self, clock: _Clock) -> None:
        """The header a caller receives. The key is valid and the limiter is far from its budget, so this 429 is the throttle's."""
        throttle = FailedAuthThrottle(max_failures=1, window_seconds=10)
        throttle.record_failure("testclient")
        clock.now = _START + 0.6
        client = TestClient(_app(RateLimiter(requests_per_minute=1000, enabled=True), throttle, api_keys=["s3cret"]))

        response = client.get("/v1/network", headers={"X-API-Key": "s3cret"})

        assert response.status_code == 429
        assert response.headers["Retry-After"] == "10"
        assert response.json()["detail"] == "Too many failed authentication attempts. Try again in 10 seconds."

    @pytest.mark.parametrize(("window", "elapsed", "expected"), [(10, 3.4, "7"), (10, 9.7, "1"), (120, 0.5, "120")])
    def test_the_pre_auth_429_names_the_wait_not_the_window(self, clock: _Clock, window: int, elapsed: float, expected: str) -> None:
        """6.6s left says 7 (truncation says 6, the window is 10); 0.3s left says 1, never 0; 119.5s left says 120.

        The point in the test above (0.6 s into a 10 s window) rounds up to 10, which is also the window: a
        middleware that sends the whole window, a constant, or a capped value passes it.
        """
        throttle = FailedAuthThrottle(max_failures=1, window_seconds=window)
        throttle.record_failure("testclient")
        clock.now = _START + elapsed
        client = TestClient(_app(RateLimiter(requests_per_minute=1000, enabled=True), throttle, api_keys=["s3cret"]))

        response = client.get("/v1/network", headers={"X-API-Key": "s3cret"})

        assert response.status_code == 429
        assert response.headers["Retry-After"] == expected
        assert response.json()["detail"] == f"Too many failed authentication attempts. Try again in {expected} seconds."

    def test_a_window_longer_than_a_minute_is_not_capped(self, clock: _Clock) -> None:
        """Every other throttle test uses a 1 s, 10 s or 60 s window, so a wait capped at 60 passes them all."""
        throttle = FailedAuthThrottle(max_failures=1, window_seconds=120)
        throttle.record_failure("1.2.3.4")

        clock.now = _START + 0.5

        assert throttle.check("1.2.3.4") == (True, 120)
