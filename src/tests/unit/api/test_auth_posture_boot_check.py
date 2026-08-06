"""Tests for the SEC-F01 boot-time auth-posture self-check.

The lifespan calls juniper-service-core's ``enforce_auth_posture(settings.api_keys,
require_auth=settings.require_auth, service_name="juniper-cascor")`` right after the
bind-attestation guard — before serving — so an empty/placeholder
``JUNIPER_CASCOR_API_KEYS`` secret (which silently disables ``APIKeyAuth`` and serves
protected routes open behind a healthy health check — the HO-2 incident class) is
LOUD at boot, and — with ``JUNIPER_CASCOR_REQUIRE_AUTH=true`` (default false) — a
boot FAILURE instead (the fail-closed posture for deployments where secrets are
provisioned).

The wiring test monkeypatches the module attribute with a recorder that raises a
sentinel, proving the lifespan invokes the check (with the resolved keys, before
any background machinery starts) without running the heavy startup tail. The
behavioural tests exercise the helper directly.
"""

import asyncio

import pytest

from api.app import create_app, lifespan
from api.settings import Settings

pytestmark = pytest.mark.unit


class _Sentinel(Exception):
    """Raised by the recorder to stop the lifespan right after the posture check."""


class TestAuthPostureLifespanWiring:
    """The check is actually invoked at application startup (before serving)."""

    @pytest.mark.parametrize("require_auth", [False, True])
    def test_lifespan_invokes_posture_check_with_resolved_keys(self, monkeypatch, require_auth):
        calls: list[tuple[list[str], bool, str]] = []

        def _recorder(api_keys, *, require_auth, service_name, logger=None, **_kwargs):
            calls.append((list(api_keys or []), require_auth, service_name))
            raise _Sentinel

        monkeypatch.setattr("api.app.enforce_auth_posture", _recorder)
        app = create_app(Settings(api_keys=["k1", "k2"], auto_start=False, require_auth=require_auth))

        async def _enter() -> None:
            async with lifespan(app):
                pass  # pragma: no cover — the recorder raises before yield

        with pytest.raises(_Sentinel):
            asyncio.run(_enter())
        assert calls == [(["k1", "k2"], require_auth, "juniper-cascor")]

    def test_require_auth_defaults_to_false(self):
        # Default keeps today's loud-WARNING posture; deployments opt in to
        # fail-closed explicitly (the composed stack sets the env flag).
        assert Settings.model_fields["require_auth"].default is False

    def test_env_flag_flips_posture(self, monkeypatch):
        monkeypatch.setenv("JUNIPER_CASCOR_REQUIRE_AUTH", "true")
        assert Settings().require_auth is True

    def test_required_with_no_keys_refuses_startup(self, monkeypatch):
        # The fail-closed posture end-to-end: the REAL lifespan raises
        # AuthPostureError right after the bind guard, before any background
        # machinery starts, so uvicorn startup fails instead of serving open.
        from juniper_service_core import AuthPostureError

        monkeypatch.delenv("JUNIPER_SKIP_AUTH_POSTURE_CHECK", raising=False)
        app = create_app(Settings(api_keys=None, auto_start=False, require_auth=True))

        async def _enter() -> None:
            async with lifespan(app):
                pass  # pragma: no cover — the posture check raises before yield

        with pytest.raises(AuthPostureError):
            asyncio.run(_enter())

    def test_create_app_itself_does_not_invoke_the_check(self, monkeypatch):
        # Construction must stay check-free — the posture fires at startup
        # (lifespan), mirroring the bind guard's construction/startup split.
        monkeypatch.setattr("api.app.enforce_auth_posture", lambda *a, **k: (_ for _ in ()).throw(_Sentinel))
        app = create_app(Settings(api_keys=None, auto_start=False))
        assert app.state.settings.api_keys is None

    def test_empty_api_keys_file_with_require_auth_refuses_startup(self, monkeypatch, tmp_path):
        """HO-2 end-to-end: empty Docker secret mount + require_auth must
        fail closed at lifespan — Settings resolves ``None``, posture raises
        before any background machinery starts."""
        from juniper_service_core import AuthPostureError

        secret_path = tmp_path / "juniper_cascor_api_keys"
        secret_path.write_text("")
        monkeypatch.setenv("JUNIPER_CASCOR_API_KEYS_FILE", str(secret_path))
        monkeypatch.delenv("JUNIPER_CASCOR_API_KEYS", raising=False)
        monkeypatch.delenv("JUNIPER_SKIP_AUTH_POSTURE_CHECK", raising=False)

        settings = Settings(auto_start=False, require_auth=True)
        assert settings.api_keys is None
        app = create_app(settings)

        async def _enter() -> None:
            async with lifespan(app):
                pass  # pragma: no cover — posture raises before yield

        with pytest.raises(AuthPostureError):
            asyncio.run(_enter())

    def test_whitespace_api_keys_file_with_require_auth_refuses_startup(self, monkeypatch, tmp_path):
        """Whitespace-only secret files strip to empty; require_auth must refuse."""
        from juniper_service_core import AuthPostureError

        secret_path = tmp_path / "juniper_cascor_api_keys"
        secret_path.write_text("  \n\t  ")
        monkeypatch.setenv("JUNIPER_CASCOR_API_KEYS_FILE", str(secret_path))
        monkeypatch.delenv("JUNIPER_CASCOR_API_KEYS", raising=False)
        monkeypatch.delenv("JUNIPER_SKIP_AUTH_POSTURE_CHECK", raising=False)

        settings = Settings(auto_start=False, require_auth=True)
        assert settings.api_keys is None
        app = create_app(settings)

        async def _enter() -> None:
            async with lifespan(app):
                pass  # pragma: no cover — posture raises before yield

        with pytest.raises(AuthPostureError):
            asyncio.run(_enter())

    def test_commas_only_api_keys_file_with_require_auth_refuses_startup(self, monkeypatch, tmp_path):
        """``, ,`` parses to ``[]`` (unconfigured); require_auth must refuse
        the same way as ``None`` — not serve open behind a healthy probe."""
        from juniper_service_core import AuthPostureError

        secret_path = tmp_path / "juniper_cascor_api_keys"
        secret_path.write_text(", ,")
        monkeypatch.setenv("JUNIPER_CASCOR_API_KEYS_FILE", str(secret_path))
        monkeypatch.delenv("JUNIPER_CASCOR_API_KEYS", raising=False)
        monkeypatch.delenv("JUNIPER_SKIP_AUTH_POSTURE_CHECK", raising=False)

        settings = Settings(auto_start=False, require_auth=True)
        assert settings.api_keys == []
        app = create_app(settings)

        async def _enter() -> None:
            async with lifespan(app):
                pass  # pragma: no cover — posture raises before yield

        with pytest.raises(AuthPostureError):
            asyncio.run(_enter())


class TestAuthPostureBehaviour:
    """The helper's three outcomes, exercised directly (hermetic)."""

    def test_no_keys_and_not_required_warns_open(self, monkeypatch, caplog):
        from juniper_service_core import enforce_auth_posture

        monkeypatch.delenv("JUNIPER_SKIP_AUTH_POSTURE_CHECK", raising=False)
        with caplog.at_level("WARNING"):
            enforce_auth_posture(None, require_auth=False, service_name="juniper-cascor")
        assert any("running OPEN" in rec.getMessage() and "juniper-cascor" in rec.getMessage() for rec in caplog.records)

    def test_blank_key_counts_as_unset(self, monkeypatch, caplog):
        # Exactly what an empty secret file resolves to (the HO-2 class).
        from juniper_service_core import auth_is_configured, enforce_auth_posture

        assert not auth_is_configured([""])
        assert not auth_is_configured(["   "])
        monkeypatch.delenv("JUNIPER_SKIP_AUTH_POSTURE_CHECK", raising=False)
        with caplog.at_level("WARNING"):
            enforce_auth_posture(["   "], require_auth=False, service_name="juniper-cascor")
        assert any("running OPEN" in rec.getMessage() for rec in caplog.records)

    def test_real_key_passes_quietly(self, monkeypatch, caplog):
        from juniper_service_core import enforce_auth_posture

        monkeypatch.delenv("JUNIPER_SKIP_AUTH_POSTURE_CHECK", raising=False)
        with caplog.at_level("INFO"):
            enforce_auth_posture(["a-real-cascor-key"], require_auth=True, service_name="juniper-cascor")
        assert not any(rec.levelname in ("WARNING", "CRITICAL") for rec in caplog.records)

    def test_required_with_no_key_raises(self, monkeypatch):
        # The fail-closed posture the follow-up flag will enable.
        from juniper_service_core import AuthPostureError, enforce_auth_posture

        monkeypatch.delenv("JUNIPER_SKIP_AUTH_POSTURE_CHECK", raising=False)
        with pytest.raises(AuthPostureError):
            enforce_auth_posture([], require_auth=True, service_name="juniper-cascor")

    def test_escape_hatch_bypasses_the_check(self, monkeypatch):
        from juniper_service_core import enforce_auth_posture

        monkeypatch.setenv("JUNIPER_SKIP_AUTH_POSTURE_CHECK", "1")
        enforce_auth_posture([], require_auth=True, service_name="juniper-cascor")
