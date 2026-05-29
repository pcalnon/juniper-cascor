"""Tests for API settings module."""

import pytest


@pytest.mark.unit
class TestSettings:
    """Test Settings configuration."""

    def test_default_settings(self):
        """Test default settings values."""
        from api.settings import Settings

        settings = Settings()
        assert settings.host == "127.0.0.1"
        assert settings.port == 8200
        assert settings.log_level == "INFO"
        assert settings.cors_origins == []
        assert settings.ws_max_connections == 50
        assert settings.ws_heartbeat_interval_sec == 30

    def test_settings_env_override(self, monkeypatch):
        """Test settings override via environment variables."""
        monkeypatch.setenv("JUNIPER_CASCOR_HOST", "0.0.0.0")
        monkeypatch.setenv("JUNIPER_CASCOR_PORT", "9999")
        monkeypatch.setenv("JUNIPER_CASCOR_LOG_LEVEL", "DEBUG")

        from api.settings import Settings

        settings = Settings()
        assert settings.host == "0.0.0.0"
        assert settings.port == 9999
        assert settings.log_level == "DEBUG"

    def test_worker_settings_defaults(self):
        """Test default worker settings for remote WebSocket workers."""
        from api.settings import Settings

        settings = Settings()
        assert settings.remote_workers_heartbeat_timeout == 30.0
        assert settings.remote_workers_task_reassignment_timeout == 120.0

    def test_worker_settings_env_override(self, monkeypatch):
        """Test worker settings override via environment variables."""
        monkeypatch.setenv("JUNIPER_CASCOR_REMOTE_WORKERS_HEARTBEAT_TIMEOUT", "15.0")
        monkeypatch.setenv("JUNIPER_CASCOR_REMOTE_WORKERS_TASK_REASSIGNMENT_TIMEOUT", "60.0")

        from api.settings import Settings

        settings = Settings()
        assert settings.remote_workers_heartbeat_timeout == 15.0
        assert settings.remote_workers_task_reassignment_timeout == 60.0

    def test_get_settings_cached(self):
        """Test that get_settings returns cached instance."""
        from api.settings import get_settings

        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
        get_settings.cache_clear()


@pytest.mark.unit
class TestApiKeysParser:
    """``_parse_api_keys`` must normalise every shape the secret-file
    pipeline can produce to a ``list[str] | None``.

    The historical ``_empty_string_to_none`` validator only handled the
    empty-string case; a bare string like
    ``CHANGE_BEFORE_PRODUCTION_USE`` (the placeholder shipped in
    juniper-deploy's ``secrets.example/juniper_cascor_api_keys.txt``)
    fell through to pydantic's ``list[str]`` coercion and crashed the
    container with ``list_type`` ValidationError.

    Mirrors the contract pinned by juniper-data's ``_parse_api_keys``.
    """

    def test_none_becomes_none(self):
        from api.settings import Settings

        settings = Settings(api_keys=None)
        assert settings.api_keys is None

    def test_empty_string_becomes_none(self):
        """An empty Docker secret file (0-byte) yields ``""`` from
        ``get_secret``; that must disable auth, not crash."""
        from api.settings import Settings

        settings = Settings(api_keys="")
        assert settings.api_keys is None

    def test_bare_string_becomes_single_element_list(self):
        """A non-empty secret file containing a single token (the typical
        out-of-the-box deploy placeholder) becomes a one-element list."""
        from api.settings import Settings

        settings = Settings(api_keys="CHANGE_BEFORE_PRODUCTION_USE")
        assert settings.api_keys == ["CHANGE_BEFORE_PRODUCTION_USE"]

    def test_comma_separated_string_is_split(self):
        """CSV input is the documented shape for multi-key secret files —
        same convention as juniper-data."""
        from api.settings import Settings

        settings = Settings(api_keys="key1,key2,key3")
        assert settings.api_keys == ["key1", "key2", "key3"]

    def test_comma_separated_string_strips_whitespace_and_empties(self):
        """``"key1, key2 , ,"`` → ``["key1", "key2"]`` (trim + drop empties)."""
        from api.settings import Settings

        settings = Settings(api_keys="key1, key2 , ,")
        assert settings.api_keys == ["key1", "key2"]

    def test_list_input_passed_through(self):
        """Direct list input (e.g. from a JSON-deserialised
        ``pydantic-settings`` env var, or a programmatic constructor)
        must round-trip without re-parsing."""
        from api.settings import Settings

        settings = Settings(api_keys=["key1", "key2"])
        assert settings.api_keys == ["key1", "key2"]

    def test_json_list_string_auto_parsed_by_pydantic(self, monkeypatch):
        """``["k1","k2"]`` as a string env var should reach the validator
        as a list (pydantic-settings auto-parses JSON for complex
        fields). This pins the path the deploy-side JSON-array
        placeholder fix relies on."""
        monkeypatch.setenv("JUNIPER_CASCOR_API_KEYS", '["k1","k2"]')
        from api.settings import Settings

        settings = Settings()
        assert settings.api_keys == ["k1", "k2"]

    def test_secret_file_with_bare_placeholder_does_not_crash(self, monkeypatch, tmp_path):
        """End-to-end regression: a Docker secret file containing the
        bare ``CHANGE_BEFORE_PRODUCTION_USE`` placeholder must NOT
        raise ``list_type`` ValidationError. This is the exact failure
        that caused the ``juniper-cascor`` restart loop after the
        CVE-2026-48710 alertmanager port-conflict recovery."""
        secret_path = tmp_path / "juniper_cascor_api_keys"
        secret_path.write_text("CHANGE_BEFORE_PRODUCTION_USE")
        monkeypatch.setenv("JUNIPER_CASCOR_API_KEYS_FILE", str(secret_path))
        # Also clear any direct env var that would short-circuit the file
        # read path inside ``get_secret``.
        monkeypatch.delenv("JUNIPER_CASCOR_API_KEYS", raising=False)
        from api.settings import Settings

        settings = Settings()
        assert settings.api_keys == ["CHANGE_BEFORE_PRODUCTION_USE"]


@pytest.mark.unit
class TestWsControlAllowedOriginsParser:
    """E.2 PR-2-B regression: ``ws_control_allowed_origins`` accepts the
    juniper-deploy compose env-var form (comma-CSV) as well as the
    JSON-array form that pydantic-settings auto-emits for ``list[str]``
    fields. See juniper-ml notes/STACK_REGRESSION_CORRECTIONS_2026-05-27.md
    §E.2.

    The default allowlist is preserved (``localhost:8050`` /
    ``127.0.0.1:8050``); when the operator sets
    ``JUNIPER_CASCOR_WS_CONTROL_ALLOWED_ORIGINS`` to
    ``"http://juniper-canopy:8050,http://localhost:8050"``, the parser
    splits on commas — without this PR, that input raised
    ``SettingsError: error parsing value for field
    "ws_control_allowed_origins" from source "EnvSettingsSource"`` and
    blocked the canopy reconnection unblock.
    """

    def test_default_allowlist_kept_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("JUNIPER_CASCOR_WS_CONTROL_ALLOWED_ORIGINS", raising=False)
        from api.settings import Settings

        settings = Settings()
        assert settings.ws_control_allowed_origins == [
            "http://localhost:8050",
            "http://127.0.0.1:8050",
            "https://localhost:8050",
            "https://127.0.0.1:8050",
        ]

    def test_csv_env_var_parsed_into_list(self, monkeypatch):
        monkeypatch.setenv(
            "JUNIPER_CASCOR_WS_CONTROL_ALLOWED_ORIGINS",
            "http://juniper-canopy:8050,http://localhost:8050,http://127.0.0.1:8050",
        )
        from api.settings import Settings

        settings = Settings()
        assert settings.ws_control_allowed_origins == [
            "http://juniper-canopy:8050",
            "http://localhost:8050",
            "http://127.0.0.1:8050",
        ]

    def test_json_array_env_var_parsed_into_list(self, monkeypatch):
        """JSON-array form (the default pydantic-settings shape for
        ``list[str]`` env vars) must still work after the ``NoDecode``
        annotation defers parsing to the validator.
        """
        monkeypatch.setenv(
            "JUNIPER_CASCOR_WS_CONTROL_ALLOWED_ORIGINS",
            '["http://x:1","http://y:2"]',
        )
        from api.settings import Settings

        settings = Settings()
        assert settings.ws_control_allowed_origins == ["http://x:1", "http://y:2"]

    def test_csv_whitespace_trimmed(self, monkeypatch):
        monkeypatch.setenv(
            "JUNIPER_CASCOR_WS_CONTROL_ALLOWED_ORIGINS",
            "  http://a:1 ,   http://b:2 ,  ",
        )
        from api.settings import Settings

        settings = Settings()
        assert settings.ws_control_allowed_origins == ["http://a:1", "http://b:2"]

    def test_empty_env_var_yields_empty_list(self, monkeypatch):
        """Operator opting out of all origins (``X=`` in env-file).
        Distinct from "env-var unset" (default applies).
        """
        monkeypatch.setenv("JUNIPER_CASCOR_WS_CONTROL_ALLOWED_ORIGINS", "")
        from api.settings import Settings

        settings = Settings()
        assert settings.ws_control_allowed_origins == []

    def test_programmatic_list_passthrough(self, monkeypatch):
        """Tests that construct ``Settings(ws_control_allowed_origins=[…])``
        directly still receive the list unchanged.
        """
        monkeypatch.delenv("JUNIPER_CASCOR_WS_CONTROL_ALLOWED_ORIGINS", raising=False)
        from api.settings import Settings

        settings = Settings(ws_control_allowed_origins=["http://prog:1", "http://prog:2"])
        assert settings.ws_control_allowed_origins == ["http://prog:1", "http://prog:2"]
