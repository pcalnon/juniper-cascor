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

    def test_get_settings_cached(self):
        """Test that get_settings returns cached instance."""
        from api.settings import get_settings

        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
        get_settings.cache_clear()
