"""Tests for Docker secrets utility (api.secrets)."""

import pytest


@pytest.mark.unit
class TestGetSecret:
    """Test get_secret() reads secrets from files and environment variables."""

    def test_returns_none_when_neither_set(self, monkeypatch):
        """get_secret returns None when no env var or file is available."""
        monkeypatch.delenv("MY_SECRET", raising=False)
        monkeypatch.delenv("MY_SECRET_FILE", raising=False)

        from api.secrets import get_secret

        assert get_secret("MY_SECRET") is None

    def test_reads_from_env_var(self, monkeypatch):
        """get_secret returns the plain env var value when set."""
        monkeypatch.setenv("MY_SECRET", "env-value")
        monkeypatch.delenv("MY_SECRET_FILE", raising=False)

        from api.secrets import get_secret

        assert get_secret("MY_SECRET") == "env-value"

    def test_reads_from_file(self, monkeypatch, tmp_path):
        """get_secret reads from a file pointed to by the _FILE env var."""
        secret_file = tmp_path / "secret.txt"
        secret_file.write_text("file-value\n")

        monkeypatch.delenv("MY_SECRET", raising=False)
        monkeypatch.setenv("MY_SECRET_FILE", str(secret_file))

        from api.secrets import get_secret

        assert get_secret("MY_SECRET") == "file-value"

    def test_file_takes_precedence_over_env_var(self, monkeypatch, tmp_path):
        """When both file and env var are set, file wins."""
        secret_file = tmp_path / "secret.txt"
        secret_file.write_text("file-value\n")

        monkeypatch.setenv("MY_SECRET", "env-value")
        monkeypatch.setenv("MY_SECRET_FILE", str(secret_file))

        from api.secrets import get_secret

        assert get_secret("MY_SECRET") == "file-value"

    def test_default_file_env_var_naming(self, monkeypatch, tmp_path):
        """Default file_env_var is <env_var>_FILE."""
        secret_file = tmp_path / "key.txt"
        secret_file.write_text("auto-named\n")

        monkeypatch.setenv("JUNIPER_DATA_API_KEY_FILE", str(secret_file))
        monkeypatch.delenv("JUNIPER_DATA_API_KEY", raising=False)

        from api.secrets import get_secret

        assert get_secret("JUNIPER_DATA_API_KEY") == "auto-named"

    def test_custom_file_env_var(self, monkeypatch, tmp_path):
        """A custom file_env_var name can be supplied explicitly."""
        secret_file = tmp_path / "custom.txt"
        secret_file.write_text("custom-path\n")

        monkeypatch.setenv("MY_CUSTOM_FILE_VAR", str(secret_file))
        monkeypatch.delenv("MY_SECRET", raising=False)
        monkeypatch.delenv("MY_SECRET_FILE", raising=False)

        from api.secrets import get_secret

        assert get_secret("MY_SECRET", file_env_var="MY_CUSTOM_FILE_VAR") == "custom-path"

    def test_file_not_found_falls_back_to_env_var(self, monkeypatch):
        """When _FILE points to a nonexistent path, fall back to env var."""
        monkeypatch.setenv("MY_SECRET", "fallback-value")
        monkeypatch.setenv("MY_SECRET_FILE", "/nonexistent/path/secret.txt")

        from api.secrets import get_secret

        assert get_secret("MY_SECRET") == "fallback-value"

    def test_strips_whitespace_from_file(self, monkeypatch, tmp_path):
        """Secret values read from files are stripped of surrounding whitespace."""
        secret_file = tmp_path / "secret.txt"
        secret_file.write_text("  spaced-value  \n")

        monkeypatch.setenv("MY_SECRET_FILE", str(secret_file))
        monkeypatch.delenv("MY_SECRET", raising=False)

        from api.secrets import get_secret

        assert get_secret("MY_SECRET") == "spaced-value"

    def test_empty_file_returns_empty_string(self, monkeypatch, tmp_path):
        """An empty mounted secret file is a present file — returns '' after strip.

        Distinct from a missing file (which falls back to the plain env var).
        Compose placeholder mounts often create empty files; callers must treat
        '' as an explicit empty secret, not as unset.
        """
        secret_file = tmp_path / "secret.txt"
        secret_file.write_text("")

        monkeypatch.delenv("MY_SECRET", raising=False)
        monkeypatch.setenv("MY_SECRET_FILE", str(secret_file))

        from api.secrets import get_secret

        assert get_secret("MY_SECRET") == ""

    def test_whitespace_only_file_returns_empty_string(self, monkeypatch, tmp_path):
        """Whitespace-only secret files strip to '' (same contract as empty)."""
        secret_file = tmp_path / "secret.txt"
        secret_file.write_text("  \n\t\n")

        monkeypatch.delenv("MY_SECRET", raising=False)
        monkeypatch.setenv("MY_SECRET_FILE", str(secret_file))

        from api.secrets import get_secret

        assert get_secret("MY_SECRET") == ""

    def test_empty_file_wins_over_env_var(self, monkeypatch, tmp_path):
        """File presence wins even when the file content is blank — no env fallback.

        This is the dangerous Docker-secrets footgun: a mounted empty
        ``*_FILE`` suppresses the plain env var and yields an empty API key /
        open-auth posture rather than the non-empty env fallback.
        """
        secret_file = tmp_path / "secret.txt"
        secret_file.write_text("")

        monkeypatch.setenv("MY_SECRET", "fallback-value")
        monkeypatch.setenv("MY_SECRET_FILE", str(secret_file))

        from api.secrets import get_secret

        assert get_secret("MY_SECRET") == ""
