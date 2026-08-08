"""Tests for the experiment YAML config layer (Wave 3.1 -- CLI experimentation plan SS5.1/SS5.2/SS5.6).

``ExperimentYamlSettingsSource`` projects ONLY the experiment YAML's ``service:`` block
into ``Settings`` (YAML > env in the SS5.1 precedence), validating fail-loud: unknown
top-level blocks, ``schema_version``, unknown ``service:`` keys (the model is
``extra="ignore"``, so silent dropping must be impossible), and the launcher-owned
infra keys (SS5.6 rule 6). The layer is inert when ``JUNIPER_CASCOR_CONFIG_FILE`` is
unset, and init-kwargs (the CLI tier) still beat the YAML.
"""

import pytest

pytestmark = pytest.mark.unit

_BASE_YAML = """
schema_version: 1
experiment:
  name: layer-test
  seed: 1
service:
  log_level: DEBUG
  metrics_enabled: true
dataset:
  generator: spiral
  params:
    seed: 1
training:
  params:
    max_iterations: 2
runtime:
  blas_threads: 2
outputs:
  plots: []
"""


def _write_yaml(tmp_path, body):
    path = tmp_path / "experiment.yaml"
    path.write_text(body, encoding="utf-8")
    return path


class TestExperimentYamlLayer:
    """SS5.1 precedence + inertness."""

    def test_inert_without_env_var(self, monkeypatch):
        monkeypatch.delenv("JUNIPER_CASCOR_CONFIG_FILE", raising=False)
        monkeypatch.setenv("JUNIPER_CASCOR_LOG_LEVEL", "WARNING")
        from api.settings import Settings

        assert Settings().log_level == "WARNING"

    def test_yaml_beats_env(self, tmp_path, monkeypatch):
        path = _write_yaml(tmp_path, _BASE_YAML)
        monkeypatch.setenv("JUNIPER_CASCOR_CONFIG_FILE", str(path))
        monkeypatch.setenv("JUNIPER_CASCOR_LOG_LEVEL", "ERROR")
        monkeypatch.setenv("JUNIPER_CASCOR_METRICS_ENABLED", "false")
        from api.settings import Settings

        settings = Settings()
        assert settings.log_level == "DEBUG"
        assert settings.metrics_enabled is True

    def test_init_kwargs_beat_yaml(self, tmp_path, monkeypatch):
        path = _write_yaml(tmp_path, _BASE_YAML)
        monkeypatch.setenv("JUNIPER_CASCOR_CONFIG_FILE", str(path))
        from api.settings import Settings

        assert Settings(log_level="WARNING").log_level == "WARNING"

    def test_env_still_wins_for_unprojected_fields(self, tmp_path, monkeypatch):
        path = _write_yaml(tmp_path, _BASE_YAML)
        monkeypatch.setenv("JUNIPER_CASCOR_CONFIG_FILE", str(path))
        monkeypatch.setenv("JUNIPER_CASCOR_AUTO_START", "true")
        from api.settings import Settings

        settings = Settings()
        assert settings.auto_start is True
        assert settings.log_level == "DEBUG"

    def test_non_service_blocks_are_ignored_by_settings(self, tmp_path, monkeypatch):
        path = _write_yaml(tmp_path, _BASE_YAML)
        monkeypatch.setenv("JUNIPER_CASCOR_CONFIG_FILE", str(path))
        from api.settings import Settings

        settings = Settings()
        assert settings.host == "127.0.0.1"
        assert settings.port == 8200


class TestExperimentYamlValidation:
    """SS5.6 rules 1/2/6 -- fail loud before boot."""

    def _expect_error(self, tmp_path, monkeypatch, body, match):
        from api.settings import ExperimentConfigError, Settings

        path = _write_yaml(tmp_path, body)
        monkeypatch.setenv("JUNIPER_CASCOR_CONFIG_FILE", str(path))
        with pytest.raises(ExperimentConfigError, match=match):
            Settings()

    @pytest.mark.parametrize("key", ["host", "port", "juniper_data_url", "eval_metrics_enabled"])
    def test_launcher_owned_service_keys_rejected(self, tmp_path, monkeypatch, key):
        body = f"schema_version: 1\nservice:\n  {key}: anything\n"
        self._expect_error(tmp_path, monkeypatch, body, match="rule 6")

    def test_unknown_service_key_rejected(self, tmp_path, monkeypatch):
        body = "schema_version: 1\nservice:\n  log_levle: DEBUG\n"
        self._expect_error(tmp_path, monkeypatch, body, match="log_levle")

    def test_unknown_top_level_block_rejected(self, tmp_path, monkeypatch):
        body = "schema_version: 1\nsurprise: {}\nservice:\n  log_level: DEBUG\n"
        self._expect_error(tmp_path, monkeypatch, body, match="surprise")

    def test_schema_version_required(self, tmp_path, monkeypatch):
        body = "service:\n  log_level: DEBUG\n"
        self._expect_error(tmp_path, monkeypatch, body, match="schema_version")

    def test_future_schema_version_rejected(self, tmp_path, monkeypatch):
        body = "schema_version: 99\nservice:\n  log_level: DEBUG\n"
        self._expect_error(tmp_path, monkeypatch, body, match="schema_version")

    def test_missing_file_fails_loud(self, tmp_path, monkeypatch):
        from api.settings import ExperimentConfigError, Settings

        monkeypatch.setenv("JUNIPER_CASCOR_CONFIG_FILE", str(tmp_path / "nope.yaml"))
        with pytest.raises(ExperimentConfigError, match="unreadable"):
            Settings()

    def test_non_mapping_yaml_rejected(self, tmp_path, monkeypatch):
        self._expect_error(tmp_path, monkeypatch, "- just\n- a\n- list\n", match="mapping")

    def test_invalid_yaml_rejected(self, tmp_path, monkeypatch):
        self._expect_error(tmp_path, monkeypatch, "service: [unclosed\n", match="not valid YAML")
