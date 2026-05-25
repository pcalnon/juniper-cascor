"""Regression: a developer's local .env must not pollute the test session.

Background. ``src/api/settings.py`` configures pydantic-settings with
``env_file=".env"``, so every ``Settings()`` constructor call reads
``./.env`` (the gitignored, developer-local copy of ``.env.example``).
pydantic-settings layers .env *under* ``os.environ``, which means
``monkeypatch.delenv("JUNIPER_DATA_URL")`` removes the OS-level value
but leaves the .env value in effect.

This silently breaks every test that asserts "field default applies
when no env var is set" — concretely, 18 tests across
``test_cfg_04_juniper_data_url_settings.py``,
``test_data_provider_coverage.py``, ``test_spiral_data_provider.py``,
``test_spiral_problem_juniper_data_integration.py``,
``test_api_health.py``, and ``test_r2_1_4_wire_compat.py`` — whenever
a developer has e.g. ``JUNIPER_DATA_URL=http://127.0.0.1:8100`` in
their local ``.env``. CI never reproduces it because runner checkouts
have no ``.env``.

The fix is an autouse session-scoped fixture in
``src/tests/conftest.py`` (``_disable_settings_env_file_for_tests``)
that sets ``Settings.model_config["env_file"] = None`` for the
session. This regression test pins that behavior so a future refactor
that drops or breaks the fixture fails loudly here, not via 18
mysterious test failures elsewhere.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Mirror conftest.py's sys.path bootstrapping so this test can import
# `api.settings` whether it's run via ``pytest src/tests/...`` or
# ``cd src && pytest tests/...``.
_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from api.settings import Settings  # noqa: E402

pytestmark = pytest.mark.unit


class TestSettingsEnvFileIsolation:
    """Pin the conftest autouse fixture that disables .env loading in tests."""

    def test_settings_env_file_is_none_during_test_session(self):
        """The session-scoped autouse fixture must set env_file to None.

        Direct read of ``Settings.model_config["env_file"]`` after pytest
        has loaded conftest.py. If this assertion fails, the
        ``_disable_settings_env_file_for_tests`` fixture has been
        dropped, renamed, or its scope/autouse semantics broken.
        """
        assert Settings.model_config.get("env_file") is None, "Settings.model_config['env_file'] should be None for the test session. " "Check src/tests/conftest.py::_disable_settings_env_file_for_tests."

    def test_local_dot_env_in_cwd_does_not_leak_into_settings(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Behavioral check: a .env in CWD must not override class defaults.

        Synthesizes a temp directory with a polluting ``.env`` that
        sets ``JUNIPER_DATA_URL`` and ``JUNIPER_CASCOR_HOST``, chdirs
        into it, clears the same env vars from ``os.environ``, and
        verifies that ``Settings()`` returns the class defaults rather
        than the values written to the file. This exercises the actual
        anti-regression contract — even if the fixture's mechanism
        changes (e.g. from ``env_file=None`` to a chdir+stub
        approach), as long as ``.env`` does not leak in, this test
        passes.
        """
        env_file = tmp_path / ".env"
        env_file.write_text(
            "JUNIPER_DATA_URL=http://leaked-from-dot-env:8100\n" "JUNIPER_CASCOR_HOST=192.0.2.123\n",
            encoding="utf-8",
        )

        monkeypatch.chdir(tmp_path)
        # Strip both alias forms in case the host environment has them
        # exported (devs running tests inside an activated shell often do).
        for var in ("JUNIPER_DATA_URL", "JUNIPER_CASCOR_JUNIPER_DATA_URL", "JUNIPER_CASCOR_HOST"):
            monkeypatch.delenv(var, raising=False)

        # Sanity: confirm the file we just wrote is visible from CWD.
        assert (Path.cwd() / ".env").exists(), "Test setup failed: .env not written to tmp_path"

        # Sanity: confirm the env vars are actually unset.
        assert "JUNIPER_DATA_URL" not in os.environ
        assert "JUNIPER_CASCOR_HOST" not in os.environ

        settings = Settings()

        assert settings.juniper_data_url is None, "Settings.juniper_data_url leaked from the .env in CWD — the autouse " "fixture in conftest.py is no longer preventing pydantic-settings from " "reading the developer's local .env."
        assert settings.host == "127.0.0.1", f"Settings.host leaked from the .env in CWD (got {settings.host!r}, " "expected the class default '127.0.0.1'). The .env file is no longer " "isolated from the test session."
