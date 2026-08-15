"""Q-6 (CLI experimentation plan §11 / H-7) — ``JUNIPER_CASCOR_LOG_DIR`` override.

Modelled on ``test_w6_snapshots_dir_override.py``; the two overrides are the same
shape against the same class of problem (a checkout-shared path that concurrent
cascor processes collide on).

Service tier: both ``_resolve_log_dir`` helpers (``api/observability.py`` and
``api/service_launcher.py``) must honour the env override at call time, fall back to
the constants default when unset, and treat a set-but-blank value as unset (the
blank-env guard class).

The arm that distinguishes this from W-6: each helper carries an ``ImportError``
fallback that never consults the constants. Reading the env at call time means the
override holds there too — an import-time-only override would silently write to the
shared checkout path in exactly the degraded environment the fallback exists for.

Direct-CLI tier: ``constants.py`` resolves ``_PROJECT_LOG_DIR_DEFAULT`` from the same
env var at import time — pinned via a module re-exec so the test does not depend on
this process's import order.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api import observability as _observability
from api import service_launcher as _service_launcher

pytestmark = pytest.mark.unit

_SRC_DIR = Path(__file__).resolve().parent.parent.parent.parent
_REPO_DIR = _SRC_DIR.parent
_CONSTANTS = _SRC_DIR / "cascor_constants" / "constants.py"

_ENV = "JUNIPER_CASCOR_LOG_DIR"

# Both service-tier helpers are the same contract; drive them from one parametrisation
# so a future divergence between the two copies fails rather than hides.
_RESOLVERS = [
    pytest.param(_observability._resolve_log_dir, "observability", id="observability"),
    pytest.param(_service_launcher._resolve_log_dir, "service_launcher", id="service_launcher"),
]


def _reexec_constants(env_value: "str | None", monkeypatch) -> Path:
    """Re-execute constants.py under a controlled env and return its log dir.

    A fresh module object (not ``importlib.reload``) so the pin is hermetic to this
    test regardless of what the suite already imported.
    """
    if env_value is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, env_value)
    spec = importlib.util.spec_from_file_location("_q6_constants_probe", _CONSTANTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return Path(module._PROJECT_LOG_DIR_DEFAULT)


class TestServiceTierOverride:
    @pytest.mark.parametrize("resolver,_name", _RESOLVERS)
    def test_env_override_wins(self, resolver, _name, tmp_path, monkeypatch):
        target = tmp_path / "run" / "logs"
        monkeypatch.setenv(_ENV, str(target))
        assert resolver() == target

    @pytest.mark.parametrize("resolver,_name", _RESOLVERS)
    def test_unset_falls_back_to_repo_logs(self, resolver, _name, monkeypatch):
        monkeypatch.delenv(_ENV, raising=False)
        assert resolver() == _REPO_DIR / "logs"

    @pytest.mark.parametrize("resolver,_name", _RESOLVERS)
    def test_blank_value_is_treated_as_unset(self, resolver, _name, monkeypatch):
        monkeypatch.setenv(_ENV, "   ")
        assert resolver() == _REPO_DIR / "logs"

    @pytest.mark.parametrize("resolver,_name", _RESOLVERS)
    def test_call_time_read_not_cached(self, resolver, _name, tmp_path, monkeypatch):
        """Two calls under different env values resolve differently — the override is
        read per call, never captured at import or first-call time."""
        first = tmp_path / "a"
        second = tmp_path / "b"
        monkeypatch.setenv(_ENV, str(first))
        assert resolver() == first
        monkeypatch.setenv(_ENV, str(second))
        assert resolver() == second

    @pytest.mark.parametrize("resolver,_name", _RESOLVERS)
    def test_user_home_is_expanded(self, resolver, _name, monkeypatch):
        monkeypatch.setenv(_ENV, "~/q6-logs")
        assert resolver() == Path.home() / "q6-logs"


class TestServiceTierImportErrorFallback:
    """The arm W-6 has no equivalent of.

    Each helper falls back to a checkout-relative ``logs/`` when ``cascor_constants``
    cannot be imported. That path never reads the constants, so an import-time-only
    override would be silently dropped there — writing to the shared checkout log in
    precisely the degraded case the fallback exists to cover.
    """

    @pytest.mark.parametrize("resolver,_name", _RESOLVERS)
    def test_override_still_wins_when_constants_unimportable(self, resolver, _name, tmp_path, monkeypatch):
        target = tmp_path / "fallback" / "logs"
        monkeypatch.setenv(_ENV, str(target))
        # A ``None`` entry in sys.modules makes the ``import`` statement raise ImportError.
        with patch.dict(sys.modules, {"cascor_constants": None, "cascor_constants.constants": None}):
            assert resolver() == target

    @pytest.mark.parametrize("resolver,_name", _RESOLVERS)
    def test_unset_keeps_checkout_relative_fallback(self, resolver, _name, monkeypatch):
        monkeypatch.delenv(_ENV, raising=False)
        with patch.dict(sys.modules, {"cascor_constants": None, "cascor_constants.constants": None}):
            assert resolver() == _REPO_DIR / "logs"


class TestDirectCliTierOverride:
    def test_env_override_wins(self, tmp_path, monkeypatch):
        target = tmp_path / "cli-logs"
        assert _reexec_constants(str(target), monkeypatch) == target

    def test_unset_keeps_repo_logs(self, monkeypatch):
        assert _reexec_constants(None, monkeypatch) == _REPO_DIR / "logs"

    def test_blank_value_is_treated_as_unset(self, monkeypatch):
        assert _reexec_constants("", monkeypatch) == _REPO_DIR / "logs"

    def test_whitespace_only_value_is_treated_as_unset(self, monkeypatch):
        assert _reexec_constants("  \t ", monkeypatch) == _REPO_DIR / "logs"

    def test_user_home_is_expanded(self, monkeypatch):
        assert _reexec_constants("~/q6-cli-logs", monkeypatch) == Path.home() / "q6-cli-logs"

    def test_downstream_log_file_path_follows_the_override(self, tmp_path, monkeypatch):
        """The override must reach the constants the logger actually consumes.

        ``_PROJECT_LOG_FILE_PATH`` is what ``log_config``/``logger`` default their
        handler paths to, so pinning only ``_PROJECT_LOG_DIR_DEFAULT`` would let the
        two drift apart and leave the file log on the shared checkout path.
        """
        target = tmp_path / "downstream"
        monkeypatch.setenv(_ENV, str(target))
        spec = importlib.util.spec_from_file_location("_q6_constants_downstream_probe", _CONSTANTS)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert Path(module._PROJECT_LOG_FILE_PATH) == target
        assert Path(module._LOGGER_LOG_FILE_PATH) == target
        assert Path(module._LOG_CONFIG_LOG_FILE_PATH) == target
