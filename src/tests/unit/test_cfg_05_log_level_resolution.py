#!/usr/bin/env python
"""
Unit tests for CFG-05: _resolve_log_level_env() in
src/cascor_constants/constants.py.

CFG-05 converges two historically-divergent log-level env vars onto a
single name that matches the ecosystem convention:

    JUNIPER_CASCOR_LOG_LEVEL  (preferred; matches pydantic
                               ``Settings`` env_prefix='JUNIPER_CASCOR_')
    CASCOR_LOG_LEVEL          (legacy; accepted with DeprecationWarning,
                               slated for removal in a future release)

These tests pin the precedence + warning behavior of the helper so the
deprecation period does not regress.

The helper is imported directly (without re-executing constants.py's
module-level validation block) so each case can drive the env vars
deterministically.
"""

import os
import sys
import warnings

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cascor_constants.constants import _resolve_log_level_env  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_log_level_env(monkeypatch):
    """Start every test with both log-level env vars unset.

    The host environment may have either variable exported (e.g. a
    developer running tests inside an activated Juniper shell).
    Clearing both up front makes each case deterministic.
    """
    monkeypatch.delenv("JUNIPER_CASCOR_LOG_LEVEL", raising=False)
    monkeypatch.delenv("CASCOR_LOG_LEVEL", raising=False)


class TestResolveLogLevelEnv:
    """Pin CFG-05 precedence + deprecation-warning contract."""

    def test_prefixed_only_returns_value_no_warning(self, monkeypatch, capsys):
        """JUNIPER_CASCOR_LOG_LEVEL set, legacy unset -> returns value, no warning."""
        monkeypatch.setenv("JUNIPER_CASCOR_LOG_LEVEL", "DEBUG")
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # Would raise if a DeprecationWarning fired.
            result = _resolve_log_level_env()
        assert result == "DEBUG"
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_prefixed_lowercased_value_is_uppercased(self, monkeypatch):
        """Lower-cased prefixed value is normalized to uppercase (existing contract)."""
        monkeypatch.setenv("JUNIPER_CASCOR_LOG_LEVEL", "debug")
        result = _resolve_log_level_env()
        assert result == "DEBUG"

    def test_legacy_only_returns_value_and_warns(self, monkeypatch, capsys):
        """CASCOR_LOG_LEVEL set, prefixed unset -> returns value + DeprecationWarning."""
        monkeypatch.setenv("CASCOR_LOG_LEVEL", "WARNING")
        with pytest.warns(DeprecationWarning, match="CASCOR_LOG_LEVEL is deprecated"):
            result = _resolve_log_level_env()
        assert result == "WARNING"
        # No stderr line should be emitted in the legacy-only path; the
        # deprecation channel is the warning, not stderr.
        captured = capsys.readouterr()
        assert "CFG-05 WARNING" not in captured.err

    def test_both_set_same_value_no_warning_no_stderr(self, monkeypatch, capsys):
        """Both env vars set to the same value -> no warning, no stderr."""
        monkeypatch.setenv("JUNIPER_CASCOR_LOG_LEVEL", "INFO")
        monkeypatch.setenv("CASCOR_LOG_LEVEL", "INFO")
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            result = _resolve_log_level_env()
        assert result == "INFO"
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_both_set_same_value_case_insensitive_no_warning(self, monkeypatch, capsys):
        """Both env vars set, differing only in case -> normalized equal, no warning, no stderr."""
        monkeypatch.setenv("JUNIPER_CASCOR_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("CASCOR_LOG_LEVEL", "debug")
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            result = _resolve_log_level_env()
        assert result == "DEBUG"
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_both_set_different_values_prefixed_wins_stderr_emitted(self, monkeypatch, capsys):
        """Both env vars set, different values -> prefixed wins + stderr line."""
        monkeypatch.setenv("JUNIPER_CASCOR_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("CASCOR_LOG_LEVEL", "WARNING")
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # Legacy-only path is the only one that warns; with prefixed
            # set, no DeprecationWarning should fire.
            result = _resolve_log_level_env()
        assert result == "DEBUG"
        captured = capsys.readouterr()
        assert "CFG-05 WARNING" in captured.err
        assert "JUNIPER_CASCOR_LOG_LEVEL takes precedence" in captured.err
        assert "CASCOR_LOG_LEVEL" in captured.err

    def test_neither_set_returns_empty_string(self, capsys):
        """Neither env var set -> returns "" (constants.py then falls back to INFO)."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            result = _resolve_log_level_env()
        assert result == ""
        captured = capsys.readouterr()
        assert captured.err == ""
