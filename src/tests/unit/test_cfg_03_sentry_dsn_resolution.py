#!/usr/bin/env python
"""
Unit tests for CFG-03: _resolve_sentry_dsn() in src/main.py.

CFG-03 converges two historically-divergent Sentry DSN env vars onto a
single name that matches the ecosystem convention:

    JUNIPER_CASCOR_SENTRY_DSN   (preferred; matches pydantic
                                 ``Settings`` env_prefix='JUNIPER_CASCOR_')
    SENTRY_SDK_DSN              (legacy; accepted with DeprecationWarning,
                                 slated for removal in a future release)

These tests pin the precedence + warning behavior of the helper so the
deprecation period does not regress.
"""

import os
import sys
import warnings

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main import _resolve_sentry_dsn  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_sentry_env(monkeypatch):
    """Start every test with both Sentry DSN env vars unset.

    The host environment may have either variable exported (e.g. a
    developer running tests inside an activated Juniper shell). Clearing
    both up front makes each case deterministic.
    """
    monkeypatch.delenv("JUNIPER_CASCOR_SENTRY_DSN", raising=False)
    monkeypatch.delenv("SENTRY_SDK_DSN", raising=False)


class TestResolveSentryDsn:
    """Pin CFG-03 precedence + deprecation-warning contract."""

    def test_prefixed_only_returns_value_no_warning(self, monkeypatch, capsys):
        """JUNIPER_CASCOR_SENTRY_DSN set, legacy unset -> returns value, no warning."""
        monkeypatch.setenv("JUNIPER_CASCOR_SENTRY_DSN", "https://prefixed@sentry.example/1")
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # Would raise if a DeprecationWarning fired.
            result = _resolve_sentry_dsn()
        assert result == "https://prefixed@sentry.example/1"
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_legacy_only_returns_value_and_warns(self, monkeypatch, capsys):
        """SENTRY_SDK_DSN set, prefixed unset -> returns value + DeprecationWarning."""
        monkeypatch.setenv("SENTRY_SDK_DSN", "https://legacy@sentry.example/2")
        with pytest.warns(DeprecationWarning, match="SENTRY_SDK_DSN is deprecated"):
            result = _resolve_sentry_dsn()
        assert result == "https://legacy@sentry.example/2"
        # No stderr line should be emitted in the legacy-only path; the
        # deprecation channel is the warning, not stderr.
        captured = capsys.readouterr()
        assert "CFG-03 WARNING" not in captured.err

    def test_both_set_same_value_no_warning_no_stderr(self, monkeypatch, capsys):
        """Both env vars set to the same value -> no warning, no stderr."""
        dsn = "https://same@sentry.example/3"
        monkeypatch.setenv("JUNIPER_CASCOR_SENTRY_DSN", dsn)
        monkeypatch.setenv("SENTRY_SDK_DSN", dsn)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            result = _resolve_sentry_dsn()
        assert result == dsn
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_both_set_different_values_prefixed_wins_stderr_emitted(self, monkeypatch, capsys):
        """Both env vars set, different values -> prefixed wins + stderr line."""
        monkeypatch.setenv("JUNIPER_CASCOR_SENTRY_DSN", "https://prefixed@sentry.example/4")
        monkeypatch.setenv("SENTRY_SDK_DSN", "https://legacy@sentry.example/5")
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # Legacy-only path is the only one that warns; with prefixed
            # set, no DeprecationWarning should fire.
            result = _resolve_sentry_dsn()
        assert result == "https://prefixed@sentry.example/4"
        captured = capsys.readouterr()
        assert "CFG-03 WARNING" in captured.err
        assert "JUNIPER_CASCOR_SENTRY_DSN takes precedence" in captured.err
        assert "SENTRY_SDK_DSN" in captured.err

    def test_neither_set_returns_none(self, capsys):
        """Neither env var set -> returns None (Sentry stays disabled)."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            result = _resolve_sentry_dsn()
        assert result is None
        captured = capsys.readouterr()
        assert captured.err == ""
