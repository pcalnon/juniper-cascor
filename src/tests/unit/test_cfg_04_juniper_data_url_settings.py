#!/usr/bin/env python
"""
Unit tests for CFG-04: ``Settings.juniper_data_url`` field in
``src/api/settings.py``.

CFG-04 consolidates 8 raw ``os.environ.get("JUNIPER_DATA_URL")`` /
``os.getenv("JUNIPER_DATA_URL")`` call sites across:

    - src/api/app.py                       (3 sites)
    - src/api/routes/health.py
    - src/api/lifecycle/manager.py
    - src/main.py
    - src/spiral_problem/spiral_problem.py
    - src/spiral_problem/data_provider.py

onto a single validated pydantic field. ``JUNIPER_DATA_URL`` is the
canonical, cross-service env-var name (also used by juniper-data and
juniper-canopy) so the field is exposed via ``AliasChoices`` rather
than the default ``env_prefix='JUNIPER_CASCOR_'`` lookup (which would
force the awkward ``JUNIPER_CASCOR_JUNIPER_DATA_URL``).

This is **not a deprecation migration** (no rename, no legacy-form to
discourage) — the env-var name stays the same. The field is added so
the lookup is centralized, validated, and discoverable; the prefixed
form is additionally accepted to give operators the ability to apply
a per-service override that doesn't leak into other services on the
same host.

These tests pin the contract:

1. Default ``None`` when neither env var is set.
2. ``JUNIPER_DATA_URL`` (canonical, unprefixed) populates the field.
3. ``JUNIPER_CASCOR_JUNIPER_DATA_URL`` (per-service override)
   populates the field.
4. Both set -> the canonical (unprefixed) name wins (AliasChoices
   precedence is left-to-right).
5. Direct ``Settings(juniper_data_url=...)`` kwarg overrides env.
6. Empty-string env value is preserved as-is (operator's choice,
   downstream falsy check handles it the same as ``None``).
"""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.settings import Settings  # noqa: E402

pytestmark = pytest.mark.unit

_CANONICAL_ENV = "JUNIPER_DATA_URL"
_PREFIXED_ENV = "JUNIPER_CASCOR_JUNIPER_DATA_URL"


@pytest.fixture(autouse=True)
def _clean_juniper_data_url_env(monkeypatch):
    """Start every test with both env vars unset.

    The host environment may have either variable exported (e.g. a
    developer running tests inside an activated Juniper shell).
    Clearing both up front makes each case deterministic.
    """
    monkeypatch.delenv(_CANONICAL_ENV, raising=False)
    monkeypatch.delenv(_PREFIXED_ENV, raising=False)


class TestJuniperDataUrlField:
    """Pin CFG-04 ``Settings.juniper_data_url`` precedence + default."""

    def test_default_none_when_neither_env_set(self):
        """Field defaults to ``None`` so callers can decide fallback policy."""
        settings = Settings()
        assert settings.juniper_data_url is None

    def test_canonical_unprefixed_env_populates_field(self, monkeypatch):
        """``JUNIPER_DATA_URL`` (the ecosystem-shared name) populates the field."""
        monkeypatch.setenv(_CANONICAL_ENV, "http://data.example:8100")
        settings = Settings()
        assert settings.juniper_data_url == "http://data.example:8100"

    def test_prefixed_env_populates_field(self, monkeypatch):
        """``JUNIPER_CASCOR_JUNIPER_DATA_URL`` (per-service override) populates the field."""
        monkeypatch.setenv(_PREFIXED_ENV, "http://override.example:8100")
        settings = Settings()
        assert settings.juniper_data_url == "http://override.example:8100"

    def test_both_set_canonical_wins(self, monkeypatch):
        """When both are set, the canonical (unprefixed) name wins.

        ``AliasChoices`` resolves left-to-right and we list the
        canonical name first, so deployments that pin a global
        ``JUNIPER_DATA_URL`` are not silently overridden by a leftover
        prefixed value.
        """
        monkeypatch.setenv(_CANONICAL_ENV, "http://canonical.example:8100")
        monkeypatch.setenv(_PREFIXED_ENV, "http://override.example:8100")
        settings = Settings()
        assert settings.juniper_data_url == "http://canonical.example:8100"

    def test_explicit_kwarg_overrides_env(self, monkeypatch):
        """Passing ``juniper_data_url=...`` to ``Settings()`` wins over env.

        Test fixtures and integration tests rely on this to inject a
        URL without polluting the process env.
        """
        monkeypatch.setenv(_CANONICAL_ENV, "http://env.example:8100")
        settings = Settings(juniper_data_url="http://kwarg.example:8100")
        assert settings.juniper_data_url == "http://kwarg.example:8100"

    def test_empty_string_env_preserved_as_empty(self, monkeypatch):
        """Empty string is kept as the empty string (not coerced to ``None``).

        Downstream call sites use ``settings.juniper_data_url or DEFAULT``
        or ``if not settings.juniper_data_url:`` — both handle the empty
        string the same as ``None`` so behavior matches the pre-CFG-04
        ``os.environ.get("JUNIPER_DATA_URL")`` semantics (which returns
        ``""`` for an exported-but-empty env var).
        """
        monkeypatch.setenv(_CANONICAL_ENV, "")
        settings = Settings()
        assert settings.juniper_data_url == ""

    def test_field_is_optional_str(self):
        """Type contract: ``str | None``.

        Pinning the annotation guards against accidental tightening
        (e.g., to ``str``) that would break the ``Settings()`` default
        path used by callers that supply the fallback themselves.
        """
        annotations = Settings.__annotations__
        assert "juniper_data_url" in annotations
        # In pydantic v2 the annotation is the raw type expression.
        # Accept either ``str | None`` (PEP 604) or
        # ``Optional[str]`` rendering for forward-compat with future
        # annotation styles.
        annotation_repr = repr(annotations["juniper_data_url"])
        assert "None" in annotation_repr and "str" in annotation_repr
