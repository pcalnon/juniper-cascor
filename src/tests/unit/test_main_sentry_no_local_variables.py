#!/usr/bin/env python
"""
Unit test: ``src/main.py``'s bootstrap ``sentry_sdk.init`` never captures frame locals.

The SDK defaults ``include_local_variables`` to ``True``, which snapshots every frame's
locals into each error event. juniper-canopy#683's validation (2026-09-24) found an event
carrying the API-key comparison loop's ``candidate`` -- the real configured key -- because
the SDK's scrubber redacts by NAME, and ``candidate`` is not a name it knows.

cascor configures Sentry in two places. The service path (``src/api/app.py``) goes through
juniper-observability's ``configure_sentry``, which carries the setting from its next
release. The direct CLI calls ``sentry_sdk.init`` itself, at import time of ``src/main.py``,
so it needs the setting of its own.

A source-level guard, like ``test_cfg_02_sentry_sdk_optional.py``: the init runs at module
import and only when a DSN is set, so a behavioural test would have to re-import ``main``
with a DSN in the environment, re-running ``load_dotenv`` and the module's other
import-time side effects. The check reads the call's KEYWORDS from the AST rather than
searching for a substring, so a comment or a string that mentions the option cannot
satisfy it.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_MAIN_PY = Path(__file__).resolve().parents[2] / "main.py"


def _sentry_init_calls(tree: ast.AST) -> list[ast.Call]:
    """Every ``sentry_sdk.init(...)`` call in ``tree``."""
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "init" and isinstance(node.func.value, ast.Name) and node.func.value.id == "sentry_sdk"]


def _turns_local_variables_off(call: ast.Call) -> bool:
    """True when the call passes ``include_local_variables=False`` literally."""
    value = {keyword.arg: keyword.value for keyword in call.keywords}.get("include_local_variables")
    return isinstance(value, ast.Constant) and value.value is False


def test_every_bootstrap_sentry_init_turns_local_variables_off() -> None:
    calls = _sentry_init_calls(ast.parse(_MAIN_PY.read_text(encoding="utf-8")))
    assert calls, "no sentry_sdk.init(...) call found in src/main.py, so this guard would check nothing"
    for call in calls:
        assert _turns_local_variables_off(call), f"src/main.py:{call.lineno}: sentry_sdk.init must pass include_local_variables=False (the SDK default, True, ships every frame's locals with each error event)"


@pytest.mark.parametrize(
    "source",
    [
        "sentry_sdk.init(dsn='x')",
        "sentry_sdk.init(dsn='x', include_local_variables=True)",
        "sentry_sdk.init(dsn='x', include_local_variables=flag)",
    ],
)
def test_the_guard_rejects_an_init_that_leaves_locals_on(source: str) -> None:
    """Negative control: the matcher finds the call, and the check fails on each wrong form."""
    (call,) = _sentry_init_calls(ast.parse(source))
    assert not _turns_local_variables_off(call)
