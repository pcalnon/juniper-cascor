"""Pin the public surface of juniper-cascor-protocol.

Consumers import from ``juniper_cascor_protocol`` (worker symbols),
``juniper_cascor_protocol.envelope`` (Pydantic schemas), or
``juniper_cascor_protocol.worker`` (StrEnum + BinaryFrame). If a symbol
is renamed or removed this test fails first — protecting the cross-
service contract.
"""

import re
import sys
import tomllib
from pathlib import Path

import juniper_cascor_protocol
import juniper_cascor_protocol.envelope as env_pkg
import juniper_cascor_protocol.worker as worker_pkg


TOP_LEVEL_EXPECTED = {"__version__", "WorkerMessageType", "BinaryFrame"}

ENVELOPE_EXPECTED = {
    "BaseEnvelope",
    "UnknownEnvelope",
    "MetricsEnvelope",
    "StateEnvelope",
    "TopologyEnvelope",
    "EventEnvelope",
    "CascadeAddEnvelope",
    "CandidateProgressEnvelope",
    "InitialMetricsEnvelope",
    "InitialMetricsData",
    "ChunkedMessageEnvelope",
    "ChunkedMessageData",
    "CommandResponseEnvelope",
    "CommandResponseData",
    "ConnectionEstablishedEnvelope",
    "ConnectionEstablishedData",
    "validate_envelope",
    "KnownEnvelope",
    "KNOWN_ENVELOPES",
    "UNMATCHED_TYPE_LABEL",
    "UNKNOWN_TYPE_BUDGET",
    "reset_unknown_label_state",
}

WORKER_EXPECTED = {"WorkerMessageType", "BinaryFrame"}


def test_top_level_all_matches_expected():
    assert set(juniper_cascor_protocol.__all__) == TOP_LEVEL_EXPECTED
    for sym in TOP_LEVEL_EXPECTED:
        assert hasattr(juniper_cascor_protocol, sym), f"missing top-level symbol: {sym}"


def test_envelope_all_matches_expected():
    assert set(env_pkg.__all__) == ENVELOPE_EXPECTED
    for sym in ENVELOPE_EXPECTED:
        assert hasattr(env_pkg, sym), f"missing envelope symbol: {sym}"


def test_worker_all_matches_expected():
    assert set(worker_pkg.__all__) == WORKER_EXPECTED
    for sym in WORKER_EXPECTED:
        assert hasattr(worker_pkg, sym), f"missing worker symbol: {sym}"


def test_version_is_stable_string():
    """``__version__`` is a stable semver string and matches the packaging source.

    Derived from the package's own version sources rather than pinned to a
    literal (the cascor#436 derive-from-source precedent), so a routine release
    bump cannot re-red this test. It still fails on the two invariants it
    actually exists to protect:

    1. **Form** — the published version stays *stable* semver. This package
       deliberately left the alpha shape behind at R2.2.3 (``0.1.0a0`` remains
       on PyPI only for build reproducibility), so an ``a``/``b``/``rc``/
       ``.dev``/``+local`` suffix reaching a release is a regression.
    2. **Lockstep** — ``juniper_cascor_protocol/_version.py`` (which supplies
       ``__version__``) and ``pyproject.toml``'s ``[project].version`` are two
       separately edited files that the release train bumps together. Comparing
       them here is a real cross-file check, not a tautology: bump one without
       the other and this test goes red.
    """
    version = juniper_cascor_protocol.__version__

    assert re.fullmatch(r"\d+\.\d+\.\d+", version), f"__version__ is not a stable semver string (no pre-release/dev/local suffix allowed): {version!r}"

    # tests/ is not packaged, so these only ever run from a source checkout —
    # requiring the pyproject keeps the lockstep half from silently skipping.
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert pyproject_path.is_file(), f"expected the package pyproject.toml at {pyproject_path}"
    declared = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]["version"]

    assert version == declared, f"version lockstep drift: __version__ is {version!r} (juniper_cascor_protocol/_version.py) but pyproject [project].version is {declared!r} — bump both"


def test_worker_subpackage_does_not_import_pydantic():
    """METRICS-MON R2.2 design Q3: worker stays Pydantic-free at runtime.

    Re-imports the worker subpackage in a fresh subinterpreter view by
    inspecting ``sys.modules`` from a clean baseline. Any future refactor
    that accidentally pulls Pydantic into ``juniper_cascor_protocol.worker``
    will surface here before the worker repo CI catches it.
    """
    # Remove any pydantic shims that other tests in this run may have
    # brought in via the envelope import. Then reimport ``worker``.
    # ``importlib.reload`` is deliberate so the assertion targets the
    # post-reload module's own import edges, not the cached version.
    import importlib

    import juniper_cascor_protocol.worker as worker_mod
    import juniper_cascor_protocol.worker.binary_frame as bf_mod
    import juniper_cascor_protocol.worker.messages as msg_mod

    importlib.reload(msg_mod)
    importlib.reload(bf_mod)
    importlib.reload(worker_mod)

    # Walk the module objects and assert none of them reference pydantic.
    for mod in (worker_mod, bf_mod, msg_mod):
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            attr_module = getattr(attr, "__module__", None)
            if attr_module is not None:
                assert not attr_module.startswith("pydantic"), f"{mod.__name__}.{attr_name} resolves into pydantic ({attr_module}) — worker must stay pydantic-free"


def test_worker_only_import_does_not_pull_envelope():
    """Importing only the worker subpackage must not transitively load envelope."""
    # If a previous test loaded envelope, ``juniper_cascor_protocol.envelope``
    # is already in ``sys.modules`` — check that the worker subpackage's
    # source files do not declare ``from juniper_cascor_protocol.envelope``.
    import juniper_cascor_protocol.worker.binary_frame as bf
    import juniper_cascor_protocol.worker.messages as wm

    for mod in (bf, wm):
        # ``__file__`` points at the .py source.
        with open(mod.__file__, "r", encoding="utf-8") as fp:
            src = fp.read()
        assert "juniper_cascor_protocol.envelope" not in src, f"{mod.__name__} imports from envelope subpackage — breaks the worker pydantic-free guarantee"
        # Also confirm pydantic is not directly imported.
        assert "import pydantic" not in src, f"{mod.__name__} imports pydantic directly"
        assert "from pydantic" not in src, f"{mod.__name__} imports pydantic directly"


def test_envelope_subpackage_does_not_import_numpy():
    """Symmetric guarantee: envelope path stays numpy-free at module-load time."""
    import juniper_cascor_protocol.envelope.base as base_mod
    import juniper_cascor_protocol.envelope.control as ctrl_mod
    import juniper_cascor_protocol.envelope.training as train_mod
    import juniper_cascor_protocol.envelope.validate as val_mod

    for mod in (base_mod, ctrl_mod, train_mod, val_mod):
        with open(mod.__file__, "r", encoding="utf-8") as fp:
            src = fp.read()
        assert "import numpy" not in src, f"{mod.__name__} imports numpy at module load — envelope path should stay numpy-free"
        assert "from numpy" not in src, f"{mod.__name__} imports numpy at module load"


def test_top_level_does_not_load_pydantic():
    """Importing ``juniper_cascor_protocol`` (top level) must not load pydantic.

    The top-level ``__init__`` only re-exports the worker subpackage
    symbols — pydantic enters ``sys.modules`` only when a caller
    explicitly imports ``juniper_cascor_protocol.envelope``.

    Inspects actual ``import``/``from`` statements rather than substring
    presence so docstrings mentioning the envelope subpackage don't
    trip the assertion.
    """
    import ast

    with open(juniper_cascor_protocol.__file__, "r", encoding="utf-8") as fp:
        src = fp.read()
    tree = ast.parse(src)
    forbidden = {"pydantic", "juniper_cascor_protocol.envelope"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root != "pydantic", f"top-level __init__ imports pydantic via 'import {alias.name}'"
                assert alias.name not in forbidden, f"top-level __init__ imports {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            mod_root = mod.split(".")[0]
            assert mod_root != "pydantic", f"top-level __init__ imports from pydantic via 'from {mod} import ...'"
            assert mod != "juniper_cascor_protocol.envelope", f"top-level __init__ imports from {mod}"


# Silence the unused-import linter — these are imported for the
# ``hasattr`` checks above.
_ = sys
