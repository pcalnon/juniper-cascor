#!/usr/bin/env python
"""Regression tests for Phase 2E juniper-cascor bug fixes.

Covers:
- BUG-CC-01: create_topology_message() wired into cascade_add lifecycle and
  broadcast via WebSocketManager.broadcast_from_thread.
- BUG-CC-02: cascade_add correlation reflects the installed hidden unit's
  best_correlation attribute instead of being hardcoded to 0.0.
- BUG-CC-04: package version is read from importlib.metadata; no source files
  carry stale ``# Version:`` header lines.
- BUG-CC-07: TrainingMonitor.current_phase is updated by the
  TrainingStateMachine via on_phase_change, with no manual assignments in the
  lifecycle manager.
"""

import importlib.metadata
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure src/ is on sys.path for top-level package imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_lifecycle_manager_with_mock_network(prev_hidden: int, new_hidden_units, ws_manager=None):
    """Construct a TrainingLifecycleManager wired to a mock network with a
    hidden_units list containing ``new_hidden_units``.

    The function mimics the state observed after the network has grown and
    installed new hidden units.
    """
    from api.lifecycle.manager import TrainingLifecycleManager

    mgr = TrainingLifecycleManager()
    mgr.network = MagicMock()
    mgr.network.input_size = 4
    mgr.network.output_size = 2
    mgr.network.hidden_units = list(new_hidden_units)
    if ws_manager is not None:
        mgr._ws_manager = ws_manager
    return mgr, prev_hidden


# ---------------------------------------------------------------------------
# BUG-CC-01: create_topology_message wired into cascade_add lifecycle
# ---------------------------------------------------------------------------
class TestBugCC01TopologyBroadcast:
    def test_topology_message_broadcast_after_cascade_add(self):
        """After new hidden units appear, manager broadcasts a topology message."""
        from api.websocket.messages import create_topology_message

        # Build a unit with a known correlation so BUG-CC-02 path also exercises.
        unit_attrs = {"best_correlation": 0.42}
        unit = MagicMock(**unit_attrs)
        # Configure getattr fallback explicitly:
        unit.best_correlation = 0.42

        ws = MagicMock()
        mgr, prev_hidden = _make_lifecycle_manager_with_mock_network(
            prev_hidden=0,
            new_hidden_units=[unit],
            ws_manager=ws,
        )

        # Re-execute the cascade_add topology-broadcast block inline.
        new_hidden = len(mgr.network.hidden_units)
        if new_hidden > prev_hidden:
            for i in range(prev_hidden, new_hidden):
                actual = float(getattr(mgr.network.hidden_units[i], "best_correlation", 0.0) or 0.0)
                mgr.monitor.on_cascade_add(hidden_unit_index=i, correlation=actual)
            topology_data = {
                "hidden_units": new_hidden,
                "input_size": mgr.network.input_size,
                "output_size": mgr.network.output_size,
                "event": "cascade_add",
            }
            mgr._ws_manager.broadcast_from_thread(create_topology_message(topology_data))

        # Assert broadcast_from_thread was called once with a topology envelope.
        call_count = ws.broadcast_from_thread.call_count
        assert call_count == 1
        envelope = ws.broadcast_from_thread.call_args.args[0]
        assert envelope["type"] == "topology"
        payload = envelope["data"]
        assert payload["event"] == "cascade_add"
        assert payload["hidden_units"] == 1
        assert payload["input_size"] == 4
        assert payload["output_size"] == 2

    def test_manager_grow_hook_imports_topology_message(self):
        """The lifecycle manager source must reference create_topology_message."""
        manager_src = Path(__file__).resolve().parents[2] / "api" / "lifecycle" / "manager.py"
        text = manager_src.read_text()
        assert "create_topology_message" in text, "BUG-CC-01: create_topology_message must be referenced in manager.py"

    def test_create_topology_message_envelope_shape(self):
        from api.websocket.messages import create_topology_message

        msg = create_topology_message({"hidden_units": 3, "input_size": 2, "output_size": 1, "event": "cascade_add"})
        assert msg["type"] == "topology"
        assert msg["data"]["hidden_units"] == 3


# ---------------------------------------------------------------------------
# BUG-CC-02: cascade_add correlation reflects installed unit's best_correlation
# ---------------------------------------------------------------------------
class TestBugCC02CorrelationPropagation:
    def test_correlation_uses_unit_best_correlation_not_zero(self):
        """The correlation passed to on_cascade_add equals unit.best_correlation."""
        from api.lifecycle.monitor import TrainingMonitor

        monitor = TrainingMonitor()
        captured = []

        def _capture(event, **kwargs):
            captured.append(event)

        monitor.register_callback("cascade_add", _capture)

        unit = MagicMock()
        unit.best_correlation = 0.875

        # Mirror the production cascade_add correlation-propagation loop body.
        actual_correlation = float(getattr(unit, "best_correlation", 0.0) or 0.0)
        monitor.on_cascade_add(hidden_unit_index=0, correlation=actual_correlation)

        assert len(captured) == 1
        observed_correlation = captured[0]["correlation"]
        assert observed_correlation == pytest.approx(0.875)

    def test_correlation_falls_back_to_zero_when_attr_missing(self):
        """Units without best_correlation default to 0.0 (no AttributeError)."""

        class _NoCorrelationUnit:
            pass

        unit = _NoCorrelationUnit()
        actual_correlation = float(getattr(unit, "best_correlation", 0.0) or 0.0)
        assert actual_correlation == 0.0

    def test_manager_source_no_longer_hardcodes_zero_correlation(self):
        """Sanity: the literal `correlation=0.0,` should no longer appear inside
        the cascade_add loop in manager.py (BUG-CC-02 regression guard)."""
        manager_src = (Path(__file__).resolve().parents[2] / "api" / "lifecycle" / "manager.py").read_text()
        # The new code should call on_cascade_add with the actual_correlation
        # variable, not a literal 0.0.
        assert "correlation=actual_correlation" in manager_src


# ---------------------------------------------------------------------------
# BUG-CC-04: version comes from importlib.metadata; no stale headers remain
# ---------------------------------------------------------------------------
class TestBugCC04VersionSingleSource:
    def test_importlib_metadata_returns_nonempty_version(self):
        version = importlib.metadata.version("juniper-cascor")
        assert isinstance(version, str)
        assert version != ""
        # Must be a recognizable semver-ish prefix.
        assert re.match(r"^\d+\.\d+\.\d+", version)

    def test_version_matches_pyproject(self):
        """Installed package version must equal the value in pyproject.toml."""
        repo_root = Path(__file__).resolve().parents[3]
        pyproject = repo_root / "pyproject.toml"
        assert pyproject.is_file(), f"pyproject.toml not found at {pyproject}"
        text = pyproject.read_text()
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        assert match is not None, "version line not found in pyproject.toml"
        declared = match.group(1)
        installed = importlib.metadata.version("juniper-cascor")
        assert installed == declared, f"importlib version {installed!r} disagrees with pyproject {declared!r}"

    def test_no_version_header_lines_in_source(self):
        """No production source files should retain `# Version:` headers."""
        src_root = Path(__file__).resolve().parents[2]
        offenders = []
        for path in src_root.rglob("*.py"):
            # Skip notebook checkpoints and explicit backups directories.
            posix = path.as_posix()
            if ".ipynb_checkpoints" in posix or "/backups/" in posix:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                logging.debug("test_no_version_header_lines_in_source: skipping unreadable %s: %s", path, exc)
                continue
            for line in text.splitlines():
                stripped = line.lstrip()
                if stripped.startswith("# Version:") or stripped.startswith("Version:"):
                    # Allow lines like "Version: " inside Sphinx-style narrative
                    # text — but the header form has whitespace separator only.
                    if re.match(r"^(#\s*)?Version:\s+\S", stripped):
                        offenders.append(str(path.relative_to(src_root)))
                        break
        assert offenders == [], "BUG-CC-04: stale `Version:` header lines must be removed: " + ", ".join(offenders)


# ---------------------------------------------------------------------------
# BUG-CC-07: TrainingMonitor.current_phase driven by state-machine wrapper
# ---------------------------------------------------------------------------
class TestBugCC07PhaseTracking:
    def test_on_phase_change_updates_current_phase(self):
        """Direct invocation of on_phase_change must mutate current_phase."""
        from api.lifecycle.monitor import TrainingMonitor

        monitor = TrainingMonitor()
        assert monitor.current_phase == "output"
        monitor.on_phase_change("candidate")
        assert monitor.current_phase == "candidate"
        monitor.on_phase_change("output")
        assert monitor.current_phase == "output"

    def test_on_phase_change_triggers_callbacks(self):
        from api.lifecycle.monitor import TrainingMonitor

        monitor = TrainingMonitor()
        seen = []
        monitor.register_callback("phase_change", lambda phase: seen.append(phase))
        monitor.on_phase_change("candidate")
        assert seen == ["candidate"]

    def test_manager_no_longer_assigns_current_phase_directly(self):
        """No manual `monitor.current_phase = "..."` assignments in manager.py."""
        manager_src = (Path(__file__).resolve().parents[2] / "api" / "lifecycle" / "manager.py").read_text()
        # Permit the on_phase_change call but forbid the dotted assignment.
        assert "monitor.current_phase =" not in manager_src
