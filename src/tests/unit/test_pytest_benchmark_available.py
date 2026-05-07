#!/usr/bin/env python
"""
Project:       Juniper
Sub-Project:   JuniperCascor
Application:   juniper_cascor
File Name:     test_pytest_benchmark_available.py
File Path:     src/tests/unit/

Author:        Paul Calnon

Date Created:  2026-05-06
Last Modified: 2026-05-06

License:       MIT License
Copyright:     Copyright (c) 2024-2026 Paul Calnon

Description:
    Regression guard for the pytest-benchmark dependency.

    Performance tests under src/tests/performance/ rely on the `benchmark`
    fixture provided by pytest-benchmark. The fixture is contributed via the
    `pytest11` entry-point group, so a missing wheel in the active env causes
    every benchmark-based test to fail at fixture resolution with the
    notoriously unhelpful "fixture 'benchmark' not found" error.

    These tests fail fast at collection-adjacent unit scope so the missing
    dependency surfaces as a single clear failure long before --run-performance
    is exercised, and provide an actionable message pointing at the test extra
    in pyproject.toml.
"""

from importlib.metadata import entry_points

import pytest


@pytest.mark.unit
class TestPytestBenchmarkAvailable:
    """Guard against pytest-benchmark silently disappearing from test extras."""

    def test_pytest_benchmark_is_importable(self):
        try:
            import pytest_benchmark
        except ImportError as exc:
            pytest.fail("pytest-benchmark is not installed in the active env. " "Performance tests will fail with 'fixture benchmark not found'. " "Add it to the [project.optional-dependencies] test extra in " f"pyproject.toml and reinstall (pip install -e .[test]). Underlying error: {exc}")

        assert pytest_benchmark.__version__, "pytest-benchmark exposes no __version__"

    def test_benchmark_plugin_entry_point_registered(self):
        names = {ep.name for ep in entry_points(group="pytest11")}
        assert "benchmark" in names, "pytest-benchmark plugin entry point 'benchmark' is not registered " "in the active env. Reinstall the test extra: pip install -e .[test]. " f"Discovered pytest11 entry points: {sorted(names)}"

    def test_benchmark_fixture_resolvable(self, request):
        fixture_names = request.fixturenames if hasattr(request, "fixturenames") else []
        registered = request.session._fixturemanager._arg2fixturedefs  # noqa: SLF001
        assert "benchmark" in registered, "Pytest fixture 'benchmark' is not registered. pytest-benchmark " "appears installed but its plugin did not load — check that " "PYTEST_DISABLE_PLUGIN_AUTOLOAD is unset and that no addopts entry " "blocks the plugin (e.g. '-p no:benchmark'). Currently visible " f"fixturenames sample: {sorted(fixture_names)[:10]}"
