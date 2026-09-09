#!/usr/bin/env python
"""Unit tests for the micro-tier timing-reference helpers (juniper-ml perf lane P2 item 2.4 / PF-4).

``tests/performance/timing_reference.py`` is what turns a ``baseline_YYYYMMDD.json`` entry from a
parameter record into a timing record, and what stamps host identity and load condition into
pytest-benchmark's saved JSON. The performance tier itself is never collected in CI (it needs
``--run-performance``), so these tests pin the helpers from the unit tier instead. Three
contracts, each of which failed silently for months:

* the statistics come out in MILLISECONDS and are absent (``{}``), not wrong, when the fixture
  ran without timing -- the docstring of ``save_baseline`` promised ``mean_ms`` for a year while
  every entry held only ``hidden_units``;
* ``git_sha`` and the load average never raise -- a reference cut on a host without ``git`` or
  ``os.getloadavg`` must still be a reference;
* ``machine_info`` (compared on ``--benchmark-compare``, warns on change) carries identity only,
  and the load condition lives beside it -- a load average inside ``machine_info`` would make
  every comparison warn and the warning would stop meaning anything.

The module is loaded by path: ``src/tests`` is not a package, so the performance tier cannot be
imported by name from here.
"""

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "performance" / "timing_reference.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("perf_timing_reference_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tr = _load_module()


def _fixture_with_stats(rounds=5, iterations=3, **seconds):
    """A stand-in for a completed pytest-benchmark fixture: ``.stats`` is a Metadata-like
    object whose ``.stats`` holds the ``Stats`` fields in SECONDS."""
    stats = SimpleNamespace(rounds=rounds, **seconds)
    return SimpleNamespace(stats=SimpleNamespace(stats=stats, iterations=iterations))


@pytest.mark.unit
class TestBenchmarkStatsMs:
    def test_seconds_are_converted_to_milliseconds(self):
        fixture = _fixture_with_stats(mean=0.002, median=0.0015, stddev=0.0001, min=0.001, max=0.004)

        out = tr.benchmark_stats_ms(fixture)

        assert out == {
            "mean_ms": pytest.approx(2.0),
            "median_ms": pytest.approx(1.5),
            "stddev_ms": pytest.approx(0.1),
            "min_ms": pytest.approx(1.0),
            "max_ms": pytest.approx(4.0),
            "rounds": 5,
            "iterations": 3,
        }
        assert isinstance(out["rounds"], int)
        assert isinstance(out["iterations"], int)

    def test_the_promised_keys_are_exactly_the_documented_ones(self):
        out = tr.benchmark_stats_ms(_fixture_with_stats(mean=1, median=1, stddev=0, min=1, max=1))
        assert set(out) == {"mean_ms", "median_ms", "stddev_ms", "min_ms", "max_ms", "rounds", "iterations"}

    def test_a_fixture_that_never_ran_yields_nothing_not_garbage(self):
        # ``--benchmark-disable`` and an un-invoked fixture both leave ``stats`` as None.
        assert tr.benchmark_stats_ms(SimpleNamespace(stats=None)) == {}
        assert tr.benchmark_stats_ms(SimpleNamespace()) == {}

    def test_zero_rounds_yields_nothing(self):
        # ``Stats.mean`` raises on empty data; the guard must fire BEFORE any field is read.
        empty = SimpleNamespace(stats=SimpleNamespace(stats=SimpleNamespace(rounds=0), iterations=1))
        assert tr.benchmark_stats_ms(empty) == {}

    def test_a_missing_iteration_count_defaults_to_one(self):
        fixture = SimpleNamespace(stats=SimpleNamespace(stats=SimpleNamespace(rounds=2, mean=1, median=1, stddev=0, min=1, max=1)))
        assert tr.benchmark_stats_ms(fixture)["iterations"] == 1


@pytest.mark.unit
class TestHostIdentityAndCondition:
    def test_collect_environment_carries_identity_and_condition(self):
        env = tr.collect_environment()

        for key in ("python_version", "torch_version", "numpy_version", "platform", "cpu_count", "cpu_model", "torch_num_threads", "omp_num_threads", "mkl_num_threads", "openblas_num_threads", "loadavg_1m", "git_sha"):
            assert key in env, key
        assert isinstance(env["cpu_model"], str) and env["cpu_model"]
        assert env["loadavg_1m"] is None or isinstance(env["loadavg_1m"], float)
        assert env["git_sha"] == tr.UNKNOWN or (len(env["git_sha"]) == 40 and all(c in "0123456789abcdef" for c in env["git_sha"]))

    def test_git_sha_outside_a_repository_is_unknown_not_an_exception(self, tmp_path):
        assert tr.git_sha(cwd=tmp_path) == tr.UNKNOWN

    def test_git_sha_survives_a_missing_git_binary(self, monkeypatch):
        def _no_git(*args, **kwargs):
            raise FileNotFoundError("git")

        monkeypatch.setattr(tr.subprocess, "run", _no_git)
        assert tr.git_sha() == tr.UNKNOWN

    def test_git_sha_survives_a_hung_git(self, monkeypatch):
        def _hang(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="git", timeout=5)

        monkeypatch.setattr(tr.subprocess, "run", _hang)
        assert tr.git_sha() == tr.UNKNOWN

    def test_load_average_is_none_where_the_platform_has_none(self, monkeypatch):
        monkeypatch.delattr(tr.os, "getloadavg", raising=False)
        assert tr.load_average() is None
        assert tr.collect_environment()["loadavg_1m"] is None

    def test_cpu_model_falls_back_when_cpuinfo_is_unreadable(self, monkeypatch):
        def _unreadable(*args, **kwargs):
            raise OSError("no /proc")

        monkeypatch.setattr("builtins.open", _unreadable)
        monkeypatch.setattr(tr.platform, "processor", lambda: "")
        assert tr.cpu_model() == tr.UNKNOWN


@pytest.mark.unit
class TestSavedJsonSplit:
    """Identity is compared; condition is recorded. Mixing them breaks the compare warning."""

    def test_machine_fields_are_identity_only(self):
        fields = tr.stable_machine_fields()
        assert set(fields) == {"cpu_model", "cpu_count", "torch_version", "numpy_version", "thread_pin"}
        assert fields["thread_pin"] == tr.BENCHMARK_THREAD_PIN == 1
        assert not any("load" in key or "utc" in key or "time" in key for key in fields)

    def test_run_condition_fields_carry_load_and_time_and_nothing_compared(self):
        fields = tr.run_condition_fields()
        assert set(fields) == {"loadavg", "saved_utc", "git_sha"}
        assert fields["loadavg"] is None or set(fields["loadavg"]) == {"1m", "5m", "15m"}
        assert fields["saved_utc"].endswith("+00:00")
