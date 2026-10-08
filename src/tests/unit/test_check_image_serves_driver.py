"""
The docker driver of ``util/check_image_serves.py``, and the verdict edges the pure suite does not pin.

``src/tests/unit/test_check_image_serves.py`` drives ``evaluate`` with synthetic observations and
pins the workflow wiring. It never runs ``main`` past the usage check, so a regression in how the
script starts the image, reads the probes, or removes the container would still be green. This
file drives ``main`` with a scripted ``docker`` and no daemon.

The contract under test is the one the publish job relies on: the serve container runs with the
image's own entrypoint, a container that never answers liveness fails closed, and the container
is removed on every path that obtained an id. ``--health-version optional`` tolerates an absent
field and still rejects a present field that disagrees.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SCRIPT = Path(__file__).resolve().parents[3] / "util" / "check_image_serves.py"

IMAGE = "ghcr.io/pcalnon/juniper-cascor@sha256:abc"
CID = "abc123def"
BASE = ["--image", IMAGE, "--dist", "juniper-cascor", "--port", "8200", "--expect-version", "0.12.0"]


def _load():
    spec = importlib.util.spec_from_file_location("check_image_serves_driver", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Completed:
    __slots__ = ("returncode", "stdout", "stderr")

    def __init__(self, code: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = code
        self.stdout = stdout
        self.stderr = stderr


class _Clock:
    """``time.sleep`` advances the same clock ``time.monotonic`` reads, so polls never wait."""

    def __init__(self) -> None:
        self.t = 1_000.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def _version_line(*, metadata="0.12.0", module_version=None, module_error=None, prefix: str = "") -> str:
    payload = json.dumps({"metadata": metadata, "module_version": module_version, "module_error": module_error})
    return f"{prefix}\n{payload}\n" if prefix else payload + "\n"


def _http(status: int, body) -> _Completed:
    text = body if isinstance(body, str) else json.dumps(body)
    return _Completed(0, stdout=f"{status}\n{text}\n")


def _scripted(monkeypatch, mod, responses):
    queue = list(responses)
    calls = []

    def run(argv, **kwargs):
        calls.append((list(argv), kwargs.get("timeout")))
        if not queue:
            raise AssertionError(f"unexpected docker call: {argv}")
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    clock = _Clock()
    monkeypatch.setattr(mod.subprocess, "run", run)
    monkeypatch.setattr(mod.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(mod.time, "sleep", clock.sleep)
    return calls, queue


def _argv(calls):
    return [argv for argv, _timeout in calls]


def _evaluate(**overrides):
    kwargs = {
        "expect": "0.12.0",
        "metadata": "0.12.0",
        "module": None,
        "module_version": None,
        "module_error": None,
        "health_status": 200,
        "health_body": {"status": "ok", "version": "0.12.0"},
        "health_version_required": True,
        "enveloped": {},
    }
    kwargs.update(overrides)
    return _load().evaluate(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Verdict edges the existing evaluate suite does not pin
# ─────────────────────────────────────────────────────────────────────────────────────────────
class TestVerdictEdges:
    def test_an_absent_health_version_passes_when_optional(self):
        """``--health-version optional`` exists so a body of ``{"status": "ok"}`` is not a failure."""
        assert _evaluate(health_version_required=False, health_body={"status": "ok"}) == []

    def test_a_present_health_version_that_disagrees_fails_even_when_optional(self):
        """Optional relaxes absence only. A field that is present and wrong is still the 0.6.0 defect."""
        failures = _evaluate(health_version_required=False, health_body={"status": "ok", "version": "0.6.0"})
        assert failures == ["the liveness body reports version 0.6.0 != metadata 0.12.0"]

    def test_a_non_dict_liveness_body_is_a_missing_version_not_a_crash(self):
        for body in ("ok", None, ["ok"]):
            failures = _evaluate(health_body=body)
            assert failures == ["the liveness body carries no version field"], body

    def test_a_module_that_does_not_import_names_the_error(self):
        """Distinct from an import that succeeds and simply has no ``__version__``."""
        failures = _evaluate(module="juniper_cascor", module_version=None, module_error="ImportError: boom")
        assert failures == ["juniper_cascor does not import: ImportError: boom"]

    def test_a_module_version_that_disagrees_with_metadata_fails(self):
        """The class-2 defect: the distribution says 0.12.0 and the module still says 0.6.0."""
        failures = _evaluate(module="juniper_cascor", module_version="0.6.0")
        assert failures == ["juniper_cascor.__version__ 0.6.0 != metadata 0.12.0"]

    def test_a_missing_distribution_is_still_reported_when_the_module_has_a_version(self):
        """The probe can set both: metadata lookup failed, then the module imported anyway."""
        failures = _evaluate(
            metadata=None,
            module="juniper_cascor",
            module_version="0.12.0",
            module_error="no distribution 'juniper-cascor'",
            health_version_required=False,
            health_body={"status": "ok"},
        )
        assert failures == [
            "no installed distribution metadata (no distribution 'juniper-cascor')",
            "juniper_cascor.__version__ 0.12.0 != metadata None",
        ]

    def test_every_enveloped_path_is_reported_in_path_order(self):
        """A check that returns on the first bad path would publish an image whose other routes lie."""
        failures = _evaluate(
            enveloped={
                "/v1/z": (500, {"meta": {"version": "0.12.0"}}),
                "/v1/a": (404, "missing"),
                "/v1/m": (200, {"status": "success"}),
            }
        )
        assert failures == [
            "/v1/a answered 404, not 200",
            "/v1/m carries no meta.version",
            "/v1/z answered 500, not 200",
        ]

    def test_a_non_json_envelope_body_is_a_missing_version_not_a_crash(self):
        failures = _evaluate(enveloped={"/v1/workers": (200, "not-json")})
        assert failures == ["/v1/workers carries no meta.version"]


# ─────────────────────────────────────────────────────────────────────────────────────────────
# main(): scripted docker, no daemon
# ─────────────────────────────────────────────────────────────────────────────────────────────
class TestDriver:
    def test_a_version_probe_timeout_is_an_environment_error_and_starts_nothing(self, monkeypatch):
        mod = _load()
        calls, queue = _scripted(monkeypatch, mod, [subprocess.TimeoutExpired(cmd="docker", timeout=180)])

        code = mod.main(BASE)

        assert code == 2
        assert _argv(calls) == [["docker", "run", "--rm", "--entrypoint", "python", IMAGE, "-c", mod._VERSION_PROBE, "juniper-cascor", ""]]
        assert calls[0][1] == 180
        assert queue == []

    def test_a_version_probe_that_cannot_run_python_does_not_start_the_service(self, monkeypatch, capsys):
        mod = _load()
        calls, _queue = _scripted(monkeypatch, mod, [_Completed(1, stderr="python: not found")])

        assert mod.main(BASE) == 2
        assert len(calls) == 1
        err = capsys.readouterr().err
        assert "could not run python" in err
        assert "python: not found" in err

    def test_a_version_probe_that_prints_no_json_is_an_environment_error(self, monkeypatch, capsys):
        mod = _load()
        _scripted(monkeypatch, mod, [_Completed(0, stdout="ready\n")])

        assert mod.main(BASE) == 2
        assert "printed no JSON" in capsys.readouterr().err

    def test_json_that_is_not_the_last_line_is_not_a_version(self, monkeypatch, capsys):
        """A warning printed after the probe must not be parsed as success, and must not be ignored into a pass."""
        mod = _load()
        stdout = _version_line() + "WARNING: leftover\n"
        calls, _queue = _scripted(monkeypatch, mod, [_Completed(0, stdout=stdout)])

        assert mod.main(BASE) == 2
        assert len(calls) == 1
        assert "printed no JSON" in capsys.readouterr().err

    def test_a_banner_before_the_json_is_ignored_and_the_service_keeps_its_entrypoint(self, monkeypatch, capsys):
        """``docker run -d IMAGE`` carries no entrypoint, port, or command override. The probe is the only override."""
        mod = _load()
        calls, queue = _scripted(
            monkeypatch,
            mod,
            [
                _Completed(0, stdout=_version_line(prefix="pulling layers")),
                _Completed(0, stdout=f"digest: something\n{CID}\n"),
                _Completed(0, stdout="true\n"),
                _http(200, {"status": "ok", "version": "0.12.0"}),
                _Completed(0, stdout=CID),
            ],
        )

        assert mod.main([*BASE, "--timeout", "10"]) == 0
        argv = _argv(calls)
        assert argv[1] == ["docker", "run", "-d", IMAGE]
        assert calls[1][1] == 120
        assert "--entrypoint" not in argv[1]
        assert argv[2] == ["docker", "inspect", "-f", "{{.State.Running}}", CID]
        assert argv[3] == ["docker", "exec", CID, "python", "-c", mod._GET_PROBE, "8200", "/v1/health"]
        assert calls[3][1] == 20
        assert argv[4] == ["docker", "rm", "-f", CID]
        assert calls[4][1] == 60
        assert queue == []
        assert "serves /v1/health and reports 0.12.0" in capsys.readouterr().out

    def test_an_image_that_does_not_start_exits_1_and_removes_nothing(self, monkeypatch, capsys):
        """No id was captured, so there is nothing to ``rm``. Removing ``""`` would target the wrong container."""
        mod = _load()
        calls, _queue = _scripted(
            monkeypatch,
            mod,
            [
                _Completed(0, stdout=_version_line()),
                _Completed(125, stderr="port is already allocated"),
            ],
        )

        assert mod.main(BASE) == 1
        assert _argv(calls) == [
            ["docker", "run", "--rm", "--entrypoint", "python", IMAGE, "-c", mod._VERSION_PROBE, "juniper-cascor", ""],
            ["docker", "run", "-d", IMAGE],
        ]
        assert "did not start" in capsys.readouterr().err

    def test_a_container_that_exits_before_liveness_is_removed_and_not_probed(self, monkeypatch, capsys):
        mod = _load()
        calls, queue = _scripted(
            monkeypatch,
            mod,
            [
                _Completed(0, stdout=_version_line()),
                _Completed(0, stdout=CID + "\n"),
                _Completed(0, stdout="false\n"),
                _Completed(0, stdout="Traceback: boom"),
                _Completed(0, stdout=CID),
            ],
        )

        assert mod.main([*BASE, "--enveloped-path", "/v1/workers", "--timeout", "30"]) == 1
        argv = _argv(calls)
        assert ["docker", "exec", CID, "python", "-c", mod._GET_PROBE, "8200", "/v1/health"] not in argv
        assert ["docker", "logs", "--tail", "40", CID] in argv
        assert argv[-1] == ["docker", "rm", "-f", CID]
        assert queue == []
        err = capsys.readouterr().err
        assert "exited before liveness" in err
        assert "Traceback: boom" in err

    def test_an_inspect_that_times_out_fails_closed_and_still_removes_the_container(self, monkeypatch):
        """Anything other than the literal ``true`` is not a running service, including a timed-out inspect."""
        mod = _load()
        calls, queue = _scripted(
            monkeypatch,
            mod,
            [
                _Completed(0, stdout=_version_line()),
                _Completed(0, stdout=CID + "\n"),
                subprocess.TimeoutExpired(cmd="docker", timeout=20),
                _Completed(0, stdout=""),
                _Completed(0, stdout=CID),
            ],
        )

        assert mod.main([*BASE, "--timeout", "30"]) == 1
        assert _argv(calls)[-1] == ["docker", "rm", "-f", CID]
        assert queue == []

    def test_liveness_that_never_arrives_fails_and_the_container_is_removed(self, monkeypatch, capsys):
        mod = _load()
        calls, queue = _scripted(
            monkeypatch,
            mod,
            [
                _Completed(0, stdout=_version_line()),
                _Completed(0, stdout=CID + "\n"),
                _Completed(0, stdout="true\n"),
                _http(503, {"status": "starting"}),
                _http(503, {"status": "starting"}),
                _Completed(0, stdout="still starting"),
                _Completed(0, stdout=CID),
            ],
        )

        assert mod.main([*BASE, "--timeout", "1", "--enveloped-path", "/v1/workers"]) == 1
        argv = _argv(calls)
        # One poll, then the deadline. The envelope is still probed; the container is still removed.
        assert argv.count(["docker", "inspect", "-f", "{{.State.Running}}", CID]) == 1
        assert ["docker", "exec", CID, "python", "-c", mod._GET_PROBE, "8200", "/v1/workers"] in argv
        assert ["docker", "logs", "--tail", "40", CID] in argv
        assert argv[-1] == ["docker", "rm", "-f", CID]
        assert queue == []
        err = capsys.readouterr().err
        assert "liveness answered 503, not 200" in err

    def test_a_matching_envelope_exits_0_and_the_container_is_still_removed(self, monkeypatch):
        mod = _load()
        calls, queue = _scripted(
            monkeypatch,
            mod,
            [
                _Completed(0, stdout=_version_line()),
                _Completed(0, stdout=CID + "\n"),
                _Completed(0, stdout="true\n"),
                _http(200, {"status": "ok", "version": "0.12.0"}),
                _http(200, {"meta": {"version": "0.12.0"}}),
                _Completed(0, stdout=CID),
            ],
        )

        assert mod.main([*BASE, "--timeout", "10", "--enveloped-path", "/v1/workers"]) == 0
        assert _argv(calls)[-1] == ["docker", "rm", "-f", CID]
        assert queue == []

    def test_a_wrong_envelope_exits_1_and_the_container_is_still_removed(self, monkeypatch, capsys):
        """The 0.11.0 image: liveness is right, the envelope still says 0.6.0. Removal is not conditional on the verdict."""
        mod = _load()
        calls, queue = _scripted(
            monkeypatch,
            mod,
            [
                _Completed(0, stdout=_version_line(metadata="0.11.0")),
                _Completed(0, stdout=CID + "\n"),
                _Completed(0, stdout="true\n"),
                _http(200, {"status": "ok", "version": "0.11.0"}),
                _http(200, {"meta": {"version": "0.6.0"}}),
                _Completed(0, stdout=CID),
            ],
        )

        assert mod.main(["--image", IMAGE, "--dist", "juniper-cascor", "--port", "8200", "--expect-version", "0.11.0", "--timeout", "10", "--enveloped-path", "/v1/workers"]) == 1
        assert _argv(calls)[-1] == ["docker", "rm", "-f", CID]
        assert queue == []
        assert "/v1/workers meta.version 0.6.0 != metadata 0.11.0" in capsys.readouterr().err

    def test_liveness_that_arrives_on_the_second_poll_passes(self, monkeypatch):
        mod = _load()
        calls, queue = _scripted(
            monkeypatch,
            mod,
            [
                _Completed(0, stdout=_version_line()),
                _Completed(0, stdout=CID + "\n"),
                _Completed(0, stdout="true\n"),
                _http(503, {"status": "starting"}),
                _Completed(0, stdout="true\n"),
                _http(200, {"status": "ok", "version": "0.12.0"}),
                _Completed(0, stdout=CID),
            ],
        )

        assert mod.main([*BASE, "--timeout", "10"]) == 0
        assert _argv(calls).count(["docker", "inspect", "-f", "{{.State.Running}}", CID]) == 2
        assert queue == []

    def test_an_unreadable_probe_does_not_count_as_liveness(self, monkeypatch, capsys):
        """A status line that is not an integer is the same outcome as no answer: fail, and do not publish."""
        mod = _load()
        calls, queue = _scripted(
            monkeypatch,
            mod,
            [
                _Completed(0, stdout=_version_line()),
                _Completed(0, stdout=CID + "\n"),
                _Completed(0, stdout="true\n"),
                _Completed(0, stdout="WARNING: probe\n200\n{}"),
                _Completed(0, stdout="log"),
                _Completed(0, stdout=CID),
            ],
        )

        assert mod.main([*BASE, "--timeout", "1"]) == 1
        assert "liveness answered None, not 200" in capsys.readouterr().err
        assert _argv(calls)[-1] == ["docker", "rm", "-f", CID]
        assert queue == []

    def test_optional_health_version_absent_exits_0(self, monkeypatch):
        mod = _load()
        _calls, queue = _scripted(
            monkeypatch,
            mod,
            [
                _Completed(0, stdout=_version_line()),
                _Completed(0, stdout=CID + "\n"),
                _Completed(0, stdout="true\n"),
                _http(200, {"status": "ok"}),
                _Completed(0, stdout=CID),
            ],
        )

        assert mod.main([*BASE, "--timeout", "10", "--health-version", "optional"]) == 0
        assert queue == []

    def test_optional_health_version_that_disagrees_exits_1(self, monkeypatch, capsys):
        mod = _load()
        _calls, queue = _scripted(
            monkeypatch,
            mod,
            [
                _Completed(0, stdout=_version_line()),
                _Completed(0, stdout=CID + "\n"),
                _Completed(0, stdout="true\n"),
                _http(200, {"status": "ok", "version": "0.4.0"}),
                _Completed(0, stdout=CID),
            ],
        )

        assert mod.main([*BASE, "--timeout", "10", "--health-version", "optional"]) == 1
        assert queue == []
        assert "reports version 0.4.0 != metadata 0.12.0" in capsys.readouterr().err

    @pytest.mark.parametrize("version", ["0.12.0-rc.1", "0.12.0+githash"])
    def test_a_prerelease_or_local_version_is_not_a_usage_error(self, monkeypatch, capsys, version):
        """The usage check rejects ``v0.12.0``. A real prerelease must get past it and still fail closed when docker cannot run."""
        mod = _load()
        calls, _queue = _scripted(monkeypatch, mod, [_Completed(1, stderr="no such image")])

        assert mod.main(["--image", IMAGE, "--dist", "juniper-cascor", "--port", "8200", "--expect-version", version]) == 2
        assert len(calls) == 1
        err = capsys.readouterr().err
        assert "is not X.Y.Z" not in err
        assert "could not run python" in err

    def test_a_failing_removal_does_not_hide_a_passing_verdict(self, monkeypatch):
        """``rm`` is best-effort. A non-zero removal after a healthy serve is still a pass."""
        mod = _load()
        calls, queue = _scripted(
            monkeypatch,
            mod,
            [
                _Completed(0, stdout=_version_line()),
                _Completed(0, stdout=CID + "\n"),
                _Completed(0, stdout="true\n"),
                _http(200, {"status": "ok", "version": "0.12.0"}),
                _Completed(1, stderr="removal failed"),
            ],
        )

        assert mod.main([*BASE, "--timeout", "10"]) == 0
        assert _argv(calls)[-1] == ["docker", "rm", "-f", CID]
        assert queue == []


class TestGetProbe:
    def test_a_nonzero_exec_is_not_an_http_status(self, monkeypatch):
        mod = _load()
        _scripted(monkeypatch, mod, [_Completed(1, stdout="200\n{}")])

        status, body = mod._get(CID, 8200, "/v1/health")
        assert status is None
        assert "200" in body

    def test_empty_output_is_not_an_http_status(self, monkeypatch):
        mod = _load()
        _scripted(monkeypatch, mod, [_Completed(0, stdout="   \n")])

        status, body = mod._get(CID, 8200, "/v1/health")
        assert status is None
        assert body == ""

    def test_a_non_json_body_keeps_the_status_and_the_text(self, monkeypatch):
        """Stderr concatenated onto a JSON body must not be parsed into a passing version."""
        mod = _load()
        _scripted(monkeypatch, mod, [_Completed(0, stdout="200\n{", stderr="truncated")])

        status, body = mod._get(CID, 8200, "/v1/health")
        assert status == 200
        assert body == "{truncated"

    def test_stderr_with_no_stdout_is_the_probe_text(self, monkeypatch):
        """Docker merges the streams in stdout-then-stderr order. A warning and no body is not a status."""
        mod = _load()
        _scripted(monkeypatch, mod, [_Completed(0, stdout="", stderr="warning: no response\n")])

        status, body = mod._get(CID, 8200, "/v1/health")
        assert status is None
        assert body == "warning: no response"
