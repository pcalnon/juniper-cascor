#!/usr/bin/env python
"""Regression tests for plan unit C4 (JUNIPER_2026-07-11_JUNIPER-CANOPY_TRAINING-RUNTIME-DEFECTS-PLAN §5 T5 / §7 C4).

Incident (2026-07-10 18:15 session): cascor's uvicorn access log went permanently
silent the moment training started — the last access record is the 18:19:56
``POST /v1/training/dataset`` right before ``start_training``; every later API
request (snapshot 422s, param 422s, start attempts) was invisible server-side.

Mechanism: ``start_training``'s create-on-start path constructs the first
``CascadeCorrelationNetwork`` inside the uvicorn process
(``api/lifecycle/manager.py`` ``_create_network_locked``), whose
``_init_logging_system`` builds ``LogConfig`` -> ``Logger.__init__``, which applies
``logging.config.dictConfig`` on ``conf/logging_config.yaml``. That YAML omitted
``disable_existing_loggers``; the stdlib defaults it to **True**, so every logger
created before training start — ``uvicorn.access`` / ``uvicorn.error`` / ``uvicorn`` /
``juniper_cascor.api`` — was disabled, and the YAML ``root:`` section replaced the
handlers installed by ``api.observability.configure_logging``.

Pinned here:

1. The REAL ``_init_logging_system`` (via the ``real_init_logging_system``
   conftest fixture — the suite-wide fast-logging fixture stays in place for
   every other test) must leave pre-existing host-application loggers enabled,
   keep their handlers attached, and leave root's handlers untouched.
2. Route-level start failures must log at WARNING (not DEBUG) so rejected
   starts are visible in server logs (``api/routes/training.py``).
"""

import logging
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.app import create_app
from api.settings import Settings
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
from log_config.logger.logger import Logger

pytestmark = pytest.mark.unit

_HOST_LOGGER_NAMES = ("uvicorn.access", "uvicorn.error", "uvicorn", "juniper_cascor.api")


def _make_network(**overrides):
    defaults = {
        "input_size": 2,
        "output_size": 2,
        "random_seed": 42,
        "candidate_pool_size": 2,
        "candidate_epochs": 3,
        "output_epochs": 3,
        "max_hidden_units": 2,
        "patience": 1,
    }
    defaults.update(overrides)
    return CascadeCorrelationNetwork(config=CascadeCorrelationConfig(**defaults))


class TestUvicornAccessLogSurvivesTrainingLoggingInit:
    """The engine's logging init must not clobber the host application's logging."""

    def test_host_loggers_survive_real_logging_init(self, real_init_logging_system, tmp_path):
        """uvicorn.* and the api logger stay enabled — and root keeps its handlers — after the REAL ``_init_logging_system`` runs its ``dictConfig`` path.

        Simulates the API server's state at training start: uvicorn's loggers exist
        with handlers (uvicorn's own dictConfig ran at startup) and root carries the
        handlers installed by ``configure_logging``. Forces the Logger singleton flag
        False so the genuine first-configuration ``dictConfig`` path executes, exactly
        as in the incident process.
        """
        root = logging.getLogger()
        juniper_logger = logging.getLogger("juniper")
        host_loggers = {name: logging.getLogger(name) for name in _HOST_LOGGER_NAMES}

        saved_states = {name: (lg.disabled, list(lg.handlers), lg.level, lg.propagate) for name, lg in host_loggers.items()}
        saved_root = (list(root.handlers), root.level)
        saved_juniper = (juniper_logger.disabled, list(juniper_logger.handlers), juniper_logger.level, juniper_logger.propagate)
        saved_flag = Logger.SingletonLoggingConfigured

        access_logger = host_loggers["uvicorn.access"]
        access_handler = logging.StreamHandler()
        access_handler.set_name("c4_sentinel_access_handler")
        root_sentinel = logging.StreamHandler()
        root_sentinel.set_name("c4_sentinel_root_handler")

        try:
            # Arrange: uvicorn-like startup state (uvicorn's LOGGING_CONFIG gives
            # uvicorn.access its own handler, level INFO, propagate False) plus a
            # root handler standing in for configure_logging's console/file pair.
            for lg in host_loggers.values():
                lg.disabled = False
            access_logger.addHandler(access_handler)
            access_logger.setLevel(logging.INFO)
            access_logger.propagate = False
            root.addHandler(root_sentinel)

            # Force the genuine first-configuration path so dictConfig actually runs.
            Logger.SingletonLoggingConfigured = False

            log_dir = tmp_path / "logs"
            log_dir.mkdir()
            network = _make_network(log_file_name="c4_access_log_survival", log_file_path=str(log_dir), log_level_name="WARNING")
            real_init_logging_system(network)

            # The seam must actually have been exercised — otherwise the assertions below are vacuous.
            assert Logger.is_configured(), "dictConfig did not run — the test failed to exercise the reconfiguration seam"

            assert not access_logger.disabled, "uvicorn.access was disabled by the training logging init (C4 incident: access log dead after start_training)"
            assert access_handler in access_logger.handlers, "uvicorn.access lost its handler across the training logging init"
            assert access_logger.getEffectiveLevel() == logging.INFO, "uvicorn.access effective level changed across the training logging init"
            assert not host_loggers["uvicorn.error"].disabled, "uvicorn.error was disabled by the training logging init"
            assert not host_loggers["uvicorn"].disabled, "uvicorn was disabled by the training logging init"
            assert not host_loggers["juniper_cascor.api"].disabled, "the api logger was disabled by the training logging init (C4 incident: API activity invisible server-side)"
            assert root_sentinel in root.handlers, "root handlers were clobbered by the engine's YAML root section (host application owns root)"
        finally:
            for name, lg in host_loggers.items():
                disabled, handlers, level, propagate = saved_states[name]
                lg.disabled = disabled
                lg.handlers[:] = handlers
                lg.setLevel(level)
                lg.propagate = propagate
            root.handlers[:] = saved_root[0]
            root.setLevel(saved_root[1])
            juniper_logger.disabled = saved_juniper[0]
            juniper_logger.handlers[:] = saved_juniper[1]
            juniper_logger.setLevel(saved_juniper[2])
            juniper_logger.propagate = saved_juniper[3]
            Logger.SingletonLoggingConfigured = saved_flag


class TestStartFailureLogsAtWarning:
    """Rejected training starts must be visible in server logs (T5: they logged at DEBUG)."""

    @pytest.fixture
    def client(self):
        settings = Settings(auto_start=False)
        app = create_app(settings)
        with TestClient(app) as c:
            yield c

    def test_start_training_rejection_logs_warning(self, client, caplog):
        """A 409-rejected start emits a WARNING record naming the reason."""
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})
        try:
            with caplog.at_level(logging.WARNING, logger="juniper_cascor.api.routes.training"):
                response = client.post("/v1/training/start")

            assert response.status_code == 409
            warning_records = [r for r in caplog.records if r.name == "juniper_cascor.api.routes.training" and r.levelno == logging.WARNING and "Start training failed" in r.getMessage()]
            assert warning_records, "rejected start did not log at WARNING on juniper_cascor.api.routes.training (C4: start failures were DEBUG-only and invisible in server logs)"
            assert "Training data not provided" in warning_records[0].getMessage()
        finally:
            client.post("/v1/training/stop")
