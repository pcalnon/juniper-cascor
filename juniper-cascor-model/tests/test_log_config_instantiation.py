#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     test_log_config_instantiation.py
# Author:        Paul Calnon
#
# Date Created:  2026-07-03
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#     Real-instantiation coverage for the resilient logging core shipped with
#     juniper-cascor-model. The other logger/log_config suites exercise the
#     getters/setters against ``Fake*`` subclasses that bypass ``__init__``;
#     these tests drive the actual ``Logger.__init__`` and ``LogConfig.__init__``
#     bootstrap paths (custom-level registration, YAML dictConfig success and the
#     best-effort fallback, the already-configured short-circuit) so the package's
#     first CI coverage gate (per-file >=90 / sub-module pooled >=95, C-5) reflects
#     the bootstrap the distributed worker relies on. Every path is hermetic:
#     config + log files are redirected to ``tmp_path`` and the global logging
#     singleton state is snapshotted and restored, so the tests are order-independent.
#
#####################################################################################################################################################################################################

from __future__ import annotations

import logging
import textwrap

import pytest

from log_config.log_config import LogConfig
from log_config.logger.logger import Logger

# A minimal, self-contained logging.config.dictConfig document with a FileHandler so the
# __init__ handler-filename-override branch (absolute path injection) is exercised. The
# FileHandler filename is overridden by Logger.__init__ to <log_file_path>/<log_file_name>.
_VALID_LOG_CONFIG_YAML = textwrap.dedent(
    """
    version: 1
    disable_existing_loggers: false
    formatters:
      simple:
        format: "%(levelname)s %(message)s"
    handlers:
      console:
        class: logging.StreamHandler
        formatter: simple
      file:
        class: logging.FileHandler
        filename: placeholder.log
        formatter: simple
    root:
      level: INFO
      handlers: [console, file]
    """
)


@pytest.fixture
def restore_logging_state():
    """Snapshot and restore all global logging state mutated by real instantiation.

    ``Logger.__init__`` flips the ``SingletonLoggingConfigured`` class flag, mutates the
    class ``_log_level``, registers custom level names globally, and (on the dictConfig
    success path) reconfigures the root logger's handlers. Restoring these keeps the tests
    order-independent so they cannot leak into the ported logger/log_config suites.
    """
    saved_configured = Logger.SingletonLoggingConfigured
    saved_level = Logger._log_level
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_root_level = root.level
    try:
        yield
    finally:
        Logger.SingletonLoggingConfigured = saved_configured
        Logger._log_level = saved_level
        root.handlers[:] = saved_handlers
        root.setLevel(saved_root_level)


class TestLoggerRealInstantiation:
    """Drive the real ``Logger.__init__`` bootstrap (not the Fake* subclasses)."""

    def test_missing_config_file_degrades_to_basic_configuration(self, tmp_path, restore_logging_state):
        # A non-existent config file must be caught and degrade to a basic StreamHandler
        # rather than raising — the resilient-logging contract (CW-05 gap #3).
        Logger.SingletonLoggingConfigured = False
        instance = Logger(
            _Logger__log_config_file_path=str(tmp_path),
            _Logger__log_config_file_name="does-not-exist.yaml",
            _Logger__log_file_path=str(tmp_path),
            _Logger__log_level_redefinition=True,
        )

        assert instance.handlers, "basic-configuration fallback must attach a handler"
        assert instance.log_config_file_name == "does-not-exist.yaml"

    def test_valid_config_applies_dictconfig_and_marks_configured(self, tmp_path, restore_logging_state):
        # A valid dictConfig document applies successfully, overrides the FileHandler
        # filename to an absolute path under the log dir, and flips the singleton flag.
        (tmp_path / "logging_config.yaml").write_text(_VALID_LOG_CONFIG_YAML)
        Logger.SingletonLoggingConfigured = False

        instance = Logger(
            _Logger__log_config_file_path=str(tmp_path),
            _Logger__log_config_file_name="logging_config.yaml",
            _Logger__log_file_path=str(tmp_path),
            _Logger__log_level_redefinition=True,
        )

        assert Logger.is_configured() is True
        assert instance.get_logger() is instance
        # The FileHandler filename override + makedirs must have targeted the tmp log dir.
        assert (tmp_path / "juniper_cascor.log").exists()

    def test_second_instantiation_short_circuits_when_already_configured(self, tmp_path, restore_logging_state):
        # Once configured globally, a subsequent Logger() must skip YAML configuration
        # entirely (the ``else`` branch of the singleton guard).
        (tmp_path / "logging_config.yaml").write_text(_VALID_LOG_CONFIG_YAML)
        Logger.SingletonLoggingConfigured = True

        instance = Logger(
            _Logger__log_config_file_path=str(tmp_path),
            _Logger__log_config_file_name="logging_config.yaml",
            _Logger__log_file_path=str(tmp_path),
            _Logger__log_level_redefinition=True,
        )

        assert Logger.is_configured() is True
        assert instance.handlers

    def test_instantiation_registers_custom_level_methods(self, tmp_path, restore_logging_state):
        # With redefinition allowed, the custom-level registration loop must attach the
        # dynamically-generated log methods (trace/verbose/fatal) onto the instance.
        Logger.SingletonLoggingConfigured = False
        instance = Logger(
            _Logger__log_config_file_path=str(tmp_path),
            _Logger__log_config_file_name="missing.yaml",
            _Logger__log_file_path=str(tmp_path),
            _Logger__log_level_redefinition=True,
        )

        for method_name in ("trace", "verbose", "fatal"):
            assert callable(getattr(instance, method_name))


class TestLoggerClassMethodEdges:
    """Cover the residual class-method edges the ported suite leaves uncovered."""

    def test_is_enabled_for_defaults_to_notset_when_level_unmappable(self, restore_logging_state):
        # When the configured level name cannot be mapped to a number, isEnabledFor must
        # fall back to NOTSET (0) so every level is considered enabled.
        Logger._log_level = "NOT-A-REAL-LEVEL"
        assert Logger.isEnabledFor(logging.CRITICAL) is True
        assert Logger.isEnabledFor(0) is True

    def test_set_level_round_trips_name_and_number(self, restore_logging_state):
        Logger.set_level("DEBUG")
        assert Logger.get_level() == "DEBUG"
        # An invalid level resets to the configured default rather than raising.
        Logger.set_level(object())
        assert Logger.get_level() in Logger._level_names.values()


class TestLogConfigRealInstantiation:
    """Drive the real ``LogConfig.__init__`` bootstrap (constructs a Logger internally)."""

    def test_log_config_constructs_and_exposes_uuid(self, tmp_path, restore_logging_state):
        (tmp_path / "logging_config.yaml").write_text(_VALID_LOG_CONFIG_YAML)
        Logger.SingletonLoggingConfigured = False

        log_config = LogConfig(
            _LogConfig__log_config_file_path=str(tmp_path),
            _LogConfig__log_config_file_name="logging_config.yaml",
            _LogConfig__log_file_path=str(tmp_path),
        )

        assert log_config.get_uuid()
        assert log_config.get_logger() is not None
        assert log_config.get_custom_logger() is not None
        # A handful of the getters over the freshly-built instance.
        assert log_config.get_log_file_path() == str(tmp_path)
        assert log_config.get_log_level_name()

    def test_log_config_get_uuid_generates_when_unset(self, tmp_path, restore_logging_state):
        (tmp_path / "logging_config.yaml").write_text(_VALID_LOG_CONFIG_YAML)
        Logger.SingletonLoggingConfigured = False
        log_config = LogConfig(
            _LogConfig__log_config_file_path=str(tmp_path),
            _LogConfig__log_config_file_name="logging_config.yaml",
            _LogConfig__log_file_path=str(tmp_path),
        )

        # Clearing the uuid forces the regeneration branch inside get_uuid().
        log_config.uuid = None
        regenerated = log_config.get_uuid()
        assert regenerated
        assert log_config.uuid == regenerated

    def test_log_config_set_uuid_twice_is_fatal(self, tmp_path, restore_logging_state):
        (tmp_path / "logging_config.yaml").write_text(_VALID_LOG_CONFIG_YAML)
        Logger.SingletonLoggingConfigured = False
        log_config = LogConfig(
            _LogConfig__log_config_file_path=str(tmp_path),
            _LogConfig__log_config_file_name="logging_config.yaml",
            _LogConfig__log_file_path=str(tmp_path),
        )

        # UUID is already set by __init__; re-setting must fail closed (guards identity).
        with pytest.raises(SystemExit):
            log_config.set_uuid("a-second-uuid")
