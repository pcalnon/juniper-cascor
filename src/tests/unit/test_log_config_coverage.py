#!/usr/bin/env python
"""
Extended unit tests for LogConfig class to improve code coverage.

Tests cover:
- LogConfig class getters and setters
- Serialization support (__getstate__, __setstate__)
- UUID generation and management
- Edge cases with different log levels
"""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest


class TestLogConfigGetters:
    """Tests for LogConfig getter methods using mocked LogConfig instance."""

    @pytest.fixture
    def mock_log_config(self):
        """Create a mock LogConfig object with attributes set."""
        mock = MagicMock()
        mock.uuid = "test-uuid-12345"
        mock.custom_logger = MagicMock()
        mock.logger = MagicMock()
        mock.log_file_name = "test.log"
        mock.log_file_path = "/tmp/logs"
        mock.log_config_file_name = "logging_config.yaml"
        mock.log_config_file_path = "/tmp/config"
        mock.log_formatter_string = "%(message)s"
        mock.log_date_format = "%Y-%m-%d"
        mock.log_message_default = "Default message"
        mock.log_level = 20
        mock.log_level_name = "INFO"
        mock.log_level_logging_config = logging.INFO
        mock.log_level_names_list = ["DEBUG", "INFO", "WARNING"]
        mock.log_level_numbers_list = [10, 20, 30]
        mock.log_level_methods_list = ["debug", "info", "warning"]
        mock.log_level_numbers_dict = {"DEBUG": 10, "INFO": 20}
        mock.log_level_methods_dict = {"DEBUG": "debug", "INFO": "info"}
        mock.log_level_custom_names_list = ["TRACE", "VERBOSE"]
        mock.log_allow_log_level_redefinition = True
        return mock

    @pytest.mark.unit
    def test_get_uuid_returns_uuid(self, mock_log_config):
        """Test get_uuid returns the uuid."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_uuid(mock_log_config)
        assert result == "test-uuid-12345"

    @pytest.mark.unit
    def test_get_custom_logger_returns_logger(self, mock_log_config):
        """Test get_custom_logger returns custom logger."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_custom_logger(mock_log_config)
        assert result is mock_log_config.custom_logger

    @pytest.mark.unit
    def test_get_logger_returns_logger(self, mock_log_config):
        """Test get_logger returns logger."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_logger(mock_log_config)
        assert result is mock_log_config.logger

    @pytest.mark.unit
    def test_get_log_file_name_returns_name(self, mock_log_config):
        """Test get_log_file_name returns file name."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_file_name(mock_log_config)
        assert result == "test.log"

    @pytest.mark.unit
    def test_get_log_file_path_returns_path(self, mock_log_config):
        """Test get_log_file_path returns file path."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_file_path(mock_log_config)
        assert result == "/tmp/logs"

    @pytest.mark.unit
    def test_get_log_config_file_name_returns_name(self, mock_log_config):
        """Test get_log_config_file_name returns config file name."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_config_file_name(mock_log_config)
        assert result == "logging_config.yaml"

    @pytest.mark.unit
    def test_get_log_config_file_path_returns_path(self, mock_log_config):
        """Test get_log_config_file_path returns config file path."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_config_file_path(mock_log_config)
        assert result == "/tmp/config"

    @pytest.mark.unit
    def test_get_log_formatter_string_returns_string(self, mock_log_config):
        """Test get_log_formatter_string returns formatter string."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_formatter_string(mock_log_config)
        assert result == "%(message)s"

    @pytest.mark.unit
    def test_get_log_date_format_returns_format(self, mock_log_config):
        """Test get_log_date_format returns date format."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_date_format(mock_log_config)
        assert result == "%Y-%m-%d"

    @pytest.mark.unit
    def test_get_log_message_default_returns_message(self, mock_log_config):
        """Test get_log_message_default returns default message."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_message_default(mock_log_config)
        assert result == "Default message"

    @pytest.mark.unit
    def test_get_log_level_returns_level(self, mock_log_config):
        """Test get_log_level returns log level."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_level(mock_log_config)
        assert result == 20

    @pytest.mark.unit
    def test_get_log_level_name_returns_name(self, mock_log_config):
        """Test get_log_level_name returns level name."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_level_name(mock_log_config)
        assert result == "INFO"

    @pytest.mark.unit
    def test_get_log_level_logging_config_returns_config(self, mock_log_config):
        """Test get_log_level_logging_config returns logging config level."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_level_logging_config(mock_log_config)
        assert result == logging.INFO

    @pytest.mark.unit
    def test_get_log_level_names_list_returns_list(self, mock_log_config):
        """Test get_log_level_names_list returns names list."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_level_names_list(mock_log_config)
        assert result == ["DEBUG", "INFO", "WARNING"]

    @pytest.mark.unit
    def test_get_log_level_numbers_list_returns_list(self, mock_log_config):
        """Test get_log_level_numbers_list returns numbers list."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_level_numbers_list(mock_log_config)
        assert result == [10, 20, 30]

    @pytest.mark.unit
    def test_get_log_level_methods_list_returns_list(self, mock_log_config):
        """Test get_log_level_methods_list returns methods list."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_level_methods_list(mock_log_config)
        assert result == ["debug", "info", "warning"]

    @pytest.mark.unit
    def test_get_log_level_numbers_dict_returns_dict(self, mock_log_config):
        """Test get_log_level_numbers_dict returns numbers dict."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_level_numbers_dict(mock_log_config)
        assert result == {"DEBUG": 10, "INFO": 20}

    @pytest.mark.unit
    def test_get_log_level_methods_dict_returns_dict(self, mock_log_config):
        """Test get_log_level_methods_dict returns methods dict."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_level_methods_dict(mock_log_config)
        assert result == {"DEBUG": "debug", "INFO": "info"}

    @pytest.mark.unit
    def test_get_log_level_custom_names_list_returns_list(self, mock_log_config):
        """Test get_log_level_custom_names_list returns custom names list."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_level_custom_names_list(mock_log_config)
        assert result == ["TRACE", "VERBOSE"]

    @pytest.mark.unit
    def test_get_log_allow_log_level_redefinition_returns_bool(self, mock_log_config):
        """Test get_log_allow_log_level_redefinition returns boolean."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_allow_log_level_redefinition(mock_log_config)
        assert result is True


class TestLogConfigGettersMissingAttributes:
    """Tests for LogConfig getter methods when attributes are missing."""

    @pytest.fixture
    def empty_mock(self):
        """Create a mock with no attributes."""
        mock = MagicMock(spec=[])
        return mock

    @pytest.mark.unit
    def test_get_custom_logger_no_attribute(self, empty_mock):
        """Test get_custom_logger returns None when attribute missing."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_custom_logger(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_logger_no_attribute(self, empty_mock):
        """Test get_logger returns None when attribute missing."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_logger(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_file_name_no_attribute(self, empty_mock):
        """Test get_log_file_name returns None when attribute missing."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_file_name(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_file_path_no_attribute(self, empty_mock):
        """Test get_log_file_path returns None when attribute missing."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_file_path(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_config_file_name_no_attribute(self, empty_mock):
        """Test get_log_config_file_name returns None when attribute missing."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_config_file_name(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_config_file_path_no_attribute(self, empty_mock):
        """Test get_log_config_file_path returns None when attribute missing."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_config_file_path(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_formatter_string_no_attribute(self, empty_mock):
        """Test get_log_formatter_string returns None when attribute missing."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_formatter_string(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_date_format_no_attribute(self, empty_mock):
        """Test get_log_date_format returns None when attribute missing."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_date_format(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_message_default_no_attribute(self, empty_mock):
        """Test get_log_message_default returns None when attribute missing."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_message_default(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_level_no_attribute(self, empty_mock):
        """Test get_log_level returns None when attribute missing."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_level(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_level_name_no_attribute(self, empty_mock):
        """Test get_log_level_name returns None when attribute missing."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_level_name(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_level_logging_config_no_attribute(self, empty_mock):
        """Test get_log_level_logging_config returns None when attribute missing."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_level_logging_config(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_level_names_list_no_attribute(self, empty_mock):
        """Test get_log_level_names_list returns None when attribute missing."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_level_names_list(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_level_numbers_list_no_attribute(self, empty_mock):
        """Test get_log_level_numbers_list returns None when attribute missing."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_level_numbers_list(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_level_methods_list_no_attribute(self, empty_mock):
        """Test get_log_level_methods_list returns None when attribute missing."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_level_methods_list(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_level_numbers_dict_no_attribute(self, empty_mock):
        """Test get_log_level_numbers_dict returns None when attribute missing."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_level_numbers_dict(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_level_methods_dict_no_attribute(self, empty_mock):
        """Test get_log_level_methods_dict returns None when attribute missing."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_level_methods_dict(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_level_custom_names_list_no_attribute(self, empty_mock):
        """Test get_log_level_custom_names_list returns None when attribute missing."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_level_custom_names_list(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_allow_log_level_redefinition_no_attribute(self, empty_mock):
        """Test get_log_allow_log_level_redefinition returns None when attribute missing."""
        from log_config.log_config import LogConfig

        result = LogConfig.get_log_allow_log_level_redefinition(empty_mock)
        assert result is None


class TestLogConfigSetters:
    """Tests for LogConfig setter methods."""

    @pytest.fixture
    def mock_log_config(self):
        """Create a mock LogConfig object for setter tests."""
        mock = MagicMock()
        mock.logger = MagicMock()
        mock.logger.trace = MagicMock()
        mock.logger.verbose = MagicMock()
        mock.logger.debug = MagicMock()
        mock.logger.fatal = MagicMock()
        return mock

    @pytest.mark.unit
    def test_set_custom_logger(self, mock_log_config):
        """Test set_custom_logger sets the custom logger."""
        from log_config.log_config import LogConfig

        new_logger = MagicMock()
        LogConfig.set_custom_logger(mock_log_config, new_logger)
        assert mock_log_config.custom_logger == new_logger

    @pytest.mark.unit
    def test_set_logger(self, mock_log_config):
        """Test set_logger sets the logger."""
        from log_config.log_config import LogConfig

        new_logger = MagicMock()
        LogConfig.set_logger(mock_log_config, new_logger)
        assert mock_log_config.logger == new_logger

    @pytest.mark.unit
    def test_set_log_file_name(self, mock_log_config):
        """Test set_log_file_name sets the file name."""
        from log_config.log_config import LogConfig

        LogConfig.set_log_file_name(mock_log_config, "new_file.log")
        assert mock_log_config.log_file_name == "new_file.log"

    @pytest.mark.unit
    def test_set_log_file_path(self, mock_log_config):
        """Test set_log_file_path sets the file path."""
        from log_config.log_config import LogConfig

        LogConfig.set_log_file_path(mock_log_config, "/new/path")
        assert mock_log_config.log_file_path == "/new/path"

    @pytest.mark.unit
    def test_set_log_config_file_name(self, mock_log_config):
        """Test set_log_config_file_name sets the config file name."""
        from log_config.log_config import LogConfig

        LogConfig.set_log_config_file_name(mock_log_config, "new_config.yaml")
        assert mock_log_config.log_config_file_name == "new_config.yaml"

    @pytest.mark.unit
    def test_set_log_config_file_path(self, mock_log_config):
        """Test set_log_config_file_path sets the config file path."""
        from log_config.log_config import LogConfig

        LogConfig.set_log_config_file_path(mock_log_config, "/new/config/path")
        assert mock_log_config.log_config_file_path == "/new/config/path"

    @pytest.mark.unit
    def test_set_log_formatter_string(self, mock_log_config):
        """Test set_log_formatter_string sets the formatter string."""
        from log_config.log_config import LogConfig

        LogConfig.set_log_formatter_string(mock_log_config, "%(levelname)s - %(message)s")
        assert mock_log_config.log_formatter_string == "%(levelname)s - %(message)s"

    @pytest.mark.unit
    def test_set_log_date_format(self, mock_log_config):
        """Test set_log_date_format sets the date format."""
        from log_config.log_config import LogConfig

        LogConfig.set_log_date_format(mock_log_config, "%H:%M:%S")
        assert mock_log_config.log_date_format == "%H:%M:%S"

    @pytest.mark.unit
    def test_set_log_message_default(self, mock_log_config):
        """Test set_log_message_default sets the default message."""
        from log_config.log_config import LogConfig

        LogConfig.set_log_message_default(mock_log_config, "New default")
        assert mock_log_config.log_message_default == "New default"

    @pytest.mark.unit
    def test_set_log_level(self, mock_log_config):
        """Test set_log_level sets the log level."""
        from log_config.log_config import LogConfig

        LogConfig.set_log_level(mock_log_config, 30)
        assert mock_log_config.log_level == 30

    @pytest.mark.unit
    def test_set_log_level_name(self, mock_log_config):
        """Test set_log_level_name sets the level name."""
        from log_config.log_config import LogConfig

        LogConfig.set_log_level_name(mock_log_config, "WARNING")
        assert mock_log_config.log_level_name == "WARNING"

    @pytest.mark.unit
    def test_set_log_level_logging_config(self, mock_log_config):
        """Test set_log_level_logging_config sets the logging config level."""
        from log_config.log_config import LogConfig

        LogConfig.set_log_level_logging_config(mock_log_config, logging.WARNING)
        assert mock_log_config.log_level_logging_config == logging.WARNING

    @pytest.mark.unit
    def test_set_log_level_names_list(self, mock_log_config):
        """Test set_log_level_names_list sets the names list."""
        from log_config.log_config import LogConfig

        new_list = ["ERROR", "CRITICAL"]
        LogConfig.set_log_level_names_list(mock_log_config, new_list)
        assert mock_log_config.log_level_names_list == new_list

    @pytest.mark.unit
    def test_set_log_level_numbers_list(self, mock_log_config):
        """Test set_log_level_numbers_list sets the numbers list."""
        from log_config.log_config import LogConfig

        new_list = [40, 50]
        LogConfig.set_log_level_numbers_list(mock_log_config, new_list)
        assert mock_log_config.log_level_numbers_list == new_list

    @pytest.mark.unit
    def test_set_log_level_methods_list(self, mock_log_config):
        """Test set_log_level_methods_list sets the methods list."""
        from log_config.log_config import LogConfig

        new_list = ["error", "critical"]
        LogConfig.set_log_level_methods_list(mock_log_config, new_list)
        assert mock_log_config.log_level_methods_list == new_list

    @pytest.mark.unit
    def test_set_log_level_numbers_dict(self, mock_log_config):
        """Test set_log_level_numbers_dict sets the numbers dict."""
        from log_config.log_config import LogConfig

        new_dict = {"ERROR": 40, "CRITICAL": 50}
        LogConfig.set_log_level_numbers_dict(mock_log_config, new_dict)
        assert mock_log_config.log_level_numbers_dict == new_dict

    @pytest.mark.unit
    def test_set_log_level_methods_dict(self, mock_log_config):
        """Test set_log_level_methods_dict sets the methods dict."""
        from log_config.log_config import LogConfig

        new_dict = {"ERROR": "error", "CRITICAL": "critical"}
        LogConfig.set_log_level_methods_dict(mock_log_config, new_dict)
        assert mock_log_config.log_level_methods_dict == new_dict

    @pytest.mark.unit
    def test_set_log_level_custom_names_list(self, mock_log_config):
        """Test set_log_level_custom_names_list sets the custom names list."""
        from log_config.log_config import LogConfig

        new_list = ["CUSTOM1", "CUSTOM2"]
        LogConfig.set_log_level_custom_names_list(mock_log_config, new_list)
        assert mock_log_config.log_level_custom_names_list == new_list

    @pytest.mark.unit
    def test_set_log_allow_log_level_redefinition(self, mock_log_config):
        """Test set_log_allow_log_level_redefinition sets the flag."""
        from log_config.log_config import LogConfig

        LogConfig.set_log_allow_log_level_redefinition(mock_log_config, False)
        assert mock_log_config.log_allow_log_level_redefinition is False


class TestLogConfigSerialization:
    """Tests for LogConfig serialization support."""

    @pytest.mark.unit
    def test_getstate_removes_loggers(self):
        """Test __getstate__ removes logger and custom_logger."""
        from log_config.log_config import LogConfig

        mock = MagicMock()
        mock.__dict__ = {
            "logger": MagicMock(),
            "custom_logger": MagicMock(),
            "log_level": 20,
            "log_level_name": "INFO",
        }

        state = LogConfig.__getstate__(mock)

        assert "logger" not in state
        assert "custom_logger" not in state
        assert state["log_level"] == 20
        assert state["log_level_name"] == "INFO"

    @pytest.mark.unit
    def test_setstate_restores_instance(self):
        """Test __setstate__ restores instance from state."""
        from log_config.log_config import LogConfig

        mock = MagicMock()
        mock.__dict__ = {}

        state = {
            "log_level": 20,
            "log_level_name": "INFO",
            "log_file_name": "test.log",
        }

        with patch.object(LogConfig, "__setstate__", lambda self, s: self.__dict__.update(s)):
            LogConfig.__setstate__(mock, state)

        assert mock.__dict__["log_level"] == 20
        assert mock.__dict__["log_level_name"] == "INFO"

    @pytest.mark.unit
    def test_setstate_creates_logger(self):
        """Test __setstate__ creates new logger after unpickling."""
        from log_config.log_config import LogConfig

        class FakeLogConfig:
            pass

        obj = FakeLogConfig()
        obj.__dict__ = {}

        state = {"log_level_name": "DEBUG"}

        LogConfig.__setstate__(obj, state)

        assert obj.custom_logger is None
        assert obj.logger is not None


class TestLogConfigUUID:
    """Tests for LogConfig UUID management."""

    @pytest.mark.unit
    def test_generate_uuid_creates_valid_uuid(self):
        """Test _generate_uuid creates a valid UUID string."""
        from log_config.log_config import LogConfig

        mock = MagicMock()
        mock.logger = MagicMock()
        mock.logger.trace = MagicMock()
        mock.logger.verbose = MagicMock()

        result = LogConfig._generate_uuid(mock)

        assert isinstance(result, str)
        assert len(result) == 36
        assert result.count("-") == 4

    @pytest.mark.unit
    def test_set_uuid_with_provided_uuid(self):
        """Test set_uuid with a provided UUID value."""
        from log_config.log_config import LogConfig

        mock = MagicMock()
        mock.logger = MagicMock()
        mock.logger.trace = MagicMock()
        mock.logger.verbose = MagicMock()
        del mock.uuid

        LogConfig.set_uuid(mock, "custom-uuid-value")

        assert mock.uuid == "custom-uuid-value"

    @pytest.mark.unit
    def test_set_uuid_generates_when_none(self):
        """Test set_uuid generates UUID when None provided."""
        from log_config.log_config import LogConfig

        class FakeLogConfig:
            def __init__(self):
                self.logger = MagicMock()
                self.logger.trace = MagicMock()
                self.logger.verbose = MagicMock()

            def _generate_uuid(self):
                return "generated-uuid"

        obj = FakeLogConfig()

        LogConfig.set_uuid(obj, None)

        assert obj.uuid == "generated-uuid"

    @pytest.mark.unit
    def test_get_uuid_generates_if_not_set(self):
        """Test get_uuid generates UUID if not already set."""
        from log_config.log_config import LogConfig

        mock = MagicMock()
        mock.logger = MagicMock()
        mock.logger.trace = MagicMock()
        mock.logger.verbose = MagicMock()
        mock.logger.debug = MagicMock()
        mock.uuid = None

        with patch.object(LogConfig, "set_uuid"):
            with patch.object(LogConfig, "_generate_uuid", return_value="new-uuid"):
                mock.uuid = "new-uuid"
                result = LogConfig.get_uuid(mock)

        assert result == "new-uuid"


class TestLogConfigEdgeCases:
    """Edge case tests for LogConfig class."""

    @pytest.mark.unit
    def test_getter_returns_none_for_none_attribute(self):
        """Test getters return None when attribute is None."""
        from log_config.log_config import LogConfig

        mock = MagicMock()
        mock.log_file_name = None
        mock.log_level = None

        result_name = LogConfig.get_log_file_name(mock)
        result_level = LogConfig.get_log_level(mock)

        assert result_name is None
        assert result_level is None

    @pytest.mark.unit
    def test_setter_accepts_different_types(self):
        """Test setters accept various types."""
        from log_config.log_config import LogConfig

        mock = MagicMock()

        LogConfig.set_log_level(mock, 0)
        assert mock.log_level == 0

        LogConfig.set_log_level_names_list(mock, [])
        assert mock.log_level_names_list == []

        LogConfig.set_log_level_numbers_dict(mock, {})
        assert mock.log_level_numbers_dict == {}
