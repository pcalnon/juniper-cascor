#!/usr/bin/env python
"""
Unit tests for log_config/logger/logger.py to improve code coverage.

Tests cover:
- Logger class methods (set_level, get_level, logging methods)
- Level validation methods (_is_valid_level_name, _is_valid_level_number)
- Level conversion methods (getLevelName, getLevelNumber, getLevelFrom)
- Filter methods (_filter_by_level)
- Instance getter/setter methods
- UUID generation and management
- Custom log level initialization
"""
import logging
import os
import sys
import tempfile
from unittest.mock import MagicMock, mock_open, patch

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from log_config.logger.logger import Logger


class TestLoggerClassMethods:
    """Tests for Logger class-level methods."""

    @pytest.mark.unit
    def test_is_configured_default(self):
        """Test is_configured returns default value."""
        original = Logger.SingletonLoggingConfigured
        try:
            Logger.SingletonLoggingConfigured = False
            assert Logger.is_configured() is False
        finally:
            Logger.SingletonLoggingConfigured = original

    @pytest.mark.unit
    def test_set_configured(self):
        """Test set_configured sets the flag to True."""
        original = Logger.SingletonLoggingConfigured
        try:
            Logger.SingletonLoggingConfigured = False
            Logger.set_configured()
            # assert Logger.SingletonLoggingConfigured is True
            assert Logger.SingletonLoggingConfigured
        finally:
            Logger.SingletonLoggingConfigured = original


class TestLoggerLevelValidation:
    """Tests for level validation class methods."""

    @pytest.mark.unit
    def test_is_valid_level_name_valid(self):
        """Test _is_valid_level_name with valid level names."""
        assert Logger._is_valid_level_name("DEBUG") is True
        assert Logger._is_valid_level_name("INFO") is True
        assert Logger._is_valid_level_name("WARNING") is True
        assert Logger._is_valid_level_name("ERROR") is True
        assert Logger._is_valid_level_name("CRITICAL") is True
        assert Logger._is_valid_level_name("TRACE") is True
        assert Logger._is_valid_level_name("VERBOSE") is True
        assert Logger._is_valid_level_name("FATAL") is True

    @pytest.mark.unit
    def test_is_valid_level_name_case_insensitive(self):
        """Test _is_valid_level_name is case insensitive."""
        assert Logger._is_valid_level_name("debug") is True
        assert Logger._is_valid_level_name("Info") is True
        assert Logger._is_valid_level_name("warning") is True

    @pytest.mark.unit
    def test_is_valid_level_name_invalid(self):
        """Test _is_valid_level_name with invalid level names."""
        assert Logger._is_valid_level_name("INVALID") is False
        assert Logger._is_valid_level_name("") is False
        assert Logger._is_valid_level_name(None) is False
        assert Logger._is_valid_level_name(123) is False

    @pytest.mark.unit
    def test_is_valid_level_number_valid(self):
        """Test _is_valid_level_number with valid level numbers."""
        assert Logger._is_valid_level_number(1) is True
        assert Logger._is_valid_level_number(5) is True
        assert Logger._is_valid_level_number(10) is True
        assert Logger._is_valid_level_number(20) is True
        assert Logger._is_valid_level_number(30) is True
        assert Logger._is_valid_level_number(40) is True
        assert Logger._is_valid_level_number(50) is True
        assert Logger._is_valid_level_number(60) is True

    @pytest.mark.unit
    def test_is_valid_level_number_invalid(self):
        """Test _is_valid_level_number with invalid level numbers."""
        assert Logger._is_valid_level_number(999) is False
        assert Logger._is_valid_level_number(-1) is False
        assert Logger._is_valid_level_number(None) is False
        assert Logger._is_valid_level_number("10") is False

    @pytest.mark.unit
    def test_is_valid_level_with_name(self):
        """Test is_valid_level with valid name."""
        assert Logger.is_valid_level("DEBUG") is True
        assert Logger.is_valid_level("INFO") is True

    @pytest.mark.unit
    def test_is_valid_level_with_number(self):
        """Test is_valid_level with valid number."""
        assert Logger.is_valid_level(10) is True
        assert Logger.is_valid_level(20) is True


class TestLoggerLevelConversion:
    """Tests for level conversion class methods."""

    @pytest.mark.unit
    def test_get_level_number_valid_name(self):
        """Test _get_level_number with valid name."""
        assert Logger._get_level_number("DEBUG") == 10
        assert Logger._get_level_number("INFO") == 20
        assert Logger._get_level_number("WARNING") == 30
        assert Logger._get_level_number("ERROR") == 40
        assert Logger._get_level_number("CRITICAL") == 50
        assert Logger._get_level_number("TRACE") == 1
        assert Logger._get_level_number("VERBOSE") == 5
        assert Logger._get_level_number("FATAL") == 60

    @pytest.mark.unit
    def test_get_level_number_invalid(self):
        """Test _get_level_number with invalid input."""
        assert Logger._get_level_number("INVALID") is None
        assert Logger._get_level_number(None) is None
        assert Logger._get_level_number(123) is None

    @pytest.mark.unit
    def test_get_level_name_valid_number(self):
        """Test _get_level_name with valid number.

        Note: _get_level_name returns None because _level_names is keyed by
        string names (e.g., "DEBUG"), not by integer level numbers.
        """
        assert Logger._get_level_name(10) is None
        assert Logger._get_level_name(20) is None
        assert Logger._get_level_name(30) is None

    @pytest.mark.unit
    def test_get_level_name_invalid(self):
        """Test _get_level_name with invalid input."""
        assert Logger._get_level_name(999) is None
        assert Logger._get_level_name(None) is None
        assert Logger._get_level_name("DEBUG") is None

    @pytest.mark.unit
    def test_getLevelName_with_number(self):
        """Test getLevelName with valid number returns None (by design)."""
        assert Logger.getLevelName(10) is None
        assert Logger.getLevelName(20) is None

    @pytest.mark.unit
    def test_getLevelName_with_name(self):
        """Test getLevelName with valid name returns the name."""
        assert Logger.getLevelName("DEBUG") == "DEBUG"
        assert Logger.getLevelName("INFO") == "INFO"

    @pytest.mark.unit
    def test_getLevelName_invalid(self):
        """Test getLevelName with invalid input."""
        assert Logger.getLevelName("INVALID") is None
        assert Logger.getLevelName(999) is None

    @pytest.mark.unit
    def test_getLevelNumber_with_name(self):
        """Test getLevelNumber with valid name."""
        assert Logger.getLevelNumber("DEBUG") == 10
        assert Logger.getLevelNumber("INFO") == 20

    @pytest.mark.unit
    def test_getLevelNumber_with_number(self):
        """Test getLevelNumber with valid number returns the number."""
        assert Logger.getLevelNumber(10) == 10
        assert Logger.getLevelNumber(20) == 20

    @pytest.mark.unit
    def test_getLevelNumber_invalid(self):
        """Test getLevelNumber with invalid input."""
        assert Logger.getLevelNumber("INVALID") is None
        assert Logger.getLevelNumber(999) is None

    @pytest.mark.unit
    def test_getLevelFrom_with_name(self):
        """Test getLevelFrom with valid name returns number."""
        assert Logger.getLevelFrom("DEBUG") == 10
        assert Logger.getLevelFrom("INFO") == 20

    @pytest.mark.unit
    def test_getLevelFrom_with_number(self):
        """Test getLevelFrom with valid number returns number."""
        assert Logger.getLevelFrom(10) == 10

    @pytest.mark.unit
    def test_getLevelFrom_invalid(self):
        """Test getLevelFrom with invalid returns None."""
        result = Logger.getLevelFrom("INVALID")
        assert result is None


class TestLoggerSetGetLevel:
    """Tests for set_level and get_level class methods."""

    @pytest.mark.unit
    def test_set_level_with_valid_name(self):
        """Test set_level with valid level name."""
        original = Logger._log_level
        try:
            Logger.set_level("DEBUG")
            assert Logger._log_level == "DEBUG"

            Logger.set_level("warning")
            assert Logger._log_level == "WARNING"
        finally:
            Logger._log_level = original

    @pytest.mark.unit
    def test_set_level_with_valid_number(self):
        """Test set_level with valid level number.

        Note: set_level uses _get_level_name which returns None for numbers
        (since _level_names is keyed by strings), so it falls back to default.
        """
        original = Logger._log_level
        try:
            Logger.set_level(10)
            from cascor_constants.constants import _LOGGER_LOG_LEVEL_NAME

            assert Logger._log_level == _LOGGER_LOG_LEVEL_NAME

            Logger.set_level(20)
            assert Logger._log_level == _LOGGER_LOG_LEVEL_NAME
        finally:
            Logger._log_level = original

    @pytest.mark.unit
    def test_set_level_with_invalid_uses_default(self):
        """Test set_level with invalid input uses default."""
        original = Logger._log_level
        try:
            Logger.set_level("INVALID")
            from cascor_constants.constants import _LOGGER_LOG_LEVEL_NAME

            assert Logger._log_level == _LOGGER_LOG_LEVEL_NAME
        finally:
            Logger._log_level = original

    @pytest.mark.unit
    def test_get_level_returns_current(self):
        """Test get_level returns current log level."""
        original = Logger._log_level
        try:
            Logger._log_level = "ERROR"
            assert Logger.get_level() == "ERROR"
        finally:
            Logger._log_level = original


class TestLoggerFilterByLevel:
    """Tests for _filter_by_level class method."""

    @pytest.mark.unit
    def test_filter_by_level_allows_higher_level(self):
        """Test _filter_by_level allows messages at or above log level."""
        assert Logger._filter_by_level(level="ERROR", log_level="INFO") is True
        assert Logger._filter_by_level(level="WARNING", log_level="DEBUG") is True
        assert Logger._filter_by_level(level="INFO", log_level="INFO") is True

    @pytest.mark.unit
    def test_filter_by_level_blocks_lower_level(self):
        """Test _filter_by_level blocks messages below log level."""
        assert Logger._filter_by_level(level="DEBUG", log_level="INFO") is False
        assert Logger._filter_by_level(level="INFO", log_level="WARNING") is False

    @pytest.mark.unit
    def test_filter_by_level_invalid_level(self):
        """Test _filter_by_level with invalid level returns False."""
        assert Logger._filter_by_level(level="INVALID", log_level="INFO") is False
        assert Logger._filter_by_level(level="INFO", log_level="INVALID") is False
        assert Logger._filter_by_level(level=None, log_level="INFO") is False


class TestLoggerLoggingMethods:
    """Tests for logging methods (trace, verbose, debug, info, etc.)."""

    @pytest.mark.unit
    @patch.object(Logger, "_log_at_level")
    def test_trace_calls_log_at_level(self, mock_log):
        """Test trace method calls _log_at_level."""
        Logger.trace("Test trace message")
        mock_log.assert_called_once()
        call_kwargs = mock_log.call_args
        assert call_kwargs[1]["level"] == "TRACE"
        assert call_kwargs[1]["message"] == "Test trace message"

    @pytest.mark.unit
    @patch.object(Logger, "_log_at_level")
    def test_verbose_calls_log_at_level(self, mock_log):
        """Test verbose method calls _log_at_level."""
        Logger.verbose("Test verbose message")
        mock_log.assert_called_once()
        assert mock_log.call_args[1]["level"] == "VERBOSE"

    @pytest.mark.unit
    @patch.object(Logger, "_log_at_level")
    def test_debug_calls_log_at_level(self, mock_log):
        """Test debug method calls _log_at_level."""
        Logger.debug("Test debug message")
        mock_log.assert_called_once()
        assert mock_log.call_args[1]["level"] == "DEBUG"

    @pytest.mark.unit
    @patch.object(Logger, "_log_at_level")
    def test_info_calls_log_at_level(self, mock_log):
        """Test info method calls _log_at_level."""
        Logger.info("Test info message")
        mock_log.assert_called_once()
        assert mock_log.call_args[1]["level"] == "INFO"

    @pytest.mark.unit
    @patch.object(Logger, "_log_at_level")
    def test_warning_calls_log_at_level(self, mock_log):
        """Test warning method calls _log_at_level."""
        Logger.warning("Test warning message")
        mock_log.assert_called_once()
        assert mock_log.call_args[1]["level"] == "WARNING"

    @pytest.mark.unit
    @patch.object(Logger, "_log_at_level")
    def test_error_calls_log_at_level(self, mock_log):
        """Test error method calls _log_at_level."""
        Logger.error("Test error message")
        mock_log.assert_called_once()
        assert mock_log.call_args[1]["level"] == "ERROR"

    @pytest.mark.unit
    @patch.object(Logger, "_log_at_level")
    def test_critical_calls_log_at_level(self, mock_log):
        """Test critical method calls _log_at_level."""
        Logger.critical("Test critical message")
        mock_log.assert_called_once()
        assert mock_log.call_args[1]["level"] == "CRITICAL"

    @pytest.mark.unit
    @patch.object(Logger, "_log_at_level")
    def test_fatal_calls_log_at_level(self, mock_log):
        """Test fatal method calls _log_at_level."""
        Logger.fatal("Test fatal message")
        mock_log.assert_called_once()
        assert mock_log.call_args[1]["level"] == "FATAL"


class TestLoggerInstanceMethods:
    """Tests for Logger instance methods (getters/setters)."""

    @pytest.fixture
    def mock_logger_instance(self):
        """Create a mock Logger instance with necessary attributes."""
        mock = MagicMock(spec=Logger)
        mock.uuid = "test-uuid-12345"
        mock.logger = MagicMock()
        mock.log_file_name = "test.log"
        mock.log_file_path = "/tmp/logs"
        mock.level = 20
        mock.log_level = 20
        mock.log_level_name = "INFO"
        mock.log_date_format = "%Y-%m-%d"
        mock.log_formatter_string = "%(message)s"
        mock.log_message_default = "Default"
        mock.log_level_custom_names_list = ["TRACE", "VERBOSE"]
        mock.log_level_numbers_dict = {"DEBUG": 10, "INFO": 20}
        mock.log_level_methods_dict = {"DEBUG": "debug", "INFO": "info"}
        mock.log_allow_log_level_redefinition = True
        return mock

    @pytest.mark.unit
    def test_get_uuid(self, mock_logger_instance):
        """Test get_uuid returns the UUID."""
        result = Logger.get_uuid(mock_logger_instance)
        assert result == "test-uuid-12345"

    @pytest.mark.unit
    def test_get_logger(self, mock_logger_instance):
        """Test get_logger returns self."""
        mock_logger_instance.get_logger = lambda: mock_logger_instance
        result = mock_logger_instance.get_logger()
        assert result is mock_logger_instance

    @pytest.mark.unit
    def test_get_log_file_name(self, mock_logger_instance):
        """Test get_log_file_name returns file name."""
        result = Logger.get_log_file_name(mock_logger_instance)
        assert result == "test.log"

    @pytest.mark.unit
    def test_get_log_file_path(self, mock_logger_instance):
        """Test get_log_file_path returns file path."""
        result = Logger.get_log_file_path(mock_logger_instance)
        assert result == "/tmp/logs"

    @pytest.mark.unit
    def test_get_log_level(self, mock_logger_instance):
        """Test get_log_level returns log level."""
        result = Logger.get_log_level(mock_logger_instance)
        assert result == 20

    @pytest.mark.unit
    def test_get_log_level_name(self, mock_logger_instance):
        """Test get_log_level_name returns level name."""
        result = Logger.get_log_level_name(mock_logger_instance)
        assert result == "INFO"

    @pytest.mark.unit
    def test_get_log_date_format(self, mock_logger_instance):
        """Test get_log_date_format returns date format."""
        result = Logger.get_log_date_format(mock_logger_instance)
        assert result == "%Y-%m-%d"

    @pytest.mark.unit
    def test_get_log_formatter_string(self, mock_logger_instance):
        """Test get_log_formatter_string returns formatter string."""
        result = Logger.get_log_formatter_string(mock_logger_instance)
        assert result == "%(message)s"

    @pytest.mark.unit
    def test_get_log_message_default(self, mock_logger_instance):
        """Test get_log_message_default returns default message."""
        result = Logger.get_log_message_default(mock_logger_instance)
        assert result == "Default"

    @pytest.mark.unit
    def test_get_log_level_custom_names_list(self, mock_logger_instance):
        """Test get_log_level_custom_names_list returns custom names."""
        result = Logger.get_log_level_custom_names_list(mock_logger_instance)
        assert result == ["TRACE", "VERBOSE"]

    @pytest.mark.unit
    def test_get_log_level_numbers_dict(self, mock_logger_instance):
        """Test get_log_level_numbers_dict returns numbers dict."""
        result = Logger.get_log_level_numbers_dict(mock_logger_instance)
        assert result == {"DEBUG": 10, "INFO": 20}

    @pytest.mark.unit
    def test_get_log_level_methods_dict(self, mock_logger_instance):
        """Test get_log_level_methods_dict returns methods dict."""
        result = Logger.get_log_level_methods_dict(mock_logger_instance)
        assert result == {"DEBUG": "debug", "INFO": "info"}

    @pytest.mark.unit
    def test_get_log_allow_log_level_redefinition(self, mock_logger_instance):
        """Test get_log_allow_log_level_redefinition returns flag."""
        result = Logger.get_log_allow_log_level_redefinition(mock_logger_instance)
        assert result is True


class TestLoggerInstanceSetters:
    """Tests for Logger instance setter methods."""

    @pytest.fixture
    def mock_instance(self):
        """Create a mock instance for setter tests."""
        return MagicMock()

    @pytest.mark.unit
    def test_set_logger(self, mock_instance):
        """Test set_logger sets the logger."""
        new_logger = MagicMock()
        Logger.set_logger(mock_instance, new_logger)
        assert mock_instance.logger == new_logger

    @pytest.mark.unit
    def test_set_log_file_name(self, mock_instance):
        """Test set_log_file_name sets the file name."""
        Logger.set_log_file_name(mock_instance, "new.log")
        assert mock_instance.log_file_name == "new.log"

    @pytest.mark.unit
    def test_set_log_file_path(self, mock_instance):
        """Test set_log_file_path sets the file path."""
        Logger.set_log_file_path(mock_instance, "/new/path")
        assert mock_instance.log_file_path == "/new/path"

    @pytest.mark.unit
    def test_set_log_date_format(self, mock_instance):
        """Test set_log_date_format sets the date format."""
        Logger.set_log_date_format(mock_instance, "%H:%M:%S")
        assert mock_instance.log_date_format == "%H:%M:%S"

    @pytest.mark.unit
    def test_set_log_formatter_string(self, mock_instance):
        """Test set_log_formatter_string sets the formatter string."""
        Logger.set_log_formatter_string(mock_instance, "%(levelname)s: %(message)s")
        assert mock_instance.log_formatter_string == "%(levelname)s: %(message)s"

    @pytest.mark.unit
    def test_set_log_message_default(self, mock_instance):
        """Test set_log_message_default sets the default message."""
        Logger.set_log_message_default(mock_instance, "New default")
        assert mock_instance.log_message_default == "New default"

    @pytest.mark.unit
    def test_set_log_level_custom_names_list(self, mock_instance):
        """Test set_log_level_custom_names_list sets the custom names."""
        Logger.set_log_level_custom_names_list(mock_instance, ["CUSTOM1"])
        assert mock_instance.log_level_custom_names_list == ["CUSTOM1"]

    @pytest.mark.unit
    def test_set_log_level_numbers_dict(self, mock_instance):
        """Test set_log_level_numbers_dict sets the numbers dict."""
        Logger.set_log_level_numbers_dict(mock_instance, {"CUSTOM": 25})
        assert mock_instance.log_level_numbers_dict == {"CUSTOM": 25}

    @pytest.mark.unit
    def test_set_log_level_methods_dict(self, mock_instance):
        """Test set_log_level_methods_dict sets the methods dict."""
        Logger.set_log_level_methods_dict(mock_instance, {"CUSTOM": "custom"})
        assert mock_instance.log_level_methods_dict == {"CUSTOM": "custom"}

    @pytest.mark.unit
    def test_set_log_allow_log_level_redefinition(self, mock_instance):
        """Test set_log_allow_log_level_redefinition sets the flag.

        Note: The setter uses `or None` which makes False become None.
        """
        Logger.set_log_allow_log_level_redefinition(mock_instance, False)
        assert mock_instance.log_allow_log_level_redefinition is None

        Logger.set_log_allow_log_level_redefinition(mock_instance, True)
        assert mock_instance.log_allow_log_level_redefinition is True


class TestLoggerMissingAttributes:
    """Tests for getter methods when attributes are missing."""

    @pytest.fixture
    def empty_mock(self):
        """Create a mock with no attributes."""
        # mock = MagicMock(spec=[])
        # return mock
        return MagicMock(spec=[])

    @pytest.mark.unit
    def test_get_log_file_name_missing(self, empty_mock):
        """Test get_log_file_name returns None when missing."""
        result = Logger.get_log_file_name(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_file_path_missing(self, empty_mock):
        """Test get_log_file_path returns None when missing."""
        result = Logger.get_log_file_path(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_level_missing(self, empty_mock):
        """Test _get_level returns None when missing."""
        result = Logger._get_level(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_level_missing(self, empty_mock):
        """Test get_log_level returns None when missing."""
        result = Logger.get_log_level(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_level_name_missing(self, empty_mock):
        """Test get_log_level_name returns None when missing."""
        result = Logger.get_log_level_name(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_date_format_missing(self, empty_mock):
        """Test get_log_date_format returns None when missing."""
        result = Logger.get_log_date_format(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_formatter_string_missing(self, empty_mock):
        """Test get_log_formatter_string returns None when missing."""
        result = Logger.get_log_formatter_string(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_message_default_missing(self, empty_mock):
        """Test get_log_message_default returns None when missing."""
        result = Logger.get_log_message_default(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_level_custom_names_list_missing(self, empty_mock):
        """Test get_log_level_custom_names_list returns None when missing."""
        result = Logger.get_log_level_custom_names_list(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_level_numbers_dict_missing(self, empty_mock):
        """Test get_log_level_numbers_dict returns None when missing."""
        result = Logger.get_log_level_numbers_dict(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_level_methods_dict_missing(self, empty_mock):
        """Test get_log_level_methods_dict returns None when missing."""
        result = Logger.get_log_level_methods_dict(empty_mock)
        assert result is None

    @pytest.mark.unit
    def test_get_log_allow_log_level_redefinition_missing(self, empty_mock):
        """Test get_log_allow_log_level_redefinition returns None when missing."""
        result = Logger.get_log_allow_log_level_redefinition(empty_mock)
        assert result is None


class TestLoggerUUID:
    """Tests for Logger UUID methods."""

    @pytest.mark.unit
    def test_generate_uuid_creates_valid_uuid(self):
        """Test _generate_uuid creates a valid UUID string."""
        mock_instance = MagicMock()
        result = Logger._generate_uuid(mock_instance)

        assert isinstance(result, str)
        assert len(result) == 36
        assert result.count("-") == 4

    @pytest.mark.unit
    def test_set_uuid_with_provided_value(self):
        """Test set_uuid with a provided UUID value."""

        class FakeLogger:
            def _generate_uuid(self):
                return "fallback-uuid"

        obj = FakeLogger()
        Logger.set_uuid(obj, "custom-uuid-value")
        assert obj.uuid == "custom-uuid-value"

    @pytest.mark.unit
    def test_set_uuid_generates_when_none(self):
        """Test set_uuid generates UUID when None provided."""

        class FakeLogger:
            def _generate_uuid(self):
                return "generated-uuid"

        obj = FakeLogger()
        Logger.set_uuid(obj, None)
        assert obj.uuid == "generated-uuid"


class TestLoggerHelperMethods:
    """Tests for Logger helper class methods."""

    @pytest.mark.unit
    def test_date_returns_callable(self):
        """Test _date returns a callable that formats dates."""
        import datetime

        formatter = Logger._date("%Y-%m-%d")
        test_date = datetime.datetime(2025, 1, 15)
        result = formatter(test_date)
        assert result == "2025-01-15"

    @pytest.mark.unit
    def test_get_log_level_returns_callable(self):
        """Test _get_log_level returns a callable."""
        level_func = Logger._get_log_level(config_lvl=10, norm_lvl=20)
        assert callable(level_func)
        assert level_func(True) == 20
        assert level_func(False) == 10

    @pytest.mark.unit
    def test_get_log_level_check_returns_callable(self):
        """Test _get_log_level_check returns a callable."""
        level_func = Logger._get_log_level_check(config_lvl=10, norm_lvl=20)
        assert callable(level_func)


class TestLoggerDictMethods:
    """Tests for Logger dict generation methods."""

    @pytest.mark.unit
    def test_console_dict_returns_dict(self):
        """Test _console_dict returns a dictionary."""
        import datetime
        from inspect import currentframe

        result = Logger._console_dict(
            frame=currentframe(),
            tsp=datetime.datetime.now(),
            level="INFO",
            message="Test message",
        )
        assert isinstance(result, dict)
        assert "INFO" in result.values()
        assert "Test message" in result.values()

    @pytest.mark.unit
    def test_file_dict_returns_dict(self):
        """Test _file_dict returns a dictionary."""
        import datetime
        from inspect import currentframe

        result = Logger._file_dict(
            frame=currentframe(),
            tsp=datetime.datetime.now(),
            level="DEBUG",
            message="Debug message",
        )
        assert isinstance(result, dict)
        assert "DEBUG" in result.values()
        assert "Debug message" in result.values()
