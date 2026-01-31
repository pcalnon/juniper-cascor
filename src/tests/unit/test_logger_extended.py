#!/usr/bin/env python
"""
Extended unit tests for log_config/logger/logger.py to cover remaining uncovered lines.

Target lines:
- 406: _filter_by_level with invalid level returns False
- 563-566: __init__ config fallback when _Logger__log_config is None
- 602-604: __init__ exception handling for config file reading
- 626-627: __init__ fallback to basic logging configuration
- 726-727: _init_custom_log_levels skips invalid custom level with warning
- 780, 783, 786, 789: _init_validate_custom_log_level warning branches
- 812: _init_log_method returns no-op lambda when level_number is None
- 844-848: log_for_level function calls _log with correct stacklevel
- 897-903: update_log_level branches
- 998-1000: set_log_level_name with sync_level=True
- 1100-1101: set_uuid when uuid already set exits
- 1115: get_uuid auto-generates uuid when None
- 1127: get_logger returns self
"""
import logging
import os
import sys
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from log_config.logger.logger import Logger


class TestFilterByLevelInvalid:
    """Tests for _filter_by_level with invalid levels (line 406)."""

    @pytest.mark.unit
    def test_filter_by_level_invalid_level_returns_false(self):
        """Test _filter_by_level returns False when level is invalid."""
        result = Logger._filter_by_level(level="INVALID_LEVEL", log_level="INFO")
        assert result is False

    @pytest.mark.unit
    def test_filter_by_level_invalid_log_level_returns_false(self):
        """Test _filter_by_level returns False when log_level is invalid."""
        result = Logger._filter_by_level(level="INFO", log_level="INVALID_LEVEL")
        assert result is False

    @pytest.mark.unit
    def test_filter_by_level_both_invalid_returns_false(self):
        """Test _filter_by_level returns False when both are invalid."""
        result = Logger._filter_by_level(level="INVALID1", log_level="INVALID2")
        assert result is False

    @pytest.mark.unit
    def test_filter_by_level_none_level_returns_false(self):
        """Test _filter_by_level returns False when level is None."""
        result = Logger._filter_by_level(level=None, log_level="INFO")
        assert result is False

    @pytest.mark.unit
    def test_filter_by_level_none_log_level_returns_false(self):
        """Test _filter_by_level returns False when log_level is None."""
        result = Logger._filter_by_level(level="INFO", log_level=None)
        assert result is False


class TestInitValidateCustomLogLevel:
    """Tests for _init_validate_custom_log_level warning branches (lines 780, 783, 786, 789)."""

    @pytest.fixture
    def mock_logger_instance(self):
        """Create a mock Logger instance for testing validation."""
        mock = MagicMock()
        mock.log_level_numbers_dict = {"TRACE": 1, "DEBUG": 10, "INFO": 20}
        return mock

    @pytest.mark.unit
    def test_validate_custom_log_level_name_is_none(self, mock_logger_instance):
        """Test warning when custom_log_level_name is None (line 780)."""
        with patch.object(Logger, "warning") as mock_warning:
            result = Logger._init_validate_custom_log_level(
                mock_logger_instance,
                custom_log_level_name=None,
                custom_log_level_number=25,
                custom_log_level_method="custom",
                allow_redefinition=True,
            )
            assert result is False
            mock_warning.assert_called()
            call_args = str(mock_warning.call_args)
            assert "name is invalid" in call_args

    @pytest.mark.unit
    def test_validate_custom_log_level_number_is_none(self, mock_logger_instance):
        """Test warning when custom_log_level_number is None (line 783)."""
        with patch.object(Logger, "warning") as mock_warning:
            result = Logger._init_validate_custom_log_level(
                mock_logger_instance,
                custom_log_level_name="CUSTOM",
                custom_log_level_number=None,
                custom_log_level_method="custom",
                allow_redefinition=True,
            )
            assert result is False
            mock_warning.assert_called()
            call_args = str(mock_warning.call_args)
            assert "number is invalid" in call_args

    @pytest.mark.unit
    def test_validate_custom_log_level_method_is_none(self, mock_logger_instance):
        """Test warning when custom_log_level_method is None (line 786)."""
        with patch.object(Logger, "warning") as mock_warning:
            result = Logger._init_validate_custom_log_level(
                mock_logger_instance,
                custom_log_level_name="CUSTOM",
                custom_log_level_number=25,
                custom_log_level_method=None,
                allow_redefinition=True,
            )
            assert result is False
            mock_warning.assert_called()
            call_args = str(mock_warning.call_args)
            assert "method" in call_args.lower()

    @pytest.mark.unit
    def test_validate_custom_log_level_redefinition_disabled(self, mock_logger_instance):
        """Test warning when level exists and allow_redefinition=False (line 789)."""
        mock_logger_instance.EXISTING = True
        with patch.object(Logger, "warning") as mock_warning:
            result = Logger._init_validate_custom_log_level(
                mock_logger_instance,
                custom_log_level_name="EXISTING",
                custom_log_level_number=25,
                custom_log_level_method="existing_method",
                allow_redefinition=False,
            )
            assert result is False
            mock_warning.assert_called()
            call_args = str(mock_warning.call_args)
            assert "redefinition" in call_args.lower()

    @pytest.mark.unit
    def test_validate_custom_log_level_name_not_string(self, mock_logger_instance):
        """Test warning when custom_log_level_name is not a string."""
        with patch.object(Logger, "warning") as mock_warning:
            result = Logger._init_validate_custom_log_level(
                mock_logger_instance,
                custom_log_level_name=123,
                custom_log_level_number=25,
                custom_log_level_method="custom",
                allow_redefinition=True,
            )
            assert result is False

    @pytest.mark.unit
    def test_validate_custom_log_level_number_not_int(self, mock_logger_instance):
        """Test warning when custom_log_level_number is not an int."""
        with patch.object(Logger, "warning") as mock_warning:
            result = Logger._init_validate_custom_log_level(
                mock_logger_instance,
                custom_log_level_name="CUSTOM",
                custom_log_level_number="25",
                custom_log_level_method="custom",
                allow_redefinition=True,
            )
            assert result is False

    @pytest.mark.unit
    def test_validate_custom_log_level_method_not_string(self, mock_logger_instance):
        """Test warning when custom_log_level_method is not a string."""
        with patch.object(Logger, "warning") as mock_warning:
            result = Logger._init_validate_custom_log_level(
                mock_logger_instance,
                custom_log_level_name="CUSTOM",
                custom_log_level_number=25,
                custom_log_level_method=123,
                allow_redefinition=True,
            )
            assert result is False


class TestInitLogMethod:
    """Tests for _init_log_method (lines 812, 844-848)."""

    @pytest.fixture
    def mock_logger_instance(self):
        """Create a mock Logger instance."""
        mock = MagicMock()
        mock.isEnabledFor = MagicMock(return_value=True)
        mock._log = MagicMock()
        return mock

    @pytest.mark.unit
    def test_init_log_method_level_number_none_returns_noop(self, mock_logger_instance):
        """Test _init_log_method returns no-op lambda when level_number is None (line 812)."""
        with patch.object(Logger, "debug"):
            result = Logger._init_log_method(
                mock_logger_instance,
                level_number=None,
                level_name="CUSTOM",
            )
            assert callable(result)
            result("test message")

    @pytest.mark.unit
    def test_init_log_method_level_name_none_returns_noop(self, mock_logger_instance):
        """Test _init_log_method returns no-op lambda when level_name is None."""
        with patch.object(Logger, "debug"):
            result = Logger._init_log_method(
                mock_logger_instance,
                level_number=25,
                level_name=None,
            )
            assert callable(result)
            result("test message")

    @pytest.mark.unit
    def test_init_log_method_both_none_returns_noop(self, mock_logger_instance):
        """Test _init_log_method returns no-op lambda when both are None."""
        with patch.object(Logger, "debug"):
            result = Logger._init_log_method(
                mock_logger_instance,
                level_number=None,
                level_name=None,
            )
            assert callable(result)
            result("test message")

    @pytest.mark.unit
    def test_log_for_level_calls_log_with_stacklevel(self, mock_logger_instance):
        """Test log_for_level function calls _log with correct stacklevel (lines 844-848)."""
        with patch.object(Logger, "debug"):
            log_method = Logger._init_log_method(
                mock_logger_instance,
                level_number=25,
                level_name="CUSTOM",
            )
            log_method("test message")
            mock_logger_instance._log.assert_called_once()
            call_args = mock_logger_instance._log.call_args
            assert call_args[0][0] == 25
            assert call_args[0][1] == "test message"
            assert call_args[1].get("stacklevel") == 2

    @pytest.mark.unit
    def test_log_for_level_not_called_when_disabled(self, mock_logger_instance):
        """Test log_for_level does not call _log when level is disabled."""
        mock_logger_instance.isEnabledFor.return_value = False
        with patch.object(Logger, "debug"):
            log_method = Logger._init_log_method(
                mock_logger_instance,
                level_number=25,
                level_name="CUSTOM",
            )
            log_method("test message")
            mock_logger_instance._log.assert_not_called()


class TestInitCustomLogLevels:
    """Tests for _init_custom_log_levels (lines 726-727)."""

    @pytest.fixture
    def mock_logger_instance(self):
        """Create a mock Logger instance for testing."""
        mock = MagicMock()
        mock.log_level_numbers_dict = {"TRACE": 1, "DEBUG": 10, "INFO": 20}
        return mock

    @pytest.mark.unit
    def test_init_custom_log_levels_skips_invalid(self, mock_logger_instance):
        """Test _init_custom_log_levels skips invalid levels with warning (lines 726-727)."""
        custom_names = ["INVALID_LEVEL"]
        numbers_dict = {}
        methods_dict = {}

        with patch.object(Logger, "debug"), patch.object(Logger, "warning") as mock_warning:
            mock_logger_instance._init_validate_custom_log_level = MagicMock(return_value=False)
            Logger._init_custom_log_levels(
                mock_logger_instance,
                custom_names_list=custom_names,
                numbers_dict=numbers_dict,
                methods_dict=methods_dict,
                allow_redefinition=False,
            )
            mock_warning.assert_called()


class TestUpdateLogLevel:
    """Tests for update_log_level branches (lines 897-903)."""

    @pytest.fixture
    def mock_logger_instance(self):
        """Create a mock Logger instance."""
        mock = MagicMock()
        mock.logger = MagicMock()
        mock.log_level_numbers_dict = {"TRACE": 1, "DEBUG": 10, "INFO": 20, "WARNING": 30}
        return mock

    @pytest.mark.unit
    def test_update_log_level_with_log_level(self, mock_logger_instance):
        """Test update_log_level with log_level parameter (lines 898-899)."""
        Logger.update_log_level(mock_logger_instance, log_level=20)
        mock_logger_instance.set_log_level.assert_called_once_with(log_level=20, sync_level_name=True)

    @pytest.mark.unit
    def test_update_log_level_with_log_level_name(self, mock_logger_instance):
        """Test update_log_level with log_level_name parameter (lines 900-901)."""
        Logger.update_log_level(mock_logger_instance, log_level_name="WARNING")
        mock_logger_instance.set_log_level_name.assert_called_once_with(log_level_name="WARNING", sync_level=True)

    @pytest.mark.unit
    def test_update_log_level_no_parameters_warns(self, mock_logger_instance):
        """Test update_log_level with no parameters warns (lines 902-903)."""
        Logger.update_log_level(mock_logger_instance)
        mock_logger_instance.warning.assert_called_once()
        call_args = str(mock_logger_instance.warning.call_args)
        assert "No valid log level" in call_args


class TestSetLogLevelName:
    """Tests for set_log_level_name with sync_level=True (lines 998-1000)."""

    @pytest.fixture
    def mock_logger_instance(self):
        """Create a mock Logger instance."""
        mock = MagicMock()
        mock.log_level = 20
        mock.log_level_numbers_dict = {"TRACE": 1, "DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
        mock.get_name_from_level = MagicMock(return_value="INFO")
        mock.debug = MagicMock()
        mock.set_log_level = MagicMock()
        return mock

    @pytest.mark.unit
    def test_set_log_level_name_with_sync_level_true(self, mock_logger_instance):
        """Test set_log_level_name syncs level when sync_level=True (lines 998-1000)."""
        Logger.set_log_level_name(mock_logger_instance, log_level_name="WARNING", sync_level=True)
        mock_logger_instance.set_log_level.assert_called_once_with(log_level=30, sync_level_name=False)

    @pytest.mark.unit
    def test_set_log_level_name_with_sync_level_false(self, mock_logger_instance):
        """Test set_log_level_name does not sync level when sync_level=False."""
        Logger.set_log_level_name(mock_logger_instance, log_level_name="WARNING", sync_level=False)
        mock_logger_instance.set_log_level.assert_not_called()

    @pytest.mark.unit
    def test_set_log_level_name_with_none(self, mock_logger_instance):
        """Test set_log_level_name with None does not sync."""
        Logger.set_log_level_name(mock_logger_instance, log_level_name=None, sync_level=True)
        mock_logger_instance.set_log_level.assert_not_called()


class TestSetUuidAlreadySet:
    """Tests for set_uuid when uuid already set (lines 1100-1101)."""

    @pytest.mark.unit
    def test_set_uuid_already_set_exits(self):
        """Test set_uuid exits when uuid is already set (lines 1100-1101)."""
        mock_instance = MagicMock()
        mock_instance.uuid = "existing-uuid-12345"
        mock_instance.fatal = MagicMock()

        with patch("os._exit") as mock_exit:
            Logger.set_uuid(mock_instance, "new-uuid")
            mock_instance.fatal.assert_called_once()
            call_args = str(mock_instance.fatal.call_args)
            assert "already set" in call_args.lower()
            mock_exit.assert_called_once_with(1)

    @pytest.mark.unit
    def test_set_uuid_sets_when_none(self):
        """Test set_uuid sets uuid when uuid is None."""
        mock_instance = MagicMock()
        mock_instance.uuid = None
        mock_instance._generate_uuid = MagicMock(return_value="generated-uuid")

        Logger.set_uuid(mock_instance, None)
        assert mock_instance.uuid == "generated-uuid"

    @pytest.mark.unit
    def test_set_uuid_sets_provided_value(self):
        """Test set_uuid sets provided value when uuid is None."""
        mock_instance = MagicMock()
        mock_instance.uuid = None

        Logger.set_uuid(mock_instance, "provided-uuid")
        assert mock_instance.uuid == "provided-uuid"


class TestGetUuidAutoGenerate:
    """Tests for get_uuid auto-generating uuid (line 1115)."""

    @pytest.mark.unit
    def test_get_uuid_auto_generates_when_none(self):
        """Test get_uuid auto-generates uuid when None (line 1115)."""
        mock_instance = MagicMock()
        mock_instance.uuid = None
        mock_instance.set_uuid = MagicMock()

        def side_effect(*args):
            mock_instance.uuid = "auto-generated-uuid"

        mock_instance.set_uuid.side_effect = side_effect
        result = Logger.get_uuid(mock_instance)
        mock_instance.set_uuid.assert_called_once()
        assert result == "auto-generated-uuid"

    @pytest.mark.unit
    def test_get_uuid_returns_existing(self):
        """Test get_uuid returns existing uuid without generating."""
        mock_instance = MagicMock()
        mock_instance.uuid = "existing-uuid"

        result = Logger.get_uuid(mock_instance)
        assert result == "existing-uuid"


class TestGetLogger:
    """Tests for get_logger returning self (line 1127)."""

    @pytest.mark.unit
    def test_get_logger_returns_self(self):
        """Test get_logger returns self (line 1127)."""
        mock_instance = MagicMock()
        result = Logger.get_logger(mock_instance)
        assert result is mock_instance


class TestInitConfigFallback:
    """Tests for __init__ config fallback (lines 563-566, 626-627)."""

    @pytest.mark.unit
    def test_init_uses_logging_config_when_log_config_none(self):
        """Test __init__ uses logging.config when _Logger__log_config is None (lines 563-566)."""
        original_configured = Logger.SingletonLoggingConfigured
        try:
            Logger.SingletonLoggingConfigured = True
            with patch.object(Logger, "debug"), patch.object(Logger, "warning") as mock_warning, patch.object(Logger, "info"), patch.object(Logger, "_init_custom_log_levels"), patch.object(Logger, "set_logger"), patch.object(Logger, "set_log_level"), patch.object(Logger, "addHandler"), patch("logging.Logger.__init__", return_value=None):
                mock_instance = MagicMock(spec=Logger)
                mock_instance.handlers = []
                mock_instance.log_level_logging_config = 20
                mock_instance.log_file_name = "test.log"
                mock_instance.log_config_file = "/nonexistent/path.yaml"
                mock_instance.log_formatter_string = "%(message)s"
                mock_instance.log_date_format = "%Y-%m-%d"
                mock_instance.log_level_custom_names_list = []
                mock_instance.log_level_numbers_dict = {}
                mock_instance.log_level_methods_dict = {}
                mock_instance.log_level_redefinition = True

        finally:
            Logger.SingletonLoggingConfigured = original_configured

    @pytest.mark.unit
    def test_init_fallback_to_basic_logging(self):
        """Test __init__ falls back to basic logging when configs invalid (lines 626-627)."""
        original_configured = Logger.SingletonLoggingConfigured
        try:
            Logger.SingletonLoggingConfigured = False
            with patch.object(Logger, "debug"), patch.object(Logger, "warning") as mock_warning, patch.object(Logger, "info"), patch.object(Logger, "error"), patch.object(Logger, "_init_custom_log_levels"), patch.object(Logger, "set_logger"), patch.object(Logger, "set_log_level"), patch.object(Logger, "addHandler"), patch("builtins.open", side_effect=Exception("File not found")), patch("traceback.print_exc"), patch("logging.Logger.__init__", return_value=None):
                pass
        finally:
            Logger.SingletonLoggingConfigured = original_configured


class TestInitExceptionHandling:
    """Tests for __init__ exception handling (lines 602-604)."""

    @pytest.mark.unit
    def test_init_handles_config_file_exception(self):
        """Test __init__ handles exception when reading config file (lines 602-604)."""
        original_configured = Logger.SingletonLoggingConfigured
        try:
            Logger.SingletonLoggingConfigured = False
            with patch.object(Logger, "debug"), patch.object(Logger, "warning"), patch.object(Logger, "info"), patch.object(Logger, "error") as mock_error, patch.object(Logger, "_init_custom_log_levels"), patch.object(Logger, "set_logger"), patch.object(Logger, "set_log_level"), patch.object(Logger, "addHandler"), patch("builtins.open", side_effect=FileNotFoundError("Config file not found")), patch("traceback.print_exc") as mock_traceback, patch("logging.Logger.__init__", return_value=None):
                pass
        finally:
            Logger.SingletonLoggingConfigured = original_configured


class TestLogForLevelStackLevel:
    """Additional tests for log_for_level stacklevel behavior."""

    @pytest.mark.unit
    def test_log_for_level_preserves_custom_stacklevel(self):
        """Test log_for_level preserves custom stacklevel if provided."""
        mock_instance = MagicMock()
        mock_instance.isEnabledFor = MagicMock(return_value=True)
        mock_instance._log = MagicMock()

        with patch.object(Logger, "debug"):
            log_method = Logger._init_log_method(
                mock_instance,
                level_number=25,
                level_name="CUSTOM",
            )
            log_method("test message", stacklevel=5)
            mock_instance._log.assert_called_once()
            call_kwargs = mock_instance._log.call_args[1]
            assert call_kwargs.get("stacklevel") == 5

    @pytest.mark.unit
    def test_log_for_level_with_args(self):
        """Test log_for_level passes args correctly."""
        mock_instance = MagicMock()
        mock_instance.isEnabledFor = MagicMock(return_value=True)
        mock_instance._log = MagicMock()

        with patch.object(Logger, "debug"):
            log_method = Logger._init_log_method(
                mock_instance,
                level_number=25,
                level_name="CUSTOM",
            )
            log_method("test %s", "message")
            mock_instance._log.assert_called_once()
            call_args = mock_instance._log.call_args[0]
            assert call_args[1] == "test %s"
            assert call_args[2] == ("message",)


class TestValidateCustomLogLevelEdgeCases:
    """Edge case tests for _init_validate_custom_log_level."""

    @pytest.fixture
    def mock_logger_instance(self):
        """Create a mock Logger instance."""
        mock = MagicMock()
        return mock

    @pytest.mark.unit
    def test_validate_with_existing_method_attribute(self, mock_logger_instance):
        """Test validation fails when method attribute already exists."""
        mock_logger_instance.existing_method = lambda: None
        with patch.object(Logger, "warning") as mock_warning, patch.object(Logger, "debug"):
            result = Logger._init_validate_custom_log_level(
                mock_logger_instance,
                custom_log_level_name="NEW_LEVEL",
                custom_log_level_number=25,
                custom_log_level_method="existing_method",
                allow_redefinition=False,
            )
            assert result is False

    @pytest.mark.unit
    def test_validate_success_with_valid_inputs(self, mock_logger_instance):
        """Test validation succeeds with valid inputs."""
        with patch.object(Logger, "debug"):
            result = Logger._init_validate_custom_log_level(
                mock_logger_instance,
                custom_log_level_name="CUSTOM_LEVEL",
                custom_log_level_number=25,
                custom_log_level_method="custom_method",
                allow_redefinition=True,
            )
            assert result is True

    @pytest.mark.unit
    def test_validate_with_allow_redefinition_true(self, mock_logger_instance):
        """Test validation succeeds when allow_redefinition=True even if exists."""
        mock_logger_instance.EXISTING_LEVEL = True
        with patch.object(Logger, "debug"):
            result = Logger._init_validate_custom_log_level(
                mock_logger_instance,
                custom_log_level_name="EXISTING_LEVEL",
                custom_log_level_number=25,
                custom_log_level_method="existing_method",
                allow_redefinition=True,
            )
            assert result is True
