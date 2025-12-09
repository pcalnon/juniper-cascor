#!/usr/bin/env python
####################################################################################################################################################################################################
# Project:       Juniper
# Application:   Dynamic Neural Network
# File Name:     logger.py
# Author:        Paul Calnon
# Version:       0.7.3
#
# Date:          2024-04-02
# Last Modified: 2025-09-05
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
#
# Description: 
#     This module provides a custom logger with custom log levels and a logging configuration.
#
#####################################################################################################################################################################################################
# Notes:
#     - This file contains a custom Logger class that extends the built-in logging.Logger class to add custom log levels.
#     - The Logger class allows for dynamic creation of custom log levels based on provided names, numbers, and methods.
#     - The Logger class also supports loading a logging configuration from a YAML file.
#
#
#     - Global flag to track if logging configuration has been applied
#     - global _LOGGER_SINGLETON_LOGGING_CONFIGURED
#     - _LOGGER_SINGLETON_LOGGING_CONFIGURED = _LOGGER_LOG_LOGGING_CONFIGURED_DEFAULT
#
#
#     - filename='<stdin>', lineno=4, function='__init__', code_context=None, index=None,
#     - frameinfo = getframeinfo(currentframe())
#     - print(f"Cascor: Logger: File: {frameinfo.filename}, Line: {frameinfo.lineno}")
#     - "[%(filename)s:%(lineno)d] (%(asctime)s) [%(levelname)s] %(message)s"
#
#
#####################################################################################################################################################################################################
# References:
#     References to setting custom log levels in Python's logging module
#     https://docs.python.org/3/library/logging.html#logging.addLevelName  
#     https://docs.python.org/3/library/logging.html#logging.Logger.setLevel
#     https://stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility/35804945
#
#####################################################################################################################################################################################################
# TODO :
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#
#####################################################################################################################################################################################################
import datetime
import logging
import logging.config
import os
import traceback
import uuid
import yaml

# from inspect import currentframe, getframeinfo, getouterframes
from inspect import currentframe, getouterframes

from logging import (
    addLevelName,
)
from constants.constants import (
    _LOGGER_LOG_DATE_FORMAT,
    _LOGGER_LOG_CONFIG_FILE_NAME,
    _LOGGER_LOG_CONFIG_FILE_PATH,
    _LOGGER_LOG_FILE_NAME,
    _LOGGER_LOG_FILE_PATH,
    _LOGGER_LOG_FORMATTER_STRING,
    # _LOGGER_LOG_LEVEL,
    _LOGGER_LOG_LEVEL_CUSTOM_NAMES_LIST,
    _LOGGER_LOG_LEVEL_LOGGING_CONFIG,
    _LOGGER_LOG_LEVEL_NAME,
    _LOGGER_LOG_LEVEL_NUMBERS_DICT,
    _LOGGER_LOG_LEVEL_NUMBERS_LIST,
    _LOGGER_LOG_LEVEL_NAMES_LIST,
    _LOGGER_LOG_LEVEL_METHODS_DICT,
    _LOGGER_LOG_LEVEL_METHODS_LIST,
    _LOGGER_LOG_LEVEL_REDEFINITION,
    _LOGGER_LOG_LOGGING_CONFIGURED_DEFAULT,
    _LOGGER_LOG_MESSAGE_DEFAULT,
    #_LOGGER_NAME,
    _LOGGER_PREFIX_FIELD_NAME_FILENAME,
    _LOGGER_PREFIX_FIELD_NAME_LINE_NUMBER,
    # The above code seems to be defining a constant variable named
    # `_LOGGER_PREFIX_FIELD_NAME_FUNCTION_NAME` in Python.
    _LOGGER_PREFIX_FIELD_NAME_FUNCTION_NAME,
    _LOGGER_PREFIX_FIELD_NAME_ASCTIME,
    _LOGGER_CONTENT_FIELD_NAME_LEVELNAME,
    _LOGGER_CONTENT_FIELD_NAME_MESSAGE,
    # _LOGGER_PREFIX_FRAME_FUNCTION_NAME,
    # _LOGGER_PREFIX_FILENAME,
    # _LOGGER_PREFIX_LINE_NUMBER,
    # _LOGGER_PREFIX_FUNCTION_NAME,
    # _LOGGER_PREFIX_ASCTIME,
    # _LOGGER_CONTENT_LEVELNAME,
    # _LOGGER_CONTENT_MESSAGE,
    # _LOGGER_DATA_FIELD_NAMES,
    # _LOGGER_FORMAT_FIELD_NAMES,
    _LOGGER_PREFIX_FRAME_FILE_NAME,
    _LOGGER_PREFIX_FRAME_LINE_NAME,
    _LOGGER_PREFIX_FRAME_FUNC_NAME,
    _LOGGER_PREFIX_FIELD_NAMES_CONSOLE,
    _LOGGER_CONTENT_FIELD_NAMES_CONSOLE,
    # _LOGGER_PREFIX_FIELDS_CONSOLE,
    # _LOGGER_CONTENT_FIELDS_CONSOLE,
    _LOGGER_PREFIX_FIELD_NAMES_FILE,
    _LOGGER_CONTENT_FIELD_NAMES_FILE,
    # _LOGGER_PREFIX_FIELDS_FILE,
    # _LOGGER_CONTENT_FIELDS_FILE,
    # _LOGGER_PREFIX_FORMAT_CONSOLE,
    # _LOGGER_PREFIX_FORMAT_FILE,
    # _LOGGER_CONTENT_FORMAT_CONSOLE,
    # _LOGGER_CONTENT_FORMAT_FILE,
    _LOGGER_DESTINATION_NAME_CONSOLE,
    _LOGGER_DESTINATION_NAME_FILE,
    _LOGGER_PREFIX_NAME,
    _LOGGER_CONTENT_NAME,
    _LOGGER_LOG_FORMATTER_STRING_FILE,
    _LOGGER_LOG_FORMATTER_STRING_FILE_PREFIX,
    _LOGGER_LOG_FORMATTER_STRING_FILE_CONTENT,
    _LOGGER_LOG_FORMATTER_STRING_CONSOLE,
    _LOGGER_LOG_FORMATTER_STRING_CONSOLE_PREFIX,
    _LOGGER_LOG_FORMATTER_STRING_CONSOLE_CONTENT,
)


########################################################################################################################################
# Define the Logger class that extends the logging.Logger class (to add custom log levels)
########################################################################################################################################
class Logger(logging.getLoggerClass()):


    ####################################################################################################################################
    # Define Singleton to determine if Logging has already been Configured
    SingletonLoggingConfigured = _LOGGER_LOG_LOGGING_CONFIGURED_DEFAULT

    @classmethod
    def is_configured(cls):
        return cls.SingletonLoggingConfigured

    @classmethod
    def set_configured(cls):
        cls.SingletonLoggingConfigured = True


    ####################################################################################################################################
    # Logger Class Attributes
    _frm = currentframe
    _tsp = datetime.datetime.now

    _date_format=_LOGGER_LOG_DATE_FORMAT

    _console_name = _LOGGER_DESTINATION_NAME_CONSOLE
    _file_name = _LOGGER_DESTINATION_NAME_FILE

    _logging_file_path = _LOGGER_LOG_FILE_PATH
    _logging_file_name = _LOGGER_LOG_FILE_NAME
    _logging_file = os.path.join(_logging_file_path, _logging_file_name)

    _prefix_name = _LOGGER_PREFIX_NAME
    _content_name = _LOGGER_CONTENT_NAME

    _formatter_file_prefix = _LOGGER_LOG_FORMATTER_STRING_FILE_PREFIX
    _formatter_file_content = _LOGGER_LOG_FORMATTER_STRING_FILE_CONTENT
    _formatter_console_prefix = _LOGGER_LOG_FORMATTER_STRING_CONSOLE_PREFIX
    _formatter_console_content = _LOGGER_LOG_FORMATTER_STRING_CONSOLE_CONTENT

    _fieldnames_file_prefix = _LOGGER_PREFIX_FIELD_NAMES_FILE
    _fieldnames_file_content = _LOGGER_CONTENT_FIELD_NAMES_FILE
    _fieldnames_console_prefix = _LOGGER_PREFIX_FIELD_NAMES_CONSOLE
    _fieldnames_console_content = _LOGGER_CONTENT_FIELD_NAMES_CONSOLE

    _file_name = _LOGGER_PREFIX_FIELD_NAME_FILENAME
    _line_number = _LOGGER_PREFIX_FIELD_NAME_LINE_NUMBER
    _function_name = _LOGGER_PREFIX_FIELD_NAME_FUNCTION_NAME
    _asctime = _LOGGER_PREFIX_FIELD_NAME_ASCTIME
    _level_name = _LOGGER_CONTENT_FIELD_NAME_LEVELNAME
    _message_name = _LOGGER_CONTENT_FIELD_NAME_MESSAGE

    _level_logger_config = _LOGGER_LOG_LEVEL_LOGGING_CONFIG
    # _level_logger_name = _LOGGER_LOG_LEVEL_NAME
    _level_logger_name = "INFO"

    _field_names_file = [_file_name, _function_name, _line_number, _asctime, _level_name, _message_name]
    _field_names_console = [_file_name, _line_number, _asctime, _level_name, _message_name]

    _formatter_string_file = _LOGGER_LOG_FORMATTER_STRING_FILE
    _formatter_string_console = _LOGGER_LOG_FORMATTER_STRING_CONSOLE

    _frame_file = _LOGGER_PREFIX_FRAME_FILE_NAME
    _frame_line = _LOGGER_PREFIX_FRAME_LINE_NAME
    _frame_func = _LOGGER_PREFIX_FRAME_FUNC_NAME

    _log_level = _LOGGER_LOG_LEVEL_NAME
    # _check_log_level = None

    # _level_trace = logging.getLevelName(1)
    _level_trace = "TRACE"
    # _level_verbose = logging.getLevelName(5)
    _level_verbose = "VERBOSE"
    _level_debug = logging.getLevelName(logging.DEBUG)
    _level_info = logging.getLevelName(logging.INFO)
    _level_warning = logging.getLevelName(logging.WARNING)
    _level_error = logging.getLevelName(logging.ERROR)
    _level_critical = logging.getLevelName(logging.CRITICAL)
    # _level_fatal = logging.getLevelName(60)
    _level_fatal = "FATAL"

    _level_names = {
        "TRACE": _level_trace,
        "VERBOSE": _level_verbose,
        "DEBUG": _level_debug,
        "INFO": _level_info,
        "WARNING": _level_warning,
        "ERROR": _level_error,
        "CRITICAL": _level_critical,
        "FATAL": _level_fatal,
    }
    _level_numbers = {
        "TRACE": 1,
        "VERBOSE": 5,
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50,
        "FATAL": 60,
    }


    ####################################################################################################################################
    # Logger Class Methods
    @classmethod
    def _date(cls, format=None):
        return lambda d: d.strftime(format)


    ####################################################################################################################################
    @classmethod
    def _frame_info(cls, frame=None,):
        return lambda name: getattr(getouterframes(frame)[1], name)


    ####################################################################################################################################
    @classmethod
    def _console_dict(cls, frame=None, tsp=None, level=None, message=None):
        _get_frame_info=cls._frame_info(frame=frame)
        _time_info = cls._date(format=cls._date_format)
        return dict(zip(cls._field_names_console, [os.path.basename(_get_frame_info(cls._frame_file)), _get_frame_info(cls._frame_line), _time_info(tsp), level, message], strict=False))

    @classmethod
    def _file_dict(cls, frame=None, tsp=None, level=None, message=None):
        _get_frame_info=cls._frame_info(frame=frame)
        _time_info = cls._date(format=cls._date_format)
        return dict(zip(cls._field_names_file, [os.path.basename(_get_frame_info(cls._frame_file)), _get_frame_info(cls._frame_func), _get_frame_info(cls._frame_line), _time_info(tsp), level, message], strict=False))


    ####################################################################################################################################
    # Define class methods to Test for valid log level name or number
    @classmethod
    def is_valid_level(cls, level = None) -> bool:
        return  cls._is_valid_level_name(level=level) or cls._is_valid_level_number(level==level)

    @classmethod
    def _is_valid_level_name(cls, level: str = None,) -> bool:
        if isinstance(level, str):
            return level.upper() in cls._level_names.values()
        return False

    @classmethod
    def _is_valid_level_number(cls, level: int = None,) -> bool:
        if isinstance(level, int):
            return level in cls._level_numbers.values()
        return False


    ####################################################################################################################################
    # Define class methods to get log level name or number
    @classmethod
    def _get_level_number(cls, level: str = None,) -> int or None:
        if isinstance(level, str):
            valid_name = level.upper() if level.upper() in cls._level_names.values() else None
            return cls._level_numbers.get(valid_name) if valid_name is not None else None
        return None

    @classmethod
    def _get_level_name(cls, level: int = None,) -> str or None:
        if isinstance(level, int):
            valid_level = level if level in cls._level_numbers.values() else None
            return cls._level_names.get(valid_level) if valid_level is not None else None
        return None


    ####################################################################################################################################
    # Define class methods to determine if Logger config level of Logger normal level should be used for log level filtering
    @classmethod
    def _get_log_level(cls, config_lvl=None, norm_lvl=None,) -> int:
        return lambda c: norm_lvl if c else config_lvl

    @classmethod
    def _get_log_level_check(cls, config_lvl=None, norm_lvl=None):
        return cls._get_log_level(config_lvl=config_lvl, norm_lvl=norm_lvl)


    ####################################################################################################################################
    # Define class methods to convert log level from Number to and/or from Name.
    @classmethod
    def getLevelName(cls, level=None,) -> str or None:
        if cls._is_valid_level_number(level=level):
            return cls._get_level_name(level=level)
        elif cls._is_valid_level_name(level=level):
            return level
        else:
            return None

    @classmethod
    def getLevelNumber(cls, level=None,) -> int or None:
        if cls._is_valid_level_name(level=level):
            return cls._get_level_number(level=level)
        elif cls._is_valid_level_number(level=level):
            return level
        else:
            return None

    @classmethod
    def getLevelFrom(cls, level=None,) -> str or int:
        return cls.getLevelNumber(level=level) or cls.getLevelName(level=level)


    ####################################################################################################################################
    # Define log level methods
    @classmethod
    def set_level(cls, level=None,) -> None:
        if cls._is_valid_level_name(level=level):
            cls._log_level = level.upper() or _LOGGER_LOG_LEVEL_NAME
        elif cls._is_valid_level_number(level=level):
            cls._log_level = cls._get_level_name(level=level) or _LOGGER_LOG_LEVEL_NAME
        else:
            cls._log_level = _LOGGER_LOG_LEVEL_NAME

    @classmethod
    def get_level(cls,) -> str:
        return cls._log_level if hasattr(cls, "_log_level") else _LOGGER_LOG_LEVEL_NAME


    ####################################################################################################################################
    @classmethod
    def _logging_message(cls, formatter_string=None, dict_method=None, a=None, b=None,) -> str:
        return lambda frame, tsp, level, message: formatter_string % dict_method(frame=frame, tsp=tsp, level=level, message=message)


    ####################################################################################################################################
    @classmethod
    def _filter_by_level(cls, level=None, log_level=None) -> bool:
        if ( valid_level_num := cls.getLevelNumber(cls.is_valid_level(level=level)) )  and  ( valid_loglevel_num := cls.getLevelNumber(cls.is_valid_level(level=log_level)) ):
            return valid_loglevel_num >= valid_level_num
        else:
            return False


    ####################################################################################################################################
    # Define Logger class log level Logging with Filtering methods
    @classmethod
    def _log_at_level(cls, frame=None, tsp=None, level=None, message=None) -> None:
        if cls._filter_by_level(level=level, log_level=cls._get_log_level_check(config_lvl=cls._level_logger_config, norm_lvl=cls._level_logger_name)(cls.is_configured())):
            _console_message = cls._logging_message(cls._formatter_string_console, cls._console_dict)
            _file_message = cls._logging_message(cls._formatter_string_file, cls._file_dict)
            print(f"+{_console_message(frame, tsp, level, message)}")
            with open(cls._logging_file, "a") as f:
                f.write(f"+{_file_message(frame, tsp, level, message)}\n")


    ####################################################################################################################################
    # Define logging methods
    @classmethod
    def trace(cls, message=None,) -> None:
        cls._log_at_level(frame=cls._frm(), tsp=cls._tsp(), level=cls._level_trace, message=message)

    @classmethod
    def verbose(cls, message=None,) -> None:
        cls._log_at_level(frame=cls._frm(), tsp=cls._tsp(), level=cls._level_verbose, message=message)

    @classmethod
    def debug(cls, message=None,) -> None:
        cls._log_at_level(frame=cls._frm(), tsp=cls._tsp(), level=cls._level_debug, message=message)

    @classmethod
    def info(cls, message=None,) -> None:
        cls._log_at_level(frame=cls._frm(), tsp=cls._tsp(), level=cls._level_info, message=message)

    @classmethod
    def warning(cls, message=None,) -> None:
        cls._log_at_level(frame=cls._frm(), tsp=cls._tsp(), level=cls._level_warning, message=message)

    @classmethod
    def error(cls, message=None,) -> None:
        cls._log_at_level(frame=cls._frm(), tsp=cls._tsp(), level=cls._level_error, message=message)

    @classmethod
    def critical(cls, message=None,) -> None:
        cls._log_at_level(frame=cls._frm(), tsp=cls._tsp(), level=cls._level_critical, message=message)

    @classmethod
    def fatal(cls, message=None,) -> None:
        cls._log_at_level(frame=cls._frm(), tsp=cls._tsp(), level=cls._level_fatal, message=message)


    ####################################################################################################################################
    # Define the init method for the Logger class
    def __init__(
        self,
        _Logger__log_config: logging.config = None,
        _Logger__log_config_file_path: str = _LOGGER_LOG_CONFIG_FILE_PATH,
        _Logger__log_config_file_name: str = _LOGGER_LOG_CONFIG_FILE_NAME,
        _Logger__log_date_format: str = _LOGGER_LOG_DATE_FORMAT,
        _Logger__log_file_name: str = _LOGGER_LOG_FILE_NAME,
        _Logger__log_file_path: str = _LOGGER_LOG_FILE_PATH,
        _Logger__log_formatter_string: str = _LOGGER_LOG_FORMATTER_STRING,
        # _Logger__log_level: int = _LOGGER_LOG_LEVEL,
        _Logger__log_level_custom_names_list: list = _LOGGER_LOG_LEVEL_CUSTOM_NAMES_LIST,
        _Logger__log_level_logging_config: int = _LOGGER_LOG_LEVEL_LOGGING_CONFIG,
        _Logger__log_level_methods_dict: dict = _LOGGER_LOG_LEVEL_METHODS_DICT,
        _Logger__log_level_methods_list: list = _LOGGER_LOG_LEVEL_METHODS_LIST,
        _Logger__log_level_name: str = _LOGGER_LOG_LEVEL_NAME,
        _Logger__log_level_names_list: list = _LOGGER_LOG_LEVEL_NAMES_LIST,
        _Logger__log_level_numbers_dict: dict = _LOGGER_LOG_LEVEL_NUMBERS_DICT,
        _Logger__log_level_numbers_list: list = _LOGGER_LOG_LEVEL_NUMBERS_LIST,
        _Logger__log_level_redefinition: bool = _LOGGER_LOG_LEVEL_REDEFINITION,
        _Logger__log_message_default: str = _LOGGER_LOG_MESSAGE_DEFAULT,
        _Logger__uuid: str = None,
        **kwargs,
    ):

        Logger.debug("Logger: __init__: Starting Logger initialization with parameters:")
        for key, value in kwargs.items():
            Logger.debug(f"Logger: __init__:   {key}: {value}")


        # Initialize Logger class file name and path attributes
        Logger.debug("Logger: __init__: Initializing Logger class file name and path attributes")
        self.log_file_name = _Logger__log_file_name or __name__
        self.log_file_path = _Logger__log_file_path or str(os.path.join(os.getcwd(), "logs"))

        # Initialize Logger class config file name and path attributes
        Logger.debug("Logger: __init__: Initializing Logger class config file name and path attributes")
        self.log_config_file_name = str(_Logger__log_config_file_name or _LOGGER_LOG_CONFIG_FILE_NAME)
        self.log_config_file_path = _Logger__log_config_file_path or str(os.path.join(os.getcwd(), "conf"))
        self.log_config_file = str(os.path.join(self.log_config_file_path, self.log_config_file_name))

        # Initialize Logger class attributes for log formatting and config
        Logger.debug("Logger: __init__: Initializing Logger class attributes for log formatting and config")
        self.log_formatter_string = _Logger__log_formatter_string or _LOGGER_LOG_FORMATTER_STRING
        self.log_message_default = _Logger__log_message_default or _LOGGER_LOG_MESSAGE_DEFAULT
        self.log_level_redefinition = _Logger__log_level_redefinition or _LOGGER_LOG_LEVEL_REDEFINITION

        # Initialize Logger class attributes for logging configuration
        Logger.debug("Logger: __init__: Initializing Logger class attributes for logging configuration")
        self.log_level_custom_names_list = _Logger__log_level_custom_names_list or _LOGGER_LOG_LEVEL_CUSTOM_NAMES_LIST
        self.log_level_methods_dict = _Logger__log_level_methods_dict or _LOGGER_LOG_LEVEL_METHODS_DICT
        self.log_level_methods_list = _Logger__log_level_methods_list or _LOGGER_LOG_LEVEL_METHODS_LIST
        self.log_level_name = _Logger__log_level_name or _LOGGER_LOG_LEVEL_NAME
        self.log_level_logging_config = _Logger__log_level_logging_config or _LOGGER_LOG_LEVEL_LOGGING_CONFIG
        self.log_level_names_list = _Logger__log_level_names_list or _LOGGER_LOG_LEVEL_NAMES_LIST
        self.log_level_numbers_dict = _Logger__log_level_numbers_dict or _LOGGER_LOG_LEVEL_NUMBERS_DICT
        self.log_level_numbers_list = _Logger__log_level_numbers_list or _LOGGER_LOG_LEVEL_NUMBERS_LIST

        # Initialize Logger class attributes for log date formatting
        Logger.debug("Logger: __init__: Initializing Logger class attributes for log date formatting")
        self.log_date_format = _Logger__log_date_format or _LOGGER_LOG_DATE_FORMAT

        # Call the superclass __init__ method for Logging class using the named logger instead of root logger
        Logger.debug(f"Logger: __init__: Calling superclass __init__ method for Logger class with name: {self.log_file_name}, level: {self.log_level_logging_config}")
        super().__init__(name=self.log_file_name, level=self.log_level_logging_config)  # Use project logger name
        # super().__init__(name=_LOGGER_NAME, level=self.log_level_logging_config)  # Use project logger name

        Logger.debug(f"Logger: __init__: Logging Config: logging.config: Type: {type(logging.config)} Value: {logging.config}")
        Logger.debug(f"Logger: __init__: Logging Config: _Logger__log_config: Type: {type(_Logger__log_config)} Value: {_Logger__log_config}")
        # add the logging.config to the Logger class for config file loading and initialization
        Logger.debug("Logger: __init__: add the logging.config to the Logger class for config file loading and initialization")
        if _Logger__log_config and hasattr(_Logger__log_config, 'dictConfig'):
            self.config = _Logger__log_config
        else:
            # Use logging.config when _Logger__log_config is None or not a valid logging config object
            Logger.debug("Logger: __init__: Use logging.config when _Logger__log_config is None or not a valid logging config object")
            self.config = logging.config
            if _Logger__log_config is None:
                Logger.warning("Logger: __init__: Warning: _Logger__log_config doesn't have dictConfig method, using logging.config instead")

        Logger.debug(f"Logger: __init__: Added Logging.config to the Logger class instance: Type: {type(self.config)} Value set to: {self.config}")
        Logger.debug(f"Logger: __init__: Config has dictConfig method: {hasattr(self.config, 'dictConfig')}")

        # Configure the logger with all custom log levels FIRST. This must happen before loading the YAML configuration
        Logger.debug("Logger: __init__: Configure the logger with all custom log levels FIRST. This must happen before loading the YAML configuration")
        Logger.debug(f"Logger: __init__: Configuring custom log levels for Logger class with names: {self.log_level_names_list}, numbers: {self.log_level_numbers_dict}, methods: {self.log_level_methods_dict}")
        self._init_custom_log_levels(
            custom_names_list = self.log_level_custom_names_list,
            numbers_dict = self.log_level_numbers_dict,
            methods_dict = self.log_level_methods_dict,
            allow_redefinition = self.log_level_redefinition,
        )

        # TODO: This might be a good place to check if custom log levels are being successfully assigned
        # Update the Logger instance simple logging config with the provided log config values
        Logger.debug(f"Logger: __init__: Updating Logger instance config with handler values: {self.handlers}")
        if not self.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(fmt=self.log_formatter_string, datefmt=self.log_date_format)
            handler.setFormatter(formatter)
            self.addHandler(handler)
        Logger.debug(f"Logger: __init__: Completed Updating Logger instance config with handler values: {self.handlers}")

        # Only configure logging once globally to avoid conflicts:   if not self.__class__._LOGGER_SINGLETON_LOGGING_CONFIGURED
        # IMPORTANT: Custom log levels must be defined before configuring Logging using the log config yaml file.
        Logger.debug("Logger: __init__: Only configure logging once globally to avoid conflicts:   if not self.__class__._LOGGER_SINGLETON_LOGGING_CONFIGURED")
        Logger.debug("Logger: __init__: IMPORTANT: Custom log levels must be defined before configuring Logging using the log config yaml file.")
        if not Logger.is_configured():
            Logger.debug(f"Logger: __init__: Configuring logging using the config dict read from logging config file: {self.log_config_file}")
            try:
                with open(self.log_config_file, "r") as self.log_conf:
                    self.logger_configs = yaml.safe_load(self.log_conf.read())           # Read logging config yaml and create Config Dict
                    Logger.debug(f"Logger: __init__: self.logger_configs: Type: {type(self.logger_configs)}, Has dictConfig: {hasattr(self.logger_configs, 'dictConfig')}")
                Logger.debug(f"Logger: __init__: self.logger_configs: Type: {type(self.logger_configs)}, Has dictConfig: {hasattr(self.logger_configs, 'dictConfig')}")
            except Exception as e:
                Logger.error(f"Logger: __init__: Error reading config file for Juniper project logging: {e}")
                traceback.print_exc()

            # Logger.info(f"Logger: __init__: Loaded logging config from {self.log_config_file_path}")
            Logger.debug(f"Logger: __init__: Loaded logging config from {self.log_config_file_path}")
            # Logger.info(f"Logger: __init__: self.config: Type: {type(self.config)}, Has dictConfig: {hasattr(self.config, 'dictConfig')}")
            Logger.debug(f"Logger: __init__: self.config: Type: {type(self.config)}, Has dictConfig: {hasattr(self.config, 'dictConfig')}")

            Logger.debug(f"Logger: __init__: Configure as Singleton: Class Variable: logging configured flag: {self.__class__.SingletonLoggingConfigured}")
            Logger.debug("Logger: __init__: Configuring logging using the config dict read from logging config file")
            Logger.debug(f"Logger: __init__: Loaded logging config from {self.log_config_file_path}")
            Logger.debug(f"Logger: __init__: self.config: Type: {type(self.config)}, Has dictConfig: {hasattr(self.config, 'dictConfig')}")

            # Only configure if we have a valid config dictionary and a valid config object
            if self.logger_configs and isinstance(self.logger_configs, dict) and hasattr(self.config, 'dictConfig'):
                Logger.debug(f"Logger: __init__: Applying dictConfig with loaded configuration: {self.logger_configs}")
                self.config.dictConfig(self.logger_configs)                              # Load the config Dict into Logging Object
                # Logger.info(f"Logger: __init__: Successfully applied dictConfig: {self.config}")
                Logger.debug(f"Logger: __init__: Successfully applied dictConfig: {self.config}")
                Logger.set_configured()
                # Logger.info("Logger: __init__: Successfully applied dictConfig")
                Logger.debug("Logger: __init__: Successfully applied dictConfig")
            else:
                Logger.warning("Logger: __init__: Falling back to basic logging configuration")
                Logger.warning(f"Logger: __init__: Skipping dictConfig - configs invalid: {bool(self.logger_configs)}, is dict: {isinstance(self.logger_configs, dict)}, has dictConfig: {hasattr(self.config, 'dictConfig')}")
        else:
            Logger.warning("Logger: __init__: Logging already configured globally, skipping YAML configuration")

        # Set the logger to the current instance
        # Logger.info("Logger: __init__: Setting Logger instance to self")
        Logger.debug("Logger: __init__: Setting Logger instance to self")
        self.set_logger(self)
        Logger.debug(f"Logger: __init__: Logger instance set to self: Type: {type(self.logger)}, Value: {self.logger}")
        # self.debug(f"Logger: __init__: Logger instance set to self: Type: {type(self.logger)}, Value: {self.logger}")

        # Set log level to the provided log level
        # Logger.info(f"Logger: __init__: Setting Temporary log level for configuring Logger instance: {_Logger__log_level_logging_config}")
        Logger.debug(f"Logger: __init__: Setting Temporary log level for configuring Logger instance: {_Logger__log_level_logging_config}")
        # self.info(f"Logger: __init__: Setting Temporary log level for configuring Logger instance: {_Logger__log_level_logging_config}")
        self.set_log_level(_Logger__log_level_logging_config)  # Set the log level for the Logger instance

        # Logger.info(f"Logger: __init__: Logger instance log level set to: {self.get_log_level()}, log level name: {self.get_log_level_name()}")
        # self.info(f"Logger: __init__: Logger instance log level set to: {self.get_log_level()}, log level name: {self.get_log_level_name()}")
        self.debug(f"Logger: __init__: Logger instance log level set to: {self.get_log_level()}, log level name: {self.get_log_level_name()}")

        # Logger.debug(f"Logger: __init__: Logger class custom log levels configured with names: {self.log_level_names_list}, numbers: {self.log_level_numbers_dict}, methods: {self.log_level_methods_dict}")
        self.debug(f"Logger: __init__: Logger class custom log levels configured with names: {self.log_level_names_list}, numbers: {self.log_level_numbers_dict}, methods: {self.log_level_methods_dict}")
        # Logger.debug(f"Logger: __init__: Logger class custom log levels configured with updated level set: Value: {self.getEffectiveLevel()}, Name: {logging.getLevelName(self.getEffectiveLevel())}")
        self.debug(f"Logger: __init__: Logger class custom log levels configured with updated level set: Value: {self.getEffectiveLevel()}, Name: {logging.getLevelName(self.getEffectiveLevel())}")
        # Logger.debug("Logger: __init__: Completed configuration of custom log levels for Logger class.")
        self.debug("Logger: __init__: Completed configuration of custom log levels for Logger class.")

        # Logger.info("Logger: __init__: Completed initialization of Logging for Logger class.")
        # self.info("Logger: __init__: Completed initialization of Logging for Logger class.")
        self.debug("Logger: __init__: Completed initialization of Logging for Logger class.")

        # self.trace("Logger: __init__: Completed configuration of Logger class.")
        # self.verbose("Logger: __init__: Completed configuration of Logger class.")
        # self.debug("Logger: __init__: Completed configuration of Logger class.")
        # self.info("Logger: __init__: Completed configuration of Logger class.")
        # self.warning("Logger: __init__: Completed configuration of Logger class.")
        # self.error("Logger: __init__: Completed configuration of Logger class.")
        # self.critical("Logger: __init__: Completed configuration of Logger class.")
        # self.fatal("Logger: __init__: Completed configuration of Logger class.")



    ####################################################################################################################################
    # Define private methods for the Logger class
    ####################################################################################################################################

    ####################################################################################################################################
    # Define an init method to configure custom log levels
    def _init_custom_log_levels(
        self,
        custom_names_list: list[str] = None,
        numbers_dict: dict[str, int] = None,
        methods_dict: dict[str, str] = None,
        allow_redefinition: bool = False,
        **kwargs,
    ) -> None:
        """
        Description:
            Initialize custom log levels for the Logger class.
        Args:
            custom_names_list: List of custom log level names.
            numbers_dict: Dictionary mapping custom log level names to their corresponding numbers.
            methods_dict: Dictionary mapping custom log level names to their corresponding methods.
            allow_redefinition: Boolean indicating whether to allow redefinition of custom log levels.
        Returns:
            None
        Notes:
            This method configures the Logger class with custom log levels based on the provided lists and dictionaries.
            It dynamically creates log methods for each custom log level and assigns them to the Logger class.
        """
        # Configure the logger with all custom log levels
        # self.debug("Logger: __init_custom_log_levels: Configuring custom log levels for Logger class")
        Logger.debug("Logger: __init_custom_log_levels: Configuring custom log levels for Logger class")
        # self.debug(f"Logger: __init_custom_log_levels: Custom Log Level Names List: {custom_names_list}")
        Logger.debug(f"Logger: __init_custom_log_levels: Custom Log Level Names List: {custom_names_list}")
        # self.debug(f"Logger: __init_custom_log_levels: Log Level Numbers Dict: {numbers_dict}")
        Logger.debug(f"Logger: __init_custom_log_levels: Log Level Numbers Dict: {numbers_dict}")
        # self.debug(f"Logger: __init_custom_log_levels: Log Level Methods Dict: {methods_dict}")
        Logger.debug(f"Logger: __init_custom_log_levels: Log Level Methods Dict: {methods_dict}")
        # self.debug(f"Logger: __init_custom_log_levels: Log Allow Log Level Redefinition: {allow_redefinition}")
        Logger.debug(f"Logger: __init_custom_log_levels: Log Allow Log Level Redefinition: {allow_redefinition}")
        for custom_log_level_name in custom_names_list:
            # self.debug(f"Logger: __init_custom_log_levels: Configuring custom log level: {custom_log_level_name}")
            Logger.debug(f"Logger: __init_custom_log_levels: Configuring custom log level: {custom_log_level_name}")

            custom_log_level_number = numbers_dict.get(custom_log_level_name)
            # self.debug(f"Logger: __init_custom_log_levels: Custom log level number for {custom_log_level_name}: {custom_log_level_number}")
            Logger.debug(f"Logger: __init_custom_log_levels: Custom log level number for {custom_log_level_name}: {custom_log_level_number}")
            custom_log_level_method = methods_dict.get(custom_log_level_name)
            # self.debug(f"Logger: __init_custom_log_levels: Custom log level method for {custom_log_level_name}: {custom_log_level_method}")
            Logger.debug(f"Logger: __init_custom_log_levels: Custom log level method for {custom_log_level_name}: {custom_log_level_method}")

            # Validate custom log level names, methods, and numbers
            if not self._init_validate_custom_log_level(
                custom_log_level_name=custom_log_level_name,
                custom_log_level_number=custom_log_level_number,
                custom_log_level_method=custom_log_level_method,
                allow_redefinition=allow_redefinition,
            ):
                # self.warning(f"Logger: __init_custom_log_levels: Warning: Custom log level {custom_log_level_name} is invalid. Skipping...")
                Logger.warning(f"Logger: __init_custom_log_levels: Warning: Custom log level {custom_log_level_name} is invalid. Skipping...")
                continue
            # self.debug(f"Logger: __init_custom_log_levels: Validated Custom log level name: {custom_log_level_name}, number: {custom_log_level_number}, method: {custom_log_level_method}")
            Logger.debug(f"Logger: __init_custom_log_levels: Validated Custom log level name: {custom_log_level_name}, number: {custom_log_level_number}, method: {custom_log_level_method}")

            # Add the custom log level attribute for custom log level name and number
            # self.debug("Logger: __init_custom_log_levels: Add the custom log level attributes for current custom name and number")
            Logger.debug("Logger: __init_custom_log_levels: Add the custom log level attributes for current custom name and number")
            addLevelName(custom_log_level_number, custom_log_level_name)

            # Assign the custom log level method to the logger class
            # self.debug(f"Logger: __init_custom_log_levels: Assigning custom log level method: {custom_log_level_method} for custom log level name: {custom_log_level_name}")
            Logger.debug(f"Logger: __init_custom_log_levels: Assigning custom log level method: {custom_log_level_method} for custom log level name: {custom_log_level_name}")
            setattr(
                self,
                custom_log_level_method,
                self._init_log_method(
                    level_number=custom_log_level_number,
                    level_name=custom_log_level_name,
                ),
            )

            # self.info(f"Logger: __init_custom_log_levels: Added custom log level Name: {custom_log_level_name}, Number: {custom_log_level_number}, Method: {custom_log_level_method} for Logger class")
            # Logger.info(f"Logger: __init_custom_log_levels: Added custom log level Name: {custom_log_level_name}, Number: {custom_log_level_number}, Method: {custom_log_level_method} for Logger class")
            Logger.debug(f"Logger: __init_custom_log_levels: Added custom log level Name: {custom_log_level_name}, Number: {custom_log_level_number}, Method: {custom_log_level_method} for Logger class")
            # self.debug(f"Logger: __init_custom_log_levels: Added custom log level method: {getattr(self, custom_log_level_method)} for Logger class")
            Logger.debug(f"Logger: __init_custom_log_levels: Added custom log level method: {getattr(self, custom_log_level_method)} for Logger class")


    ####################################################################################################################################
    # Define a private method to validate custom log level names, numbers, and methods
    def _init_validate_custom_log_level(
        self,
        custom_log_level_name: str = None,
        custom_log_level_number: int = None,
        custom_log_level_method: str = None,
        allow_redefinition: bool = False,
    ) -> bool:
        """
        Description:
            Validate custom log level name, number, and method.
        Args:
            custom_log_level_name: The custom log level name.
            custom_log_level_number: The custom log level number.
            custom_log_level_method: The custom log level method.
            allow_redefinition: Boolean indicating whether to allow redefinition of custom log levels.
        Returns:
            bool: True if the custom log level is valid, False otherwise.
        """
        # Validate the custom log level name, number, and method
        # self.debug(f"Logger: _init_validate_custom_log_level: Validating custom log level: {custom_log_level_name}, number: {custom_log_level_number}, method: {custom_log_level_method}")
        Logger.debug(f"Logger: _init_validate_custom_log_level: Validating custom log level: {custom_log_level_name}, number: {custom_log_level_number}, method: {custom_log_level_method}")
        invalid = False
        if (invalid := (custom_log_level_name is None or not isinstance(custom_log_level_name, str))):
            # self.warning(f"Logger: _init_validate_custom_log_level: Custom log level name is invalid for {custom_log_level_name}.")
            Logger.warning(f"Logger: _init_validate_custom_log_level: Custom log level name is invalid for {custom_log_level_name}.")
        elif (invalid := (custom_log_level_number is None or not isinstance(custom_log_level_number, int))):
            # self.warning(f"Logger: _init_validate_custom_log_level: Custom log level number is invalid for {custom_log_level_name}.")
            Logger.warning(f"Logger: _init_validate_custom_log_level: Custom log level number is invalid for {custom_log_level_name}.")
        elif (invalid := (custom_log_level_method is None or not isinstance(custom_log_level_method, str))):
            # self.warning(f"Logger: _init_validate_custom_log_level: Custom log level method (Name) is invalid for {custom_log_level_name}.")
            Logger.warning(f"Logger: _init_validate_custom_log_level: Custom log level method (Name) is invalid for {custom_log_level_name}.")
        elif (invalid := (not allow_redefinition and (hasattr(self, custom_log_level_name) or hasattr(self, str(custom_log_level_number)) or hasattr(self, custom_log_level_method)))):
            # self.warning(f"Logger: _init_validate_custom_log_level: Log level redefinition is disabled and Custom log level name, number, or method already exists for {custom_log_level_name}.")
            Logger.warning(f"Logger: _init_validate_custom_log_level: Log level redefinition is disabled and Custom log level name, number, or method already exists for {custom_log_level_name}.")
        else:
            # self.debug(f"Logger: _init_validate_custom_log_level: Custom log level validation for {custom_log_level_name} returned: {not invalid}")
            Logger.debug(f"Logger: _init_validate_custom_log_level: Custom log level validation for {custom_log_level_name} returned: {not invalid}")
        return not invalid  # Return True if the custom log level is valid, False otherwise


    ####################################################################################################################################
    # Define the init method that dynamically generates the named logging method for custom log levels
    def _init_log_method(
        self,
        level_number: int = None,
        level_name: str = None,
    ) -> callable:
        """
        Description:
            Create a custom log method for the custom log level name.
        Args:
            custom_log_level_number: The custom log level number.
            custom_log_level_name: The custom log level name.
        Returns:
            A function that logs a message at the custom log level.
        """
        if level_number is None or level_name is None:
            return lambda message=_LOGGER_LOG_MESSAGE_DEFAULT, *args, **kwargs: None

        # self.debug(f"Logger: _init_log_method: make_log_method: Creating custom log method for level name: {level_name}, level number: {level_number}")
        Logger.debug(f"Logger: _init_log_method: make_log_method: Creating custom log method for level name: {level_name}, level number: {level_number}")
        def log_for_level(
            message: str = _LOGGER_LOG_MESSAGE_DEFAULT,
            *args,
            **kwargs,
        ) -> None:
            """
            Description:
                Log a message at the custom log level.
            Args:
                message: The message to be logged.
                *args: Additional positional arguments.
                **kwargs: Additional keyword arguments.
            Notes:
                This method builds a unique log method for each custom log level defined in the LogConfig class.
                The method is dynamically assigned to the Logger class using setattr, and assigned the name of the custom log level method.
            Returns:
                None
            """
            # This is log level check is being called from the current logger instance (not the Logger class)
            # self.debug("Logger: log_for_level: This is log level check is being called from the current logger instance (not the Logger class)")
            # self.trace("Logger: log_for_level: This is log level check is being called from the current logger instance (not the Logger class)")
            # Logger.debug("Logger: log_for_level: This is log level check is being called from the current logger instance (not the Logger class)")
            if self.isEnabledFor(level_number):
                # Set stacklevel to 2 to skip this wrapper function and point to the actual caller. This ensures the correct file and line number are logged
                # self.debug("Logger: log_for_level: Set stacklevel to 2 to skip this wrapper function and point to the actual caller. This ensures the correct file and line number are logged")
                # self.trace("Logger: log_for_level: Set stacklevel to 2 to skip this wrapper function and point to the actual caller. This ensures the correct file and line number are logged")
                # Logger.debug("Logger: log_for_level: Set stacklevel to 2 to skip this wrapper function and point to the actual caller. This ensures the correct file and line number are logged")
                kwargs.setdefault('stacklevel', 2)
                # Call the logger's log method with the custom log level number
                # self.trace("Logger: log_for_level: Call the logger's log method with the custom log level number")
                # Logger.debug("Logger: log_for_level: Call the logger's log method with the custom log level number")
                self._log(level_number, message, args, **kwargs)

        # Return the log method for the custom log level
        # self.verbose(f"Logger: _init_log_method: Returning log method for custom log level: {level_name}, number: {level_number}")
        # self.verbose(f"Logger: _init_log_method: log_for_level: Returning custom level log Function: {log_for_level}")
        return log_for_level


    def _generate_uuid(self) -> str:
        """
        Description:
            Generate a new UUID for the Logger class.
        Args:
            self: The instance of the class.
        Returns:
            str: The generated UUID.
        """
        new_uuid = str(uuid.uuid4())
        # self.debug(f"Logger: _generate_uuid: Generated new UUID: {new_uuid}")
        Logger.debug(f"Logger: _generate_uuid: Generated new UUID: {new_uuid}")
        return new_uuid

    def isEnabledFor(self, level: int) -> bool:
        """
        Description:
            Check if the logger is enabled for the specified log level.
        Args:
            level: The log level to check.
        Returns:
            bool: True if the logger is enabled for the specified log level, False otherwise.
        """
        # return super().isEnabledFor(level or self.log_level or logging.NOTSET)
        # return level >= self._get_level_number(level=self.log_level_name)
        return level >= self.log_level_numbers_dict.get(self.log_level_name, logging.NOTSET)


    ####################################################################################################################################
    # Define the Logger Class Public Update methods
    def update_log_level(self, log_level_name: str = None, log_level: int = None) -> None:
        """
        Description:
            Update the log level for the Logger class.
        Args:
            log_level_name: The name of the log level to be set.
            log_level_number: The numeric value of the log level to be set.
        Returns:
            None
        Notes:
            This method updates the log level for the Logger class based on the provided name or number.
        """
        # self.logger.info("Logger: update_log_level: Updating log level for Logger class")
        self.logger.debug("Logger: update_log_level: Updating log level for Logger class")
        if log_level is not None:
            self.set_log_level(log_level=log_level, sync_level_name=True)
        elif log_level_name is not None:
            self.set_log_level_name(log_level_name=log_level_name, sync_level=True)
        else:
            self.warning("Logger: update_log_level: No valid log level name or number provided. No changes made.")


    ####################################################################################################################################
    # Define the Logger Class Public setter methods

    def set_logger(self, logger: logging.Logger):
        """
        Description:
            Set the logger instance for the Logger class.
        Args:
            self: The instance of the Logger class.
            logger: The logger instance to be set.
        Returns:
            None
        """
        self.logger = logger or None

    def set_log_file_name(self, log_file_name: str):
        """
        Description:
            Set the log file name for the Logger class.
        Args:
            self: The instance of the Logger class.
            log_file_name: The name of the log file to be set.
        Returns:
            None
        """
        self.log_file_name = log_file_name or None

    def set_log_file_path(self, log_file_path: str):
        """
        Description:
            Set the log file path for the Logger class.
        Args:
            self: The instance of the Logger class.
            log_file_path: The path of the log file to be set.
        Returns:
            None
        """
        self.log_file_path = log_file_path or None

    def _set_level(self, level: int = None):
        """
        Description:
            Set the logging level for the Logger class.
        Args:
            self: The instance of the Logger class.
            level: The logging level to be set.
        Returns:
            None
        """
        self.setLevel(level or self.log_level or logging.NOTSET)

    def set_log_level(
        self,
        log_level: int = None,
        sync_level_name: bool = True
    ):
        """
        Description:
            Set the logging level for the Logger class.
        Args:
            self: The instance of the Logger class.
            log_level: The logging level to be set.
            sync_level_name: Boolean indicating whether to synchronize the log level name with the log level.
        Notes:
            If log_level is None or not a valid logging level, it will not change the current log level.
            If sync_level_name is True, it will also set the log level name based on the log level.
        Returns:
            None
        """
        self.log_level = log_level if log_level and self.get_name_from_level(log_level) else None
        self.debug(f"Logger: set_log_level: Completed Setting log level to: {self.log_level}")
        self._set_level(self.log_level)
        if log_level and sync_level_name:
            log_level_name = self.get_name_from_level(log_level) or None
            self.set_log_level_name(log_level_name=log_level_name, sync_level=False,)
        self.debug(f"Logger: set_log_level: Completed Setting log level to: {self.log_level} with name: {self.log_level_name}")

    def set_log_level_name(
        self,
        log_level_name: str = None,
        sync_level: bool = True
    ):
        """
        Description:
            Set the logging level name for the Logger class.
        Args:
            self: The instance of the Logger class.
            log_level_name: The logging level name to be set.
            sync_level: Boolean indicating whether to synchronize the log level with the log level name.
        Notes:
            If log_level_name is None or not a valid logging level name, it will not change the current log level name.
            If sync_level is True, it will also set the log level based on the log level name.
        Returns:
            None
        """
        self.log_level_name = log_level_name if log_level_name and self.get_name_from_level(self.log_level) else None
        self.debug(f"Logger: set_log_level_name: Completed Setting log level name to: {self.log_level_name}")
        if log_level_name and sync_level:
            log_level = self.log_level_numbers_dict.get(log_level_name, logging.NOTSET)
            self.debug(f"Logger: set_log_level_name: Setting log level to: {log_level} for log level name: {log_level_name}")
            self.set_log_level(log_level=log_level, sync_level_name=False)
        self.debug(f"Logger: set_log_level_name: Completed Setting log level name to: {self.log_level_name} with level: {self.log_level}")

    def set_log_date_format(self, log_date_format: str):
        """
        Description:
            Set the log date format for the Logger class.
        Args:
            self: The instance of the Logger class.
            log_date_format: The date format for the log messages.
        Returns:
            None
        """
        self.log_date_format = log_date_format or None

    def set_log_formatter_string(self, log_formatter_string: str):
        """
        Description:
            Set the log formatter string for the Logger class.
        Args:
            self: The instance of the Logger class.
            log_formatter_string: The formatter string for the log messages.
        Returns:
            None
        """
        self.log_formatter_string = log_formatter_string or None

    def set_log_message_default(self, log_message_default: str):
        """
        Description:
            Set the default log message for the Logger class.
        Args:
            self: The instance of the Logger class.
            log_message_default: The default message to be logged if no message is provided.
        Returns:
            None
        """
        self.log_message_default = log_message_default or None

    def set_log_level_custom_names_list(self, log_level_custom_names_list: list[str]):
        """
        Description:
            Set the custom names for the log levels.
        Args:
            self: The instance of the Logger class.
            log_level_custom_names_list: A list of custom names for the log levels.
        Returns:
            None
        """
        self.log_level_custom_names_list = log_level_custom_names_list or None

    def set_log_level_numbers_dict(self, log_level_numbers_dict: dict[str, int]):
        """
        Description:
            Set the log level numbers dictionary for the Logger class.
        Args:
            self: The instance of the Logger class.
            log_level_numbers_dict: A dictionary mapping log level names to their numeric values.
        Returns:
            None
        """
        self.log_level_numbers_dict = log_level_numbers_dict or None

    def set_log_level_methods_dict(self, log_level_methods_dict: dict[str, str]):
        """
        Description:
            Set the log level methods dictionary for the Logger class.
        Args:
            self: The instance of the Logger class.
            log_level_methods_dict: A dictionary mapping log level names to their methods.
        Returns:
            None
        """
        self.log_level_methods_dict = log_level_methods_dict or None

    def set_log_allow_log_level_redefinition(self, log_allow_log_level_redefinition: bool):
        """
        Description:
            Set the log allow log level redefinition flag for the Logger class.
        Args:
            self: The instance of the Logger class.
            log_allow_log_level_redefinition: A boolean indicating if log level redefinition is allowed.
        Returns:
            None
        """
        self.log_allow_log_level_redefinition = log_allow_log_level_redefinition or None

    def set_uuid(self, uuid: str | None):
        """
        Description:
            Set the UUID for the Logger class.
        Args:
            self: The instance of the Logger class.
            uuid: The UUID to be set for the Logger class.
        Returns:
            None
        """
        if not hasattr(self, "uuid") or self.uuid is None:
            self.uuid = (uuid, self._generate_uuid())[uuid is None]  # Generate a new UUID if none is provided
        else:
            self.fatal(f"Logger: set_uuid: Fatal Error: UUID already set: {self.uuid}. Changing UUID is bad Juju.  Exiting...")
            os._exit(1)


    ####################################################################################################################################
    # Define the Logger Class Public getter methods
    def get_uuid(self) -> str:
        """
        Description:
            Get the UUID for the Logger class.
        Args:
            self: The instance of the Logger class.
        Returns:
            str: The UUID of the Logger class.
        """
        if self.uuid is None or not hasattr(self, "uuid"):
            self.set_uuid()
        return self.uuid

    def get_logger(self) -> logging.Logger:
        """
        Description:
            Get the logger instance for the Logger class.
        Args:
            self: The instance of the Logger class.
        Returns:
            logging.Logger: The logger instance for the Logger class.
        """
        return self

    def get_log_file_name(self) -> str or None:
        """
        Description:
            Get the log file name for the Logger class.
        Args:
            self: The instance of the Logger class.
        Returns:
            str or None: The log file name for the Logger class.
        """
        return self.log_file_name if hasattr(self, "log_file_name") else None

    def get_log_file_path(self) -> str or None:
        """
        Description:
            Get the log file path for the Logger class.
        Args:
            self: The instance of the Logger class.
        Returns:
            str or None: The log file path for the Logger class.
        """
        return self.log_file_path if hasattr(self, "log_file_path") else None

    def _get_level(self) -> int or None:
        """
        Description:
            Get the logging level for the Logger class.
        Args:
            self: The instance of the Logger class.
        Returns:
            int or None: The logging level for the Logger class.
        """
        return self.level if hasattr(self, "level") else None

    def get_log_level(self) -> int or None:
        """
        Description:
            Get the log level for the Logger class.
        Args:
            self: The instance of the Logger class.
        Returns:
            int or None: The log level for the Logger class.
        """
        return self.log_level if hasattr(self, "log_level") else None

    def get_log_level_name(self) -> str or None:
        """
        Description:
            Get the log level name for the Logger class.
        Args:
            self: The instance of the Logger class.
        Returns:
            str or None: The log level name for the Logger class.
        """
        return self.log_level_name if hasattr(self, "log_level_name") else None

    def get_name_from_level(self, log_level: int) -> str or None:
        """
        Description:
            Get the log level name from the log level number.
        Args:
            self: The instance of the Logger class.
            log_level: The log level number to get the name for.
        Returns:
            str or None: The log level name for the given log level number.
        """
        log_level_name = list(self.log_level_numbers_dict.keys())[list(self.log_level_numbers_dict.values()).index(log_level)] if log_level in self.log_level_numbers_dict.values() else None
        self.debug(f"Logger: get_name_from_level: Returning log level name: {log_level_name} for log level: {log_level}")
        return log_level_name

    def get_log_date_format(self) -> str or None:
        """
        Description:
            Get the log date format for the Logger class.
        Args:
            self: The instance of the Logger class.
        Returns:
            str or None: The log date format for the Logger class.
        """
        return self.log_date_format if hasattr(self, "log_date_format") else None

    def get_log_formatter_string(self) -> str or None:
        """
        Description:
            Get the log formatter string for the Logger class.
        Args:
            self: The instance of the Logger class.
        Returns:
            str or None: The log formatter string for the Logger class.
        """
        return self.log_formatter_string if hasattr(self, "log_formatter_string") else None

    def get_log_message_default(self) -> str or None:
        """
        Description:
            Get the default log message for the Logger class.
        Args:
            self: The instance of the Logger class.
        Returns:
            str or None: The default log message for the Logger class.
        """
        return self.log_message_default if hasattr(self, "log_message_default") else None

    def get_log_level_custom_names_list(self) -> list[str] or None:
        """
        Description:
            Get the custom names list for the log levels.
        Args:
            self: The instance of the Logger class.
        Returns:
            list[str] or None: The custom names list for the log levels.
        """
        return self.log_level_custom_names_list if hasattr(self, "log_level_custom_names_list") else None

    def get_log_level_numbers_dict(self) -> dict[str, int] or None:
        """
        Description:
            Get the log level numbers dictionary for the Logger class.
        Args:
            self: The instance of the Logger class
        Returns:
            dict[str, int] or None: The log level numbers dictionary for the Logger class.
        """
        return self.log_level_numbers_dict if hasattr(self, "log_level_numbers_dict") else None

    def get_log_level_methods_dict(self) -> dict[str, str] or None:
        """
        Description:
            Get the log level methods dictionary for the Logger class.
        Args:
            self: The instance of the Logger class.
        Returns:
            dict[str, str] or None: The log level methods dictionary for the Logger class.
        """
        return self.log_level_methods_dict if hasattr(self, "log_level_methods_dict") else None

    def get_log_allow_log_level_redefinition(self) -> bool or None:
        """
        Description:
            Get the log allow log level redefinition for the Logger class.
        Args:
            self: The instance of the Logger class.
        Returns:
            bool or None: The log allow log level redefinition for the Logger class.
        """
        return self.log_allow_log_level_redefinition if hasattr(self, "log_allow_log_level_redefinition") else None
