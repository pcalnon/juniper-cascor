#!/usr/bin/env python
####################################################################################################################################################################################################
# Project:       Juniper
# Application:   Dynamic Neural Network
# File Name:     LogConfig.py
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
#     This module provides a logging configuration for the Juniper project.
#
#####################################################################################################################################################################################################
# Notes:
#    - This file contains a LogConfig class that sets up logging configuration for the Juniper project.
#    - The LogConfig class allows for dynamic creation of custom log levels based on provided names, numbers, and methods.
#    - The LogConfig class also supports loading a logging configuration from a YAML file.
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
#    - fix this to allow the Trace log level to be specified in the logging config file
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#
#####################################################################################################################################################################################################
import os
# import yaml
import uuid
import logging
import logging.config

# from inspect import currentframe, getframeinfo

from logging import setLoggerClass
from log_config.logger.logger import Logger
from constants.constants import (
    _LOG_CONFIG_LOG_CONFIG_FILE_NAME,
    _LOG_CONFIG_LOG_CONFIG_FILE_PATH,
    _LOG_CONFIG_LOG_LEVEL_CUSTOM_NAMES_LIST,
    _LOG_CONFIG_LOG_DATE_FORMAT,
    _LOG_CONFIG_LOG_FILE_NAME,
    _LOG_CONFIG_LOG_FILE_PATH,
    _LOG_CONFIG_LOG_FORMATTER_STRING,
    _LOG_CONFIG_LOG_LEVEL,
    _LOG_CONFIG_LOG_LEVEL_LOGGING_CONFIG,
    _LOG_CONFIG_LOG_LEVEL_METHODS_DICT,
    _LOG_CONFIG_LOG_LEVEL_METHODS_LIST,
    _LOG_CONFIG_LOG_LEVEL_NAME,
    _LOG_CONFIG_LOG_LEVEL_NAMES_LIST,
    _LOG_CONFIG_LOG_LEVEL_NUMBERS_DICT,
    _LOG_CONFIG_LOG_LEVEL_NUMBERS_LIST,
    _LOG_CONFIG_LOG_LEVEL_REDEFINITION,
    _LOG_CONFIG_LOG_MESSAGE_DEFAULT,
    _LOGGER_NAME,
)


########################################################################################################################################
# LogConfig Class Definition
class LogConfig(object):

    ####################################################################################################################################
    # Define the init method for the LogConfig class
    def __init__(
        self,
        _LogConfig__log_config: logging.config = None,
        _LogConfig__log_config_file_name: str = _LOG_CONFIG_LOG_CONFIG_FILE_NAME,
        _LogConfig__log_config_file_path: str = _LOG_CONFIG_LOG_CONFIG_FILE_PATH,
        _LogConfig__log_file_name: str = _LOG_CONFIG_LOG_FILE_NAME,
        _LogConfig__log_file_path: str = _LOG_CONFIG_LOG_FILE_PATH,
        _LogConfig__log_formatter_string: str = _LOG_CONFIG_LOG_FORMATTER_STRING,
        _LogConfig__log_date_format: str = _LOG_CONFIG_LOG_DATE_FORMAT,
        _LogConfig__log_level: int =_LOG_CONFIG_LOG_LEVEL,
        _LogConfig__log_level_custom_names_list: list[str] = _LOG_CONFIG_LOG_LEVEL_CUSTOM_NAMES_LIST,
        _LogConfig__log_level_logging_config: logging = _LOG_CONFIG_LOG_LEVEL_LOGGING_CONFIG,
        _LogConfig__log_level_methods_dict: dict[str, str] = _LOG_CONFIG_LOG_LEVEL_METHODS_DICT,
        _LogConfig__log_level_methods_list: list[str] = _LOG_CONFIG_LOG_LEVEL_METHODS_LIST,
        _LogConfig__log_level_name: str =_LOG_CONFIG_LOG_LEVEL_NAME,
        _LogConfig__log_level_names_list: list[str] = _LOG_CONFIG_LOG_LEVEL_NAMES_LIST,
        _LogConfig__log_level_numbers_dict: dict[str, int] = _LOG_CONFIG_LOG_LEVEL_NUMBERS_DICT,
        _LogConfig__log_level_numbers_list: list[int] = _LOG_CONFIG_LOG_LEVEL_NUMBERS_LIST,
        _LogConfig__log_level_redefinition: bool = _LOG_CONFIG_LOG_LEVEL_REDEFINITION,
        _LogConfig__log_message_default: str = _LOG_CONFIG_LOG_MESSAGE_DEFAULT,
        _LogConfig__uuid=None,
        **kwargs,
    ):
        super().__init__()

        self.log_config_file_name = (_LogConfig__log_config_file_name, _LOG_CONFIG_LOG_CONFIG_FILE_NAME)[_LogConfig__log_config_file_name is None]
        self.log_config_file_path = (_LogConfig__log_config_file_path, _LOG_CONFIG_LOG_CONFIG_FILE_PATH)[_LogConfig__log_config_file_path is None]
        self.log_date_format = (_LogConfig__log_date_format, _LOG_CONFIG_LOG_DATE_FORMAT)[_LogConfig__log_date_format is None]
        self.log_file_name = _LogConfig__log_file_name or __name__
        self.log_file_path = _LogConfig__log_file_path or str(os.path.join(os.getcwd(), "logs"))
        self.log_file = str(os.path.join(self.log_file_path, self.log_file_name))
        self.log_formatter_string = _LogConfig__log_formatter_string or _LOG_CONFIG_LOG_FORMATTER_STRING
        self.log_level_custom_names_list = _LogConfig__log_level_custom_names_list or _LOG_CONFIG_LOG_LEVEL_CUSTOM_NAMES_LIST
        self.log_level_methods_dict = _LogConfig__log_level_methods_dict or _LOG_CONFIG_LOG_LEVEL_METHODS_DICT
        self.log_level_methods_list = _LogConfig__log_level_methods_list or _LOG_CONFIG_LOG_LEVEL_METHODS_LIST
        self.log_level_names_list = _LogConfig__log_level_names_list or _LOG_CONFIG_LOG_LEVEL_NAMES_LIST
        self.log_level_numbers_dict = _LogConfig__log_level_numbers_dict or _LOG_CONFIG_LOG_LEVEL_NUMBERS_DICT
        self.log_level_numbers_list = _LogConfig__log_level_numbers_list or _LOG_CONFIG_LOG_LEVEL_NUMBERS_LIST
        self.log_level_redefinition = _LogConfig__log_level_redefinition or _LOG_CONFIG_LOG_LEVEL_REDEFINITION
        self.log_message_default = _LogConfig__log_message_default or _LOG_CONFIG_LOG_MESSAGE_DEFAULT
        self.log_level = _LogConfig__log_level or _LOG_CONFIG_LOG_LEVEL
        self.log_level_name = _LogConfig__log_level_name or _LOG_CONFIG_LOG_LEVEL_NAME
        self.log_level_logging_config = _LogConfig__log_level_logging_config or _LOG_CONFIG_LOG_LEVEL_LOGGING_CONFIG
        self.log_level_names_list = _LogConfig__log_level_names_list or _LOG_CONFIG_LOG_LEVEL_NAMES_LIST
        self.log_level_numbers_list = _LogConfig__log_level_numbers_list or _LOG_CONFIG_LOG_LEVEL_NUMBERS_LIST
        Logger.debug(f"Cascor: LogConfig: __init__: Logging Config: Type: {type(_LogConfig__log_config)} Value: {_LogConfig__log_config}")

        # Create a Logger object from the Logger class: Configure the logging system first, then get the actual root logger
        if (custom_logger := Logger(
            _Logger__log_config=_LogConfig__log_config,
            _Logger__log_config_file_name=self.log_config_file_name,
            _Logger__log_config_file_path=self.log_config_file_path,
            _Logger__log_date_format=self.log_date_format,
            _Logger__log_file_name=self.log_file_name,
            _Logger__log_file_path=self.log_file_path,
            _Logger__log_formatter_string=self.log_formatter_string,
            _Logger__log_level_custom_names_list=self.log_level_custom_names_list,
            _Logger__log_level_logging_config=_LogConfig__log_level_logging_config,
            _Logger__log_level_methods_dict=self.log_level_methods_dict,
            _Logger__log_level_methods_list=self.log_level_methods_list,
            _Logger__log_level_name=self.log_level_name,
            _Logger__log_level_names_list=self.log_level_names_list,
            _Logger__log_level_numbers_dict=self.log_level_numbers_dict,
            _Logger__log_level_numbers_list=self.log_level_numbers_list,
            _Logger__log_level_redefinition=self.log_level_redefinition,
            _Logger__log_message_default=self.log_message_default,
            _Logger__uuid=None,
            **kwargs,
        )) is None:
            Logger.critical(f"LogConfig: __init__:  Failed to create Logger class: {custom_logger}")
            os._exit(1)

        # Initialize Log Config class with custom logger: Use the actual root logger which has the correct handlers
        self.custom_logger = custom_logger
        Logger.debug(f"LogConfig: __init__: Custom Logger: Type: {type(self.custom_logger)}, Value: {self.custom_logger}")

        # Get the actual configured root logger instead of the custom logger instance
        self.logger = logging.getLogger(_LOGGER_NAME)  # This gets the properly configured named logger
        
        # Copy the custom logging methods from the custom logger to the named logger
        # TODO: Use method names (lowercase) from methods_dict instead of level names (uppercase) from custom_names_list
        for level_name in self.custom_logger.log_level_custom_names_list:
            method_name = self.custom_logger.log_level_methods_dict[level_name]  # Convert "TRACE" -> "trace"
            if hasattr(self.custom_logger, method_name):
                setattr(self.logger, method_name, getattr(self.custom_logger, method_name).__get__(self.logger, type(self.logger)))
        Logger.debug(f"LogConfig: __init__: Using named logger '{_LOGGER_NAME}': Type: {type(self.logger)}, Value: {self.logger}")
        Logger.debug(f"LogConfig: __init__: Named logger handlers: {self.logger.handlers}")
        Logger.debug(f"LogConfig: __init__: Named logger has trace method: {hasattr(self.logger, 'trace')}")
        Logger.debug(f"LogConfig: __init__: Named logger has verbose method: {hasattr(self.logger, 'verbose')}")

        # setLoggerClass(self.logger)  # Set the Logger class as the default logger class
        setLoggerClass(self.logger.__class__)  # Set the Logger class as the default logger class
        self.logger.info("LogConfig: __init__: Created custom logger by instantiating Logger class")
        self.logger.verbose(f"LogConfig: __init__: Created custom logger: {self.logger}, Type: {type(self.logger)}")

        # Set UUID for the LogConfig class
        self.set_uuid(_LogConfig__uuid)
        self.logger.debug(f"LogConfig: __init__: UUID set to: {self.get_uuid()}")

        self.logger.debug(f"LogConfig: __init__: Default Log Level set to {self.get_log_level()}")
        self.logger.info("LogConfig: __init__: Completed Initialization of LogConfig class")
        self.logger.trace("LogConfig: __init__: Completed the LogConfig class __init__ method")


    # ####################################################################################################################################
    # # Define an abstract method that must be implemented by inheriting subclasses
    # def instantiatable_initialize_subclass(self):
    #     """
    #     This is an abstract base method that must be implemented by subclasses.

    #     Args:
    #         self: The instance of the class.

    #     Returns:
    #         None
    #     """
    #     pass


    ####################################################################################################################################
    # Define private methods for the LogConfig class
    def _generate_uuid(self):
        """
        Description:
            Generate a new UUID for the LogConfig class.
        Args:
            self: The instance of the class.
        Returns:
            str: The generated UUID.
        """
        self.logger.trace("LogConfig: _generate_uuid: Inside the LogConfig class Generate UUID method")
        new_uuid = str(uuid.uuid4())
        self.logger.verbose(f"LogConfig: _generate_uuid: UUID: {new_uuid}")
        self.logger.trace("LogConfig: _generate_uuid: Completed the LogConfig class Generate UUID method")
        return new_uuid


    ####################################################################################################################################
    # Define LogConfig class Setters
    def set_custom_logger(self, custom_logger):
        """
        Description:
            Set the custom logger for the LogConfig class.
        Args:
            custom_logger (logging.Logger): The custom logger to be set.
        Returns:
            None
        """
        self.custom_logger = custom_logger

    def set_logger(self, logger):
        """
        Description:
            Set the logger for the LogConfig class.
        Args:
            logger (logging.Logger): The logger to be set.
        Returns:
            None
        """
        self.logger = logger

    def set_log_file_name(self, log_file_name):
        """
        Description:
            Set the log file name for the LogConfig class.
        Args:
            log_file_name (str): The log file name to be set.
        Returns:
            None
        """
        self.log_file_name = log_file_name

    def set_log_file_path(self, log_file_path):
        """
        Description:
            Set the log file path for the LogConfig class.
        Args:
            log_file_path (str): The log file path to be set.
        Returns:
            None
        """
        self.log_file_path = log_file_path

    def set_log_config_file_name(self, log_config_file_name):
        """
        Description:
            Set the config file name for the LogConfig class.
        Args:
            config_file_name (str): The config file name to be set.
        Returns:
            None
        """
        self.log_config_file_name = log_config_file_name

    def set_log_config_file_path(self, log_config_file_path):
        """
        Description:
            Set the config file path for the LogConfig class.
        Args:
            config_file_path (str): The config file path to be set.
        Returns:
            None
        """
        self.log_config_file_path = log_config_file_path

    def set_log_formatter_string(self, log_formatter_string):
        """
        Description:
            Set the log formatter string for the LogConfig class.
        Args:
            log_formatter_string (str): The log formatter string to be set.
        Returns:
            None
        """
        self.log_formatter_string = log_formatter_string

    def set_log_date_format(self, log_date_format):
        """
        Description:
            Set the log date format for the LogConfig class.
        Args:
            log_date_format (str): The log date format to be set.
        Returns:
            None
        """
        self.log_date_format = log_date_format

    def set_log_message_default(self, log_message_default):
        """
        Description:
            Set the log message default for the LogConfig class.
        Args:
            log_message_default (str): The log message default to be set.
        Returns:
            None
        """
        self.log_message_default = log_message_default

    def set_log_level(self, log_level):
        """
        Description:
            Set the log level for the LogConfig class.
        Args:
            log_level (int): The log level to be set.
        Returns:
            None
        """
        self.log_level = log_level

    def set_log_level_name(self, log_level_name):
        """
        Description:
            Set the log level name for the LogConfig class.
        Args:
            log_level_name (str): The log level name to be set.
        Returns:
            None
        """
        self.log_level_name = log_level_name

    def set_log_level_logging_config(self, log_level_logging_config):
        """
        Description:
            Set the log level for the logging config file.
        Args:
            log_level_logging_config (int): The log level for the logging config file.
        Returns:
            None
        """
        self.log_level_logging_config = log_level_logging_config

    def set_log_level_names_list(self, log_level_names_list):
        """
        Description:
            Set the log level names list for the LogConfig class.
        Args:
            log_level_names_list (list): The log level names list to be set.
        Returns:
            None
        """
        self.log_level_names_list = log_level_names_list

    def set_log_level_numbers_list(self, log_level_numbers_list):
        """
        Description:
            Set the log level numbers list for the LogConfig class.
        Args:
            log_level_numbers_list (list): The log level numbers list to be set.
        Returns:
            None
        """
        self.log_level_numbers_list = log_level_numbers_list

    def set_log_level_methods_list(self, log_level_methods_list):
        """
        Description:
            Set the log level methods list for the LogConfig class.
        Args:
            log_level_methods_list (list): The log level methods list to be set.
        Returns:
            None
        """
        self.log_level_methods_list = log_level_methods_list

    def set_log_level_numbers_dict(self, log_level_numbers_dict):
        """
        Description:
            Set the log level numbers dict for the LogConfig class.
        Args:
            log_level_numbers_dict (dict): The log level numbers dict to be set.
        Returns:
            None
        """
        self.log_level_numbers_dict = log_level_numbers_dict

    def set_log_level_methods_dict(self, log_level_methods_dict):
        """
        Description:
            Set the log level methods dict for the LogConfig class.
        Args:
            log_level_methods_dict (dict): The log level methods dict to be set.
        Returns:
            None
        """
        self.log_level_methods_dict = log_level_methods_dict

    def set_log_level_custom_names_list(self, log_level_custom_names_list):
        """
        Description:
            Set the log level custom names list for the LogConfig class.
        Args:
            log_level_custom_names_list (list): The log level custom names list to be set.
        Returns:
            None
        """
        self.log_level_custom_names_list = log_level_custom_names_list

    def set_log_allow_log_level_redefinition(self, log_allow_log_level_redefinition):
        """
        Description:
            Set whether to allow log level redefinition for the LogConfig class.
        Args:
            log_allow_log_level_redefinition (bool): Whether to allow log level redefinition.
        Returns:
            None
        """
        self.log_allow_log_level_redefinition = log_allow_log_level_redefinition

    def set_uuid(self, uuid=None):
        """
        Description:
            This method sets the UUID for the LogConfig class.  If no UUID is provided, a new UUID will be generated.
        Args:
            uuid (str): The UUID to be set. If None, a new UUID will be generated.
        Returns:
            None
        """
        self.logger.trace("LogConfig: set_uuid: Starting to set UUID for LogConfig class")
        if not hasattr(self, "uuid") or self.uuid is None:
            self.uuid = (uuid, self._generate_uuid())[uuid is None]  # Generate a new UUID if none is provided
        else:
            self.logger.fatal(f"LogConfig: set_uuid: Fatal Error: UUID already set: {self.uuid}. Changing UUID is bad Juju.  Exiting...")
            os._exit(1)
        self.logger.verbose(f"LogConfig: set_uuid: UUID set to: {self.uuid}")
        self.logger.trace("LogConfig: set_uuid: Completed setting UUID for LogConfig class")


    ####################################################################################################################################
    # Define LogConfig class Getters
    def get_uuid(self):
        """
        Description:
            This method returns the UUID for the LogConfig class.
        Args:
            self: The instance of the class.
        Returns:
            str: The UUID for the LogConfig class.
        """
        if not hasattr(self, "uuid") or self.uuid is None:
            self.set_uuid()  # Ensure UUID is set if not already
            self.logger.debug("LogConfig: get_uuid: UUID was not set, generated a new one.")
        self.logger.verbose(f"LogConfig: get_uuid: Returning UUID: {self.uuid}")
        self.logger.trace("LogConfig: get_uuid: Completed getting UUID for LogConfig class")
        return self.uuid
    
    def get_custom_logger(self):
        """
        Description:
            Returns the custom logger for the LogConfig class.
        Args:
            self: The instance of the class.
        Returns:
            self.custom_logger: The custom logger instance of the LogConfig class to be returned.
        """
        return self.custom_logger if hasattr(self, "custom_logger") else None

    def get_logger(self):
        """ 
        Description:
            Returns the logger for the LogConfig class.
        Args:
            self: The instance of the class.
        Returns:
            self.logger: The custom logger instance of the LogConfig class to be returned.
        """
        return self.logger if hasattr(self, "logger") else None

    def get_log_file_name(self):
        """ 
        Description:
            Returns the log file name for the LogConfig class.
        Args:
            self: The instance of the class.
        Returns:
            str: The log file name for the LogConfig class.
        """
        return self.log_file_name if hasattr(self, "log_file_name") else None

    def get_log_file_path(self):
        """
        Description:
            Returns the log file path for the LogConfig class.
        Args:
            self: The instance of the class.
        Returns:
            str: The log file path for the LogConfig class.
        """
        return self.log_file_path if hasattr(self, "log_file_path") else None

    def get_log_config_file_name(self):
        """
        Description:
            Returns the log config file name for the LogConfig class.
        Args:
            self: The instance of the class.
        Returns:
            str: The path to the logging config file.
        """
        return self.log_config_file_name if hasattr(self, "log_config_file_name") else None

    def get_log_config_file_path(self):
        """
        Description:
            Returns the log config file path for the LogConfig class.
        Args:
            self: The instance of the class.
        Returns:
            str: The path to the logging config file.
        """
        return self.log_config_file_path if hasattr(self, "log_config_file_path") else None

    def get_log_formatter_string(self):
        """
        Description:
            Returns the log formatter string for the LogConfig class.
        Args:
            self: The instance of the class.
        Returns:
            str: The log formatter string to be set.
        """
        return self.log_formatter_string if hasattr(self, "log_formatter_string") else None

    def get_log_date_format(self):
        """
        Description:
            Returns the log date format for the LogConfig class.
        Args:
            self: The instance of the class.
        Returns:
            str: The log date format to be set.
        """
        return self.log_date_format if hasattr(self, "log_date_format") else None

    def get_log_message_default(self):
        """
        Description:
            Returns the log message default for the LogConfig class.
        Args:
            self: The instance of the class.
        Returns:
            str: The log message default to be set.
        """
        return self.log_message_default if hasattr(self, "log_message_default") else None

    def get_log_level(self):
        """ 
        Description:
            Returns the log level for the LogConfig class.
        Args:
            self: The instance of the class.
        Returns:
            int: The log level to be set.
        """
        return self.log_level if hasattr(self, "log_level") else None

    def get_log_level_name(self):
        """
        Description:
            Returns the log level name for the LogConfig class.
        Args:
            self: The instance of the class.
        Returns:
            str: The log level name to be set.
        """
        return self.log_level_name if hasattr(self, "log_level_name") else None

    def get_log_level_logging_config(self):
        """
        Description:
            Returns the log level for the logging config file.
        Args:
            self: The instance of the class.
        Returns:
            int: The log level for the logging config file.
        """
        return self.log_level_logging_config if hasattr(self, "log_level_logging_config") else None

    def get_log_level_names_list(self):
        """
        Description:
            Returns the log level names list for the LogConfig class.
        Args:
            self: The instance of the class.
        Returns:
            list: The log level names list to be set.
        """
        return self.log_level_names_list if hasattr(self, "log_level_names_list") else None

    def get_log_level_numbers_list(self):
        """
        Description:
            Returns the log level numbers list for the LogConfig class.
        Args:
            self: The instance of the class.
        Returns:
            list: The log level numbers list to be set.
        """
        return self.log_level_numbers_list if hasattr(self, "log_level_numbers_list") else None

    def get_log_level_methods_list(self):
        """
        Description:
            Returns the log level methods list for the LogConfig class.
        Args:
            self: The instance of the class.
        Returns:
            list: The log level methods list to be set.
        """
        return self.log_level_methods_list if hasattr(self, "log_level_methods_list") else None

    def get_log_level_numbers_dict(self):
        """
        Description:
            Returns the log level numbers dictionary for the LogConfig class.
        Args:
            self: The instance of the class.
        Returns:
            dict: The log level numbers dictionary to be set.
        """
        return self.log_level_numbers_dict if hasattr(self, "log_level_numbers_dict") else None

    def get_log_level_methods_dict(self):
        """
        Description:
            Returns the log level methods dictionary for the LogConfig class.
        Args:
            self: The instance of the class.
        Returns:
            dict: The log level methods dictionary to be set.
        """
        return self.log_level_methods_dict if hasattr(self, "log_level_methods_dict") else None

    def get_log_level_custom_names_list(self):
        """
        Description:
            Returns the log level custom names list for the LogConfig class.
        Args:
            self: The instance of the class.
        Returns:
            list: The log level custom names list to be set.
        """
        return self.log_level_custom_names_list if hasattr(self, "log_level_custom_names_list") else None

    def get_log_allow_log_level_redefinition(self):
        """
        Description:
            Returns whether to allow log level redefinition for the LogConfig class.
        Args:
            self: The instance of the class.
        Returns:
            bool: Whether to allow log level redefinition.
        """
        return self.log_allow_log_level_redefinition if hasattr(self, "log_allow_log_level_redefinition") else None