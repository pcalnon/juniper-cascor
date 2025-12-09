#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     constants_model.py
# Author:        Paul Calnon
# Version:       0.3.1 (0.7.3)
#
# Date:          2025-08-27
# Last Modified: 2025-09-05
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
#
# Description:
#    This file contains constants used in the Cascade Correlation Neural Network implementation.
#
#####################################################################################################################################################################################################
# Notes:
#    - This file contains constants that are used throughout the Cascade Correlation Neural Network implementation.
#    - These constants include logger field names, format strings, and other configuration parameters.
#    - The constants are organized into sections for easy reference and modification.
#    - The logger constants define the structure and format of log messages, including field names and format strings for console and file logging.
#    - The constants are defined using descriptive variable names to enhance code readability and maintainability.
#
#
#         format_spec ::= [[fill]align][sign]["z"]["#"]["0"][width][grouping_option]["." precision][type]
#
#
#         Component 	    Description 	                                                                                    Possible Values
#
#         fill 	            Specifies the character to use for padding values that don’t occupy the entire field width 	        Any character
#         align 	        Specifies how to justify values that don’t occupy the entire field width 	                        <, >, =, or ^
#         sign 	            Controls whether a leading sign is included for numeric values 	                                    +, -, or a space
#         z 	            Coerces negative zeros 	                                                                            z
#         # 	            Selects an alternate output form for certain presentation types, such as integers 	                #
#         0 	            Causes values to be padded on the left with zeros instead of ASCII space characters 	            0
#         width 	        Specifies the minimum width of the output 	                                                        Integer value
#         grouping_option 	Specifies a grouping character for numeric output 	                                                _ or ,
#         precision 	    Specifies the number of digits after the decimal point for floating-point presentation types, 
#                             and the maximum output width for string presentations types 	                                Integer value
#         type 	            Specifies the presentation type, the type of conversion performed on the corresponding argument 	b, c, d, e, E, f, F, g, G, n, o, s, x, X, or %
#
#
#         Value 	Presentation Type
#         b 	Binary integer
#         c 	Unicode character
#         d 	Decimal integer
#         e or E 	Exponential
#         f or F 	Floating-point number
#         g or G 	Floating-point or exponential number
#         n 	Decimal integer
#         o 	Octal integer
#         s 	String
#         x or X 	Hexadecimal integer
#         % 	Percentage value
#
#####################################################################################################################################################################################################
# References:
#     - https://realpython.com/python-formatted-output/
#     - https://docs.python.org/3/library/logging.html
#     - https://pyformat.info/
#
#####################################################################################################################################################################################################
# TODO:
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#
#####################################################################################################################################################################################################
import logging



# #####################################################################################################################################################################################################

# # TODO: Integrate this block into Constants files
# _LOG_CONFIG_DEFAULT = {
#     "version": 1,
#     "disable_existing_loggers": False,
#     "formatters": {
#         "default": {
#             "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#             "datefmt": "%Y-%m-%d %H:%M:%S",
#         },
#     },
#     "handlers": {
#         "console": {
#             "class": "logging.StreamHandler",
#             "level": _CASCOR_SPIRAL_DATASET_LOG_LEVEL_DEFAULT,
#             "formatter": "default",
#         },
#     },
#     "loggers": {
#         _CASCOR_SPIRAL_DATASET_LOG_NAME: {
#             "level": _CASCOR_SPIRAL_DATASET_LOG_LEVEL_DEFAULT,
#             "handlers": ["console"],
#             "propagate": False,
#         },
#     },
#     "root": {
#         "level": _CASCOR_SPIRAL_DATASET_LOG_LEVEL_DEFAULT,
#         "handlers": ["console"],
#     },
# }

# #####################################################################################################################################################################################################



#####################################################################################################################################################################################################
# Define logging format strings and field names for console and file logging
#####################################################################################################################################################################################################

#####################################################################################################################################################################################################
# Define logging level for class logging
_LOGGER_CLASS_LOGGER_LOG_LEVEL_DEFAULT = logging.DEBUG
# _LOGGER_CLASS_LOGGER_LOG_LEVEL_DEFAULT = logging.INFO
# _LOGGER_CLASS_LOGGER_LOG_LEVEL_DEFAULT = logging.WARNING
# _LOGGER_CLASS_LOGGER_LOG_LEVEL_DEFAULT = logging.ERROR
# _LOGGER_CLASS_LOGGER_LOG_LEVEL_DEFAULT = logging.CRITICAL


#####################################################################################################################################################################################################
# Define logging field names
_LOGGER_PREFIX_FIELD_NAME_FILENAME = "filename"
_LOGGER_PREFIX_FIELD_NAME_LINE_NUMBER = "lineno"
# _LOGGER_PREFIX_FIELD_NAME_FUNCTION_NAME = "funcName"
_LOGGER_PREFIX_FIELD_NAME_FUNCTION_NAME = "function"
_LOGGER_PREFIX_FIELD_NAME_ASCTIME = "asctime"

_LOGGER_CONTENT_FIELD_NAME_LEVELNAME = "levelname"
_LOGGER_CONTENT_FIELD_NAME_MESSAGE = "message"

_LOGGER_PREFIX_FRAME_FILE_NAME = _LOGGER_PREFIX_FIELD_NAME_FILENAME
_LOGGER_PREFIX_FRAME_LINE_NAME = _LOGGER_PREFIX_FIELD_NAME_LINE_NUMBER
_LOGGER_PREFIX_FRAME_FUNC_NAME = _LOGGER_PREFIX_FIELD_NAME_FUNCTION_NAME


#####################################################################################################################################################################################################
# Define logging format string components
_LOGGER_PREFIX_FILENAME = f"%({_LOGGER_PREFIX_FIELD_NAME_FILENAME})s"
_LOGGER_PREFIX_LINE_NUMBER = f"%({_LOGGER_PREFIX_FIELD_NAME_LINE_NUMBER})d"
_LOGGER_PREFIX_FUNCTION_NAME = f"%({_LOGGER_PREFIX_FIELD_NAME_FUNCTION_NAME})s"
_LOGGER_PREFIX_ASCTIME = f"%({_LOGGER_PREFIX_FIELD_NAME_ASCTIME})s"
_LOGGER_CONTENT_LEVELNAME = f"%({_LOGGER_CONTENT_FIELD_NAME_LEVELNAME})s"
_LOGGER_CONTENT_MESSAGE = f"%({_LOGGER_CONTENT_FIELD_NAME_MESSAGE})s"


#####################################################################################################################################################################################################
# Define formatting for fileT and console logging
#     "[%(filename)s: %(funcName)s:%(lineno)d] (%(asctime)s) [%(levelname)s] %(message)s"
#     "[%(filename)s:%(lineno)d] (%(asctime)s) [%(levelname)s] %(message)s"

_LOGGER_LOG_FORMATTER_STRING_FILE_PREFIX = f"[{_LOGGER_PREFIX_FILENAME}: {_LOGGER_PREFIX_FUNCTION_NAME}:{_LOGGER_PREFIX_LINE_NUMBER}] ({_LOGGER_PREFIX_ASCTIME}) "
_LOGGER_LOG_FORMATTER_STRING_FILE_CONTENT = f"[{_LOGGER_CONTENT_LEVELNAME}] {_LOGGER_CONTENT_MESSAGE}"
_LOGGER_LOG_FORMATTER_STRING_FILE = f"{_LOGGER_LOG_FORMATTER_STRING_FILE_PREFIX}{_LOGGER_LOG_FORMATTER_STRING_FILE_CONTENT}"

_LOGGER_LOG_FORMATTER_STRING_CONSOLE_PREFIX = f"[{_LOGGER_PREFIX_FILENAME}: {_LOGGER_PREFIX_LINE_NUMBER}] ({_LOGGER_PREFIX_ASCTIME}) "
_LOGGER_LOG_FORMATTER_STRING_CONSOLE_CONTENT = f"[{_LOGGER_CONTENT_LEVELNAME}] {_LOGGER_CONTENT_MESSAGE}"
_LOGGER_LOG_FORMATTER_STRING_CONSOLE = f"{_LOGGER_LOG_FORMATTER_STRING_CONSOLE_PREFIX}{_LOGGER_LOG_FORMATTER_STRING_CONSOLE_CONTENT}"



#####################################################################################################################################################################################################
# Define format_spec Option Values:  [[fill]align][sign]["z"]["#"]["0"][width][grouping_option]["." precision][type]
_LOGGER_NEGATIVE_ZERO_OPTION_LIST = [ "z" ]
_LOGGER_ALTERNATE_FORM_OPTION_LIST = [ "#" ]
_LOGGER_ZERO_PAD_OPTION_LIST = [ "0" ]
_LOGGER_SIGN_OPTION_LIST = [ "+", "-", " " ]
_LOGGER_ALIGN_OPTION_LIST = [ "<", ">", "=", "^" ]
_LOGGER_GROUPING_OPTION_LIST = [ "_", "," ]
_LOGGER_NUMERIC_TYPE_LIST = [ "b", "c", "d", "e", "E", "f", "F", "g", "G", "n", "o", "s", "x", "X", "%" ]


#####################################################################################################################################################################################################
# Define constants for names of logging destinations
_LOGGER_DESTINATION_NAME_CONSOLE = "console"
_LOGGER_DESTINATION_NAME_FILE = "file"


#####################################################################################################################################################################################################
# Define constants for names of logging message sections
_LOGGER_PREFIX_NAME = "prefix"
_LOGGER_CONTENT_NAME = "content"



#####################################################################################################################################################################################################
# Logging field names for console and file logging
#####################################################################################################################################################################################################

#####################################################################################################################################################################################################
# Logging field names complete list
_LOGGER_DATA_FIELD_NAMES = [ _LOGGER_PREFIX_FIELD_NAME_FILENAME, _LOGGER_PREFIX_FIELD_NAME_LINE_NUMBER, _LOGGER_PREFIX_FIELD_NAME_FUNCTION_NAME, _LOGGER_PREFIX_FIELD_NAME_ASCTIME, _LOGGER_CONTENT_FIELD_NAME_LEVELNAME, _LOGGER_CONTENT_FIELD_NAME_MESSAGE, ]
_LOGGER_FORMAT_FIELD_NAMES = [ _LOGGER_PREFIX_FILENAME, _LOGGER_PREFIX_LINE_NUMBER, _LOGGER_PREFIX_FUNCTION_NAME, _LOGGER_PREFIX_ASCTIME, _LOGGER_CONTENT_LEVELNAME, _LOGGER_CONTENT_MESSAGE, ]

_LOGGER_FIELD_NAMES_DICT = dict(zip(_LOGGER_DATA_FIELD_NAMES, _LOGGER_FORMAT_FIELD_NAMES, strict=False))


#####################################################################################################################################################################################################
#  Console logging Prefix field names
_LOGGER_PREFIX_FIELD_NAMES_CONSOLE = [_LOGGER_PREFIX_FIELD_NAME_FILENAME, _LOGGER_PREFIX_FIELD_NAME_LINE_NUMBER, _LOGGER_PREFIX_FIELD_NAME_ASCTIME]
_LOGGER_PREFIX_FIELDS_CONSOLE = [_LOGGER_PREFIX_FILENAME, _LOGGER_PREFIX_LINE_NUMBER, _LOGGER_PREFIX_ASCTIME]

_LOGGER_PREFIX_CONSOLE_DICT = dict(zip(_LOGGER_PREFIX_FIELD_NAMES_CONSOLE, _LOGGER_PREFIX_FIELDS_CONSOLE, strict=False))


#####################################################################################################################################################################################################
# Console logging Content field names
_LOGGER_CONTENT_FIELD_NAMES_CONSOLE = [_LOGGER_CONTENT_FIELD_NAME_LEVELNAME, _LOGGER_CONTENT_FIELD_NAME_MESSAGE]
_LOGGER_CONTENT_FIELDS_CONSOLE = [_LOGGER_CONTENT_LEVELNAME, _LOGGER_CONTENT_MESSAGE]

_LOGGER_CONTENT_CONSOLE_DICT = dict(zip(_LOGGER_CONTENT_FIELD_NAMES_CONSOLE, _LOGGER_CONTENT_FIELDS_CONSOLE, strict=False))


#######################################################################################################################################################################################################
# File logging Prefix field names
_LOGGER_PREFIX_FIELD_NAMES_FILE = [_LOGGER_PREFIX_FIELD_NAME_FILENAME, _LOGGER_PREFIX_FIELD_NAME_LINE_NUMBER, _LOGGER_PREFIX_FIELD_NAME_FUNCTION_NAME, _LOGGER_PREFIX_FIELD_NAME_ASCTIME]
_LOGGER_PREFIX_FIELDS_FILE = [_LOGGER_PREFIX_FILENAME, _LOGGER_PREFIX_LINE_NUMBER, _LOGGER_PREFIX_FUNCTION_NAME, _LOGGER_PREFIX_ASCTIME]

_LOGGER_PREFIX_FILE_DICT = dict(zip(_LOGGER_PREFIX_FIELD_NAMES_FILE, _LOGGER_PREFIX_FIELDS_FILE, strict=False))


#####################################################################################################################################################################################################
# File logging Content field names
_LOGGER_CONTENT_FIELD_NAMES_FILE = [_LOGGER_CONTENT_FIELD_NAME_LEVELNAME, _LOGGER_CONTENT_FIELD_NAME_MESSAGE]
_LOGGER_CONTENT_FIELDS_FILE = [_LOGGER_CONTENT_LEVELNAME, _LOGGER_CONTENT_MESSAGE]

_LOGGER_CONTENT_FILE_DICT = dict(zip(_LOGGER_CONTENT_FIELD_NAMES_FILE, _LOGGER_CONTENT_FIELDS_FILE, strict=False))



#####################################################################################################################################################################################################
# Logging format strings for console and file logging
#####################################################################################################################################################################################################

#####################################################################################################################################################################################################
#  Define the log format strings for console logging Prefix
_LOGGER_PREFIX_FORMAT_CONSOLE= f"[{_LOGGER_PREFIX_FILENAME}:{_LOGGER_PREFIX_LINE_NUMBER}] ({_LOGGER_PREFIX_ASCTIME})"

_LOGGER_PREFIX_CONSOLE_FORMAT_DICT = dict(zip(_LOGGER_PREFIX_FIELD_NAMES_CONSOLE, _LOGGER_PREFIX_FIELDS_CONSOLE, strict=False))


#####################################################################################################################################################################################################
# Define the log format strings for console logging content
_LOGGER_CONTENT_FORMAT_CONSOLE=f"[{_LOGGER_CONTENT_LEVELNAME}] {_LOGGER_CONTENT_MESSAGE}"

_LOGGER_CONTENT_CONSOLE_FORMAT_DICT = dict(zip(_LOGGER_CONTENT_FIELD_NAMES_CONSOLE, _LOGGER_CONTENT_FORMAT_CONSOLE, strict=False))


#####################################################################################################################################################################################################
# Define the log format strings for file logging Prefix
_LOGGER_PREFIX_FORMAT_FILE= f"[{_LOGGER_PREFIX_FILENAME}:{_LOGGER_PREFIX_LINE_NUMBER} - {_LOGGER_PREFIX_FUNCTION_NAME}] ({_LOGGER_PREFIX_ASCTIME})"

_LOGGER_PREFIX_FILE_FORMAT_DICT = dict(zip(_LOGGER_PREFIX_FIELD_NAMES_FILE, _LOGGER_PREFIX_FORMAT_FILE, strict=False))


#####################################################################################################################################################################################################
# Define the log format strings for file logging Content
_LOGGER_CONTENT_FORMAT_FILE=f"[{_LOGGER_CONTENT_LEVELNAME}] {_LOGGER_CONTENT_MESSAGE}"

_LOGGER_CONTENT_FILE_FORMAT_DICT = dict(zip(_LOGGER_CONTENT_FIELD_NAMES_FILE, _LOGGER_CONTENT_FORMAT_FILE, strict=False))

