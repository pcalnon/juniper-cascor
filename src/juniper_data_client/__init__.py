#!/usr/bin/env python
"""
Project:       Juniper
Prototype:     Cascade Correlation Neural Network
File Name:     __init__.py
Author:        Paul Calnon
Version:       0.3.18 (0.7.4)

Date Created:  2026-01-29
Last Modified: 2026-02-12

License:       MIT License
Copyright:     Copyright (c) 2024-2026 Paul Calnon

Description:
    Package initialization for juniper_data_client module.
    Re-exports from the shared juniper-data-client package (DATA-012).
    Falls back to local implementation if shared package is not installed.
"""

try:
    from juniper_data_client import JuniperDataClient, JuniperDataClientError, JuniperDataConfigurationError, JuniperDataConnectionError, JuniperDataNotFoundError, JuniperDataTimeoutError, JuniperDataValidationError, __version__

    _USING_SHARED_PACKAGE = True
except ImportError:
    from .client import JuniperDataClient

    _USING_SHARED_PACKAGE = False
    __version__ = "0.3.18-local"

    class JuniperDataClientError(Exception):
        """Base exception for all JuniperData client errors."""

        pass

    class JuniperDataConnectionError(JuniperDataClientError):
        """Raised when connection to JuniperData service fails."""

        pass

    class JuniperDataTimeoutError(JuniperDataClientError):
        """Raised when a request to JuniperData times out."""

        pass

    class JuniperDataNotFoundError(JuniperDataClientError):
        """Raised when a requested resource is not found (404)."""

        pass

    class JuniperDataValidationError(JuniperDataClientError):
        """Raised when request parameters fail validation (400/422)."""

        pass

    class JuniperDataConfigurationError(JuniperDataClientError):
        """Raised when JuniperData configuration is missing or invalid."""

        pass


__all__ = [
    "JuniperDataClient",
    "JuniperDataClientError",
    "JuniperDataConfigurationError",
    "JuniperDataConnectionError",
    "JuniperDataNotFoundError",
    "JuniperDataTimeoutError",
    "JuniperDataValidationError",
    "__version__",
    "_USING_SHARED_PACKAGE",
]
