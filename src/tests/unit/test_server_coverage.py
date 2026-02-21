#!/usr/bin/env python
"""
Unit tests for server.py to improve code coverage.

Tests cover:
- main() function with mocked dependencies (uvicorn, settings, app)
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

pytestmark = pytest.mark.unit


class TestServerMain:
    """Tests for server.main() function."""

    def test_main_calls_uvicorn_run(self):
        """main() should call uvicorn.run with the app and settings."""
        mock_settings = MagicMock()
        mock_settings.host = "127.0.0.1"
        mock_settings.port = 8200
        mock_settings.log_level = "INFO"

        mock_app = MagicMock()

        with patch("server.get_settings", return_value=mock_settings) as mock_get_settings, patch("server.create_app", return_value=mock_app) as mock_create_app, patch("server.uvicorn.run") as mock_uvicorn_run:
            from server import main

            main()

            mock_get_settings.assert_called_once()
            mock_create_app.assert_called_once_with(mock_settings)
            mock_uvicorn_run.assert_called_once_with(
                mock_app,
                host="127.0.0.1",
                port=8200,
                log_level="info",
            )

    def test_main_lowercases_log_level(self):
        """main() should lowercase the log level from settings."""
        mock_settings = MagicMock()
        mock_settings.host = "0.0.0.0"
        mock_settings.port = 9000
        mock_settings.log_level = "WARNING"

        with patch("server.get_settings", return_value=mock_settings), patch("server.create_app", return_value=MagicMock()), patch("server.uvicorn.run") as mock_uvicorn_run:
            from server import main

            main()

            call_kwargs = mock_uvicorn_run.call_args
            assert call_kwargs[1]["log_level"] == "warning" or call_kwargs.kwargs["log_level"] == "warning"

    def test_main_uses_settings_host_and_port(self):
        """main() should pass host and port from settings to uvicorn."""
        mock_settings = MagicMock()
        mock_settings.host = "192.168.1.100"
        mock_settings.port = 3000
        mock_settings.log_level = "DEBUG"

        with patch("server.get_settings", return_value=mock_settings), patch("server.create_app", return_value=MagicMock()), patch("server.uvicorn.run") as mock_uvicorn_run:
            from server import main

            main()

            _, kwargs = mock_uvicorn_run.call_args
            assert kwargs["host"] == "192.168.1.100"
            assert kwargs["port"] == 3000
