#!/usr/bin/env python
"""
Unit tests for main.py to improve code coverage.

Tests cover:
- parse_args() function with various flag combinations
- main() function with mocked dependencies
- Error handling paths (os._exit calls)
"""

import argparse
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

pytestmark = pytest.mark.unit


class TestParseArgs:
    """Tests for parse_args function."""

    def test_parse_args_default_values(self):
        """Test parse_args with no arguments returns default values."""
        with patch("sys.argv", ["main.py"]):
            from main import parse_args

            args = parse_args()

            assert args.profile is False
            assert args.profile_memory is False
            assert args.profile_output == "./profiles"
            assert args.profile_top_n == 30

    def test_parse_args_with_profile_flag(self):
        """Test parse_args with --profile flag."""
        with patch("sys.argv", ["main.py", "--profile"]):
            from main import parse_args

            args = parse_args()

            assert args.profile is True
            assert args.profile_memory is False

    def test_parse_args_with_profile_memory_flag(self):
        """Test parse_args with --profile-memory flag."""
        with patch("sys.argv", ["main.py", "--profile-memory"]):
            from main import parse_args

            args = parse_args()

            assert args.profile is False
            assert args.profile_memory is True

    def test_parse_args_with_profile_output(self):
        """Test parse_args with --profile-output flag."""
        with patch("sys.argv", ["main.py", "--profile-output", "/custom/path"]):
            from main import parse_args

            args = parse_args()

            assert args.profile_output == "/custom/path"

    def test_parse_args_with_profile_top_n(self):
        """Test parse_args with --profile-top-n flag."""
        with patch("sys.argv", ["main.py", "--profile-top-n", "50"]):
            from main import parse_args

            args = parse_args()

            assert args.profile_top_n == 50

    def test_parse_args_with_all_profile_flags(self):
        """Test parse_args with all profiling flags combined."""
        with patch(
            "sys.argv",
            [
                "main.py",
                "--profile",
                "--profile-output",
                "/my/profiles",
                "--profile-top-n",
                "100",
            ],
        ):
            from main import parse_args

            args = parse_args()

            assert args.profile is True
            assert args.profile_memory is False
            assert args.profile_output == "/my/profiles"
            assert args.profile_top_n == 100

    def test_parse_args_memory_with_options(self):
        """Test parse_args with --profile-memory and additional options."""
        with patch(
            "sys.argv",
            [
                "main.py",
                "--profile-memory",
                "--profile-output",
                "./mem_profiles",
                "--profile-top-n",
                "25",
            ],
        ):
            from main import parse_args

            args = parse_args()

            assert args.profile_memory is True
            assert args.profile is False
            assert args.profile_output == "./mem_profiles"
            assert args.profile_top_n == 25


class TestMainFunction:
    """Tests for main() function."""

    @pytest.fixture
    def mock_dependencies(self):
        """Set up common mocks for main function."""
        with patch("main.Logger") as mock_logger_class, patch("main.LogConfig") as mock_log_config_class, patch("main.SpiralProblem") as mock_spiral_problem_class:
            mock_logger = MagicMock()
            mock_logger_class.info = MagicMock()
            mock_logger_class.debug = MagicMock()
            mock_logger_class.error = MagicMock()

            mock_log_config = MagicMock()
            mock_log_config_class.return_value = mock_log_config

            mock_logger_instance = MagicMock()
            mock_logger_instance.info = MagicMock()
            mock_logger_instance.debug = MagicMock()
            mock_logger_instance.verbose = MagicMock()
            mock_log_config.get_logger.return_value = mock_logger_instance
            mock_log_config.get_log_level.return_value = 20
            mock_log_config.get_log_level_name.return_value = "INFO"

            mock_spiral_problem = MagicMock()
            mock_spiral_problem_class.return_value = mock_spiral_problem
            mock_spiral_problem.evaluate = MagicMock()

            yield {
                "logger_class": mock_logger_class,
                "log_config_class": mock_log_config_class,
                "log_config": mock_log_config,
                "logger_instance": mock_logger_instance,
                "spiral_problem_class": mock_spiral_problem_class,
                "spiral_problem": mock_spiral_problem,
            }

    def test_main_happy_path(self, mock_dependencies):
        """Test main() happy path - successful execution."""
        from main import main

        main()

        mock_dependencies["log_config_class"].assert_called_once()
        mock_dependencies["log_config"].get_logger.assert_called_once()
        mock_dependencies["spiral_problem_class"].assert_called_once()
        mock_dependencies["spiral_problem"].evaluate.assert_called_once()

    def test_main_log_config_returns_none(self):
        """Test main() when LogConfig returns None - should call os._exit(1)."""
        with patch("main.Logger") as mock_logger_class, patch("main.LogConfig") as mock_log_config_class, patch("main.os._exit") as mock_exit:
            mock_logger_class.info = MagicMock()
            mock_logger_class.debug = MagicMock()
            mock_logger_class.error = MagicMock()

            mock_log_config_class.return_value = None
            mock_exit.side_effect = SystemExit(1)

            from main import main

            with pytest.raises(SystemExit) as exc_info:
                main()

            mock_exit.assert_called_once_with(1)
            assert exc_info.value.code == 1

    def test_main_get_logger_returns_none(self):
        """Test main() when log_config.get_logger() returns None - should call os._exit(2)."""
        with patch("main.Logger") as mock_logger_class, patch("main.LogConfig") as mock_log_config_class, patch("main.os._exit") as mock_exit:
            mock_logger_class.info = MagicMock()
            mock_logger_class.debug = MagicMock()
            mock_logger_class.error = MagicMock()

            mock_log_config = MagicMock()
            mock_log_config_class.return_value = mock_log_config
            mock_log_config.get_logger.return_value = None
            mock_exit.side_effect = SystemExit(2)

            from main import main

            with pytest.raises(SystemExit) as exc_info:
                main()

            mock_exit.assert_called_once_with(2)
            assert exc_info.value.code == 2

    def test_main_logs_startup_messages(self, mock_dependencies):
        """Test that main() logs appropriate startup messages."""
        from main import main

        main()

        mock_dependencies["logger_class"].info.assert_called()
        calls = mock_dependencies["logger_class"].info.call_args_list
        call_messages = [str(call) for call in calls]
        assert any("Starting" in msg for msg in call_messages)

    def test_main_creates_spiral_problem_with_correct_params(self, mock_dependencies):
        """Test that main() creates SpiralProblem with expected parameters."""
        from main import main

        main()

        call_kwargs = mock_dependencies["spiral_problem_class"].call_args[1]
        assert "_SpiralProblem__spiral_config" in call_kwargs
        assert "_SpiralProblem__input_size" in call_kwargs
        assert "_SpiralProblem__output_size" in call_kwargs
        assert "_SpiralProblem__learning_rate" in call_kwargs

    def test_main_calls_evaluate_with_correct_params(self, mock_dependencies):
        """Test that main() calls sp.evaluate with expected parameters."""
        from main import main

        main()

        call_kwargs = mock_dependencies["spiral_problem"].evaluate.call_args[1]
        assert "n_points" in call_kwargs
        assert "n_spirals" in call_kwargs
        assert "n_rotations" in call_kwargs
        assert "clockwise" in call_kwargs
        assert "train_ratio" in call_kwargs
        assert "test_ratio" in call_kwargs


class TestMainFunctionLogging:
    """Tests for main() function logging behavior."""

    def test_main_logs_log_config_creation_success(self):
        """Test that main() logs successful LogConfig creation."""
        with patch("main.Logger") as mock_logger_class, patch("main.LogConfig") as mock_log_config_class, patch("main.SpiralProblem") as mock_spiral_problem_class:
            mock_logger_class.info = MagicMock()
            mock_logger_class.debug = MagicMock()
            mock_logger_class.error = MagicMock()

            mock_log_config = MagicMock()
            mock_log_config_class.return_value = mock_log_config

            mock_logger_instance = MagicMock()
            mock_log_config.get_logger.return_value = mock_logger_instance
            mock_log_config.get_log_level.return_value = 20
            mock_log_config.get_log_level_name.return_value = "INFO"

            mock_spiral_problem = MagicMock()
            mock_spiral_problem_class.return_value = mock_spiral_problem

            from main import main

            main()

            assert mock_logger_class.debug.called

    def test_main_logs_error_on_log_config_failure(self):
        """Test that main() logs error when LogConfig creation fails."""
        with patch("main.Logger") as mock_logger_class, patch("main.LogConfig") as mock_log_config_class, patch("main.os._exit") as mock_exit:
            mock_logger_class.info = MagicMock()
            mock_logger_class.debug = MagicMock()
            mock_logger_class.error = MagicMock()

            mock_log_config_class.return_value = None
            mock_exit.side_effect = SystemExit(1)

            from main import main

            with pytest.raises(SystemExit):
                main()

            mock_logger_class.error.assert_called()
            error_calls = [str(call) for call in mock_logger_class.error.call_args_list]
            assert any("Failed to create LogConfig" in msg for msg in error_calls)

    def test_main_logs_error_on_get_logger_failure(self):
        """Test that main() logs error when get_logger returns None."""
        with patch("main.Logger") as mock_logger_class, patch("main.LogConfig") as mock_log_config_class, patch("main.os._exit") as mock_exit:
            mock_logger_class.info = MagicMock()
            mock_logger_class.debug = MagicMock()
            mock_logger_class.error = MagicMock()

            mock_log_config = MagicMock()
            mock_log_config_class.return_value = mock_log_config
            mock_log_config.get_logger.return_value = None
            mock_exit.side_effect = SystemExit(2)

            from main import main

            with pytest.raises(SystemExit):
                main()

            mock_logger_class.error.assert_called()
            error_calls = [str(call) for call in mock_logger_class.error.call_args_list]
            assert any("Failed to get Logger" in msg for msg in error_calls)


class TestParseArgsArgumentParser:
    """Tests for parse_args argument parser configuration."""

    def test_parse_args_help_text(self):
        """Test that parse_args configures proper help text."""
        with patch("sys.argv", ["main.py", "--help"]):
            from main import parse_args

            with pytest.raises(SystemExit) as exc_info:
                parse_args()

            assert exc_info.value.code == 0

    def test_parse_args_invalid_top_n_type(self):
        """Test parse_args with invalid --profile-top-n type."""
        with patch("sys.argv", ["main.py", "--profile-top-n", "not_a_number"]):
            from main import parse_args

            with pytest.raises(SystemExit) as exc_info:
                parse_args()

            assert exc_info.value.code == 2
