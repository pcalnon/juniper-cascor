#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     cascor.py
# Author:        Paul Calnon
#
# Date Created:  2025-06-11
# Last Modified: 2026-02-24
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
#
# Description:
#    This file contains the functions and code needed to solve the two spiral problem using a Cascade Correlation Neural Network.
#
#####################################################################################################################################################################################################
# Notes:
#    - This file serves as the main entry point for the Cascade Correlation Neural Network project.
#    - It initializes logging, sets up configurations, and runs the main logic to solve the two spiral problem.
#    - It uses the `setup_logging` function to configure the logging system.
#    - It creates an instance of the `SpiralProblem` class and calls its `run` method to start the problem-solving process.
#
#####################################################################################################################################################################################################
# References:
#
#
#####################################################################################################################################################################################################
# TODO :
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#
#####################################################################################################################################################################################################
import argparse
import logging
import logging.config
import os
import sys

# PARALLEL-FIX (RC-1), corrected in #531: the BLAS thread policy is now shared with the SERVICE
# entry point instead of living here alone. Must run BEFORE any BLAS-importing module (numpy,
# torch, scipy) -- these variables are read once at library load and are inherited, unchangeably,
# by every forkserver-created candidate worker.
#
# This used to hard-cap all three to 2. The service enters via `uvicorn api.app:create_app`, never
# executed it, and so loaded BLAS uncapped -- two entry points into the same trainer running
# different thread pools by accident of which file started the process. Measured cost: the capped
# path's candidate phase ran 1.52x the uncapped path's on identical data and initialisation, with
# 1.30x attributable to the cap (juniper-cascor#531). The default is now a no-op, matching the
# service; `JUNIPER_CASCOR_BLAS_THREADS` opts back in.
#
# RC-1's actual oversubscription fix -- torch.set_num_threads() per worker and for the parent --
# is untouched and does not depend on these variables.
from parallelism.blas_threads import configure_blas_threads

configure_blas_threads()

from dotenv import load_dotenv

# TODO: F401 - unused imports, may be needed for future use
# from cascor_constants.constants import _CASCOR_MAX_NEW  # trunk-ignore(ruff/F401)
# from cascor_constants.constants import _CASCOR_MAX_ORIG  # trunk-ignore(ruff/F401)
# from cascor_constants.constants import _CASCOR_MIN_NEW  # trunk-ignore(ruff/F401)
# from cascor_constants.constants import _CASCOR_MIN_ORIG  # trunk-ignore(ruff/F401)
# from cascor_constants.constants import _CASCOR_ORIG_POINTS  # trunk-ignore(ruff/F401)
from cascor_constants.constants import (
    _CASCOR_ACTIVATION_FUNCTION,
    _CASCOR_CANDIDATE_DISPLAY_FREQUENCY,
    _CASCOR_CANDIDATE_EPOCHS,
    _CASCOR_CANDIDATE_POOL_SIZE,
    _CASCOR_CLOCKWISE,
    _CASCOR_CORRELATION_THRESHOLD,
    _CASCOR_DEFAULT_ORIGIN,
    _CASCOR_DEFAULT_RADIUS,
    _CASCOR_DISTRIBUTION_FACTOR,
    _CASCOR_EPOCHS_MAX,
    _CASCOR_GENERATE_PLOTS_DEFAULT,
    _CASCOR_INPUT_SIZE,
    _CASCOR_LEARNING_RATE,
    _CASCOR_LOG_CONFIG_FILE_NAME,
    _CASCOR_LOG_CONFIG_FILE_PATH,
    _CASCOR_LOG_DATE_FORMAT,
    _CASCOR_LOG_FILE_NAME,
    _CASCOR_LOG_FILE_PATH,
    _CASCOR_LOG_FORMATTER_STRING,
    _CASCOR_LOG_LEVEL,
    _CASCOR_LOG_LEVEL_CUSTOM_NAMES_LIST,
    _CASCOR_LOG_LEVEL_LOGGING_CONFIG,
    _CASCOR_LOG_LEVEL_METHODS_DICT,
    _CASCOR_LOG_LEVEL_METHODS_LIST,
    _CASCOR_LOG_LEVEL_NAME,
    _CASCOR_LOG_LEVEL_NAMES_LIST,
    _CASCOR_LOG_LEVEL_NUMBERS_DICT,
    _CASCOR_LOG_LEVEL_NUMBERS_LIST,
    _CASCOR_LOG_LEVEL_REDEFINITION,
    _CASCOR_LOG_MESSAGE_DEFAULT,
    _CASCOR_MAX_HIDDEN_UNITS,
    _CASCOR_NOISE_FACTOR_DEFAULT,
    _CASCOR_NUM_ROTATIONS,
    _CASCOR_NUM_SPIRALS,
    _CASCOR_NUMBER_POINTS_PER_SPIRAL,
    _CASCOR_OUTPUT_EPOCHS,
    _CASCOR_OUTPUT_SIZE,
    _CASCOR_PATIENCE,
    _CASCOR_RANDOM_SEED,
    _CASCOR_RANDOM_VALUE_SCALE,
    _CASCOR_STATUS_DISPLAY_FREQUENCY,
    _CASCOR_TEST_RATIO,
    _CASCOR_TRAIN_RATIO,
)
from log_config.log_config import LogConfig
from log_config.logger.logger import Logger
from spiral_problem.spiral_problem import SpiralProblem

# import sys  # TODO: F401 - unused import, may be needed for future use


# import columnar as col
# import torch
# import numpy as np
# import random
# import utils.utils

# from inspect import currentframe, getframeinfo


def _resolve_sentry_dsn() -> str | None:
    """CFG-03: pick the Sentry DSN env var, preferring the prefixed name.

    Historically the bootstrap Sentry init at module-import time read the
    standalone ``SENTRY_SDK_DSN`` env var, while ``Settings.sentry_dsn``
    (used by ``configure_sentry`` in ``src/api/app.py``) reads the
    prefixed ``JUNIPER_CASCOR_SENTRY_DSN`` (via
    ``env_prefix='JUNIPER_CASCOR_'`` on the pydantic ``Settings`` class).
    Two env vars for the same feature is operator-hostile — converge on
    the prefixed name (the ecosystem convention) and keep
    ``SENTRY_SDK_DSN`` accepted with a ``DeprecationWarning`` so existing
    deployments are not broken by this PR. The next major release should
    drop the legacy name.

    Returns:
        The DSN string to pass to ``sentry_sdk.init``, or ``None`` if
        neither env var is set (Sentry stays disabled).

    Precedence:
        1. ``JUNIPER_CASCOR_SENTRY_DSN`` — preferred; matches the
           ecosystem env-prefix convention and the pydantic Settings
           field.
        2. ``SENTRY_SDK_DSN`` — legacy; accepted with a
           ``DeprecationWarning``.

    When both are set:
        - Same value -> no warning, prefixed name wins (no-op).
        - Different values -> stderr line warning of split-config drift;
          prefixed name wins.
    """
    prefixed = os.getenv("JUNIPER_CASCOR_SENTRY_DSN")
    legacy = os.getenv("SENTRY_SDK_DSN")
    if not prefixed and legacy:
        import warnings as _cfg_03_warnings

        _cfg_03_warnings.warn(
            "SENTRY_SDK_DSN is deprecated; set JUNIPER_CASCOR_SENTRY_DSN instead. " "SENTRY_SDK_DSN will be removed in a future release. Until then, the " "legacy variable continues to work but the prefixed form takes " "precedence whenever both are set.",
            DeprecationWarning,
            stacklevel=2,
        )
        return legacy
    if prefixed and legacy and prefixed != legacy:
        # Both set with different values — the prefixed one wins. Tell
        # the operator on stderr so split-config drift is visible at
        # startup.
        print(
            "[juniper-cascor] CFG-03 WARNING: both JUNIPER_CASCOR_SENTRY_DSN and SENTRY_SDK_DSN are set " "to different values; JUNIPER_CASCOR_SENTRY_DSN takes precedence. " "Unset SENTRY_SDK_DSN to silence this message.",
            file=sys.stderr,
        )
    return prefixed


load_dotenv()
_sentry_dsn = _resolve_sentry_dsn()
if _sentry_dsn:
    # CFG-02 (v7 roadmap §13524): ``sentry-sdk`` is now an optional dep
    # declared in the ``[observability]`` extra. Lazy-import here so that
    # ``pip install juniper-cascor`` (no extras) still loads when no
    # DSN is configured. If the operator set a DSN but did not install
    # the extra, emit a clear stderr warning and skip init rather than
    # crashing — bootstrap-time Sentry is opportunistic.
    try:
        import sentry_sdk
    except ImportError:
        print(
            "[juniper-cascor] CFG-02 WARNING: JUNIPER_CASCOR_SENTRY_DSN " "(or legacy SENTRY_SDK_DSN) is set but the ``sentry-sdk`` package " "is not installed. Bootstrap-time Sentry init skipped. Install " "with ``pip install juniper-cascor[observability]`` (or " "``pip install sentry-sdk``) to enable error reporting.",
            file=sys.stderr,
        )
    else:
        # Match configure_sentry()'s behavior at the application-bootstrap
        # site as well: default PII off (SEC-15) and scrub any residual
        # sensitive headers via before_send.
        from api.observability import _strip_sensitive_headers as _sentry_strip_sensitive_headers

        sentry_sdk.init(
            dsn=_sentry_dsn,
            # SEC-15: do not upload default PII (request headers, IP addresses,
            # user identifiers). The before_send hook strips any sensitive
            # headers that other integrations may still attach to events.
            send_default_pii=False,
            enable_logs=True,
            traces_sample_rate=1.0,
            profile_session_sample_rate=1.0,
            profile_lifecycle="trace",
            before_send=_sentry_strip_sensitive_headers,
        )
# app = FastAPI()


#####################################################################################################################################################################################################
# TODO: don't think this is needed with Logger class implementing singleton pattern, and Class methods for initial logging
global logger
global log_config


#####################################################################################################################################################################################################
# Define the main function for Juniper Cascor

#####################################################################################################################################################################################################
# W-11 (CLI experimentation plan SS11 / Wave 3.6): direct-CLI experiment-YAML mapping.
# --config (Wave 3.1) already threads JUNIPER_CASCOR_CONFIG_FILE and the Settings source
# fail-loud-validates the file (SS5.6) when main() constructs Settings below -- so these
# helpers may read the blocks leniently. The adapter maps the YAML's dataset.params /
# training.params onto the direct CLI's overridable knobs, with cascor_constants as the
# fallback tier (SS5.1: CLI env-var threading > YAML > constants). Keys with no
# direct-CLI counterpart are reported loudly, never dropped silently (the W-1 doctrine).

# YAML key -> resolved knob name used by main()'s SpiralProblem/evaluate call sites.
_W11_DATASET_KEY_MAP = {
    "n_points_per_spiral": "n_points",
    "n_spirals": "n_spirals",
    "n_rotations": "n_rotations",
    "noise": "noise",
    "train_ratio": "train_ratio",
    "test_ratio": "test_ratio",
    "seed": "random_seed",
}
_W11_TRAINING_KEY_MAP = {
    "learning_rate": "learning_rate",
    "correlation_threshold": "correlation_threshold",
    "max_hidden_units": "max_hidden_units",
    "patience": "patience",
    "candidate_epochs": "candidate_epochs",
    "candidate_pool_size": "candidate_pool_size",
    "output_epochs": "output_epochs",
    # C2b semantics: TrainingParams.max_epochs is the initial output-training pass budget
    # and defaults to output_epochs -- the closest direct-CLI knob. An explicit
    # output_epochs wins over the alias.
    "max_epochs": "output_epochs",
}


def _load_experiment_blocks():
    """Return (dataset.params, training.params) from the --config experiment YAML ({} when unset)."""
    config_path = os.environ.get("JUNIPER_CASCOR_CONFIG_FILE")
    if not config_path:
        return {}, {}
    import yaml as _yaml

    try:
        data = _yaml.safe_load(open(config_path, encoding="utf-8").read()) or {}
    except OSError as exc:
        # The Settings source raises first in practice; this guard keeps the adapter honest
        # if the file vanishes between the two reads.
        Logger.error(f"Cascor: W-11: experiment config unreadable: {exc}")
        sys.exit(3)
    dataset_params = (data.get("dataset") or {}).get("params") or {}
    training_params = (data.get("training") or {}).get("params") or {}
    return dict(dataset_params), dict(training_params)


def _resolve_cli_overrides(dataset_params, training_params):
    """Map YAML blocks onto direct-CLI knob names; return (overrides, unmapped_keys).

    ``overrides`` is {knob_name: value}; ``unmapped_keys`` lists YAML keys with no
    direct-CLI counterpart (service-tier-only knobs like max_iterations or
    candidate_pool_size) -- the caller reports them loudly.
    """
    overrides = {}
    unmapped = []
    for key, value in dataset_params.items():
        knob = _W11_DATASET_KEY_MAP.get(key)
        if knob is None:
            unmapped.append(f"dataset.params.{key}")
        else:
            overrides[knob] = value
    for key, value in training_params.items():
        knob = _W11_TRAINING_KEY_MAP.get(key)
        if knob is None:
            unmapped.append(f"training.params.{key}")
        elif knob == "output_epochs" and key == "max_epochs" and "output_epochs" in training_params:
            continue  # explicit output_epochs wins over the max_epochs alias
        else:
            overrides[knob] = value
    return overrides, sorted(unmapped)


def main(generate_plots: bool = _CASCOR_GENERATE_PLOTS_DEFAULT):
    """
    Run the spiral problem end to end.

    Args:
        generate_plots: Whether to build the dataset / decision-boundary / training-history
            figures. Defaults to the project constant; `--no-plots` turns them off, which is
            what an automated or headless run wants (F-P1-3) since the figures are
            display-only and are never saved to disk.
    """
    Logger.info("Cascor: main: Starting the Cascade Correlation Neural Network project")
    Logger.info(f"Cascor: main: Project constants: Log Level: {_CASCOR_LOG_LEVEL}, Log Level Name: {_CASCOR_LOG_LEVEL_NAME}")
    Logger.info(f"Cascor: main: Project constants: Log File Name: {_CASCOR_LOG_FILE_NAME}, Log File Path: {_CASCOR_LOG_FILE_PATH}")

    if (
        log_config := LogConfig(
            _LogConfig__log_config=logging.config,
            # _LogConfig__log_config=None,
            _LogConfig__log_config_file_name=_CASCOR_LOG_CONFIG_FILE_NAME,
            _LogConfig__log_config_file_path=_CASCOR_LOG_CONFIG_FILE_PATH,
            _LogConfig__log_date_format=_CASCOR_LOG_DATE_FORMAT,
            _LogConfig__log_file_name=_CASCOR_LOG_FILE_NAME,
            _LogConfig__log_file_path=_CASCOR_LOG_FILE_PATH,
            _LogConfig__log_formatter_string=_CASCOR_LOG_FORMATTER_STRING,
            _LogConfig__log_level_custom_names_list=_CASCOR_LOG_LEVEL_CUSTOM_NAMES_LIST,
            _LogConfig__log_level_logging_config=_CASCOR_LOG_LEVEL_LOGGING_CONFIG,
            _LogConfig__log_level_methods_dict=_CASCOR_LOG_LEVEL_METHODS_DICT,
            _LogConfig__log_level_methods_list=_CASCOR_LOG_LEVEL_METHODS_LIST,
            _LogConfig__log_level_names_list=_CASCOR_LOG_LEVEL_NAMES_LIST,
            _LogConfig__log_level_numbers_dict=_CASCOR_LOG_LEVEL_NUMBERS_DICT,
            _LogConfig__log_level_numbers_list=_CASCOR_LOG_LEVEL_NUMBERS_LIST,
            _LogConfig__log_level_redefinition=_CASCOR_LOG_LEVEL_REDEFINITION,
            _LogConfig__log_level=_CASCOR_LOG_LEVEL,
            _LogConfig__log_level_name=_CASCOR_LOG_LEVEL_NAME,
            _LogConfig__log_message_default=_CASCOR_LOG_MESSAGE_DEFAULT,
        )
    ) is None:
        Logger.error("Cascor: main: Error: Failed to create LogConfig class")
        sys.exit(1)
    elif (logger := log_config.get_logger()) is None:
        Logger.error("Cascor: main: Error: Failed to get Logger object from LogConfig class")
        sys.exit(2)

    Logger.debug(f"Cascor: main: Successfully created LogConfig class and Logger object: Type: {type(log_config)}, Value:\n{log_config}")
    Logger.debug(f"Cascor: main: Successfully created LogConfig class and Logger object: Type: {type(logger)}, Value:\n{logger}")
    Logger.debug(f"Cascor: main: Successfully created LogConfig class & Logger object: log level: Type: {type(log_config.get_log_level())}, Value: {log_config.get_log_level()}")
    Logger.debug(f"Cascor: main: Successfully created LogConfig class & Logger object: log level name: Type: {type(log_config.get_log_level_name())}, Value: {log_config.get_log_level_name()}")
    Logger.debug(f"Cascor: main: Successfully created Logger object: logger level: Type: {type(logger.level)}, Value: {logger.level}")
    # Logger.debug(f"Cascor: main: Successfully created Logger object: logger name: '{logger.name}', handlers: {len(logger.handlers)}")  # B907
    Logger.debug(f"Cascor: main: Successfully created Logger object: logger name: {logger.name!r}, handlers: {len(logger.handlers)}")

    logger.verbose(f"Cascor: main: Successfully created LogConfig class: Type: {type(log_config)}, Value: {log_config}, and Logger object: Type: {type(logger)}, Value: {logger}")
    logger.debug(f"Cascor: main: Successfully created LogConfig class: {log_config} and Logger object: {logger}")
    logger.info(f"Cascor: main: Successfully created LogConfig class & Logger object: log level: {log_config.get_log_level()}")
    logger.info("Cascor: main: Inside Main function")
    logger.info("Cascor: main: Completed Initialization of Project Logger")

    # #####################################################################################################################################################################################################

    #     generated_datasets = GeneratedDatasets(
    #         _GeneratedDatasets__spiral_config=config,
    #         _GeneratedDatasets__dataset_tensors=None,
    #         _GeneratedDatasets__dataset_file_info=None,
    #         _GeneratedDatasets__num_spirals=_SPIRAL_DATASET_DEFAULT_NUM_SPIRALS,
    #         _GeneratedDatasets__num_points_per_spiral=_SPIRAL_DATASET_DEFAULT_NUM_POINTS_PER_SPIRAL,
    #         _GeneratedDatasets__noise_level=_SPIRAL_DATASET_DEFAULT_NOISE_LEVEL,
    #         _GeneratedDatasets__num_rotations=_SPIRAL_DATASET_DEFAULT_NUM_ROTATIONS,
    #         _GeneratedDatasets__min_radius=_SPIRAL_DATASET_DEFAULT_MIN_RADIUS,
    #         _GeneratedDatasets__max_radius=_SPIRAL_DATASET_DEFAULT_MAX_RADIUS,
    #         _GeneratedDatasets__clockwise_rotation=_SPIRAL_DATASET_DEFAULT_CLOCKWISE_ROTATION,
    #         _GeneratedDatasets__seed_value=_SPIRAL_DATASET_DEFAULT_SEED_VALUE,
    #         _GeneratedDatasets__dataset_dir=_SPIRAL_DATASET_DATASET_DIR_DEFAULT,
    #         _GeneratedDatasets__visualization_dir=_SPIRAL_DATASET_VISUALIZATION_DIR_DEFAULT,
    #         _GeneratedDatasets__log_file_name=_CASCOR_SPIRAL_DATASET_LOG_NAME,
    #         _GeneratedDatasets__log_formatter_string=_CASCOR_SPIRAL_DATASET_LOG_FORMATTER_STRING,
    #         _GeneratedDatasets__log_date_format=_CASCOR_SPIRAL_DATASET_LOG_DATE_FORMAT,
    #         _GeneratedDatasets__log_file_path=_CASCOR_SPIRAL_DATASET_LOG_FILE_PATH,
    #         _GeneratedDatasets__log_level=_CASCOR_SPIRAL_DATASET_LOG_LEVEL_DEFAULT,
    #         _GeneratedDatasets__dataset_train_ratio=_SPIRAL_DATASET_DEFAULT_TRAIN_RATIO,
    #         _GeneratedDatasets__dataset_test_ratio=_SPIRAL_DATASET_DEFAULT_TEST_RATIO,
    #         _GeneratedDatasets__dataset_val_ratio=_SPIRAL_DATASET_DEFAULT_VAL_RATIO,
    #     )

    #     config = SpiralConfig(
    #         _SpiralConfig__num_spirals=_SPIRAL_DATASET_DEFAULT_NUM_SPIRALS,
    #         _SpiralConfig__num_points_per_spiral=_SPIRAL_DATASET_DEFAULT_NUM_POINTS_PER_SPIRAL,
    #         _SpiralConfig__noise_level=_SPIRAL_DATASET_DEFAULT_NOISE_LEVEL,
    #         _SpiralConfig__num_rotations=_SPIRAL_DATASET_DEFAULT_NUM_ROTATIONS,
    #         _SpiralConfig__min_radius=_SPIRAL_DATASET_DEFAULT_MIN_RADIUS,
    #         _SpiralConfig__max_radius=_SPIRAL_DATASET_DEFAULT_MAX_RADIUS,
    #         _SpiralConfig__clockwise_rotation=_SPIRAL_DATASET_DEFAULT_CLOCKWISE_ROTATION,
    #         _SpiralConfig__seed_value=_SPIRAL_DATASET_DEFAULT_SEED_VALUE,
    #         _SpiralConfig__visualization_dir=_SPIRAL_DATASET_VISUALIZATION_DIR_DEFAULT,
    #         _SpiralConfig__dataset_dir=_SPIRAL_DATASET_DATASET_DIR_DEFAULT,
    #         _SpiralConfig__log_file_path=_CASCOR_SPIRAL_DATASET_LOG_FILE_PATH,
    #         _SpiralConfig__log_name=_CASCOR_SPIRAL_DATASET_LOG_NAME,
    #         _SpiralConfig__log_level=_CASCOR_SPIRAL_DATASET_LOG_LEVEL_DEFAULT,
    #     )

    # #####################################################################################################################################################################################################

    # Pre-flight check: Validate JuniperData service connectivity before expensive initialization
    # CFG-04: Settings field consolidates the JUNIPER_DATA_URL env-var
    # lookup. Required at startup (no fallback) — fail loudly with the
    # same exit code as before so deployment manifests catch
    # misconfiguration immediately.
    from api.settings import Settings as _CfgSettings  # local import: keep startup import graph minimal

    juniper_data_url = _CfgSettings().juniper_data_url
    if not juniper_data_url:
        logger.error("Cascor: main: JUNIPER_DATA_URL environment variable is not set. " "Set it to the JuniperData service URL (e.g., 'http://localhost:8100'). " "See AGENTS.md for configuration details.")
        sys.exit(3)

    logger.info(f"Cascor: main: Pre-flight check: Verifying JuniperData service at {juniper_data_url}")
    try:
        import urllib.request

        health_url = f"{juniper_data_url.rstrip('/')}/v1/health"
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310 - internal health check URL from env var
            logger.info(f"Cascor: main: Pre-flight check: JuniperData service is healthy (HTTP {resp.status})")
    except Exception as e:
        logger.error(f"Cascor: main: Pre-flight check FAILED: JuniperData service at {juniper_data_url} is not reachable. " f"Error: {e}\n" f"    Please start the JuniperData service before running JuniperCascor:\n" f"        cd juniper-data && conda activate JuniperData && ./try\n" f"    Or:  conda activate JuniperData && python -m juniper_data")
        sys.exit(4)

    # W-11: resolve experiment-YAML overrides (constants are the fallback tier).
    _w11_dataset, _w11_training = _load_experiment_blocks()
    _w11, _w11_unmapped = _resolve_cli_overrides(_w11_dataset, _w11_training)
    if _w11:
        logger.info(f"Cascor: main: W-11 experiment-YAML overrides active: {sorted(_w11)}")
    if _w11_unmapped:
        logger.warning(f"Cascor: main: W-11: experiment-YAML keys with no direct-CLI counterpart (service-tier only), IGNORED here: {_w11_unmapped}")

    # Instantiate the SpiralProblem class
    logger.info("Cascor: main: Creating SpiralProblem instance")
    sp = SpiralProblem(
        _SpiralProblem__spiral_config=logging.config,
        _SpiralProblem__dataset_tensors=None,
        _SpiralProblem__dataset_file_info=None,
        _SpiralProblem__activation_function=_CASCOR_ACTIVATION_FUNCTION,
        _SpiralProblem__candidate_display_frequency=_CASCOR_CANDIDATE_DISPLAY_FREQUENCY,
        _SpiralProblem__candidate_epochs=_w11.get("candidate_epochs", _CASCOR_CANDIDATE_EPOCHS),
        _SpiralProblem__candidate_pool_size=_w11.get("candidate_pool_size", _CASCOR_CANDIDATE_POOL_SIZE),
        _SpiralProblem__clockwise=_CASCOR_CLOCKWISE,
        _SpiralProblem__correlation_threshold=_w11.get("correlation_threshold", _CASCOR_CORRELATION_THRESHOLD),
        _SpiralProblem__default_origin=_CASCOR_DEFAULT_ORIGIN,
        _SpiralProblem__default_radius=_CASCOR_DEFAULT_RADIUS,
        _SpiralProblem__distribution=_CASCOR_DISTRIBUTION_FACTOR,
        _SpiralProblem__epochs_max=_CASCOR_EPOCHS_MAX,
        _SpiralProblem__generate_plots_default=generate_plots,
        _SpiralProblem__input_size=_CASCOR_INPUT_SIZE,
        _SpiralProblem__learning_rate=_w11.get("learning_rate", _CASCOR_LEARNING_RATE),
        _SpiralProblem__log_config=log_config,
        _SpiralProblem__log_file_name=_CASCOR_LOG_FILE_NAME,
        _SpiralProblem__log_file_path=_CASCOR_LOG_FILE_PATH,
        _SpiralProblem__log_level_name=_CASCOR_LOG_LEVEL_NAME,
        _SpiralProblem__max_hidden_units=_w11.get("max_hidden_units", _CASCOR_MAX_HIDDEN_UNITS),
        _SpiralProblem__n_points=_w11.get("n_points", _CASCOR_NUMBER_POINTS_PER_SPIRAL),
        _SpiralProblem__n_rotations=_w11.get("n_rotations", _CASCOR_NUM_ROTATIONS),
        _SpiralProblem__n_spirals=_w11.get("n_spirals", _CASCOR_NUM_SPIRALS),
        _SpiralProblem__noise=_w11.get("noise", _CASCOR_NOISE_FACTOR_DEFAULT),
        _SpiralProblem__output_size=_CASCOR_OUTPUT_SIZE,
        _SpiralProblem__patience=_w11.get("patience", _CASCOR_PATIENCE),
        _SpiralProblem__output_epochs=_w11.get("output_epochs", _CASCOR_OUTPUT_EPOCHS),
        _SpiralProblem__status_display_frequency=_CASCOR_STATUS_DISPLAY_FREQUENCY,
        _SpiralProblem__random_seed=_w11.get("random_seed", _CASCOR_RANDOM_SEED),
        _SpiralProblem__train_ratio=_w11.get("train_ratio", _CASCOR_TRAIN_RATIO),
        _SpiralProblem__test_ratio=_w11.get("test_ratio", _CASCOR_TEST_RATIO),
    )
    logger.debug(f"Main: sp: Type: {type(sp)}, Value:\n{sp}")

    # #####################################################################################################################################################################################################

    #     complex_dataset_output = generated_datasets.generate_spiral_datasets(
    #         num_spirals=_SPIRAL_DATASET_DEFAULT_NUM_SPIRALS,
    #         num_points=_SPIRAL_DATASET_DEFAULT_NUM_POINTS_PER_SPIRAL,
    #         num_rotations=_SPIRAL_DATASET_DEFAULT_NUM_ROTATIONS,
    #         noise_level=_SPIRAL_DATASET_DEFAULT_NOISE_LEVEL,
    #         min_radius=_SPIRAL_DATASET_DEFAULT_MIN_RADIUS,
    #         max_radius=_SPIRAL_DATASET_DEFAULT_MAX_RADIUS,
    #         clockwise_rotation=_SPIRAL_DATASET_DEFAULT_CLOCKWISE_ROTATION,
    #         seed_value=_SPIRAL_DATASET_DEFAULT_SEED_VALUE,
    #         visualization_dir=_SPIRAL_DATASET_VISUALIZATION_DIR_DEFAULT,
    #         dataset_dir=_SPIRAL_DATASET_DATASET_DIR_DEFAULT,
    #         log_file_path=_CASCOR_SPIRAL_DATASET_LOG_FILE_PATH,
    #         dataset_train_ratio=_SPIRAL_DATASET_DEFAULT_TRAIN_RATIO,
    #         dataset_test_ratio=_SPIRAL_DATASET_DEFAULT_TEST_RATIO,
    #         dataset_val_ratio=_SPIRAL_DATASET_DEFAULT_VAL_RATIO,
    #     )
    #     logger.info("cascor_spiral: main: Creating DatasetTensors instance for the complex dataset...")
    #     (complex_dataset_file_info, complex_dataset_tensors) = complex_dataset_output
    #     logger.info(f"cascor_spiral: main: Generated complex dataset file info: {complex_dataset_file_info}")
    #     logger.info(f"cascor_spiral: main: Generated complex dataset tensors: {complex_dataset_tensors}")
    #     # Convert Complex dataset to tensors dict and send to CUDA if available
    #     complex_dataset_tensors.to_cuda()

    # #####################################################################################################################################################################################################

    # Solve the two spiral problem using the SpiralProblem instance
    logger.info("Main: Solving SpiralProblem instance")
    sp.evaluate(
        n_points=_w11.get("n_points", _CASCOR_NUMBER_POINTS_PER_SPIRAL),
        n_spirals=_w11.get("n_spirals", _CASCOR_NUM_SPIRALS),
        n_rotations=_w11.get("n_rotations", _CASCOR_NUM_ROTATIONS),
        clockwise=_CASCOR_CLOCKWISE,
        distribution=_CASCOR_DISTRIBUTION_FACTOR,
        random_value_scale=_CASCOR_RANDOM_VALUE_SCALE,
        default_origin=_CASCOR_DEFAULT_ORIGIN,
        default_radius=_CASCOR_DEFAULT_RADIUS,
        train_ratio=_w11.get("train_ratio", _CASCOR_TRAIN_RATIO),
        test_ratio=_w11.get("test_ratio", _CASCOR_TEST_RATIO),
        noise=_w11.get("noise", _CASCOR_NOISE_FACTOR_DEFAULT),
        plot=generate_plots,
    )
    logger.info("Main: Completed solving SpiralProblem instance")


#####################################################################################################################################################################################################
# Command Line Argument Parsing
# P3-NEW-001: Development Profiling Infrastructure


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Juniper Cascor - Cascade Correlation Neural Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                      # Run normally
  python main.py --no-plots           # Run headless / automated (no blocking figures)
  python main.py --profile            # Run with cProfile profiling
  python main.py --profile-memory     # Run with memory profiling
  python main.py --profile --profile-output ./my_profiles
        """,
    )

    parser.add_argument("--profile", action="store_true", help="Enable cProfile deterministic profiling")

    parser.add_argument("--profile-memory", action="store_true", help="Enable tracemalloc memory profiling")

    parser.add_argument("--profile-output", type=str, default="./profiles", help="Directory for profile output files (default: ./profiles)")

    parser.add_argument("--profile-top-n", type=int, default=30, help="Number of top functions to display in profile output (default: 30)")
    parser.add_argument("--config", type=str, default=None, help="Experiment YAML whose service: block overrides env (sets JUNIPER_CASCOR_CONFIG_FILE before settings load; Wave 3.1)")
    parser.add_argument("--no-plots", action="store_true", help="Skip the dataset / decision-boundary / training-history figures. Recommended for automated and headless runs: the figures are display-only (never saved), and under an interactive backend showing them blocks the process after training finishes (F-P1-3)")

    return parser.parse_args()


#####################################################################################################################################################################################################
# Main function to run the two spiral problem solution
# This is the entry point for the script.
if __name__ == "__main__":
    args = parse_args()

    if args.config:
        # Wave 3.1: must land before the first Settings()/get_settings() use (SS5.2).
        os.environ["JUNIPER_CASCOR_CONFIG_FILE"] = args.config

    # F-P1-3: resolved once so every entry path below (plain, cProfile, tracemalloc) honours
    # --no-plots. A profiling run is automated by definition, so it is exactly the path that
    # must not park in a GUI event loop after training finishes.
    generate_plots = _CASCOR_GENERATE_PLOTS_DEFAULT and not args.no_plots

    if args.profile:
        from profiling.deterministic import ProfileContext

        Logger.info("Cascor: Starting with cProfile profiling enabled")
        with ProfileContext("main_training", output_dir=args.profile_output) as profiler:
            main(generate_plots=generate_plots)
        profiler.print_stats(top_n=args.profile_top_n)
        profiler.save()

    elif args.profile_memory:
        from profiling.memory import MemoryTracker

        Logger.info("Cascor: Starting with memory profiling enabled")
        with MemoryTracker("main_training") as tracker:
            main(generate_plots=generate_plots)
        tracker.print_summary()
        tracker.print_top_allocations(top_n=args.profile_top_n)
        tracker.print_diff(top_n=args.profile_top_n)

    else:
        main(generate_plots=generate_plots)
