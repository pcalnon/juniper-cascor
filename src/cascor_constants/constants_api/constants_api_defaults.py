#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     constants_api_defaults.py
# Author:        Paul Calnon
#
# Date Created:  2026-04-09
# Last Modified: 2026-04-09
#
# License:       MIT License
# Copyright:     Copyright (c) 2025-2026 Paul Calnon
#
# Description:
#    This file contains default constants for the JuniperCascor REST API layer.
#    These constants define default values for Pydantic model fields, lifecycle manager
#    configuration, service URLs, timeout values, security settings, and HTTP route parameters.
#
#####################################################################################################################################################################################################
# Notes:
#    - All constants use the _PROJECT_API_* naming prefix, consistent with the existing
#      _PROJECT_MODEL_* and _PROJECT_PROBLEM_* patterns in the cascor_constants hierarchy.
#    - These constants are consumed in Wave 2 of the hardcoded values refactor.
#    - Log rotation and observability constants are in constants_logging/constants_logging.py.
#    - Candidate unit constants are in constants_candidates/constants_candidates.py.
#
#####################################################################################################################################################################################################
# References:
#    - api/models/network.py              -- NetworkCreateRequest Field defaults
#    - api/models/training.py             -- DatasetSource and InlineDataset defaults
#    - api/lifecycle/manager.py           -- Lifecycle manager defaults
#    - api/lifecycle/monitor.py           -- Metrics buffer size
#    - api/app.py                         -- Service URLs, startup timeouts
#    - api/service_launcher.py            -- Process and service termination timeouts
#    - api/workers/security.py            -- TLS, rate limiter, anomaly detection defaults
#    - api/middleware.py                  -- Request body limit, HTTP status codes
#    - api/routes/decision_boundary.py   -- Resolution bounds and HTTP status codes
#
#####################################################################################################################################################################################################
# TODO:
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#
#####################################################################################################################################################################################################


#####################################################################################################################################################################################################
# API Network Model Defaults (api/models/network.py — NetworkCreateRequest Field defaults)
#####################################################################################################################################################################################################

_PROJECT_API_NETWORK_INPUT_SIZE_DEFAULT: int = 2
_PROJECT_API_NETWORK_OUTPUT_SIZE_DEFAULT: int = 2
_PROJECT_API_NETWORK_LEARNING_RATE_DEFAULT: float = 0.01
_PROJECT_API_NETWORK_CANDIDATE_LEARNING_RATE_DEFAULT: float = 0.005
_PROJECT_API_NETWORK_MAX_HIDDEN_UNITS_DEFAULT: int = 10
_PROJECT_API_NETWORK_CANDIDATE_POOL_SIZE_DEFAULT: int = 8
_PROJECT_API_NETWORK_CORRELATION_THRESHOLD_DEFAULT: float = 0.1
_PROJECT_API_NETWORK_PATIENCE_DEFAULT: int = 5
_PROJECT_API_NETWORK_CANDIDATE_EPOCHS_DEFAULT: int = 50
_PROJECT_API_NETWORK_OUTPUT_EPOCHS_DEFAULT: int = 25
_PROJECT_API_NETWORK_EPOCHS_MAX_DEFAULT: int = 200
_PROJECT_API_NETWORK_MAX_ITERATIONS_DEFAULT: int = 1000
_PROJECT_API_NETWORK_INIT_OUTPUT_WEIGHTS_DEFAULT: str = "zero"
# CAN-011 (Phase 6E Sprint A-3): output-layer activation function default.
# Mirrors cascade_correlation.py::_init_activation_function's fallback.
_PROJECT_API_NETWORK_ACTIVATION_FUNCTION_DEFAULT: str = "Tanh"


#####################################################################################################################################################################################################
# API Training Model Defaults (api/models/training.py — DatasetSource and InlineDataset defaults)
#####################################################################################################################################################################################################

_PROJECT_API_DATASET_SOURCE_DEFAULT: str = "inline"
_PROJECT_API_MAX_DATASET_SAMPLES: int = 100000
_PROJECT_API_MAX_DATASET_TARGETS: int = 100000


#####################################################################################################################################################################################################
# Lifecycle Manager Defaults (api/lifecycle/manager.py and api/lifecycle/monitor.py)
#####################################################################################################################################################################################################

# Lifecycle training-limit caps are intentionally large so the user (or the
# canopy UI), not the API default, chooses when to stop. Aligned with the
# canopy / requirements-spec values per
# juniper-ml/notes/code-review/CANOPY_CASCOR_INTERFACE_ROADMAP_2026-04-08.md
# §3.5 (PR #122 raised these from 1e3/1e3/1e3; rolled back to small values
# during the Waves 1-6 hardcoded-values cleanup #123 — V35d re-aligns).
_PROJECT_API_LIFECYCLE_DEFAULT_MAX_HIDDEN_UNITS: int = 10_000
_PROJECT_API_LIFECYCLE_DEFAULT_EPOCHS_MAX: int = 100_000_000_000
_PROJECT_API_LIFECYCLE_DEFAULT_MAX_ITERATIONS: int = 1_000_000
_PROJECT_API_METRICS_BUFFER_SIZE: int = 10000
_PROJECT_API_PROGRESS_QUEUE_WAIT_TIMEOUT: float = 0.1
_PROJECT_API_PROGRESS_QUEUE_GET_TIMEOUT: float = 0.25
_PROJECT_API_DRAIN_THREAD_JOIN_TIMEOUT: float = 2.0
_PROJECT_API_LIFECYCLE_DEFAULT_CANDIDATE_PATIENCE: int = 30


#####################################################################################################################################################################################################
# Service URLs and Endpoints (api/app.py)
#####################################################################################################################################################################################################

_PROJECT_API_JUNIPER_DATA_URL_DEFAULT: str = "http://localhost:8100"
_PROJECT_API_JUNIPER_DATA_READY_TIMEOUT: int = 60
_PROJECT_API_SELF_HEALTH_CHECK_URL_TEMPLATE: str = "http://localhost:{port}/v1/health"
_PROJECT_API_CANOPY_STARTUP_WAIT_TIMEOUT: float = 30.0
_PROJECT_API_CANOPY_STARTUP_CHECK_INTERVAL: float = 1.0
_PROJECT_API_CANOPY_DEMO_MODE_DISABLED: str = "false"
_PROJECT_API_CANOPY_HEALTH_CHECK_URL: str = "http://localhost:8050/v1/health"


#####################################################################################################################################################################################################
# Service Launcher Timeouts (api/service_launcher.py)
#####################################################################################################################################################################################################

_PROJECT_API_SERVICE_DEFAULT_TERMINATE_TIMEOUT: float = 10.0
_PROJECT_API_PROCESS_TERMINATION_TIMEOUT: int = 5
_PROJECT_API_SERVICE_TERMINATION_TIMEOUT: int = 5
_PROJECT_API_SERVICE_HEALTH_POLL_TIMEOUT: float = 60.0
_PROJECT_API_SERVICE_HEALTH_POLL_INTERVAL: float = 2.0
_PROJECT_API_HEALTH_CHECK_HTTP_TIMEOUT: int = 5


#####################################################################################################################################################################################################
# Security Defaults (api/workers/security.py)
#####################################################################################################################################################################################################

_PROJECT_API_TLS_MIN_VERSION_DEFAULT: str = "TLSv1.3"
_PROJECT_API_RATE_LIMITER_CLEANUP_INTERVAL: float = 300.0
_PROJECT_API_ANOMALY_STALE_CORR_THRESHOLD: float = 0.001
_PROJECT_API_ANOMALY_DUPLICATE_CORR_WINDOW: int = 10


#####################################################################################################################################################################################################
# Middleware Constants (api/middleware.py)
#####################################################################################################################################################################################################

_PROJECT_API_MAX_REQUEST_BODY_BYTES: int = 10 * 1024 * 1024  # 10 MB
_PROJECT_API_HTTP_400_BAD_REQUEST: int = 400
_PROJECT_API_HTTP_413_PAYLOAD_TOO_LARGE: int = 413


#####################################################################################################################################################################################################
# Decision Boundary Route Constants (api/routes/decision_boundary.py)
#####################################################################################################################################################################################################

_PROJECT_API_DECISION_BOUNDARY_RESOLUTION_DEFAULT: int = 50
_PROJECT_API_DECISION_BOUNDARY_RESOLUTION_MIN: int = 5
_PROJECT_API_DECISION_BOUNDARY_RESOLUTION_MAX: int = 200
_PROJECT_API_HTTP_503_SERVICE_UNAVAILABLE: int = 503
_PROJECT_API_HTTP_404_NOT_FOUND: int = 404
_PROJECT_API_HTTP_500_INTERNAL_SERVER_ERROR: int = 500
