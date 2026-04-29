"""API configuration settings using pydantic-settings."""

from functools import lru_cache
from typing import Any

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from api.secrets import get_secret

# Define Safe and Reasonable Defaults for API Settings
_JUNIPER_CASCOR_ENV_PREFIX: str = "JUNIPER_CASCOR_"

_JUNIPER_CASCOR_API_HOST_LOCAL: str = "127.0.0.1"
_JUNIPER_CASCOR_API_HOST_DEFAULT: str = _JUNIPER_CASCOR_API_HOST_LOCAL

_JUNIPER_CASCOR_API_PORT: int = 8200
_JUNIPER_CASCOR_API_PORT_DEFAULT: int = _JUNIPER_CASCOR_API_PORT

_JUNIPER_CASCOR_API_LOGLEVEL_INFO: str = "INFO"
_JUNIPER_CASCOR_API_LOGLEVEL_DEFAULT: str = _JUNIPER_CASCOR_API_LOGLEVEL_INFO

_JUNIPER_CASCOR_API_CORS_ORIGINS_ALL: list[str] = ["*"]
_JUNIPER_CASCOR_API_CORS_ORIGINS_NONE: list[str] = []
_JUNIPER_CASCOR_API_CORS_ORIGINS_DEFAULT: list[str] = _JUNIPER_CASCOR_API_CORS_ORIGINS_NONE

_JUNIPER_CASCOR_API_WS_MAX_CONNECTIONS: int = 50
_JUNIPER_CASCOR_API_WS_MAX_CONNECTIONS_DEFAULT: int = _JUNIPER_CASCOR_API_WS_MAX_CONNECTIONS

_JUNIPER_CASCOR_API_WS_HEARTBEAT_INTERVAL_SEC: int = 30
_JUNIPER_CASCOR_API_WS_HEARTBEAT_INTERVAL_SEC_DEFAULT: int = _JUNIPER_CASCOR_API_WS_HEARTBEAT_INTERVAL_SEC

_JUNIPER_CASCOR_API_WS_HEARTBEAT_PONG_TIMEOUT_SEC: int = 10
_JUNIPER_CASCOR_API_WS_HEARTBEAT_PONG_TIMEOUT_SEC_DEFAULT: int = _JUNIPER_CASCOR_API_WS_HEARTBEAT_PONG_TIMEOUT_SEC

# Phase 0-cascor: WebSocket sequencing, replay, and resume settings
# Env vars match canonical plan kill switch names (JUNIPER_WS_* prefix)
_JUNIPER_CASCOR_API_WS_REPLAY_BUFFER_SIZE: int = 1024
_JUNIPER_CASCOR_API_WS_REPLAY_BUFFER_SIZE_DEFAULT: int = _JUNIPER_CASCOR_API_WS_REPLAY_BUFFER_SIZE

_JUNIPER_CASCOR_API_WS_SEND_TIMEOUT_SECONDS: float = 0.5
_JUNIPER_CASCOR_API_WS_SEND_TIMEOUT_SECONDS_DEFAULT: float = _JUNIPER_CASCOR_API_WS_SEND_TIMEOUT_SECONDS

_JUNIPER_CASCOR_API_WS_RESUME_HANDSHAKE_TIMEOUT_S: float = 5.0
_JUNIPER_CASCOR_API_WS_RESUME_HANDSHAKE_TIMEOUT_S_DEFAULT: float = _JUNIPER_CASCOR_API_WS_RESUME_HANDSHAKE_TIMEOUT_S

_JUNIPER_CASCOR_API_WS_STATE_THROTTLE_COALESCE_MS: int = 1000
_JUNIPER_CASCOR_API_WS_STATE_THROTTLE_COALESCE_MS_DEFAULT: int = _JUNIPER_CASCOR_API_WS_STATE_THROTTLE_COALESCE_MS

_JUNIPER_CASCOR_API_WS_PENDING_MAX_DURATION_S: float = 10.0
_JUNIPER_CASCOR_API_WS_PENDING_MAX_DURATION_S_DEFAULT: float = _JUNIPER_CASCOR_API_WS_PENDING_MAX_DURATION_S

_JUNIPER_CASCOR_API_KEYS_LIST_EMPTY: list[str] | None = None
_JUNIPER_CASCOR_API_RATELIMIT_DISABLED: bool = False
_JUNIPER_CASCOR_API_RATELIMIT_ENABLED: bool = True
_JUNIPER_CASCOR_API_RATELIMIT_DEFAULT: int = 60

_JUNIPER_CASCOR_API_LOG_FORMAT_TEXT: str = "text"
_JUNIPER_CASCOR_API_LOG_FORMAT_DEFAULT: str = _JUNIPER_CASCOR_API_LOG_FORMAT_TEXT

_JUNIPER_CASCOR_API_SENTRY_DSN_NONE: str | None = None
_JUNIPER_CASCOR_API_SENTRY_DSN_DEFAULT: str | None = _JUNIPER_CASCOR_API_SENTRY_DSN_NONE

_JUNIPER_CASCOR_API_METRICS_ENABLED_DISABLED: bool = False
_JUNIPER_CASCOR_API_METRICS_ENABLED_DEFAULT: bool = _JUNIPER_CASCOR_API_METRICS_ENABLED_DISABLED

_JUNIPER_CASCOR_API_AUTO_START_DISABLED: bool = False
_JUNIPER_CASCOR_API_AUTO_START_ENABLED: bool = True
_JUNIPER_CASCOR_API_AUTO_START_DEFAULT: bool = _JUNIPER_CASCOR_API_AUTO_START_ENABLED

_JUNIPER_CASCOR_API_AUTO_START_DATA_SERVICE_DEFAULT: bool = False
_JUNIPER_CASCOR_API_AUTO_START_DATA_SERVICE_COMMAND_DEFAULT: str = "python -m juniper_data"

_JUNIPER_CASCOR_API_AUTO_START_CANOPY_DEFAULT: bool = False
_JUNIPER_CASCOR_API_AUTO_START_CANOPY_COMMAND_DEFAULT: str = "python -m juniper_canopy"

_JUNIPER_CASCOR_API_AUTO_DATASET_SPIRAL: str = "spiral"
_JUNIPER_CASCOR_API_AUTO_DATASET_DEFAULT: str = _JUNIPER_CASCOR_API_AUTO_DATASET_SPIRAL

_JUNIPER_CASCOR_API_AUTO_DATASET_PARAMS_EMPTY: str = "{}"
_JUNIPER_CASCOR_API_AUTO_DATASET_PARAMS_DEFAULT: str = _JUNIPER_CASCOR_API_AUTO_DATASET_PARAMS_EMPTY

_JUNIPER_CASCOR_API_AUTO_NETWORK_EMPTY: str = "{}"
_JUNIPER_CASCOR_API_AUTO_NETWORK_DEFAULT: str = _JUNIPER_CASCOR_API_AUTO_NETWORK_EMPTY

_JUNIPER_CASCOR_API_AUTO_TRAIN_EPOCHS: int = 200
_JUNIPER_CASCOR_API_AUTO_TRAIN_EPOCHS_DEFAULT: int = _JUNIPER_CASCOR_API_AUTO_TRAIN_EPOCHS

_JUNIPER_CASCOR_API_REMOTE_WORKERS_HEARTBEAT_TIMEOUT: float = 30.0
_JUNIPER_CASCOR_API_REMOTE_WORKERS_HEARTBEAT_TIMEOUT_DEFAULT: float = _JUNIPER_CASCOR_API_REMOTE_WORKERS_HEARTBEAT_TIMEOUT

_JUNIPER_CASCOR_API_REMOTE_WORKERS_TASK_REASSIGNMENT_TIMEOUT: float = 120.0
_JUNIPER_CASCOR_API_REMOTE_WORKERS_TASK_REASSIGNMENT_TIMEOUT_DEFAULT: float = _JUNIPER_CASCOR_API_REMOTE_WORKERS_TASK_REASSIGNMENT_TIMEOUT

# Worker Security (Phase 4) defaults
_JUNIPER_CASCOR_API_WORKER_RATE_LIMIT_ENABLED_DEFAULT: bool = False
_JUNIPER_CASCOR_API_WORKER_RATE_LIMIT_CONNECTIONS_PER_MINUTE_DEFAULT: int = 10
_JUNIPER_CASCOR_API_WORKER_RATE_LIMIT_BURST_SIZE_DEFAULT: int = 3

_JUNIPER_CASCOR_API_WORKER_ANOMALY_DETECTION_ENABLED_DEFAULT: bool = False
_JUNIPER_CASCOR_API_WORKER_ANOMALY_MIN_TRAINING_TIME_DEFAULT: float = 0.1
_JUNIPER_CASCOR_API_WORKER_ANOMALY_PERFECT_CORR_THRESHOLD_DEFAULT: float = 0.999

_JUNIPER_CASCOR_API_WORKER_AUDIT_LOGGING_ENABLED_DEFAULT: bool = False
_JUNIPER_CASCOR_API_WORKER_METRICS_ENABLED_DEFAULT: bool = False


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be overridden via environment variables with the
    JUNIPER_CASCOR_ prefix (e.g., JUNIPER_CASCOR_PORT).
    """

    model_config = SettingsConfigDict(
        env_prefix=_JUNIPER_CASCOR_ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    host: str = _JUNIPER_CASCOR_API_HOST_DEFAULT
    port: int = _JUNIPER_CASCOR_API_PORT_DEFAULT
    log_level: str = _JUNIPER_CASCOR_API_LOGLEVEL_DEFAULT
    cors_origins: list[str] = _JUNIPER_CASCOR_API_CORS_ORIGINS_DEFAULT

    ws_max_connections: int = _JUNIPER_CASCOR_API_WS_MAX_CONNECTIONS_DEFAULT
    # SEC-03: per-IP cap on concurrent WebSocket connections. Mirrors the
    # canopy ``max_connections_per_ip`` pattern so a single hostile client
    # cannot monopolize the global ``ws_max_connections`` pool. Applies to
    # every endpoint routed through ``WebSocketManager.connect*``.
    ws_max_connections_per_ip: int = 5
    ws_heartbeat_interval_sec: int = _JUNIPER_CASCOR_API_WS_HEARTBEAT_INTERVAL_SEC_DEFAULT
    ws_heartbeat_pong_timeout_sec: int = _JUNIPER_CASCOR_API_WS_HEARTBEAT_PONG_TIMEOUT_SEC_DEFAULT

    # Phase 0-cascor: WebSocket sequencing, replay, and resume
    # Kill-switch env vars use JUNIPER_WS_* prefix per canonical plan §3.1
    ws_replay_buffer_size: int = Field(
        default=_JUNIPER_CASCOR_API_WS_REPLAY_BUFFER_SIZE_DEFAULT,
        description="Maximum number of messages in the WebSocket replay buffer (0 disables replay)",
        ge=0,
        validation_alias=AliasChoices("ws_replay_buffer_size", "JUNIPER_WS_REPLAY_BUFFER_SIZE"),
    )
    ws_send_timeout_seconds: float = Field(
        default=_JUNIPER_CASCOR_API_WS_SEND_TIMEOUT_SECONDS_DEFAULT,
        description="Timeout in seconds for individual WebSocket send operations (GAP-WS-07)",
        gt=0,
        validation_alias=AliasChoices("ws_send_timeout_seconds", "JUNIPER_WS_SEND_TIMEOUT_SECONDS"),
    )
    ws_resume_handshake_timeout_s: float = Field(
        default=_JUNIPER_CASCOR_API_WS_RESUME_HANDSHAKE_TIMEOUT_S_DEFAULT,
        description="Timeout in seconds for the resume handshake window on /ws/training",
        gt=0,
    )
    ws_state_throttle_coalesce_ms: int = Field(
        default=_JUNIPER_CASCOR_API_WS_STATE_THROTTLE_COALESCE_MS_DEFAULT,
        description="Minimum interval in milliseconds between non-terminal state broadcasts",
        gt=0,
    )
    ws_pending_max_duration_s: float = Field(
        default=_JUNIPER_CASCOR_API_WS_PENDING_MAX_DURATION_S_DEFAULT,
        description="Maximum duration in seconds a connection can stay in pending state",
        gt=0,
    )
    ws_initial_metrics_count: int = Field(
        default=100,
        description=(
            "GAP-WS-16: number of recent metrics to send as initial_metrics on fresh "
            "/ws/training connect. 0 disables the initial burst (clients must request "
            "via subscribe_metrics or fall back to REST)."
        ),
        ge=0,
        validation_alias=AliasChoices("ws_initial_metrics_count", "JUNIPER_WS_INITIAL_METRICS_COUNT"),
    )

    api_keys: list[str] | None = _JUNIPER_CASCOR_API_KEYS_LIST_EMPTY

    @model_validator(mode="before")
    @classmethod
    def _load_api_keys_from_secret_file(cls, data: Any) -> Any:
        """Load api_keys from a Docker secrets file when not set directly.

        Checks for ``JUNIPER_CASCOR_API_KEYS_FILE`` (via ``get_secret``)
        and injects the value into the data dict so Pydantic can parse it.
        """
        if isinstance(data, dict) and not data.get("api_keys"):
            secret_value = get_secret("JUNIPER_CASCOR_API_KEYS")
            if secret_value:
                data["api_keys"] = secret_value
        return data

    @field_validator("api_keys", mode="before")
    @classmethod
    def _empty_string_to_none(cls, v: Any) -> list[str] | None:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    rate_limit_enabled: bool = _JUNIPER_CASCOR_API_RATELIMIT_DISABLED
    rate_limit_requests_per_minute: int = _JUNIPER_CASCOR_API_RATELIMIT_DEFAULT

    log_format: str = _JUNIPER_CASCOR_API_LOG_FORMAT_DEFAULT
    sentry_dsn: str | None = _JUNIPER_CASCOR_API_SENTRY_DSN_DEFAULT
    metrics_enabled: bool = _JUNIPER_CASCOR_API_METRICS_ENABLED_DEFAULT

    auto_start: bool = _JUNIPER_CASCOR_API_AUTO_START_DEFAULT
    auto_start_data_service: bool = _JUNIPER_CASCOR_API_AUTO_START_DATA_SERVICE_DEFAULT
    auto_start_data_service_command: str = _JUNIPER_CASCOR_API_AUTO_START_DATA_SERVICE_COMMAND_DEFAULT
    auto_start_canopy: bool = _JUNIPER_CASCOR_API_AUTO_START_CANOPY_DEFAULT
    auto_start_canopy_command: str = _JUNIPER_CASCOR_API_AUTO_START_CANOPY_COMMAND_DEFAULT
    auto_dataset: str = _JUNIPER_CASCOR_API_AUTO_DATASET_DEFAULT
    auto_dataset_params: str = _JUNIPER_CASCOR_API_AUTO_DATASET_PARAMS_DEFAULT
    auto_network: str = _JUNIPER_CASCOR_API_AUTO_NETWORK_DEFAULT
    auto_train_epochs: int = _JUNIPER_CASCOR_API_AUTO_TRAIN_EPOCHS_DEFAULT

    # Remote WebSocket worker configuration
    remote_workers_heartbeat_timeout: float = _JUNIPER_CASCOR_API_REMOTE_WORKERS_HEARTBEAT_TIMEOUT_DEFAULT
    remote_workers_task_reassignment_timeout: float = _JUNIPER_CASCOR_API_REMOTE_WORKERS_TASK_REASSIGNMENT_TIMEOUT_DEFAULT

    # Worker Security (Phase 4) — all disabled by default for zero behavior change
    worker_rate_limit_enabled: bool = _JUNIPER_CASCOR_API_WORKER_RATE_LIMIT_ENABLED_DEFAULT
    worker_rate_limit_connections_per_minute: int = _JUNIPER_CASCOR_API_WORKER_RATE_LIMIT_CONNECTIONS_PER_MINUTE_DEFAULT
    worker_rate_limit_burst_size: int = _JUNIPER_CASCOR_API_WORKER_RATE_LIMIT_BURST_SIZE_DEFAULT

    worker_anomaly_detection_enabled: bool = _JUNIPER_CASCOR_API_WORKER_ANOMALY_DETECTION_ENABLED_DEFAULT
    worker_anomaly_min_training_time: float = _JUNIPER_CASCOR_API_WORKER_ANOMALY_MIN_TRAINING_TIME_DEFAULT
    worker_anomaly_perfect_corr_threshold: float = _JUNIPER_CASCOR_API_WORKER_ANOMALY_PERFECT_CORR_THRESHOLD_DEFAULT

    worker_audit_logging_enabled: bool = _JUNIPER_CASCOR_API_WORKER_AUDIT_LOGGING_ENABLED_DEFAULT
    worker_metrics_enabled: bool = _JUNIPER_CASCOR_API_WORKER_METRICS_ENABLED_DEFAULT

    # Phase B-pre-b: Control-path security (§S8)
    ws_control_allowed_origins: list[str] = [
        "http://localhost:8050",
        "http://127.0.0.1:8050",
        "https://localhost:8050",
        "https://127.0.0.1:8050",
    ]
    ws_control_rate_limit_per_sec: int = 10  # leaky bucket: 10 tokens, 10/s refill
    ws_control_idle_timeout_sec: int = 120  # bidirectional idle timeout
    ws_control_cooldown_rejections: int = 10  # rejections in cooldown window before IP block
    ws_control_cooldown_window_sec: int = 60  # cooldown window in seconds
    ws_control_cooldown_block_sec: int = 300  # 5-minute IP block after cooldown triggers
    disable_ws_control_endpoint: bool = False  # emergency kill switch (CSWSH hard-disable)


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
