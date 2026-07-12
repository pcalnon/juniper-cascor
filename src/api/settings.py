"""API configuration settings using pydantic-settings."""

from functools import lru_cache
from typing import Annotated, Any

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

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

_JUNIPER_CASCOR_API_WS_EMISSION_SUMMARY_INTERVAL_SEC: float = 60.0
_JUNIPER_CASCOR_API_WS_EMISSION_SUMMARY_INTERVAL_SEC_DEFAULT: float = _JUNIPER_CASCOR_API_WS_EMISSION_SUMMARY_INTERVAL_SEC

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
# Default OFF. Auto-start trains on boot onto a default (empty) network, which violates
# the clean-STOPPED initial-state assumption every API / Canopy / automation caller makes
# (it left the #319 probe staring at epoch 7576 / 0 hidden units on a fresh stack). It is
# a demo/dev convenience only; juniper-deploy's demo opts in explicitly via
# JUNIPER_CASCOR_AUTO_START=true, so flipping the default does NOT change the demo stack.
# See notes/CASCOR_STARTUP_SECRET_INDIRECTION_INVESTIGATION_2026-06-14.md (3.3).
_JUNIPER_CASCOR_API_AUTO_START_DEFAULT: bool = _JUNIPER_CASCOR_API_AUTO_START_DISABLED

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

    # SEC-F22 / D2 — startup bind-guard attestation (two-flag scheme, identical
    # across canopy / cascor / juniper-deploy). Both default False
    # (fail-closed). When the service is configured to bind a NON-loopback
    # interface (``host`` not 127.0.0.0/8, ::1, localhost, or an IPv4-mapped
    # loopback), the startup bind-guard refuses to start unless AT LEAST ONE of
    # these attestations is True. Each names a *distinct* reason a non-loopback
    # bind is safe, so operators attest the control that actually applies
    # instead of one conflated flag:
    #
    # * ``loopback_publish_attested`` (env
    #   ``JUNIPER_CASCOR_LOOPBACK_PUBLISH_ATTESTED``) — the port is reachable
    #   ONLY via a loopback-only host publish (the containerized default:
    #   ``127.0.0.1:8200:8200`` in compose; verifiable by the juniper-deploy
    #   preflight).
    # * ``auth_proxy_attested`` (env ``JUNIPER_CASCOR_AUTH_PROXY_ATTESTED``) — a
    #   fronting authenticating reverse proxy terminates access before the port
    #   (Phase-4; attestation only, no in-process verification).
    #
    # Loopback binds always start. This turns the load-bearing "loopback is the
    # trust boundary" precondition into an enforced invariant and closes the
    # silent ``JUNIPER_CASCOR_HOST=0.0.0.0`` footgun. See juniper-ml
    # ``notes/JUNIPER_CANOPY_CONTROL_SURFACE_AUTH_AND_NAT_DESIGN_2026-07-03.md``
    # §4 Option A / §8 D2 and this repo's
    # ``notes/JUNIPER_CASCOR_CONTROL_SURFACE_AUTH_AND_NAT_SECURITY_NOTE_2026-07-04.md``.
    loopback_publish_attested: bool = False
    auth_proxy_attested: bool = False

    log_level: str = _JUNIPER_CASCOR_API_LOGLEVEL_DEFAULT
    cors_origins: list[str] = _JUNIPER_CASCOR_API_CORS_ORIGINS_DEFAULT

    ws_max_connections: int = _JUNIPER_CASCOR_API_WS_MAX_CONNECTIONS_DEFAULT
    # SEC-F19 / D4a — stack-absolute GLOBAL WebSocket connection cap across
    # ALL WS endpoints (/ws/training, /ws/control, /ws/v1/workers combined),
    # env ``JUNIPER_CASCOR_WS_MAX_CONNECTIONS_GLOBAL``. Bounds total socket
    # resource regardless of source IP or identity — the availability /
    # DoS-dampening backstop that keeps working when the per-IP cap collapses
    # behind Docker NAT (every client presents as the bridge gateway). Must
    # exceed the expected legitimate total (training clients + control +
    # the worker fleet). Best-effort, NOT authentication.
    ws_max_connections_global: int = 200
    # SEC-F19 / D4b — per-identity WebSocket connection cap, keyed on the
    # authenticated principal (the presented ``X-API-Key`` token hash) where
    # one exists, env ``JUNIPER_CASCOR_WS_MAX_CONNECTIONS_PER_IDENTITY``.
    # Enforced on /ws/control (a small-cardinality principal — canopy/browser)
    # alongside the global cap. NOT enforced on /ws/v1/workers: a worker fleet
    # shares one token so keying on it would cap horizontal scaling, and the
    # unique server-assigned worker_id is only known post-registration — so
    # meaningful worker per-identity keying is not cleanly available and the
    # global cap is the worker minimum (design §8 OQ-2; a follow-up).
    # Best-effort fairness, NOT authentication.
    ws_max_connections_per_identity: int = 5
    # SEC-03 / SEC-F19 — per-IP cap on concurrent WebSocket connections.
    # Mirrors the canopy ``max_connections_per_ip`` pattern so a single
    # hostile client cannot monopolize the global ``ws_max_connections`` pool.
    # Applies to every endpoint routed through ``WebSocketManager.connect*``
    # (today: /ws/training).
    #
    # NOTE (inert-behind-NAT): keyed on the raw socket peer
    # (``websocket.client[0]``), this cap is DoS-dampening only, NOT
    # authentication. Behind Docker NAT every client collapses to the bridge
    # gateway IP, so the cap becomes a single shared bucket (one client's 5
    # sockets exhaust it for everyone — the HO-3 self-DoS). The stack-absolute
    # ``ws_max_connections_global`` + per-identity caps are the controls that
    # survive NAT; genuine per-client identity requires the deferred
    # fronting-proxy + trusted-XFF milestone (design §5, D6).
    ws_max_connections_per_ip: int = 5
    # Phase F / C3 — the application-level WebSocket heartbeat contract:
    # /ws/training and /ws/control send ``{"type":"ping","ts":<float>}`` every
    # ``ws_heartbeat_interval_sec`` seconds; a client that sends NOTHING (a
    # ``{"type":"pong"}`` reply, or any other well-formed frame — C3 counts
    # any inbound traffic as liveness) within ``ws_heartbeat_pong_timeout_sec``
    # seconds of a ping is closed with 1011 ("Heartbeat timeout"). Set the
    # interval <= 0 to disable the heartbeat entirely (escape hatch for legacy
    # clients that cannot answer pings; the control-channel idle timeout still
    # applies). juniper-cascor-client >= 0.7.0 answers pings automatically.
    ws_heartbeat_interval_sec: int = Field(
        default=_JUNIPER_CASCOR_API_WS_HEARTBEAT_INTERVAL_SEC_DEFAULT,
        description="Seconds between application-level WS heartbeat pings on /ws/training and /ws/control (<= 0 disables the heartbeat)",
        validation_alias=AliasChoices("ws_heartbeat_interval_sec", "JUNIPER_WS_HEARTBEAT_INTERVAL_SEC"),
    )
    ws_heartbeat_pong_timeout_sec: int = Field(
        default=_JUNIPER_CASCOR_API_WS_HEARTBEAT_PONG_TIMEOUT_SEC_DEFAULT,
        description="Seconds after a heartbeat ping within which the client must send a pong (or any frame — C3 tolerance) before the connection is closed with 1011",
        gt=0,
        validation_alias=AliasChoices("ws_heartbeat_pong_timeout_sec", "JUNIPER_WS_HEARTBEAT_PONG_TIMEOUT_SEC"),
    )
    ws_emission_summary_interval_sec: float = Field(
        default=_JUNIPER_CASCOR_API_WS_EMISSION_SUMMARY_INTERVAL_SEC_DEFAULT,
        description="C3/T5: minimum seconds between periodic INFO summaries of WS frames emitted by type (checked on each send, incl. heartbeat pings); <= 0 disables the summary",
        validation_alias=AliasChoices("ws_emission_summary_interval_sec", "JUNIPER_WS_EMISSION_SUMMARY_INTERVAL_SEC"),
    )

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
        description=("GAP-WS-16: number of recent metrics to send as initial_metrics on fresh " "/ws/training connect. 0 disables the initial burst (clients must request " "via subscribe_metrics or fall back to REST)."),
        ge=0,
        validation_alias=AliasChoices("ws_initial_metrics_count", "JUNIPER_WS_INITIAL_METRICS_COUNT"),
    )
    ws_max_message_size_bytes: int = Field(
        default=60_000,
        description=("GAP-WS-18: serialized JSON size threshold above which broadcasts are " "split into chunked_message envelopes. Default 60_000 leaves headroom " "below the WebSocket 64 KB per-frame limit imposed by some intermediaries. " "Set to 0 to disable chunking entirely (oversized broadcasts will be sent " "as-is and may cause silent connection teardown — only use for tests)."),
        ge=0,
        validation_alias=AliasChoices("ws_max_message_size_bytes", "JUNIPER_WS_MAX_MESSAGE_SIZE_BYTES"),
    )
    ws_chunk_payload_size_bytes: int = Field(
        default=32_000,
        description=("GAP-WS-18: maximum payload bytes per chunk when chunking is triggered. " "Default 32_000 keeps each chunked envelope (payload + JSON wrapper) safely " "under 60_000. Must be > 0."),
        gt=0,
        validation_alias=AliasChoices("ws_chunk_payload_size_bytes", "JUNIPER_WS_CHUNK_PAYLOAD_SIZE_BYTES"),
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

    @field_validator("ws_control_allowed_origins", mode="before")
    @classmethod
    def _parse_ws_control_allowed_origins(cls, v: Any) -> list[str]:
        """Normalise ``ws_control_allowed_origins`` to ``list[str]``.

        Accepts:

        * ``None`` → empty list (caller has explicitly removed all origins).
          Most operators want to keep the *default* list; the default is
          materialised via the ``Field(default=...)`` mechanism and never
          reaches this validator when the env var is unset.
        * Plain string ``"http://x:1,http://y:2"`` → ``["http://x:1", "http://y:2"]``
          (CSV form, mirrors ``_parse_api_keys``).
        * Plain string ``'["http://x:1","http://y:2"]'`` → list (JSON-array
          form, the default pydantic-settings env-coercion shape).
        * Already-list input → returned as-is (covers the programmatic
          ``Settings(ws_control_allowed_origins=[...])`` callsite used by
          tests + the ``Field(default=...)`` path when the env var is unset).
        """
        if v is None:
            return []
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            # Try JSON-array first to preserve compatibility with the
            # pydantic-settings auto-coercion for ``list[str]`` env vars.
            if s.startswith("[") and s.endswith("]"):
                import json as _json

                try:
                    parsed = _json.loads(s)
                except (_json.JSONDecodeError, ValueError):
                    parsed = None
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            # Fall back to comma-CSV.
            return [item.strip() for item in s.split(",") if item.strip()]
        return list(v)

    @field_validator("api_keys", mode="before")
    @classmethod
    def _parse_api_keys(cls, v: Any) -> list[str] | None:
        """Normalise ``api_keys`` to ``list[str] | None``.

        Accepts (in order of likelihood):

        * ``None`` or empty string → ``None`` (auth disabled).
        * Plain string ``"key1,key2"`` → ``["key1", "key2"]`` (the format
          a single-string Docker secret file naturally produces; matches
          the juniper-data parser).
        * Already a list (including JSON-deserialised lists from
          ``pydantic-settings``) → returned as-is.

        Without this validator the bare string read from a Docker secrets
        file (e.g. ``CHANGE_BEFORE_PRODUCTION_USE`` from
        ``secrets.example/juniper_cascor_api_keys.txt``) would hit
        pydantic's ``list[str]`` coercion and fail with
        ``ValidationError: api_keys ... Input should be a valid list``,
        putting the ``juniper-cascor`` container into a restart loop on
        any default ``docker compose up``. Mirrors the
        ``_parse_api_keys`` validator on ``juniper-data``'s ``Settings``.
        """
        if v is None or v == "":
            return None
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v  # type: ignore[return-value]

    rate_limit_enabled: bool = _JUNIPER_CASCOR_API_RATELIMIT_DISABLED
    rate_limit_requests_per_minute: int = _JUNIPER_CASCOR_API_RATELIMIT_DEFAULT

    log_format: str = _JUNIPER_CASCOR_API_LOG_FORMAT_DEFAULT
    sentry_dsn: str | None = _JUNIPER_CASCOR_API_SENTRY_DSN_DEFAULT
    metrics_enabled: bool = _JUNIPER_CASCOR_API_METRICS_ENABLED_DEFAULT

    # SEC-16 parity with juniper-data: loopback-only by default. Set
    # ``JUNIPER_CASCOR_METRICS_TRUSTED_IPS='["10.0.0.5","172.18.0.0/16"]'``
    # (JSON list) or a comma-separated string. Accepts bare IP literals and
    # CIDR ranges; the in-process ``MetricsAuthMiddleware`` normalises IPv6
    # zone-ids and IPv4-mapped IPv6 client addresses before membership check,
    # so a Docker container appearing as ``::ffff:172.18.0.5`` matches an
    # IPv4 ``172.18.0.0/16`` allowlist entry. Mirrors
    # ``juniper-data.api.settings.metrics_trusted_ips`` (SEC-16, POC §3.1).
    metrics_trusted_ips: list[str] = ["127.0.0.1", "::1"]

    @field_validator("metrics_trusted_ips")
    @classmethod
    def _validate_metrics_trusted_ips(cls, v: list[str]) -> list[str]:
        """Fail loud at startup if any allowlist entry is unparseable.

        Without this guard a typo like ``172.18.0.0/164`` would silently
        compile to a working-but-empty allowlist that 403s every scrape.
        Uses the shared ``parse_trusted_networks`` from
        ``juniper-observability`` so cascor's fail-loud message stays in
        lockstep with juniper-data and any future consumer.
        """
        from juniper_observability import parse_trusted_networks

        parse_trusted_networks(v)
        return v

    # CFG-04: JuniperData service URL. Canonical cross-service env var
    # is ``JUNIPER_DATA_URL`` (unprefixed) — shared by juniper-data,
    # juniper-canopy, and juniper-cascor — so we expose it via
    # ``AliasChoices`` rather than the default ``env_prefix='JUNIPER_CASCOR_'``
    # lookup (which would force operators to write the awkward
    # ``JUNIPER_CASCOR_JUNIPER_DATA_URL``). ``AliasChoices`` replaces
    # the prefix-derived lookup entirely, so the prefixed form is
    # additionally listed for parity with the other ``Settings`` fields
    # and to keep the ``JUNIPER_CASCOR_`` prefix viable for operators
    # who want a per-service override. Default is ``None`` — callers
    # that want the legacy ``http://localhost:8100`` fallback should
    # apply it explicitly via
    # ``settings.juniper_data_url or _PROJECT_API_JUNIPER_DATA_URL_DEFAULT``;
    # callers where the URL is required (e.g., ``main.py`` pre-flight,
    # ``SpiralProblem``) check for ``None`` and fail loudly. This field
    # consolidates the 8 raw ``os.environ.get('JUNIPER_DATA_URL')`` call
    # sites that drifted across ``src/api/app.py``, ``src/main.py``,
    # ``src/api/routes/health.py``, ``src/api/lifecycle/manager.py``,
    # ``src/spiral_problem/spiral_problem.py``, and
    # ``src/spiral_problem/data_provider.py`` into a single validated
    # config surface.
    juniper_data_url: str | None = Field(
        default=None,
        description=("JuniperData service URL (e.g., 'http://localhost:8100'). " "Canonical env var: JUNIPER_DATA_URL (unprefixed, ecosystem-wide). " "Per-service override: JUNIPER_CASCOR_JUNIPER_DATA_URL."),
        validation_alias=AliasChoices("juniper_data_url", "JUNIPER_DATA_URL", "JUNIPER_CASCOR_JUNIPER_DATA_URL"),
    )

    auto_start: bool = _JUNIPER_CASCOR_API_AUTO_START_DEFAULT
    auto_start_data_service: bool = _JUNIPER_CASCOR_API_AUTO_START_DATA_SERVICE_DEFAULT
    auto_start_data_service_command: str = _JUNIPER_CASCOR_API_AUTO_START_DATA_SERVICE_COMMAND_DEFAULT
    auto_start_canopy: bool = _JUNIPER_CASCOR_API_AUTO_START_CANOPY_DEFAULT
    auto_start_canopy_command: str = _JUNIPER_CASCOR_API_AUTO_START_CANOPY_COMMAND_DEFAULT
    auto_dataset: str = _JUNIPER_CASCOR_API_AUTO_DATASET_DEFAULT
    auto_dataset_params: str = _JUNIPER_CASCOR_API_AUTO_DATASET_PARAMS_DEFAULT
    auto_network: str = _JUNIPER_CASCOR_API_AUTO_NETWORK_DEFAULT
    # DEPRECATED (C2b / Q1): previously seeded the network's ``epochs_max``
    # attribute on the auto-start path — an attribute the training loop never
    # read (the granular limits gate training; ``epochs_max`` is now derived
    # from them, see TrainingLifecycleManager.derive_epochs_cap). Retained as
    # a no-op so existing JUNIPER_CASCOR_AUTO_TRAIN_EPOCHS env vars do not
    # break Settings construction; slated for removal.
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
    #
    # Env binding: ``JUNIPER_CASCOR_WS_CONTROL_ALLOWED_ORIGINS`` (resolved
    # automatically via the ``env_prefix="JUNIPER_CASCOR_"`` model config).
    # Two input shapes are accepted:
    #
    # 1. JSON-array string: ``'["http://x:1","http://y:2"]'``
    #    (the form ``pydantic-settings`` produces by default for
    #    ``list[str]`` env vars).
    # 2. Comma-CSV string: ``'http://x:1,http://y:2'``
    #    (the more operator-friendly form; matches the ``_parse_api_keys``
    #    convention for ``JUNIPER_CASCOR_API_KEYS``).
    #
    # The CSV path closes the regression class where juniper-canopy
    # (running inside docker compose) connects to ``/ws/control`` from
    # the container hostname → the Origin is ``http://juniper-canopy:8050``
    # which is **not** in the default allowlist. juniper-deploy E.2 PR-2-D
    # sets the env var to ``http://juniper-canopy:8050,http://localhost:8050,…``
    # in compose; without the CSV parser the operator would have to
    # JSON-escape the list, which is brittle for env-file authoring.
    #
    # See juniper-ml ``notes/STACK_REGRESSION_CORRECTIONS_2026-05-27.md``
    # §E.2 PR-2-B for the regression + remediation context.
    # ``Annotated[..., NoDecode]`` tells pydantic-settings to *skip* its
    # default JSON decoding for this env var (which would raise
    # ``SettingsError`` on CSV-form input before our ``mode='before'``
    # validator gets a chance to run).  The validator below then handles
    # both JSON-array and comma-CSV forms uniformly.
    ws_control_allowed_origins: Annotated[list[str], NoDecode] = Field(
        default=[
            "http://localhost:8050",
            "http://127.0.0.1:8050",
            "https://localhost:8050",
            "https://127.0.0.1:8050",
        ],
        description=("WebSocket /ws/control Origin allowlist. " "Env var: JUNIPER_CASCOR_WS_CONTROL_ALLOWED_ORIGINS. " "Accepts JSON-array (default ``pydantic-settings`` form) or comma-CSV. " "Empty string disables the allowlist (use only when explicitly opting out)."),
        validation_alias=AliasChoices("ws_control_allowed_origins", "JUNIPER_CASCOR_WS_CONTROL_ALLOWED_ORIGINS"),
    )
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
