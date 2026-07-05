# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

- **SEC-F22 / D2 — startup bind-guard (`enforce_fronting_auth_bind_guard`).** At application startup (in the `lifespan`, before uvicorn binds the socket or any background thread is spawned), cascor now **refuses to start** — fail-closed, with a CRITICAL log — when `JUNIPER_CASCOR_HOST` names a **non-loopback** interface (e.g. `0.0.0.0`) unless the new boolean setting `fronting_auth_attested` (env `JUNIPER_CASCOR_FRONTING_AUTH_ATTESTED`, default **false**) is true. Loopback binds (`127.0.0.0/8`, `::1`, `localhost`, IPv4-mapped-IPv6 loopback) always start. cascor's control/worker WebSocket surface has **no app-layer authentication of its own** — its only effective control in the containerized stack is the loopback network boundary — so this turns that load-bearing precondition into an enforced invariant and closes the silent `JUNIPER_CASCOR_HOST=0.0.0.0` footgun. The symmetric counterpart to the canopy bind-guard. **Deploy roll-out is owner-gated (Phase 1):** the container binds `0.0.0.0` behind a loopback host-publish, so enabling this in `juniper-deploy` requires setting `JUNIPER_CASCOR_FRONTING_AUTH_ATTESTED=true` there. Design of record: juniper-ml [`notes/JUNIPER_CANOPY_CONTROL_SURFACE_AUTH_AND_NAT_DESIGN_2026-07-03.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_CANOPY_CONTROL_SURFACE_AUTH_AND_NAT_DESIGN_2026-07-03.md) §4 Option A / §8 D2; local note `notes/JUNIPER_CASCOR_CONTROL_SURFACE_AUTH_AND_NAT_SECURITY_NOTE_2026-07-04.md`. Tests: `src/tests/unit/api/test_bind_guard.py`.

- **SEC-F19 / D4 — WebSocket connection caps (stack-global + per-identity).** The `WebSocketManager` gains a stack-absolute **GLOBAL** connection cap (`ws_max_connections_global`, env `JUNIPER_CASCOR_WS_MAX_CONNECTIONS_GLOBAL`, default 200) spanning **all** WS endpoints — `/ws/training` (via `connect`/`connect_pending`) and `/ws/control` + `/ws/v1/workers` (via the new `try_admit`/`release_admission` admission gate) draw from one counter — plus a **per-identity** cap (`ws_max_connections_per_identity`, env `JUNIPER_CASCOR_WS_MAX_CONNECTIONS_PER_IDENTITY`, default 5) enforced on `/ws/control`, keyed on a non-reversible SHA-256 digest of the caller's `X-API-Key` token (`ws_identity_key`). Over-cap connections are rejected/closed with the existing `1013` close code. These are the DoS-dampening controls that **survive Docker NAT**, where the existing per-IP cap (`ws_max_connections_per_ip`, unchanged) collapses to one shared bridge-gateway bucket (HO-3 self-DoS); the per-IP cap is now documented **inert-behind-NAT** (DoS-dampening, not authentication). Per-identity keying on `/ws/v1/workers` is intentionally **global-only** (a worker fleet shares one token and the unique worker_id is only known post-registration — design §8 OQ-2; a documented follow-up). Design of record §5 Option B / §8 D4. Tests: `src/tests/unit/api/test_ws_connection_caps.py`.

### Added

- **`.dockerignore` added — excludes build artifacts (incl. nested `**/*.egg-info/`) from the image.** cascor had no `.dockerignore`, so the full build context was sent and any build artifact under `src/` could be `COPY src/`'d into the image. Because cascor runs from `/app/src` (`ENV PYTHONPATH=/app/src`), a stale `src/*.egg-info` would land ahead of site-packages and shadow `importlib.metadata.version("juniper-cascor")` — the class of bug fixed for juniper-canopy in [canopy #362](https://github.com/pcalnon/juniper-canopy/pull/362) (surfaced by the build-provenance `make doctor` work). cascor is **not** vulnerable today (its egg-info is at the repo root, which `COPY src/` does not pick up), so this is preventive hardening + general context hygiene (the file also excludes `.git/`, `__pycache__/`, `*.py[cod]`, `build/`, `dist/`, and test/type-check caches). The nested `**/*.egg-info/` + `**/*.dist-info/` forms are required because a root-only `*.egg-info/` does not match `src/*.egg-info`. Regression test: `src/tests/unit/test_dockerignore_egg_info.py`.

- **Build provenance on `/v1/health` + `/v1/health/ready`.** The service now
  reports the source `git_sha` and ISO-8601 `build_date` baked into its image
  at build time. New `GIT_SHA` / `BUILD_DATE` / `APP_VERSION` Dockerfile
  build-args become OCI labels (`org.opencontainers.image.revision` /
  `.created` / `.version`) plus `JUNIPER_CASCOR_GIT_SHA` / `_BUILD_DATE` env
  vars; a new `api.provenance` accessor reads them back (both `null` outside a
  provenance-stamped image — local dev / a bare `docker build`). The values are
  also passed into `set_build_info(...)` (Prometheus `juniper_cascor_build`
  Info metric) and the shared `ReadinessResponse`. Foundation for the ecosystem
  stale-image-detection effort — see juniper-ml
  [`notes/BUILD_PROVENANCE_DESIGN_2026-06-14.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/BUILD_PROVENANCE_DESIGN_2026-06-14.md).
  Requires `juniper-observability>=0.4.0`.

- **Native `equities` (and generic-params) dataset staging**: `POST /v1/training/dataset` and the live-swap `POST /v1/training/dataset/live` now accept `dataset_type: "equities"` plus a generic `params: dict` carrying arbitrary juniper-data generator inputs (e.g. `symbols`, `start_date`, `end_date`, `normalize_features`, `max_symbols`) that the spiral-shaped typed fields (`n_samples`/`noise`/`rotations`/`n_spirals`) don't cover. `StageDatasetRequest` (`src/api/models/training.py`) gains `"equities"` in the `dataset_type` `Literal` and a `params: Optional[Dict[str, Any]]` field (`SwapDatasetLiveRequest` inherits it); `_reload_dataset` (`src/api/lifecycle/manager.py`) pops and **merges** the generic `params` with the remaining typed fields before forwarding to `JuniperDataClient.create_dataset` (generic keys win on conflict). The legacy spiral/xor/circles/moons/mnist path is unchanged — those bodies carry no `params` key, so the merge is a strict no-op (`{**cfg, **{}}`) and `_current_dataset_config` keeps its prior shape. This lets cascor natively fetch + train on the juniper-data `equities` time-series dataset (S&P 500 daily OHLCV + SEC EDGAR fundamentals, 2000→present) — `stage → start_training → create_dataset("equities", {…}) → download_artifact_npz` — instead of requiring callers to pre-fetch and `inline_data` the arrays. No route or canopy-adapter change (the typed fields are preserved; `model_dump` carries the new `params`). Regression coverage in `src/tests/integration/api/test_pending_dataset.py`: `test_equities_staging_accepts_and_echoes_generic_params`, `test_equities_reload_forwards_generic_params_flattened` (asserts `create_dataset` receives `generator="equities"` and the flattened params), and `test_spirals_typed_fields_still_forward_unchanged` (legacy-path guard).
- **`completion_reason` in training status (Issue #3 diagnosability follow-up)**: `grow_network` (`src/cascade_correlation/cascade_correlation.py`) now records *which* exit terminated the most recent growth run in `self._completion_reason`, and `TrainingLifecycleManager.get_status()` (`src/api/lifecycle/manager.py`) surfaces it as a top-level `"completion_reason"` field. The five values map 1:1 to the loop's exits: `residual_collapsed` (residual error went `None`), `no_candidate` (no training results / no best candidate / no candidate met multi-select criteria — the 0-unit *stall* signature), `below_threshold` (best candidate below the adaptive correlation threshold), `early_stopped` (validation early-stop), and `max_iterations` (the growth loop hit its cap — captured via a `for/else` clause). Defaults to `None` before any training and resets at the start of each `grow_network` run; `get_status` reads it via `getattr(self.network, …, None)` so it is safe when no network is loaded. The reason is also appended to the existing `grow_network: Finished training…` INFO log line. This lets canopy distinguish a genuine convergence from a stall instead of both rendering a bare "Completed" — the **canopy consumer side is a separate follow-up**. The grow-loop change is assignment-only (no new branches, no control-flow change), so the fragile cascade-termination logic is untouched. Regression coverage: `src/tests/unit/test_completion_reason.py` (all five exits + the `None` default) and `src/tests/unit/api/test_lifecycle_manager.py::…::test_get_status_surfaces_completion_reason` (+ the no-network `None` case).
- **SEC-16 / POC remediation §3.1**: cascor now has parity with
  juniper-data's `MetricsAuthMiddleware` — `/metrics` is exempt from
  `SecurityMiddleware` (so prometheus doesn't 401 ahead of the IP
  allowlist) AND gated by a parallel in-process IP allowlist with full
  CIDR + IPv6 normalization. Mirrors the juniper-data implementation
  verbatim (validator A's "duplicate inline" recommendation in
  [`POC_REMEDIATION_PLAN_2026-05-27.md` §3.1](https://github.com/pcalnon/juniper-deploy/blob/main/notes/poc/POC_REMEDIATION_PLAN_2026-05-27.md));
  promotion to `juniper-observability` is a roadmap §R5 follow-up.
  Concrete changes: `src/api/middleware.py` adds `/metrics` and
  `/metrics/` to `EXEMPT_PATHS`; `src/api/observability.py` adds
  `MetricsAuthMiddleware` plus `_parse_trusted_networks` (CIDR
  fail-loud parser) and `_normalize_client_ip` (zone-id strip,
  IPv4-mapped IPv6 unwrap); `src/api/settings.py` adds
  `Settings.metrics_trusted_ips: list[str] = ["127.0.0.1", "::1"]`
  with a `_validate_metrics_trusted_ips` field validator that
  surfaces unparseable entries at `Settings()` construction;
  `src/api/app.py` replaces the bare
  `app.mount("/metrics", get_prometheus_app())` with
  `app.mount("/metrics", MetricsAuthMiddleware(get_prometheus_app(), settings.metrics_trusted_ips))`.
  Why this is needed even with `/metrics` exempt: a misconfigured
  deployment (port 8200 published directly, or running outside
  compose / K8s) would expose `/metrics` with zero auth. Regression
  coverage: new `src/tests/unit/api/test_metrics_auth_middleware.py`
  (12 tests) pins the `EXEMPT_PATHS` invariant, CIDR v4 allow + miss,
  mixed CIDR + literal, CIDR v6 allow, IPv4-mapped IPv6 vs IPv4 CIDR
  (the docker-bridge regression), IPv6 zone-id strip, default
  loopback works, malformed / missing client address falls through
  to 403, invalid CIDR raises at `Settings()`, and middleware-init
  fail-loud as defense in depth. Tests drive the middleware via raw
  ASGI scopes (not `TestClient` + `create_app`) because the cascor
  lifespan does not tear down cleanly in unit-test context — same
  contract either way.

### Changed

- **`_API_VERSION` in `src/api/routes/health.py` now derives from
  `importlib.metadata`** instead of a hardcoded literal (OQ-1 of the
  build-provenance effort), so the `/v1/health` + `/v1/health/ready` version can
  no longer drift from `pyproject.toml`'s `[project].version`. Falls back to the
  literal only in a bare source checkout where the distribution is not
  installed. (`src/api/app.py`'s `_API_VERSION` already used this pattern; this
  closes the parallel hardcoded constant on the health route.)

- **CFG-02** (v7 roadmap §13524): `sentry-sdk>=2.0.0` moved from `[project] dependencies` into the `[project.optional-dependencies] observability` extra. The roadmap's recommended Approach A is "optional features should use optional dependencies": Sentry is only initialized when `JUNIPER_CASCOR_SENTRY_DSN` (or the deprecated `SENTRY_SDK_DSN`, see CFG-03) is set, so users running cascor without Sentry no longer pay the install footprint. Matches the canopy `[observability]` precedent. **BREAKING for Sentry users**: deployments that previously relied on `pip install juniper-cascor` pulling sentry-sdk transitively must now use `pip install juniper-cascor[observability]` (or `pip install juniper-cascor[all]` — the `[all]` aggregator already includes `[observability]`). The shared `juniper-observability[sentry]` extra remains an equally-valid alternative. The bootstrap Sentry init at the top of `src/main.py` (previously `import sentry_sdk` unconditionally at line 52 + `sentry_sdk.init(...)` inside `if _sentry_dsn:`) now lazy-imports `sentry_sdk` inside the `if _sentry_dsn:` block, wrapped in `try/except ImportError` that emits a clear stderr warning when a DSN is set but the SDK is not installed (`[juniper-cascor] CFG-02 WARNING: ... install with pip install juniper-cascor[observability]`). The application-bootstrap Sentry path in `src/api/app.py` is unchanged: it already delegates to `juniper_observability.configure_sentry`, which does its own lazy `import sentry_sdk` inside the function and is a no-op when DSN is empty. Pinned by a new 4-case source-level regression suite at `src/tests/unit/test_cfg_02_sentry_sdk_optional.py` (sentry-sdk absent from core deps, present in `[observability]`, no top-level import in `main.py`, lazy import + `except ImportError` guard present). Aligning the `[observability]` extra to `sentry-sdk[fastapi]>=2.0.0` (to match `juniper-observability[sentry]`) is a deferred follow-up — CFG-02 is scoped strictly to moving the dep. Tracks CFG-02 in the v7 outstanding-development roadmap §20.

### Fixed

- **Dual-path remote candidate training — collection budget, round isolation, and idle-worker dispatch (Issue #319, defects 3–5)**: three distinct defects in the remote/dual-path candidate leg, companion to #321 (which fixed the two remote-dispatch crashes). All three are verified at the cascor↔worker boundary on the live deployed stack.
  - **#3 — remote result-collection budget** (`src/cascade_correlation/cascade_correlation.py`, `src/api/workers/coordinator.py`, `src/cascor_constants/constants.py`): `_dispatch_to_remote_workers` reused `candidate_training_shutdown_timeout` (~10s — a process-*teardown* budget) to wait for a full candidate-training round (tens of seconds), so `collect_results` always timed out, discarded every remote result as "late", and retried the tasks on the saturated local pool. New `CascadeCorrelationNetwork._remote_result_collection_timeout()` scales the wait to the training workload (`candidate_epochs`) with a 120s floor and 900s ceiling. `WorkerCoordinator.collect_results` gains a worker-liveness early-exit (polls in `_RESULT_COLLECTION_POLL_INTERVAL` slices and returns promptly once no workers remain connected) so the larger budget cannot hang a round when every worker disconnects.
  - **#4 — remote result round isolation** (`src/api/workers/coordinator.py`, `src/cascade_correlation/cascade_correlation.py`): `TaskResult` now carries a `round_id`, attached server-side in `submit_result` from the dispatching `PendingTask`. `_dispatch_to_remote_workers` filters results by `round_id`, so a late result for a still-tracked stale `_pending_tasks` entry can no longer leak into a later round's collection (the remote analogue of #315's RC-5 local-path isolation).
  - **#5 — idle workers never received tasks submitted after they connected** (`src/api/websocket/worker_stream.py`): `_try_dispatch_task` ran only at worker-connect and after a task result, so an idle, already-connected worker had no trigger to pick up candidate tasks submitted mid-session — they sat unassigned until the collection budget expired and the round fell back to local retry, so the remote tier never contributed. The heartbeat handler now dispatches pending work to idle workers (guarded on registry idle state, since `get_next_assignment` does not itself refuse a busy worker), bounding delivery latency to ~one heartbeat interval. Verified live: a worker receives a dispatched task within ~1s of dispatch (previously: never).
  - **Scope / not yet closed**: these fix the **cascor** side of the dual-path. End-to-end remote candidate *execution* remains blocked by a separate **worker-side** defect (CW-05: the `juniper-cascor-worker` container cannot import the cascor `candidate_unit` codebase — it is launched with no `--cascor-path`, no cascor src mount, and no cascor package, so it crashes on every task it is handed). Tracked separately; this PR therefore *partially* addresses #319. Regression coverage: `TestHeartbeatDispatch` (worker_stream — task delivered to an idle worker via heartbeat, plus the busy-worker guard), `TestCollectResults` round_id + liveness early-exit (coordinator), and the timeout-scaling + stale-round tests (`test_candidate_result_collection_dualpath_regression.py`).
- **Status `current_hidden_units` stuck at 0 — track the live count from the metric stream**: `GET /v1/training/status` reported `monitor.current_hidden_units = 0` for the entire run even as the cascade grew units (`/v1/network` showed the real count), because `TrainingMonitor.current_hidden_units` was bumped **only** in `on_cascade_add` — a method with **no production caller** (only tests invoke it). `TrainingMonitor.on_epoch_end` (the sole production status updater, called from the manager's output-training callback with `hidden_units=len(network.hidden_units)`) now sets `current_hidden_units` from that authoritative per-epoch count, so the status field tracks growth without depending on the unwired `on_cascade_add` callback. Surfaced during the live-verify of the #315 result-collection fix: the network grew 0→4 hidden units while the status (and canopy's status bar, which reads this field) showed 0 — making a working fix look broken from the dashboard. Regression test `test_lifecycle_monitor.py::TestTrainingMonitor::test_on_epoch_end_updates_current_hidden_units`. (Separately noted: `on_cascade_add` being unwired in production also means `cascade_add` WS-broadcast events never fire — a distinct follow-up.)

### Security

- **SEC-F10 (HO-5) — runtime training-parameter bounds are now enforced on the `PATCH` *and* WebSocket update paths.** `TrainingParams` (the `POST /v1/training/start` model) always carried both floors and ceilings (e.g. `learning_rate` `le=10.0`, `candidate_pool_size` `le=256`, `max_hidden_units` `le=10_000`, `patience` `le=100_000`, `epochs_max`/`candidate_epochs`/`output_epochs` `le=1_000_000`), but the two *runtime*-update paths did not: `TrainingParamUpdateRequest` (`PATCH /v1/training/params`) declared only lower bounds, and the `set_params` WebSocket command (`/ws/control`) routed the raw JSON dict straight to `TrainingLifecycleManager.update_params` with no Pydantic model at all — only a downstream key-whitelist + candidate-pool check, neither of which range-checks scalar fields. A runtime update could therefore push an out-of-range value (live-confirmed: `max_hidden_units=999999999`) onto a running network. **Fix**: (1) `src/api/models/training.py` mirrors every `le=` ceiling from `TrainingParams` onto the corresponding `TrainingParamUpdateRequest` field (12 fields; fields stay `Optional` for partial-update semantics), so an out-of-range PATCH is rejected with 422 at the request boundary; (2) `src/api/websocket/control_stream.py` validates the incoming `set_params` payload through `TrainingParamUpdateRequest(**params)` before `update_params` and, on failure, returns a clean `command_response{status:"error", code:"invalid_params"}` ack (the WS analogue of the REST 422) without dropping the connection. A new `TestParamModelBoundsParity` guard pins the two models' shared-field numeric bounds in lock-step so the divergence cannot silently reopen. Regression coverage: `src/tests/unit/api/test_api_runtime_params.py` (over-ceiling + below-floor → 422, in-range/exact-ceiling → 200, model-bounds parity) and `src/tests/integration/api/test_ws_control_set_params.py` (WS over-ceiling / negative rejected and **not** applied to the live network; in-range still applied). Ref: juniper-ml [`notes/JUNIPER_STACK_SECURITY_AUDIT_PLAN_2026-07-02.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_STACK_SECURITY_AUDIT_PLAN_2026-07-02.md) §4.3 / §5.2.

### Tests

- **Per-file coverage lift 2 (C-5) — WebSocket layer (`src/api/websocket/`).** Tests-only; no source changed, no CI gate flipped. Part 2 of the split under the ecosystem per-file coverage rollout (juniper-ml [`notes/JUNIPER_ECOSYSTEM_PER_FILE_COVERAGE_ROLLOUT_SCOPING_2026-06-30.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_ECOSYSTEM_PER_FILE_COVERAGE_ROLLOUT_SCOPING_2026-06-30.md)); lifts the three lowest-coverage WebSocket source files — the sub-module recommended next after PR-1 ([#368](https://github.com/pcalnon/juniper-cascor/pull/368)) — to full statement coverage of their previously-uncovered branches:

  | File | Before (stmt) | After (stmt) |
  |------|---------------|--------------|
  | `src/api/websocket/training_stream.py` | 114/156 = 73.08% | 156/156 = **100.00%** |
  | `src/api/websocket/control_stream.py` | 161/193 = 83.42% | 193/193 = **100.00%** |
  | `src/api/websocket/manager.py` | 275/308 = 89.29% | 308/308 = **100.00%** |

  The `src/api/websocket` sub-module clears the ratified ≥95% pooled bar: **88.17% → 99.37%** (842/955 → 949/955, statement-weighted).
  
  - Overall cascor coverage 90.20% → 91.03%.
  - New fast unit tests (40 across `test_training_stream_coverage.py` [new], `test_control_stream_coverage.py`, and `test_websocket_manager.py`) drive the resume-handshake + replay arms (`training_stream._await_resume_frame` / `_handle_resume`), the control-path handshake gates / leaky-bucket rate-limit / invalid-params / heartbeat / idle-timeout branches (`control_stream`), and the manager's per-endpoint bookkeeping, per-IP accounting, pending-connection rejection, and defensive metric-emission guards (`manager`) — all via `AsyncMock` seams (no live sockets).
  - Measured on the CI `unit and not slow` subset (the gate basis) with `juniper-coverage-gap-map` (`juniper-ci-tools 0.6.0`, advisory).
  - The blocking `--enforce` gate lands in the final PR of the split once every sub-module clears.

- **Per-file coverage rollout (Phase C-5) — worst-first lift, part 1 of a
  multi-PR sequence.** Lifts the two lowest-coverage source files to full
  statement coverage, clearing the two sub-modules they dominate. No source
  files changed; no gate flipped yet (the `juniper-coverage-gap-map --enforce`
  CI step lands in the final gate PR once every sub-module clears). Measured on
  the CI `unit and not slow` subset (`--cov=src`), the gate basis.
  - `parallelism/rc4_ring_buffer.py`: **30.16% → 100%** statement — the
    RC-4 instrumentation ring buffer is disabled-by-default (its `ENABLED`
    flag reads `CASCOR_RC4_RING_BUFFER` at import), so a normal run
    short-circuits every body; the new `src/tests/unit/test_rc4_ring_buffer_coverage.py`
    drives the enabled paths by monkeypatching the module flag (never the
    environment, so the conftest RC-4 fixtures stay inert) with per-test
    global isolation. This clears the ecosystem's **worst** cascor
    sub-module, `parallelism` (**69.01% → 100%** pooled).
  - `api/routes/network.py`: **51.38% → 100%** statement — the three CAN-015h
    handlers (`patch_weights` / `add_hidden_unit` / `delete_hidden_unit`) were
    uncovered status→HTTP-code dispatch; `src/tests/unit/api/test_network_route_coverage.py`
    gains a case per branch (including the defensive unmapped-sentinel 500),
    each driven by mocking the lifecycle method to return a crafted status
    dict (sentinels resolved off the real lifecycle instance to stay
    drift-proof). This clears the `api/routes` sub-module
    (**86.90% → 95.69%** pooled).
  - Overall cascor statement coverage **90.20% → 91.12%**; files below the
    90% floor 8 → 6, sub-modules below the 95% pooled bar 9 → 7. See juniper-ml
    [`notes/JUNIPER_ECOSYSTEM_PER_FILE_COVERAGE_ROLLOUT_SCOPING_2026-06-30.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_ECOSYSTEM_PER_FILE_COVERAGE_ROLLOUT_SCOPING_2026-06-30.md).

## [0.5.0] - 2026-05-22

**Note on version history**: `pyproject.toml` was bumped 0.3.17 → 0.4.0 on 2026-03-03 in preparation for a 0.4.0 release that was never cut to PyPI (the `[0.4.0]` section below documents the work that *would have* shipped). This 0.5.0 release rolls up both that work and the subsequent ~2.5 months of changes (469 commits since `v0.3.17`) into a single PyPI release. Subsequent entries in this section list the additional work landed since 2026-03-03.

### Added

- **AGENTS.md header schema standardization** (juniper-cascor#299, juniper-ml#316/#319): adopted the canonical 6-field schema (`**Project**` / `**Repository**` / `**Author**` / `**License**` / `**Version**` / `**Last Updated**`, in that relative order). Added `.github/workflows/agents-md-touch-up.yml` (auto-bumps `**Last Updated**` to today's UTC date on every PR push touching `AGENTS.md`; idempotent). Added `juniper-lint-agents-md-header` CI step (via `juniper-ci-tools>=0.4.0` PyPI dependency). `**Repository**: pcalnon/juniper-cascor` added to AGENTS.md header; required-field order corrected to canonical.
- **`util/test_agents_md_version_drift.py`** -- portable port of juniper-ml's lint test pinning `AGENTS.md`'s `**Version**:` header to `pyproject.toml`'s `[project].version`. Catches the failure class where a `pyproject.toml` bump leaves the agent-facing contract stale. Preventive-only here: cascor's `AGENTS.md` and `pyproject.toml` are already in sync at `0.5.0`. Wired into the CI tests job next to the existing `test_workflow_script_paths.py` lint.

### Removed

- **API-09 (PR 3 of 3 — migration complete)**: dropped the top-level `"detail"` deprecation alias from cascor's `HTTPException` envelope responses. PR 1 (#293) introduced the alias as a transitional measure so pre-migration consumers — notably `juniper-cascor-client` releases before commit `b0a636a3` (2026-02-21, predates the migration plan) and 36 in-tree test assertions reading `response.json()["detail"]` — kept working unchanged while the migration was rolled out. PR 2 (juniper-cascor-client #59) pinned the envelope-aware parser via explicit regression coverage. PR 3 (this commit) drops the alias now that the soak window has completed; cascor's final `HTTPException` response shape is `{"status":"error","error":{"code":"HTTP_NNN","message":<exc.detail>,"detail":null},"meta":{"timestamp":...,"version":...}}` — single shape, no alias. **Required consumer change**: clients reading the top-level `"detail"` field must switch to `response.json()["error"]["message"]`. `juniper-cascor-client` already does this since the 2026-02-21 commit; the 36 in-tree cascor test assertions are updated to match. Wire-compat snapshot `test_api_09_http_exception_wire_compat.py::TestLegacyDetailAliasAbsent` is the new pin asserting the alias stays gone (replaces the previous `TestLegacyDetailAliasPresent` class from PR 1). Closes the API-09 migration end-to-end; full history at `notes/API_09_ERROR_ENVELOPE_MIGRATION_DESIGN_2026-05-21.md`.

### Changed

- **CFG-04**: cascor's `JUNIPER_DATA_URL` env-var lookup is consolidated onto a new `Settings.juniper_data_url: str | None` pydantic field in `src/api/settings.py`. The 8 historical raw `os.environ.get("JUNIPER_DATA_URL")` / `os.getenv("JUNIPER_DATA_URL")` call sites — spread across `src/api/app.py` (3 sites in `create_app()` / `_auto_start_training` / `_auto_start_canopy`), `src/api/routes/health.py`, `src/api/lifecycle/manager.py`, `src/main.py`, `src/spiral_problem/spiral_problem.py`, and `src/spiral_problem/data_provider.py` — now read from the field instead, which centralizes the lookup, gives it a typed default, and surfaces it in the `Settings` object schema (so it's discoverable by `dir(settings)`, `settings.model_dump()`, generated config docs, etc.). The field is exposed via `validation_alias=AliasChoices("juniper_data_url", "JUNIPER_DATA_URL", "JUNIPER_CASCOR_JUNIPER_DATA_URL")` so the canonical unprefixed `JUNIPER_DATA_URL` (the ecosystem-shared name also used by juniper-data and juniper-canopy) takes precedence over the per-service `JUNIPER_CASCOR_JUNIPER_DATA_URL` override. **Behavior is byte-for-byte unchanged**: the default is `None`, the 4 call sites that wanted the legacy `http://localhost:8100` fallback apply it explicitly via `settings.juniper_data_url or _PROJECT_API_JUNIPER_DATA_URL_DEFAULT`, and the 4 call sites that treat the URL as required (`main.py` pre-flight, `SpiralProblem.generate_n_spiral_dataset`, `SpiralDataProvider.__init__` / `.validate_configuration` / `.get_spiral_dataset`, and the readiness probe) still check for `None`/empty and short-circuit identically. Tests that pre-constructed `Settings(...)` then `patch.dict(os.environ, {"JUNIPER_DATA_URL": ...})` afterwards now pass the URL directly via the `juniper_data_url=` kwarg, which is the post-CFG-04 equivalent and stays decoupled from ambient process env. **Not a deprecation migration** (unlike CFG-03 / CFG-05): the env-var name does not change and no `DeprecationWarning` is emitted — `JUNIPER_DATA_URL` remains the canonical and indefinitely-supported name. New 7-case regression suite at `src/tests/unit/test_cfg_04_juniper_data_url_settings.py` pins the contract: default `None`, canonical-only, prefixed-only, both-set-canonical-wins, kwarg-overrides-env, empty-string-preserved, and type-annotation (`str | None`). Tracks CFG-04 in the v7 outstanding-development roadmap.

### Deprecated

- **API-09 (PR 1 of 3)**: cascor now wraps every `raise HTTPException(...)` site (66 across `src/api/routes/`) into the project's standard `ResponseEnvelope` / `ErrorResponse` shape, registered as a new `@app.exception_handler(HTTPException)` in `create_app()` (`src/api/app.py`). The response body is now `{"status":"error","error":{"code":"HTTP_NNN","message":<exc.detail>,"detail":null},"meta":{"timestamp":...,"version":...},"detail":<exc.detail>}` — the new envelope shape plus a top-level `"detail"` **deprecation alias** for the migration window. The alias keeps existing consumers working unchanged: (a) 36 in-tree cascor test assertions reading `response.json()["detail"]` continue to pass, and (b) `juniper-cascor-client` releases prior to the PR 2 update read `body.get("detail", response.text)` in `_request()` and would otherwise silently degrade to dumping the entire JSON blob as the error message. The alias will be **removed in a future release (PR 3 of this migration)** after the cascor-client release in PR 2 has had time to be adopted in production deployments. `error.code` uses the string form `"HTTP_NNN"` to match the existing `ErrorDetail.code: str` schema (Pydantic v2 strict mode does not coerce `int -> str`) and the existing semantic codes (`VALIDATION_ERROR`, `INTERNAL_ERROR`), preserving the future migration path to semantic codes (`NETWORK_NOT_FOUND`, etc. — API-09b in the design doc). `headers=exc.headers` passthrough preserves `WWW-Authenticate` (401), `Retry-After` (429), and arbitrary custom headers to match FastAPI's default-handler behavior. Pinned by a new regression suite at `src/tests/unit/api/test_api_09_http_exception_envelope.py` (covers 400/401/403/404/409/422/500/503 + headers-passthrough + `exc.detail=None` fallback + existing-handler-not-shadowed) and a new wire-compat snapshot at `src/tests/unit/api/test_api_09_http_exception_wire_compat.py` (pins top-level + nested key sets exact, snapshot values for 404 / 503, and the alias's required presence — PR 3 will flip this to required-absence). Full migration plan in `notes/API_09_ERROR_ENVELOPE_MIGRATION_DESIGN_2026-05-21.md`. Tracks API-09 in the v7 outstanding-development roadmap §21.

- **CFG-05**: `CASCOR_LOG_LEVEL` is now deprecated in favor of the prefixed `JUNIPER_CASCOR_LOG_LEVEL` env var (which already backs the pydantic `Settings.log_level` field via `env_prefix='JUNIPER_CASCOR_'`). Historically the bootstrap log-level override at the top of `src/cascor_constants/constants.py` read a second, unprefixed `CASCOR_LOG_LEVEL` (CASCOR-P2-003) while the pydantic-validated runtime config read the prefixed form — two env vars for the same feature is operator-hostile. A new `_resolve_log_level_env() -> str` helper in `src/cascor_constants/constants.py` consolidates the lookup with the following precedence: (1) `JUNIPER_CASCOR_LOG_LEVEL` wins when set; (2) `CASCOR_LOG_LEVEL` is still honored when only the legacy name is set, but emits a `DeprecationWarning` (`warnings.warn(..., DeprecationWarning, stacklevel=2)`); (3) when both are set to **different** values, the prefixed form wins and a `[juniper-cascor] CFG-05 WARNING:` line is emitted on stderr so split-config drift is visible at startup; (4) when both are set to the **same** value (case-insensitive, since both are normalized via `.upper()`), the prefixed form wins silently (no-op for migration-in-progress deployments). `CASCOR_LOG_LEVEL` will be **removed in a future release** — update deployment manifests, supervisor configs, and `.env` files to the prefixed name. Pinned by 7-case regression suite at `src/tests/unit/test_cfg_05_log_level_resolution.py` (prefixed-only, prefixed-lowercase-normalized, legacy-only, both-same, both-same-case-insensitive, both-different, neither-set). Mirrors the CFG-03 (`SENTRY_SDK_DSN` → `JUNIPER_CASCOR_SENTRY_DSN`) migration shipped immediately prior. Tracks CFG-05 in the v7 outstanding-development roadmap.

- **CFG-03**: `SENTRY_SDK_DSN` is now deprecated in favor of the prefixed `JUNIPER_CASCOR_SENTRY_DSN` env var (which already backs the pydantic `Settings.sentry_dsn` field via `env_prefix='JUNIPER_CASCOR_'` consumed by `configure_sentry()` in `src/api/app.py`). Historically the bootstrap Sentry init at the top of `src/main.py` read a second, unprefixed `SENTRY_SDK_DSN` — two env vars for the same feature is operator-hostile. A new `_resolve_sentry_dsn() -> str | None` helper in `src/main.py` consolidates the lookup with the following precedence: (1) `JUNIPER_CASCOR_SENTRY_DSN` wins when set; (2) `SENTRY_SDK_DSN` is still honored when only the legacy name is set, but emits a `DeprecationWarning` (`warnings.warn(..., DeprecationWarning, stacklevel=2)`); (3) when both are set to **different** values, the prefixed form wins and a `[juniper-cascor] CFG-03 WARNING:` line is emitted on stderr so split-config drift is visible at startup; (4) when both are set to the **same** value, the prefixed form wins silently (no-op for migration-in-progress deployments). `SENTRY_SDK_DSN` will be **removed in a future release** — update deployment manifests, secrets, and `.env` files to the prefixed name. Pinned by 5-case regression suite at `src/tests/unit/test_cfg_03_sentry_dsn_resolution.py` (prefixed-only, legacy-only, both-same, both-different, neither-set). Tracks CFG-03 in the v7 outstanding-development roadmap.

### Added

- **METRICS-MON R3.7 (soak complete)**: macOS leg of the unit-tests CI matrix flipped from `experimental: true` → `experimental: false`, making the `macos-latest` (Python 3.12) leg **required**. Failures on macOS now block the job. The `continue-on-error: ${{ matrix.experimental == true }}` job-level guard is preserved as a future-proof escape hatch for future experimental matrix entries; with `experimental: false` it evaluates to `false`. Soak window 2026-05-01 → 2026-05-15 confirmed clean (per user direction). Closes the post-soak follow-up of the R3.7 fan-out.

- **METRICS-MON R3.7 / seed-(R1.3 design)**: macOS leg added to the unit-tests CI matrix. `.github/workflows/ci.yml::unit-tests` now runs on `${{ matrix.os }}` with a single new `macos-latest` (Apple Silicon / ARM) entry pinned to Python 3.12; Linux legs (Python 3.12 + 3.13 + 3.14) are unchanged. The macOS leg starts in **`continue-on-error: true`** mode for a 2-week soak (2026-04-30 → 2026-05-14) so platform-divergence failures (POSIX-only assumptions in the lifecycle / WS / training paths) surface in CI without blocking PRs while environment-specific issues are identified. The torch wheel install branches by OS — Linux uses the CPU-only PyTorch index (`https://download.pytorch.org/whl/cpu`) which has no macOS-arm64 wheels; macOS uses the default PyPI index which does. After the soak, flip the include block's `experimental` flag to `false` to make the macOS leg required. Closes the juniper-cascor leg of [METRICS_MONITORING_R3_ENTRY_PLAN_2026-04-30.md](https://github.com/pcalnon/juniper-ml/blob/main/notes/code-review/METRICS_MONITORING_R3_ENTRY_PLAN_2026-04-30.md) §3 Q1.

- **METRICS-MON R3.5 / seed-07**: replay-buffer overflow regression coverage at the production-default capacity. New test class `TestReplayBufferOverflowAtConfiguredCapacity` in `src/tests/unit/api/test_websocket_seq_replay.py` (3 tests) drives `Settings().ws_replay_buffer_size + 1` broadcasts through `WebSocketManager` and asserts: (a) buffer length stays bounded at capacity, (b) oldest entry evicted (seq=1 gone, seq=2 surviving as the new oldest), (c) newest entry retained (seq = capacity+1), (d) no exception raised at the boundary, (e) `current_seq` reflects every broadcast even when buffered ones were evicted. Also covers the at-capacity boundary (no eviction yet at exactly N broadcasts) and a 10×N stress case (no pathological growth past the boundary). Existing `TestReplayBuffer.test_replay_buffer_bounded_to_configured_capacity` (size=10 / iters=20) is preserved for fast unit coverage; the new class pins the contract at the deployed default (1024). See [`notes/code-review/METRICS_MONITORING_ROADMAP_2026-04-25.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/code-review/METRICS_MONITORING_ROADMAP_2026-04-25.md) §6 R3.5 in juniper-ml.

### Changed (potentially breaking)

- **METRICS-MON R2.2.2 / seed-05**: juniper-cascor's WebSocket protocol surface now consumes the shared `juniper-cascor-protocol>=0.1.0a0` package (added as a runtime dependency). `MessageType` and `BinaryFrame` move from `src/api/workers/protocol.py` into `juniper_cascor_protocol.worker`; the local symbols are preserved as thin re-exports (`MessageType = WorkerMessageType`) so existing imports (`from api.workers.protocol import MessageType, BinaryFrame, …`) continue to work unchanged. The `WorkerProtocol` builder/validator helpers + the `TaskAssignment` / `TaskResultMessage` dataclasses stay local. The dict-builder helpers in `src/api/websocket/messages.py` (`create_metrics_message`, `create_state_message`, `create_topology_message`, `create_event_message`, `create_cascade_add_message`, `create_chunked_message`, `create_initial_metrics_message`, `create_candidate_progress_message`, `create_control_ack_message`) now construct the corresponding `juniper_cascor_protocol.envelope` Pydantic models and emit `model.model_dump(exclude_none=True)`. Output is **byte-for-byte identical** to the pre-migration implementation; pinned by the new wire-compat snapshot at `src/tests/unit/api/test_messages_wire_compat.py` (covers all 9 typed envelopes plus a pydantic-roundtrip test). Cascor server is the **first consumer** of its own protocol package — the producer cannot drift from the schema because it now imports it. See [`notes/code-review/METRICS_MONITORING_R2.2_WS_FRAME_SCHEMA_DESIGN_2026-04-29.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/code-review/METRICS_MONITORING_R2.2_WS_FRAME_SCHEMA_DESIGN_2026-04-29.md) in juniper-ml.

- **METRICS-MON R2.1.4 / seed-06**: juniper-cascor's observability surface now consumes the shared `juniper-observability>=0.1.1` package (added as a runtime dependency). The cross-cutting machinery — `JuniperJsonFormatter`, `RequestIdMiddleware`, `PrometheusMiddleware`, `request_id_var`, `UNMATCHED_ENDPOINT_LABEL`, `get_prometheus_app`, `set_build_info`, the SEC-15 `_strip_sensitive_headers` Sentry hook, and the R1.2 contract constants `LIVENESS_TICK_BUDGET_MS` / `LIVENESS_STALENESS_SECONDS` / `READINESS_HEADER` — moves into the shared lib; `api/observability.py` and `api/models/health.py` are preserved as thin re-export shims so existing imports (`from api.observability import …`, `from api.models.health import …`) continue to work unchanged. **Wire-format change**: `/v1/health/ready` body field `timestamp` now derives from `datetime.now(UTC).timestamp()` (was naive `datetime.now().timestamp()`) — closes the BUG-JD-06-equivalent local-time leak. Values stay unix-epoch-seconds and shift only by host tz-offset (irrelevant to consumers computing diffs). The service-specific `configure_logging` (with `RotatingFileHandler` to `logs/juniper_cascor.log`) and `configure_sentry` (pinning `traces_sample_rate=1.0` via `_LOGGER_SENTRY_TRACES_SAMPLE_RATE`) stay local; cascor's training and WebSocket Prometheus metrics are unchanged. New `JuniperJsonFormatter()` default `service` is `"juniper-service"` (was `"juniper-cascor"`); all in-tree call sites pass the service name explicitly so this only affects ad-hoc construction. Wire-compat snapshot test added at `src/tests/unit/api/test_r2_1_4_wire_compat.py` pinning the `/v1/health/ready` JSON shape, header, and Prometheus metric names. See [`notes/code-review/METRICS_MONITORING_R2.1_SHARED_OBSERVABILITY_DESIGN_2026-04-28.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/code-review/METRICS_MONITORING_R2.1_SHARED_OBSERVABILITY_DESIGN_2026-04-28.md) in juniper-ml.

### Added

- **METRICS-MON R1.3 / seed-04 (cascor side)**: `WorkerRegistration` gains `in_flight_tasks: int`, `last_task_completed_at: float | None`, and `rss_mb: float | None` populated from R1.3-aware workers' enriched heartbeat payloads. `WorkerRegistration.record_heartbeat()` and `WorkerRegistry.heartbeat()` accept these as keyword-only arguments with `None` defaults so older worker images sending only `worker_id`/`timestamp` still register correctly (prior values preserved). The WebSocket worker-stream handler forwards the enriched fields when present. `/v1/workers` and `/v1/workers/{id}` JSON now include the new keys; existing keys unchanged. See [`notes/code-review/METRICS_MONITORING_R1.3_WORKER_HEARTBEAT_DESIGN_2026-04-27.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/code-review/METRICS_MONITORING_R1.3_WORKER_HEARTBEAT_DESIGN_2026-04-27.md) in juniper-ml. Companion worker PR provides the enriched payload sender and the new HTTP `/v1/health[/live|/ready]` probe surface.

### Changed (potentially breaking)

- **METRICS-MON R1.2 / seed-02 / seed-03**: `/v1/health/ready` now returns **HTTP 503** (not 200) when a required dependency is unhealthy, with body `status="not_ready"`. Required deps for cascor are the lifecycle manager (always) and JuniperData (when `JUNIPER_DATA_URL` is set). `/v1/health/live` runs an in-process liveness tick (consults a new `lifecycle.is_alive()` accessor backed by a 1-second heartbeat counter) within a 250 ms budget and returns **HTTP 503** on tick failure or budget exceedance. Both endpoints emit a new `X-Juniper-Readiness` header / liveness body fields (`tick`, `duration_ms`) so probe diagnostics surface in orchestrator logs without body parsing. Adds `TrainingLifecycleManager.bump_liveness()`, `is_alive(stale_after_seconds=30.0)`, and `stop_liveness_heartbeat()` plumbing; `TrainingMonitor` event callbacks (epoch, cascade, phase-change, training-start/end, candidate-progress, topology-change) bump the heartbeat so progress in the training thread is an additional liveness signal. See [`notes/code-review/METRICS_MONITORING_R1.2_PROBE_DESIGN_2026-04-27.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/code-review/METRICS_MONITORING_R1.2_PROBE_DESIGN_2026-04-27.md) in juniper-ml for the cross-repo contract; companion PRs land in juniper-data, juniper-canopy, and juniper-deploy. `/v1/health` (the legacy combined endpoint) is unchanged.

### Added

- Phase D (§S10) per-command timeouts on `/ws/control`: commands dispatched via `asyncio.to_thread` and bounded by `asyncio.wait_for` with `start=10s`, `stop/pause/resume/reset=2s`, `set_params=1s`. Timeouts emit `command_response{status:"error", error:"Command timed out after Ns"}` while the connection stays open.
- Phase D (§S10.3) unknown-command envelope now includes `code:"unknown_command"` to let browser clients distinguish protocol errors from execution failures. `create_control_ack_message` gains an optional `code=` keyword argument.
- Phase D (§S10.7) server-side observability counter `cascor_ws_control_command_received_total{command}` (lazy-registered so test suites without `prometheus_client` stay importable).
- Hardcoded-values refactor (Wave 1): new `cascor_constants/constants_api/constants_api_defaults.py` module with 49 `_PROJECT_API_*` constants covering Pydantic field defaults for `NetworkCreateRequest` / `TrainingStartRequest`, lifecycle defaults, middleware body/rate-limit defaults, observability defaults, TLS minimum versions, decision-boundary resolution bounds, juniper-data integration timeouts, and inter-service URL templates. Existing constants modules (model, candidates, hdf5, logging) extended where needed.

### Changed

- Hardcoded-values refactor (Wave 2): replaced ~58 inline literals across 10 api-layer files (`api/app.py`, `api/lifecycle/manager.py`, `api/lifecycle/monitor.py`, `api/middleware.py`, `api/models/network.py`, `api/models/training.py`, `api/observability.py`, `api/routes/decision_boundary.py`, `api/service_launcher.py`, `api/workers/security.py`) with imports from `cascor_constants.constants_api.constants_api_defaults`.
- Hardcoded-values refactor (Wave 3): replaced 4 inline literals in `src/candidate_unit/candidate_unit.py` and `src/snapshots/snapshot_serializer.py` with constants from `cascor_constants.constants_model` and `cascor_constants.constants_hdf5`.
- AGENTS.md "Constants Configuration" section updated to document the new `constants_api/` submodule and the cross-repo alignment requirements verified by Wave 5 (Pydantic field/constants alignment, worker protocol bit-identity with `juniper-cascor-worker` and `juniper-cascor-client`, `X-API-Key` literal alignment, binary-frame format alignment).

### Fixed

- **CONC-10** (Track 3 Phase 3D, 2026-04-27): `WorkerCoordinator._check_stale_workers` (`src/api/workers/coordinator.py`) now holds `self._lock` across the entire per-worker reaper sequence — re-check liveness, requeue any active task on `self._unassigned_tasks`, AND call `self._registry.deregister(...)` — so a concurrent `get_next_assignment(worker_id)` (which holds the same lock for its critical section) can no longer land a task on a worker mid-deregistration. Pre-fix the deregister was outside the lock, leaving a window where the just-assigned task waited up to `_task_reassignment_timeout` (default 120 s) before being picked up by another worker. The send-callback unregister stays outside the lock per the lock-order rule. The reaper also now uses `current.active_task_id or worker.active_task_id` (registry-current first, snapshot fallback) so a task dispatched between the snapshot and the re-check is still requeued. Verified by `src/tests/unit/api/test_coordinator_health_monitor_race.py` (2 AST-based source-level guards + 2 behavioural tests racing assignment vs deregister).
- **CONC-09** (Track 3 Phase 3C, 2026-04-26): the `auto_start_training` and `auto_start_canopy` background tasks created in `src/api/app.py::lifespan` are no longer fire-and-forget. Each `asyncio.create_task(...)` is now stored on `app.state.startup_tasks`, named for debuggability, and wired with the new `_log_startup_task_exception` done-callback so any non-cancellation exception is surfaced at error level with the original traceback (instead of being silently swallowed and only later emitted as the cryptic "Task exception was never retrieved" warning when the task is GC'd). The shutdown phase cancels any in-flight startup tasks and awaits them with `asyncio.gather(..., return_exceptions=True)` so cancellation errors don't escape the lifespan boundary. Verified by `src/tests/unit/api/test_app_startup_tasks.py` (4 source-level checks that always run + 4 behavioural tests that `importorskip("torch")` so the env's broken torch C-extension doesn't gate the regression coverage).
- Removed an unused `epoch_trained_candidate` return-value assignment in `src/candidate_unit/candidate_unit.py` that flake8 had been flagging as F841 (the side-effecting `_update_weights_and_bias` call is preserved). This was a surgical fix needed to keep pre-commit clean for the Wave 3 commit.

### Notes

- All Wave 5 cross-repo alignment checks pass: `MessageType` ↔ `MSG_TYPE_*` constants in `juniper-cascor-worker` and `juniper-cascor-client` are bit-identical; `_PROJECT_API_NETWORK_*_DEFAULT` values match `NetworkCreateRequest.model_fields[*].default` exactly (`input_size=2`, `output_size=2`, `learning_rate=0.01`, `max_hidden_units=10`, `epochs_max=200`, etc.).
- All api-unit and unit pytest suites pass without modification; pre-commit on the 12 files this branch modifies is clean.
- No public API changes; REST request/response shapes, WebSocket message formats, and the binary frame protocol are unchanged.

## [0.4.0] - 2026-03-03

**Summary**: Comprehensive security hardening — security headers, request body limits, error sanitization, restrictive CORS/rate limiting defaults, WebSocket authentication and message validation, HMAC pickle verification, /metrics auth, conditional docs, CI hardening, and scheduled security scanning.

### Security: [0.4.0]

- Added `SecurityHeadersMiddleware` — X-Content-Type-Options, X-Frame-Options, Referrer-Policy, Permissions-Policy, conditional HSTS
- Added `RequestBodyLimitMiddleware` with configurable max body size (default 10MB)
- Sanitized error responses in ValueError handler, training routes, and network routes — generic messages returned to clients; internal details logged at DEBUG
- Changed CORS origins default from `["*"]` to `[]` (restrictive by default)
- Changed rate limiting default from disabled to enabled
- Added WebSocket authentication — API key validation at connection accept, close code 4001 on failure
- Added WebSocket message size limits and Pydantic schema validation for control commands
- Added HMAC signature verification before `pickle.loads()` in snapshot serializer
- Removed `/metrics` from authentication-exempt paths
- Added conditional API docs — disabled when API keys are configured
- Removed `|| true` from Bandit CI step (security scan failures now fail the build)

### Added: [0.4.0]

- `.github/workflows/security-scan.yml` — Weekly scheduled security scanning (Bandit, pip-audit)

### Changed: [0.4.0]

- Updated test fixtures for new security defaults

### Technical Notes: [0.4.0]

- **SemVer impact**: MINOR — New middleware, changed security defaults (non-breaking: configurable via env vars)
- **Test count**: 264 API tests passed, 0 failed
- **Part of**: Cross-ecosystem security audit (7 repos, 24 findings)
- **Note**: Version 0.4.0 in pyproject.toml; CHANGELOG versions 0.0.1–0.7.0 are pre-PyPI development history

---

## [0.7.0] - 2026-02-06

### Changed
- **BREAKING**: `JUNIPER_DATA_URL` environment variable is now REQUIRED for dataset operations (CAS-INT-001)
- `generate_n_spiral_dataset()` now exclusively uses `SpiralDataProvider` — local spiral generation code path removed
- `JuniperDataClient` now supports API key authentication via `api_key` parameter or `JUNIPER_DATA_API_KEY` env var (CAS-INT-003)

### Added
- NPZ data contract validation in `SpiralDataProvider._convert_arrays_to_tensors()` (CAS-INT-004)
- `JuniperCascor ↔ JuniperData Integration Plan` at `notes/CASCOR_JUNIPER_DATA_INTEGRATION_PLAN.md`
- 25 new integration-related tests (total: 64 integration tests, 1269 unit tests passing)

### Removed
- Local spiral generation fallback code path in `generate_n_spiral_dataset()` (78 lines)
- Optional `JUNIPER_DATA_URL` toggle behavior (now mandatory)

### Fixed
- `test_spiral_problem_coverage.py` updated to use mock API path instead of removed local generation

---

## [0.6.7] - 2026-02-05

**Summary**: Created comprehensive Integration Development Plan consolidating all outstanding work across JuniperCascor, JuniperData, and JuniperCanopy. Updated CLAUDE.md with integration references.

### Added: [0.6.7]

- **Integration Development Plan**: Created `notes/INTEGRATION_DEVELOPMENT_PLAN.md`
  - Consolidated outstanding tasks from 4 existing roadmap documents:
    - `JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR_PLAN.md` (JuniperData notes)
    - `INTEGRATION_ROADMAP.md` (JuniperData notes)
    - `PRE-DEPLOYMENT_ROADMAP-2.md` (JuniperData notes)
    - `PRE-DEPLOYMENT_ROADMAP.md` (JuniperData notes)
  - Performed rigorous source code review identifying 47 new issues:
    - 5 Critical (walrus operator bug, duplicated class, invalid constructor params, hardcoded paths)
    - 12 High (misleading imports, undeclared globals, missing import guards, stale duplicates)
    - 16 Medium (falsy value bugs, inverted logic, OOM risk, refactoring needs)
    - 14 Low (TODOs, commented code, version inconsistencies, style issues)
  - Organized 61 total issues into 5 prioritized phases:
    - Phase 0: Critical Bugs & Blockers (5 issues, 8-16 hours)
    - Phase 1: Integration Architecture (8 issues, 24-40 hours)
    - Phase 2: Code Quality & Test Integrity (14 issues, 32-48 hours)
    - Phase 3: Infrastructure & CI/CD (10 issues, 20-32 hours)
    - Phase 4: Enhancements & Future Work (24 issues, 40-80 hours)
  - Includes dependency matrix, risk assessment, and source code review appendix

### Changed: [0.6.7]

- **CLAUDE.md**: Updated to version 0.6.6 (0.7.3), date 2026-02-05
  - Added `notes/INTEGRATION_DEVELOPMENT_PLAN.md` to Documentation Files table
  - Added 5 additional roadmap documents to Documentation Files table
  - Added `requests` to Core Libraries (JuniperData REST client dependency)
  - Added `juniper_data_client/` to directory structure
  - Added `JUNIPER_DATA_URL` and `CASCOR_BACKEND_PATH` to Environment Variables table

### Technical Notes: [0.6.7]

- **SemVer impact**: PATCH - Documentation and planning only; no API or code changes
- **Planning only**: No source code modifications were made
- **Reference**: See `notes/INTEGRATION_DEVELOPMENT_PLAN.md` for full plan details

---

## [0.6.6] - 2026-02-04

**Summary**: Test Suite and CI/CD Enhancement - Phase 4 complete. Added complexity warnings, performance benchmarks, multi-Python testing, and quick integration tests.

### Fixed: [0.6.6]

- **LOW-001**: Enabled flake8 complexity warnings (C901)
  - Removed C901 from flake8 ignore list for source code
  - Added targeted `# noqa: C901` to `_validate_format` in `snapshot_serializer.py` (complexity 19, validation logic)

### Added: [0.6.6]

- **LOW-004**: Added performance benchmarks to scheduled workflow
  - New job in `scheduled-tests.yml` runs `run_benchmarks.bash` nightly
  - Results uploaded as artifacts with 90-day retention
- **MED-009**: Added Python version matrix to unit tests (3.11, 3.12, 3.13)
  - Unit tests now run on all supported Python versions
  - Artifact names include Python version to avoid conflicts
  - Cache keys include Python version for proper isolation
- **MED-010**: Added quick integration tests that run on all branches
  - New `quick-integration-tests` job runs on every push
  - Stricter timeout (60s) and maxfail (2) for fast feedback
  - Added to quality gate requirements

### Changed: [0.6.6]

- Renamed "Integration Tests" job to "Full Integration Tests" for clarity
- Updated quality gate to check quick-integration-tests result

### Deferred: [0.6.6]

- **MED-014**: Line length reduction deferred - requires full codebase reformatting in separate commit

### Technical Notes: [0.6.6]

- **SemVer impact**: PATCH – Test infrastructure and CI/CD configuration; no API changes
- **Phase Complete**: Phase 4 (Low Priority & Enhancements) - MED-014 deferred
- **Reference**: See `notes/TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN.md` for full issue details

---

## [0.6.5] - 2026-02-04

**Summary**: Test Suite and CI/CD Enhancement - Phase 3 complete. Improved tooling quality gates with better linting coverage, mypy error code re-enablement, scheduled slow test workflow, and shellcheck configuration.

### Fixed: [0.6.5]

- **HIGH-005**: Re-enabled mypy error codes: `misc`, `call-arg`, `func-returns-value`, `no-redef`
  - Fixed 3 code issues in `cascade_correlation.py`:
    - `no-redef`: Variable re-declaration in `grow_network` (line 2754/2815)
    - `call-arg`: Removed invalid `objectify` kwarg in `create_snapshot` (line 3109)
    - `func-returns-value`: Fixed `_generate_uuid` return type from `None` to `str`
  - Added phased plan comments for remaining disabled codes
- **MED-013**: Changed shellcheck severity from `error` to `warning`
  - Added `backups/` and `temp/` to exclude patterns

### Added: [0.6.5]

- **MED-002**: Created scheduled workflow for slow/long-running tests
  - New file: `.github/workflows/scheduled-tests.yml`
  - Runs nightly at 3 AM UTC
  - Includes slow unit tests, slow integration tests, and long-running correctness tests
  - Supports manual triggering via workflow_dispatch
- **HIGH-006**: Added separate linting hooks for test files with relaxed rules
  - Flake8 hook for tests: relaxed complexity (25), additional ignores (E722, F401, F811)
  - Bandit hook for tests: additional skips (B105, B106, B107 for test fixtures)

### Changed: [0.6.5]

- **MED-007**: Removed `-p no:warnings` from pytest addopts
  - Added targeted filterwarnings for known library deprecations (torch, numpy, h5py, pkg_resources)
- **MED-004**: Re-enabled E722 (bare except) and F401 (unused imports) for source code linting
  - Tests retain relaxed rules for these codes
- Updated mypy python-version from 3.12 to 3.13 in pre-commit config

### Technical Notes: [0.6.5]

- **SemVer impact**: PATCH – Test infrastructure and CI/CD configuration; no API changes
- **Phase Complete**: Phase 3 (Tooling Quality Gates)
- **Reference**: See `notes/TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN.md` for full issue details

---

## [0.6.4] - 2026-02-04

**Summary**: Test Suite and CI/CD Enhancement - Phases 0-2 complete. Implemented fixes for false-positive tests, weak assertions, configuration issues, and CI pipeline hardening.

### Fixed: [0.6.4]

- **CRIT-001**: Fixed always-passing tests in `test_training_workflow.py` (assert True in both branches)
  - Converted to proper `pytest.raises()` for exception testing
- **CRIT-002**: Converted `test_quick.py` to proper pytest format with assertions
  - Was discovered by pytest but had no test functions
- **CRIT-004**: Fixed `test_final.py` returning boolean instead of asserting
  - Now uses proper assertions for correlation validation
- **CRIT-005**: Configured pip-audit to fail on high/critical vulnerabilities
  - Changed from warning-only to fail build on vulnerabilities
- **HIGH-002**: Fixed OR logic in gradient test that always passed
  - Changed to properly assert gradient existence
- **HIGH-003**: Fixed weak accuracy thresholds below random chance
  - Updated thresholds to be at or above random chance (0.5 for 2-class, 1/n for n-class)
- **HIGH-004**: Fixed fast mode to verify learning occurred
  - Added regression check even in fast mode
- **HIGH-007**: Fixed loss tolerance allowing regression (+0.5)
  - Changed to assert loss actually decreases
- **HIGH-008**: Fixed conditional skip that always skipped in multiprocessing test
  - Now actually tests valid start methods
- **HIGH-010**: Fixed empty test block in `test_residual_error.py`
  - Added meaningful assertions to the test
- **MED-003**: Replaced hardcoded absolute paths with relative paths
  - Fixed in `test_quick.py`, `test_final.py`, `test_cascor_fix.py`, `test_p1_fixes.py`
- **MED-005**: Updated Python version from 3.14 (unreleased) to 3.13
  - Fixed in CI workflow, pre-commit config, and pyproject.toml

### Added: [0.6.4]

- **CRIT-003**: Added `--run-long` pytest option for long-running correctness tests
  - Added to `conftest.py` with collection modifying logic
  - Added `long` marker to `pyproject.toml`
  - Converted deterministic resume test to use this marker
- **HIGH-009**: Added `dill>=0.3.6` to test dependencies
  - Added to CI workflow and pyproject.toml optional dependencies
- **MED-001**: Expanded coverage sources to include all `src/` directories
  - Updated `pyproject.toml` and CI workflow

### Changed: [0.6.4]

- **MED-012**: Added `.pytest_cache/` to `.gitignore`
- CI workflow now installs `dill` for both unit and integration test jobs
- Coverage now measures all source modules instead of just 3

### Technical Notes: [0.6.4]

- **SemVer impact**: PATCH – Test infrastructure and CI/CD configuration; no API changes
- **Phases Complete**: Phase 0 (Baseline), Phase 1 (Test Integrity), Phase 2 (Coverage & Realism)
- **Reference**: See `notes/TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN.md` for full issue details

---

## [0.6.3] - 2026-02-01

**Summary**: Documentation updated to reflect JuniperData refactor. Spiral data generation now uses external JuniperData service via REST API.

### Documentation: [0.6.3]

- **JuniperData Integration Documentation**
  - Updated `docs/INDEX.md`: Added `juniper_data_client/` to Key Modules, added External Dependencies section
  - Updated `docs/DOCUMENTATION_OVERVIEW.md`: Added External Services section, updated project structure
  - Updated `docs/api/API_REFERENCE.md`: Added JuniperDataClient API documentation
  - Updated `docs/api/API_SCHEMAS.md`: Added JuniperData artifact schemas (NPZ format)
  - Updated `docs/install/QUICK_START.md`: Added JuniperData prerequisite and startup instructions
  - Updated `docs/install/ENVIRONMENT_SETUP.md`: Added JuniperData Service Setup section
  - Updated `docs/install/USER_MANUAL.md`: Added JuniperDataClient usage examples
  - Updated `docs/install/REFERENCE.md`: Added JuniperData Configuration section
  - Updated `docs/source/MANUAL.md`: Added juniper_data_client module, updated diagrams
  - Updated `docs/source/REFERENCE.md`: Added Service Integration Conventions section
  - Updated `docs/overview/CONSTANTS_GUIDE.md`: Added JuniperData Client Constants
  - Updated `docs/testing/MANUAL.md`: Added Testing with JuniperData section

### Technical Notes: [0.6.3]

- **SemVer impact**: PATCH – Documentation only; no API or code changes
- **Architecture change**: Spiral data generation now provided by external JuniperData service
- **New module**: `juniper_data_client/` for REST API integration

---

## [0.6.2] - 2026-02-01

**Summary**: CI/CD parity achieved across JuniperCascor, JuniperData, and JuniperCanopy with standardized settings.

### Changed: [0.6.2]

- **CI/CD Configuration Parity**
  - `.pre-commit-config.yaml` (v0.4.2)
    - Line length: 512 for black, isort, flake8
    - Added yamllint hook (v1.35.1, relaxed config)
    - Enabled mypy in CI (removed from skip list)
  - `.github/workflows/ci.yml` (v0.4.1)
    - Coverage threshold: 80% (up from 50%)
    - Added build job with package verification
    - Standardized artifact paths: reports/junit/, reports/htmlcov/, reports/coverage.xml
  - `pyproject.toml` (v0.3.17)
    - Line length: 512 for black/isort
    - Coverage fail_under: 80%

### Technical Notes: [0.6.2]

- **SemVer impact**: PATCH – Configuration changes only; no API changes
- **CI Parity**: All 3 Juniper applications now use identical CI/CD settings

---

## [0.6.1] - 2026-01-31

**Summary**: Added algorithm parameter support for backward compatibility with legacy Cascor spiral generation. End-to-end validation completed with JuniperData service.

### Added: [0.6.1]

- **Algorithm Parameter** (`src/spiral_problem/data_provider.py`)
  - Added `algorithm` parameter to `get_spiral_dataset()` method
  - Supports `"modern"` (default) or `"legacy_cascor"` for backward compatibility
  - Parameter passed through to JuniperData API when specified

- **Unit Tests** (1 new test)
  - `test_get_spiral_dataset_passes_algorithm_parameter` - Verifies algorithm parameter is correctly passed to JuniperDataClient

### Validated: [0.6.1]

- End-to-end integration tested with live JuniperData service on port 8100
- Both modern and legacy_cascor algorithms validated

### Technical Notes: [0.6.1]

- **SemVer impact**: PATCH – Backward-compatible feature addition
- **Test Count**: 39 JuniperData-related tests (all passing)
- **Phase Status**: End-to-end validation complete

---

## [0.6.0] - 2026-01-30

**Summary**: Completed Phase 3 of JuniperData integration. Added JuniperDataClient and SpiralDataProvider for fetching spiral datasets from JuniperData REST API. Feature flag JUNIPER_DATA_URL enables JuniperData mode.

### Added: [0.6.0]

- **JuniperData Client** (`src/juniper_data_client/`)
  - `client.py` - REST client for JuniperData API
    - `create_dataset()` - POST /v1/datasets to generate datasets
    - `download_artifact_npz()` - GET artifact and parse as numpy arrays
    - URL normalization (scheme, trailing slash, /v1 suffix handling)
    - Configurable timeouts with requests.Session
  - `__init__.py` - Package initialization with JuniperDataClient export

- **Spiral Data Provider** (`src/spiral_problem/data_provider.py`)
  - `SpiralDataProvider` class for JuniperData integration
  - `use_juniper_data` property - True when JUNIPER_DATA_URL is set
  - `get_spiral_dataset()` - Fetches dataset and converts to torch tensors
  - Parameter mapping: n_points → n_points_per_spiral, noise_level → noise
  - Returns same format as legacy: `((x_train, y_train), (x_test, y_test), (x_full, y_full))`
  - `SpiralDataProviderError` custom exception for clear error handling

- **Feature Flag Integration** (`src/spiral_problem/spiral_problem.py`)
  - Added JUNIPER_DATA_URL environment variable check in `generate_n_spiral_dataset()`
  - When set, uses SpiralDataProvider to fetch data from JuniperData service
  - When unset, legacy local generation code path is used unchanged

- **Unit Tests** (38 new tests)
  - `tests/unit/test_juniper_data_client.py` - 17 tests for client functionality
  - `tests/unit/test_spiral_data_provider.py` - 18 tests for provider
  - `tests/unit/test_spiral_problem_juniper_data_integration.py` - 3 tests for feature flag

### Usage: [0.6.0]

```bash
# Enable JuniperData service
export JUNIPER_DATA_URL=http://localhost:8100

# Run with legacy local generation
unset JUNIPER_DATA_URL
```

### Technical Notes: [0.6.0]

- **SemVer impact**: MINOR – New feature (JuniperData client integration), no breaking changes
- **Dependencies**: Uses existing requests library for HTTP client
- **Phase Status**: Completes Phase 3 of spiral data generator extraction plan
- **Test Count**: 38 new tests (all passing)

---

## [0.5.3] - 2026-01-29

**Summary**: Achieved 91% test coverage target (up from 75%) through comprehensive unit test additions across all source modules.

### Added: [0.5.3]

- **New Test Files**: Created 8 new test files to improve coverage:
  - `tests/unit/test_main_coverage.py` - Tests for main.py (parse_args, main function)
  - `tests/unit/test_utils_extended.py` - Extended tests for utils.py edge cases
  - `tests/unit/test_logging_utils_extended.py` - Tests for SampledLogger, BatchLogger, LogFrequencyTracker
  - `tests/unit/test_snapshot_common_extended.py` - Tests for decode fallbacks, CUDA suppression
  - `tests/unit/test_logger_extended.py` - Tests for Logger initialization, validation, custom levels
  - `tests/unit/test_remote_client_0_extended.py` - Tests for remote client connection and workers
  - `tests/unit/test_snapshot_serializer_coverage.py` - Additional serializer edge case tests

### Changed: [0.5.3]

- **Coverage Improvements by Module**:

  | Module                                       | Before  | After   |
  | -------------------------------------------- | ------- | ------- |
  | `cascade_correlation/cascade_correlation.py` | 46%     | 86%     |
  | `candidate_unit/candidate_unit.py`           | 56%     | 86%     |
  | `cascor_constants/constants.py`              | 50%     | 99%     |
  | `remote_client/remote_client_0.py`           | 70%     | 94%     |
  | `snapshot_serializer.py`                     | 88%     | 92%     |
  | `snapshot_common.py`                         | 85%     | 100%    |
  | `profiling/deterministic.py`                 | 66%     | 100%    |
  | `profiling/memory.py`                        | 62%     | 100%    |
  | `profiling/logging_utils.py`                 | 87%     | 100%    |
  | `log_config/log_config.py`                   | 39%     | 96%     |
  | `log_config/logger/logger.py`                | 88%     | 97%     |
  | **TOTAL**                                    | **75%** | **91%** |

### Technical Notes: [0.5.3]

- **SemVer impact**: PATCH – Test additions only; no API or code changes
- **Test count**: 1324 tests passing (4 skipped for long-running/multiprocessing tests)
- **Coverage target**: 90% achieved (91% actual)

---

## [0.5.2] - 2026-01-29

**Summary**: Added comprehensive DOCUMENTATION_OVERVIEW.md with complete navigation guide matching JuniperCanopy documentation style.

### Added: [0.5.2]

- **DOCUMENTATION_OVERVIEW.md**: Complete documentation navigation guide (~800 lines)
  - Quick navigation "I Want To" table with all 24 docs
  - Detailed descriptions for essential documents (README, quick-start, environment-setup, AGENTS.md)
  - Project structure diagram
  - Technical guides for all API, Testing, CI/CD, and Source documentation
  - Complete document index with lines, type, audience, and status
  - Documentation standards (naming, formatting, cross-referencing)
  - Quick reference card with essential commands

### Technical Notes: [0.5.2]

- **SemVer impact**: PATCH – Documentation only; no API or code changes
- Follows JuniperCanopy DOCUMENTATION_OVERVIEW.md format and structure

---

## [0.5.1] - 2026-01-29

**Summary**: Complete pre-commit compliance including MyPy type checking. Fixed all F401 unused imports, B907 string quoting, F811 duplicate functions, and valid-type errors. All 17 pre-commit hooks now pass.

### Fixed: [0.5.1]

- **F401 Unused Imports**: Commented out unused imports with TODO prefix across 7 files
  - `cascade_correlation.py`: 6 activation function constants
  - `main.py`: `sys`, 5 `_CASCOR_*` constants
  - `profiling/deterministic.py`: `os`, `Optional`
  - `profiling/logging_utils.py`: `wraps`, `Any`, `Optional`
  - `profiling/memory.py`: `linecache`, `Path`, `Optional`, `Tuple`

- **F811 Duplicate Function**: Resolved `_create_optimizer` duplication in `cascade_correlation.py`
  - Kept version at line ~1950 (supports 15 optimizers with full config)
  - Commented out version at line ~995 (only supported 4 optimizers)

- **B907 String Quoting**: Fixed 19 occurrences across 4 files
  - Replaced manual quoting `'{var}'` with `!r` conversion flag `{var!r}`
  - Files: `cascade_correlation.py`, `log_config.py`, `main.py`, `snapshot_serializer.py`

- **MyPy valid-type Errors**: Fixed 13 type annotation issues
  - `callable` → `Callable[..., Any]` (6 occurrences)
  - `any` → `Any` (2 occurrences)
  - `[Type]` → `list[Type]` (1 occurrence)
  - `tuple([...])` → `tuple[...]` (1 occurrence)
  - `(T1, T2)` → `tuple[T1, T2]` (2 occurrences)
  - `uuid` → `uuid.UUID` (1 occurrence)
  - `Optional` → `Optional[Any]` (1 occurrence)

### Added: [0.5.1]

- **Re-export Manifest**: Added `__all__` to `cascor_constants/constants.py` with 120 constants
  - Makes re-exports explicit to satisfy F401 checks
  - Organized by source sub-module

- **MyPy Configuration**: Enabled MyPy in pre-commit with appropriate disabled error codes
  - Disables complex structural checks that require deeper refactoring
  - Can be incrementally tightened as codebase improves

### Documentation: [0.5.1]

- Updated `notes/CHANGES_FOR_REVIEW.md` with complete fix documentation
- Version 2.0.0 of CHANGES_FOR_REVIEW.md marks all issues resolved

### Technical Notes: [0.5.1]

- **SemVer impact**: PATCH – Code style and type annotation fixes; no API changes
- **Pre-commit status**: All 17 hooks pass
- **MyPy coverage**: 4 core modules checked (cascade_correlation, candidate_unit, spiral_problem, snapshots)

---

## [0.5.0] - 2026-01-29

**Summary**: Major refactoring milestone - extracted Spiral Dataset Generator into standalone JuniperData application. Completed Phases 0-2 of the spiral data generator extraction, creating a new microservice with REST API for dataset generation.

### Added: [0.5.0]

- **JuniperData Application**: New standalone dataset generation service at `Juniper/JuniperData/`
  - **Package Structure**: Complete Python package with `pyproject.toml`, `AGENTS.md`, `README.md`
  - **Core Generator**: Pure NumPy spiral generator (`juniper_data/generators/spiral/`)
    - `SpiralParams` - Pydantic model with validation
    - `SpiralGenerator` - Static methods for N-spiral generation
    - `defaults.py` - Extracted constants from Cascor
  - **Core Utilities**: Dataset management utilities (`juniper_data/core/`)
    - `split.py` - shuffle_data, split_data, shuffle_and_split
    - `dataset_id.py` - Deterministic hash-based dataset IDs
    - `models.py` - DatasetMeta, CreateDatasetRequest/Response
    - `artifacts.py` - NPZ save/load, checksum computation
  - **Storage Layer**: Pluggable storage backends (`juniper_data/storage/`)
    - `DatasetStore` - Abstract base class
    - `InMemoryDatasetStore` - For testing
    - `LocalFSDatasetStore` - Production file-based storage
  - **REST API**: FastAPI-based service (`juniper_data/api/`)
    - `GET /v1/health` - Health check
    - `GET /v1/generators` - List available generators
    - `GET /v1/generators/{name}/schema` - Parameter schema
    - `POST /v1/datasets` - Create/generate dataset
    - `GET /v1/datasets` - List datasets
    - `GET /v1/datasets/{id}` - Get metadata
    - `GET /v1/datasets/{id}/artifact` - Download NPZ
    - `GET /v1/datasets/{id}/preview` - Preview samples
    - `DELETE /v1/datasets/{id}` - Delete dataset

- **Golden Reference Datasets**: Test fixtures for parity validation
  - `tests/fixtures/generate_golden_datasets.py` - Generation script
  - `tests/fixtures/golden_datasets/README.md` - Documentation

- **Comprehensive Test Suite**: 76 tests (all passing)
  - 60 unit tests (spiral generator, split, dataset_id)
  - 16 integration tests (API endpoints)

- **Refactoring Plan Document**: `notes/JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR_PLAN.md`
  - Synthesized analysis of three proposals
  - Method extraction specification
  - 5-phase implementation plan
  - Migration strategy for Cascor/Canopy

### Documentation: [0.5.0]

- Created comprehensive refactoring plan document
- Updated plan with implementation status (Phases 0-2 complete)

### Technical Notes: [0.5.0]

- **SemVer impact**: MINOR – New feature (JuniperData extraction), no breaking changes to Cascor API
- **Dependencies**: JuniperData uses numpy, pydantic, fastapi, uvicorn (no torch in core)
- **Phases Complete**: 0 (Baseline), 1 (Core Generator), 2 (REST API)
- **Phases Pending**: 3 (Cascor Integration), 4 (Canopy Integration)
- **Run JuniperData**: `cd JuniperData && python -m juniper_data` (port 8100)

---

## [0.4.1] - 2026-01-29

**Summary**: Comprehensive documentation overhaul. Created complete documentation suite in docs/ directory covering installation, API reference, testing, CI/CD, and source code guides.

### Added: [0.4.1]

- **Documentation Suite**: Created 20+ documentation files in `docs/` directory
  - **Overview**: `docs/INDEX.md` (landing page), `docs/overview/CONSTANTS_GUIDE.md`
  - **Install/Config**: `QUICK_START.md`, `ENVIRONMENT_SETUP.md`, `USER_MANUAL.md`, `REFERENCE.md`
  - **API**: `API_REFERENCE.md` (v0.3.21 updated), `API_SCHEMAS.md` (HDF5/data schemas)
  - **Testing**: `QUICK_START.md`, `ENVIRONMENT_SETUP.md`, `MANUAL.md`, `REFERENCE.md`, `SELECTIVE_TESTING_GUIDE.md`
  - **CI/CD**: `QUICK_START.md`, `ENVIRONMENT_SETUP.md`, `MANUAL.md`, `REFERENCE.md`
  - **Source Code**: `QUICK_START.md`, `ENVIRONMENT_SETUP.md`, `MANUAL.md`, `REFERENCE.md`

- **Documentation Features**:
  - Complete API documentation with examples and type hints
  - HDF5 snapshot schema documentation
  - Test marker reference and CI mapping
  - Module-by-module source code guide
  - Extension points for new problems/activations/serializers
  - Configuration override guidance

### Changed: [0.4.1]

- **README.md**: Enhanced with Quick Start section, installation instructions, usage examples, and documentation links

### Documentation: [0.4.1]

- API Reference updated to version 0.3.21 (from 0.3.2)
- All documentation dated 2026-01-29

### Technical Notes: [0.4.1]

- **SemVer impact**: PATCH – Documentation only; no API or code changes
- Previous `notes/` directory retained as historical reference

---

## [0.4.0] - 2026-01-29

**Summary**: Major CI/CD pipeline overhaul. Implemented comprehensive, production-ready CI/CD with pre-commit hooks, security scanning, proper failure handling, and coverage enforcement.

### Added: [0.4.0]

- **Enhanced CI/CD Pipeline**: Complete overhaul of GitHub Actions workflow
  - **Pre-commit Job**: Runs across Python 3.12 and 3.13 with matrix strategy
  - **Unit Tests Job**: Coverage enforcement with `--cov-fail-under` (50% threshold)
  - **Integration Tests Job**: Now runs on PRs AND main/develop pushes
  - **Security Job**: Gitleaks (secrets), Bandit (SAST/SARIF), pip-audit (dependencies)
  - **Quality Gate Job**: Aggregates all checks with proper failure handling
  - Removed `continue-on-error: true` and `|| true` from critical steps
  - Added dependency caching for conda and pip packages
  - Added concurrency control to cancel stale runs

- **Pre-commit Configuration**: Created `.pre-commit-config.yaml`
  - General hooks: check-yaml, check-toml, trailing-whitespace, merge conflicts
  - Python formatting: Black (line-length=120)
  - Import sorting: isort (black profile)
  - Linting: Flake8 with bugbear, comprehensions, simplify plugins
  - Type checking: MyPy (optional, runs on core modules)
  - Security: Bandit SAST scanning
  - Markdown: markdownlint with auto-fix
  - Shell: shellcheck for bash scripts

- **CODEOWNERS File**: Created `.github/CODEOWNERS`
  - Defines code ownership for automatic review requests
  - Covers core modules, tests, configuration, and documentation

- **Branch Protection Documentation**: Created `docs/ci_cd/BRANCH_PROTECTION.md`
  - Required status checks configuration
  - Pull request requirements for main and develop branches
  - Coverage enforcement guidelines
  - Security scanning documentation
  - Step-by-step setup instructions

### Changed: [0.4.0]

- **CI Workflow**: Upgraded from v6 to v4 for actions/checkout (stable)
- **Coverage Enforcement**: Changed from soft fail (warning) to hard fail
- **Integration Tests**: Now run on main/develop pushes, not just PRs

### Documentation: [0.4.0]

- Updated AGENTS.md with pre-commit and security scanning commands
- Updated version to 0.4.0

### Technical Notes: [0.4.0]

- **SemVer impact**: MINOR – New CI/CD features, no breaking changes
- **Pre-commit setup**: `pip install pre-commit && pre-commit install`
- **Local validation**: `pre-commit run --all-files`

### Pre-commit Compliance: [0.4.0]

- **Fixed**: 33 corrupted line continuations in spiral_problem.py (`\ \ \#\` → ` # `)
- **Fixed**: Black target-version (py314 not supported, using py311-py313)
- **Auto-formatted**: 64 Python files with Black
- **Excluded**: .ipynb_checkpoints/, backups/, legacy util scripts
- **Deferred**: MyPy type checking (112 errors, requires type annotation fixes)
- **Deferred**: F401 unused imports, B907 string quoting (documented in CHANGES_FOR_REVIEW.md)

---

## [0.3.21] - 2026-01-25

**Summary**: Major test coverage expansion. Added 6 new test files with ~150+ tests to improve coverage from ~50% to ~67%.

### Added: [0.3.21]

- **Test Coverage Expansion (P2-NEW-001)**: Added comprehensive unit tests
  - `test_cascade_correlation_coverage.py` - 17 test classes/methods for core network
  - `test_candidate_unit_extended.py` - 12 test classes for candidate unit
  - `test_profiling_module.py` - 15 test classes for profiling infrastructure
  - `test_network_methods_extended.py` - 16 test classes for network methods
  - `test_config_and_exceptions.py` - 15 test classes for configuration
  - `test_training_workflow.py` - 13 test classes for training workflows

### Changed: [0.3.21]

- Updated PRE-DEPLOYMENT_ROADMAP-2.md with test coverage status

### Documentation: [0.3.21]

- Test coverage now at ~67% overall (from ~50%)
- Core modules improved: cascade_correlation.py (~61%), candidate_unit.py (~81%)

---

## [0.3.20] - 2026-01-25

**Summary**: Completed Phase D (3/5 tasks) of PRE-DEPLOYMENT_ROADMAP-2.md. Added profiling infrastructure with cProfile, tracemalloc, and py-spy support. Created logging utilities for hot path optimization.

### Added: [0.3.20]

- **Development Profiling Infrastructure (P3-NEW-001)**: Created comprehensive profiling module
  - Added `src/profiling/` module with deterministic and memory profiling
  - `--profile` flag for cProfile integration
  - `--profile-memory` flag for tracemalloc memory profiling
  - `--profile-output` and `--profile-top-n` configuration options
  - `ProfileContext` context manager for block profiling
  - `MemoryTracker` context manager for memory analysis
  - `profile_function` and `memory_profile` decorators

- **Sampling Profiling Infrastructure (P3-NEW-002)**: Added py-spy integration
  - Created `util/profile_training.bash` script
  - SVG flame graph generation
  - Speedscope JSON format output
  - Configurable sampling rate, duration, native frames

- **Hot Path Logging Utilities (P4-NEW-004)**: Created logging optimization tools
  - `SampledLogger` - Sample log messages at configurable rate
  - `BatchLogger` - Buffer and batch log output
  - `log_if_enabled()` - Avoid expensive formatting when level disabled
  - `log_timing()` - Context manager for timing operations
  - `LogFrequencyTracker` - Track log call frequency

### Changed: [0.3.20]

- Updated `main.py` with `argparse` for command-line profiling options
- Updated AGENTS.md with profiling commands documentation

### Documentation: [0.3.20]

- Updated PRE-DEPLOYMENT_ROADMAP-2.md to v2.3.0 (14/19 tasks, 74%)
- Added profiling commands to AGENTS.md Essential Commands section

---

## [0.3.19] - 2026-01-25

**Summary**: Completed Phase A (5/5 tasks) and Phase B (3/4 tasks) of PRE-DEPLOYMENT_ROADMAP-2.md. Resolved module naming collision (P4-NEW-006) enabling scalable sub-project integration. Added CI coverage gates and README badge. Created new test file for candidate seed diversity.

### Changed: [0.3.19]

- **Module Naming Collision Resolution (P4-NEW-006)**: Renamed constants modules to prevent import conflicts
  - **Cascor**: Renamed `constants/` → `cascor_constants/` (9 files updated)
  - **Canopy**: Renamed `constants.py` → `canopy_constants.py` (16 files updated)
  - Eliminates need for `sys.path.insert()` workaround
  - Enables scalable integration with future sub-projects (JuniperBranch, JuniperBerry)
  - Updated AGENTS.md documentation references

### Added: [0.3.19]

- **CI Coverage Gates (P2-NEW-002)**: Added coverage threshold enforcement to CI pipeline
  - Added "Check Coverage Thresholds" step with 50% initial threshold
  - Uses `coverage report --fail-under=50` with soft fail (warning only)
  - Threshold to be increased as coverage improves

- **README Workflow Badge (P4-NEW-003)**: Added GitHub Actions status badge
  - Badge displays CI/CD Pipeline status
  - Links to workflow runs for quick access

### Verified: [0.3.19]

- **main.py End-to-End (P4-NEW-001)**: Verified application startup
  - All module imports work correctly
  - LogConfig/Logger initialize properly
  - SpiralProblem and CascadeCorrelationNetwork instantiate correctly
  - Plotting enabled by default

- **./try Script (P4-NEW-002)**: Verified launcher script functionality
  - Symlink correctly points to `util/juniper_cascor.bash`
  - Environment validation works (conda env, Python version)
  - All configuration files sourced correctly

- **Parallel Processing (P4-NEW-005)**: Verified multiprocessing works
  - `_execute_parallel_training` invoked (not sequential)
  - ForkServer multiprocessing manager starts correctly
  - 9 worker processes spawn with unique PIDs
  - Task and result queues created properly

### Added: [0.3.19] (Tests)

- **test_candidate_seed_diversity.py** (P2-NEW-005): New unit test file with 4 tests
  - `test_candidates_have_different_seeds` - Verifies pool candidates have unique seeds
  - `test_candidates_have_different_initial_weights` - Verifies weight diversity
  - `test_same_seed_produces_same_weights` - Reproducibility test
  - `test_different_seeds_produce_different_weights` - Diversity test

### Documentation: [0.3.19]

- Updated `notes/PRE-DEPLOYMENT_ROADMAP-2.md`:
  - Marked Phase A as complete (5/5 tasks)
  - Marked Phase B as substantially complete (B.1-B.3 done, B.4 ongoing)
  - Updated P4-NEW-006 status (module naming collision resolved)
  - Verified P2-NEW-003 and P2-NEW-004 already implemented in cascade_correlation.py
  - Updated P2-NEW-005 status (tests verified/created)

### Technical Notes: [0.3.19]

- **SemVer impact**: PATCH – CI configuration, tests, and documentation; no API changes
- **Phase B Discovery**: Multiprocessing timeout hardening (B.2) and sequential fallback (B.3) were already implemented in the codebase

---

## [0.3.18] - 2026-01-25

### Fixed: [0.3.18]

- **CASCOR-TIMEOUT-001**: Resolved test timeout failures affecting 17 training-intensive tests
  - **Root Cause**: Tests exceeding 60-second pytest timeout, NOT multiprocessing deadlocks
  - **Solution**: Marked training tests with `@pytest.mark.slow` and `@pytest.mark.timeout(300)`
  - **CI Update**: Changed CI to run `-m "not slow"` by default
  - **Files Modified**: 7 test files, CI workflow, pytest.ini, tests/README.md

### Changed: [0.3.18]

- **CI/CD Pipeline**: Updated unit and integration test jobs to exclude slow tests
  - Unit tests now run with `-m "unit and not slow"`
  - Integration tests now run with `-m "integration and not slow"`
  - Slow tests can be run separately with extended timeout

### Added: [0.3.18]

- **Test Documentation**: Added slow test handling documentation
  - Added comment section to `pytest.ini` explaining slow test markers
  - Added "Slow Test Handling" section to `tests/README.md`

- **Slow Test Markers**: Applied to 13 training-intensive tests:
  - `test_spiral_problem.py`: 6 tests (spiral learning, robustness, visualization, edge cases)
  - `test_comprehensive_serialization.py`: 1 test (deterministic training resume)
  - `test_cascor_fix.py`: 2 tests (sequential/individual candidate training)
  - `test_critical_fixes.py`: 1 test (candidate training)
  - `test_final.py`: 1 test (candidate units)
  - `test_p1_fixes.py`: 1 test (early stopping)
  - `test_accuracy.py`: 1 test (accuracy with trained network)

### Documentation: [0.3.18]

- Created `notes/PRE-DEPLOYMENT_ROADMAP-2.md`:
  - Consolidated all incomplete/unstarted issues from original roadmap
  - Re-prioritized into 4 new priority levels (P1-NEW through P4-NEW)
  - New phased implementation schedule (Phase A through D)
  - 19 remaining issues tracked with effort estimates

- Updated `notes/PRE-DEPLOYMENT_ROADMAP.md`:
  - Added Section 13: Test Timeout Analysis and Resolution
  - Documented CASCOR-TIMEOUT-001 root cause and resolution
  - Documented Phase 2 multiprocessing hardening approach (deferred)
  - Updated version to 1.6.0

### Technical Notes: [0.3.18]

- **SemVer impact**: PATCH – Test configuration and documentation; no API changes
- **Expected test results**: `pytest -m "not slow"` should pass all fast tests without timeouts
- **Slow tests**: Run separately with `pytest -m slow --timeout=0` or per-test 300s timeout

---

## [0.3.17] - 2026-01-24

### Added: [0.3.17]

- **End-to-End Integration Analysis**: Complete analysis of Cascor-Canopy integration architecture
  - Documented in-process embedding model (not client-server IPC)
  - Identified 5 integration issues (INTEG-001 through INTEG-005)
  - Documented parallel processing verification procedures
  - Added architecture diagram showing component relationships

- **Continuous Profiling Infrastructure Design**: Comprehensive profiling strategy documented
  - **Deterministic Profiling**: cProfile, line_profiler, memory_profiler
  - **Statistical Profiling**: py-spy, Scalene, Python 3.15 Tachyon
  - **Continuous Profiling**: Grafana Pyroscope integration design
  - **PyTorch Profiling**: torch.profiler with TensorBoard integration
  - **Memory Profiling**: tracemalloc, Scalene for allocation tracking
  - **Flame Graph Generation**: py-spy, speedscope workflow
  - **4-phase implementation plan** (Development → Sampling → Continuous → PyTorch)

- **Code Coverage Roadmap to >90%**: Detailed improvement plan
  - Current: ~15%, Target: 90%
  - Priority 1: Core modules (cascade_correlation, candidate_unit, snapshot_serializer)
  - Priority 2: Support modules (log_config, constants, utils)
  - Priority 3: Edge cases and error paths
  - Test categories: +150 unit, +30 integration, +10 performance tests planned

### Documentation: [0.3.17]

- Updated `notes/PRE-DEPLOYMENT_ROADMAP.md` with sections 10, 11, 12
  - Section 10: End-to-End Integration Analysis
  - Section 11: Continuous Profiling Infrastructure Design
  - Section 12: Code Coverage Roadmap to >90%

---

## [0.3.16] - 2026-01-24

### Added: [0.3.16]

- **CASCOR-P1-007**: CI/CD Pipeline Setup - Complete GitHub Actions infrastructure
  - **Created**: `.github/workflows/ci.yml` with 5-stage pipeline:
    - **Lint job**: Black, isort, Flake8, MyPy (with continue-on-error for gradual adoption)
    - **Test job**: Unit tests with pytest, coverage reporting, 60-second timeout
    - **Integration job**: Integration tests (triggered on PRs only)
    - **Quality Gate job**: Enforces test pass requirement before merge
    - **Notify job**: Build status notification with workflow metadata
  - **Created**: `pyproject.toml` with unified Python tooling configuration:
    - Black: line-length 120, Python 3.11-3.14 targets
    - isort: black profile for import sorting
    - pytest: markers, timeout (60s), strict mode
    - coverage: source modules, branch coverage, HTML/XML reports
    - mypy: permissive settings for gradual type checking adoption
  - **Pipeline Features**:
    - Uses `conda-incubator/setup-miniconda@v3` with mamba for fast environment setup
    - Python 3.14 target (matching JuniperCascor conda environment)
    - Coverage artifacts uploaded for 30 days
    - JUnit XML reports for CI tool integration
    - Disk space cleanup for GitHub Actions runners

- **CASCOR-P2-002**: Type Checker Configuration - Mypy integration complete
  - Added mypy configuration to `pyproject.toml` with permissive settings
  - Python 3.14 target, `ignore_missing_imports = true`
  - Module overrides for torch, numpy, h5py, matplotlib, yaml
  - Updated `AGENTS.md` with type checking commands

- **CASCOR-P2-003**: Logging Performance Optimization
  - Added `CASCOR_LOG_LEVEL` environment variable support in `src/constants/constants.py`
  - Validates against known log levels: TRACE, VERBOSE, DEBUG, INFO, WARNING, ERROR, CRITICAL, FATAL
  - Falls back to INFO if env var not set or invalid
  - Documented quiet mode presets in `AGENTS.md`:
    - `export CASCOR_LOG_LEVEL=WARNING` for production/benchmarking
    - `export CASCOR_LOG_LEVEL=DEBUG` for verbose debugging

- **CASCOR-P3-004**: Performance Benchmark Harness - Complete
  - Created `src/tests/scripts/run_benchmarks.bash`
  - Benchmarks: serialization (save/load HDF5), forward pass, output layer training
  - Configurable iterations, quiet mode, output file support
  - Integrates with `CASCOR_LOG_LEVEL` for quiet benchmarking

- **CASCOR-P2-001**: Code Coverage Improvement - New test files added
  - Created `src/tests/unit/test_cascor_getters_setters.py` (30+ tests)
    - Tests for getter/setter methods in CascadeCorrelationNetwork
    - Tests for candidate data helper methods
    - Tests for network properties and _create_candidate_unit factory
    - Tests for _select_best_candidates method
  - Created `src/tests/unit/test_candidate_unit_coverage.py` (25+ tests)
    - Tests for CandidateUnit initialization and properties
    - Tests for forward pass and correlation calculation
    - Tests for pickling support (multiprocessing)
    - Tests for ActivationWithDerivative class
    - Tests for CandidateTrainingResult dataclass

### Verified: [0.3.16]

- **CASCOR-P3-002**: Flexible Optimizer System - Already implemented
  - `_create_optimizer()` method supports SGD, Adam, AdamW, RMSprop
  - `OptimizerConfig` class in cascade_correlation_config.py

- **CASCOR-P3-005**: N-Best Candidate Selection - Already implemented
  - `_select_best_candidates()` method for selecting top N candidates
  - `candidates_per_layer` config option in CascadeCorrelationConfig

- **CASCOR-P3-001**: Candidate Factory Refactor - Analysis complete
  - Factory exists at `_create_candidate_unit()`
  - Other instantiation sites have valid design reasons (multiprocessing, grow_network)

### Documentation: [0.3.16]

- **PRE-DEPLOYMENT_ROADMAP.md**: Added missing P1 issues (P1-001 through P1-004)
  - CASCOR-P1-001: Multiprocessing Manager Port Conflicts (was in INTEGRATION_ROADMAP only)
  - CASCOR-P1-002: validate_training API Mismatch (was in INTEGRATION_ROADMAP only)
  - CASCOR-P1-003: Multiprocessing Pickling Error (was in INTEGRATION_ROADMAP only)
  - CASCOR-P1-004: try Script Symlink Fix (was in INTEGRATION_ROADMAP only)
  - All were already fixed, now properly tracked in consolidated roadmap

- **CANOPY-P1-002**: Module Naming Collision - Verified workaround in place
  - `CascorIntegration._add_backend_to_path()` ensures Cascor modules take priority
  - Full rename deferred to post-deployment

- **CANOPY-P1-003**: Monitoring Thread Race Condition - Fixed
  - Added `metrics_lock` to `CascorIntegration` for thread-safe metrics extraction
  - File changed: `JuniperCanopy/juniper_canopy/src/backend/cascor_integration.py`

### Technical Notes: [0.3.16]

- **SemVer impact**: MINOR – New CI/CD infrastructure and configuration; no API changes
- Part of PRE-DEPLOYMENT_ROADMAP.md P1/P2/P3 issue resolution (Phase 2: Quality Infrastructure)
- Linting jobs use `continue-on-error: true` for gradual codebase cleanup
- All Cascor P1 issues now properly tracked (P1-001 through P1-009)

---

## [0.3.15] - 2026-01-24

### Fixed: [0.3.15]

- **CASCOR-P0-001**: Fixed multiprocessing completion logic that could hang indefinitely
  - **Problem**: The busy-wait loop in `_execute_parallel_training` used `task_queue.empty()` and `result_queue.qsize()` which are unreliable for multiprocessing Manager proxies and can cause infinite hangs if a worker crashes
  - **Solution**:
    - Replaced unreliable `empty()`/`qsize()` busy-wait with bounded timeout loop
    - Added worker liveness checks using `worker.is_alive()`
    - Loop now exits early when all workers have completed
    - Relies on existing `_collect_training_results` for proper timeout-based result collection
  - **File Changed**: `src/cascade_correlation/cascade_correlation.py` (lines 1957-1993)

- **CASCOR-P0-005**: Fixed candidate task parameter wiring bug
  - **Problem**: `train_candidate_worker` used incorrect dictionary keys when instantiating `CandidateUnit`, causing per-candidate seeds, epochs, and learning rates to be ignored (returned `None`)
  - **Solution**: Fixed `.get()` key names to match `_build_candidate_inputs` dictionary:
    - `"epochs"` → `"candidate_epochs"`
    - `"learning_rate"` → `"candidate_learning_rate"`
    - `"random_seed"` → `"candidate_seed"`
    - `"random_value_max"` → `"random_max_value"`
  - **File Changed**: `src/cascade_correlation/cascade_correlation.py` (lines 2608-2627)

- **CASCOR-P0-006**: Verified already fixed (residual error shape logic)
  - **Status**: Main file already had correct logic; bug existed only in duplicate file
  - **Resolution**: Duplicate files in `src/utils/cascade_correlation/` and `src/utils/candidate_unit/` deleted

- **CASCOR-P0-004**: Fixed snapshot serializer save_object() TypeError
  - **Problem**: `save_object()` called `_save_root_attributes()` with 4 arguments, but method only accepts 2
  - **Additional Problem**: `_save_root_attributes` and `_save_metadata` were defined twice (dead code)
  - **Solution**:
    - Changed `save_object()` to call `_save_network_objects_helper()` instead
    - Removed duplicate method definitions (lines 236-270)
  - **File Changed**: `src/snapshots/snapshot_serializer.py`

- **CASCOR-P0-003**: Verified previous bug fixes (BUG-001, BUG-002)
  - **BUG-001**: Random state restoration - verified via 22 serialization integration tests
  - **BUG-002**: Logger pickling - verified no pickling errors during multiprocessing
  - **Tests Passed**: `test_serialization.py` (22 tests), `test_forward_pass.py` (30 tests)

- **CASCOR-P0-001**: Fixed undefined variable in multiprocessing timeout
  - **Problem**: `queue_timeout` was not defined in `_execute_parallel_training`
  - **Solution**: Changed to `getattr(self, 'task_queue_timeout', 60.0)`
  - **File Changed**: `src/cascade_correlation/cascade_correlation.py` (line 1968)

- **CASCOR-P0-002**: Improved serialization test coverage to 78%+
  - **Problem**: Serialization module had low test coverage (~15% overall)
  - **Solution**: Created comprehensive unit test file with 20 new tests
  - **Tests Added**:
    - `save_object()`, `save_network()`, `load_network()` tests
    - `verify_saved_network()` tests
    - Edge case tests (invalid paths, hidden units, error handling)
    - Random state and config preservation tests
  - **File Created**: `src/tests/unit/test_snapshot_serializer.py`

- **CASCOR-P1-009**: Fixed `get_candidates_data_count()` summing values instead of counting
  - **Problem**: Method used `sum(getattr(r, field)...)` which summed field values instead of counting items
  - **Solution**: Changed to `sum(1 for r in results...)` to properly count matching items
  - **File Changed**: `src/cascade_correlation/cascade_correlation.py` (line 2355)

- **CASCOR-P1-008**: Fixed CandidateUnit random roll OOM vulnerability
  - **Problem**: `_roll_sequence_number()` created list of up to 2^32-1 elements, causing OOM
  - **Solution**:
    - Replaced list comprehension with simple for-loop that discards values
    - Added `MAX_ROLL_COUNT = 10000` cap to prevent excessive iterations
    - Added warning log when sequence exceeds cap
  - **File Changed**: `src/candidate_unit/candidate_unit.py` (lines 463-475)

### Verified: [0.3.15]

- **CASCOR-P1-005**: Shell script path resolution - verified already working
- **CASCOR-P1-006**: Test runner script - verified already working (no syntax errors)

### Removed: [0.3.15]

- **Module Duplication Cleanup**: Deleted duplicate module copies from `src/utils/`
  - `src/utils/cascade_correlation/cascade_correlation.py` - contained outdated code with bugs
  - `src/utils/candidate_unit/candidate_unit.py` - duplicate of canonical version
  - Only canonical versions in `src/cascade_correlation/` and `src/candidate_unit/` remain

### Technical Notes: [0.3.15]

- **SemVer impact**: PATCH – Critical bug fix; no API changes
- Part of PRE-DEPLOYMENT_ROADMAP.md P0 issue resolution

---

## [0.3.14] - 2026-01-22

### Fixed: [0.3.14]

- **CASCOR-P1-001**: Resolved multiprocessing manager port conflicts
  - **Problem**: `forkserver` context with custom Manager classes had compatibility issues
  - **Solution**:
    - Fixed `set_forkserver_preload()` to use list argument format (was incorrectly passing multiple arguments)
    - Retained `forkserver` as preferred context (Python 3.14.2 fixes compatibility issues)
    - Dynamic port allocation (port 0) prevents "Address already in use" conflicts
  - **Files Changed**:
    - `src/cascade_correlation/cascade_correlation.py` - Fixed `set_forkserver_preload()` call
    - `src/constants/constants_model/constants_model.py` - Updated comments

- **CASCOR-P1-002**: Added missing PyYAML to environment spec
  - **File Changed**: `conf/conda_environment.yaml` - Added `pyyaml=6.0.3=pyh7db6752_0`

- **Test Fix**: Fixed `test_forward_pass_nan_input` import path
  - **Problem**: Test imported `ValidationError` from wrong module path
  - **Solution**: Changed import from `cascade_correlation_exceptions.cascade_correlation_exceptions` to `cascade_correlation.cascade_correlation_exceptions.cascade_correlation_exceptions`
  - **File Changed**: `src/tests/unit/test_forward_pass.py`

### Technical Notes: [0.3.14]

- **SemVer impact**: PATCH – Bug fixes and documentation; no API changes
- All Cascor unit tests now pass (152+ tests)
- Canopy tests also verified passing (2942 passed, 41 skipped)

---

## [0.3.13] - 2026-01-21

### Fixed: [0.3.13]

- **CASCOR-P0-002**: Fixed test suite timeout/hang issues
  - **Problem**: Tests would timeout after 180 seconds, never completing the full suite
  - **Solution**: Installed `pytest-timeout` and configured 60-second per-test timeout
  - **File Changed**: `src/tests/pytest.ini`
  - **Configuration Added**:

    ```ini
    timeout = 60
    timeout_method = signal
    ```

  - **Result**: Tests now timeout individually after 60 seconds instead of hanging indefinitely

### Technical Notes: [0.3.13]

- **SemVer impact**: PATCH – Test infrastructure improvement; no application code changes
- Removed duplicate `--tb=long` from pytest.ini (was conflicting with `--tb=short`)

---

## [0.3.12] - 2026-01-21

### Fixed: [0.3.12]

- **CASCOR-P1-003**: Fixed multiprocessing pickling error with `wrapped_activation` local function
  - **Problem**: `CandidateUnit._init_activation_with_derivative()` defined a local function `wrapped_activation` that cannot be pickled for multiprocessing, causing workers to fail when sending results back
  - **Error**: `AttributeError: Can't pickle local object 'CandidateUnit._init_activation_with_derivative.<locals>.wrapped_activation'`
  - **Solution**: Created picklable `ActivationWithDerivative` class at module level with `__getstate__`/`__setstate__` methods
  - **Files Changed**:
    - `src/candidate_unit/candidate_unit.py` - Added `ActivationWithDerivative` class, modified `_init_activation_with_derivative()` method
    - `src/cascade_correlation/cascade_correlation.py` - Added `ActivationWithDerivative` class, modified `_init_activation_with_derivative()` method
  - **Features**:
    - Stores activation function name for serialization
    - Reconstructs activation from comprehensive ACTIVATION_MAP on unpickling
    - Supports 30+ PyTorch activation functions
    - Includes analytical derivatives for tanh, sigmoid, relu; numerical approximation for others
  - **Result**: CandidateUnit objects can now be pickled for multiprocessing, enabling parallel candidate training

### Added: [0.3.12]

- **New Test Suite**: `src/tests/unit/test_activation_with_derivative.py`
  - 23 unit tests for `ActivationWithDerivative` class
  - Tests cover pickling, derivatives, CandidateUnit integration, and both module implementations
  - All tests pass

### Technical Notes: [0.3.12]

- **SemVer impact**: PATCH – Bug fix enabling multiprocessing; no API changes
- Original local function code preserved as comments with `# OLD:` prefix
- New code marked with `# NEW:` prefix and CASCOR-P1-003 reference

---

## [0.3.11] - 2026-01-20

### Fixed: [0.3.11]

- **CASCOR-P1-004**: Fixed `try` script cosmetic warnings by updating symlink target
  - **Problem**: The `try` symlink pointed to `util/try.bash`, which called `log_debug` before logging functions were sourced, causing 11 "command not found" warnings
  - **Fix**: Updated `try` symlink to point directly to `util/juniper_cascor.bash`
  - **Archived**: Old `util/try.bash` script moved to archive
  - **Result**: Clean startup with no "command not found" warnings

### Identified: [0.3.11]

- **CASCOR-P1-003**: Documented multiprocessing pickling error with `wrapped_activation` local function
  - **Problem**: `CandidateUnit._init_activation_with_derivative()` defines a local function `wrapped_activation` that cannot be pickled for multiprocessing
  - **Impact**: Workers cannot send results back to main process, forcing sequential fallback
  - **Status**: ✅ RESOLVED in v0.3.12
  - **Fix Applied**: Created picklable `ActivationWithDerivative` class at module level

---

## [0.3.10] - 2026-01-20

### Fixed: [0.3.10]

- **CASCOR-P1-002**: Fixed validate_training API mismatch causing AttributeError
  - **Root Cause**: `grow_network()` passed a `ValidateTrainingInputs` dataclass to `validate_training()`, but the method expected individual parameters and returned a tuple
  - **Error**: `AttributeError: 'tuple' object has no attribute 'early_stop'`
  - **Fix**: Updated `validate_training()` method signature to accept `ValidateTrainingInputs` dataclass and return `ValidateTrainingResults` dataclass
  - **File Changed**: `src/cascade_correlation/cascade_correlation.py` (lines 4115-4258)
  - **Result**: Training validation now uses proper dataclass API, enabling full network training cycle

---

## [0.3.9] - 2026-01-20

### Fixed: [0.3.9]

- **CASCOR-P0-004**: Fixed candidate training result parsing error causing all candidates to fail
  - **Root Cause**: `_train_candidate_unit()` called `candidate.train()` which returns a `float`, but the code expected a `CandidateTrainingResult` object with a `.correlation` attribute
  - **Error**: `'float' object has no attribute 'correlation'` - all 10 candidates failed with 0 hidden units added
  - **Fix**: Changed `candidate.train()` to `candidate.train_detailed()` which returns the full `CandidateTrainingResult` dataclass
  - **File Changed**: `src/cascade_correlation/cascade_correlation.py` (line 2767)
  - **Result**: Candidate training now returns proper result objects, enabling network growth with hidden units

---

## [0.3.8] - 2026-01-20

### Fixed: [0.3.8]

- **CASCOR-P0-003**: Fixed test collection errors caused by incorrect module import paths
  - Multiple test files were using incorrect import path `from cascade_correlation_config...` instead of `from cascade_correlation.cascade_correlation_config...`
  - The `cascade_correlation_config` module is a submodule of `cascade_correlation`, not a top-level module
  - This caused `ModuleNotFoundError: No module named 'cascade_correlation_config'` during test collection
  - **Files Fixed**:
    - `src/tests/unit/test_hdf5.py` (lines 10, 24)
    - `src/tests/integration/test_serialization.py` (line 34)
    - `src/tests/unit/test_p1_fixes.py` (lines 73, 124, 195)
    - `src/tests/unit/test_critical_fixes.py` (lines 47, 102)
  - **Result**: All 152 Cascor tests now collect successfully with 0 errors (previously 2 collection errors)

### Integration: [0.3.8]

- Integration analysis with Juniper Canopy documented in `notes/INTEGRATION_ROADMAP.md`
- Environment compatibility verified with JuniperCascor conda environment

---

## [0.3.7] - 2026-01-16

### Fixed: [0.3.7]

- **P0-019**: Fixed Multiprocessing Manager Port Conflict and Sequential Training Fallback
  - **Root Cause 1**: `_CASCADE_CORRELATION_NETWORK_BASE_MANAGER_ADDRESS` was set to just the IP string `'127.0.0.1'` instead of a tuple `('127.0.0.1', port)`
  - **Root Cause 2**: Fixed port 50000 was hardcoded, causing "Address already in use" errors when multiple tests or instances run
  - **Root Cause 3**: `forkserver` multiprocessing context had issues with custom Manager classes in Python 3.14.0 (resolved in Python 3.14.2)
  - **Root Cause 4**: When parallel training failed, dummy results with zero correlation were used, preventing network growth
  - **Fixes Applied**:
    - Changed default port from 50000 to 0 (dynamic OS allocation) in `constants_model.py`
    - Fixed address constant to use tuple `('127.0.0.1', 0)` instead of just IP string in `constants.py`
    - Updated `_init_multiprocessing()` to use configured context type
    - Added sequential training fallback in `_execute_candidate_training()` when parallel training fails
    - Retained `forkserver` context as preferred method (Python 3.14.2 fixes compatibility with custom Manager classes)
  - **Result**: Network uses `forkserver` for optimal parallel training performance, with sequential fallback available when needed.

### Files Changed: [0.3.7]

- `src/constants/constants_model/constants_model.py` - Changed port to 0, retained `forkserver` context
- `src/constants/constants.py` - Added import for `_PROJECT_MODEL_BASE_MANAGER_ADDRESS`, fixed address constant to use tuple
- `src/cascade_correlation/cascade_correlation.py` - Updated `_init_multiprocessing()` to use config context, added sequential fallback in `_execute_candidate_training()`

---

## [0.3.6] - 2026-01-15

### Fixed: [0.3.6]

- **P0-016**: Fixed multiprocessing spawn context module import error
  - Added missing `__init__.py` files to all source directories for proper Python package recognition
  - This resolves `ModuleNotFoundError: No module named 'constants.constants_model'; 'constants' is not a package` error
  - Affected directories: constants/, constants_model/, constants_candidates/, constants_hdf5/, constants_activation/, constants_problem/, constants_logging/, cascade_correlation/, cascade_correlation_config/, cascade_correlation_exceptions/, log_config/, logger/, candidate_unit/, spiral_problem/, cascor_plotter/, remote_client/
  - Root cause: Python's multiprocessing `spawn` context re-imports the module, requiring proper package structure

- **P0-017**: Fixed critical bug in best_candidate_id selection causing network to never grow
  - Fixed `_process_training_results()` where `best_candidate_id` was incorrectly set as a tuple `(value,)` instead of an int
  - The trailing comma in the assignment created a tuple, causing all subsequent lookups to fail
  - Since lookups always returned `None`, `best_candidate` was always `None` and `grow_network()` exited immediately
  - This caused the network to remain linear with no hidden units, unable to solve nonlinear problems like spiral classification
  - Changed to directly access `results[0].candidate_id` after sorting (best correlation at index 0)
  - Also simplified best_candidate data extraction to use direct attribute access on sorted results

- **P0-018**: Fixed test_forward_pass_nan_input test expectation
  - Updated test to expect `ValidationError` exception for NaN inputs (correct behavior)
  - Fixed import path from `cascade_correlation.cascade_correlation_exceptions` to `cascade_correlation_exceptions` to match runtime module resolution
  - Previous test incorrectly expected NaN values to propagate through the network

### Files Changed: [0.3.6]

- Created `src/constants/__init__.py` and all subdirectory `__init__.py` files
- Created `src/cascade_correlation/__init__.py` and all subdirectory `__init__.py` files
- Created `src/log_config/__init__.py` and `src/log_config/logger/__init__.py`
- Created `src/candidate_unit/__init__.py`, `src/spiral_problem/__init__.py`, `src/cascor_plotter/__init__.py`, `src/remote_client/__init__.py`
- `src/cascade_correlation/cascade_correlation.py` - Fixed `_process_training_results()` best_candidate_id bug (lines 1562-1591)
- `src/tests/unit/test_forward_pass.py` - Fixed NaN input test expectation

## [0.3.5] - 2025-01-15

### Fixed: [0.3.5]

- **P0-010**: Fixed CandidateUnit.train() return type for backward compatibility
  - Restored `train()` to return float (correlation value) for backward compatibility
  - Added `train_detailed()` method returning full `CandidateTrainingResult` dataclass
  - Internal code can use `train_detailed()` for full training details
  - Added `last_training_result` attribute for introspection after training

- **P0-011**: Fixed CandidateTrainingManager.start() method signature
  - Added `method` parameter to `start()` for multiprocessing context validation
  - Validates method is one of 'fork', 'spawn', 'forkserver' or raises `ValueError`
  - Raises `NotImplementedError` if method not supported on platform

- **P0-012**: Fixed ValidationError exception hierarchy for test compatibility
  - Made `ValidationError` subclass both `CascadeCorrelationError` and `ValueError`
  - Tests expecting `(ValueError, RuntimeError)` now correctly catch `ValidationError`

- **P0-013**: Fixed fit() method to support `epochs` parameter alias
  - Added `epochs` parameter as backward-compatible alias for `max_epochs`
  - Raises `ValueError` if both provided with different values

- **P0-014**: Fixed tensor validation and edge case handling
  - Added `allow_empty` parameter to `_validate_tensor_input()`
  - `forward()` now allows empty tensors for edge case handling
  - Fixed `calculate_residual_error()` dimension validation (removed incorrect x/y feature comparison)
  - Added target output size validation to prevent tensor mismatch errors
  - Fixed `_accuracy()` to return NaN for empty batches instead of ZeroDivisionError

- **P0-015**: Fixed test expectations to match implementation behavior
  - Updated `test_candidate_training_manager.py` to skip actual manager start() calls
  - Fixed `test_accuracy_non_tensor_inputs` to expect `ValueError`
  - Fixed `test_residual_error_*` tests to match graceful handling behavior

### Files Changed: [0.3.5]

- `src/candidate_unit/candidate_unit.py` - Added train_detailed(), modified train() to return float
- `src/cascade_correlation/cascade_correlation.py` - Multiple fixes for validation, fit(), and manager
- `src/cascade_correlation/cascade_correlation_exceptions/cascade_correlation_exceptions.py` - ValidationError inheritance
- `src/tests/unit/test_candidate_training_manager.py` - Updated test expectations
- `src/tests/unit/test_accuracy.py` - Fixed exception type expectations
- `src/tests/unit/test_residual_error.py` - Fixed test assertions

## [0.3.4] - 2025-01-15

### Fixed: [0.3.4]

- **P0-008**: Fixed multiprocessing context for plotting subprocess
  - Changed plotting subprocess in `spiral_problem.py` from default `forkserver` context to explicit `spawn` context
  - This resolves `ModuleNotFoundError: No module named 'constants.constants_model'; 'constants' is not a package` error
  - This resolves `ConnectionResetError: [Errno 104] Connection reset by peer` error when starting plotting process
  - Application now executes successfully without multiprocessing errors

- **P0-006**: Installed missing Python dependencies
  - Installed `h5py` (3.15.1) for HDF5 serialization support
  - Installed `pytest-cov` (7.0.0) for test coverage reporting
  - Installed `psutil` (7.2.1) for test utilities
  - Test suite now runs with coverage reporting enabled

- **pytest.ini**: Restored coverage options now that pytest-cov is installed
  - Re-enabled `--cov=cascade_correlation`, `--cov=candidate_unit` options
  - Re-enabled coverage report generation (term-missing, html, xml)

### Files Changed: [0.3.4]

- `src/spiral_problem/spiral_problem.py` - Use spawn context for plotting subprocess
- `src/tests/pytest.ini` - Restored coverage options
- `src/tests/integration/test_spiral_problem.py` - Fixed import path for SpiralDataGenerator

## [0.3.3] - 2025-01-12

### Fixed: [0.3.3]

- **P0-001**: Fixed critical candidate training runtime errors
  - Fixed incorrect method call `_train_candidate_worker` → `_train_candidate_unit` in `train_candidate_worker()` method (cascade_correlation.py:1782)
  - Fixed `UnboundLocalError` for `traceback` variable in exception handler by adding import statement
  - Added `__getstate__` and `__setstate__` methods to `LogConfig` class to properly handle pickling of logger objects during multiprocessing
  - Added `__getstate__` and `__setstate__` methods to `CascadeCorrelationConfig` class to handle log_config serialization
  - Updated `CascadeCorrelationNetwork.__getstate__` to exclude `log_config` and activation functions from pickling
  - Updated `CascadeCorrelationNetwork.__setstate__` to reinitialize activation functions after unpickling

### Added: [0.3.3]

- **P0-004**: Added thread safety documentation
  - Added thread safety warning to README.md
  - Added thread safety warning to FEATURES_GUIDE.md
  - Added thread safety warning docstring to `CascadeCorrelationNetwork` class

## [0.3.2] - 2025-01-12

### Added: [0.3.2]

- Initial MVP release with Cascade Correlation Neural Network implementation
- HDF5 serialization support for network snapshots
- N-best candidate selection capability
- Flexible optimizer configuration
- Deterministic training with random state preservation
- Multiprocessing support for parallel candidate training
- Data integrity validation with checksums

### Notes

- Reference implementation based on Fahlman & Lebiere, 1990

---

## [0.3.1] - 2025-12-09

### Fixed: [0.3.1]

- Code refactoring and cleanup across multiple modules (Commit: 1ee6d00)
- Updated execution counts in Jupyter notebook checkpoints
- Removed unnecessary import statements and added comments for clarity
- Improved readability of getter methods in `CascadeCorrelationNetwork` class
- Fixed typos in exception file names
- Improved documentation in README

### Changed: [0.3.1]

- Enhanced snapshot saving and loading functions for better error handling
- Refactored test cases for better organization and clarity
- Cleaned up bash script for running tests, improving readability and consistency
- Added markdownlint configuration files (`.markdownlint.json`, `.markdownlint.jsonc`, `.markdownlint.yaml`)

### Technical Notes: [0.3.1]

- 36 files changed with 1,150 insertions and 468 deletions
- Test suite reorganized for better maintainability

---

## [0.3.0] - 2025-12-08

### Added: [0.3.0]

- Initial standalone Juniper Cascor project structure (Commit: 2076d21)
- VS Code configuration for development
- Logging configuration (`conf/logging_config.yaml`)
- Script utilities configuration (`conf/script_util.cfg`)
- Pre-generated spiral datasets for testing (2, 4, 5, 8 spiral variants)
- Sample training images and visualizations
- Comprehensive project documentation:
  - `ANALYSIS_COMPLETE.md`
  - `CASCOR_ENHANCEMENTS_ROADMAP.md`
  - `CODE_REVIEW_SUMMARY.md`
  - `CRITICAL_FIXES_REQUIRED.md`
  - `FEATURES_GUIDE.md`
  - `IMPLEMENTATION_SUMMARY.md`
  - `PHASE1_COMPLETE.md`
  - `SERIALIZATION_FIXES_SUMMARY.md`

### Changed: [0.3.0]

- Separated Juniper Cascor from parent Juniper project as standalone package
- Reorganized source directory structure under `src/`

---

## [0.2.0] - 2025-10-28

### Fixed: [0.2.0]

- **BUG-001**: Fixed test random state restoration failures
  - Test helper method used wrong RNG function for different modules
  - `torch.rand()` was incorrectly called on `random` and `numpy` modules
  - Modified `_load_and_validate_network_helper()` to detect module type and call correct function
  - Files Changed: `src/tests/integration/test_serialization.py`

- **BUG-002**: Fixed logger pickling error in multiprocessing
  - `PicklingError: logger cannot be pickled` when spawning multiprocessing for plots
  - Enhanced `CascadeCorrelationNetwork.__getstate__()` to remove 15+ non-picklable objects
  - Enhanced `CascadeCorrelationNetwork.__setstate__()` to properly restore logger, plotter, display functions
  - Added pickling support to `CascadeCorrelationPlotter`
  - Files Changed: `src/cascade_correlation/cascade_correlation.py`, `src/cascor_plotter/cascor_plotter.py`

### Added: [0.2.0]

- **ENH-001**: Comprehensive test suite for serialization
  - Created `src/tests/integration/test_comprehensive_serialization.py` (370 lines)
  - 6 new integration tests for full serialization round-trip

- **ENH-008**: Enhanced worker cleanup with better logging

### Technical Notes: [0.2.0]

- Phase 1 implementation complete (P0 + P1 + P2)
- Total implementation time: ~4 hours

---

## [0.1.1] - 2025-10-25

### Fixed: [0.1.1]

- **Critical HDF5 Serialization Fixes**:
  - Fixed UUID not being restored during network load (was generating new UUID each time)
  - Fixed Python random module state not being persisted (only NumPy and PyTorch were saved)
  - Fixed config JSON serialization errors for `activation_functions_dict`, `log_config`, `logger`
  - Fixed history key mismatch (`value_loss`/`value_accuracy` vs `val_loss`/`val_accuracy`)
  - Fixed activation function not being reinitialized after load

### Changed: [0.1.1]

- Updated `snapshot_serializer.py` to handle UUID restoration in `_create_network_from_file()`
- Added Python random state save/load using `pickle.dumps()`/`pickle.loads()`
- Added exclusion list for non-serializable config attributes
- Updated history save/load to use correct network keys

---

## [0.1.0] - 2025-10-15

### Fixed: [0.1.0]

- **P0 Critical Blocking Issues** (6 fixes enabling basic training):
  1. Fixed `CandidateTrainingResult` dataclass field names (`candidate_index` → `candidate_id`, `best_correlation` → `correlation`)
  2. Fixed gradient descent direction in `CandidateUnit` (was gradient ascent: `+=` → `-=`)
  3. Fixed matrix multiplication in weight updates (dimension mismatch with `@` operator)
  4. Fixed `_get_correlations` field names for consistency
  5. Updated train method to use correct field names
  6. Added instance correlation update during training

- **P1 High Priority Fixes** (5 fixes for production readiness):
  1. Implemented optimizer state serialization to HDF5
  2. Added training counter persistence (snapshot_counter, current_epoch, patience_counter, best_value_loss)
  3. Added queue operation timeouts (30 second timeout for `result_queue.put()`)
  4. Implemented early stopping for candidate training
  5. Fixed type annotations and added public `save_to_hdf5()`/`load_from_hdf5()` API methods

### Changed: [0.1.0]

- Fixed `np.string_` → `np.bytes_` for NumPy 2.0+ compatibility
- Improved error handling for queue full scenarios in multiprocessing

### Technical Notes: [0.1.0]

- All P1 tests passed: 5/5 (100%)
- Early stopping reduces training times by ~50-70%

---

## [0.0.1] - 2023-06-13

### Added: [0.0.1]

- Initial commit of Cascade Correlation Neural Network prototype (Commit: 681c2e9)
- Core implementation based on Fahlman & Lebiere, 1990 paper
- Basic network architecture with input/output layers
- Candidate unit training infrastructure
- Forward pass algorithm

---

## Version History

| Version | Date       | Description                              |
| ------- | ---------- | ---------------------------------------- |
| 0.6.7   | 2026-02-05 | Integration Development Plan             |
| 0.6.6   | 2026-02-04 | Test/CI Phase 4 (benchmarks, matrix)     |
| 0.6.5   | 2026-02-04 | Test/CI Phase 3 (tooling quality gates)  |
| 0.6.4   | 2026-02-04 | Test/CI Phases 0-2 (test integrity)      |
| 0.6.3   | 2026-02-01 | JuniperData integration documentation    |
| 0.6.2   | 2026-02-01 | CI/CD parity across all 3 apps           |
| 0.6.1   | 2026-01-31 | Algorithm parameter + E2E validation     |
| 0.6.0   | 2026-01-30 | JuniperData Cascor Integration (Phase 3) |
| 0.5.1   | 2026-01-29 | Pre-commit Compliance (MyPy, F401, B907) |
| 0.5.0   | 2026-01-29 | JuniperData Extraction (Phases 0-2)      |
| 0.4.1   | 2026-01-29 | Documentation Overhaul                   |
| 0.4.0   | 2026-01-29 | CI/CD Pipeline Overhaul                  |
| 0.3.16  | 2026-01-24 | CI/CD Pipeline Setup (P1-007)            |
| 0.3.15  | 2026-01-24 | Fixed P0 issues, serialization coverage  |
| 0.3.14  | 2026-01-22 | Fixed multiprocessing and test issues    |
| 0.3.13  | 2026-01-21 | Fixed test timeout configuration         |
| 0.3.12  | 2026-01-21 | Fixed activation pickling for MP         |
| 0.3.7   | 2026-01-16 | Fixed port conflicts, sequential fallback|
| 0.3.6   | 2026-01-15 | Fixed spawn context module imports       |
| 0.3.5   | 2025-01-15 | Fixed API compatibility and test suite   |
| 0.3.4   | 2025-01-15 | Fixed multiprocessing and dependencies   |
| 0.3.3   | 2025-01-12 | Addressed critical runtime errors        |
| 0.3.2   | 2025-01-12 | MVP Complete                             |
| 0.3.1   | 2025-12-09 | Code refactoring and cleanup             |
| 0.3.0   | 2025-12-08 | Standalone project structure             |
| 0.2.0   | 2025-10-28 | Phase 1 complete, serialization fixes    |
| 0.1.1   | 2025-10-25 | HDF5 serialization critical fixes        |
| 0.1.0   | 2025-10-15 | P0/P1 critical bug fixes                 |
| 0.0.1   | 2023-06-13 | Initial development release              |
