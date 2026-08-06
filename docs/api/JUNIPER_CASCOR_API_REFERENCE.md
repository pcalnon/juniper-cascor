# juniper-cascor API Reference

**Application**: juniper-cascor
**Version covered**: 0.4.0
**Source repo**: `/home/pcalnon/Development/python/Juniper/juniper-cascor/`
**Compiled**: 2026-05-08
**Audience**: Engineers integrating with the juniper-cascor REST + WebSocket surface (juniper-canopy, juniper-cascor-client, juniper-cascor-worker, ad-hoc tooling).

---

## Table of Contents

- [Conventions](#conventions)
  - [Base URL & versioning](#base-url--versioning)
  - [Authentication](#authentication)
  - [Response envelope](#response-envelope)
  - [Error envelope](#error-envelope)
  - [Common HTTP status codes](#common-http-status-codes)
  - [Common WebSocket close codes](#common-websocket-close-codes)
  - [Middleware stack](#middleware-stack)
- [Health & readiness](#health--readiness)
  - [GET `/v1/health`](#get-v1health)
  - [GET `/v1/health/live`](#get-v1healthlive)
  - [GET `/v1/health/ready`](#get-v1healthready)
- [Network management](#network-management)
  - [POST `/v1/network`](#post-v1network)
  - [GET `/v1/network`](#get-v1network)
  - [DELETE `/v1/network`](#delete-v1network)
  - [GET `/v1/network/topology`](#get-v1networktopology)
  - [GET `/v1/network/stats`](#get-v1networkstats)
  - [PATCH `/v1/network/weights`](#patch-v1networkweights)
  - [POST `/v1/network/hidden-units`](#post-v1networkhidden-units)
  - [DELETE `/v1/network/hidden-units/{idx}`](#delete-v1networkhidden-unitsidx)
- [Training control](#training-control)
  - [POST `/v1/training/start`](#post-v1trainingstart)
  - [POST `/v1/training/stop`](#post-v1trainingstop)
  - [POST `/v1/training/pause`](#post-v1trainingpause)
  - [POST `/v1/training/resume`](#post-v1trainingresume)
  - [POST `/v1/training/reset`](#post-v1trainingreset)
  - [POST `/v1/training/metrics/clear`](#post-v1trainingmetricsclear)
  - [POST `/v1/training/metrics/clear/undo`](#post-v1trainingmetricsclearundo)
  - [GET `/v1/training/status`](#get-v1trainingstatus)
  - [GET `/v1/training/params`](#get-v1trainingparams)
  - [PATCH `/v1/training/params`](#patch-v1trainingparams)
- [Metrics](#metrics)
  - [C7 scalar evaluation metrics](#c7-scalar-evaluation-metrics)
  - [GET `/v1/metrics`](#get-v1metrics)
  - [GET `/v1/metrics/history`](#get-v1metricshistory)
  - [GET `/v1/metrics/transport`](#get-v1metricstransport)
- [Dataset](#dataset)
  - [GET `/v1/dataset`](#get-v1dataset)
  - [GET `/v1/dataset/data`](#get-v1datasetdata)
- [Decision boundary](#decision-boundary)
  - [GET `/v1/decision-boundary`](#get-v1decision-boundary)
- [Snapshots](#snapshots)
  - [POST `/v1/snapshots`](#post-v1snapshots)
  - [GET `/v1/snapshots`](#get-v1snapshots)
  - [GET `/v1/snapshots/{snapshot_id}`](#get-v1snapshotssnapshot_id)
  - [POST `/v1/snapshots/{snapshot_id}/restore`](#post-v1snapshotssnapshot_idrestore)
  - [POST `/v1/snapshots/{snapshot_id}/retrain`](#post-v1snapshotssnapshot_idretrain)
  - [POST `/v1/snapshots/{snapshot_id}/resume`](#post-v1snapshotssnapshot_idresume)
  - [POST `/v1/snapshots/{snapshot_id}/replay`](#post-v1snapshotssnapshot_idreplay)
  - [POST `/v1/snapshots/{snapshot_id}/replay/control`](#post-v1snapshotssnapshot_idreplaycontrol)
- [Workers](#workers)
  - [GET `/v1/workers`](#get-v1workers)
  - [GET `/v1/workers/stats`](#get-v1workersstats)
  - [GET `/v1/workers/{worker_id}`](#get-v1workersworker_id)
- [WebSocket endpoints](#websocket-endpoints)
  - [WS `/ws/training`](#ws-wstraining)
  - [WS `/ws/control`](#ws-wscontrol)
  - [WS `/ws/v1/workers`](#ws-wsv1workers)
- [State-modifying endpoints summary](#state-modifying-endpoints-summary)

---

## Conventions

### Base URL & versioning

The FastAPI app is defined at `src/api/app.py:399`. All REST routers are mounted under the `/v1` prefix (`src/api/app.py:461-468`); WebSocket endpoints are mounted directly on the app (`src/api/app.py:471-473`). Default service ports per the ecosystem are `8201` (host) → `8200` (container).

### Startup bind guard

The service defaults to a loopback bind (`JUNIPER_CASCOR_HOST=127.0.0.1`, `JUNIPER_CASCOR_PORT=8200`). During lifespan startup, `enforce_bind_attestation_guard()` refuses non-loopback binds such as `0.0.0.0`, `::`, or non-local hostnames unless at least one bind attestation is set: `JUNIPER_CASCOR_LOOPBACK_PUBLISH_ATTESTED=true` or `JUNIPER_CASCOR_AUTH_PROXY_ATTESTED=true`.

Treat each flag as an operational attestation: `JUNIPER_CASCOR_LOOPBACK_PUBLISH_ATTESTED=true` asserts the port is reachable only via a loopback-only host publish; `JUNIPER_CASCOR_AUTH_PROXY_ATTESTED=true` asserts a fronting authenticating reverse proxy terminates access. Without at least one, startup raises `NonLoopbackBindError` before the server begins accepting REST or WebSocket traffic.

### Authentication

Optional at the request layer; loud at boot via SEC-F01.

- REST: `X-API-Key` header validated by `APIKeyAuth` middleware (`src/api/security.py`). When `settings.api_keys` is empty/blank, auth is disabled (dev mode) and protected routes are served open.
- WebSocket: same `X-API-Key` header, validated in `ws_authenticate()` (`src/api/websocket/manager.py`). On failure the socket is closed with code `4001`.
- Boot posture: the lifespan calls `juniper_service_core.enforce_auth_posture(...)` immediately after the bind guard (`src/api/app.py`). `JUNIPER_CASCOR_REQUIRE_AUTH` (default `false`) selects WARNING-and-continue vs refuse-with-`AuthPostureError`. Deployments that provision secrets (composed juniper-deploy) should set it `true`. Bypass: `JUNIPER_SKIP_AUTH_POSTURE_CHECK=1` (logged loudly).
- Docker secrets: `JUNIPER_CASCOR_API_KEYS_FILE` is read by `api.secrets.get_secret()`. An existing empty/whitespace-only file returns `""` with **no** fallback to the plain env var. In the usual compose `_FILE`-only pattern that leaves `settings.api_keys` unset (HO-2 empty-placeholder class) unless `REQUIRE_AUTH=true` fails the boot.

Rate limiting is also optional; per-IP REST limiter defaults to 100 req/min, and the worker WebSocket has its own per-IP connection rate limiter. WebSocket admission also has a stack-global cap across all WS endpoints (`JUNIPER_CASCOR_WS_MAX_CONNECTIONS_GLOBAL`, default 200). `/ws/control` adds a per-identity cap (`JUNIPER_CASCOR_WS_MAX_CONNECTIONS_PER_IDENTITY`, default 5) keyed on a non-reversible digest of the presented `X-API-Key`.

### Startup bind guard

The service fails closed at startup when `JUNIPER_CASCOR_HOST` is a non-loopback bind target (for example `0.0.0.0`) unless `JUNIPER_CASCOR_LOOPBACK_PUBLISH_ATTESTED=true` or `JUNIPER_CASCOR_AUTH_PROXY_ATTESTED=true`. Loopback binds (`127.0.0.0/8`, `::1`, `localhost`, IPv4-mapped IPv6 loopback) always start. The guard runs from the FastAPI lifespan before the server begins accepting connections and raises `NonLoopbackBindError` on unsafe non-loopback startup (no warning-only mode).

Use `JUNIPER_CASCOR_LOOPBACK_PUBLISH_ATTESTED=true` when a loopback host-publish fronts the port, or `JUNIPER_CASCOR_AUTH_PROXY_ATTESTED=true` when an authenticating reverse proxy does. Container runs normally bind `0.0.0.0` inside the container and publish the host port on loopback (so they attest the loopback-publish flag).

### Response envelope

Successful REST responses are wrapped by `success_response()` (`src/api/models/common.py:85-97`):

```json
{
  "status": "success",
  "data": { /* endpoint-specific payload */ },
  "meta": {
    "timestamp": 1714000000.123,
    "version": "0.4.0"
  }
}
```

The wrapper recursively coerces NumPy scalars to Python natives via `coerce_native_scalars()` (`src/api/models/common.py:11-43`). The two health endpoints (`/v1/health`, `/v1/health/ready`) intentionally bypass the envelope for backward compatibility with existing health probes.

### Error envelope

The global handlers in `src/api/app.py:480-494` produce:

```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters"
  },
  "meta": { "timestamp": 1714000000.123, "version": "0.4.0" }
}
```

`HTTPException`s raised by route handlers pass through with their configured `status_code` and `detail`. Pydantic body validation produces FastAPI's standard 422 error shape.

### Common HTTP status codes

| Code | Meaning in juniper-cascor                                                                                            |
|------|----------------------------------------------------------------------------------------------------------------------|
| 200  | Success                                                                                                              |
| 400  | Bad request — invalid `snapshot_id` format, bad shape, invalid replay action params                                  |
| 404  | Not found — no network created, snapshot/worker not found, hidden-unit index out of range, dataset not loaded        |
| 409  | Conflict — invalid FSM state, training already active, network at `max_hidden_units`, stale replay session id in URL |
| 422  | Unprocessable Entity — Pydantic validation failure on request body, NaN/Inf weights, unknown activation name         |
| 500  | Internal server error — topology / decision-boundary extraction failed, unhandled exception                          |
| 503  | Service unavailable — lifecycle / registry / WebSocket manager not bound (startup not complete or shutting down)     |

### Common WebSocket close codes

| Code | Meaning                                                                                                                                                                                             |
|------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1000 | Normal closure                                                                                                                                                                                      |
| 1011 | Heartbeat timeout — the client sent no pong (nor any other frame) within the pong window. Reason text: `Heartbeat timeout: no pong or traffic within <N>s`                                          |
| 1006 | Abnormal closure — reported client-side only when the connection dies without a close frame (RFC 6455 §7.4.1 forbids sending 1006 on the wire; pre-C3 heartbeat closes attempted it and never delivered a close frame) |
| 1013 | WebSocket admission cap exceeded                                                                                                                                                                    |
| 4001 | Authentication required / `X-API-Key` invalid                                                                                                                                                       |
| 4003 | Origin header not permitted (control + worker streams)                                                                                                                                             |
| 4004 | Worker subsystem not initialized                                                                                                                                                                    |
| 1013 | WebSocket connection cap reached                                                                                                                                                                    |
| 4029 | Connection rate limited (worker stream)                                                                                                                                                             |

### Middleware stack

Registered in `src/api/app.py:426-458` (LIFO execution order):

1. CORS (only if origins are configured)
2. `RequestBodyLimitMiddleware`
3. `SecurityHeadersMiddleware`
4. `SecurityMiddleware` (`APIKeyAuth` + `RateLimiter`)
5. `PrometheusMiddleware` (only when `metrics_enabled`)
6. `RequestIdMiddleware` (always adds `X-Request-Id`)

---

## Health & readiness

Defined in `src/api/routes/health.py`. These endpoints intentionally do **not** use the success envelope so existing probes (Kubernetes, Docker, juniper-canopy, juniper-deploy) keep working.

### GET `/v1/health`

**Summary** — Always-on liveness probe.

**Detailed description** — Cheapest possible health check. Returns immediately without consulting any subsystem. Used by older probes and external watchdogs that just need a 200 to confirm the process is up. Implemented at `src/api/routes/health.py:53-61`.

**Syntax**:

```http
GET /v1/health HTTP/1.1
Host: localhost:8201
```

**Example call**:

```bash
curl -s http://localhost:8201/v1/health
```

**State changes** — None.

**Returns** — `200 OK` with body `{"status": "ok", "version": "0.4.0"}`.

**Error handling** — None; the handler has no failure paths.

---

### GET `/v1/health/live`

**Summary** — In-process liveness tick conforming to METRICS-MON R2.1.4.

**Detailed description** — Performs a tiny in-process operation timed against `LIVENESS_TICK_BUDGET_MS`. If the heartbeat is stale or the lifecycle isn't bound, the response is `503` with `status: "unresponsive"`. Used by container orchestrators that want to evict an in-process hang. Implemented at `src/api/routes/health.py:64-100`.

**Syntax**:

```http
GET /v1/health/live HTTP/1.1
```

**Example call**:

```bash
curl -i http://localhost:8201/v1/health/live
```

**State changes** — None.

**Returns**:

```json
{
  "status": "alive",
  "tick": "juniper-cascor",
  "duration_ms": 0
}
```

When degraded the body is `{"status": "unresponsive", "tick": "juniper-cascor", "duration_ms": <int>, "error": "<reason>"}`.

**Error handling** — `503` for stale heartbeat or missing lifecycle.

---

### GET `/v1/health/ready`

**Summary** — Readiness probe with per-dependency status.

**Detailed description** — Walks the registered dependency probes (juniper-data, snapshot store, worker registry, etc.) and returns aggregate readiness plus a per-dep dict. Sets the `X-Juniper-Readiness` response header so reverse proxies can route on it. Implemented at `src/api/routes/health.py:103-163`.

**Syntax**:

```http
GET /v1/health/ready HTTP/1.1
```

**Example call**:

```bash
curl -i http://localhost:8201/v1/health/ready
```

**State changes** — None.

**Returns** — `ReadinessResponse` (re-exported from `juniper_observability`):

```json
{
  "status": "ready",
  "version": "0.4.0",
  "service": "juniper-cascor",
  "dependencies": {
    "juniper_data": {"name": "juniper_data", "status": "healthy", "message": "..."},
    "snapshot_store": {"name": "snapshot_store", "status": "healthy", "message": "..."}
  },
  "details": {}
}
```

**Error handling** — `503` when any required dep is `unhealthy`. Optional deps in `not_configured` state do not flip the aggregate.

---

## Network management

Router defined in `src/api/routes/network.py`, prefix `/v1/network`. All endpoints depend on `lifecycle` from app state; `503` if the lifecycle isn't bound. Network mutations are FSM-gated where noted.

### POST `/v1/network`

**Summary** — Create a new CasCor network.

**Detailed description** — Allocates a fresh CasCor network with the supplied hyperparameters via `lifecycle.create_network()` (`src/api/routes/network.py:27`). Replaces any prior in-memory network. Returns the canonical metadata (input/output sizes, learning rate, hidden-unit cap, current count, and the network UUID). Implemented at `src/api/routes/network.py:22-31`.

**Syntax**:

```http
POST /v1/network HTTP/1.1
Content-Type: application/json

{
  "input_size": 2,
  "output_size": 1,
  "learning_rate": 0.01,
  "candidate_learning_rate": 0.1,
  "max_hidden_units": 50,
  "candidate_pool_size": 8,
  "correlation_threshold": 0.5,
  "patience": 20,
  "candidate_epochs": 50,
  "output_epochs": 10,
  "max_iterations": 100,
  "init_output_weights": "zero",
  "optimizer_type": "Adam",
  "activation_function_name": "Tanh"
}
```

**Body model** — `NetworkCreateRequest` (`src/api/models/network.py`). All fields optional; defaults match the snippet above. `optimizer_type` ∈ `{Adam, AdamW, SGD, RMSprop, NAdam, RAdam, Adamax, Adagrad, Adadelta, Adafactor, ASGD, LBFGS, Rprop, Muon}`. `activation_function_name` ∈ `{Identity, Tanh, Sigmoid, ReLU, LeakyReLU, ELU, SELU, GELU, Softmax, Softplus, Hardtanh, Softshrink, Tanhshrink}`.

**Example call**:

```bash
curl -s -X POST http://localhost:8201/v1/network \
  -H 'Content-Type: application/json' \
  -d '{"input_size": 2, "output_size": 1, "learning_rate": 0.01}'
```

**State changes** — Allocates a new network on the lifecycle; replaces any pre-existing network (legacy data is discarded).

**Returns** — `200` envelope with `data` containing `input_size`, `output_size`, `hidden_units`, `max_hidden_units`, `learning_rate`, `uuid`, plus the full hyperparameter snapshot.

**Error handling**:

| Code | Trigger                                                                           |
|------|-----------------------------------------------------------------------------------|
| 409  | Network already exists in an incompatible FSM state (`HTTPException(409, "...")`) |
| 422  | Pydantic validation (negative sizes, unknown optimizer/activation, etc.)          |
| 503  | `lifecycle` not bound                                                             |

---

### GET `/v1/network`

**Summary** — Read the current network's metadata.

**Detailed description** — Returns the same payload as `POST /v1/network` for the live network. Implemented at `src/api/routes/network.py:34-40`.

**Syntax**:

```http
GET /v1/network HTTP/1.1
```

**Example call**:

```bash
curl -s http://localhost:8201/v1/network
```

**State changes** — None.

**Returns** — Envelope with the network metadata dict.

**Error handling** — `404` if no network has been created; `503` if the lifecycle isn't bound.

---

### DELETE `/v1/network`

**Summary** — Tear down the current network.

**Detailed description** — Invokes `lifecycle.delete_network()` (`src/api/routes/network.py:48`). Frees memory, drops snapshots from RAM, and resets the FSM. Implemented at `src/api/routes/network.py:43-52`.

**Syntax**:

```http
DELETE /v1/network HTTP/1.1
```

**Example call**:

```bash
curl -s -X DELETE http://localhost:8201/v1/network
```

**State changes** — Deallocates the network and resets associated lifecycle state.

**Returns** — `200` envelope with `{"deleted": true}`.

**Error handling** — `409` if the FSM is in a state where deletion is not allowed; `503` if lifecycle is unbound.

---

### GET `/v1/network/topology`

**Summary** — Return the current cascade topology suitable for visualization.

**Detailed description** — Extracts the per-unit weights, biases, activation labels, and cascade input wiring. Used by juniper-canopy's network panel. Implemented at `src/api/routes/network.py:55-64`.

**Syntax**:

```http
GET /v1/network/topology HTTP/1.1
```

**Example call**:

```bash
curl -s http://localhost:8201/v1/network/topology
```

**State changes** — None.

**Returns** — Envelope with topology object.

**Error handling** — `404` if no network; `500` if extraction fails (caught and re-raised as `HTTPException(500, ...)`); `503` if lifecycle unbound.

---

### GET `/v1/network/stats`

**Summary** — Return weight statistics for the current network.

**Detailed description** — Aggregate stats (per-layer mean/std/min/max, parameter counts). Implemented at `src/api/routes/network.py:67-73`.

**Syntax / example**:

```bash
curl -s http://localhost:8201/v1/network/stats
```

**State changes** — None.

**Returns** — Envelope with statistics dict.

**Error handling** — `404` if no network; `503` if lifecycle unbound.

---

### PATCH `/v1/network/weights`

**Summary** — Surgically rewrite output or per-unit weights/bias (CAN-015h-1).

**Detailed description** — FSM-gated to the `Investigating` state. Used by juniper-canopy's investigation tools to nudge weights without retraining. The handler validates exact tensor shape, rejects NaN/Inf, and supports `float32` / `float64` dtypes. Implemented at `src/api/routes/network.py:76-121`.

**Syntax**:

```http
PATCH /v1/network/weights HTTP/1.1
Content-Type: application/json

{
  "target": "output",
  "field": "weights",
  "values": [[0.12, -0.04, 0.31, ...]],
  "hidden_unit_index": null,
  "dtype": "float32"
}
```

**Body model** — `PatchWeightsRequest`. Required: `target` ∈ `{output, hidden_unit}`, `field` ∈ `{weights, bias}`, `values` (list with exact required shape). `hidden_unit_index` is required when `target == "hidden_unit"`.

**Example call**:

```bash
curl -s -X PATCH http://localhost:8201/v1/network/weights \
  -H 'Content-Type: application/json' \
  -d '{"target":"hidden_unit","field":"bias","values":[0.05],"hidden_unit_index":2}'
```

**State changes** — Writes new tensor values into the live network's parameter group. Updates FSM bookkeeping for investigation.

**Returns** — Envelope with the updated network info plus `operation` and `fsm_state`.

**Error handling**:

| Code | Trigger                                        |
|------|------------------------------------------------|
| 400  | Shape mismatch, unknown target/field           |
| 404  | No network or `hidden_unit_index` out of range |
| 409  | FSM not in `Investigating`                     |
| 422  | NaN/Inf in `values`, unknown dtype             |
| 500  | Defensive fallback for an unmapped lifecycle status sentinel |
| 503  | Lifecycle unbound                              |

---

### POST `/v1/network/hidden-units`

**Summary** — Manually append a hidden unit at the cascade tail (CAN-015h-2).

**Detailed description** — FSM-gated to `Investigating`. Initializes output weights to zero so the new unit is benign. Useful for white-box experiments. Implemented at `src/api/routes/network.py:124-176`.

**Syntax**:

```http
POST /v1/network/hidden-units HTTP/1.1
Content-Type: application/json

{
  "weights": [0.1, -0.2, 0.05],
  "bias": 0.0,
  "activation": "Tanh",
  "position": "tail"
}
```

**Body model** — `AddHiddenUnitRequest`. `weights` shape must equal `[input_size + num_existing_hidden_units]`. `position` is `"tail"` only in V1. `activation` defaults to `"Tanh"`.

**Example call**:

```bash
curl -s -X POST http://localhost:8201/v1/network/hidden-units \
  -H 'Content-Type: application/json' \
  -d '{"weights":[0.1,-0.2,0.05],"bias":0.0,"activation":"Tanh"}'
```

**State changes** — Adds a unit to the cascade, expands every downstream unit's input vector, and zero-initializes its output projection.

**Returns** — Envelope with `unit_index`, `num_hidden_units`, `operation`, `fsm_state`, and refreshed network metadata.

**Error handling**:

| Code | Trigger                                                     |
|------|-------------------------------------------------------------|
| 400  | Bad weight shape                                            |
| 404  | No network                                                  |
| 409  | FSM not in `Investigating` or already at `max_hidden_units` |
| 422  | NaN/Inf, unknown activation                                 |
| 500  | Defensive fallback for an unmapped lifecycle status sentinel |
| 503  | Lifecycle unbound                                           |

---

### DELETE `/v1/network/hidden-units/{idx}`

**Summary** — Manually remove the hidden unit at index `idx` (CAN-015h-3).

**Detailed description** — FSM-gated to `Investigating`. Subsequent units shift down; cascade input wiring is rewritten to keep dimensionality consistent. Implemented at `src/api/routes/network.py:179-221`.

**Syntax**:

```http
DELETE /v1/network/hidden-units/3 HTTP/1.1
```

**Example call**:

```bash
curl -s -X DELETE http://localhost:8201/v1/network/hidden-units/3
```

**State changes** — Removes the unit, renumbers downstream units, and rewires every consumer's input slice.

**Returns** — Envelope with `removed_index`, `num_hidden_units`, `operation`, `fsm_state`, and refreshed network metadata.

**Error handling**:

| Code | Trigger                          |
|------|----------------------------------|
| 404  | No network or `idx` out of range |
| 409  | FSM not in `Investigating`       |
| 500  | Defensive fallback for an unmapped lifecycle status sentinel |
| 503  | Lifecycle unbound                |

---

## Training control

Router defined in `src/api/routes/training.py`, prefix `/v1/training`.

### POST `/v1/training/start`

**Summary** — Kick off a training run.

**Detailed description** — Accepts inline data, a generator-based dataset (e.g., `spiral`), or relies on a pre-loaded dataset. Validates `params` against `TrainingParams` (SEC-07: unknown keys produce `422`). Coerces data to `torch.float32` tensors before invoking `lifecycle.start_training()` (`src/api/routes/training.py:68`). Implemented at `src/api/routes/training.py:23-73`.

**Syntax**:

```http
POST /v1/training/start HTTP/1.1
Content-Type: application/json

{
  "epochs": 250,
  "dataset": {"source": "generator", "generator": "spiral", "params": {"n": 200}},
  "inline_data": null,
  "params": {
    "learning_rate": 0.01,
    "patience": 20,
    "max_hidden_units": 50
  }
}
```

**Body model** — `TrainingStartRequest` with sub-models `DatasetSource`, `InlineDataset` (≤100 train + ≤100 val samples, list-of-list-of-float), and `TrainingParams` (all fields optional). Either `inline_data` or `dataset` may be provided; if neither, the loaded dataset is reused.

**`start_fresh` (C5 / Q4 use-case 2, default `false`)** — Retention posture for the run:

- **`false` (default) — continue the current model.** The existing network AND its retained metrics/history are kept, so the run continues training the model as-is (the cross-dataset continual-training use case, Q4 use-case 1). Metrics/history are now RETAINED across run boundaries by default (pre-C5 every run start emptied the metrics buffer); a continuing run appends only its new rows.
- **`true` — clean-launch start.** The current model and all retained metrics/history are DISCARDED before the run, and a vanilla, untrained network is rebuilt from the dataset dims — functionally identical to a fresh stack launch, **EXCEPT that on-disk snapshot artifacts are preserved** (nothing on this path deletes a snapshot). Independent of the snapshot-driven `POST /v1/snapshots/{id}/retrain` and the FSM-level `POST /v1/training/reset`.

Backward compatible: pre-C5 callers omit `start_fresh` and get the retain (continue) path.

**Example call**:

```bash
curl -s -X POST http://localhost:8201/v1/training/start \
  -H 'Content-Type: application/json' \
  -d '{"dataset":{"source":"generator","generator":"spiral","params":{"n":200}},"params":{"learning_rate":0.01}}'
```

**State changes** — Transitions the FSM into `Training` and spins up the training loop. Begins emitting `epoch_end` and `state` events to all `/ws/training` subscribers.

**Returns** — Envelope with the lifecycle's training-start result dict (training id, dataset summary, effective params).

**Error handling**:

| Code | Trigger                                                                  |
|------|--------------------------------------------------------------------------|
| 409  | Cannot start in current FSM state (e.g., already running, replay active) |
| 422  | Body validation, unknown `params` key, NaN/Inf in `inline_data`          |
| 503  | Lifecycle unbound                                                        |

---

### POST `/v1/training/stop`

**Summary** — Halt the currently running training.

**Detailed description** — Idempotent; calling on an idle FSM returns gracefully. Implemented at `src/api/routes/training.py:75-80`.

**Syntax / example**:

```bash
curl -s -X POST http://localhost:8201/v1/training/stop
```

**State changes** — Transitions FSM out of `Training`/`Paused` to idle; cancels the training loop coroutine; flushes the WebSocket replay buffer's terminal `state` event.

**Returns** — Envelope with `{"stopped": true, "epoch": <int>, ...}` from lifecycle.

**Error handling** — `503` if lifecycle unbound. (No `409`; stop is permissive.)

---

### POST `/v1/training/pause`

**Summary** — Pause an active training loop.

**Detailed description** — Implemented at `src/api/routes/training.py:83-92`.

**Syntax / example**:

```bash
curl -s -X POST http://localhost:8201/v1/training/pause
```

**State changes** — FSM transitions `Training → Paused`. Loop coroutine awaits a resume signal; metrics history is preserved.

**Returns** — Envelope with lifecycle pause result.

**Error handling** — `409` if not currently `Training`; `503` if lifecycle unbound.

---

### POST `/v1/training/resume`

**Summary** — Resume from `Paused`.

**Detailed description** — Implemented at `src/api/routes/training.py:95-104`.

**Syntax / example**:

```bash
curl -s -X POST http://localhost:8201/v1/training/resume
```

**State changes** — FSM `Paused → Training`; loop coroutine continues.

**Returns** — Envelope with lifecycle resume result.

**Error handling** — `409` if not currently `Paused`; `503` if lifecycle unbound.

---

### POST `/v1/training/reset`

**Summary** — Reset the training state (clears history & counters but preserves the network).

**Detailed description** — Implemented at `src/api/routes/training.py:107-112`.

**Syntax / example**:

```bash
curl -s -X POST http://localhost:8201/v1/training/reset
```

**State changes** — Clears training history arrays, epoch counters, auto-snap-best ratchet; transitions FSM to idle. This is the heavy reset (FSM + counters + metrics). For clearing metrics/history ONLY — with an undo — use `POST /v1/training/metrics/clear` (C5).

**Returns** — Envelope with reset result.

**Error handling** — `503` if lifecycle unbound.

---

### POST `/v1/training/metrics/clear`

**Summary** — Clear the retained training metrics/history, with undo (C5 / Q4 use-case 1).

**Detailed description** — Metrics/history are now retained across run boundaries by default (C5 / Q4), so this is the explicit control that empties the metrics/history buffer between runs. The clear stashes an in-memory undo snapshot and is reversible via `POST /v1/training/metrics/clear/undo` at any point **until the next training run starts** (starting a run finalizes the clear and drops the snapshot). Unlike `POST /v1/training/reset`, this touches metrics/history only — it does not reset the FSM, counters, or the model. The undo snapshot is bounded by the metrics buffer size (10000 rows), so a pending undo costs at most one extra buffer's worth of memory. Implemented at `src/api/routes/training.py`.

**Syntax / example**:

```bash
curl -s -X POST http://localhost:8201/v1/training/metrics/clear
```

**State changes** — Empties the metrics/history buffer; stashes the removed rows as the undo snapshot. `GET /v1/training/status` reports `metrics_clear_undo_available: true` afterwards.

**Returns** — Envelope with `{"status": "cleared", "cleared_count": <int>, "undo_available": true}`.

**Error handling** — `503` if lifecycle unbound.

---

### POST `/v1/training/metrics/clear/undo`

**Summary** — Undo the most recent metrics/history clear (C5 / Q4 use-case 1 fallback).

**Detailed description** — Restores the rows removed by the last `POST /v1/training/metrics/clear`. Valid only until the next training run starts; returns `409` when there is no clear to undo (nothing cleared, or a run has started since and finalized the clear). Implemented at `src/api/routes/training.py`.

**Syntax / example**:

```bash
curl -s -X POST http://localhost:8201/v1/training/metrics/clear/undo
```

**State changes** — Repopulates the metrics/history buffer from the undo snapshot and drops the snapshot. `GET /v1/training/status` reports `metrics_clear_undo_available: false` afterwards.

**Returns** — Envelope with `{"status": "restored", "restored_count": <int>, "undo_available": false}`.

**Error handling** — `409` if there is no clear to undo; `503` if lifecycle unbound.

---

### GET `/v1/training/status`

**Summary** — Read the current training status snapshot.

**Detailed description** — Returns a coherent snapshot taken under the WebSocket manager lock so `snapshot_seq` and `server_instance_id` align with the latest streamed state. Implemented at `src/api/routes/training.py:115-127`.

**Syntax / example**:

```bash
curl -s http://localhost:8201/v1/training/status
```

**State changes** — None.

**Returns** — Envelope with: `training_state`, `training_active`, `network_loaded`, `state_machine`, `monitor`, `completion_reason`, `metrics_clear_undo_available`, `snapshot_seq`, `server_instance_id`. `metrics_clear_undo_available` (C5, additive) is `true` while an explicit metrics clear (`POST /v1/training/metrics/clear`) can still be undone — i.e. no run has started since — so a UI can render the undo affordance across a page reload without a separate poll.

**Counter semantics (C2b — the contract UI consumers should render):**

| Field | Block | Meaning |
|-------|-------|---------|
| `current_epoch` / `current_step` | `training_state` | Completed **training steps** — entries in the engine's per-pass history: one initial output-training pass plus one per cascade growth iteration. NOT inner output-training epochs. Single writer (the metrics drain); the two fields are aliases today. |
| `max_epochs` | `training_state` | The **derived** total-epoch cap implied by the granular limits: `output_epochs + min(max_iterations, max_hidden_units) * (candidate_epochs + output_epochs)`. A display budget (the natural `Epoch: X / Y` denominator), not an enforced abort — the granular limits do the gating. Refreshed at network create / param apply / snapshot load. |
| `output_epoch` / `output_total_epochs` | `training_state` | Live progress **within the current output-training pass** (inner epoch vs. that pass's budget, sampled ~every 25th epoch). Zeroed at run start, growth-phase exit, and run end. Output-phase sibling of the `candidate_epoch` pair. |
| `candidate_epoch` / `candidate_total_epochs` | `training_state` | Live progress within the current candidate-pool training pass (from the worker progress stream). |
| `grow_iteration` / `grow_max` | `training_state` | Cascade growth iteration counter vs. its `max_iterations` limit. |
| `learning_rate`, `max_hidden_units`, `max_iterations` | `training_state` | Projections of the live network's effective values (synced at create / apply / snapshot-load) — the same values `/v1/network` and `GET /v1/training/params` report. |
| `current_epoch` | `monitor` | Completed training steps (same unit as `training_state.current_epoch`). E.g. `20` after 14 hidden units = 20 completed passes, not 20 inner epochs. |
| `current_hidden_units` | `monitor` | Installed cascade unit count as of the latest metrics row. |
| `total_metrics` | `monitor` | Buffered metrics ROW count across both row kinds (per-step rows + throttled within-pass samples) — not comparable to `current_epoch`. |

Rows returned by `GET /v1/metrics/history` carry a `kind` discriminator: `"training_step"` rows use step numbering in `epoch`; `"output_epoch"` rows use within-pass inner-epoch numbering.

**Error handling** — `503` if lifecycle unbound.

---

### GET `/v1/training/params`

**Summary** — Read the active training parameter set.

**Detailed description** — Implemented at `src/api/routes/training.py:130-136`.

**Syntax / example**:

```bash
curl -s http://localhost:8201/v1/training/params
```

**State changes** — None.

**Returns** — Envelope with the live params dict. `epochs_max` is **derived read-only** (C2b / Q1): `output_epochs + min(max_iterations, max_hidden_units) * (candidate_epochs + output_epochs)` — it can never contradict the granular limits, and it is always admissible if echoed back to `PATCH` (pre-C2b the echoed construction-time default `1e11` exceeded the PATCH ceiling `1e6`, wholesale-rejecting seeded full-form applies with 422).

**Error handling** — `404` if no network; `503` if lifecycle unbound.

---

### PATCH `/v1/training/params`

**Summary** — Update one or more runtime training parameters.

**Detailed description** — All fields optional (PATCH semantics). Updates the live params, allowing in-flight tuning of learning rates, candidate pool, patience, etc. Implemented at `src/api/routes/training.py:139-153`.

**Syntax**:

```http
PATCH /v1/training/params HTTP/1.1
Content-Type: application/json

{
  "learning_rate": 0.005,
  "candidate_learning_rate": 0.05,
  "patience": 30
}
```

**Body model** — `TrainingParamUpdateRequest` (same field set as `TrainingParams`).

**Example call**:

```bash
curl -s -X PATCH http://localhost:8201/v1/training/params \
  -H 'Content-Type: application/json' \
  -d '{"learning_rate":0.005,"patience":30}'
```

**State changes** — Mutates lifecycle params; takes effect on the next epoch / candidate phase.

**Returns** — Envelope with the merged params dict plus the C2a accounting fields `applied` (keys that landed) and `skipped` (`{"key", "reason"}` rows). `epochs_max` is **deprecated as an input** (C2b / Q1): submitted values are accepted at the request boundary (floor `ge=1` only) but never applied — they are reported as `skipped(not-updatable)`, since the value is derived from the granular limits (see `GET /v1/training/params`).

**Error handling** — `404` if no network; `422` for unknown keys / out-of-range values; `503` if lifecycle unbound.

---

## Metrics

Router in `src/api/routes/metrics.py`, prefix `/v1/metrics`.

### C7 scalar evaluation metrics

**Intent** — Phase 1 of C7 (U-4) attaches classification quality scalars (F1, precision, recall, ROC-AUC) to the existing metrics surfaces so canopy and clients can chart them without a new protocol package. Distinct from Prometheus (`JUNIPER_CASCOR_METRICS_ENABLED`).

**Codepaths** — `TrainingLifecycleManager._compute_eval_scalar_metrics` / `_extract_and_record_metrics` (`src/api/lifecycle/manager.py`); row attachment in `TrainingMonitor.on_epoch_end` (`src/api/lifecycle/monitor.py`); math in `src/api/lifecycle/classification_metrics.py` (torch-native, no scikit-learn).

**Cadence & split**

- Computed once per completed **training step** (initial output pass + one per growth iteration), not per inner output epoch.
- Eval split is validation (`_val_x`/`_val_y`) when present, otherwise training.
- On each metrics drain, scalars attach to the **terminal** `kind="training_step"` row only; older backfilled rows and `kind="output_epoch"` rows keep `f1`/`precision`/`recall`/`roc_auc` as `null` (one forward pass reflects current network state).

**Config** — `JUNIPER_CASCOR_EVAL_METRICS_ENABLED` (default on). Parsed by `_env_flag` in the lifecycle manager (not `api.settings.Settings`). Disable with `0` / `false` / `no` / `off`. Multi-class averaging is fixed to `macro` in the manager today (`_eval_metrics_average`).

**Surfaces**

| Surface | What you get |
|---------|----------------|
| `GET /v1/metrics` | Flat `f1`/`precision`/`recall`/`roc_auc` plus self-describing `eval_metrics` (`enabled`, `average`, `split`, `n_samples`, `n_classes`, `undefined`) |
| `GET /v1/metrics/history` | Same flat keys on each history row (nullable; populated on terminal training-step rows) |
| `WS /ws/training` `metrics` / `initial_metrics` | Same row dict as history (additive keys; protocol envelopes allow extras) |

**Not on** `/v1/training/status` (per-epoch loss/accuracy live on the metrics surfaces only).

**Decode & degradation** — Binary (single output column): threshold 0.5, positive-class scores, `average` reported as `"binary"`. Multi-class: argmax vs one-hot, macro-average by default. Whole-metric `null` with `undefined` reasons: `empty_batch`, `single_class`, `invalid_output`. Failures never raise into the training thread.

**Disable example**:

```bash
JUNIPER_CASCOR_EVAL_METRICS_ENABLED=0 JUNIPER_CASCOR_PORT=8201 python server.py
```

**Pitfalls**

- Do not confuse with `JUNIPER_CASCOR_METRICS_ENABLED` (Prometheus scrape endpoint).
- Expect `null` scalars on within-pass `output_epoch` rows and before the first drain of a run.
- No `juniper-cascor-protocol` bump required — additive nullable keys on existing envelopes.

### GET `/v1/metrics`

**Summary** — Latest metrics snapshot.

**Detailed description** — Same payload schema the `/ws/training` stream emits as `metrics_update`. Implemented at `src/api/routes/metrics.py:17-23`. **C7:** includes flat `f1`/`precision`/`recall`/`roc_auc` and the `eval_metrics` metadata block (see [C7 scalar evaluation metrics](#c7-scalar-evaluation-metrics)).

**Syntax / example**:

```bash
curl -s http://localhost:8201/v1/metrics
```

**State changes** — None.

**Returns** — Envelope with the most recent metrics object (epoch, loss/accuracy, C7 scalars + `eval_metrics`, etc.).

**Error handling** — `404` if no network; `503` if lifecycle unbound.

---

### GET `/v1/metrics/history`

**Summary** — Recent metric history.

**Detailed description** — Returns the most recent `count` entries (or all if `count` is omitted). Used by canopy on initial load to backfill charts before subscribing to the WebSocket stream. Implemented at `src/api/routes/metrics.py:26-33`. **Retention (C5 / Q4):** the history is now retained across training run boundaries by default, so this endpoint stays populated after a run completes and across a subsequent run (pre-C5 the buffer was emptied at each run start). Empty it explicitly — with undo — via `POST /v1/training/metrics/clear`, or start a run with `start_fresh: true` for a clean slate. **C7:** each row always carries nullable `f1`/`precision`/`recall`/`roc_auc` (populated on the terminal `training_step` row of each drain when eval metrics are enabled).

**Syntax**:

```http
GET /v1/metrics/history?count=100 HTTP/1.1
```

**Query params** — `count` (`int ≥ 1`, optional). Without it, the lifecycle's full retained history is returned.

**Example call**:

```bash
curl -s 'http://localhost:8201/v1/metrics/history?count=100'
```

**State changes** — None.

**Returns** — Envelope with `metrics: [<list>]`.

**Error handling** — `503` if lifecycle unbound.

---

### GET `/v1/metrics/transport`

**Summary** — Cumulative WebSocket transport stats (GAP-WS-16).

**Detailed description** — Surfaces counters maintained by the WebSocket manager: bytes/messages sent (overall and per-type), connection counts, replay-buffer state. Useful for diagnosing slow consumers. Implemented at `src/api/routes/metrics.py:36-49`.

**Syntax / example**:

```bash
curl -s http://localhost:8201/v1/metrics/transport
```

**State changes** — None.

**Returns** — Envelope with the transport stats dict.

**Error handling** — `503` if the WebSocket manager isn't initialized.

---

## Dataset

Router in `src/api/routes/dataset.py`, prefix `/v1/dataset`.

### GET `/v1/dataset`

**Summary** — Dataset metadata for the current run.

**Detailed description** — Returns the dataset descriptor (source URL or generator + params, sample counts, feature/label shape, optional checksum). Implemented at `src/api/routes/dataset.py:17-21`.

**Syntax / example**:

```bash
curl -s http://localhost:8201/v1/dataset
```

**State changes** — None.

**Returns** — Envelope with dataset metadata.

**Error handling** — `503` if lifecycle unbound.

---

### GET `/v1/dataset/data`

**Summary** — Full training/validation arrays for visualization.

**Detailed description** — Returns the actual `X_train`/`y_train`/`X_val`/`y_val` data so the canopy front-end can plot points alongside the decision boundary. Implemented at `src/api/routes/dataset.py:24-31`.

**Syntax / example**:

```bash
curl -s http://localhost:8201/v1/dataset/data
```

**State changes** — None.

**Returns** — Envelope with the dataset arrays.

**Error handling** — `404` if no dataset is loaded; `503` if lifecycle unbound.

---

## Decision boundary

Router in `src/api/routes/decision_boundary.py`, prefix `/v1/decision-boundary`.

### GET `/v1/decision-boundary`

**Summary** — Compute the network's decision boundary on a 2D grid.

**Detailed description** — Requires a 2D-input network and loaded training data. Computation runs on a grid of `resolution × resolution` points and is offloaded via `asyncio.to_thread()` so it doesn't block the event loop. Implemented at `src/api/routes/decision_boundary.py:20-39`.

**Syntax**:

```http
GET /v1/decision-boundary?resolution=200 HTTP/1.1
```

**Query params** — `resolution` (`int`, default `100`, range `[10, 512]`).

**Example call**:

```bash
curl -s 'http://localhost:8201/v1/decision-boundary?resolution=200'
```

**State changes** — None.

**Returns** — Envelope with grid points, predictions, and bounding box.

**Error handling** — `404` if no network or no training data; `500` if the boundary computation raises; `503` if lifecycle unbound.

---

## Snapshots

Router in `src/api/routes/snapshots.py`, prefix `/v1/snapshots`. All snapshot IDs in the URL are validated by `_validate_snapshot_id()` (alphanumerics / `_` / `-` only, 1–128 chars; SEC-17 path-traversal hardening). HDF5 I/O is offloaded with `asyncio.to_thread()` to keep the event loop responsive (PERF-CC-01).

### POST `/v1/snapshots`

**Summary** — Save the current network as a new snapshot.

**Detailed description** — Serializes weights, topology, training params, and metric history to HDF5. Implemented at `src/api/routes/snapshots.py:129-142`.

**Syntax**:

```http
POST /v1/snapshots HTTP/1.1
Content-Type: application/json

{ "description": "spiral baseline" }
```

**Body model** — `SnapshotCreateRequest{ description: str = "" }`.

**Example call**:

```bash
curl -s -X POST http://localhost:8201/v1/snapshots \
  -H 'Content-Type: application/json' \
  -d '{"description":"spiral baseline"}'
```

**State changes** — Writes a new HDF5 file under the snapshot store; updates the in-memory snapshot index.

**Returns** — Envelope with snapshot metadata (`snapshot_id`, file path, created_at, description, network summary).

**Error handling** — `404` if no network to snapshot; `503` if lifecycle unbound.

---

### GET `/v1/snapshots`

**Summary** — List all snapshots.

**Detailed description** — Implemented at `src/api/routes/snapshots.py:145-149`.

**Syntax / example**:

```bash
curl -s http://localhost:8201/v1/snapshots
```

**State changes** — None.

**Returns** — Envelope with `snapshots: [<list>]`.

**Error handling** — `503` if lifecycle unbound.

---

### GET `/v1/snapshots/{snapshot_id}`

**Summary** — Read a snapshot's metadata.

**Detailed description** — Implemented at `src/api/routes/snapshots.py:152-160`.

**Syntax / example**:

```bash
curl -s http://localhost:8201/v1/snapshots/snap_20260508_120000
```

**State changes** — None.

**Returns** — Envelope with snapshot metadata.

**Error handling** — `400` if `snapshot_id` fails the regex validator; `404` if not found; `503` if lifecycle unbound.

---

### POST `/v1/snapshots/{snapshot_id}/restore`

**Summary** — Restore a snapshot for inspection / modification (CAN-015d).

**Detailed description** — Loads weights into the live network and transitions the FSM to `Investigating`, where the manual network-mutation endpoints are unlocked. Implemented at `src/api/routes/snapshots.py:163-210`.

**Syntax / example**:

```bash
curl -s -X POST http://localhost:8201/v1/snapshots/snap_20260508_120000/restore
```

**State changes** — Loads snapshot into the live network; transitions FSM `* → Investigating`.

**Returns** — Envelope with `snapshot_id`, `operation: "restore"`, `fsm_state`, `time_index{snapshot_window: ...}`, post-restore `training_params`, and `status: "restored"` (legacy field).

**Error handling**:

| Code | Trigger                                                    |
|------|------------------------------------------------------------|
| 400  | Invalid `snapshot_id` format                               |
| 404  | Snapshot not found                                         |
| 409  | FSM in `Started`/`Paused` (training must be stopped first) |
| 503  | Lifecycle unbound                                          |

---

### POST `/v1/snapshots/{snapshot_id}/retrain`

**Summary** — Restore a snapshot and reset training history so the next start begins at epoch 0 (CAN-015a).

**Detailed description** — Restores weights, topology, and meta-params, but clears history, counters, FSM, and the auto-snap-best ratchet. Implemented at `src/api/routes/snapshots.py:213-253`.

**Syntax / example**:

```bash
curl -s -X POST http://localhost:8201/v1/snapshots/snap_20260508_120000/retrain
```

**State changes** — Loads snapshot; resets history arrays and counters; resets auto-snap ratchet; transitions FSM to idle.

**Returns** — Envelope with `snapshot_id`, `operation: "retrain"`, `fsm_state`, `time_index_default: 0`, post-restore `training_params`, `status: "ready"`.

**Error handling** — `400` invalid id, `404` not found, `503` lifecycle unbound.

---

### POST `/v1/snapshots/{snapshot_id}/resume`

**Summary** — Restore a snapshot preserving training history so the next start continues epoch numbering (CAN-015b).

**Detailed description** — Same as `restore` but keeps history arrays and transitions FSM to `RESUME_READY`. The next `start_training` extends history from the snapshot's terminal epoch. Implemented at `src/api/routes/snapshots.py:256-302`.

**Syntax / example**:

```bash
curl -s -X POST http://localhost:8201/v1/snapshots/snap_20260508_120000/resume
```

**State changes** — Loads snapshot; preserves history; FSM `* → RESUME_READY`.

**Returns** — Envelope with `snapshot_id`, `operation: "resume"`, `fsm_state`, `resume_point_epoch` (snapshot's terminal epoch), `training_params`, `status: "ready"`.

**Error handling**:

| Code | Trigger                   |
|------|---------------------------|
| 400  | Invalid id format         |
| 404  | Snapshot not found        |
| 409  | FSM in `Started`/`Paused` |
| 503  | Lifecycle unbound         |

---

### POST `/v1/snapshots/{snapshot_id}/replay`

**Summary** — Begin a synthetic replay of a snapshot's training history (CAN-015c).

**Detailed description** — Loads the snapshot and spawns a background driver thread (via `asyncio.to_thread()`) that emits `epoch_end` events from the stored history at a configurable speed. Replay starts paused at index 0; control with `replay/control`. V1 covers metric arrays + topology evolution only. Implemented at `src/api/routes/snapshots.py:317-358`.

**Syntax / example**:

```bash
curl -s -X POST http://localhost:8201/v1/snapshots/snap_20260508_120000/replay
```

**State changes** — Loads snapshot; FSM `* → REPLAYING`; spawns the replay driver thread.

**Returns** — Envelope with `snapshot_id`, `operation: "replay"`, `fsm_state`, `time_index_default: "start"`, session summary (current_index, length, speed), `training_params`, `status: "replaying"`.

**Error handling** — `400` invalid id, `404` not found, `409` if FSM is `Started`/`Paused`, `503` lifecycle unbound.

---

### POST `/v1/snapshots/{snapshot_id}/replay/control`

**Summary** — Drive an active replay session (play / pause / seek / speed / range / stop).

**Detailed description** — Controls the replay session bound to the URL's `snapshot_id` (mismatch returns `409` to prevent stale-tab accidents). Implemented at `src/api/routes/snapshots.py:361-407`.

**Syntax**:

```http
POST /v1/snapshots/{snapshot_id}/replay/control HTTP/1.1
Content-Type: application/json

{
  "action": "seek",
  "time_index": 42,
  "value": null,
  "start": null,
  "end": null
}
```

**Body model** — `ReplayControlRequest` with discriminator `action`:

| `action` | Required fields            | Behavior                                     |
|----------|----------------------------|----------------------------------------------|
| `play`   | –                          | Advance from current index                   |
| `pause`  | –                          | Stop advancing                               |
| `seek`   | `time_index` (int)         | Jump to index, clamped to `[0, length)`      |
| `speed`  | `value` (float, `-10..10`) | Set playback speed; `\|value\| < 0.1` pauses |
| `range`  | `start` (int), `end` (int) | Restrict playback to `[start, end)`          |
| `stop`   | –                          | Tear down the session, exit `REPLAYING`      |

**Example call**:

```bash
curl -s -X POST http://localhost:8201/v1/snapshots/snap_20260508_120000/replay/control \
  -H 'Content-Type: application/json' \
  -d '{"action":"seek","time_index":42}'
```

**State changes** — Mutates the active replay session. `stop` transitions FSM `REPLAYING → idle`.

**Returns** — Envelope with `snapshot_id`, `operation: "replay_control"`, `action`, `result` (action-specific summary), and `fsm_state` when applicable.

**Error handling**:

| Code | Trigger                                                                 |
|------|-------------------------------------------------------------------------|
| 400  | Invalid action params (e.g., `seek` outside range, malformed `range`)   |
| 409  | URL `snapshot_id` doesn't match the active session, or no active replay |
| 503  | Lifecycle unbound                                                       |

---

## Workers

Router in `src/api/routes/workers.py`, prefix `/v1/workers`. The worker pool is populated by juniper-cascor-worker connections to `/ws/v1/workers`.

### GET `/v1/workers`

**Summary** — List all currently registered workers.

**Detailed description** — Returns one entry per registered worker with status, health score, task counters, in-flight tasks, last completion time, RSS, last task duration, recent task durations, and GPU utilization (METRICS-MON R1.3 + R4.4 fields). Implemented at `src/api/routes/workers.py:61-71`.

**Syntax / example**:

```bash
curl -s http://localhost:8201/v1/workers
```

**State changes** — None.

**Returns** — Envelope with `{workers: [...], count: <int>}`.

**Error handling** — `503` if the worker registry isn't bound.

---

### GET `/v1/workers/stats`

**Summary** — Aggregate worker pool statistics.

**Detailed description** — Health-state counts, totals, and average health score; useful for dashboards. Implemented at `src/api/routes/workers.py:74-97`.

**Syntax / example**:

```bash
curl -s http://localhost:8201/v1/workers/stats
```

**State changes** — None.

**Returns** — Envelope with `total`, `idle`, `busy`, `stale`, `total_tasks_completed`, `total_tasks_failed`, `average_health_score`, `timestamp`.

**Error handling** — `503` if the registry isn't bound.

---

### GET `/v1/workers/{worker_id}`

**Summary** — Read one worker's full status.

**Detailed description** — Implemented at `src/api/routes/workers.py:100-107`.

**Syntax / example**:

```bash
curl -s http://localhost:8201/v1/workers/worker-abc123
```

**State changes** — None.

**Returns** — Envelope with the worker's full status object.

**Error handling** — `404` if not found; `503` if registry unbound.

---

## WebSocket endpoints

All three sockets share these properties:

- `X-API-Key` authenticated via `ws_authenticate()` (`src/api/websocket/manager.py`); failures close with `4001`.
- Admission caps: every WebSocket reserves from the stack-global cap (default 200); `/ws/control` also reserves from the per-identity cap (default 5, keyed on the API-key digest). Over-cap attempts close with `1013`.
- The legacy per-peer-IP cap remains DoS-dampening only. Behind Docker NAT, every client may present as the bridge gateway and therefore share one IP bucket; use the global and per-identity caps for limits that survive NAT.
- Application-layer heartbeat: server sends `{"type":"ping","ts":<float>}` every 30s; the client must send a `{"type":"pong"}` (or any other frame — C3 tolerance) within 10s or the connection is closed with `1011`.
- All payloads are JSON unless explicitly noted as binary.
- Over-cap handshakes close with `1013`. `JUNIPER_CASCOR_WS_MAX_CONNECTIONS_GLOBAL` (default 200) spans all WebSocket endpoints; `JUNIPER_CASCOR_WS_MAX_CONNECTIONS_PER_IDENTITY` (default 5) applies to `/ws/control`; `JUNIPER_CASCOR_WS_MAX_CONNECTIONS_PER_IP` (default 5) is DoS dampening and can collapse to one shared bucket behind Docker NAT.

`/ws/training` and `/ws/control` also use an application-layer heartbeat — the explicit contract (C3):

- The server sends `{"type":"ping","ts":<float>}` every `ws_heartbeat_interval_sec` seconds (default `30`; env `JUNIPER_WS_HEARTBEAT_INTERVAL_SEC`). A value `<= 0` disables the heartbeat entirely (escape hatch for legacy clients; the `/ws/control` bidirectional idle timeout, `ws_control_idle_timeout_sec` default `120`, still applies).
- The client SHOULD reply `{"type":"pong"}`. Any well-formed inbound frame received within `ws_heartbeat_pong_timeout_sec` seconds (default `10`; env `JUNIPER_WS_HEARTBEAT_PONG_TIMEOUT_SEC`) of a ping also counts as proof of liveness — the heartbeat performs dead-peer detection, not frame-type compliance, so an actively-sending client is never reaped.
- A client that sends nothing within the pong window is closed with code `1011`, reason `Heartbeat timeout: no pong or traffic within <N>s`, and a server-side WARNING log line. (Pre-C3 the close used `1006`, which RFC 6455 §7.4.1 forbids on the wire — the `websockets` server implementation rejects it, so the close frame never reached the peer and clients were left holding a silent half-open socket.)
- `juniper-cascor-client >= 0.7.0` answers pings automatically on both streams (CL1) and exposes `is_alive(window)` / `last_frame_at` liveness surfaces for supervisors.
- T5 observability: every heartbeat ping is recorded in the transport counters (`GET /v1/metrics/transport`, `messages_sent_by_type.ping`), and the WS manager logs a periodic INFO emission summary (`WS emission summary (last <N>s): metrics=…, ping=… (<K> active connections)`, interval `ws_emission_summary_interval_sec`, default `60`, env `JUNIPER_WS_EMISSION_SUMMARY_INTERVAL_SEC`, `<= 0` disables) so "connected but nothing flowing" is diagnosable server-side.

### WS `/ws/training`

**Summary** — Streaming training events for dashboards (juniper-canopy).

**Detailed description** — Optionally accepts a `resume` handshake within a configurable timeout to replay buffered events from a sequence number.
On a fresh connect, the server sends `connection_established` followed by `initial_status`, `state`, and `initial_metrics` (configurable burst size, default 100).
During training it broadcasts `metrics`, `state`, `topology`, `cascade_add`, `candidate_progress`, and `event` frames (builders in `src/api/websocket/messages.py`; emission call sites registered in `src/api/lifecycle/manager.py:1333-1355`).
Replay buffer default 10,000 messages, with stack-global and per-IP admission limits (defaults: 200 global across all WebSocket endpoints, 5 per IP), max message size 16 MB, chunk payload size 1 MB, send timeout 10 s, state coalescing 50 ms.
Handler at `src/api/websocket/training_stream.py`.

**Connect** — `ws://localhost:8201/ws/training` (mounted in `src/api/app.py:471`).

**Resume handshake (optional, client → server within timeout)**:

```json
{
  "type": "resume",
  "data": {
    "last_seq": 12345,
    "server_instance_id": "server-uuid-from-connection_established"
  }
}
```

Resume responses:

| Type | Data | Meaning |
|------|------|---------|
| `resume_ok` | `{"replayed_count": <int>}` | Replay succeeded; buffered events follow as personal messages. |
| `resume_failed` | `{"reason": "malformed_resume"}` | Frame omitted `data.last_seq` or `data.server_instance_id`. |
| `resume_failed` | `{"reason": "server_restarted"}` | Client's `server_instance_id` did not match this process. |
| `resume_failed` | `{"reason": "out_of_range"}` | `last_seq` is older than the retained replay buffer or replay is disabled. |

If no resume frame arrives within the handshake timeout, or a non-resume/non-JSON
frame arrives during that window, the connection proceeds as a fresh connect.

**Other client→server messages**:

```json
{ "type": "pong" }
```

```json
{ "type": "subscribe_metrics", "data": { "max_count": 50 } }
```

`subscribe_metrics` replies with an `initial_metrics` personal message. The
requested count is clamped to `[1, ws_initial_metrics_count]` (or `100` when the
initial burst is disabled).

**Server→client message types**:

| Type                     | When                                      |
|--------------------------|-------------------------------------------|
| `connection_established` | After auth                                |
| `initial_status`         | Fresh connect                             |
| `state`                  | FSM transitions / coalesced state updates |
| `initial_metrics`        | Fresh connect (back-fill)                 |
| `resume_ok`              | Successful resume handshake               |
| `resume_failed`          | Failed resume handshake                   |
| `epoch_end`              | Each epoch completion                     |
| `candidate_progress`     | Candidate worker progress                 |
| `cascade_add`            | New hidden unit added                     |
| `topology`               | When the network grows / mutates          |
| `chunked_message`        | Fragment of an oversized JSON envelope    |
| `event`                  | Generic lifecycle event                   |
| `ping`                   | Heartbeat ping every 30 s by default      |

Broadcast messages get a monotonic `seq` and `emitted_at_monotonic` before
fan-out and replay buffering. Personal messages (`connection_established`,
`initial_status`, `initial_metrics`, `resume_ok`, `resume_failed`) do not get a
`seq`; `initial_metrics.data.current_seq` tells the client which broadcast
sequence is current at the time of the back-fill.

Oversized JSON envelopes are sent as one or more `chunked_message` envelopes:

```json
{
  "type": "chunked_message",
  "timestamp": 1714000000.123,
  "seq": 42,
  "data": {
    "chunk_id": "uuid",
    "chunk_index": 0,
    "total_chunks": 3,
    "original_type": "topology",
    "payload": "{\"type\":\"topology\",..."
  }
}
```

Clients reconstruct by grouping on `chunk_id`, sorting by `chunk_index`, joining
`payload`, and parsing the resulting JSON. Each chunk receives its own `seq` and
replay-buffer slot.

**Example call**:

```bash
websocat -H "X-API-Key: $API_KEY" ws://localhost:8201/ws/training
```

**State changes** — None directly. The stream is read-only; only the server-side connection registry and replay buffer are mutated.

**Error handling / close codes**:

| Code | Trigger                                   |
|------|-------------------------------------------|
| 1013 | Manager, stack-global, or per-IP cap hit  |
| 4001 | Auth failure                              |
| 1013 | Global or per-IP connection cap reached  |
| 1006 | Heartbeat pong timeout, message too large |
| 1000 | Normal close                              |

---

### WS `/ws/control`

**Summary** — Authenticated command channel for training lifecycle control.

**Detailed description** — Origin header is rejected with `4003` if present (Phase B-pre-b: machine-to-machine only). Admission reserves a stack-global slot and a per-identity slot keyed on a non-reversible per-process HMAC of the presented `X-API-Key`; over-cap closes with `1013`. Per-connection leaky-bucket rate limit (default 10 cmd/s). Bidirectional 120 s idle timeout. Per-origin handshake cooldown. Phase D execution timeouts: `start` 10 s; `stop`/`pause`/`resume`/`reset` 2 s; `set_params` 1 s. Phase D §S10.7 lazily registers Prometheus counter `cascor_ws_control_command_received_total{command}` via `register_or_reuse`. Handler at `src/api/websocket/control_stream.py`.

**Connect** — `ws://localhost:8201/ws/control` (mounted in `src/api/app.py:472`).

**Client→server command frame**:

```json
{
  "command": "start",
  "command_id": "uuid-or-omit",
  "params": {
    "dataset": {"source": "generator", "generator": "spiral", "params": {"n": 200}},
    "params": {"learning_rate": 0.01}
  }
}
```

Valid commands: `start`, `stop`, `pause`, `resume`, `reset`, `set_params`. The schemas mirror their REST equivalents.

**Server→client response frame** (D-03 canonical — no `seq` field)

```json
{
  "type": "command_response",
  "timestamp": 1714000000.123,
  "data": {
    "command": "start",
    "command_id": "uuid",
    "status": "success",
    "result": { "...": "..." }
  }
}
```

Validation and command failures keep the socket open and return the same
envelope with `data.status: "error"` plus `data.error` and, when available,
`data.code` (for example `unknown_command` or `invalid_params`).
The rate-limited response is the legacy flat shape emitted directly by the
handler and also has no `seq`:

```json
{
  "type": "command_response",
  "command": "pause",
  "command_id": "uuid",
  "status": "rate_limited",
  "retry_after": 0.1
}
```

**Example call**:

```bash
websocat -H "X-API-Key: $API_KEY" ws://localhost:8201/ws/control \
  <<< '{"command":"pause","command_id":"abc"}'
```

**State changes** — `start`, `stop`, `pause`, `resume`, `reset` mutate the lifecycle and FSM identically to their REST counterparts. `set_params` mutates training params identically to `PATCH /v1/training/params`.

**Error handling / close codes**:

| Code | Trigger                          |
|------|----------------------------------|
| 1013 | Stack-global or per-identity cap hit |
| 4001 | Auth failure                     |
| 4003 | Origin header present (B2B-only) |
| 1013 | Global or per-identity connection cap reached |
| 1008 | Rate limit exceeded              |
| 1006 | Heartbeat timeout, idle timeout  |

Rate limiting and per-command failures arrive in-band as `command_response`
messages rather than closing the socket. Oversized command messages (over 64 KB)
also receive an error response and the connection stays open. Malformed JSON
receives an error response and then closes with `1003`.

---

### WS `/ws/v1/workers`

**Summary** — Worker registration and task dispatch socket (juniper-cascor-worker).

**Detailed description** — Origin header is rejected with `4003` if present (Section 12.3 — machine-to-machine only). Admission reserves only a stack-global WebSocket slot: worker fleets may share one token, and the server-assigned `worker_id` is not known until after admission. Optional Phase 4 protections: per-source-IP connection rate limiter (default 10 conn/min, burst 2), anomaly detector (perfect-correlation and training-time deviation guards), audit logging, and worker performance metrics. Handler at `src/api/websocket/worker_stream.py`.

**Connect** — `ws://localhost:8201/ws/v1/workers` (mounted in `src/api/app.py:473`).

**Wire protocol** — JSON envelope plus binary tensor frames. See `src/api/workers/protocol.py`.

| Message type   | Direction       | Payload                                            |
|----------------|-----------------|----------------------------------------------------|
| `REGISTRATION` | worker → server | `{type, worker_id, capabilities{frameworks, ...}}` |
| `HEARTBEAT`    | worker → server | periodic                                           |
| `TASK_ASSIGN`  | server → worker | task spec + binary tensor frames                   |
| `TASK_RESULT`  | worker → server | result envelope + binary tensor frames             |
| `WORKER_ERROR` | worker → server | error envelope                                     |

**Limits** — JSON ≤ 65 KB; binary ≤ 100 MB.

**Example registration**:

```json
{
  "type": "registration",
  "worker_id": "worker-abc123",
  "capabilities": {"frameworks": ["torch"], "gpu": true}
}
```

**State changes** — On `REGISTRATION` the worker is added to the registry. `TASK_RESULT` updates the worker's task counters, health score, and recent durations. `WORKER_ERROR` increments failure counters and may quarantine the worker (Phase 4 anomaly detector).

**Error handling / close codes**:

| Code | Trigger                              |
|------|--------------------------------------|
| 1013 | Stack-global WebSocket cap hit       |
| 4001 | Auth failure                         |
| 4003 | Origin header present                |
| 4004 | Worker subsystem not initialized     |
| 1013 | Stack-global WebSocket cap reached   |
| 4029 | Connection rate limit exceeded       |
| 1006 | Message too large, heartbeat timeout |

---

## State-modifying endpoints summary

The endpoints below mutate application state. All others are read-only.

| Endpoint                                                              | Lifecycle method invoked                        | Effect                                                |
|-----------------------------------------------------------------------|-------------------------------------------------|-------------------------------------------------------|
| POST `/v1/network`                                                    | `lifecycle.create_network()`                    | Creates network                                       |
| DELETE `/v1/network`                                                  | `lifecycle.delete_network()`                    | Deletes network                                       |
| PATCH `/v1/network/weights`                                           | `lifecycle.patch_weights()`                     | Rewrites a parameter group (FSM-gated)                |
| POST `/v1/network/hidden-units`                                       | `lifecycle.add_hidden_unit_manual()`            | Adds a hidden unit (FSM-gated)                        |
| DELETE `/v1/network/hidden-units/{idx}`                               | `lifecycle.remove_hidden_unit_manual()`         | Removes a hidden unit (FSM-gated)                     |
| POST `/v1/training/start`                                             | `lifecycle.start_training()`                    | Starts the loop                                       |
| POST `/v1/training/stop`                                              | `lifecycle.stop_training()`                     | Stops the loop                                        |
| POST `/v1/training/pause`                                             | `lifecycle.pause_training()`                    | Pauses                                                |
| POST `/v1/training/resume`                                            | `lifecycle.resume_training()`                   | Resumes                                               |
| POST `/v1/training/reset`                                             | `lifecycle.reset()`                             | Resets history & counters                             |
| PATCH `/v1/training/params`                                           | `lifecycle.update_params()`                     | Mutates runtime params                                |
| POST `/v1/snapshots`                                                  | `lifecycle.save_snapshot()` (offloaded)         | Writes HDF5                                           |
| POST `/v1/snapshots/{id}/restore`                                     | `lifecycle.load_snapshot()` (offloaded)         | Loads + FSM → `Investigating`                         |
| POST `/v1/snapshots/{id}/retrain`                                     | `lifecycle.restore_for_retrain()` (offloaded)   | Loads + clears history                                |
| POST `/v1/snapshots/{id}/resume`                                      | `lifecycle.resume_from_snapshot()` (offloaded)  | Loads + FSM → `RESUME_READY`                          |
| POST `/v1/snapshots/{id}/replay`                                      | `lifecycle.start_replay()` (offloaded)          | Spawns replay thread + FSM → `REPLAYING`              |
| POST `/v1/snapshots/{id}/replay/control`                              | `lifecycle.replay_control()`                    | Mutates replay session, may FSM → idle                |
| WS `/ws/control` `start`/`stop`/`pause`/`resume`/`reset`/`set_params` | Same lifecycle methods as the REST counterparts | Same state changes                                    |
| WS `/ws/v1/workers` `REGISTRATION` / `TASK_RESULT` / `WORKER_ERROR`   | Worker registry mutations                       | Adds/updates worker, updates counters, may quarantine |

---

## See also

- Source: `src/api/app.py`, `src/api/routes/*.py`, `src/api/websocket/*.py`, `src/api/models/*.py`, `src/api/security.py`, `src/api/workers/*.py` in juniper-cascor.
- Ecosystem map: `/home/pcalnon/Development/python/Juniper/CLAUDE.md`.
- Service ports & env vars: `juniper-ml/docs/REFERENCE.md`.
- Observability helpers used by these handlers: `juniper-observability` (`register_or_reuse`, `ReadinessResponse`, `DependencyStatus`).
