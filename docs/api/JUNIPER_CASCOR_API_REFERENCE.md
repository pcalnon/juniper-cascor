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
  - [GET `/v1/training/status`](#get-v1trainingstatus)
  - [GET `/v1/training/params`](#get-v1trainingparams)
  - [PATCH `/v1/training/params`](#patch-v1trainingparams)
- [Metrics](#metrics)
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

### Authentication

Optional. Controlled by the application settings:

- REST: `X-API-Key` header validated by `APIKeyAuth` middleware (`src/api/security.py`). When `settings.api_keys` is empty, auth is disabled (dev mode).
- WebSocket: same `X-API-Key` header, validated in `ws_authenticate()` (`src/api/websocket/manager.py`). On failure the socket is closed with code `4001`.

Rate limiting is also optional; per-IP REST limiter defaults to 100 req/min, and the worker WebSocket has its own per-IP connection rate limiter.

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

| Code | Meaning                                                     |
|------|-------------------------------------------------------------|
| 1000 | Normal closure                                              |
| 1006 | Abnormal closure — heartbeat timeout, message size exceeded |
| 4001 | Authentication required / `X-API-Key` invalid               |
| 4003 | Origin header not permitted (control + worker streams)      |
| 4004 | Worker subsystem not initialized                            |
| 4029 | Connection rate limited (worker stream)                     |

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
  "epochs_max": 500,
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

**State changes** — Clears training history arrays, epoch counters, auto-snap-best ratchet; transitions FSM to idle.

**Returns** — Envelope with reset result.

**Error handling** — `503` if lifecycle unbound.

---

### GET `/v1/training/status`

**Summary** — Read the current training status snapshot.

**Detailed description** — Returns a coherent snapshot taken under the WebSocket manager lock so `snapshot_seq` and `server_instance_id` align with the latest streamed state. Implemented at `src/api/routes/training.py:115-127`.

**Syntax / example**:

```bash
curl -s http://localhost:8201/v1/training/status
```

**State changes** — None.

**Returns** — Envelope with: `training_state`, `training_active`, `network_loaded`, `state_machine`, `monitor`, `snapshot_seq`, `server_instance_id`.

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

**Returns** — Envelope with the live params dict.

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

**Returns** — Envelope with the merged params dict.

**Error handling** — `404` if no network; `422` for unknown keys / out-of-range values; `503` if lifecycle unbound.

---

## Metrics

Router in `src/api/routes/metrics.py`, prefix `/v1/metrics`.

### GET `/v1/metrics`

**Summary** — Latest metrics snapshot.

**Detailed description** — Same payload schema the `/ws/training` stream emits as `metrics_update`. Implemented at `src/api/routes/metrics.py:17-23`.

**Syntax / example**:

```bash
curl -s http://localhost:8201/v1/metrics
```

**State changes** — None.

**Returns** — Envelope with the most recent metrics object (epoch, loss, accuracy, candidate scores, etc.).

**Error handling** — `404` if no network; `503` if lifecycle unbound.

---

### GET `/v1/metrics/history`

**Summary** — Recent metric history.

**Detailed description** — Returns the most recent `count` entries (or all if `count` is omitted). Used by canopy on initial load to backfill charts before subscribing to the WebSocket stream. Implemented at `src/api/routes/metrics.py:26-33`.

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
- Application-layer heartbeat: server sends `{"type":"ping","ts":<float>}` every 30s; client must reply `{"type":"pong"}` within 10s or the connection is closed.
- All payloads are JSON unless explicitly noted as binary.

### WS `/ws/training`

**Summary** — Streaming training events for dashboards (juniper-canopy).

**Detailed description** — Optionally accepts a `resume` handshake within a configurable timeout to replay buffered events from a sequence number.
On a fresh connect, the server sends `connection_established` followed by `initial_status`, `state`, and `initial_metrics` (configurable burst size, default 100).
During training it broadcasts `epoch_end`, `state`, and `topology_update` events.
Replay buffer default 10,000 messages, with per-IP and global connection limits (defaults: 100 total, 10 per IP), max message size 16 MB, chunk payload size 1 MB, send timeout 10 s, state coalescing 50 ms.
Handler at `src/api/websocket/training_stream.py`.

**Connect** — `ws://localhost:8201/ws/training` (mounted in `src/api/app.py:471`).

**Resume handshake (optional, client → server within timeout)**:

```json
{ "type": "resume", "seq": 12345 }
```

**Other client→server messages** — `{"type":"pong"}`, optional `{"type":"subscribe_metrics", ...}`.

**Server→client message types**:

| Type                     | When                                      |
|--------------------------|-------------------------------------------|
| `connection_established` | After auth                                |
| `initial_status`         | Fresh connect                             |
| `state`                  | FSM transitions / coalesced state updates |
| `initial_metrics`        | Fresh connect (back-fill)                 |
| `epoch_end`              | Each epoch completion                     |
| `topology_update`        | When the network grows / mutates          |
| `heartbeat_ping`         | Every 30 s                                |

**Example call**:

```bash
websocat -H "X-API-Key: $API_KEY" ws://localhost:8201/ws/training
```

**State changes** — None directly. The stream is read-only; only the server-side connection registry and replay buffer are mutated.

**Error handling / close codes**:

| Code | Trigger                                   |
|------|-------------------------------------------|
| 4001 | Auth failure                              |
| 1006 | Heartbeat pong timeout, message too large |
| 1000 | Normal close                              |

---

### WS `/ws/control`

**Summary** — Authenticated command channel for training lifecycle control.

**Detailed description** — Origin header is rejected with `4003` if present (Phase B-pre-b: machine-to-machine only). Per-connection leaky-bucket rate limit (default 10 cmd/s). Bidirectional 120 s idle timeout. Per-origin handshake cooldown. Phase D execution timeouts: `start` 10 s; `stop`/`pause`/`resume`/`reset` 2 s; `set_params` 1 s. Phase D §S10.7 lazily registers Prometheus counter `cascor_ws_control_command_received_total{command}` via `register_or_reuse`. Handler at `src/api/websocket/control_stream.py`.

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
  "command": "start",
  "command_id": "uuid",
  "status": "success",
  "result": { "...": "..." },
  "error": null
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
| 4001 | Auth failure                     |
| 4003 | Origin header present (B2B-only) |
| 1008 | Rate limit exceeded              |
| 1006 | Heartbeat timeout, idle timeout  |

Per-command failures arrive in-band as `{"status":"error", "error": "..."}` rather than closing the socket.

---

### WS `/ws/v1/workers`

**Summary** — Worker registration and task dispatch socket (juniper-cascor-worker).

**Detailed description** — Origin header is rejected with `4003` if present (Section 12.3 — machine-to-machine only). Optional Phase 4 protections: per-source-IP connection rate limiter (default 10 conn/min, burst 2), anomaly detector (perfect-correlation and training-time deviation guards), audit logging, and worker performance metrics. Handler at `src/api/websocket/worker_stream.py`.

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
| 4001 | Auth failure                         |
| 4003 | Origin header present                |
| 4004 | Worker subsystem not initialized     |
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
