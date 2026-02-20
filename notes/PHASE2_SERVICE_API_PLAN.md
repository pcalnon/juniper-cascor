# Phase 2 — Complete CasCor Service API

**Created:** 2026-02-20
**Status:** In Progress

## Context

Phase 2 of the polyrepo migration adds a FastAPI + WebSocket service layer to JuniperCascor so Canopy can consume it as a network service instead of importing it as a library. ~60-65% of Phase 2 is already implemented (103 tests passing). The remaining work focuses on:

1. **WebSocket streaming** (critical path for Canopy integration) — completely missing (stub only)
2. **Missing REST routes** — metrics, dataset, decision-boundary endpoints
3. **Per-epoch monitoring hooks** — current hooks only fire after `fit()` returns, not per-epoch
4. **Integration tests** for the new endpoints

The existing CLI entry point (`main.py`) remains unchanged.

## Working Directory

All changes in `/home/pcalnon/Development/python/Juniper/JuniperCascor/juniper_cascor/src/`

## Steps

### Step 1 — Add Missing REST Routes (metrics, dataset, decision-boundary)

- `api/routes/metrics.py` — `GET /v1/metrics`, `GET /v1/metrics/history`
- `api/routes/dataset.py` — `GET /v1/dataset`
- `api/routes/decision_boundary.py` — `GET /v1/decision-boundary`
- `api/lifecycle/manager.py` — add `get_dataset()`, `get_decision_boundary()`
- `api/app.py` — register new routers

### Step 2 — WebSocket Infrastructure

- `api/websocket/manager.py` — Connection manager (connect/disconnect/broadcast/broadcast_from_thread)
- `api/websocket/messages.py` — Message builders ({type, timestamp, data} format)
- `api/websocket/training_stream.py` — `/ws/training` handler
- `api/websocket/control_stream.py` — `/ws/control` handler
- `api/app.py` — register WebSocket endpoints

### Step 3 — Per-Epoch Monitoring Hooks

- Enhanced `_install_monitoring_hooks()` for per-epoch metrics
- `set_ws_manager()` method on lifecycle manager
- Monitor callbacks wired to WebSocket broadcasts

### Step 4 — Training Start with Dataset Support

- Enhanced `TrainingStartRequest` model with DatasetSource
- Training start route supports inline data and juniper-data source

### Step 5 — Unit Tests

- Tests for all new routes, WebSocket manager, messages, handlers, monitoring hooks

### Step 6 — Integration Tests

- Full lifecycle tests and WebSocket streaming tests

## Deferred

- Snapshot routes (`/v1/snapshots/*`)
- Worker routes (`/v1/workers/*`)
- `PUT /v1/training/params`
