# Decouple JuniperCanopy from JuniperCascor — Comprehensive Plan

**Created:** 2026-02-21
**Status:** Planning
**Phase:** Polyrepo Migration Phase 4
**Prerequisite:** Phase 2 (CasCor Service API) and Phase 3 (juniper-cascor-client) complete
**Duration:** 2-3 weeks
**Risk:** High (core architectural change to Canopy)
**Parent Plan:** `POLYREPO_MIGRATION_PLAN.md` Phase 4

---

## 1. Objective

Replace Canopy's `CascorIntegration` class (~1,600 lines of `sys.path` injection, monkey-patching, and direct CasCor imports) with a `CascorServiceAdapter` that wraps the `juniper-cascor-client` package, communicating over REST/WebSocket instead of in-process Python imports.

---

## 2. Current State — What CascorIntegration Does

`CascorIntegration` lives at `JuniperCanopy/juniper_canopy/src/backend/cascor_integration.py` and performs:

| Category | Methods | Replacement Strategy |
|----------|---------|---------------------|
| **Backend discovery** | `_resolve_backend_path`, `_add_backend_to_path`, `_import_backend_modules` | **Eliminated** — no path injection needed |
| **Network lifecycle** | `create_network`, `connect_to_network` | `client.create_network()`, `client.get_network()` |
| **Training control** | `fit_async`, `start_training_background`, `request_training_stop`, `is_training_in_progress` | `client.start_training()`, `client.stop_training()`, `client.get_training_status()` |
| **Monitoring hooks** | `install_monitoring_hooks`, method wrapping of `fit`/`train_output`/`train_candidates`, `_on_*` callbacks | **Eliminated** — CasCor service handles monitoring and streams via WebSocket |
| **Monitoring thread** | `start_monitoring_thread`, `stop_monitoring`, `_monitoring_loop`, `_extract_current_metrics` | **Replaced** by WebSocket subscription via `CascorTrainingStream` |
| **Network data** | `get_network_topology`, `get_network_data`, `extract_cascor_topology`, `get_dataset_info`, `get_prediction_function` | `client.get_topology()`, `client.get_dataset()`, `client.get_decision_boundary()` |
| **Remote workers** | `connect_remote_workers`, `start_remote_workers`, `stop_remote_workers`, `disconnect_remote_workers`, `get_remote_worker_status` | **Stub no-ops** — workers managed server-side by CasCor service + juniper-cascor-worker |
| **Dataset generation** | `_generate_dataset_from_juniper_data`, `_create_juniper_dataset` | Handled by CasCor service; Canopy passes dataset config in `start_training()` |
| **Broadcasting** | `_broadcast_message` (to Canopy's WebSocketManager) | WebSocket relay: CasCor WS → adapter → Canopy frontend WS |

### Current Activation Logic in main.py

```python
# Current: binary mode
if force_demo_mode:
    cascor_integration = None
    demo_mode_active = True
else:
    try:
        cascor_integration = CascorIntegration(cascor_backend_path)
        demo_mode_active = False
    except FileNotFoundError:
        cascor_integration = None
        demo_mode_active = True
```

### Key Internal State Managed by CascorIntegration

- `self.network` — `CascadeCorrelationNetwork` instance
- `self._original_methods` — saved unwrapped methods for hook restoration
- `self.monitoring_thread` — background polling thread
- `self._training_executor` — `ThreadPoolExecutor(max_workers=1)` for async training
- `self._training_future` / `self._training_stop_requested` — training lifecycle state
- `self.metrics_lock`, `self.topology_lock`, `self._training_lock` — thread safety
- `self._remote_client` — `RemoteWorkerClient` for distributed training

### WebSocket Message Types Broadcast

1. `training_start` — `{type, timestamp, input_size, output_size}`
2. `training_complete` — `{type, timestamp, history, hidden_units_added}`
3. `phase_start` / `phase_end` — `{type, phase, timestamp, loss?, accuracy?, hidden_units?, epoch?}`
4. `metrics_update` — `{type, epoch, train_loss, train_accuracy, value_loss, value_accuracy, hidden_units, timestamp}`

---

## 3. Target Architecture — CascorServiceAdapter

### 3.1 juniper-cascor-client API (v0.1.0)

The adapter wraps these three client classes:

**JuniperCascorClient** (synchronous REST, `http://localhost:8200`):

| Method | Signature | Maps From |
|--------|-----------|-----------|
| `create_network` | `(**kwargs) -> Dict` | `CascorIntegration.create_network(config)` |
| `get_network` | `() -> Dict` | `CascorIntegration.connect_to_network()` |
| `delete_network` | `() -> Dict` | (new) |
| `get_topology` | `() -> Dict` | `CascorIntegration.get_network_topology()` |
| `get_statistics` | `() -> Dict` | (new) |
| `start_training` | `(epochs?, dataset?, inline_data?, params?) -> Dict` | `CascorIntegration.start_training_background()` |
| `stop_training` | `() -> Dict` | `CascorIntegration.request_training_stop()` |
| `pause_training` | `() -> Dict` | (new) |
| `resume_training` | `() -> Dict` | (new) |
| `reset_training` | `() -> Dict` | (new) |
| `get_training_status` | `() -> Dict` | `CascorIntegration.get_training_status()` |
| `get_training_params` | `() -> Dict` | (new) |
| `get_metrics` | `() -> Dict` | `CascorIntegration._extract_current_metrics()` |
| `get_metrics_history` | `(count?) -> Dict` | (new) |
| `get_dataset` | `() -> Dict` | `CascorIntegration.get_dataset_info()` |
| `get_decision_boundary` | `(resolution?) -> Dict` | `CascorIntegration.get_prediction_function()` |
| `health_check` | `() -> Dict` | (new) |
| `is_alive` / `is_ready` | `() -> bool` | (new) |
| `wait_for_ready` | `(timeout?, poll_interval?) -> bool` | (new) |

**CascorTrainingStream** (async WebSocket, `ws://localhost:8200/ws/training`):

| Method | Purpose | Maps From |
|--------|---------|-----------|
| `connect(path?)` | Connect to WS endpoint | `CascorIntegration.start_monitoring_thread()` |
| `disconnect()` | Close WS connection | `CascorIntegration.stop_monitoring()` |
| `stream()` | `AsyncIterator[Dict]` — yields messages | `CascorIntegration._monitoring_loop()` |
| `listen()` | Block and dispatch to callbacks | (new pattern) |
| `on_metrics(cb)` | Register metrics callback | `CascorIntegration.create_monitoring_callback()` |
| `on_state(cb)` | Register state change callback | (new) |
| `on_topology(cb)` | Register topology update callback | (new) |
| `on_cascade_add(cb)` | Register cascade unit callback | (new) |
| `on_event(cb)` | Register general event callback | (new) |
| `send_command(cmd, params?)` | Send control command via WS | (new) |

**CascorControlStream** (async WebSocket, `ws://localhost:8200/ws/control`):

| Method | Purpose |
|--------|---------|
| `connect()` | Connect to `/ws/control` |
| `disconnect()` | Close connection |
| `command(cmd, params?)` | Send command and await response |

### 3.2 CascorServiceAdapter Design

```python
# src/backend/cascor_service_adapter.py

import asyncio
from typing import Any, Dict, Optional
from juniper_cascor_client import (
    JuniperCascorClient,
    CascorTrainingStream,
    JuniperCascorConnectionError,
)

class CascorServiceAdapter:
    """Adapter between Canopy's internal interfaces and the CasCor service.

    Replaces CascorIntegration. Communicates with CasCor via REST API
    and WebSocket rather than direct Python imports.

    Constructor takes (service_url, api_key) — NOT (cascor_url,
    websocket_manager, data_client) as the migration plan originally
    specified. The WebSocket manager is imported internally to avoid
    coupling the adapter's constructor to Canopy's WS infrastructure.
    """

    def __init__(
        self,
        service_url: str = "http://localhost:8200",
        api_key: Optional[str] = None,
    ) -> None:
        self.service_url = service_url
        self.client = JuniperCascorClient(
            base_url=service_url,
            api_key=api_key,
        )
        ws_url = service_url.replace("http://", "ws://").replace("https://", "wss://")
        self.training_stream = CascorTrainingStream(
            base_url=ws_url,
            api_key=api_key,
        )
        self._relay_task: Optional[asyncio.Task] = None
        self._connected = False

    # --- Lifecycle ---

    async def connect(self) -> bool:
        """Wait for the CasCor service to be ready."""
        return self.client.wait_for_ready(timeout=30.0)

    async def start_metrics_relay(self) -> None:
        """Connect to CasCor WS and relay metrics to Canopy frontend."""
        from communication.websocket_manager import websocket_manager
        await self.training_stream.connect()
        self._relay_task = asyncio.create_task(
            self._relay_loop(websocket_manager)
        )
        self._connected = True

    async def _relay_loop(self, ws_manager) -> None:
        """Relay messages from CasCor WS to Canopy's frontend WS."""
        async for message in self.training_stream.stream():
            await ws_manager.broadcast(message)

    async def stop_metrics_relay(self) -> None:
        """Disconnect WS relay."""
        if self._relay_task and not self._relay_task.done():
            self._relay_task.cancel()
        await self.training_stream.disconnect()
        self._connected = False

    def shutdown(self) -> None:
        """Clean up all resources."""
        self.client.close()

    # --- Network Management (backward-compatible names) ---

    def create_network(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Create a new network. Accepts dict config like CascorIntegration."""
        kwargs = config or {}
        return self.client.create_network(**kwargs)

    def connect_to_network(self, network=None) -> bool:
        """Get current network state. Ignores `network` arg (service owns it)."""
        result = self.client.get_network()
        return bool(result)

    def get_network_topology(self) -> Optional[Dict]:
        """Get network topology for visualization."""
        return self.client.get_topology()

    def extract_cascor_topology(self) -> Optional[Dict]:
        """Alias for get_network_topology (backward compatibility)."""
        return self.get_network_topology()

    def get_network_data(self) -> Dict:
        """Get network state data."""
        return self.client.get_network()

    # --- Training Control (backward-compatible names) ---

    def start_training_background(self, *args, **kwargs) -> bool:
        """Start training. CasCor service handles async execution.

        Unlike CascorIntegration, this does NOT accept (x, y, epochs)
        positional args — training data is managed server-side.
        Accepts keyword args: epochs, dataset, inline_data, params.
        """
        try:
            self.client.start_training(**kwargs)
            return True
        except Exception:
            return False

    def request_training_stop(self) -> bool:
        """Request graceful training stop."""
        try:
            self.client.stop_training()
            return True
        except Exception:
            return False

    def is_training_in_progress(self) -> bool:
        """Check if training is currently running."""
        try:
            status = self.client.get_training_status()
            return status.get("is_training", False)
        except Exception:
            return False

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        try:
            return self.client.get_training_status()
        except Exception:
            return {"is_training": False, "network_connected": False}

    # --- Monitoring (replaced by WS relay) ---

    def install_monitoring_hooks(self) -> bool:
        """No-op. Monitoring is handled server-side."""
        return True

    def start_monitoring_thread(self, interval: float = 1.0) -> None:
        """No-op. Monitoring uses WS relay instead of polling thread."""
        pass

    def stop_monitoring(self) -> None:
        """No-op. WS relay handles this."""
        pass

    def restore_original_methods(self) -> None:
        """No-op. No method wrapping in service mode."""
        pass

    # --- Data ---

    def get_dataset_info(self, x=None, y=None) -> Optional[Dict[str, Any]]:
        """Get current dataset metadata from the service."""
        try:
            return self.client.get_dataset()
        except Exception:
            return None

    def get_prediction_function(self):
        """Not applicable in service mode. Use get_decision_boundary()."""
        return None

    def get_decision_boundary(self, resolution: int = 50) -> Dict[str, Any]:
        """Get decision boundary grid data."""
        return self.client.get_decision_boundary(resolution=resolution)

    # --- Metrics ---

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        return self.client.get_metrics()

    def get_metrics_history(self, count: Optional[int] = None) -> Dict[str, Any]:
        """Get training metrics history."""
        return self.client.get_metrics_history(count=count)

    # --- Remote Workers (stub no-ops) ---
    # Workers are managed server-side by CasCor + juniper-cascor-worker.
    # These methods exist only for interface compatibility.

    def connect_remote_workers(self, address=None, authkey=None) -> bool:
        """No-op. Workers managed server-side."""
        return True

    def start_remote_workers(self, num_workers: int = 1) -> bool:
        """No-op. Workers managed server-side."""
        return True

    def stop_remote_workers(self, timeout: int = 10) -> bool:
        """No-op. Workers managed server-side."""
        return True

    def disconnect_remote_workers(self) -> bool:
        """No-op. Workers managed server-side."""
        return True

    def get_remote_worker_status(self) -> Dict[str, Any]:
        """No-op. Workers managed server-side."""
        return {"active": False, "managed_by": "cascor-service"}
```

### 3.3 Key Design Decisions

1. **Constructor signature**: `(service_url, api_key)` — NOT `(cascor_url, websocket_manager, data_client)` as originally planned. The WebSocket manager is imported internally within `start_metrics_relay()` to avoid coupling the adapter's constructor to Canopy's WS infrastructure.

2. **Backward-compatible method names**: The adapter uses the same method names as `CascorIntegration` (e.g., `create_network`, `get_network_topology`, `start_training_background`, `request_training_stop`, `is_training_in_progress`). This minimizes changes needed in `main.py` route handlers.

3. **WS streaming method**: Uses `stream()` (async iterator), NOT `stream_metrics()` which does not exist in the client API. The relay loop iterates over `self.training_stream.stream()`.

4. **Remote workers**: Stub no-ops. Workers are managed server-side by the CasCor service and the standalone `juniper-cascor-worker` package. The adapter stubs exist only for interface compatibility during transition.

5. **Dataset handling**: The adapter does NOT generate datasets. Training data is either inline (passed via `start_training(inline_data=...)`) or sourced server-side by CasCor from JuniperData.

---

## 4. Three-Mode Activation Mechanism

The migration plan originally specified a binary switch: demo mode vs. service mode. The actual implementation should support **three modes** to enable a smooth transition:

### 4.1 Mode Definitions

| Mode | Trigger | Backend | When to Use |
|------|---------|---------|-------------|
| **Demo** | `CASCOR_DEMO_MODE=1` | `DemoMode` (unchanged) | Development without any backend |
| **Service** | `CASCOR_SERVICE_URL` is set | `CascorServiceAdapter` (new) | Production with CasCor as a service |
| **Legacy** | `CASCOR_BACKEND_PATH` is set | `CascorIntegration` (existing) | Transition period — direct imports |

### 4.2 Activation Priority

```
1. CASCOR_DEMO_MODE=1           → Demo mode (highest priority)
2. CASCOR_SERVICE_URL is set    → Service mode via CascorServiceAdapter
3. CASCOR_BACKEND_PATH is set   → Legacy mode via CascorIntegration
4. Neither set                  → Attempt legacy, fall back to demo
```

### 4.3 Updated main.py Activation Logic

```python
# New: three-mode activation
force_demo_mode = os.getenv("CASCOR_DEMO_MODE", "0") in ("1", "true", "True", "yes")
cascor_service_url = os.getenv("CASCOR_SERVICE_URL")
cascor_backend_path = os.getenv("CASCOR_BACKEND_PATH")

if force_demo_mode:
    # Mode 1: Demo
    backend = None
    demo_mode_active = True
    logger.info("CasCor backend: DEMO MODE (forced)")

elif cascor_service_url:
    # Mode 2: Service (new — via juniper-cascor-client)
    from backend.cascor_service_adapter import CascorServiceAdapter
    api_key = os.getenv("CASCOR_SERVICE_API_KEY")
    backend = CascorServiceAdapter(
        service_url=cascor_service_url,
        api_key=api_key,
    )
    demo_mode_active = False
    logger.info(f"CasCor backend: SERVICE MODE ({cascor_service_url})")

elif cascor_backend_path:
    # Mode 3: Legacy (existing — direct imports, transitional)
    try:
        from backend.cascor_integration import CascorIntegration
        backend = CascorIntegration(cascor_backend_path)
        demo_mode_active = False
        logger.info(f"CasCor backend: LEGACY MODE ({cascor_backend_path})")
    except FileNotFoundError:
        backend = None
        demo_mode_active = True
        logger.warning("CasCor backend path not found, falling back to DEMO MODE")

else:
    # No config — default to demo
    backend = None
    demo_mode_active = True
    logger.info("CasCor backend: DEMO MODE (no config)")
```

### 4.4 Transition Timeline

| Step | Legacy Mode | Service Mode | Notes |
|------|-------------|--------------|-------|
| Phase 4a: Implement adapter | Available | Available | Both modes work |
| Phase 4b: Validate service mode | Available | Tested in CI | Integration tests pass |
| Phase 4c: Remove legacy | **Removed** | Default | Delete `CascorIntegration`, remove `CASCOR_BACKEND_PATH` support |
| Phase 5: Split repos | N/A | Only mode | No CasCor source in Canopy's repo |

---

## 5. Implementation Steps

### Step 5.1 — Create CascorServiceAdapter

**File:** `JuniperCanopy/juniper_canopy/src/backend/cascor_service_adapter.py`

- Implement the adapter class as specified in Section 3.2
- All public methods match `CascorIntegration`'s interface names
- Remote worker methods are stub no-ops
- Monitoring methods are no-ops (WS relay replaces polling)
- Uses `JuniperCascorClient` for REST and `CascorTrainingStream` for WS

### Step 5.2 — Update main.py with Three-Mode Activation

**File:** `JuniperCanopy/juniper_canopy/src/main.py`

- Implement the three-mode activation logic from Section 4.3
- Replace all `cascor_integration` references with `backend` variable
- Both `CascorIntegration` and `CascorServiceAdapter` expose the same method names
- Add WS relay startup in the `lifespan` context manager for service mode:

```python
# In lifespan() startup
if isinstance(backend, CascorServiceAdapter):
    await backend.connect()
    await backend.start_metrics_relay()
```

- Add WS relay shutdown:

```python
# In lifespan() shutdown
if isinstance(backend, CascorServiceAdapter):
    await backend.stop_metrics_relay()
    backend.shutdown()
elif hasattr(backend, 'shutdown'):
    backend.shutdown()
```

### Step 5.3 — Update Route Handlers

Route handlers in `main.py` currently call methods like:

```python
cascor_integration.start_training_background()
cascor_integration.get_training_status()
cascor_integration.get_network_topology()
```

These become:

```python
backend.start_training_background()
backend.get_training_status()
backend.get_network_topology()
```

Because the adapter uses backward-compatible method names, most route handlers require only a variable rename (`cascor_integration` → `backend`).

**Route mapping:**

| Route | Current Call | Adapter Call | Notes |
|-------|-------------|--------------|-------|
| `/api/train/start` | `cascor_integration.start_training_background()` | `backend.start_training_background()` | Same name |
| `/api/train/stop` | `cascor_integration.request_training_stop()` | `backend.request_training_stop()` | Same name |
| `/api/train/status` | `cascor_integration.get_training_status()` | `backend.get_training_status()` | Same name |
| `/api/network/topology` | `cascor_integration.get_network_topology()` | `backend.get_network_topology()` | Same name |
| `/api/network/create` | `cascor_integration.create_network(config)` | `backend.create_network(config)` | Same name |
| `/ws/control` reset | `cascor_integration.restore_original_methods()` then `create_network` then `install_monitoring_hooks` | Same calls — adapter stubs the no-ops | Transparent |

### Step 5.4 — Update Dependencies

**File:** `JuniperCanopy/juniper_canopy/pyproject.toml`

```toml
dependencies = [
    # ... existing deps ...
    "juniper-data-client>=0.3.0",
    "juniper-cascor-client>=0.1.0",   # NEW
]
```

### Step 5.5 — Update Configuration

**New environment variables:**

| Variable | Purpose | Default |
|----------|---------|---------|
| `CASCOR_SERVICE_URL` | CasCor service URL | (none — triggers service mode when set) |
| `CASCOR_SERVICE_API_KEY` | API key for CasCor service | (none — optional) |

**Variables retained during transition:**

| Variable | Purpose | Notes |
|----------|---------|-------|
| `CASCOR_DEMO_MODE` | Force demo mode | Unchanged |
| `CASCOR_BACKEND_PATH` | Legacy direct-import path | Removed in Step 5.8 |

**Default port:** `8200` (matches `juniper-cascor-client` defaults), NOT `8060` as originally stated in the migration plan.

### Step 5.6 — Write Tests for CascorServiceAdapter

**Location:** `JuniperCanopy/juniper_canopy/src/tests/`

- **Unit tests**: Mock `JuniperCascorClient` and verify adapter delegates correctly
- **Interface compatibility tests**: Verify adapter exposes all methods that `main.py` calls
- **Three-mode activation tests**: Verify correct backend is selected for each env var combination
- **WS relay tests**: Mock `CascorTrainingStream.stream()` and verify messages are broadcast

```python
# Example: test interface compatibility
def test_adapter_has_all_required_methods():
    """Verify adapter exposes every method that main.py calls."""
    required_methods = [
        "create_network",
        "connect_to_network",
        "get_network_topology",
        "start_training_background",
        "request_training_stop",
        "is_training_in_progress",
        "get_training_status",
        "install_monitoring_hooks",
        "start_monitoring_thread",
        "stop_monitoring",
        "restore_original_methods",
        "get_dataset_info",
        "shutdown",
    ]
    adapter = CascorServiceAdapter.__new__(CascorServiceAdapter)
    for method_name in required_methods:
        assert hasattr(adapter, method_name), f"Missing method: {method_name}"
        assert callable(getattr(adapter, method_name)), f"Not callable: {method_name}"
```

### Step 5.7 — Integration Testing

- Start CasCor service locally
- Set `CASCOR_SERVICE_URL=http://localhost:8200`
- Start Canopy
- Verify: network creation, training start/stop, metrics streaming, topology retrieval
- Verify: demo mode still works with `CASCOR_DEMO_MODE=1`
- Verify: legacy mode still works with `CASCOR_BACKEND_PATH=...`

### Step 5.8 — Remove Legacy Mode (Post-Validation)

Once service mode is validated:

1. Delete `src/backend/cascor_integration.py` (~1,600 lines)
2. Remove `CASCOR_BACKEND_PATH` support from `main.py`
3. Remove legacy mode branch from three-mode activation
4. Remove `sys.path` manipulation code
5. Remove all direct CasCor imports
6. Remove tests that depend on CasCor source being on `sys.path`

The resulting `main.py` has only two modes: demo and service.

---

## 6. Method Name Mapping — Complete Reference

This table maps every `CascorIntegration` method to its `CascorServiceAdapter` equivalent:

| CascorIntegration Method | Adapter Method | Client Call | Notes |
|--------------------------|---------------|-------------|-------|
| `__init__(backend_path)` | `__init__(service_url, api_key)` | `JuniperCascorClient(base_url, api_key)` | Different constructor |
| `create_network(config)` | `create_network(config)` | `client.create_network(**config)` | Dict unpacked to kwargs |
| `connect_to_network(network)` | `connect_to_network(network)` | `client.get_network()` | Arg ignored |
| `get_network_topology()` | `get_network_topology()` | `client.get_topology()` | Same name |
| `extract_cascor_topology()` | `extract_cascor_topology()` | `client.get_topology()` | Alias |
| `get_network_data()` | `get_network_data()` | `client.get_network()` | Same name |
| `fit_async(*args, **kw)` | N/A | `client.start_training()` | Service is already async |
| `start_training_background(*args, **kw)` | `start_training_background(**kw)` | `client.start_training()` | No positional args |
| `request_training_stop()` | `request_training_stop()` | `client.stop_training()` | Same name |
| `is_training_in_progress()` | `is_training_in_progress()` | `client.get_training_status()` | Checks `is_training` key |
| `get_training_status()` | `get_training_status()` | `client.get_training_status()` | Same name |
| `install_monitoring_hooks()` | `install_monitoring_hooks()` | No-op (returns True) | Server-side |
| `start_monitoring_thread(interval)` | `start_monitoring_thread(interval)` | No-op | WS relay instead |
| `stop_monitoring()` | `stop_monitoring()` | No-op | WS relay instead |
| `restore_original_methods()` | `restore_original_methods()` | No-op | No wrapping |
| `get_dataset_info(x, y)` | `get_dataset_info(x, y)` | `client.get_dataset()` | Args ignored |
| `get_prediction_function()` | `get_prediction_function()` | Returns None | Use `get_decision_boundary()` |
| `_broadcast_message(msg)` | (internal relay) | `training_stream.stream()` → `ws_manager.broadcast()` | Automatic |
| `connect_remote_workers(...)` | `connect_remote_workers(...)` | No-op (returns True) | Server-side |
| `start_remote_workers(n)` | `start_remote_workers(n)` | No-op (returns True) | Server-side |
| `stop_remote_workers(timeout)` | `stop_remote_workers(timeout)` | No-op (returns True) | Server-side |
| `disconnect_remote_workers()` | `disconnect_remote_workers()` | No-op (returns True) | Server-side |
| `get_remote_worker_status()` | `get_remote_worker_status()` | No-op stub dict | Server-side |
| `shutdown()` | `shutdown()` | `client.close()` | Same name |
| `create_monitoring_callback(type, cb)` | N/A | Use `training_stream.on_metrics()` etc. | Different pattern |

---

## 7. WebSocket Relay Architecture

### Current Architecture (CascorIntegration)

```
[CasCor Network] --method wrapping--> [CascorIntegration callbacks]
                                           |
                                           v
                              [_broadcast_message()]
                                           |
                                           v
                              [Canopy WebSocketManager] ---> [Browser]
```

### New Architecture (CascorServiceAdapter)

```
[CasCor Service] --per-epoch hooks--> [CasCor WS Manager]
                                           |
                                      /ws/training
                                           |
                                           v
                              [CascorTrainingStream.stream()]
                                           |
                                       _relay_loop
                                           |
                                           v
                              [Canopy WebSocketManager] ---> [Browser]
```

### Message Format Compatibility

CasCor service WebSocket messages use the format: `{type, timestamp, data}`.

CascorIntegration currently broadcasts messages like:
```json
{"type": "metrics_update", "epoch": 5, "train_loss": 0.1, ...}
```

The CasCor service format wraps data in a `data` key:
```json
{"type": "metrics", "timestamp": "...", "data": {"epoch": 5, "train_loss": 0.1, ...}}
```

**Decision:** The relay loop should either:
- (a) Pass messages through as-is and update Canopy's frontend JS to handle the new format, OR
- (b) Transform messages in the relay to match the old format

**Recommendation:** Option (a) — pass through as-is. The frontend should adapt to the service format since that is the long-term contract.

---

## 8. Corrections from Original Migration Plan

| Topic | Migration Plan (Phase 4) | Corrected |
|-------|--------------------------|-----------|
| Default port | `8060` | **`8200`** (matches `juniper-cascor-client` defaults) |
| Constructor | `(cascor_url, websocket_manager, data_client)` | **`(service_url, api_key)`** — adapter imports WS manager internally |
| Method names | New names: `start_training`, `get_topology` | **Backward-compatible names** matching CascorIntegration API |
| Activation | Binary: demo vs. service | **Three-mode**: demo / service / legacy (transitional) |
| Legacy removal | Remove CascorIntegration immediately | **Keep during transition**, feature-flag with env vars |
| WS relay | `stream_metrics()` | **`stream()`** — correct client API method |
| Remote workers | `client.connect_workers()`, etc. | **Stub no-ops** — workers managed server-side |
| Config URL | `conf/app_config.yaml` backend section | **Environment variables** (`CASCOR_SERVICE_URL`) — consistent with existing patterns |

---

## 9. Deliverables Checklist

- [ ] `CascorServiceAdapter` implemented in `src/backend/cascor_service_adapter.py`
- [ ] Three-mode activation logic in `main.py`
- [ ] All route handlers work with both `CascorIntegration` and `CascorServiceAdapter`
- [ ] `juniper-cascor-client>=0.1.0` added to `pyproject.toml` dependencies
- [ ] Unit tests for `CascorServiceAdapter` (mocked client)
- [ ] Interface compatibility tests (adapter has all required methods)
- [ ] Integration test: service mode end-to-end
- [ ] Demo mode continues to work identically
- [ ] Legacy mode continues to work during transition
- [ ] WebSocket relay tested (CasCor WS → Canopy frontend)
- [ ] Environment variable documentation updated
- [ ] `POLYREPO_MIGRATION_PLAN.md` Phase 4 updated with corrections

### Post-Validation (Step 5.8)

- [ ] `CascorIntegration` removed (~1,600 lines)
- [ ] All `sys.path` manipulation code removed
- [ ] No direct imports of CasCor modules anywhere in Canopy
- [ ] `CASCOR_BACKEND_PATH` support removed
- [ ] Legacy-only tests removed

---

## 10. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| CasCor service API not complete (Phase 2 in progress) | Adapter methods fail | Three-mode activation allows fallback to legacy |
| WS message format mismatch | Frontend breaks | Document format contract; option to transform in relay |
| Training data handling changes | Training fails | Service mode passes dataset config to CasCor, not raw tensors |
| Performance regression (network hop) | Slower metrics updates | WS streaming is near-real-time; REST polling as fallback |
| juniper-cascor-worker not compatible | Remote training breaks | Worker is independent service; adapter stubs don't interfere |

---

## 11. Dependencies and Prerequisites

| Prerequisite | Status | Notes |
|--------------|--------|-------|
| Phase 2: CasCor Service API | In Progress | REST routes ~65% done; WS endpoints needed |
| Phase 3: juniper-cascor-client v0.1.0 | Complete | All client classes implemented |
| juniper-cascor-worker | Complete | Standalone package, no Canopy dependency |
| JuniperData service | Running | Required for dataset generation |
| juniper-data-client | Complete | Already used by Canopy |
