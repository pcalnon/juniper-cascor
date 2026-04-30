# juniper-cascor-protocol

Cross-process WebSocket frame schemas for the [juniper-cascor](https://github.com/pcalnon/juniper-cascor) ecosystem.

This package is the canonical, single-sourced wire-protocol surface for the three Juniper WebSocket endpoints:

| Endpoint | Schema source |
|---|---|
| `/ws/training` (server → client broadcast) | `juniper_cascor_protocol.envelope` (Pydantic v2) |
| `/ws/control` (bidirectional command/response) | `juniper_cascor_protocol.envelope` (Pydantic v2) |
| `/ws/v1/workers` (cascor ↔ worker JSON + binary side-channel) | `juniper_cascor_protocol.worker` (StrEnum + numpy `BinaryFrame`) |

## Why two subpackages?

The envelope subpackage uses Pydantic v2 for declarative validation of JSON-only frames.
The worker subpackage stays Pydantic-free so [`juniper-cascor-worker`](https://github.com/pcalnon/juniper-cascor-worker)
can adopt the canonical type enum and binary codec without pulling Pydantic into its slim image —
preserving the [METRICS-MON R2 exit-gate decision](https://github.com/pcalnon/juniper-ml/blob/main/notes/code-review/METRICS_MONITORING_R2_EXIT_GATE_WORKER_ADOPTION_2026-04-29.md).
Importing `juniper_cascor_protocol.worker` does **not** load Pydantic (verified by the test suite).

## Install

```bash
pip install juniper-cascor-protocol
```

Pinned by the cascor server, [juniper-cascor-client](https://github.com/pcalnon/juniper-cascor-client), [juniper-canopy](https://github.com/pcalnon/juniper-canopy), and [juniper-cascor-worker](https://github.com/pcalnon/juniper-cascor-worker) — each consumer imports only the subpackage it needs.

## Quick start

### Validating an inbound `/ws/training` frame

```python
import json
from juniper_cascor_protocol.envelope import (
    UnknownEnvelope,
    MetricsEnvelope,
    validate_envelope,
)

raw = ws.recv()                         # ``websockets`` text frame
frame = json.loads(raw)
envelope = validate_envelope(frame)     # never raises on bad input

if isinstance(envelope, MetricsEnvelope):
    handle_metrics(envelope.data)
elif isinstance(envelope, UnknownEnvelope):
    # ``envelope.type`` is the cardinality-bounded label
    # (collapses to ``"_unmatched"`` after N distinct unknowns).
    log_unrecognized_frame(envelope.type)
```

`validate_envelope` never raises on schema mismatch — chaos-testing of malformed frames is part of the package's contract so a misbehaving server cannot crash a consumer.

### Worker-side `BinaryFrame` codec

```python
import numpy as np
from juniper_cascor_protocol.worker import BinaryFrame, WorkerMessageType

# Send tensor as a binary side-channel frame
weights = np.zeros((128, 128), dtype="float32")
ws.send(BinaryFrame.encode(weights))

# Receive and decode
raw = await ws.recv()           # bytes
arr = BinaryFrame.decode(raw)   # owned numpy array

# Dispatch JSON envelope by canonical message type
msg = json.loads(text_frame)
if msg.get("type") == WorkerMessageType.TASK_ASSIGN:
    ...
```

## Cardinality-bound details

The `validate_envelope` helper tracks distinct unknown `type` strings up to `UNKNOWN_TYPE_BUDGET = 16` per process; subsequent unknowns return as `UNMATCHED_TYPE_LABEL = "_unmatched"`. This mirrors the [`juniper_observability.UNMATCHED_ENDPOINT_LABEL`](https://pypi.org/project/juniper-observability/) strategy used for HTTP cardinality bounds. Tests reset state via `reset_unknown_label_state()` in `juniper_cascor_protocol.envelope`.

## Versioning

Follows PEP 440 + [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Consumers should pin `juniper-cascor-protocol>=A.B,<A+1`. Additive envelope fields are minor bumps; field removals/renames are major bumps.

See [`CHANGELOG.md`](./CHANGELOG.md) for the per-version contract.

## License

MIT — see [LICENSE](https://github.com/pcalnon/juniper-cascor/blob/main/LICENSE).
