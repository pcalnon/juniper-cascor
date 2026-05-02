# juniper-cascor — Histogram Bucket Rationale

**Date:** 2026-05-02
**METRICS-MON sub-track:** R4.1 / seed-14
**Status:** Initial draft — bucket layouts marked **tentative pending R5.1**.
**Related:** [`METRICS_MONITORING_R4_ENTRY_PLAN_2026-05-01.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/code-review/METRICS_MONITORING_R4_ENTRY_PLAN_2026-05-01.md) §3 Q1 (hybrid: document current rationale now; mark tentative; R5.1 ratifies).

---

## 1. Inventory

juniper-cascor exposes **four** Prometheus histograms on the production
surface:

| Metric | Labels | Bucket constant | Purpose |
|---|---|---|---|
| `cascor_inference_duration_seconds` | _(none)_ | `_LOGGER_PROMETHEUS_LATENCY_BUCKETS` (in `cascor_constants/constants_logging/`) | Inference RPC latency for the `/v1/inference` endpoint. |
| `cascor_ws_resume_replayed_events` | _(none)_ | `_WS_RESUME_REPLAY_BUCKETS` (local) | Number of buffered events replayed when a WebSocket client successfully resumes via the seq-replay handshake. **Discrete count, not duration.** |
| `cascor_ws_broadcast_send_duration_seconds` | `type` | _Prometheus default_ | Wall-clock for individual WebSocket `send_*` operations on a single client. |
| `cascor_ws_command_handler_seconds` | `command` | _Prometheus default_ | Wall-clock for command-handler dispatch (e.g. `pause`, `resume`, `start_training`). |

Two histograms have **explicit buckets** chosen for their distribution
shape; two use **Prometheus defaults** (`(.005, .01, .025, .05, .075,
.1, .25, .5, .75, 1, 2.5, 5, 7.5, 10, +inf)`) and are flagged for
re-evaluation in R5.1.

---

## 2. `cascor_inference_duration_seconds`

### 2.1 Current bucket layout

```python
_LOGGER_PROMETHEUS_LATENCY_BUCKETS: tuple = (
    0.001,   # 1 ms
    0.005,   # 5 ms
    0.01,    # 10 ms
    0.05,    # 50 ms
    0.1,     # 100 ms
    0.5,     # 500 ms
    1.0,     # 1 s
    2.5,     # 2.5 s
    5.0,     # 5 s
    float("inf"),
)
```

10 buckets including `+inf`. Spans 4 orders of magnitude (1 ms → 5 s).

### 2.2 Rationale per boundary

| Boundary | What it discriminates | SLO target served | R5.1 status |
|---|---|---|---|
| **0.001 s (1 ms)** | Hot-cache inference on a small network. Sub-millisecond is achievable on small CasCor networks with cached input tensors. | None directly. Useful for the "ideal" floor. | **Tentative.** |
| **0.005 s (5 ms)** | Typical small-network inference (CasCor with ~10 hidden units). | **Candidate** for "p50 inference latency < 5 ms" SLO. | **Tentative — high confidence.** |
| **0.01 s (10 ms)** | Single-frame budget at 100 Hz. Above this, real-time inference clients (e.g. live classification demos) start to feel sluggish. | **Strong candidate** for "p95 inference latency < 10 ms" SLO. | **Tentative — high confidence.** |
| **0.05 s (50 ms)** | Display-frame budget at 60 Hz × ~3 frames. Above this, a client polling at the display refresh rate will see drops. | **Candidate** for "p99 inference latency < 50 ms" SLO. | **Tentative — moderate confidence.** |
| **0.1 s (100 ms)** | Human-perceptible interaction-lag threshold. | Useful for capacity-planning queries. | **Tentative.** |
| **0.5 s (500 ms)** | "Sluggish" territory — likely indicates a large network or upstream contention. | Useful for alerting trends. | **Tentative.** |
| **1.0 s (1 s)** | Pathological for inference. CasCor inference should never legitimately take a full second on the 32-hidden-unit nominal network. | Alert threshold candidate. | **Tentative.** |
| **2.5 s / 5.0 s** | Filler for the long-tail; rarely populated. | None directly. | **Tentative.** May be removable if R5.1 doesn't reference them. |
| **+inf** | Mandatory upper bound. | — | Required. |

### 2.3 Cross-reference

`_LOGGER_PROMETHEUS_LATENCY_BUCKETS` is canopy- and data-side reusable
in principle (same layout serves any "RPC latency" use case), but
neither currently imports it. R5.1 may consider lifting it into
`juniper-observability` as a shared default if the SLO catalog
identifies a unified RPC-latency bucket scheme.

---

## 3. `cascor_ws_resume_replayed_events`

### 3.1 Current bucket layout

```python
_WS_RESUME_REPLAY_BUCKETS = (0, 1, 5, 25, 100, 500, 1024)
```

7 buckets (Prometheus appends `+inf`). Discrete counts, not durations.

### 3.2 Rationale per boundary

The histogram tracks the **count** of buffered events replayed during
a successful WebSocket resume — not a latency. Bucket boundaries map
to operational regimes of the replay buffer (fixed at
`Settings().ws_replay_buffer_size = 1024` per R3.5).

| Boundary | What it discriminates | SLO target served | R5.1 status |
|---|---|---|---|
| **0** | Resume that found nothing to replay (client was caught up). Healthy: client disconnected briefly during a quiet window. | Frequency of "trivial resumes" — capacity signal. | **Tentative.** |
| **1** | Single missed event. Likely a network blip during a single broadcast. | Useful for distinguishing "blip" from "outage". | **Tentative.** |
| **5** | Short outage (~5 broadcasts at 1 Hz state-broadcast tick = ~5s of state buffering). | Operational signal. | **Tentative.** |
| **25** | ~25 s of buffered state. Client likely reconnected after a network reload or browser tab refresh. | Useful for resume-success-rate SLO context. | **Tentative.** |
| **100** | ~100 s of buffering. Client missed a non-trivial training segment. | Indicates client disconnected for >1 min. | **Tentative.** |
| **500** | Half the buffer capacity. Approaching the limit where eviction starts; replay is still complete but the client has been gone long enough to trigger a "did they really come back?" review. | **Candidate** for alert threshold ("rate of resume_replayed_events > 500"). | **Tentative — moderate confidence.** |
| **1024** | Buffer-full marker. Resumes that hit this boundary received the maximum-possible replay; older events were evicted (the "replay complete" guarantee weakens). | Strong alerting signal — clients reconnecting at this depth need investigation. | **Tentative — high confidence.** |
| **+inf** | Mandatory upper bound; clipped at 1024 in practice (R3.5 capacity guard). | — | Required. |

### 3.3 Trade-off

Boundaries chosen as roughly logarithmic to match the operational
regimes (blip / brief / extended / near-full). 7 buckets is sparse but
appropriate for a count metric where bucket-resolution beats
quantile-precision for alerting use cases.

---

## 4. `cascor_ws_broadcast_send_duration_seconds`

### 4.1 Current bucket layout

**Default Prometheus buckets** (no explicit `buckets=` kwarg):

```
(.005, .01, .025, .05, .075, .1, .25, .5, .75, 1, 2.5, 5, 7.5, 10, +inf)
```

15 buckets. Designed as a "general HTTP" default — likely too sparse
in the sub-10 ms region where WS broadcast send durations actually
sit.

### 4.2 Rationale

**No explicit rationale in current source.** WebSocket broadcast send
operations on a single client should typically complete in
sub-millisecond on a healthy connection (the actual write to the OS
socket buffer) — sub-5 ms in pathological cases. The default bucket
floor at 5 ms means most observations land in the first bucket and
quantile estimation in the healthy regime is impossible.

### 4.3 R5.1 ratification candidate

R5.1 should re-bucket this to bracket the actual distribution:

```python
# Proposed; not adopted in this PR (R5.1 territory):
buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, float("inf"))
```

Sub-millisecond resolution for the healthy regime, 5 ms / 10 ms /
50 ms / 100 ms boundaries to flag genuine slowness. **Status: tentative;
re-bucket in R5.1.**

---

## 5. `cascor_ws_command_handler_seconds`

### 5.1 Current bucket layout

**Default Prometheus buckets** (same as §4.1).

### 5.2 Rationale

**No explicit rationale in current source.** Command-handler durations
vary widely by `command` label:

- `pause` / `resume`: simple state flips — sub-millisecond.
- `start_training`: launches the training loop in a background thread
  — sub-100 ms.
- `update_params`: lifecycle-lock acquire + atomic-rollback path —
  sub-50 ms in nominal case.

Defaults are roughly OK for the slower commands but waste resolution
on the fast ones. R5.1 may consider splitting the metric per command
class or adopting the per-metric override approach
(`buckets_for_command={"pause": ..., "start_training": ...}`).

### 5.3 R5.1 ratification candidate

Status: tentative; re-evaluate in R5.1 once SLO catalog assigns
per-command latency budgets.

---

## 6. R5.1 ratification queue

When R5.1 designs the cascor SLO catalog:

- [ ] **Inference**: decide whether p95 < 10 ms is the load-bearing
      SLO. If yes, retain 0.005 / 0.01 / 0.05 boundaries with
      confidence. Consider removing 2.5 s / 5 s if no SLO references
      them.
- [ ] **Resume replay**: pick one of the operational thresholds (25,
      100, 500) as the "client-degraded" signal and align with the
      cascor-canopy reconnection alert.
- [ ] **Broadcast send**: re-bucket to add sub-millisecond resolution.
      The default Prometheus layout is wrong for this distribution.
- [ ] **Command handler**: decide between (a) one histogram with the
      `command` label and proposed sub-millisecond buckets, or
      (b) split into per-command-class metrics for clearer SLO
      mapping.
- [ ] Consider lifting `_LOGGER_PROMETHEUS_LATENCY_BUCKETS` into
      `juniper-observability` as a shared "general RPC latency"
      bucket constant if R5.1's SLO catalog identifies a unified
      RPC-latency scheme.

---

## 7. Process notes

- HELP-string markers: all 4 histograms carry a "tentative pending
  R5.1" suffix on their HELP lines so operators reading `/metrics`
  directly see the marker. Inline comments at each definition point
  at this rationale doc.
- Re-bucketing is a metric-version event but **not** a public-API
  break. No SemVer-major beat is required when R5.1 ratifies or
  reshapes.
- `cascor_ws_broadcast_send_duration_seconds` and
  `cascor_ws_command_handler_seconds` are flagged as
  **likely-needs-re-bucketing** (default Prometheus layout doesn't
  match their actual distribution); R5.1 should prioritize them.
