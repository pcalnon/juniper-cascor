# juniper-cascor — Histogram Bucket Rationale

**Date:** 2026-05-02 (R4.1 draft) / 2026-05-03 (R5.1b update)
**METRICS-MON sub-track:** R4.1 / seed-14, plus **R5.1b** (this PR)
**Status:** §4 and §5 layouts **implemented in R5.1b**. §2 and §3
remain **tentative pending R5.1** SLO ratification.
**Related:** [`METRICS_MONITORING_R4_ENTRY_PLAN_2026-05-01.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/code-review/METRICS_MONITORING_R4_ENTRY_PLAN_2026-05-01.md) §3 Q1 (hybrid: document current rationale now; mark tentative; R5.1 ratifies).

---

## 1. Inventory

juniper-cascor exposes **four** Prometheus histograms on the production
surface:

| Metric | Labels | Bucket constant | Purpose |
|---|---|---|---|
| `cascor_inference_duration_seconds` | _(none)_ | `_LOGGER_PROMETHEUS_LATENCY_BUCKETS` (in `cascor_constants/constants_logging/`) | Inference RPC latency for the `/v1/inference` endpoint. |
| `cascor_ws_resume_replayed_events` | _(none)_ | `_WS_RESUME_REPLAY_BUCKETS` (local) | Number of buffered events replayed when a WebSocket client successfully resumes via the seq-replay handshake. **Discrete count, not duration.** |
| `cascor_ws_broadcast_send_duration_seconds` | `type` | `_WS_SUB_MS_LATENCY_BUCKETS` (R5.1b) | Wall-clock for individual WebSocket `send_*` operations on a single client. |
| `cascor_ws_command_handler_seconds` | `command` | `_WS_SUB_MS_LATENCY_BUCKETS` (R5.1b) | Wall-clock for command-handler dispatch (e.g. `pause`, `resume`, `start_training`). |

As of **R5.1b** (this PR) all four histograms now carry explicit
buckets chosen for their distribution shape. The two WebSocket
latency histograms previously used the Prometheus default layout
(`(.005, .01, .025, .05, .075, .1, .25, .5, .75, 1, 2.5, 5, 7.5, 10,
+inf)`); they have been re-bucketed to the shared
`_WS_SUB_MS_LATENCY_BUCKETS` constant — see §4.

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

### 4.1 Bucket layout

**Implemented in R5.1b (this PR).** Adopted via the shared
`_WS_SUB_MS_LATENCY_BUCKETS` constant in `src/api/observability.py`:

```python
_WS_SUB_MS_LATENCY_BUCKETS: tuple = (
    0.0001,  # 100 µs
    0.0005,  # 500 µs
    0.001,   # 1 ms
    0.005,   # 5 ms
    0.01,    # 10 ms
    0.05,    # 50 ms
    0.1,     # 100 ms
    float("inf"),
)
```

8 buckets including `+inf`. Replaces the Prometheus default layout
(`(.005, .01, .025, .05, .075, .1, .25, .5, .75, 1, 2.5, 5, 7.5, 10,
+inf)`, 15 buckets), whose 5 ms floor was too coarse for the actual
distribution.

### 4.2 Rationale

WebSocket broadcast send operations on a single client should
typically complete in sub-millisecond on a healthy connection (the
actual write to the OS socket buffer) — sub-5 ms in pathological
cases. The previous default-bucket floor at 5 ms meant most
observations landed in the first bucket and quantile estimation in
the healthy regime was impossible. The R5.1b layout adds three
sub-millisecond boundaries (100 µs / 500 µs / 1 ms) for the healthy
regime, retains 5 ms / 10 ms as the "soft slow" markers, and uses
50 ms / 100 ms to flag genuine slowness while keeping bucket count
low (8 vs. the default's 15).

### 4.3 R5.1b implementation note

Adopted as written above. The HELP line on the metric no longer
carries the `(R4.1 buckets tentative pending R5.1)` suffix; the
inline source comment now points at this section as the rationale of
record. SLO ratification (R5.1 proper) remains pending — re-bucketing
is a metric-version event but not a SemVer-major break, so a future
ratification PR can adjust boundaries without breaking dashboards
beyond a normal metric-version bump.

---

## 5. `cascor_ws_command_handler_seconds`

### 5.1 Bucket layout

**Implemented in R5.1b (this PR).** Adopts the same
`_WS_SUB_MS_LATENCY_BUCKETS` constant defined in §4.1 — 8 boundaries
spanning 100 µs → 100 ms (+inf).

### 5.2 Rationale

Command-handler durations vary widely by `command` label:

- `pause` / `resume`: simple state flips — sub-millisecond.
- `start_training`: launches the training loop in a background thread
  — sub-100 ms.
- `update_params`: lifecycle-lock acquire + atomic-rollback path —
  sub-50 ms in nominal case.

The previous default Prometheus layout was roughly OK for the slower
commands but wasted resolution on the fast ones (everything below 5 ms
landed in the first bucket). The R5.1b sub-millisecond layout
(100 µs / 500 µs / 1 ms) gives quantile precision for the
`pause`/`resume` regime, while the 5 ms / 10 ms / 50 ms / 100 ms
boundaries continue to bracket `update_params` and `start_training`
durations. The single shared layout keeps the metric un-split and the
`command` label as the discriminator, which preserves dashboard and
alert authoring simplicity at a small cost in per-command bucket
fit.

### 5.3 R5.1b implementation note

Adopted as written. The HELP line no longer carries the
`(R4.1 buckets tentative pending R5.1)` suffix; the inline source
comment points at §4–§5 of this document as the rationale of record.
A future R5.1 SLO catalog may still split the metric per command-class
if a single layout proves insufficient — that would be a metric-rename
event, not just a re-bucket, and is out of scope for R5.1b.

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
- [x] **Broadcast send**: re-bucketed to add sub-millisecond
      resolution in **R5.1b** (this PR). New layout is the shared
      `_WS_SUB_MS_LATENCY_BUCKETS` constant — see §4.1.
- [x] **Command handler**: kept as a single histogram with the
      `command` label; adopted the same sub-millisecond layout in
      **R5.1b**. Per-command-class split deferred — see §5.3.
- [ ] Consider lifting `_LOGGER_PROMETHEUS_LATENCY_BUCKETS` into
      `juniper-observability` as a shared "general RPC latency"
      bucket constant if R5.1's SLO catalog identifies a unified
      RPC-latency scheme.

---

## 7. Process notes

- HELP-string markers: as of R5.1b, the two re-bucketed histograms
  (`cascor_ws_broadcast_send_duration_seconds` and
  `cascor_ws_command_handler_seconds`) **no longer** carry the
  "tentative pending R5.1" suffix. The remaining two
  (`juniper_cascor_inference_duration_seconds`,
  `cascor_ws_resume_replayed_events`) keep the suffix because they
  await SLO ratification (R5.1 proper), not re-bucketing. Inline
  comments at each definition point at this rationale doc.
- Re-bucketing is a metric-version event but **not** a public-API
  break. No SemVer-major beat is required when R5.1 ratifies or
  reshapes.
- R5.1b status (`cascor_ws_broadcast_send_duration_seconds` and
  `cascor_ws_command_handler_seconds`): **re-bucketed.** Both metrics
  now share the `_WS_SUB_MS_LATENCY_BUCKETS` layout defined in
  `src/api/observability.py`; default Prometheus buckets removed.
