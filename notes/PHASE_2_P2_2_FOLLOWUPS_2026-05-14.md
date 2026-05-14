# P2-2 Follow-ups: real-time broadcast + history fetch route

**Status**: Captured 2026-05-14 during the P2-2 design check-in. Not yet implemented.
**Parent PR**: P2-2 (`phase2/p2-2-dataset-swap-history`) — landed only the history persistence + serializer round-trip.

This doc records two ideas surfaced during the P2-2 scoping discussion that were intentionally deferred to keep the P2-2 PR focused on persistence. Each is a small, independent follow-up that can land any time after P2-2 merges.

---

## Follow-up A: real-time WebSocket broadcast of `dataset_swap` events

### What

Push the `dataset_swap` event over the existing cascor WebSocket immediately on swap completion, so canopy can render a timeline marker without polling or fetching a snapshot.

### Why deferred from P2-2

History events are currently HDF5-only — no other history surface broadcasts in real time today (the agent investigation 2026-05-14 confirmed `train_loss` / `value_loss` / `train_accuracy` / `value_accuracy` / `hidden_units_added` all persist to HDF5 but do not broadcast). Adding broadcast for `dataset_swap` alone would create asymmetry — the right design conversation is "should all history events broadcast?", which is bigger than P2-2.

Until this follows up, canopy P2-7 will pick up swap events via the snapshot/history GET path.

### Sketch

- In `swap_dataset_live` step 16 (after `record_dataset_swap_event` returns the event dict, in cascor `manager.py`), call `self._ws_manager.broadcast(create_dataset_swap_message(event))`.
- New helper `create_dataset_swap_message(event)` in the same module as `create_event_message` / `create_cascade_add_message` (cascade_correlation.py around line 1281 per the agent's investigation — confirm before implementing).
- Wire-format: `{"type": "dataset_swap", "data": {...event payload...}}` mirroring `create_cascade_add_message`'s envelope.

### Tests

- Lifecycle: mock `_ws_manager.broadcast`, run a successful swap, assert the call fires exactly once with the dataset_swap envelope.
- No broadcast on cancel/rollback (mirror the persistence test).
- Backpressure: if `_ws_manager` is `None` (no canopy connected), the call no-ops cleanly.

### Effort

~50–100 LOC. Small follow-up PR.

---

## Follow-up B: `GET /v1/history/dataset_swaps` route

### What

REST route that returns the current network's `history["dataset_swaps"]` list as JSON, without requiring callers to fetch a full snapshot.

### Why deferred from P2-2

P2-2 records the events into history; the snapshot serializer makes them durable. The route is purely for convenience — canopy P2-7's "Replay UI swap markers" feature wants this kind of access pattern. But until P2-7 is being implemented, we don't know whether a dedicated GET route is the right shape or whether the events would be better exposed via an existing history-fetch endpoint or via the WebSocket stream (Follow-up A above).

The PR series document at `juniper-canopy/notes/ISSUE_3_PHASE_2_LIVE_DATASET_SWAP_2026-05-09.md` §7 places P2-7 last in the chain. By the time canopy P2-7 is being written, this follow-up's shape will be clear.

### Sketch

- New route in `src/api/routes/history.py` (or wherever history GETs live today — verify before implementing):
  ```text
  GET /v1/history/dataset_swaps        — returns the full list
  GET /v1/history/dataset_swaps?since=ISO8601  — events strictly after a timestamp (optional)
  ```
- Response shape: `{"status": "success", "data": {"events": [...]}}` using the existing `success_response` envelope.
- No write surface — events are append-only via the swap-completion path.

### Tests

- Empty history → empty list.
- Single swap completed → one event in response.
- Multiple swaps → events in chronological order.
- `?since=` filter works (later events only).
- Route honours the same auth scope as other history endpoints (verify the existing convention).

### Effort

~100–150 LOC. Small follow-up PR.

---

## Pickup order

Either follow-up can land independently of the other. Suggested order if both are needed:

1. **Follow-up B (GET route)** first — gives canopy a fetch path immediately.
2. **Follow-up A (broadcast)** second — adds the push path. By the time we add broadcast, the consumer shape will be clearer from B.

If only one is needed (e.g. P2-7 ends up only needing the fetch path), Follow-up A can stay deferred indefinitely without blocking anything.
