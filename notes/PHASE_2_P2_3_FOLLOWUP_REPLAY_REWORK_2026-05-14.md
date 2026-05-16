# P2-3 follow-up: event-sourced replay rework

**Status**: Captured 2026-05-14 during P2-3 implementation. Not yet scheduled.
**Parent PR**: P2-3 (`phase2/p2-3-swap-snapshots-replay`).

P2-3's parent spec ([`juniper-canopy/notes/ISSUE_3_PHASE_2_LIVE_DATASET_SWAP_2026-05-09.md`](../../juniper-canopy/notes/ISSUE_3_PHASE_2_LIVE_DATASET_SWAP_2026-05-09.md) §3.9 / §7) calls for a *"Replay reconstruction handler (instantaneous transformation, per §8 Answer 3)"* — implying that when replay encounters a `dataset_swap` event mid-playback, a handler fires and transforms the network's dims instantaneously.

The 2026-05-14 P2-3 implementation investigation revealed that **`_ReplaySession` is a passive time-indexed metric playback engine**, not an event-sourced replay engine. The "mid-playback handler" pattern cannot be implemented on top of the current engine without substantial architectural rework. This document captures (a) why P2-3 shipped without the mid-playback handler, (b) the canopy-orchestrated alternative P2-3 enables, and (c) the rework spec to revisit if the alternative turns out insufficient.

---

## 1. Current replay engine (`_ReplaySession`)

Defined at `src/api/lifecycle/manager.py:766`.

Flow on `start_replay(snapshot_id)`:

1. Load the target HDF5 snapshot via `_load_snapshot_to_network()`. The deserializer reconstructs the entire network state — weights, topology, history arrays — exactly as it was at the moment the snapshot was taken.
2. The replay session captures `network.history` (metric arrays) and `network.weight_history` (CAN-015g per-sample weights).
3. A background thread (`_run`) advances a `time_index` integer at the configured playback speed and emits synthetic `epoch_end` events frame-by-frame via `_emit_frame(index)`.

The session **does not process individual history events**. There is no event-dispatch table. The `_run` loop is time-indexed, not event-indexed. Topology is frozen at snapshot-load time — there is no live network being mutated.

This means there is no place to plug in a per-event handler for `dataset_swap` that would mutate the replay's network state at the right moment.

## 2. What P2-3 actually shipped

P2-3 ships only the snapshot infrastructure:

- Pre-swap auto-snap in `swap_dataset_live`, capturing the network state immediately before `_resize_network_for_dataset` mutates anything.
- Post-swap auto-snap after the new training future is submitted.
- The captured snapshot IDs are threaded into `record_dataset_swap_event(pre_swap_snapshot_id=…, post_swap_snapshot_id=…)`, so the `dataset_swap` history event (from P2-2) now carries both IDs instead of the `None` placeholders.

The snapshot IDs round-trip through `_save_training_history` / `_load_training_history` (the P2-2 serializer plumbing handles them), so a replay session that loaded any post-swap snapshot can read those IDs out of its own history.

## 3. Canopy-orchestrated transitions (the P2-7 path)

Without an event-replay engine, **canopy orchestrates topology transitions at the snapshot boundary**, not in cascor's replay loop. The intended P2-7 UX:

1. When the user starts replay from a snapshot whose `history["dataset_swaps"]` is non-empty, canopy reads each event's timestamp and renders a **timeline marker** at the corresponding playback frame.
2. As the playback cursor approaches a marker, the timeline UI offers (or auto-fires) a *"continue past swap"* action.
3. The action calls `POST /v1/snapshots/{post_swap_snapshot_id}/replay/control` with `action="play"` — effectively restarting replay from the post-swap snapshot.
4. The user sees the topology "change instantaneously" at the marker — achieved by snapshot reload, not by mid-playback mutation. The visible effect matches the spec's *"instantaneous transformation"* wording even though the mechanism differs.

The §8 Answer 3 *"instantaneous transformation"* requirement is satisfied in user-visible terms. The mechanism is a property of the orchestration layer (canopy), not the playback engine (cascor).

## 4. Trigger conditions for picking up the rework

This follow-up should be revisited if any of the following surface during P2-7 or later canopy work:

* **Snapshot-reload feels jarring.** Latency of `POST /replay/control` + snapshot load + replay restart visibly stutters the playback. If the cumulative delay exceeds ~500 ms on a representative network the user experience suffers; an event-sourced engine that mutates a live replay network in place would be smoother.

* **Multi-swap-in-quick-succession.** A snapshot every swap means a replay run that crosses N swap boundaries does N snapshot reloads. For sustained interactive use (e.g., scrubbing back and forth across the timeline) this could compound.

* **Animated topology transitions.** If P2-7's UX evolves to animate the topology change (e.g., show new input/output nodes fading in over a few frames), the snapshot-reload model can't produce intermediate states. An event-sourced replay engine could call `network._resize_network_for_dataset` and interpolate.

* **Mid-replay editing.** A "what if" UX (replay this run but with a different swap dim) would need event-sourced playback so the user's hypothetical change can be applied mid-stream. Currently out of scope; flagged here because it's a natural extension that the snapshot model can't serve.

* **Replay-only snapshot mode.** If snapshot writes become a noticeable training-loop overhead and the team decides to disable auto-snap on swap, the snapshot-orchestration UX dies entirely — at that point we'd need event-replay or some other reconstruction path.

If none of the above surface, this follow-up may stay deferred indefinitely without harm.

## 5. Rework specification (to be applied when triggered)

### 5.1 Engine changes

Convert `_ReplaySession` from passive metric playback to event-sourced playback:

* Add an `_event_queue: list[dict]` field, populated at session start by merging:
  * Per-epoch metric samples (synthesised from `network.history["train_loss"]` etc., one event per index — current behaviour).
  * Cascade-add events from `network.history["hidden_units_added"]`.
  * Dataset-swap events from `network.history["dataset_swaps"]`.
  * Any future event-with-payload history keys.
* Order the queue by event timestamp (or by epoch index where timestamps aren't available — needs a backward-compat strategy for old snapshots without per-event timestamps).
* Replace the `_run` time-index loop with an event-dispatch loop that pops the next event, sleeps until its scheduled time, then dispatches to a per-type handler.

### 5.2 Per-event handlers

* `epoch_end` — current `_emit_frame` behaviour. Broadcast metric tick.
* `cascade_add` — already pre-baked into the loaded snapshot's topology; the handler just broadcasts a `cascade_add_message` for canopy to update its render.
* `dataset_swap` — call `network._resize_network_for_dataset(...)` using the event's `arch_changes` to apply the grow side, then set `network.active_output_dim` from the event payload to apply the shrink-via-pad side. Broadcast a `dataset_swap_message`.

### 5.3 Backward compatibility

Snapshots written before the rework have neither per-event timestamps nor a guarantee that `dataset_swaps` exists. The rework must:

* Treat absent `dataset_swaps` as the empty list (already covered by P2-2's load path).
* Synthesise event timestamps from epoch indices when missing.
* Provide a feature flag (env var or settings) to fall back to the passive playback engine for snapshots that don't carry enough event metadata.

### 5.4 Estimated cost

* `_ReplaySession` rewrite: 200–400 LOC.
* Per-event handlers: 50–150 LOC each.
* Tests covering each handler + ordering + backward compat: 300–500 LOC.
* Total: ~1000–1500 LOC, likely 1–2 weeks including review and integration with P2-7.

A standalone design conversation should happen first to lock in the event-queue shape, handler protocol, and backward-compat strategy before code lands.

## 6. Files touched (when triggered)

* `src/api/lifecycle/manager.py` — `_ReplaySession` class (mostly rewrite), `start_replay`, `replay_control`.
* `src/cascade_correlation/cascade_correlation.py` — possible new methods if `_resize_network_for_dataset` needs a "for replay" variant that skips bookkeeping side effects (broadcast, history recording, etc.).
* `src/snapshots/snapshot_serializer.py` — possibly extending event metadata if backward-compat timestamp synthesis needs help.
* New: `src/tests/integration/api/test_replay_event_sourced.py` for the dispatch loop + handler tests.
