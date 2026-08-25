"""Training lifecycle manager — central coordinator for CasCor training.

Wraps CascadeCorrelationNetwork with:
- Thread-safe training via ThreadPoolExecutor
- State machine for deterministic control flow
- Monitoring hooks for real-time metrics
- Topology and statistics extraction
"""

import copy
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from api.lifecycle.classification_metrics import compute_scalar_classification_metrics
from api.lifecycle.monitor import TrainingMonitor, TrainingState
from api.lifecycle.state_machine import Command, TrainingPhase, TrainingStateMachine
from api.models.cascor_model import CascorModel
from api.models.common import coerce_native_scalars as _common_coerce_native_scalars
from api.observability import TRAINING_SESSION_STATUS_CANCELLED, TRAINING_SESSION_STATUS_FAILURE, TRAINING_SESSION_STATUS_SUCCESS, dec_training_sessions, inc_training_session_completed, inc_training_sessions, observe_training_step_duration, record_training_epoch, set_hidden_units, set_training_accuracy, set_training_loss
from cascor_constants.constants_api import _PROJECT_API_DRAIN_THREAD_JOIN_TIMEOUT, _PROJECT_API_LIFECYCLE_DEFAULT_CANDIDATE_PATIENCE, _PROJECT_API_NETWORK_INPUT_SIZE_DEFAULT, _PROJECT_API_NETWORK_OUTPUT_SIZE_DEFAULT, _PROJECT_API_PROGRESS_QUEUE_GET_TIMEOUT, _PROJECT_API_PROGRESS_QUEUE_WAIT_TIMEOUT
from snapshots.snapshot_load_status import SnapshotLoadResult
from snapshots.snapshot_load_status import absent as snapshot_absent

# How long ``shutdown()`` waits for a live training thread to observe ``_stop_event`` and
# unwind through ``fit``'s ``finally`` (which releases the candidate-worker pool and unlinks
# the run's SharedMemory blocks) before releasing those resources from the shutdown thread
# itself. Budgeted so the whole lifespan shutdown stays inside the shortest stop-tool grace
# window (10 s: experiment_stack.bash / docker stop) even when the join times out and the
# explicit pool release then spends its own ``_WORKER_SHUTDOWN_GRACE_SECONDS`` (5 s) + 1.5 s
# escalation: 3 + 6.5 < 10. The interrupt itself lands within ~25 output epochs
# (milliseconds); only a stop that arrives mid-candidate-round runs the clock.
_SHUTDOWN_TRAINING_JOIN_TIMEOUT_SECONDS = 3.0


def _env_flag(name: str, *, default: bool) -> bool:
    """Parse a boolean environment variable (``1/0``, ``true/false``, ``yes/no``,
    ``on/off``; case-insensitive). Returns ``default`` when ``name`` is unset or
    blank. Used for the C7 ``JUNIPER_CASCOR_EVAL_METRICS_ENABLED`` toggle."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _read_optimizer_type(network: Any) -> str:
    """CAN-010 / ENH-006 (A-2): read ``optimizer_type`` through the nested
    ``config.optimizer_config`` path. Falls back to ``"Adam"`` if the chain
    is missing — same default as ``OptimizerConfig`` itself."""
    config = getattr(network, "config", None)
    optimizer_config = getattr(config, "optimizer_config", None) if config is not None else None
    return getattr(optimizer_config, "optimizer_type", "Adam") if optimizer_config is not None else "Adam"


def _write_optimizer_type(network: Any, value: str) -> None:
    """CAN-010 / ENH-006 (A-2): set ``optimizer_type`` through the nested
    ``config.optimizer_config`` path. Used by ``update_params`` so the
    setattr-on-network pattern in ``updatable_keys`` works for this nested
    field. Raises if the chain is missing — matches the contract of the
    other setters."""
    network.config.optimizer_config.optimizer_type = value


def _read_activation_function_name(network: Any) -> str:
    """CAN-011 (A-3): read ``activation_function_name`` from the network.
    Falls back to ``"Tanh"`` (matches ``_init_activation_function``'s
    fallback in ``cascade_correlation.py``) when the attribute is missing."""
    return getattr(network, "activation_function_name", "Tanh") or "Tanh"


def _write_activation_function_name(network: Any, value: str) -> None:
    """CAN-011 (A-3): swap ``activation_function_name`` and re-run
    ``_init_activation_function`` so ``activation_fn`` / ``activation_fn_no_diff``
    pick up the new mapping from the registry. Without the re-init the
    surface attribute would change but the network would keep computing the
    old activation. Existing cascaded units retain whatever activation they
    were trained with — this only affects future cascade growth and the
    output-layer activation chain."""
    network.config.activation_function_name = value
    network._init_activation_function()


# ``_coerce_native_scalars`` previously lived here as a private helper;
# it's been promoted to ``api.models.common.coerce_native_scalars`` and
# applied at the response envelope so every route is covered (not just
# routes that thread through ``get_training_params``). The local alias
# is kept so existing internal callers don't have to be edited in this
# PR — they can move to the public name as a follow-up.
_coerce_native_scalars = _common_coerce_native_scalars


class InvalidCandidatePoolError(ValueError):
    """Raised when a candidate-pool PATCH violates the §1.5 C2.1 invariant triple.

    A subclass of ``ValueError`` so existing ``except ValueError`` handlers in
    ``update_params`` and the PATCH route still see it; the PATCH route promotes
    instances of this specific subclass to HTTP 422 (the message is the
    human-readable violation string) while bare ``ValueError`` remains 404.
    See ``FRONTEND_ISSUES_PLAN_2026-05-09.md §1.5 C2.1`` for the full truth table.
    """


class SwapInProgressError(RuntimeError):
    """Raised by ``swap_dataset_live`` when a concurrent swap is already underway.

    Promotes to HTTP 409 Conflict in the route. Implements the §3.7 guardrail
    #3 idempotency contract: subsequent swap requests received while a swap
    is in flight are rejected rather than queued, so the user gets a clear
    "already swapping" signal instead of mystery serialisation.
    """


class NoSwapInProgressError(RuntimeError):
    """Raised by ``request_swap_cancel`` when no swap is currently in flight.

    Promotes to HTTP 404 Not Found on ``DELETE /v1/training/dataset/live`` — the
    resource (an in-flight swap) being cancelled does not exist. Distinct from
    409 (resource exists, conflicting state) so the canopy "Cancel" affordance
    can distinguish "nothing to cancel" from "swap already finished racing you".
    """


class SwapCancelledError(RuntimeError):
    """Raised by ``swap_dataset_live`` when a concurrent cancel was honoured.

    The route translates this to HTTP 200 with ``{"status": "cancelled"}`` —
    the swap operation completed with cancellation as its terminal state, the
    pre-swap snapshot has been restored, and training continues on the OLD
    dataset. From the caller's perspective the request was handled cleanly;
    distinguishing it from a generic 5xx prevents the canopy "Live Switch
    failed" toast from firing on a user-initiated cancel.
    """


class NoMetricsUndoError(RuntimeError):
    """Raised by ``undo_clear_metrics`` when there is no clear to undo.

    C5 (Q4 use-case 1): an explicit ``clear_metrics_with_undo`` stashes the
    cleared rows so the clear can be reversed until the next run starts.
    Requesting an undo when nothing was cleared — or after a training run has
    already started (which finalizes the clear and drops the snapshot) —
    raises this, which the ``POST /v1/training/metrics/clear/undo`` route
    promotes to HTTP 409 Conflict (the resource — a reversible clear — no
    longer exists in a reversible state).
    """


class _PreSwapSnapshot:  # noqa: D101 - frozen container, not part of the public surface
    __slots__ = (
        "train_x",
        "train_y",
        "val_x",
        "val_y",
        "state_dict",
        "input_size",
        "output_size",
        "dataset_config",
        "active_output_dim",
        "output_weights",
        "output_bias",
        "hidden_unit_weights",
    )

    def __init__(
        self,
        train_x,
        train_y,
        val_x,
        val_y,
        state_dict,
        input_size,
        output_size,
        dataset_config,
        active_output_dim=None,
        output_weights=None,
        output_bias=None,
        hidden_unit_weights=None,
    ):
        # Plain container for the §3.7 guardrail-#1 pre-swap state. Tensor
        # references only — we don't .clone() since the swap path immediately
        # rebinds self._train_x to new tensors, so the old refs remain alive
        # via this snapshot until rollback or successful return. ``state_dict``
        # IS deep-copied by the caller (it shares storage with live network
        # parameters that would otherwise mutate as training resumes).
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.state_dict = state_dict
        self.input_size = input_size
        self.output_size = output_size
        self.dataset_config = dataset_config
        # P2-1d: the loss-mask dim must also rollback so an aborted shrink
        # doesn't leave training masking against a stale shorter dim.
        self.active_output_dim = active_output_dim
        # P2-1d: the network's parameter tensors are mutated IN PLACE by
        # ``_resize_network_for_dataset``. The CascadeCorrelationNetwork has
        # no ``state_dict()`` (it doesn't inherit from nn.Module), so we
        # capture tensor clones here and restore them on rollback. Each is
        # a fresh clone (detached, separate storage) so the live training
        # path can't accidentally mutate the rollback view.
        self.output_weights = output_weights
        self.output_bias = output_bias
        # List of per-hidden-unit weight tensor clones, in the same order
        # as ``network.hidden_units``. Hidden-unit biases are scalars and
        # are not mutated by P2-1d (the resize path never touches them),
        # so they don't need snapshotting.
        self.hidden_unit_weights = hidden_unit_weights


class TrainingInterrupted(Exception):
    """Sentinel raised from training-loop callbacks when the user requests stop.

    Pre-2026-05-10 the ``pause_training`` and ``stop_training`` REST endpoints
    set ``_pause_event`` / ``_stop_event`` and transitioned the FSM, but the
    flags were never observed inside ``cascade_correlation.fit()`` — training
    ran to natural completion regardless. The fix wires ``_check_for_interrupt``
    into ``_handle_event``, dispatched on every ``epoch_end`` / ``phase_change`` event
    ``CascorModel.fit`` emits (the output-epoch and grow-iteration boundaries); when
    ``_stop_event`` is set it raises this sentinel which ``_run_training`` catches as a
    clean cancellation (FSM → STOP, status Stopped, cancelled-counter incremented; not
    a failure).

    See ``ISSUE_3_PHASE_2_LIVE_DATASET_SWAP_2026-05-09.md`` §3.4 audit findings
    for the original defect documentation; this fix is P2-PRE-1 of that plan.
    """


def _validate_candidate_pool_triple(s: int, t: int, r: int, p: int) -> Optional[str]:
    """Return ``None`` if (S, T, R, P) satisfy the §1.5 C2.1 invariant or a
    human-readable violation string otherwise.

    Bound to be called against the *post-merge* triple — never against the
    per-key delta — because a multi-key PATCH that's only valid as a unit
    (e.g. ``{S: 6, T: 4, R: 2}`` from a prior ``(S=2, T=2, R=0)``) must be
    accepted in one shot.
    """
    if not (1 <= s <= p):
        return f"selected_candidates {s} not in [1, candidate_pool_size={p}]"
    if t < 0 or r < 0:
        return f"top_candidates and random_candidates must be >= 0 (got T={t}, R={r})"
    if t > s or r > s:
        return f"each component must be <= selected_candidates (S={s}, T={t}, R={r})"
    if t == 0 and r == 0:
        return "top_candidates and random_candidates cannot both be 0"
    if t == 0 and r != s:
        return f"with top_candidates=0, random_candidates must equal S={s} (got R={r})"
    if r == 0 and t != s:
        return f"with random_candidates=0, top_candidates must equal S={s} (got T={t})"
    if t > 0 and r > 0 and t + r != s:
        return f"top_candidates+random_candidates must equal S={s} (got {t}+{r}={t + r})"
    return None


class _WeightHistoryRecorder:
    """CAN-015g (Phase 6E follow-on, g-6): training-loop weight history capture.

    Populates ``network.weight_history`` during a training run so the
    serializer's V2 path (g-1) actually has data to persist when the
    user calls ``POST /v1/snapshots``. Without this, V2 replay only
    works against ad-hoc test fixtures — production training would
    still produce V1-only snapshots.

    Plan: ``notes/PHASE_6E_DEFERRED_CAN-015GH_DESIGN.md`` §"Live
    capture during training (g-6)".

    Three trigger points:
      1. **Every Nth epoch** — registered as a callback on
         ``monitor.on_epoch_end``. ``N`` = network's
         ``config.weight_history_sampling_interval`` (default 50;
         set to 1 for every-epoch capture; set to 0 to disable the
         periodic trigger and rely on cascade-add only).
      2. **Cascade-grow events** — registered as a callback on
         ``monitor.on_cascade_add``. Always captures
         regardless of the periodic interval since these are the
         narrative anchors per the parent design.
      3. **Terminal capture** — call ``capture_terminal()`` from the
         lifecycle's training-completion path so the last sample
         reflects the truly-final weights even when training stops
         mid-interval.

    Memory ceiling: ``config.weight_history_max_samples`` (default
    1000) is a soft cap. On overflow the recorder decimates
    inter-cascade samples by 2× while always retaining cascade-add
    samples (they're the visually-meaningful moments).

    Idempotency: a single epoch can fire both Nth-epoch and
    cascade-add triggers. The recorder dedupes by epoch number — a
    sample for a given epoch is recorded at most once, with the
    cascade-add capture winning if both fire (it has the latest
    post-grow state).

    Thread safety: capture runs in the training thread (the same
    thread that fires ``on_epoch_end`` / ``on_cascade_add``). No
    cross-thread coordination is needed because ``network.weight_history``
    is only read by the lifecycle in (a) ``save_snapshot`` and
    (b) replay-session loading — both of which fire while training
    is paused or stopped per the FSM.
    """

    # Marker on a sample's metadata: ``True`` means this sample was
    # captured at a cascade-add event and is exempt from decimation.
    _CASCADE_FLAG_KEY = "_cascade_add"

    def __init__(self, network, monitor, *, sampling_interval: Optional[int] = None, max_samples: Optional[int] = None) -> None:
        self.network = network
        self.monitor = monitor
        config = getattr(network, "config", None)
        self.sampling_interval: int = int(sampling_interval if sampling_interval is not None else getattr(config, "weight_history_sampling_interval", 50))
        self.max_samples: int = int(max_samples if max_samples is not None else getattr(config, "weight_history_max_samples", 1000))
        self.logger = logging.getLogger(__name__)
        self._registered: bool = False
        self._init_weight_history()

    def _init_weight_history(self) -> None:
        """Ensure ``network.weight_history`` exists with the expected shape.

        Pre-g-6 networks won't have the attribute — initialize it
        with the same dict layout the g-1 serializer reads. Idempotent
        so reattaching the recorder mid-run doesn't clobber existing
        samples.
        """
        wh = getattr(self.network, "weight_history", None)
        if not isinstance(wh, dict):
            self.network.weight_history = {
                "sampling_strategy": "adaptive",
                "sampling_interval": self.sampling_interval,
                "sample_indices": [],
                "output_weights": [],
                "output_bias": [],
                # Per-unit dicts: {first_sample_index, activation, weights[], bias[]}.
                # See plan §"Hidden-unit slicing convention".
                "hidden_units": [],
                # Internal: epochs already captured (dedupe + decimation
                # bookkeeping). Not consumed by the serializer.
                "_captured_epochs": [],
                "_cascade_epochs": set(),
            }
        else:
            # Refresh the periodic interval so a runtime-tunable change
            # via PATCH /v1/training/params lands on the next trigger.
            wh["sampling_interval"] = self.sampling_interval
            wh.setdefault("sampling_strategy", "adaptive")
            wh.setdefault("sample_indices", [])
            wh.setdefault("output_weights", [])
            wh.setdefault("output_bias", [])
            wh.setdefault("hidden_units", [])
            wh.setdefault("_captured_epochs", [])
            wh.setdefault("_cascade_epochs", set())

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self) -> None:
        """Subscribe to ``on_epoch_end`` and ``on_cascade_add`` events.

        Idempotent — repeated calls are no-ops so multiple
        ``start_training`` invocations don't double-register.
        """
        if self._registered or self.monitor is None:
            return
        self.monitor.register_callback("epoch_end", self._on_epoch_end)
        self.monitor.register_callback("cascade_add", self._on_cascade_add)
        self._registered = True

    # ------------------------------------------------------------------
    # Trigger callbacks
    # ------------------------------------------------------------------

    def _on_epoch_end(self, **kwargs) -> None:
        """Periodic trigger: capture every Nth epoch.

        Skipped silently when ``sampling_interval == 0`` (cascade-add
        only mode). Best-effort — exceptions are logged but never
        crash the training thread.
        """
        if self.sampling_interval <= 0:
            return
        epoch = kwargs.get("epoch")
        if epoch is None:
            return
        try:
            epoch_int = int(epoch)
        except (TypeError, ValueError):
            return
        # Trigger on epoch numbers that are multiples of the interval
        # (epoch=1 with interval=50 does NOT fire; epoch=50 does).
        # Epoch 0 fires too — gives canopy the initial state.
        if epoch_int % self.sampling_interval != 0:
            return
        try:
            self._capture(epoch_int, is_cascade_add=False)
        except Exception:
            self.logger.exception("g-6: epoch_end capture raised; continuing")

    def _on_cascade_add(self, **kwargs) -> None:
        """Cascade-grow trigger: always capture.

        Fires after a unit has been fully installed (matches the
        existing ``hidden_units_added`` append point in
        ``cascade_correlation.add_unit``). Uses the monitor's
        ``current_epoch`` because the event payload doesn't carry it.
        """
        epoch = getattr(self.monitor, "current_epoch", None) if self.monitor is not None else None
        if epoch is None:
            return
        try:
            epoch_int = int(epoch)
        except (TypeError, ValueError):
            return
        try:
            self._capture(epoch_int, is_cascade_add=True)
        except Exception:
            self.logger.exception("g-6: cascade_add capture raised; continuing")

    def capture_terminal(self) -> None:
        """Final-epoch capture from the lifecycle's training-completion path.

        Public so the caller can invoke it explicitly; uses the
        monitor's ``current_epoch`` like ``_on_cascade_add`` and
        marks the sample as cascade-equivalent (decimation-exempt)
        so the terminal frame survives even on long runs that hit
        the soft cap.
        """
        epoch = getattr(self.monitor, "current_epoch", None) if self.monitor is not None else None
        if epoch is None:
            return
        try:
            epoch_int = int(epoch)
        except (TypeError, ValueError):
            return
        try:
            self._capture(epoch_int, is_cascade_add=True)
        except Exception:
            self.logger.exception("g-6: terminal capture raised; continuing")

    # ------------------------------------------------------------------
    # Capture mechanics
    # ------------------------------------------------------------------

    def _capture(self, epoch: int, *, is_cascade_add: bool) -> None:
        """Snapshot the network's current weights into ``weight_history``.

        Dedupes by epoch — a second trigger at the same epoch
        overwrites the previous sample's tensors (the cascade-add
        capture wins because it sees the latest post-grow state).
        Tensors are detached + ``cpu().numpy().copy()`` so the
        history holds an independent snapshot the optimizer can't
        mutate later.
        """
        self._init_weight_history()
        wh = self.network.weight_history

        captured = wh["_captured_epochs"]
        cascade_epochs = wh["_cascade_epochs"]

        if epoch in captured:
            existing_idx = captured.index(epoch)
            self._write_sample_at(existing_idx, epoch, is_cascade_add=is_cascade_add)
            if is_cascade_add:
                cascade_epochs.add(epoch)
            return

        # New sample — append.
        captured.append(epoch)
        wh["sample_indices"].append(epoch)
        wh["output_weights"].append(self._copy_output_weights())
        wh["output_bias"].append(self._copy_output_bias())
        new_index = len(captured) - 1
        # Hidden-unit per-sample slice: append a per-unit array entry.
        self._append_hidden_unit_slices(new_index, captured)
        if is_cascade_add:
            cascade_epochs.add(epoch)

        # Enforce the soft cap with cascade-aware decimation.
        if self.max_samples > 0 and len(captured) > self.max_samples:
            self._decimate(captured, cascade_epochs)

    def _write_sample_at(self, index: int, epoch: int, *, is_cascade_add: bool) -> None:
        """Overwrite an existing sample's tensors (dedupe path)."""
        wh = self.network.weight_history
        wh["output_weights"][index] = self._copy_output_weights()
        wh["output_bias"][index] = self._copy_output_bias()
        # Refresh per-unit slices for this sample only.
        captured = wh["_captured_epochs"]
        for unit_idx, unit_dict in enumerate(wh["hidden_units"]):
            first = unit_dict["first_sample_index"]
            local = index - first
            if local < 0:
                continue
            unit_w, unit_b = self._copy_unit(unit_idx)
            if unit_w is None:
                continue
            while len(unit_dict["weights"]) <= local:
                unit_dict["weights"].append(unit_w)
                unit_dict["bias"].append(unit_b)
            unit_dict["weights"][local] = unit_w
            unit_dict["bias"][local] = unit_b
        # Ensure any newly-cascade-grown unit (added between this
        # sample's first capture and the rewrite) gets a slot.
        self._append_hidden_unit_slices(index, captured, rewrite=True)

    def _append_hidden_unit_slices(self, sample_index: int, captured: list, rewrite: bool = False) -> None:
        """Ensure every current hidden unit has a per-sample slice for ``sample_index``."""
        wh = self.network.weight_history
        units = getattr(self.network, "hidden_units", None) or []
        for unit_idx, _ in enumerate(units):
            if unit_idx >= len(wh["hidden_units"]):
                # First time we've seen this unit — record its
                # ``first_sample_index`` (sample-list index, NOT
                # epoch — matches the g-2 cache convention).
                activation = self._unit_activation_name(unit_idx)
                wh["hidden_units"].append(
                    {
                        "first_sample_index": sample_index,
                        "activation": activation,
                        "weights": [],
                        "bias": [],
                    }
                )
            unit_dict = wh["hidden_units"][unit_idx]
            first = unit_dict["first_sample_index"]
            local = sample_index - first
            if local < 0:
                continue
            unit_w, unit_b = self._copy_unit(unit_idx)
            if unit_w is None:
                continue
            if rewrite and local < len(unit_dict["weights"]):
                unit_dict["weights"][local] = unit_w
                unit_dict["bias"][local] = unit_b
            else:
                while len(unit_dict["weights"]) <= local:
                    # Pad with zeros if a prior sample skipped this
                    # unit (shouldn't happen with the current trigger
                    # set but defensive against future trigger types).
                    unit_dict["weights"].append(unit_w)
                    unit_dict["bias"].append(unit_b)

    # ------------------------------------------------------------------
    # Tensor copy helpers (training-thread, no autograd retention)
    # ------------------------------------------------------------------

    def _copy_output_weights(self):
        ow = getattr(self.network, "output_weights", None)
        return self._tensor_to_numpy(ow)

    def _copy_output_bias(self):
        ob = getattr(self.network, "output_bias", None)
        return self._tensor_to_numpy(ob)

    def _copy_unit(self, unit_idx: int):
        units = getattr(self.network, "hidden_units", None) or []
        if unit_idx >= len(units):
            return None, None
        unit = units[unit_idx]
        w = unit.get("weights") if isinstance(unit, dict) else getattr(unit, "weights", None)
        b = unit.get("bias") if isinstance(unit, dict) else getattr(unit, "bias", None)
        w_np = self._tensor_to_numpy(w)
        b_np = self._tensor_to_numpy(b)
        if b_np is None:
            return w_np, 0.0
        # Per-unit bias is logically a scalar — flatten to a Python float
        # so the (g-1) serializer's atleast_1d wrap path stays uniform.
        return w_np, float(b_np.flat[0]) if b_np.size > 0 else 0.0

    @staticmethod
    def _tensor_to_numpy(t):
        if t is None:
            return None
        # Torch tensor path — detach to drop autograd, cpu() to leave
        # any device, numpy() + copy() so the buffer survives
        # subsequent in-place updates by the optimizer.
        try:
            return np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32).copy()
        except AttributeError:
            try:
                return np.ascontiguousarray(t, dtype=np.float32).copy()
            except (TypeError, ValueError):
                return None

    def _unit_activation_name(self, unit_idx: int) -> str:
        units = getattr(self.network, "hidden_units", None) or []
        if unit_idx >= len(units):
            return ""
        unit = units[unit_idx]
        if isinstance(unit, dict):
            act = unit.get("activation_fn") or unit.get("activation")
            if act is None:
                return ""
            name = getattr(act, "__name__", None) or str(act).__class__.__name__
            return str(name)
        return ""

    # ------------------------------------------------------------------
    # Decimation (memory ceiling)
    # ------------------------------------------------------------------

    def _decimate(self, captured: list, cascade_epochs: set) -> None:
        """Halve the inter-cascade sample density when the soft cap is hit.

        Drops every other non-cascade sample. Cascade-add samples
        (and the terminal sample, if marked) are always retained
        because they carry the narrative arc the user actually wants
        to scrub through.
        """
        wh = self.network.weight_history
        keep_mask: List[bool] = []
        non_cascade_seen = 0
        for epoch in captured:
            if epoch in cascade_epochs:
                keep_mask.append(True)
            else:
                # Drop every second non-cascade sample.
                keep = (non_cascade_seen % 2) == 0
                keep_mask.append(keep)
                non_cascade_seen += 1

        # Apply mask in-place to the parallel arrays.
        new_captured = [e for e, k in zip(captured, keep_mask) if k]
        new_indices = [v for v, k in zip(wh["sample_indices"], keep_mask) if k]
        new_outputs = [v for v, k in zip(wh["output_weights"], keep_mask) if k]
        new_biases = [v for v, k in zip(wh["output_bias"], keep_mask) if k]

        # Drop the same indices from each hidden unit's per-sample arrays;
        # adjust ``first_sample_index`` if that unit's first sample fell.
        for unit_dict in wh["hidden_units"]:
            old_first = unit_dict["first_sample_index"]
            unit_keep = keep_mask[old_first:]
            unit_dict["weights"] = [v for v, k in zip(unit_dict["weights"], unit_keep) if k]
            unit_dict["bias"] = [v for v, k in zip(unit_dict["bias"], unit_keep) if k]
            # Re-index ``first_sample_index`` to the new compacted list.
            kept_before = sum(1 for k in keep_mask[:old_first] if k)
            unit_dict["first_sample_index"] = kept_before

        wh["_captured_epochs"] = new_captured
        wh["sample_indices"] = new_indices
        wh["output_weights"] = new_outputs
        wh["output_bias"] = new_biases
        # ``_cascade_epochs`` already only references retained epochs.

        # Bump the *recorded* sampling_interval to reflect the
        # decimation so loaders / canopy can interpret the gap.
        wh["sampling_interval"] = max(1, wh.get("sampling_interval", self.sampling_interval) * 2)


class _WeightCache:
    """CAN-015g (Phase 6E follow-on, g-2): per-sample weight tensor cache.

    Lifts the dict-of-numpy-arrays produced by g-1's
    ``snapshot_serializer._load_weight_history`` into an LRU cache so
    ``_ReplaySession`` can serve weight payloads to canopy without
    re-walking the (potentially large) source dict on every scrubber
    move. Plan: ``notes/PHASE_6E_DEFERRED_CAN-015GH_DESIGN.md`` §
    "In-memory cache".

    The "LRU eviction with byte budget" design lives here rather than
    leaning on a generic library because:
      • The byte cost of an entry is the sum of ``ndarray.nbytes`` for
        every tensor in the sample (output + per-unit) — not something
        a generic cache can introspect.
      • The budget is a soft cap for fairness across long-running
        sessions, not a hard limit; missing the budget by one sample
        is preferable to evicting the just-requested entry.

    All public methods are thread-safe (the cache is read by the
    replay driver thread and written by the route response thread on
    cache misses).
    """

    DEFAULT_BUDGET_BYTES: int = 256 * 1024 * 1024  # 256 MB; g-2 of plan

    def __init__(self, weight_history: Optional[Dict[str, Any]], byte_budget: int = DEFAULT_BUDGET_BYTES):
        self._wh = weight_history or {}
        self._budget = max(1, int(byte_budget))
        # OrderedDict for LRU: ``move_to_end`` on access, ``popitem(last=False)``
        # for eviction. Key = sample_index (int).
        from collections import OrderedDict as _OrderedDict

        self._entries: "_OrderedDict[int, Dict[str, Any]]" = _OrderedDict()
        self._sizes: Dict[int, int] = {}
        self._total_bytes: int = 0
        self._lock = threading.Lock()
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True iff the underlying weight_history has any samples."""
        return bool(self._wh.get("sample_indices"))

    @property
    def sample_indices(self) -> List[int]:
        return list(self._wh.get("sample_indices", []))

    @property
    def num_samples(self) -> int:
        return len(self._wh.get("sample_indices", []))

    @property
    def sampling_strategy(self) -> str:
        return str(self._wh.get("sampling_strategy", ""))

    @property
    def sampling_interval(self) -> int:
        return int(self._wh.get("sampling_interval", 0))

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "entries": len(self._entries),
                "bytes": self._total_bytes,
                "budget_bytes": self._budget,
            }

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, sample_index: int) -> Optional[Dict[str, Any]]:
        """Return the per-sample weight payload, or ``None`` if the
        index is out of range / weights aren't available.

        On a cache hit the entry is promoted to most-recently-used.
        On a miss the payload is built from the source dict and
        admitted (possibly evicting older entries to fit the budget).
        """
        if not self.available:
            return None
        if sample_index < 0 or sample_index >= self.num_samples:
            return None

        with self._lock:
            if sample_index in self._entries:
                self._entries.move_to_end(sample_index)
                self._hits += 1
                return self._entries[sample_index]
            self._misses += 1

        payload = self._build_payload(sample_index)
        size = self._sizeof(payload)

        with self._lock:
            # Evict LRU entries until the new payload fits the budget.
            # Always admit at least the current payload even if it alone
            # exceeds the budget — the user explicitly asked for it.
            while self._entries and (self._total_bytes + size) > self._budget:
                evicted_key, _ = self._entries.popitem(last=False)
                self._total_bytes -= self._sizes.pop(evicted_key, 0)
                self._evictions += 1
            self._entries[sample_index] = payload
            self._sizes[sample_index] = size
            self._total_bytes += size

        return payload

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_payload(self, sample_index: int) -> Dict[str, Any]:
        """Project the per-sample slice out of the source weight_history.

        Layout matches what g-3 will emit in the synthetic ``epoch_end``
        event's ``weights`` block — keeping it stable here lets g-3
        forward this dict directly to subscribers.
        """
        output_weights = self._wh.get("output_weights", [])
        output_bias = self._wh.get("output_bias", [])
        hidden_units = self._wh.get("hidden_units", [])

        ow = output_weights[sample_index] if sample_index < len(output_weights) else None
        ob = output_bias[sample_index] if sample_index < len(output_bias) else None

        # Hidden-unit per-sample slicing. Each unit's per-sample arrays
        # are indexed relative to ``first_sample_index``, defined as the
        # **0-based index into the global ``sample_indices`` list** at
        # which this unit first appeared (NOT an epoch number). Skip
        # units whose first sample is in the future. The convention is
        # documented here because the data layout is otherwise opaque —
        # g-3's emitter and any future consumers must use the same
        # interpretation.
        hu_payload = []
        for unit in hidden_units:
            first = int(unit.get("first_sample_index", 0))
            if sample_index < first:
                continue
            local_idx = sample_index - first
            unit_w = unit.get("weights", [])
            unit_b = unit.get("bias", [])
            if local_idx >= len(unit_w):
                continue
            hu_payload.append(
                {
                    "first_sample_index": first,
                    "activation": unit.get("activation", ""),
                    "weights": unit_w[local_idx],
                    "bias": float(unit_b[local_idx]) if local_idx < len(unit_b) else 0.0,
                }
            )

        return {
            "sample_index": sample_index,
            "epoch": int(self._wh.get("sample_indices", [sample_index])[sample_index]),
            "output_weights": ow,
            "output_bias": ob,
            "hidden_units": hu_payload,
        }

    @staticmethod
    def _sizeof(payload: Dict[str, Any]) -> int:
        """Approximate the byte cost of a payload entry.

        Sums ``ndarray.nbytes`` for every tensor referenced. Skips
        Python overhead — close enough for budget-eviction decisions.
        """
        size = 0
        for key in ("output_weights", "output_bias"):
            arr = payload.get(key)
            if hasattr(arr, "nbytes"):
                size += int(arr.nbytes)
        for unit in payload.get("hidden_units", []):
            arr = unit.get("weights")
            if hasattr(arr, "nbytes"):
                size += int(arr.nbytes)
            # bias is a Python float — negligible.
        return size


class _ReplaySession:
    """CAN-015c (Phase 6E Sprint B B-3): per-snapshot replay session.

    Holds the playback state for a single replay run — current time
    index, speed (with sign for direction), pause flag, range
    sub-window — plus the background thread that ticks while playing
    and emits synthetic ``epoch_end`` events from the loaded network's
    history arrays.

    V1 scope (per ``notes/PHASE_6E_SPRINT_B_DESIGN.md`` §2.2 / §10.1):
    metric arrays + topology evolution metadata only. Per-epoch weight
    history (decision boundary playback, per-unit weight evolution) is
    deferred to CAN-015g — would require a snapshot-format extension.

    The thread emits via ``monitor._trigger_callbacks`` directly rather
    than ``monitor.on_epoch_end`` so synthetic frames don't pollute the
    live ``metrics_buffer``. Subscribers (WS broadcasters, canopy
    metrics-curve renderer) receive the events identically — only the
    replay-buffer side-effect differs.
    """

    # Allowed speed range. 0 ≡ pause, sign carries direction, magnitude
    # caps at 10× to avoid pathological CPU usage on very long
    # snapshots. Values beyond the range are clamped at /control time.
    _MIN_SPEED: float = -10.0
    _MAX_SPEED: float = 10.0
    _MIN_NONZERO_MAG: float = 0.1
    # Cap inter-frame sleeps so /pause / /seek wake up promptly even
    # at very low speeds.
    _MAX_TICK_SLEEP: float = 0.5

    def __init__(self, snapshot_id: str, history: Dict[str, list], monitor, weight_history: Optional[Dict[str, Any]] = None, weight_cache_budget_bytes: Optional[int] = None) -> None:
        self.snapshot_id = snapshot_id
        # Pre-extract the history arrays we know about so the loop
        # doesn't re-fetch every tick. Stored as plain lists (not
        # references to network.history) so a future Restore-while-
        # somehow-still-replaying race doesn't mutate them under us.
        self._history: Dict[str, list] = {key: list(history.get(key, [])) for key in ("train_loss", "value_loss", "train_accuracy", "value_accuracy")}
        # Length is the longest known array. Time index is bounded
        # exclusively by ``self.length`` (i.e. valid indices are
        # ``[0, length-1]``). Empty histories produce length=0 and the
        # loop correctly idles.
        self.length: int = max((len(v) for v in self._history.values()), default=0)
        self._monitor = monitor
        # CAN-015g (g-2): per-sample weight cache. Empty / absent
        # weight_history (V1 snapshots, or networks where g-1
        # serializer didn't load any samples) yields a cache that
        # advertises ``available=False``; canopy then knows to disable
        # the decision-boundary scrubber. The cache is constructed
        # eagerly even in the V1 case so consumers don't have to
        # branch on ``is None``.
        self.weight_cache = _WeightCache(weight_history, byte_budget=weight_cache_budget_bytes if weight_cache_budget_bytes is not None else _WeightCache.DEFAULT_BUDGET_BYTES)
        # Playback state — guarded by the lock for cross-thread reads.
        self._lock = threading.Lock()
        self.time_index: int = 0
        self.speed: float = 1.0
        self.paused: bool = True
        self.range_start: int = 0
        self.range_end: int = self.length  # exclusive
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)

    def start_thread(self) -> None:
        """Start the playback driver thread. Idempotent (a session can
        be re-started after pause without spawning a new thread)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name=f"replay-{self.snapshot_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the playback thread to exit and wait briefly for it
        to drain. Safe to call from any thread including the lifecycle
        shutdown path."""
        self._stop_event.set()
        self._wake_event.set()  # break out of any pending wait
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def play(self) -> None:
        with self._lock:
            self.paused = False
        self._wake_event.set()

    def pause(self) -> None:
        with self._lock:
            self.paused = True
        self._wake_event.set()

    def seek(self, target: int) -> int:
        """Jump to a specific time index. Returns the actual landed
        position after clamping to the active range."""
        with self._lock:
            self.time_index = self._clamp_to_range(target)
            landed = self.time_index
        self._wake_event.set()
        # Emit the seek-target frame immediately so canopy gets visual
        # feedback even if we're paused.
        self._emit_frame(landed)
        return landed

    def set_speed(self, value: float) -> float:
        """Set playback speed. ``0`` is treated as pause. Returns the
        effective (clamped) speed."""
        # Clamp magnitude to [_MIN_NONZERO_MAG, _MAX_SPEED] preserving
        # sign; treat tiny magnitudes as 0 (pause).
        if abs(value) < self._MIN_NONZERO_MAG:
            value = 0.0
        elif value > 0:
            value = min(value, self._MAX_SPEED)
        else:
            value = max(value, self._MIN_SPEED)
        with self._lock:
            self.speed = value
            # speed=0 is functionally pause; surface the flag too so
            # /play later doesn't have to also call /speed.
            if value == 0.0:
                self.paused = True
        self._wake_event.set()
        return value

    def set_range(self, start: int, end: int) -> Dict[str, int]:
        """Restrict playback to ``[start, end)``. End may be at most
        ``self.length``. Time index is re-clamped if it's now outside
        the new range. Returns the resulting range as a dict."""
        with self._lock:
            self.range_start = max(0, min(start, self.length))
            self.range_end = max(self.range_start, min(end, self.length))
            self.time_index = self._clamp_to_range(self.time_index)
            result = {"start": self.range_start, "end": self.range_end, "time_index": self.time_index}
        self._wake_event.set()
        return result

    def state_summary(self) -> Dict[str, Any]:
        """Snapshot of the current session state for the route response."""
        with self._lock:
            summary: Dict[str, Any] = {
                "snapshot_id": self.snapshot_id,
                "length": self.length,
                "time_index": self.time_index,
                "speed": self.speed,
                "paused": self.paused,
                "range": {"start": self.range_start, "end": self.range_end},
            }
        # CAN-015g (g-2): expose weight-cache state so canopy knows
        # whether decision-boundary playback is available and can
        # render the scrubber accordingly. ``sample_epochs`` is the
        # epoch number at each sample boundary — canopy uses these to
        # snap the scrubber to the closest sample.
        summary["weights_available"] = self.weight_cache.available
        if self.weight_cache.available:
            summary["weight_sampling"] = {
                "strategy": self.weight_cache.sampling_strategy,
                "interval": self.weight_cache.sampling_interval,
                "num_samples": self.weight_cache.num_samples,
                "sample_epochs": self.weight_cache.sample_indices,
            }
        return summary

    def weights_at(self, sample_index: int) -> Optional[Dict[str, Any]]:
        """Return the per-sample weight payload for ``sample_index``,
        or ``None`` if the snapshot has no weight history or the
        index is out of range.

        Wrapper around the cache so callers (g-3's emitter, ad-hoc
        ops queries) don't have to reach into ``self.weight_cache``
        directly.
        """
        return self.weight_cache.get(sample_index)

    def _clamp_to_range(self, index: int) -> int:
        if self.range_end <= self.range_start:
            return self.range_start
        return max(self.range_start, min(self.range_end - 1, index))

    def _emit_frame(self, index: int) -> None:
        """Emit a synthetic ``epoch_end`` event for the given index.

        Bypasses ``monitor.on_epoch_end`` (which would write to
        ``metrics_buffer``) and calls ``_trigger_callbacks`` directly
        so the WS broadcasters fire but live training state stays
        untouched. Per the design's read-only-history guarantee."""
        if self._monitor is None:
            return
        if index < 0 or index >= self.length:
            return

        def _series_at(key: str):
            series = self._history.get(key, [])
            return series[index] if index < len(series) else None

        metrics = {
            "epoch": index + 1,  # 1-indexed for canopy display, matches on_epoch_end
            "loss": _series_at("train_loss"),
            "accuracy": _series_at("train_accuracy"),
            "validation_loss": _series_at("value_loss"),
            "validation_accuracy": _series_at("value_accuracy"),
            "phase": "Replay",
            "replay": True,  # marker so subscribers can distinguish synthetic frames
            "snapshot_id": self.snapshot_id,
        }
        try:
            self._monitor._trigger_callbacks(
                "epoch_end",
                metrics=metrics,
                epoch=metrics["epoch"],
                loss=metrics["loss"],
                accuracy=metrics["accuracy"],
            )
        except Exception:
            # Best-effort emission — a subscriber that raises mustn't
            # crash the playback thread.
            self.logger.exception("replay session: synthetic _trigger_callbacks raised")

    def _run(self) -> None:
        """Background thread driver. Sleeps for ``1/abs(speed)`` between
        frames while playing, polls every ``_MAX_TICK_SLEEP`` while
        paused. Wake-event short-circuits any wait so /pause / /seek /
        /speed take effect immediately."""
        # Emit an initial frame on session start so subscribers see
        # the entry point (epoch 0) before any /play.
        self._emit_frame(0)
        while not self._stop_event.is_set():
            with self._lock:
                paused = self.paused
                speed = self.speed
                time_index = self.time_index
                range_start = self.range_start
                range_end = self.range_end
            if paused or abs(speed) < self._MIN_NONZERO_MAG:
                # Idle until woken up — by /play, /seek, /speed change,
                # or /stop. Bounded wait so /stop_event from a separate
                # call still terminates the thread promptly.
                if self._wake_event.wait(self._MAX_TICK_SLEEP):
                    self._wake_event.clear()
                continue
            # Compute sleep duration for this frame. Bounded above so
            # very low speeds still yield to wake events promptly.
            sleep_s = min(1.0 / abs(speed), self._MAX_TICK_SLEEP)
            if self._wake_event.wait(sleep_s):
                self._wake_event.clear()
                continue
            # Advance the time index respecting direction and range.
            with self._lock:
                step = 1 if speed > 0 else -1
                new_index = time_index + step
                if new_index < range_start or new_index >= range_end:
                    # Reached a boundary — auto-pause at the edge.
                    self.paused = True
                    self.time_index = self._clamp_to_range(new_index)
                    landed = self.time_index
                else:
                    self.time_index = new_index
                    landed = new_index
            self._emit_frame(landed)


class TrainingLifecycleManager:
    """Central coordinator for CasCor network training lifecycle.

    Manages network creation, training execution (async via ThreadPoolExecutor),
    monitoring hooks, state tracking, and metrics collection.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Core components
        # WS-6 B-phase (native model-core adoption): the manager holds a
        # model-core ``CascorModel`` (wrapping the ``CascadeCorrelationNetwork``).
        # ``self.network`` is a back-compat property over ``self.model`` (defined
        # below) so the cascor-specific reaches keep resolving to the CCN.
        self.model: Optional[CascorModel] = None
        self.state_machine = TrainingStateMachine()
        self.training_state = TrainingState()
        self.monitor = TrainingMonitor()

        # Threading
        self._lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._topology_lock = threading.Lock()
        # CONC-02 / BUG-CC-16 (Phase 3B): guard the broadcast throttle so two
        # callers cannot both pass the (now - last) < interval check and emit
        # duplicate state messages. _last_state_broadcast_time is initialized
        # here so the read in _broadcast_training_state never has to gate on
        # hasattr() (which itself was part of the original race window).
        self._broadcast_lock = threading.Lock()
        self._last_state_broadcast_time: float = 0.0
        self._executor: Optional[ThreadPoolExecutor] = None
        self._training_future: Optional[Future] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially
        self._last_emitted_history_len = 0

        # C5 (Q4/U-1 retention semantics):
        #   ``_retain_metrics_next_run`` — set by ``start_training`` and read by
        #     ``_run_training`` to decide whether the upcoming run RETAINS the
        #     metrics/history buffer (default; cross-dataset continuity) or
        #     clears it (start-fresh / resume rebuild). Defaults to retain.
        #   ``_metrics_undo_buffer`` — a snapshot of the rows removed by an
        #     explicit ``clear_metrics_with_undo``; ``None`` when no undo is
        #     available. Bounded by the deque ``maxlen``
        #     (``_PROJECT_API_METRICS_BUFFER_SIZE`` = 10000 rows), so a pending
        #     undo costs at most one extra buffer's worth of memory. Dropped
        #     when a run starts (the clear is then finalized).
        #   ``_metrics_undo_lock`` — guards the undo buffer independently of the
        #     big ``_lock`` so clear/undo never contend with training control.
        self._retain_metrics_next_run: bool = True
        self._metrics_undo_buffer: Optional[List[Dict[str, Any]]] = None
        self._metrics_undo_lock = threading.Lock()

        # WS-6 PR-B3.3: live monitoring is driven by CascorModel.fit's on_event sink
        # (_handle_event) rather than monkey-patching network.fit/grow_network. These
        # per-run bookkeeping fields are reset at the top of _run_training.
        self._step_timer_prev: Optional[float] = None
        self._grow_phase_entered: bool = False
        self._cascade_emitted_count: int = 0

        # Training data
        self._train_x: Optional[torch.Tensor] = None
        self._train_y: Optional[torch.Tensor] = None
        self._val_x: Optional[torch.Tensor] = None
        self._val_y: Optional[torch.Tensor] = None

        # C7 (U-4): expanded scalar evaluation metrics — F1 / precision / recall
        # / ROC-AUC computed over the evaluation split (the validation/test split
        # ``_val_x``/``_val_y`` when present, else the training split). Computed
        # in ``_extract_and_record_metrics`` once per metrics drain — i.e. once
        # per completed TRAINING STEP (initial output pass + one per growth
        # iteration), NOT per inner epoch — so the added cost is a single
        # ``torch.no_grad()`` forward pass over the eval split per step
        # (negligible for the 2-D research datasets; bounded for large sets since
        # it is step-cadenced, not epoch-cadenced). The scalars ride the terminal
        # training-step metrics row (see ``TrainingMonitor.on_epoch_end``) and the
        # ``/v1/metrics`` snapshot (``get_metrics``). Enabled by default; set
        # ``JUNIPER_CASCOR_EVAL_METRICS_ENABLED`` to ``0``/``false`` to disable
        # (distinct from ``JUNIPER_CASCOR_METRICS_ENABLED``, which gates the
        # Prometheus endpoint). ``_latest_scalar_metrics`` caches the newest
        # computed result for the snapshot surface; reset at each run start.
        self._eval_metrics_enabled: bool = _env_flag("JUNIPER_CASCOR_EVAL_METRICS_ENABLED", default=True)
        self._eval_metrics_average: str = "macro"
        self._latest_scalar_metrics: Optional[Dict[str, Any]] = None

        # Network creation params (for reset)
        self._params: Optional[Dict[str, Any]] = None

        # FRONTEND_ISSUES_PLAN_2026-05-09 §3.5.1 / Issue #3 Phase 1 — staged
        # dataset config. Set by ``stage_dataset_config`` (POST /v1/training/
        # dataset), consumed + cleared by the next ``start_training`` via
        # ``_reload_dataset``. ``clear_pending_dataset_config`` (DELETE) and
        # ``get_pending_dataset_config`` (GET .../pending) round it out so the
        # canopy banner can show the staged change and offer Cancel.
        self._pending_dataset_config: Optional[Dict[str, Any]] = None

        # ISSUE_3_PHASE_2_LIVE_DATASET_SWAP_2026-05-09 — Phase 2 P2-1a.
        # ``_experimental_functions_enabled`` gates ``swap_dataset_live`` and is
        # the server-side authority (F2.10): a stale frontend toggle alone cannot
        # bypass it. Read from env at boot so deployments can pre-enable it
        # without round-tripping the admin route on every restart.
        # ``_current_dataset_config`` mirrors whatever cfg was last applied via
        # ``_reload_dataset`` — drives the ``before_cfg`` field in the
        # ``swap_dataset_live`` response so canopy doesn't have to track it.
        # ``_swap_in_progress`` is the §3.7 guardrail-#3 idempotency flag.
        self._experimental_functions_enabled: bool = os.environ.get("CASCOR_EXPERIMENTAL_FUNCTIONS_ENABLED") == "1"
        self._current_dataset_config: Optional[Dict[str, Any]] = None
        self._swap_in_progress: bool = False
        # P2-1b: ``_swap_cancel_requested`` is signalled by
        # ``DELETE /v1/training/dataset/live`` and observed at safe checkpoints
        # inside ``swap_dataset_live`` (post-fetch, pre-future-resubmit). Using
        # an ``Event`` rather than a bool gives a memory-barriered read across
        # the cancel-issuer thread and the swap-driver thread without needing
        # to re-acquire ``_lock`` (which the swap holds for its full
        # duration). Cleared in the swap's ``finally`` so a future swap doesn't
        # observe a stale set from a prior aborted swap.
        self._swap_cancel_requested: threading.Event = threading.Event()

        # CAN-015g (g-6): training-loop weight history recorder.
        # Lazily attached on first ``start_training`` call; persists
        # across re-trains so callbacks register exactly once.
        self._weight_history_recorder: Optional["_WeightHistoryRecorder"] = None

        # WebSocket manager (set via set_ws_manager)
        self._ws_manager = None
        self._state_throttle_interval: float = 1.0  # seconds, configurable via set_ws_manager

        # Worker coordinator (set via set_worker_coordinator)
        self._worker_coordinator = None

        # METRICS-MON R1.2 / seed-03: liveness heartbeat. A 1-second daemon
        # bumps ``_liveness_counter`` and ``_liveness_last_tick_at``; the
        # liveness probe consults ``is_alive()`` to detect a wedged process.
        # The TrainingMonitor callbacks also bump the counter so progress
        # in the training thread is an additional liveness signal.
        self._liveness_counter: int = 0
        self._liveness_last_tick_at: float = time.monotonic()
        self._liveness_lock = threading.Lock()
        self._liveness_stop_event = threading.Event()
        self._liveness_thread: Optional[threading.Thread] = None
        self._start_liveness_thread()
        self._register_liveness_monitor_callbacks()

        # CAS-006 (Phase 6E Sprint A-4): auto-snap-best.
        # Hooks ``monitor.epoch_end`` and saves a snapshot every
        # time (validation) accuracy beats the best-seen-so-far for the
        # current run. Defaults: feature off, 50-epoch warmup. Both are
        # exposed via TrainingParams + TrainingParamUpdateRequest and
        # included in ``updatable_keys`` so users can toggle mid-run.
        self._auto_snap_best: bool = False
        self._auto_snap_min_epochs: int = 50
        self._auto_snap_best_metric: Optional[float] = None
        self._auto_snap_lock = threading.Lock()

        # CAN-015c (Phase 6E Sprint B B-3): replay session. ``None``
        # outside of an active replay; an ``_ReplaySession`` instance
        # while ``state_machine.is_replaying()``. The route layer reads
        # the session state for /control responses and dispatches
        # play/pause/seek/speed/range/stop into it.
        self._replay_session: Optional["_ReplaySession"] = None

        # CAN-015b (Phase 6E Sprint B B-2): resume-from-snapshot marker.
        # ``resume_from_snapshot`` sets this to the snapshot's terminal
        # epoch count so canopy can render a visual boundary in the
        # metrics-curve component (a vertical line separating the
        # pre-resume read-only history from the new training that
        # appends past it). Cleared once consumed by ``start_training``
        # (so a subsequent run from the same snapshot doesn't mistakenly
        # carry over the marker).
        self._resume_point_epoch: Optional[int] = None
        self.monitor.register_callback("epoch_end", self._maybe_auto_snap_callback)

        self.logger.info("TrainingLifecycleManager initialized")

    def _maybe_auto_snap_callback(self, metrics=None, epoch=None, loss=None, accuracy=None, **_kwargs) -> None:
        """CAS-006 (A-4): epoch_end callback that saves a snapshot when the
        current (validation) accuracy beats the best-seen-so-far for the
        current run. No-op when the feature is disabled, when the warmup
        threshold has not been reached, or when no usable accuracy metric
        is available.

        Tracks the best metric on the lifecycle (not the network) because
        a single network instance can be used across multiple training
        runs; ``start_training`` resets the tracker so each run starts
        fresh. Prefers ``validation_accuracy`` over ``accuracy`` so the
        snapshot reflects generalization rather than training-set fit.
        """
        with self._auto_snap_lock:
            if not self._auto_snap_best:
                return
            if epoch is None or epoch < self._auto_snap_min_epochs:
                return
            current = None
            if isinstance(metrics, dict):
                current = metrics.get("validation_accuracy")
            if current is None:
                current = accuracy
            if current is None:
                return
            best = self._auto_snap_best_metric
            if best is not None and current <= best:
                return
            self._auto_snap_best_metric = current
            description = f"auto_snap_best epoch={epoch} accuracy={current:.6f}"
        # Save outside the auto_snap_lock so a slow filesystem doesn't
        # serialize the next epoch_end callback. ``save_snapshot`` has
        # its own internal failure handling.
        try:
            self.save_snapshot(description=description)
        except Exception:
            self.logger.exception("auto_snap_best: save_snapshot failed (epoch=%s, accuracy=%s)", epoch, current)

    def _register_liveness_monitor_callbacks(self) -> None:
        """Bump the heartbeat from every training-monitor event so progress
        in the training thread is an additional liveness signal.
        """
        bump = lambda **_kw: self.bump_liveness()  # noqa: E731 — concise wrapper
        for event in ("epoch_start", "epoch_end", "cascade_add", "training_start", "training_end", "topology_change", "candidate_progress", "phase_change"):
            self.monitor.register_callback(event, bump)

    def bump_liveness(self) -> None:
        """Record that the lifecycle is making forward progress.

        Called from the 1-second daemon thread and from TrainingMonitor
        event callbacks. The probe layer reads the resulting timestamp
        via ``is_alive()`` to decide liveness.
        """
        with self._liveness_lock:
            self._liveness_counter += 1
            self._liveness_last_tick_at = time.monotonic()

    def is_alive(self, stale_after_seconds: float = 30.0) -> bool:
        """Return True if the heartbeat has been bumped within the window.

        ``stale_after_seconds`` defaults to 30 s — well above the daemon
        thread's 1-second cadence, so transient scheduling jitter does
        not flap liveness, but well below typical Helm
        ``failureThreshold`` × ``periodSeconds`` so real wedges still
        get caught.
        """
        with self._liveness_lock:
            last = self._liveness_last_tick_at
        return (time.monotonic() - last) < stale_after_seconds

    def _start_liveness_thread(self) -> None:
        """Start the 1-second daemon that bumps the heartbeat."""

        def _loop() -> None:
            while not self._liveness_stop_event.is_set():
                self.bump_liveness()
                self._liveness_stop_event.wait(1.0)

        self._liveness_thread = threading.Thread(
            target=_loop,
            name="lifecycle-liveness",
            daemon=True,
        )
        self._liveness_thread.start()

    def stop_liveness_heartbeat(self) -> None:
        """Stop the heartbeat thread (used in shutdown / tests)."""
        self._liveness_stop_event.set()
        if self._liveness_thread is not None:
            self._liveness_thread.join(timeout=2.0)

    def set_ws_manager(self, ws_manager, state_throttle_interval: float = 1.0) -> None:
        """Set the WebSocket manager for real-time broadcasting.

        Registers monitor callbacks that broadcast metrics/events via WebSocket.

        Args:
            ws_manager: WebSocketManager instance.
            state_throttle_interval: Minimum interval in seconds between
                non-terminal state broadcasts (GAP-WS-21 coalescer).
        """
        self._ws_manager = ws_manager
        self._state_throttle_interval = state_throttle_interval
        self._register_ws_callbacks()

    def set_worker_coordinator(self, coordinator) -> None:
        """Set the worker coordinator for remote WebSocket worker dispatch.

        When a coordinator is set, newly created networks will have it injected
        so they can dispatch candidate training tasks to remote workers.
        If a network already exists, the coordinator is injected immediately.
        """
        self._worker_coordinator = coordinator
        if self.network is not None and hasattr(self.network, "set_worker_coordinator"):
            self.network.set_worker_coordinator(coordinator)
            self.logger.info("Worker coordinator injected into existing network")

    def _register_ws_callbacks(self) -> None:
        """Register WebSocket broadcast callbacks on the training monitor."""
        if self._ws_manager is None:
            return

        from api.websocket.messages import create_candidate_progress_message, create_cascade_add_message, create_event_message, create_metrics_message

        ws = self._ws_manager

        self.monitor.register_callback(
            "epoch_end",
            lambda metrics, **kw: ws.broadcast_from_thread(create_metrics_message(metrics)),
        )
        self.monitor.register_callback(
            "cascade_add",
            lambda event, **kw: ws.broadcast_from_thread(create_cascade_add_message(event)),
        )
        self.monitor.register_callback(
            "training_start",
            lambda **kw: self._broadcast_training_state(force=True),
        )
        self.monitor.register_callback(
            "training_end",
            lambda **kw: ws.broadcast_from_thread(create_event_message({"event": "training_complete"})),
        )
        self.monitor.register_callback(
            "candidate_progress",
            lambda progress, **kw: ws.broadcast_from_thread(create_candidate_progress_message(progress)),
        )

        self.logger.info("WebSocket broadcast callbacks registered")

    # Terminal statuses that must always bypass the broadcast throttle (GAP-WS-21)
    _TERMINAL_STATUSES = frozenset({"Completed", "Failed", "Stopped"})

    def _broadcast_training_state(self, force: bool = False) -> None:
        """Broadcast full training state via WebSocket.

        Uses a terminal-aware debounced coalescer (GAP-WS-21):
        - Terminal transitions (Completed/Failed/Stopped) always bypass throttle
        - force=True always bypasses throttle
        - Non-terminal transitions throttled to at most once per coalesce interval
        """
        if self._ws_manager is None:
            return

        state_data = self.training_state.get_state()
        status = state_data.get("status", "")
        is_terminal = status in self._TERMINAL_STATUSES

        # CONC-02 / BUG-CC-16 (Phase 3B): the throttle is a check-then-set on
        # _last_state_broadcast_time. Without a lock two threads (training
        # thread, monitor thread, control endpoint) could both observe
        # `now - last >= interval`, both pass, and both broadcast — defeating
        # the GAP-WS-21 coalescer. Hold _broadcast_lock across the read and
        # the write so only one caller wins the throttle window. Terminal
        # transitions and force=True still bypass the throttle but still
        # update the timestamp under the lock for consistency.
        with self._broadcast_lock:
            now = time.monotonic()
            if not force and not is_terminal:
                if now - self._last_state_broadcast_time < self._state_throttle_interval:
                    # OBS-WIRE-02 (3.7): the GAP-WS-21 coalescer just
                    # dropped a non-terminal state broadcast. Bump the
                    # counter so a sudden zero-rate while broadcast
                    # count is high becomes a regression signal.
                    try:
                        from api.observability import ws_inc_state_throttle_coalesced

                        ws_inc_state_throttle_coalesced()
                    except Exception:
                        # Defensive: prometheus_client may be absent
                        # in some test environments. Match the
                        # OBS-WIRE-01 logger.debug pattern so the
                        # failure is recoverable in production logs
                        # without short-circuiting the throttle path.
                        self.logger.debug("ws_inc_state_throttle_coalesced emission failed", exc_info=True)
                    return
            self._last_state_broadcast_time = now

        from api.websocket.messages import create_state_message

        self._ws_manager.broadcast_from_thread(create_state_message(state_data))

    # ------------------------------------------------------------------
    # Network management
    # ------------------------------------------------------------------

    def create_network(self, **kwargs) -> Dict[str, Any]:
        """Create a new CascadeCorrelationNetwork.

        Args:
            **kwargs: Parameters passed to CascadeCorrelationConfig.create_simple_config()

        Returns:
            Network info dictionary
        """
        with self._lock:
            return self._create_network_locked(**kwargs)

    def _create_network_locked(self, **kwargs) -> Dict[str, Any]:
        """``create_network`` body for callers that already hold ``self._lock``.

        ``self._lock`` is a non-reentrant ``threading.Lock``, so ``start_training``'s
        create-on-start path (training-start diagnosis 2026-07-09 PR-B) cannot call
        the public ``create_network`` from inside its locked section — it calls this
        instead. Behavior is identical to the public method.
        """
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

        # Reject while STARTED (active fit), PAUSED (parked training thread still
        # owns the network), or REPLAYING (history playback session). Replacing
        # the model in those states races the parked/replay thread and corrupts
        # status surfaces.
        if self.state_machine.is_started() or self.state_machine.is_paused():
            raise RuntimeError("Cannot create network while training is active")
        if self.state_machine.is_replaying():
            raise RuntimeError("Cannot create network while replay is active")
        # INVESTIGATING owns an inspected snapshot model (patch_weights /
        # add_hidden_unit_manual / retrain). Replacing it in-place leaves the
        # FSM Investigating against a brand-new network that is not the
        # restored snapshot — start_training stays rejected and weight edits
        # target the wrong object. Require /retrain or /reset first.
        if self.state_machine.is_investigating():
            raise RuntimeError("Cannot create network while investigating a snapshot")

        self._params = kwargs.copy()
        config = CascadeCorrelationConfig.create_simple_config(**kwargs)
        # WS-6 B-phase: wrap the freshly built CCN in the model-core CascorModel.
        # PR-B3.3: no monitoring hooks to install — live monitoring rides
        # CascorModel.fit's on_event sink (_handle_event), wired per-fit in _run_training.
        self.model = CascorModel(network=CascadeCorrelationNetwork(config=config))

        # Inject worker coordinator for remote dispatch if available
        if self._worker_coordinator is not None and hasattr(self.network, "set_worker_coordinator"):
            self.network.set_worker_coordinator(self._worker_coordinator)

        # C2b (I-4 root / I-1c): seed TrainingState from the network that was
        # ACTUALLY created, not from ``kwargs.get(..., lifecycle-default)``.
        # The old seeding used a second, independent default layer
        # (``_PROJECT_API_LIFECYCLE_DEFAULT_*``) whenever a kwarg was omitted
        # — e.g. the create-on-start path passes only input/output sizes, so
        # ``/v1/training/status`` reported ``max_hidden_units: 10000,
        # learning_rate: 0.01`` while ``/v1/network`` reported the engine's
        # effective values. The live network object is now the single source
        # of truth for both surfaces (training-runtime-defects plan §4 I-4).
        self.training_state.update_state(
            status="Stopped",
            phase="Idle",
            network_name=f"CasCor-{kwargs.get('input_size', _PROJECT_API_NETWORK_INPUT_SIZE_DEFAULT)}x{kwargs.get('output_size', _PROJECT_API_NETWORK_OUTPUT_SIZE_DEFAULT)}",
        )
        self._sync_training_state_from_network()

        info = self.get_network_info()
        self.logger.info(f"Network created: {info['input_size']}x{info['output_size']}")
        return info

    def delete_network(self) -> None:
        """Delete the current network."""
        with self._lock:
            # Mirror create_network: PAUSED keeps a parked training thread that
            # still references ``self.model``; REPLAYING owns a replay session.
            # Clearing the model under either state leaves dangling futures /
            # orphaned replay workers without event/session cleanup.
            if self.state_machine.is_started() or self.state_machine.is_paused():
                raise RuntimeError("Cannot delete network while training is active")
            if self.state_machine.is_replaying():
                raise RuntimeError("Cannot delete network while replay is active")
            # Mirror create_network: INVESTIGATING still owns the inspected
            # snapshot model for patch/retrain flows. Clearing it under that
            # state strands the FSM Investigating with no model.
            if self.state_machine.is_investigating():
                raise RuntimeError("Cannot delete network while investigating a snapshot")
            self.model = None
            self._params = None
            self.state_machine.handle_command(Command.RESET)
            self.training_state.update_state(status="Stopped", phase="Idle")
            self.logger.info("Network deleted")

    def has_model(self) -> bool:
        return self.model is not None

    def has_network(self) -> bool:
        """Deprecated alias for :meth:`has_model` (WS-6 B2a seam name-align). Kept
        for any external caller; all in-repo call sites use ``has_model``."""
        return self.has_model()

    @property
    def network(self):
        """The wrapped ``CascadeCorrelationNetwork`` (or ``None``).

        WS-6 B-phase: the manager now holds a model-core :class:`CascorModel` in
        :attr:`model`; ``network`` is a back-compat property so the cascor-specific
        reaches (``self.network.<attr>`` reads, monkey-patch write-through,
        HDF5/live-swap surgery, decision-boundary ``forward``) keep resolving to the
        underlying CCN. Reads delegate to ``self.model.network``.
        """
        return self.model.network if self.model is not None else None

    @network.setter
    def network(self, value):
        """Back-compat setter: accept a bare ``CascadeCorrelationNetwork`` (wrap it),
        a ready :class:`CascorModel`, or ``None`` — storing into :attr:`model`. The
        internal assignment sites set ``self.model`` directly; this keeps any
        external ``manager.network = <ccn>`` caller working."""
        if value is None:
            self.model = None
        elif isinstance(value, CascorModel):
            self.model = value
        else:
            self.model = CascorModel(network=value)

    def get_network_info(self) -> Dict[str, Any]:
        """Get network information."""
        if self.network is None:
            return {}
        return {
            "input_size": self.network.input_size,
            "output_size": self.network.output_size,
            "hidden_units": len(self.network.hidden_units),
            "max_hidden_units": getattr(self.network, "max_hidden_units", 0),
            "learning_rate": getattr(self.network, "learning_rate", 0.0),
            "uuid": str(getattr(self.network, "uuid", "")),
        }

    @staticmethod
    def derive_epochs_cap(network) -> int:
        """C2b / Q1 outcome (c): per-run derived total-epoch cap implied by the granular limits.

        ``epochs_max`` outlived its original role (a hard stop for a simple, plateau-prone
        early model): the engine stores the attribute at construction but **never reads it**
        (``cascade_correlation.py`` sets ``self.epochs_max`` in ``_init_network_parameters``
        and no code path consults it), while the limits that actually gate training are the
        granular meta-parameters — ``output_epochs`` (per output-training pass),
        ``candidate_epochs`` (per candidate-pool pass), ``max_iterations`` (cascade growth
        iterations) and ``max_hidden_units`` (growth capacity). Per the owner decision
        (training-runtime-defects plan §12 Q1), ``epochs_max`` is now DERIVED from those
        limits instead of being an independently settable value that can contradict or
        shadow them:

            effective_iterations = min(max_iterations, max_hidden_units)
            epochs_max = output_epochs + effective_iterations * (candidate_epochs + output_epochs)

        i.e. one initial output-training pass, plus — for every growth iteration the limits
        admit — one candidate-pool pass and one output retraining pass. Candidates within a
        pool train concurrently, so the pool contributes ``candidate_epochs`` sequential
        epochs per iteration (pool size multiplies work, not sequential epochs). The value
        is a *reporting/display budget* (the ``Epoch: X / Y`` denominator canopy's N6
        consumes), not an enforced abort: enforcement stays with the granular limits
        themselves, which is exactly the no-shadowing property Q1 requires. Early stopping
        / patience can end a run well below the cap.

        Stable and computable at start time (all four inputs are known at
        ``start_training``); it changes only when a granular limit is PATCHed — the
        ``_sync_training_state_from_network`` call sites keep ``training_state.max_epochs``
        aligned at create / param-apply / snapshot-load. Robust to partial network
        stand-ins (tests): every input is read with ``getattr(..., 0)``.
        """
        output_epochs = int(getattr(network, "output_epochs", 0) or 0)
        candidate_epochs = int(getattr(network, "candidate_epochs", 0) or 0)
        max_iterations = int(getattr(network, "max_iterations", 0) or 0)
        max_hidden_units = int(getattr(network, "max_hidden_units", 0) or 0)
        effective_iterations = max(0, min(max_iterations, max_hidden_units))
        return output_epochs + effective_iterations * (candidate_epochs + output_epochs)

    def _sync_training_state_from_network(self) -> None:
        """C2b (I-4 root / I-1c): project the live network's effective parameter values into ``TrainingState``.

        ``/v1/network`` (``get_network_info``) and ``GET /v1/training/params``
        (``get_training_params``) read the network object directly, but
        ``/v1/training/status``'s ``training_state`` block is a projected copy — before
        C2b it was seeded once at create time from ``kwargs`` + a second default layer
        and never refreshed, so the two REST surfaces could disagree for the whole life
        of a network. This helper is the single projection point; call it whenever the
        network's parameters may have changed (network create, ``update_params`` apply,
        snapshot load). ``max_epochs`` is the Q1 derived cap (see ``derive_epochs_cap``).
        No-op when no network exists.
        """
        if self.network is None:
            return
        self.training_state.update_state(
            learning_rate=getattr(self.network, "learning_rate", 0.0),
            max_hidden_units=getattr(self.network, "max_hidden_units", 0),
            max_epochs=self.derive_epochs_cap(self.network),
            max_iterations=getattr(self.network, "max_iterations", 0),
        )

    # ------------------------------------------------------------------
    # Monitoring hooks (monkey-patch approach from CascorIntegration)
    # ------------------------------------------------------------------

    def _check_for_interrupt(self) -> None:
        """Raise ``TrainingInterrupted`` on stop request; block on pause.

        Called from ``_handle_event`` on every ``epoch_end`` / ``phase_change``
        event ``CascorModel.fit`` emits during training (the output-epoch and
        grow-iteration boundaries). Pre-2026-05-10 the ``pause_training`` and
        ``stop_training`` REST endpoints were observably no-ops at the
        training-loop level — they updated the FSM but the loop never observed
        the events. This method is the missing hook.

        The pause wait uses a 0.5 s timeout so a stop request received WHILE
        paused is observed promptly (the loop re-checks ``_stop_event`` on
        each wakeup). Without the timeout, a stop after pause would block forever
        since ``_pause_event`` would still be cleared.

        Runs synchronously on the training thread (``_handle_event`` is dispatched
        from CCN's bare callback sites, which don't catch), so raising here
        propagates straight out of ``fit`` — this is how stop/pause rides CCN's
        native hooks (WS-6 PR-B3.3) now that the fit/grow monkey-patches are gone.
        Kept an instance method so tests can invoke ``mgr._check_for_interrupt()``.
        """
        if self._stop_event.is_set():
            raise TrainingInterrupted("stop_requested")
        while not self._pause_event.is_set():
            self._pause_event.wait(timeout=0.5)
            if self._stop_event.is_set():
                raise TrainingInterrupted("stop_requested_during_pause")

    def _handle_event(self, event) -> None:
        """``on_event`` sink for :meth:`CascorModel.fit` (WS-6 PR-B3.3).

        Translates the model-core coarse events the model emits *live* during fit
        (``training_start`` -> ``epoch_end`` / ``phase_change`` -> ``unit_added`` ->
        ``training_end``) into the ``TrainingMonitor`` / ``TrainingState`` updates the read
        routes serialize — replacing the per-epoch / per-iteration projection the removed
        ``monitored_fit`` / ``monitored_grow`` / ``monitored_validate`` monkey-patches used to
        perform. Session-lifecycle bookkeeping (FSM start/terminal transitions, the active-
        session gauge pair, ``on_training_start`` / ``on_training_end``) is owned by
        :meth:`_run_training`, which drives the fit.

        Runs synchronously on the training thread (``on_event`` is dispatched from CCN's bare
        callback sites), so raising ``TrainingInterrupted`` here aborts ``fit`` cleanly — this
        is how stop/pause continues to ride CCN's native hooks now that the fit/grow wrappers
        are gone. The event-type vocabulary is CascorModel.fit's contract (WS-6 PR-B3.2).
        """
        etype = getattr(event, "type", None)
        payload = getattr(event, "payload", None) or {}

        if etype == "epoch_end":
            # Was _output_training_callback (bound to network._output_epoch_callback).
            self._check_for_interrupt()
            metrics = payload.get("metrics", {}) or {}
            epoch = int(payload.get("epoch", 0))
            # C2b counter semantics (I-1c / S12): ``epoch`` here is the INNER
            # output-training epoch within the CURRENT pass (1-based, throttled
            # to every 25th by CCN's ``train_output_layer`` callback; its budget
            # rides in ``payload["epochs"]``). Before C2b this value was written
            # into ``training_state.current_epoch``, racing with
            # ``_extract_and_record_metrics``' training-step write — the same
            # field flip-flopped between e.g. 10000 (inner epoch) and 12 (steps),
            # which is exactly the live "Epoch: 10000 vs 12" header confusion.
            # ``current_epoch`` now has a single writer (the history drain; it
            # counts completed training steps) and the live within-pass progress
            # is exposed under the dedicated ``output_epoch`` /
            # ``output_total_epochs`` pair (the output-phase sibling of
            # ``candidate_epoch`` / ``candidate_total_epochs``).
            self.monitor.on_epoch_end(
                epoch=epoch,
                loss=metrics.get("loss"),
                accuracy=None,
                learning_rate=getattr(self.network, "learning_rate", 0.0),
                hidden_units=len(self.network.hidden_units),
                kind="output_epoch",
            )
            self.training_state.update_state(output_epoch=epoch, output_total_epochs=int(payload.get("epochs", 0)), phase_detail="training_output")
            # METRICS-MON R5.4-pre: train-step duration histogram — delta between successive
            # output-epoch events (perf_counter; robust to wall-clock). The first event of a
            # run seeds the timer and emits no sample.
            now = time.perf_counter()
            prev = self._step_timer_prev
            if prev is not None:
                try:
                    observe_training_step_duration(now - prev)
                except Exception:
                    self.logger.debug("training_step_duration emission failed", exc_info=True)
            self._step_timer_prev = now
            self._extract_and_record_metrics()
            return

        if etype == "phase_change":
            # Was _grow_iteration_callback (bound to network._grow_iteration_callback): per
            # grow-iteration live candidate-pool state. The ``detail`` dict carries the fields
            # CCN's coarse event types would otherwise drop (plan §3.3 — extend the payload).
            self._check_for_interrupt()
            detail = payload.get("detail", {}) or {}
            if not self._grow_phase_entered:
                # First grow iteration: enter the Candidate phase once (monitored_grow set
                # this at grow_network entry).
                self._grow_phase_entered = True
                self.state_machine.set_phase(TrainingPhase.CANDIDATE)
                self.monitor.on_phase_change(self.state_machine.phase.name.lower())
                self.training_state.update_state(phase="Candidate", phase_started_at=datetime.now().isoformat())
                self._broadcast_training_state(force=True)
            self.training_state.update_state(
                grow_iteration=int(detail.get("grow_iteration", 0)),
                grow_max=int(detail.get("max_iterations", 0)),
                best_correlation=detail.get("best_correlation", 0.0),
                candidates_trained=int(detail.get("candidates_trained", 0)),
                candidates_total=int(detail.get("candidates_total", 0)),
                phase_detail=detail.get("phase_detail", "adding_candidate"),
                best_candidate_id=detail.get("best_candidate_id", -1),
                best_candidate_uuid=detail.get("best_candidate_uuid", ""),
                second_candidate_id=detail.get("second_candidate_id"),
                second_candidate_correlation=detail.get("second_candidate_correlation", 0.0),
                all_correlations=detail.get("all_correlations", []),
            )
            self._broadcast_training_state()
            self._extract_and_record_metrics()
            return

        if etype == "unit_added":
            # Was monitored_grow's post-grow_network cascade_add loop. CascorModel emits one
            # unit_added per installed unit (post-hoc, after net.fit), so advancing a cursor
            # through hidden_units reproduces the legacy "all cascade adds batched after
            # growth" timing and stays correct on retrain (baseline = units present at run
            # start). The unit's authoritative correlation is read from the network, matching
            # monitored_grow, rather than from the payload.
            hidden = self.network.hidden_units
            idx = self._cascade_emitted_count
            if 0 <= idx < len(hidden):
                unit = hidden[idx]
                actual_correlation = float(getattr(unit, "best_correlation", 0.0) or 0.0)
                self.monitor.on_cascade_add(hidden_unit_index=idx, correlation=actual_correlation)
                self._cascade_emitted_count = idx + 1
            return

        if etype == "training_end":
            # Was monitored_grow's tail: a single full-topology broadcast after growth, then
            # back to the Output phase. (cascade_add already fired per unit_added above.)
            if self._grow_phase_entered:
                if self._ws_manager is not None:
                    from api.websocket.messages import create_topology_message

                    full_topology = self.get_topology()
                    if full_topology is not None:
                        self._ws_manager.broadcast_from_thread(create_topology_message(full_topology))
                self.state_machine.set_phase(TrainingPhase.OUTPUT)
                self.monitor.on_phase_change(self.state_machine.phase.name.lower())
                # C2b: clear the within-pass progress pairs (candidate AND output)
                # at the growth-phase exit so neither lingers past its phase.
                self.training_state.update_state(phase="Output", phase_detail="", candidate_epoch=0, candidate_total_epochs=0, output_epoch=0, output_total_epochs=0)
                self._broadcast_training_state(force=True)
            self._extract_and_record_metrics()
            return

        # training_start: session setup is owned by _run_training; nothing per-event to
        # project here (training_start is required first by the model-core event contract).

    @staticmethod
    def _drain_progress_queue(network_ref, stop_event, state, monitor, manager_ref):
        """Background thread that reads candidate progress from workers.

        Uses deferred queue discovery: the persistent progress queue is created
        lazily inside grow_network() -> _ensure_worker_pool(), so it may not
        exist when this thread starts. We poll for it until it appears or the
        stop event is set.
        """
        import queue as _queue_mod

        _pq = None
        while not stop_event.is_set():
            # Deferred discovery — queue is created inside grow_network
            if _pq is None:
                _pq = getattr(network_ref, "_persistent_progress_queue", None)
                if _pq is None:
                    try:
                        stop_event.wait(timeout=_PROJECT_API_PROGRESS_QUEUE_WAIT_TIMEOUT)
                    except Exception:
                        break
                    continue
            try:
                progress = _pq.get(timeout=_PROJECT_API_PROGRESS_QUEUE_GET_TIMEOUT)
            except _queue_mod.Empty:
                continue
            except Exception:
                break
            state.update_state(
                phase_detail="training_candidates",
                candidate_epoch=progress.get("epoch", 0),
                candidate_total_epochs=progress.get("total_epochs", 0),
                best_correlation=progress.get("correlation", 0.0),
            )
            monitor.on_candidate_progress(progress)
            manager_ref._broadcast_training_state()

    def _eval_split(self) -> tuple:
        """C7 (U-4): the evaluation split for the scalar metrics — the
        validation/test tensors (``_val_x``/``_val_y``, sourced from the
        dataset's ``X_test``/``y_test``) when present, else the training split.
        Returns ``(None, None)`` when no data is loaded."""
        if self._val_x is not None and self._val_y is not None:
            return self._val_x, self._val_y
        return self._train_x, self._train_y

    def _compute_eval_scalar_metrics(self) -> Optional[Dict[str, Any]]:
        """C7 (U-4): compute F1 / precision / recall / ROC-AUC over the
        evaluation split via a single ``torch.no_grad()`` forward pass.

        Best-effort: returns ``None`` (never raises) when the feature is
        disabled, no network / eval data is available, or the forward pass /
        computation fails — a degraded metric must never crash the drain or the
        training thread. Called OUTSIDE ``_metrics_lock`` so the forward pass
        does not extend that critical section (which ``get_metrics`` also takes)."""
        if not self._eval_metrics_enabled:
            return None
        network = self.network
        x, y = self._eval_split()
        if network is None or x is None or y is None:
            return None
        try:
            with torch.no_grad():
                output = network.forward(x)
            return compute_scalar_classification_metrics(output, y, average=self._eval_metrics_average)
        except Exception:
            self.logger.debug("C7: eval scalar-metrics computation failed", exc_info=True)
            return None

    def _drain_scalars_if_new(self) -> Optional[Dict[str, Any]]:
        """C7 (U-4): cheap, lock-free pre-check gating the eval forward pass.

        Returns the computed scalar metrics only when a new history row is
        (racily) visible, so the frequent within-pass drains that add no row
        skip the forward pass entirely. The read is racy but only gates optional
        work — the authoritative new-row gate stays inside ``_metrics_lock`` in
        :meth:`_extract_and_record_metrics`."""
        try:
            approx_len = len(self.network.history.get("train_loss", []))
        except (RuntimeError, KeyError, AttributeError):
            return None
        if approx_len <= self._last_emitted_history_len:
            return None
        return self._compute_eval_scalar_metrics()

    def _extract_and_record_metrics(self) -> None:
        """Extract NEW metrics from network history and record them.

        Uses a high-water-mark (_last_emitted_history_len) to only emit
        history entries that haven't been emitted yet. Safe to call
        multiple times — idempotent when no new data exists.

        C7 (U-4): when new training-step rows exist, the scalar evaluation
        metrics are computed once (outside ``_metrics_lock``) and attached to the
        TERMINAL new row only — older backfilled rows in the same slice carry
        ``None`` because a single forward pass reflects the network's current
        state, not each historical row. The result is also cached in
        ``_latest_scalar_metrics`` for the ``/v1/metrics`` snapshot.
        """
        if self.network is None or not hasattr(self.network, "history"):
            return

        # C7 (U-4): compute the scalar evaluation metrics only when a new history
        # row is (racily) visible — the forward pass runs outside _metrics_lock.
        scalar_metrics: Optional[Dict[str, Any]] = self._drain_scalars_if_new()

        # CONC-03 / BUG-CC-17 (Phase 3B): the previous implementation
        # released self._metrics_lock between the snapshot+high-water-mark read
        # and the high-water-mark write. Two concurrent callers could both
        # observe the same `last_emitted`, both emit the slice
        # [last_emitted:current_len), and only then race on the write — so each
        # epoch in that slice was reported to TrainingMonitor twice.
        # Hold _metrics_lock across the read, the per-entry on_epoch_end calls,
        # and the high-water-mark advance so the read-process-write cycle is
        # atomic. The training_state update is idempotent and is left outside
        # the lock to keep the critical section bounded.
        with self._metrics_lock:
            try:
                history = self.network.history
                train_loss_list = list(history.get("train_loss", []))
                train_accuracy_list = list(history.get("train_accuracy", []))
                val_loss_list = list(history.get("value_loss", []))
                val_accuracy_list = list(history.get("value_accuracy", []))
                hidden_units_count = len(self.network.hidden_units)
                last_emitted = self._last_emitted_history_len
            except (RuntimeError, KeyError):
                return

            current_len = len(train_loss_list)
            if current_len <= last_emitted:
                return  # No new data

            # Emit all new entries
            for i in range(last_emitted, current_len):
                epoch = i + 1
                # C7 (U-4): the scalars reflect one forward pass over the current
                # network, so attach them to the TERMINAL new row only; earlier
                # backfilled rows keep the nullable fields at None.
                row_scalars = scalar_metrics if i == current_len - 1 else None
                self.monitor.on_epoch_end(
                    epoch=epoch,
                    loss=train_loss_list[i],
                    accuracy=train_accuracy_list[i] if i < len(train_accuracy_list) else None,
                    learning_rate=getattr(self.network, "learning_rate", 0.0),
                    hidden_units=hidden_units_count,
                    validation_loss=val_loss_list[i] if i < len(val_loss_list) else None,
                    validation_accuracy=val_accuracy_list[i] if i < len(val_accuracy_list) else None,
                    scalar_metrics=row_scalars,
                )
                # OBS-WIRE-01 (A.2): bump the per-phase epoch counter
                # exactly once per newly-emitted history row. Counters
                # MUST NOT be throttled (rate() would under-count by
                # the throttle factor); we emit one increment per row
                # here, mirroring the high-water-mark advance.
                try:
                    record_training_epoch(phase="output")
                    if i < len(val_loss_list):
                        # Validation pass also constitutes a "training
                        # epoch" from the SLI perspective. Keep the
                        # phase distinction so the validation-vs-train
                        # ratio stays observable.
                        record_training_epoch(phase="validation")
                except Exception:
                    self.logger.debug("record_training_epoch emission failed", exc_info=True)

            # OBS-WIRE-01 (A.2): set the loss / accuracy / hidden-units
            # gauges from the latest history row. Gauges are last-value
            # observations, so a single set() per drain is sufficient
            # — the per-row emit-loop above advances the underlying
            # counter; here we only need the terminal sample. ``last``
            # is the index of the newest entry (current_len - 1).
            last = current_len - 1
            try:
                set_training_loss(phase="output", loss_type="train", value=float(train_loss_list[last]))
                if last < len(train_accuracy_list) and train_accuracy_list[last] is not None:
                    set_training_accuracy(phase="output", value=float(train_accuracy_list[last]))
                # P-23: validation history is typically shorter than the
                # training history (validation may run every K epochs).
                # Indexing the validation lists with the training-side
                # ``last = current_len - 1`` skips the gauge update
                # whenever the latest training epoch had no validation
                # pass — the test
                # ``test_drain_emits_loss_accuracy_hidden_units_and_counter``
                # exercises that case. Gauges are "most recent
                # observation", so emit the last validation entry
                # whenever any exists. Single-conditional form keeps the
                # function under flake8 C901's complexity cap.
                if val_loss_list and val_loss_list[-1] is not None:
                    set_training_loss(phase="output", loss_type="validation", value=float(val_loss_list[-1]))
                if val_accuracy_list and val_accuracy_list[-1] is not None:
                    set_training_accuracy(phase="validation", value=float(val_accuracy_list[-1]))
                set_hidden_units(int(hidden_units_count))
            except Exception:
                self.logger.debug("training gauge emission failed", exc_info=True)

            # Advance the high-water-mark before releasing the lock — this is
            # the second half of the formerly-split section.
            self._last_emitted_history_len = current_len

            # C7 (U-4): cache the latest computed scalars for the /v1/metrics
            # snapshot (``get_metrics``). Only overwrite when we actually
            # computed a result this drain, so a transient computation failure
            # leaves the previous snapshot value intact rather than blanking it.
            if scalar_metrics is not None:
                self._latest_scalar_metrics = scalar_metrics

        # C2b counter semantics: ``current_epoch`` / ``current_step`` count
        # completed TRAINING STEPS — entries in the engine's ``history``
        # arrays, i.e. one initial output-training pass plus one per cascade
        # growth iteration — NOT inner output-training epochs. This drain is
        # the single writer for both fields (the ``epoch_end`` handler exposes
        # within-pass progress under ``output_epoch``/``output_total_epochs``
        # instead of racing on ``current_epoch`` — see ``_handle_event``).
        self.training_state.update_state(
            current_epoch=current_len,
            current_step=current_len,
        )

    # ------------------------------------------------------------------
    # Training control
    # ------------------------------------------------------------------

    # Network.fit()'s narrow signature — anything outside this set raises
    # TypeError if passed to fit(**kwargs). TrainingParams is intentionally
    # broader (covers every runtime-tunable param), so start_training has
    # to split the request body into "fit-shaped" and "network-attribute"
    # kwargs and route them through different paths. See
    # juniper-ml/notes/JUNIPER_2026-04-30_JUNIPER-CASCOR_FIT-KWARGS-LATENT-BUG.md for the full trace
    # and rationale (Option 1 — filter at the start_training boundary).
    _FIT_KWARGS: frozenset = frozenset({"max_epochs", "epochs", "max_iterations", "early_stopping"})

    def _attach_weight_history_recorder(self) -> None:
        """CAN-015g (g-6): instantiate + register the weight history recorder.

        Idempotent — call before each training run. The recorder
        reads ``config.weight_history_sampling_interval`` /
        ``config.weight_history_max_samples`` at attach time, so a
        runtime PATCH /v1/training/params change before this point
        affects subsequent samples (changes mid-run land at the next
        ``register`` call when re-attached).
        """
        if self.network is None or self.monitor is None:
            return
        existing = self._weight_history_recorder
        if existing is None or existing.network is not self.network:
            self._weight_history_recorder = _WeightHistoryRecorder(self.network, self.monitor)
        else:
            # Re-init in case config tunables changed since last training run.
            existing.sampling_interval = int(getattr(self.network.config, "weight_history_sampling_interval", existing.sampling_interval))
            existing.max_samples = int(getattr(self.network.config, "weight_history_max_samples", existing.max_samples))
            existing._init_weight_history()
        self._weight_history_recorder.register()

    def start_training(
        self,
        X: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        *,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        start_fresh: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Start training asynchronously.

        Args:
            X: Training features tensor
            y: Training targets tensor
            X_val: Validation features
            y_val: Validation targets
            start_fresh: C5 (Q4 use-case 2 / U-1) — when True, DISCARD the
                current model and all retained metrics/history before the run
                (a clean-launch-equivalent reset via
                :meth:`_start_fresh_reset_locked`) so training begins with a
                vanilla, untrained network created from the dataset dims.
                On-disk snapshot artifacts are never touched. When False
                (default) the current model and its metrics/history are
                RETAINED, so training continues the existing model — the
                cross-dataset continual-training use case (Q4 use-case 1).
            **kwargs: TrainingParams body. Fields in ``_FIT_KWARGS`` are
                forwarded to ``network.fit``; everything else is applied
                in-place via ``update_params`` so the next fit pass sees
                the new values. Unknown keys (not in fit and not in
                ``update_params``' whitelist) raise immediately so a
                typo at the API boundary fails loud rather than getting
                swallowed on the background thread.

        Returns:
            Status dictionary
        """
        with self._lock:
            if self.state_machine.is_started():
                raise RuntimeError("Training already in progress")
            # CAN-015d (B-4): Investigating is the inspection / modification
            # mode loaded by ``/restore``. Training commands are explicitly
            # rejected — the user must invoke ``/retrain`` or ``/resume`` to
            # transition out of Investigating before starting training.
            # Failing fast at the API boundary is much clearer than
            # letting the future submit and the FSM transition fail
            # silently on the background training thread.
            if self.state_machine.is_investigating():
                raise RuntimeError("Cannot start training while Investigating a snapshot — invoke /v1/snapshots/{id}/retrain or /resume to transition out of Investigating first")
            # CAN-015c (B-3): Replaying is read-only playback. Same
            # rejection contract — user must /replay/control stop first.
            if self.state_machine.is_replaying():
                raise RuntimeError("Cannot start training while replaying a snapshot — invoke /v1/snapshots/{id}/replay/control with action='stop' first")

            if X is not None:
                self._train_x = X
                self._train_y = y
            if X_val is not None:
                self._val_x = X_val
                self._val_y = y_val

            # FRONTEND_ISSUES_PLAN_2026-05-09 §3.5.1 / Issue #3 Phase 1 — if
            # the user staged a dataset change while training was stopped,
            # consume it now (before the future is submitted). On reload
            # failure, leave the staged config in place so the user can fix
            # the upstream juniper-data issue and Restart-and-retry without
            # losing their selection.
            if self._pending_dataset_config:
                self._reload_dataset(**self._pending_dataset_config)
                self._pending_dataset_config = None

            if self._train_x is None or self._train_y is None:
                raise ValueError("Training data not provided")

            # C5 (Q4 use-case 2 / U-1): a start-fresh run discards the current
            # model + all retained metrics/history HERE — after the data guard
            # (so a dataless start doesn't throw the model away on a doomed
            # start) and before the create-on-start block below, which then
            # rebuilds a vanilla network from the (new) dataset dims. On-disk
            # snapshots are preserved. Default (start_fresh=False) leaves the
            # current model + metrics/history intact — continue-training.
            if start_fresh:
                self._start_fresh_reset_locked()

            # Training-start diagnosis 2026-07-09 (PR-B): with ``auto_start``
            # defaulting off, nothing creates a network before the first
            # user-initiated start — every UI start on a fresh cascor died on
            # the old pre-lock "No network created" guard. Mirror
            # ``_auto_start_training``'s inference instead: size the network
            # from the actual training arrays (the staged/pending dataset was
            # consumed above, so the dims are authoritative). A bare start with
            # neither data nor a staged dataset still fails loud on the
            # "Training data not provided" check above.
            if self.network is None:
                create_cfg = {
                    "input_size": self._train_x.shape[1],
                    "output_size": self._train_y.shape[1] if self._train_y.dim() > 1 else 1,
                }
                self.logger.info(
                    "start_training: no network — creating %sx%s from dataset dims",
                    create_cfg["input_size"],
                    create_cfg["output_size"],
                )
                self._create_network_locked(**create_cfg)

            # P2-1d: cold-swap parity with swap_dataset_live. If the dataset
            # is smaller than the network's input/output dims (e.g., after a
            # snapshot-load left the network at a larger size than a new
            # smaller dataset), zero-pad up to the network's dims and set
            # the loss-mask depth so training doesn't pull gradients toward
            # zero on the dead output slots. start_training does NOT grow
            # the network — only swap_dataset_live owns the grow path; if
            # the dataset exceeds network capacity here, the pad helper
            # raises ValueError and the user must recreate the network or
            # use the live-swap path.
            if hasattr(self.network, "input_size") and hasattr(self.network, "output_size"):
                (
                    self._train_x,
                    self._train_y,
                    self._val_x,
                    self._val_y,
                    _active_input_dim,
                    active_output_dim,
                ) = self._pad_dataset_for_network(self._train_x, self._train_y, self._val_x, self._val_y)
                if hasattr(self.network, "active_output_dim"):
                    self.network.active_output_dim = active_output_dim

            self._stop_event.clear()
            self._pause_event.set()

            # CAS-006 (A-4) + CAN-015b (B-2): each training run normally
            # starts fresh — we don't want a snapshot from a previous run's
            # accuracy ceiling to suppress auto-snaps in this run. EXCEPTION:
            # when the FSM is RESUME_READY we're continuing a snapshotted
            # run, so the loaded ratchet stays as the baseline (a re-snap
            # only fires when the resumed training truly beats the prior
            # run's best). We also clear the resume marker once consumed
            # so a stop-then-restart-without-resume doesn't carry it over.
            resuming = self.state_machine.is_resume_ready()
            if not resuming:
                with self._auto_snap_lock:
                    self._auto_snap_best_metric = None
            else:
                self._resume_point_epoch = None

            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cascor-train")

            fit_kwargs = {k: v for k, v in kwargs.items() if k in self._FIT_KWARGS}
            network_kwargs = {k: v for k, v in kwargs.items() if k not in self._FIT_KWARGS and v is not None}

            # Apply network-attribute kwargs in-place BEFORE submitting the
            # training future so the next fit pass observes the new values.
            # ``_apply_params_unlocked`` shares the same whitelist + atomic-
            # rollback path as ``update_params``; calling it here while we
            # hold ``_lock`` avoids re-entering the non-reentrant
            # lock and avoids the race where the background thread could
            # start fit() before update_params lands.
            if network_kwargs:
                self._apply_params_unlocked(network_kwargs)

            # CAN-015g (g-6): attach the weight-history recorder before
            # the training future runs so the first epoch_end /
            # cascade_add events get captured. Lazy attach: the
            # recorder reads ``config.weight_history_*`` so a runtime
            # PATCH /v1/training/params change before this point lands
            # in the recorder's snapshot of those values.
            self._attach_weight_history_recorder()

            # C5 (Q4/U-1): decide this run's metrics/history retention posture
            # (read by ``_run_training`` on the background thread):
            #   * plain start (default): RETAIN — the buffer carries across the
            #     run boundary and ``_run_training`` baselines the history
            #     high-water-mark at the existing length so only THIS run's new
            #     rows are appended (no re-emit/duplication).
            #   * resume (RESUME_READY): rebuild the buffer from the full loaded
            #     history — the pre-C5 clear + re-emit-from-zero path (unchanged
            #     snapshot-resume semantics).
            #   * start-fresh: the buffer was emptied above and the network is
            #     vanilla (empty history), so this collapses to the same clear +
            #     zero-baseline path.
            self._retain_metrics_next_run = not resuming and not start_fresh
            # Starting a run finalizes any pending explicit clear: drop the undo
            # snapshot so ``undo_clear_metrics`` after this point is a 409.
            with self._metrics_undo_lock:
                self._metrics_undo_buffer = None

            self._training_future = self._executor.submit(self._run_training, self._train_x, self._train_y, self._val_x, self._val_y, **fit_kwargs)

        return {"status": "training_started", "timestamp": time.time()}

    def _run_training(self, x, y, x_val, y_val, **kwargs) -> None:
        """Execute training in the background thread (submitted by ``start_training``).

        WS-6 PR-B3.3: drives the model-core ``CascorModel.fit`` with the ``on_event`` sink
        (:meth:`_handle_event`) and owns the session lifecycle the removed ``monitored_fit``
        monkey-patch used to wrap — FSM start/terminal transitions, the active-session gauge
        pair (OBS-WIRE-01), ``on_training_start`` / ``on_training_end``, and the candidate-
        progress drain thread (the async 50 Hz ``/ws/training`` side-channel that cannot ride
        the synchronous ``on_event`` — plan H4). Per-epoch / per-iteration projection is done
        by ``_handle_event``.

        Exceptions propagate to the training future: a clean stop raises
        ``TrainingInterrupted`` (swallowed here as a successful cancellation, returning None);
        any other error transitions to Failed and re-raises so ``future.result()`` surfaces it.
        """
        monitor = self.monitor
        state = self.training_state
        sm = self.state_machine
        stop_event = self._stop_event

        # Reset per-run bookkeeping (was monitored_fit's per-fit reset + the _step_timer box).
        # _cascade_emitted_count baselines at the units already present so a retrain only emits
        # cascade_add for units grown this run.
        # C5 (Q4/U-1): on a metrics-RETAINING run, baseline the history high-water-mark at the
        # rows already emitted into the (retained) buffer so this run EXTENDS it with only its
        # new rows — re-emitting from 0 while retaining would duplicate the prior run's tail.
        # Mirrors _cascade_emitted_count. A fresh / start-fresh / resume-rebuild run uses 0.
        retain_metrics = self._retain_metrics_next_run
        self._last_emitted_history_len = self._current_history_len() if retain_metrics else 0
        self._step_timer_prev = None
        self._grow_phase_entered = False
        self._cascade_emitted_count = len(self.network.hidden_units)
        # C7 (U-4): drop the previous run's cached scalar snapshot so /v1/metrics
        # reports None until this run's first drain recomputes it (the retained
        # history rows keep their own already-computed scalars).
        self._latest_scalar_metrics = None

        # Session start (was monitored_fit's pre-fit block).
        monitor.on_training_start(retain_metrics=retain_metrics)
        # BUG-CC-07: phase via the state-machine command, then notify the monitor.
        sm.handle_command(Command.START)
        monitor.on_phase_change(sm.phase.name.lower())
        # C2b: reset the within-pass progress pairs at run start so a new run
        # never displays the previous run's terminal inner-epoch values.
        state.update_state(status="Started", phase="Output", phase_started_at=datetime.now().isoformat(), output_epoch=0, output_total_epochs=0, candidate_epoch=0, candidate_total_epochs=0)
        self._broadcast_training_state(force=True)
        # OBS-WIRE-01 (A.1): mark the session active; balanced by dec in the finally so the
        # gauge (which gates TrainingStalled / TrainingLossNotDecreasing / LowCandidateCorrelation
        # alerts) returns to zero across normal, cancelled, and failure terminal paths.
        try:
            inc_training_sessions()
        except Exception:
            self.logger.debug("inc_training_sessions emission failed", exc_info=True)

        # Candidate-progress drain (retained side-channel; deferred queue discovery — the
        # persistent progress queue is created lazily inside grow_network). Started once around
        # the whole fit rather than per grow_network call (monitored_grow's old home); the drain
        # polls until the queue appears, so starting it before grow is safe.
        drain_stop = threading.Event()
        drain_thread = threading.Thread(
            target=TrainingLifecycleManager._drain_progress_queue,
            args=(self.network, drain_stop, state, monitor, self),
            daemon=True,
            name="candidate-progress-drain",
        )
        drain_thread.start()

        try:
            # CCN.fit defaults early_stopping=True; the removed monkey-patch path
            # (self.network.fit(**fit_kwargs)) inherited that default whenever the API body
            # omitted it. CascorModel.fit (WS-6 PR-B3.2) defaults early_stopping to False, so
            # set CCN's default explicitly here to keep training — and the golden post-train
            # topology — behavior-identical. An explicit early_stopping in the body is honored.
            # (Seam note: the A-phase's generic ``self.model.fit(...)`` call should likewise
            # pass early_stopping, or CascorModel.fit's default be aligned to CCN's, so the
            # generic manager inherits this behavior.)
            kwargs.setdefault("early_stopping", True)
            self.model.fit(x, y, X_val=x_val, y_val=y_val, on_event=self._handle_event, **kwargs)

            # Catch any remaining metrics + capture terminal weights (was monitored_fit's
            # post-original_fit block).
            self._extract_and_record_metrics()
            if self._weight_history_recorder is not None:
                self._weight_history_recorder.capture_terminal()

            if stop_event.is_set():
                sm.handle_command(Command.STOP)
                state.update_state(status="Stopped", phase="Idle")
                self._broadcast_training_state(force=True)
                inc_training_session_completed(TRAINING_SESSION_STATUS_CANCELLED)
            else:
                sm.mark_completed()
                state.update_state(status="Completed", phase="Idle")
                self._broadcast_training_state(force=True)
                inc_training_session_completed(TRAINING_SESSION_STATUS_SUCCESS)
        except TrainingInterrupted:
            # Clean cancellation: _handle_event raised on a stop request from inside a CCN
            # callback. Same terminal transitions as the post-fit stop path; not an error.
            sm.handle_command(Command.STOP)
            state.update_state(status="Stopped", phase="Idle")
            self._broadcast_training_state(force=True)
            inc_training_session_completed(TRAINING_SESSION_STATUS_CANCELLED)
        except Exception as e:
            sm.mark_failed(str(e))
            state.update_state(status="Failed", phase="Idle")
            self._broadcast_training_state(force=True)
            inc_training_session_completed(TRAINING_SESSION_STATUS_FAILURE)
            raise
        finally:
            drain_stop.set()
            drain_thread.join(timeout=_PROJECT_API_DRAIN_THREAD_JOIN_TIMEOUT)
            try:
                dec_training_sessions()
            except Exception:
                self.logger.debug("dec_training_sessions emission failed", exc_info=True)
            monitor.on_training_end()

    def stop_training(self) -> Dict[str, Any]:
        """Request training stop.

        Idempotent when already Stopped / Completed / Failed (FSM reject
        is ignored; callers still receive ``stop_requested``). Rejected
        with ``RuntimeError`` while Investigating or Replaying so
        ``training_state`` cannot report Stopped while the FSM still
        blocks ``start_training``.
        """
        if self.state_machine.is_investigating() or self.state_machine.is_replaying():
            raise RuntimeError(f"Cannot stop training while {self.state_machine.status.name}")
        self._stop_event.set()
        transitioned = self.state_machine.handle_command(Command.STOP)
        if transitioned:
            self.training_state.update_state(status="Stopped", phase="Idle")
            self._broadcast_training_state(force=True)
        return {"status": "stop_requested", "timestamp": time.time()}

    def pause_training(self) -> Dict[str, Any]:
        """Pause training."""
        if not self.state_machine.is_started():
            raise RuntimeError("Training is not active")
        self._pause_event.clear()
        self.state_machine.handle_command(Command.PAUSE)
        self.training_state.update_state(status="Paused")
        self._broadcast_training_state(force=True)
        return {"status": "paused", "timestamp": time.time()}

    def resume_training(self) -> Dict[str, Any]:
        """Resume paused training."""
        if not self.state_machine.is_paused():
            raise RuntimeError("Training is not paused")
        self._pause_event.set()
        self.state_machine.handle_command(Command.RESUME)
        self.training_state.update_state(status="Started")
        self._broadcast_training_state(force=True)
        return {"status": "resumed", "timestamp": time.time()}

    def reset(self) -> Dict[str, Any]:
        """Reset training state.

        Normalises the control-event pair via ``_reset_event_state`` so a
        subsequent ``start_training`` does not inherit a stale
        ``_pause_event.clear()`` from a prior pause (BUG-CC-#5).
        """
        self._reset_event_state()
        self._last_emitted_history_len = 0
        self.state_machine.handle_command(Command.RESET)
        self.monitor.clear_metrics()
        self.training_state.update_state(
            status="Stopped",
            phase="Idle",
            current_epoch=0,
            current_step=0,
        )
        self._broadcast_training_state(force=True)
        return {"status": "reset", "timestamp": time.time()}

    def _reset_event_state(self) -> None:
        """Single source of truth for control-event normalisation.

        Post-condition: ``_stop_event`` is set (signals any in-flight
        training thread to stop) and ``_pause_event`` is set (no synthetic
        pause inherited by the next ``start_training`` call).
        """
        self._stop_event.set()
        self._pause_event.set()

    # ------------------------------------------------------------------
    # C5 (Q4/U-1) — metrics/history retention, explicit clear + undo,
    # and start-fresh (clean-launch) reset.
    # ------------------------------------------------------------------

    def _current_history_len(self) -> int:
        """Length of the live network's primary per-epoch series (``train_loss``).

        The C5 retained-run high-water-mark baseline: a retaining run resumes
        ``_last_emitted_history_len`` here so the buffer is EXTENDED with only
        the new rows this run appends (mirrors ``_extract_and_record_metrics``'
        ``train_loss`` indexing and the ``_cascade_emitted_count`` baseline).
        Returns 0 when there is no network / no history (e.g. a fresh or
        start-fresh-reset network), which reproduces the pre-C5 baseline.
        """
        net = self.network
        if net is None or not hasattr(net, "history"):
            return 0
        try:
            return len(net.history.get("train_loss", []))
        except (AttributeError, TypeError):
            return 0

    def _metrics_undo_available(self) -> bool:
        """True when an explicit metrics clear can still be undone (C5)."""
        with self._metrics_undo_lock:
            return self._metrics_undo_buffer is not None

    def clear_metrics_with_undo(self) -> Dict[str, Any]:
        """C5 (Q4 use-case 1): explicitly clear the retained metrics/history,
        stashing the cleared rows so the clear can be reversed with
        :meth:`undo_clear_metrics` at any point until the next run starts.

        The undo snapshot holds at most ``_PROJECT_API_METRICS_BUFFER_SIZE``
        (10000) rows — the same bound as the live buffer — so a pending undo
        costs at most one extra buffer's worth of memory. Starting a training
        run (:meth:`start_training`) finalizes the clear and drops the
        snapshot. Distinct from :meth:`reset` (which also clears counters + the
        FSM) — this touches metrics/history only.
        """
        with self._metrics_undo_lock:
            cleared = list(self.monitor.get_all_metrics())
            self.monitor.clear_metrics()
            self._metrics_undo_buffer = cleared
            count = len(cleared)
        self.logger.info("Metrics cleared with undo (%d rows stashed)", count)
        return {"status": "cleared", "cleared_count": count, "undo_available": True}

    def undo_clear_metrics(self) -> Dict[str, Any]:
        """C5 (Q4 use-case 1 fallback): restore the metrics/history removed by
        the most recent :meth:`clear_metrics_with_undo`.

        Valid until the next training run starts. Raises
        :class:`NoMetricsUndoError` when no undo is available (nothing was
        cleared, or a run has started since and finalized the clear).
        """
        with self._metrics_undo_lock:
            if self._metrics_undo_buffer is None:
                raise NoMetricsUndoError("No metrics clear to undo (nothing cleared, or a training run has started since)")
            rows = self._metrics_undo_buffer
            self.monitor.restore_metrics(rows)
            self._metrics_undo_buffer = None
            count = len(rows)
        self.logger.info("Metrics clear undone (%d rows restored)", count)
        return {"status": "restored", "restored_count": count, "undo_available": False}

    def _start_fresh_reset_locked(self) -> None:
        """C5 (Q4 use-case 2 / U-1): clean-launch reset for a start-fresh run.

        Discards the in-memory model and every piece of retained training data
        so the next ``start_training`` recreates a vanilla, untrained network
        from the dataset dims — functionally identical to a fresh stack launch
        EXCEPT for artifacts with a permanence expectation. **Snapshot files on
        disk (``_get_snapshots_dir``) are NEVER touched by this path** — a
        start-fresh discards the working model, not the operator's saved
        snapshots.

        Assumes ``self._lock`` is held (called inline from ``start_training``);
        combines ``delete_network`` + ``reset`` + ``restore_for_retrain``'s
        reset scope without re-entering the non-reentrant lock.
        """
        # Discard the working model (mirrors delete_network under the held lock).
        self.model = None
        self._params = None
        # Clear retained metrics/history and drop any pending undo — a fresh
        # start supersedes an in-flight explicit clear.
        self.monitor.clear_metrics()
        with self._metrics_undo_lock:
            self._metrics_undo_buffer = None
        self._last_emitted_history_len = 0
        # Fresh auto-snap ratchet + no carried-over resume marker.
        with self._auto_snap_lock:
            self._auto_snap_best_metric = None
        self._resume_point_epoch = None
        # FSM + counters back to the clean-launch baseline.
        self.state_machine.handle_command(Command.RESET)
        self.training_state.update_state(status="Stopped", phase="Idle", current_epoch=0, current_step=0)
        self.logger.info("start_fresh: discarded model + cleared retained metrics/history (snapshots on disk preserved)")

    # ------------------------------------------------------------------
    # Status & metrics
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        state_summary = self.state_machine.get_state_summary()
        monitor_state = self.monitor.get_current_state()
        training_state = self.training_state.get_state()

        if self.network is not None:
            training_state.setdefault("input_size", getattr(self.network, "input_size", 0))
            training_state.setdefault("output_size", getattr(self.network, "output_size", 0))

        return {
            "state_machine": state_summary,
            "monitor": monitor_state,
            "training_state": training_state,
            "network_loaded": self.network is not None,
            "training_active": self.state_machine.is_started(),
            # FRONTEND_ISSUES_PLAN_2026-05-09 §3.5.1 / Issue #3 Phase 1 — drives
            # the canopy "Dataset change pending — restart training to apply"
            # banner without canopy having to poll a separate route.
            "pending_dataset": self.get_pending_dataset_config(),
            # Issue #3 diagnosability follow-up: which grow_network exit fired on
            # the last growth run (residual_collapsed / no_candidate /
            # below_threshold / early_stopped / max_iterations), or None before
            # any training. Lets canopy distinguish a genuine convergence from a
            # 0-unit stall instead of both showing a bare "Completed".
            "completion_reason": getattr(self.network, "_completion_reason", None),
            # C5 (Q4/U-1): additive — True while an explicit metrics clear
            # (POST /v1/training/metrics/clear) can still be undone (i.e. no run
            # has started since). Lets canopy render the undo affordance across
            # a page reload without a separate poll. Additive field only.
            "metrics_clear_undo_available": self._metrics_undo_available(),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        if self.network is None:
            return {}
        with self._metrics_lock:
            try:
                history = self.network.history
                train_loss = list(history.get("train_loss", []))
                train_accuracy = list(history.get("train_accuracy", []))
                val_loss = list(history.get("value_loss", []))
                val_accuracy = list(history.get("value_accuracy", []))
                hidden_units = len(self.network.hidden_units)
            except (RuntimeError, KeyError):
                return {}

        # C7 (U-4): the latest scalar evaluation metrics computed over the
        # evaluation split (see ``_extract_and_record_metrics``). Additive and
        # nullable — the flat ``f1``/``precision``/``recall``/``roc_auc`` fields
        # sit alongside loss/accuracy where consumers already read them, and a
        # self-describing ``eval_metrics`` block records the averaging strategy,
        # the split used, sample/class counts, whether the feature is enabled,
        # and any per-metric undefined reasons. ``None`` before the first drain
        # of a run and whenever computation is disabled/unavailable.
        latest = self._latest_scalar_metrics
        scalars = latest if isinstance(latest, dict) else {}
        eval_x, _eval_y = self._eval_split()
        eval_split = "validation" if (self._val_x is not None and self._val_y is not None) else ("training" if eval_x is not None else None)

        return {
            "epoch": len(train_loss),
            "train_loss": train_loss[-1] if train_loss else None,
            "train_accuracy": train_accuracy[-1] if train_accuracy else None,
            "val_loss": val_loss[-1] if val_loss else None,
            "val_accuracy": val_accuracy[-1] if val_accuracy else None,
            "hidden_units": hidden_units,
            "timestamp": datetime.now().isoformat(),
            # C7 (U-4) flat scalar evaluation metrics (nullable).
            "f1": scalars.get("f1"),
            "precision": scalars.get("precision"),
            "recall": scalars.get("recall"),
            "roc_auc": scalars.get("roc_auc"),
            # C7 (U-4) self-describing metadata for the scalar metrics.
            "eval_metrics": {
                "enabled": self._eval_metrics_enabled,
                "average": scalars.get("average", self._eval_metrics_average),
                "split": eval_split,
                "n_samples": scalars.get("n_samples"),
                "n_classes": scalars.get("n_classes"),
                "undefined": scalars.get("undefined", {}),
            },
        }

    def get_metrics_history(self, count: Optional[int] = None) -> list:
        """Get metrics history."""
        if count:
            return self.monitor.get_recent_metrics(count)
        return self.monitor.get_all_metrics()

    def has_training_data(self) -> bool:
        """Check if training data is loaded."""
        return self._train_x is not None and self._train_y is not None

    # ------------------------------------------------------------------
    # Pending dataset config (FRONTEND_ISSUES_PLAN_2026-05-09 §3.5.1 + §3.5.2 P1)
    # ------------------------------------------------------------------

    def stage_dataset_config(self, **cfg: Any) -> Dict[str, Any]:
        """Stage a dataset-config change for the next ``start_training`` call.

        The actual fetch + tensor swap happens in ``_reload_dataset`` at the
        top of ``start_training``; staging just records intent so the canopy
        banner can announce "Dataset change pending" until the user restarts.

        Empty ``cfg`` is a no-op (clears any prior staging — same shape as
        ``clear_pending_dataset_config``) so the route can use ``cfg or {}``
        without separate validation.
        """
        with self._lock:
            if not cfg:
                self._pending_dataset_config = None
                return {"status": "cleared", "config": None}
            self._pending_dataset_config = dict(cfg)
            return {"status": "staged", "config": dict(cfg)}

    def clear_pending_dataset_config(self) -> Dict[str, Any]:
        """Discard any staged dataset change so the next start uses current data."""
        with self._lock:
            prior = self._pending_dataset_config
            self._pending_dataset_config = None
        return {"status": "cleared", "discarded": dict(prior) if prior else None}

    def get_pending_dataset_config(self) -> Optional[Dict[str, Any]]:
        """Return the staged dataset config (or None) — drives the canopy banner."""
        cfg = self._pending_dataset_config
        return dict(cfg) if cfg else None

    # P2-2 Follow-up B (Issue #3): canopy P2-7 timeline UI reads dataset_swap
    # events via this lifecycle method without pulling a full snapshot.
    # The method is read-only and returns a fresh list copy so a caller
    # iterating the result can't mutate persisted history through the
    # reference. See notes/PHASE_2_P2_2_FOLLOWUPS_2026-05-14.md.

    def get_dataset_swap_events(self, since: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return the network's ``dataset_swap`` history events.

        Args:
            since: Optional ISO-8601 timestamp. If supplied, only events whose
                ``timestamp`` is STRICTLY greater than ``since`` are returned
                (lexicographic compare on the ISO-8601 string is correct for
                UTC timestamps in the canonical format the recorder uses).
                Events with no timestamp (e.g. loaded from a malformed
                snapshot) are excluded when ``since`` is set, included
                otherwise.

        Returns:
            A list of event dicts in chronological order, copied so the
            caller can iterate or mutate the result without affecting
            persisted history. Returns ``[]`` when no network is loaded.
        """
        if self.network is None or not hasattr(self.network, "history"):
            return []
        events = list(self.network.history.get("dataset_swaps", []) or [])
        # Deep-copy each event so callers iterating the result can mutate
        # nested dicts (``arch_changes`` and ``appended_nodes``) without
        # corrupting persisted history. Matches the recorder's own
        # deep-copy guarantee (``record_dataset_swap_event``). Event
        # payloads are small dicts, so the cost is negligible relative
        # to the safety it provides.
        if since is None:
            return [copy.deepcopy(e) for e in events]
        return [copy.deepcopy(e) for e in events if isinstance(e.get("timestamp"), str) and e["timestamp"] > since]

    # ------------------------------------------------------------------
    # Phase 2: experimental-functions gate + live dataset swap (P2-1a)
    # See ISSUE_3_PHASE_2_LIVE_DATASET_SWAP_2026-05-09.md §3.1/§3.2/§3.7/§3.8.
    # ------------------------------------------------------------------

    def get_experimental_functions(self) -> bool:
        """Return whether the experimental-functions gate is open.

        Authoritative server-side state per F2.10: a stale frontend toggle
        cannot bypass this. Initial value comes from the
        ``CASCOR_EXPERIMENTAL_FUNCTIONS_ENABLED`` env var (``=1`` means open).
        """
        return self._experimental_functions_enabled

    def set_experimental_functions(self, enabled: bool) -> Dict[str, Any]:
        """Open or close the experimental-functions gate.

        Returns the new state in a dict suitable for direct JSON response.
        The admin route is access-controlled separately (existing
        ``JUNIPER_DATA_API_KEY`` mechanism or equivalent) so this method
        does not re-validate authorisation.
        """
        self._experimental_functions_enabled = bool(enabled)
        return {"experimental_functions_enabled": self._experimental_functions_enabled}

    def request_swap_cancel(self) -> Dict[str, Any]:
        """Signal an in-flight ``swap_dataset_live`` to abort and roll back.

        Backed by ``_swap_cancel_requested`` (a ``threading.Event``). The flag
        is observed at the next checkpoint inside ``swap_dataset_live``
        (post-fetch and pre-future-resubmit), which raises ``SwapCancelledError``
        to trip the existing §3.8 rollback path.

        Returns a small descriptor dict on success. Raises
        ``NoSwapInProgressError`` (route → 404) if no swap is currently in
        flight — important UX signal so the canopy "Cancel" button doesn't
        falsely succeed against a swap that already finished racing the click.

        The cancel is asynchronous: an in-flight ``_reload_dataset`` HTTP fetch
        cannot be interrupted mid-syscall, so the swap may take up to one
        full fetch RTT to observe the flag. The DELETE response means
        "signal accepted", not "swap aborted by the time you read this".
        """
        # Snapshot the flag under the lock — the swap mutates it inside
        # ``_lock`` so reading without the lock could see a stale
        # False during the brief window between swap-start and the very first
        # checkpoint. Holding the lock here also prevents a race where a swap
        # finishes between our check and our set().
        with self._lock:
            if not self._swap_in_progress:
                raise NoSwapInProgressError("no_swap_in_progress")
            self._swap_cancel_requested.set()
            return {"status": "cancel_requested"}

    def _check_swap_cancel(self) -> None:
        """Raise ``SwapCancelledError`` if a cancel has been signalled.

        Called at safe checkpoints inside ``swap_dataset_live`` where state
        is consistent enough that the standard §3.8 rollback path can restore
        the pre-swap snapshot. The first checkpoint sits immediately after
        ``_reload_dataset`` returns — the most likely point a user-initiated
        cancel will land, since the fetch is the long pole of a swap.
        """
        if self._swap_cancel_requested.is_set():
            raise SwapCancelledError("swap_cancelled_by_client")

    def swap_dataset_live(self, **cfg: Any) -> Dict[str, Any]:  # noqa: C901
        """In-flight dataset swap (P2-1a equal-dim skeleton + P2-1b polish).

        Phase 2 step-by-step flow per spec §3.2:
          1. Acquire ``_lock`` (entire swap held under the lock —
             Audit #2 in §3.4 confirmed read-side routes don't contend).
          2. Validate experimental-functions gate (raises PermissionError → 403)
             and ``is_started()`` (raises ValueError → 422).
          3. Set ``_swap_in_progress`` flag (concurrent swap → 409 via the
             ``SwapInProgressError`` sentinel raised here).
          4. Snapshot pre-swap state for rollback (§3.7 guardrail #1) +
             capture the in-flight candidate-pool depth so the response can
             surface it (§3.5 — "Swap discarded N in-flight candidates").
          5/6. Signal stop on the training future; await its clean exit.
             Pre-P2-PRE-1 the training thread ignored ``_stop_event``;
             the fix at ``f4453fa`` wires the signal into the training-loop
             callbacks via ``_check_for_interrupt`` so this actually works.
          7. ``_reload_dataset`` fetches the new dataset (juniper-data I/O).
             P2-1b: a ``_check_swap_cancel`` checkpoint fires here — a
             DELETE arriving during the fetch trips ``SwapCancelledError``
             and the §3.8 rollback restores the pre-swap snapshot.
          8. P2-1c architecture adapter: equal-dim is a no-op; grow-only
             (input and/or output) expands the relevant weight tensors
             in place with zero-init new connections (§3.6 zero-init
             invariant); any shrink raises
             ``ValueError("shrink_unsupported (P2-1c): ...")`` → 422.
             P2-1d will lift the shrink restriction via prepended adapter
             layers.
          10. Reset ``_auto_snap_best_metric`` so the stale ratchet from
              the old dataset doesn't suppress new auto-snaps.
          11. Candidate pool is implicitly abandoned by the future stopping
              in step 6 — workers and in-flight candidates discarded. The
              count was captured pre-stop in step 4 (it is necessarily zero
              after the future stops, so reading it later is wrong).
          12. Submit a new training future on the new tensors. ``network.fit``
              naturally starts with output training, so the §3.5 "output
              training first" semantic is implicit (resolves §8 Q5).
          14. Force a topology rebroadcast (no-op equal-dim, but plumbs the
              WebSocket path for P2-1c/1d when dims actually change).
          15-16. Clear the in-progress + cancel flags in ``finally``; return
                 structured response per §3.3. Emit the §3.7 #5 structured
                 INFO log line.

        On ANY failure between step 4 and step 14: restore the pre-swap
        snapshot, resume training on the OLD dataset (best-effort), clear
        the flag, raise the original exception (the route translates it).
        A ``SwapCancelledError`` is a clean termination — same rollback,
        but the route maps it to 200 with ``{"status": "cancelled"}``.
        """
        # Step 2: validate gate (before acquiring lock — fast-path 403)
        if not self._experimental_functions_enabled:
            raise PermissionError("experimental_functions_disabled")

        with self._lock:
            # Step 2 cont: validate training is running.
            if not self.state_machine.is_started():
                raise ValueError("training_not_running — use POST /v1/training/dataset (cold swap) instead")

            # Step 3: idempotency guard.
            if self._swap_in_progress:
                raise SwapInProgressError("swap_already_in_progress")
            self._swap_in_progress = True
            # P2-1b: clear any stale cancel signal so we start each swap with
            # a fresh cancellation slate. A DELETE that arrived during the
            # window between two consecutive swaps must not pre-cancel the
            # next one — the flag is per-swap, not sticky.
            self._swap_cancel_requested.clear()

            try:
                # Step 4: snapshot pre-swap state. P2-1d expands the snapshot
                # to cover the network's parameter tensors (output_weights,
                # output_bias, each hidden unit's weights) because
                # ``_resize_network_for_dataset`` mutates them in place; the
                # CascadeCorrelationNetwork has no ``state_dict`` of its own,
                # so we clone the tensors directly. Hidden-unit biases are
                # scalars and untouched by P2-1d, so they aren't snapshotted.
                pre = _PreSwapSnapshot(
                    train_x=self._train_x,
                    train_y=self._train_y,
                    val_x=self._val_x,
                    val_y=self._val_y,
                    state_dict=copy.deepcopy(self.network.state_dict()) if hasattr(self.network, "state_dict") else None,
                    input_size=getattr(self.network, "input_size", None),
                    output_size=getattr(self.network, "output_size", None),
                    dataset_config=dict(self._current_dataset_config) if self._current_dataset_config else None,
                    active_output_dim=getattr(self.network, "active_output_dim", None),
                    output_weights=(self.network.output_weights.detach().clone() if hasattr(self.network, "output_weights") and self.network.output_weights is not None else None),
                    output_bias=(self.network.output_bias.detach().clone() if hasattr(self.network, "output_bias") and self.network.output_bias is not None else None),
                    hidden_unit_weights=([u["weights"].detach().clone() for u in self.network.hidden_units] if hasattr(self.network, "hidden_units") else None),
                )

                # P2-1b: capture the candidate-pool depth BEFORE we stop the
                # future. Reading after the stop always sees zero (the pool
                # has been drained by ``_stop_event``). Only the
                # CANDIDATE phase has an in-flight pool — output training
                # has none, so reporting zero in those cases is correct.
                abandoned_candidate_pool_size = self._snapshot_abandoned_candidate_pool_size()

                # P2-3 (Issue #3): pre-swap auto-snap. Fires AFTER the
                # in-memory rollback snapshot (step 4) but BEFORE any
                # network mutation, so the on-disk snapshot captures the
                # genuine pre-swap state. Kept on disk even when the swap
                # is later cancelled / rolled back — the pre-swap moment
                # is a valid checkpoint regardless of swap outcome, and
                # cleanup would just add race conditions.
                # Failure-tolerant: if the HDF5 write fails the swap
                # continues without an ID; canopy P2-7 will render the
                # event without a "Restore from pre-swap" affordance.
                pre_swap_snapshot_id: Optional[str] = None
                try:
                    pre_dataset_name = (pre.dataset_config or {}).get("dataset_type") or "unknown"
                    pre_snap = self.save_snapshot(description=f"pre-swap before dataset_swap (from {pre_dataset_name})")
                    if pre_snap is not None:
                        pre_swap_snapshot_id = pre_snap.get("id")
                except Exception:
                    self.logger.exception("swap_dataset_live: pre-swap snapshot failed; swap continues without pre_swap_snapshot_id")

                # Step 5/6: signal stop, wait for training future to exit cleanly.
                # P2-PRE-1 (f4453fa) makes _stop_event actually interrupt
                # the training-loop callbacks via TrainingInterrupted, caught
                # by _run_training as clean cancellation. Pre-fix this would
                # have blocked until natural fit completion (minutes).
                self._stop_event.set()
                self._pause_event.set()  # ensure pause wait loop wakes to observe stop
                future = self._training_future
                if future is not None:
                    try:
                        future.result(timeout=10)
                    except Exception as exc:
                        # Future may raise; we don't care about the value, only
                        # that the worker has finished. ``_run_training``'s
                        # clean-cancellation handler swallows TrainingInterrupted,
                        # but other exceptions still propagate out of ``future.result``.
                        # Log at debug since we're intentionally discarding them —
                        # the swap path continues regardless.
                        self.logger.debug("swap_dataset_live: prior training future raised on join: %s", exc)
                self._training_future = None

                # Step 7: fetch new dataset. Failure here triggers rollback.
                self._reload_dataset(**cfg)

                # P2-1b: post-fetch cancel checkpoint. Most user-initiated
                # cancels land here (the fetch is the long pole). The
                # SwapCancelledError trips the §3.8 rollback below.
                self._check_swap_cancel()

                # Step 8 (P2-1d): align network capacity with the new dataset.
                #   - GROW the network's input and/or output dim if the new
                #     dataset is larger than the network on either side.
                #   - PAD the dataset's tensors up to the network's dims if
                #     the new dataset is smaller. The training methods read
                #     ``self.network.active_output_dim`` to mask loss to the
                #     real output slots (see ``train_output_layer`` and
                #     ``calculate_residual_error`` in cascade_correlation.py).
                #
                # The network is monotonically non-decreasing: never shrinks.
                # See ``notes/PHASE_2_P2_1D_DESIGN_2026-05-13.md`` for the
                # full design — this is the design that replaces the §3.6
                # prepend-adapter approach abandoned 2026-05-12.
                dataset_input_dim = self._train_x.shape[1]
                dataset_output_dim = self._train_y.shape[1]
                target_input_dim = max(int(pre.input_size or 0), dataset_input_dim)
                target_output_dim = max(int(pre.output_size or 0), dataset_output_dim)
                resize_result = self.network._resize_network_for_dataset(
                    input_size_new=target_input_dim,
                    output_size_new=target_output_dim,
                )
                (
                    self._train_x,
                    self._train_y,
                    self._val_x,
                    self._val_y,
                    active_input_dim,
                    active_output_dim,
                ) = self._pad_dataset_for_network(self._train_x, self._train_y, self._val_x, self._val_y)
                # Set the loss-mask depth on the network. Same attribute that
                # ``_resize_network_for_dataset`` reset to ``output_size_new``
                # — overriding here when the dataset is smaller than the
                # (possibly just-grown) network.
                self.network.active_output_dim = active_output_dim

                # Step 10: reset auto-snap ratchet (§3.7 guardrail #6).
                with self._auto_snap_lock:
                    self._auto_snap_best_metric = None

                # P2-1b: second cancel checkpoint — last chance to abort
                # before the new future starts. After ``submit()`` the new
                # training run is live; cancelling then is the user's
                # responsibility (pause/stop on the new run).
                self._check_swap_cancel()

                # Step 12: submit new training future. _run_training performs
                # the FSM STOPPED → STARTED transition at the start of the new
                # invocation; we just need the underlying tensors to be the
                # new ones (which they are, post-_reload_dataset).
                self._stop_event.clear()
                self._pause_event.set()
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cascor-train")
                self._training_future = self._executor.submit(
                    self._run_training,
                    self._train_x,
                    self._train_y,
                    self._val_x,
                    self._val_y,
                )

                # Step 14: topology rebroadcast (no-op equal-dim, but the path
                # is plumbed for P2-1c/1d when dims actually change).
                self._broadcast_training_state(force=True)

                # P2-3 (Issue #3): post-swap auto-snap. Captures the
                # network state AFTER resize + new training future
                # submission so canopy P2-7's "Restore from post-swap"
                # affordance and the snapshot-orchestrated replay
                # transition (see ``notes/PHASE_2_P2_3_FOLLOWUP_REPLAY_REWORK_2026-05-14.md``)
                # have a concrete handle. Same failure tolerance as the
                # pre-swap snap.
                post_swap_snapshot_id: Optional[str] = None
                try:
                    post_dataset_name = (self._current_dataset_config or {}).get("dataset_type") or "unknown"
                    post_snap = self.save_snapshot(description=f"post-swap after dataset_swap (to {post_dataset_name})")
                    if post_snap is not None:
                        post_swap_snapshot_id = post_snap.get("id")
                except Exception:
                    self.logger.exception("swap_dataset_live: post-swap snapshot failed; swap continues without post_swap_snapshot_id")

                # Step 16: structured INFO log per §3.7 guardrail #5, then
                # structured response per §3.3. The §3.3 log/response use
                # the *dataset's* dims as the right-hand side of the arrow
                # so the user sees what the dataset is, not the network's
                # (possibly-larger, post-grow) capacity. The two are equal
                # when grow fires; they differ when shrink-via-padding
                # leaves the network larger than the dataset.
                hidden_preserved = int(resize_result.get("hidden_preserved", 0))
                self.logger.info(
                    "swap: input %d→%d, output %d→%d, hidden %d preserved, candidates %d abandoned, mode→output_training",
                    pre.input_size if pre.input_size is not None else dataset_input_dim,
                    dataset_input_dim,
                    pre.output_size if pre.output_size is not None else dataset_output_dim,
                    dataset_output_dim,
                    hidden_preserved,
                    abandoned_candidate_pool_size,
                )
                # §3.3 arch_changes shape. Deltas are the dataset-vs-pre-swap
                # diff so canopy can render a meaningful "what changed" toast
                # regardless of grow vs shrink. ``appended_nodes`` is the
                # network-side growth (zero when the dataset only shrunk).
                # ``prepended_layers`` is a frozen empty list — the old §3.6
                # adapter approach was abandoned; we never prepend layers.
                response_arch = {
                    "input_delta": dataset_input_dim - int(pre.input_size or 0),
                    "output_delta": dataset_output_dim - int(pre.output_size or 0),
                    "hidden_preserved": hidden_preserved,
                    "appended_nodes": {
                        "input": int(resize_result.get("input_delta", 0)),
                        "output": int(resize_result.get("output_delta", 0)),
                    },
                    "prepended_layers": [],
                    "abandoned_candidate_pool_size": abandoned_candidate_pool_size,
                    "active_output_dim": active_output_dim,
                }
                # P2-2 (Issue #3): persist the swap into network history so it
                # round-trips through snapshot save/load and is available to
                # canopy P2-7's timeline UI. Only fires on the success branch
                # — rollback / cancel raise before reaching this point.
                # Snapshot ID fields are placeholders that P2-3 will populate
                # P2-3 (Issue #3): the snapshot IDs that P2-2 left as
                # placeholders are now populated from the pre/post-swap
                # auto-snaps above. Either may be ``None`` if the
                # corresponding snapshot write failed — the event still
                # records, just without that affordance for canopy P2-7.
                # Failure to record (e.g., missing helper on a non-cascade
                # network) is logged at warning but does NOT abort the
                # swap — the swap itself already succeeded.
                recorded_event: Optional[Dict[str, Any]] = None
                if hasattr(self.network, "record_dataset_swap_event"):
                    try:
                        recorded_event = self.network.record_dataset_swap_event(
                            before_cfg=pre.dataset_config,
                            after_cfg=dict(self._current_dataset_config) if self._current_dataset_config else None,
                            arch_changes=response_arch,
                            pre_swap_snapshot_id=pre_swap_snapshot_id,
                            post_swap_snapshot_id=post_swap_snapshot_id,
                        )
                    except Exception:
                        self.logger.exception("swap_dataset_live: record_dataset_swap_event failed; swap itself was successful")

                # P2-2 Follow-up A (Issue #3): push the event over the
                # WebSocket so canopy can render a timeline marker in
                # real time without polling the history fetch route. The
                # broadcast envelope uses the existing generic
                # ``create_event_message`` with an ``event="dataset_swap"``
                # discriminator — no new envelope type required (avoids a
                # cross-repo bump of juniper_cascor_protocol). No-op when
                # no WS clients are connected. Failure-tolerant: a
                # broadcast error never aborts the swap.
                if recorded_event is not None and self._ws_manager is not None:
                    try:
                        from api.websocket.messages import create_event_message as _create_event_message

                        self._ws_manager.broadcast_from_thread(_create_event_message({"event": "dataset_swap", "swap": recorded_event}))
                    except Exception:
                        self.logger.exception("swap_dataset_live: dataset_swap WebSocket broadcast failed; swap itself was successful")
                return {
                    "status": "swapped",
                    "before_cfg": pre.dataset_config,
                    "after_cfg": dict(self._current_dataset_config) if self._current_dataset_config else None,
                    "arch_changes": response_arch,
                    "mode": "output_training_first",
                }

            except SwapCancelledError:
                # Clean cancellation — rollback and re-raise. Route maps to
                # HTTP 200 with cancelled status; this is NOT an error path
                # from the user's perspective, just an abandoned-by-request
                # terminal state with pre-swap data restored.
                self._rollback_pre_swap_state(pre)
                raise
            except ValueError:
                # Validation-class failure (dim_change_unsupported etc.) — rollback
                # and re-raise so the route can return 422. Note: the original
                # _train_x/_y/_val_x/_val_y may already have been overwritten by
                # _reload_dataset before the dim check fired; rollback restores them.
                self._rollback_pre_swap_state(pre)
                raise
            except Exception:
                # Catch-all — anything from juniper-data fetch to executor
                # failure. Same rollback + re-raise (route translates to 5xx).
                self._rollback_pre_swap_state(pre)
                raise
            finally:
                # Step 15: clear in-progress flag + cancel signal unconditionally
                # (§3.7 #3). Clearing the cancel flag here means a DELETE
                # arriving exactly as we finish does not survive to bias the
                # NEXT swap — see ``request_swap_cancel`` for the 404 path
                # that fires when the DELETE arrives post-clear.
                self._swap_in_progress = False
                self._swap_cancel_requested.clear()

    def _snapshot_abandoned_candidate_pool_size(self) -> int:
        """Return the in-flight candidate-pool depth at swap time.

        Only the CANDIDATE phase has live candidates; reading the network's
        ``candidate_pool_size`` at any other phase returns the configured
        pool capacity, not the count of work actually being abandoned.
        Output / inference / idle / paused → 0. Resolves the §3.5 "Swap
        discarded N in-flight candidates" UX requirement without overstating
        the abandonment cost in non-candidate phases.
        """
        try:
            phase = self.state_machine.phase
        except Exception:
            return 0
        if phase != TrainingPhase.CANDIDATE:
            return 0
        return int(getattr(self.network, "candidate_pool_size", 0) or 0)

    def _pad_dataset_for_network(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        val_x: Optional[torch.Tensor],
        val_y: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], int, int]:
        """Zero-pad dataset tensors up to the network's current input/output dims.

        Returns ``(x, y, val_x, val_y, active_input_dim, active_output_dim)``
        where ``active_*_dim`` is the dataset's pre-pad dim. The caller is
        expected to update ``self.network.active_output_dim`` (the training
        methods read it to mask the loss to real output slots — see
        ``train_output_layer`` and ``calculate_residual_error``); this helper
        does not mutate network state directly so it's safe to call from
        contexts that don't yet own the live training network (e.g., a
        snapshot-load-validation path).

        Raises ``ValueError`` if the dataset exceeds network capacity — the
        caller must invoke ``network._resize_network_for_dataset`` first.
        This is an assertion against the contract, not a real-world error
        path; ``swap_dataset_live`` always resizes before padding.

        ``None`` validation tensors stay ``None`` (we don't fabricate a zero
        validation set if the caller didn't supply one). Tensors that already
        match the network dim pass through unchanged.
        """
        net_input = int(self.network.input_size)
        net_output = int(self.network.output_size)
        active_input_dim = int(x.shape[1])
        active_output_dim = int(y.shape[1])

        if active_input_dim > net_input or active_output_dim > net_output:
            raise ValueError(f"_pad_dataset_for_network: dataset ({active_input_dim}, {active_output_dim}) exceeds network capacity ({net_input}, {net_output}); resize the network first")

        if active_input_dim < net_input:
            input_pad = net_input - active_input_dim
            x = torch.cat([x, torch.zeros(x.shape[0], input_pad, dtype=x.dtype, device=x.device)], dim=1)
            if val_x is not None:
                val_x = torch.cat([val_x, torch.zeros(val_x.shape[0], input_pad, dtype=val_x.dtype, device=val_x.device)], dim=1)

        if active_output_dim < net_output:
            output_pad = net_output - active_output_dim
            y = torch.cat([y, torch.zeros(y.shape[0], output_pad, dtype=y.dtype, device=y.device)], dim=1)
            if val_y is not None:
                val_y = torch.cat([val_y, torch.zeros(val_y.shape[0], output_pad, dtype=val_y.dtype, device=val_y.device)], dim=1)

        return x, y, val_x, val_y, active_input_dim, active_output_dim

    def _rollback_pre_swap_state(self, pre: "_PreSwapSnapshot") -> None:  # noqa: C901
        """Restore the network + tensor refs from a pre-swap snapshot.

        Restores dataset tensors, the loss-mask active dim, and (P2-1d) the
        network's parameter tensors (``output_weights``, ``output_bias``,
        each hidden unit's weights) plus the ``input_size`` / ``output_size``
        bookkeeping. The cascade-correlation network has no ``state_dict``
        (it doesn't inherit from ``nn.Module``), so we restore each tensor
        directly from the snapshot's clones.

        Best-effort: each restoration is wrapped so a partial failure (e.g.,
        hidden-unit count mismatch from a logic bug) doesn't leave the
        rollback half-done — the warning lands but training can still
        attempt to resume.
        """
        self.logger.warning(
            "swap_dataset_live: rolling back to pre-swap state (input=%r output=%r cfg=%r)",
            pre.input_size,
            pre.output_size,
            pre.dataset_config,
        )
        self._train_x = pre.train_x
        self._train_y = pre.train_y
        self._val_x = pre.val_x
        self._val_y = pre.val_y
        # P2-1d: restore the network's loss-mask depth so a half-completed
        # shrink doesn't leave the next training run masking against a stale
        # active dim. ``None`` snapshot means "wasn't set pre-swap, leave alone".
        if pre.active_output_dim is not None and hasattr(self.network, "active_output_dim"):
            self.network.active_output_dim = pre.active_output_dim
        # Restore network parameter tensors. The CascadeCorrelationNetwork
        # is plain-Python (no nn.Module), so we re-bind the attributes
        # directly. Each snapshot tensor is a detached clone — re-enable
        # requires_grad on output_weights/bias so the next train_output_layer
        # builds a fresh optimizer on a leaf tensor (cascade convention).
        if pre.output_weights is not None and hasattr(self.network, "output_weights"):
            try:
                w = pre.output_weights.clone()
                w.requires_grad_(True)
                self.network.output_weights = w
            except Exception:
                self.logger.exception("swap_dataset_live rollback: restoring output_weights failed; network may be inconsistent")
        if pre.output_bias is not None and hasattr(self.network, "output_bias"):
            try:
                b = pre.output_bias.clone()
                b.requires_grad_(True)
                self.network.output_bias = b
            except Exception:
                self.logger.exception("swap_dataset_live rollback: restoring output_bias failed; network may be inconsistent")
        if pre.hidden_unit_weights is not None and hasattr(self.network, "hidden_units"):
            try:
                for unit, restored_w in zip(self.network.hidden_units, pre.hidden_unit_weights):
                    # Stay detached — cascade-correlation hidden weights are
                    # frozen post-promotion (line ~3568 cascade_correlation.py).
                    unit["weights"] = restored_w.clone()
            except Exception:
                self.logger.exception("swap_dataset_live rollback: restoring hidden_unit weights failed; network may be inconsistent")
        # Re-bind the size attributes after the tensors are restored so any
        # readers (e.g., ``forward`` validation at line 1561) see a
        # self-consistent view.
        if pre.input_size is not None and hasattr(self.network, "input_size"):
            self.network.input_size = pre.input_size
        if pre.output_size is not None and hasattr(self.network, "output_size"):
            self.network.output_size = pre.output_size
        # Legacy: state_dict path remains a no-op on the cascade-correlation
        # class (no load_state_dict), but kept for any future nn.Module-style
        # network the lifecycle might host.
        if pre.state_dict is not None and hasattr(self.network, "load_state_dict"):
            try:
                self.network.load_state_dict(pre.state_dict)
            except Exception:
                self.logger.exception("swap_dataset_live rollback: load_state_dict failed; weights may be inconsistent")
        self._current_dataset_config = dict(pre.dataset_config) if pre.dataset_config else None

    # Canopy-facing staged ``dataset_type`` values (``StageDatasetRequest``'s
    # Literal) → juniper-data ``GENERATOR_REGISTRY`` keys. Types not listed pass
    # through unchanged (their registry key already matches).
    _STAGED_GENERATOR_ALIASES: Dict[str, str] = {"spirals": "spiral", "moons": "moon"}

    @staticmethod
    def _translate_staged_config(dataset_type: str, cfg: Dict[str, Any]) -> "tuple[str, Dict[str, Any]]":
        """Translate a canopy-facing staged dataset config into juniper-data's schema.

        ``StageDatasetRequest`` speaks canopy's dialect — ``dataset_type``
        ``"spirals"``/``"moons"``, a *total* ``n_samples``, ``rotations`` — while
        juniper-data's registry keys are ``"spiral"``/``"moon"`` and its spiral/xor
        generators take per-arm / per-quadrant counts (``n_points_per_spiral``,
        ``n_points_per_quadrant``) and ``n_rotations``. Nothing translated between
        the two dialects before this helper, so every canopy-staged reload died
        with "Unknown generator 'spirals'" at juniper-data (training-start
        diagnosis 2026-07-09 — the unit stubs exercised this path with
        juniper-data names directly, masking the seam).

        Returns ``(generator, params)`` ready for ``create_dataset``. Translated
        keys are written with ``setdefault`` so caller-supplied generic ``params``
        (which won their merge upstream) keep winning on conflict.
        """
        generator = TrainingLifecycleManager._STAGED_GENERATOR_ALIASES.get(dataset_type, dataset_type)
        params = dict(cfg)
        if generator == "spiral":
            n_spirals = int(params.get("n_spirals") or 2)  # juniper-data SPIRAL_DEFAULT_N_SPIRALS
            n_samples = params.pop("n_samples", None)
            if n_samples is not None:
                params.setdefault("n_points_per_spiral", max(1, int(n_samples) // max(1, n_spirals)))
            rotations = params.pop("rotations", None)
            if rotations is not None:
                params.setdefault("n_rotations", rotations)
            params.pop("n_squares", None)
        elif generator == "xor":
            n_samples = params.pop("n_samples", None)
            if n_samples is not None:
                params.setdefault("n_points_per_quadrant", max(1, int(n_samples) // 4))
            params.pop("rotations", None)
            params.pop("n_spirals", None)
            params.pop("n_squares", None)
        elif generator == "gaussian":
            # W-3: juniper-data's GaussianParams has NO ``n_samples`` — it takes a
            # per-class count. A staged total would previously pass through and be
            # silently ignored by the (extra-tolerant) params model, generating with
            # defaults — the silent-wrong-params class. Divide by the requested
            # class count (juniper-data GAUSSIAN_DEFAULT_N_CLASSES=2) like spiral's
            # per-arm division.
            n_classes = int(params.get("n_classes") or 2)
            n_samples = params.pop("n_samples", None)
            if n_samples is not None:
                params.setdefault("n_samples_per_class", max(1, int(n_samples) // max(1, n_classes)))
            params.pop("rotations", None)
            params.pop("n_spirals", None)
            params.pop("n_squares", None)
        elif generator == "checkerboard":
            # W-3: checkerboard takes ``n_samples`` AND ``n_squares`` directly; drop
            # only the spiral-only fields.
            params.pop("rotations", None)
            params.pop("n_spirals", None)
        else:
            # circles / moon / mnist / equities take ``n_samples`` directly; drop
            # every generator-specific typed field they do not declare.
            params.pop("rotations", None)
            params.pop("n_spirals", None)
            params.pop("n_squares", None)
        return generator, params

    @staticmethod
    def _artifact_to_tensors(arrays: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Convert a juniper-data NPZ artifact into validated float32 tensors.

        Returns ``(train_x, train_y, val_x, val_y)`` with ``val_*`` as ``None``
        when the artifact carries no validation split. Split out of
        ``_reload_dataset`` so the guard ladder (missing keys, malformed
        arrays, non-2-D shapes, sample-count mismatches, partial validation
        splits) stays inside the source complexity budget; every failure is a
        ``RuntimeError`` the reload caller treats as a retryable staged-config
        state.
        """
        try:
            new_train_x = torch.tensor(arrays["X_train"], dtype=torch.float32)
            new_train_y = torch.tensor(arrays["y_train"], dtype=torch.float32)
        except KeyError as exc:
            raise RuntimeError(f"juniper-data artifact missing required key: {exc}") from exc
        except (TypeError, ValueError, RuntimeError) as exc:
            # Non-array / non-numeric payloads blow up at tensor construction;
            # surface a stable RuntimeError so swap/start callers can leave
            # pending staging intact and retry after fixing the upstream artifact.
            raise RuntimeError(f"juniper-data artifact train arrays are malformed: {exc}") from exc

        if new_train_x.ndim != 2 or new_train_y.ndim != 2:
            raise RuntimeError(f"juniper-data artifact train arrays must be 2-D; got X_train.ndim={new_train_x.ndim}, y_train.ndim={new_train_y.ndim} " "-- 3-D sequence artifacts belong to the juniper-recurrence tier, not cascade-correlation (W-2 tier boundary; " "see the OQ-4 3-D ingestion-gate design in juniper-ml notes/)")
        if new_train_x.shape[0] != new_train_y.shape[0]:
            raise RuntimeError(f"juniper-data artifact train sample count mismatch: X_train={new_train_x.shape[0]} y_train={new_train_y.shape[0]}")

        has_x_test = "X_test" in arrays
        has_y_test = "y_test" in arrays
        if has_x_test != has_y_test:
            present = "X_test" if has_x_test else "y_test"
            missing = "y_test" if has_x_test else "X_test"
            raise RuntimeError(f"juniper-data artifact has partial validation split ({present} without {missing})")

        if not has_x_test:
            return new_train_x, new_train_y, None, None

        try:
            new_val_x = torch.tensor(arrays["X_test"], dtype=torch.float32)
            new_val_y = torch.tensor(arrays["y_test"], dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"juniper-data artifact validation arrays are malformed: {exc}") from exc
        if new_val_x.ndim != 2 or new_val_y.ndim != 2:
            raise RuntimeError(f"juniper-data artifact validation arrays must be 2-D; got X_test.ndim={new_val_x.ndim}, y_test.ndim={new_val_y.ndim} " "-- 3-D sequence artifacts belong to the juniper-recurrence tier, not cascade-correlation (W-2 tier boundary)")
        if new_val_x.shape[0] != new_val_y.shape[0]:
            raise RuntimeError(f"juniper-data artifact validation sample count mismatch: X_test={new_val_x.shape[0]} y_test={new_val_y.shape[0]}")
        return new_train_x, new_train_y, new_val_x, new_val_y

    def _reload_dataset(self, **cfg: Any) -> None:
        """Fetch a fresh dataset from juniper-data and replace the live tensors.

        Mirrors ``api/app.py::_auto_start_training``'s pattern: instantiate a
        ``JuniperDataClient`` from env vars, ``create_dataset`` with the
        staged generator + params, ``download_artifact_npz``, convert to
        ``torch.float32`` tensors, swap ``_train_x/_train_y`` (and val if
        the artifact carries them).

        The staged config is kept in canopy's dialect everywhere it is stored
        (``_pending_dataset_config`` / ``_current_dataset_config``); it is
        translated to juniper-data's generator key + params schema via
        ``_translate_staged_config`` at the fetch boundary only.

        Held under ``_lock`` by the caller (``start_training``);
        any I/O failure surfaces as ``RuntimeError`` so the caller can leave
        ``_pending_dataset_config`` in place for the user to retry.
        """
        try:
            from juniper_data_client import JuniperDataClient
        except ImportError as exc:  # juniper-data-client is an optional extra
            raise RuntimeError("juniper-data-client is not installed; cannot reload dataset") from exc

        # Pop the generator name; everything else is forwarded as ``params``.
        cfg = dict(cfg)
        dataset_type = cfg.pop("dataset_type", None)
        if not dataset_type:
            raise RuntimeError("Pending dataset config missing required 'dataset_type'")

        # Merge the generic ``params`` dict (StageDatasetRequest.params — for
        # generators whose inputs are not covered by the typed convenience
        # fields, e.g. ``equities``) with the remaining top-level typed fields
        # (n_samples/noise/… for the legacy generators). Generic params win on
        # key conflict. The legacy path is unchanged: spiral/xor/… bodies carry
        # no ``params`` key, so ``generic_params`` is empty and ``cfg`` is the
        # typed fields exactly as before.
        generic_params = cfg.pop("params", None) or {}
        cfg = {**cfg, **generic_params}

        # Resolve the juniper-data API key the same way inbound auth (settings.py)
        # and the auto-start path (app.py) do: ``api.secrets.get_secret`` honors the
        # ``JUNIPER_DATA_API_KEY_FILE`` Docker-secret indirection, falling back to the
        # plain ``JUNIPER_DATA_API_KEY`` env var. The previous ``from secrets_util
        # import get_secret`` referenced a module that never existed in this repo, so
        # the ``except ImportError`` branch silently substituted a ``None``-returning
        # lambda -> the JuniperDataClient sent no ``X-API-Key`` -> juniper-data 401 ->
        # cascor 502 on every live dataset swap. Both linters that would have caught it
        # were suppressed (``# type: ignore`` + ``# noqa``). See
        # notes/CASCOR_STARTUP_SECRET_INDIRECTION_INVESTIGATION_2026-06-14.md.
        from api.secrets import get_secret
        from api.settings import Settings
        from cascor_constants.constants_api import _PROJECT_API_JUNIPER_DATA_URL_DEFAULT

        # CFG-04: Settings field consolidates the JUNIPER_DATA_URL env-var
        # lookup; ``or DEFAULT`` preserves the legacy localhost:8100
        # fallback when neither the canonical nor the prefixed env var
        # is set. Fresh ``Settings()`` (not ``get_settings()``) so this
        # runtime path picks up env changes between pending-dataset
        # reloads, matching the pre-migration ``os.environ.get`` behavior.
        data_url = Settings().juniper_data_url or _PROJECT_API_JUNIPER_DATA_URL_DEFAULT
        api_key = get_secret("JUNIPER_DATA_API_KEY")
        client = JuniperDataClient(base_url=data_url, api_key=api_key)

        generator, jd_params = self._translate_staged_config(dataset_type, cfg)
        try:
            result = client.create_dataset(generator=generator, params=jd_params, persist=True)
            dataset_id = result["dataset_id"]
            arrays = client.download_artifact_npz(dataset_id)
        except Exception as exc:
            raise RuntimeError(f"juniper-data fetch failed: {exc}") from exc

        new_train_x, new_train_y, new_val_x, new_val_y = self._artifact_to_tensors(arrays)
        self._val_x = new_val_x
        self._val_y = new_val_y
        self._train_x = new_train_x
        self._train_y = new_train_y
        # ISSUE_3_PHASE_2_LIVE_DATASET_SWAP §3.2 step 4d — track the canonical
        # cfg so ``swap_dataset_live`` can report it as ``before_cfg`` and so
        # the rollback path can restore it if a swap fails.
        self._current_dataset_config = {"dataset_type": dataset_type, **dict(cfg)} if dataset_type else None
        self.logger.info("Reloaded dataset %r (%d train samples)", dataset_type, new_train_x.shape[0])

    def get_dataset(self) -> Dict[str, Any]:
        """Return dataset metadata."""
        if self._train_x is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "train_samples": self._train_x.shape[0],
            "test_samples": self._val_x.shape[0] if self._val_x is not None else 0,
            "input_features": self._train_x.shape[1],
            "output_features": self._train_y.shape[1],
        }

    def get_dataset_data(self) -> Optional[Dict[str, Any]]:
        """Return dataset arrays for visualization."""
        if self._train_x is None:
            return None
        result = {
            "train_x": self._train_x.detach().cpu().tolist(),
            "train_y": self._train_y.detach().cpu().tolist(),
        }
        if self._val_x is not None:
            result["val_x"] = self._val_x.detach().cpu().tolist()
            result["val_y"] = self._val_y.detach().cpu().tolist()
        return result

    def get_training_params(self) -> Dict[str, Any]:
        """Get current training parameters.

        Returns every field listed in ``update_params``' ``updatable_keys`` so that
        clients reconciling UI state after a reconnect observe the live network
        values rather than falling back to stale defaults.

        ``epochs_max`` is the exception (C2b / Q1): it is no longer a settable
        parameter — the echoed value is the per-run cap DERIVED from the granular
        limits (see :meth:`derive_epochs_cap`), so the value a client reads here can
        never contradict the limits that actually gate training. Before C2b this
        echoed the network's construction-time attribute (default 1e11), which
        exceeded the PATCH model's own ceiling (le=1e6) — canopy seeded its form
        from this echo and every full-form apply was wholesale-rejected 422
        (training-runtime-defects plan §4 I-4 root cause 1).

        Numpy scalars are coerced to Python natives via ``.item()`` so the
        result round-trips through pydantic-core's JSON serializer. After a
        snapshot restore the network's scalar attributes come back from
        HDF5 as ``numpy.int64`` / ``numpy.float64`` (h5py's default), which
        pydantic-core rejects with ``PydanticSerializationError``;
        coercing here keeps the wire format clean without forcing every
        snapshot consumer to do it.
        """
        if self.network is None:
            return {}
        return _coerce_native_scalars(
            {
                "learning_rate": getattr(self.network, "learning_rate", 0.0),
                "candidate_learning_rate": getattr(self.network, "candidate_learning_rate", 0.0),
                "max_hidden_units": getattr(self.network, "max_hidden_units", 0),
                # C2b / Q1 outcome (c): derived read-only cap, NOT the (dead)
                # construction-time network attribute. See derive_epochs_cap.
                "epochs_max": self.derive_epochs_cap(self.network),
                "max_iterations": getattr(self.network, "max_iterations", 0),
                "patience": getattr(self.network, "patience", 0),
                "candidate_pool_size": getattr(self.network, "candidate_pool_size", 0),
                "correlation_threshold": getattr(self.network, "correlation_threshold", 0.0),
                "convergence_threshold": getattr(self.network, "convergence_threshold", 0.001),
                "candidate_patience": getattr(self.network, "candidate_patience", _PROJECT_API_LIFECYCLE_DEFAULT_CANDIDATE_PATIENCE),
                "candidate_convergence_threshold": getattr(self.network, "candidate_convergence_threshold", 0.001),
                "candidate_epochs": getattr(self.network, "candidate_epochs", 0),
                # CAS-002 (Phase 6E Sprint A-1): per-output-training-phase budget
                # (one of the granular limits the C2b derived cap is computed from).
                "output_epochs": getattr(self.network, "output_epochs", 0),
                "init_output_weights": getattr(self.network, "init_output_weights", "zero"),
                # CAN-010 / ENH-006 (Phase 6E Sprint A-2): output-layer optimizer.
                # Reads through the nested ``config.optimizer_config`` so a runtime
                # patch via ``update_params`` is reflected here on the next GET.
                "optimizer_type": _read_optimizer_type(self.network),
                # CAN-011 (Phase 6E Sprint A-3): hidden-unit activation function.
                "activation_function_name": _read_activation_function_name(self.network),
                # CAS-006 (Phase 6E Sprint A-4): auto-snap-best lifecycle flags.
                # These live on the lifecycle (not the network) so a single
                # network instance can be re-used across runs while the auto-
                # snap counter resets each ``start_training``.
                "auto_snap_best": self._auto_snap_best,
                "auto_snap_min_epochs": self._auto_snap_min_epochs,
                # FRONTEND_ISSUES_PLAN_2026-05-09 §1.5 C2 / Issue #1 — candidate-pool
                # selection knobs. PR-4a stores them; PR-4b consumes them in
                # ``_select_best_candidates`` and ``grow_network``.
                "multi_candidate": getattr(self.network, "multi_candidate", False),
                "candidate_selection": getattr(self.network, "candidate_selection", "top"),
                "selected_candidates": getattr(self.network, "selected_candidates", 1),
                "top_candidates": getattr(self.network, "top_candidates", 1),
                "random_candidates": getattr(self.network, "random_candidates", 0),
            }
        )

    def update_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update runtime-modifiable training parameters (thread-safe).

        Modifies the live network's attributes directly. Parameters that are
        safe to update while training is running: learning_rate,
        candidate_learning_rate, correlation_threshold, candidate_pool_size.
        Parameters effective at next cascade/epoch: max_hidden_units, patience.

        ``epochs_max`` left the whitelist in C2b (Q1 outcome (c)): it is a derived
        read-only value now — a submitted ``epochs_max`` is accepted at the request
        boundary (so pre-N5 canopy full-form applies keep succeeding) and reported
        as ``skipped(not-updatable)`` by the C2a accounting instead of being applied.

        GAP-WS-28: applies all updates atomically. If any setattr raises,
        every previously-applied key is reverted to its pre-call value
        before re-raising, so the network is never left in a half-updated
        state. The ``_lock`` already prevents the race itself; this
        adds the all-or-nothing semantics for the case where a property
        setter rejects a value (currently no setters do, but adding a
        defensive guard now means future validation can be wired in
        without re-introducing torn writes).

        Args:
            params: Dict of parameter names and new values (None values excluded).

        Returns:
            Updated training parameters dict, plus additive per-key reporting
            (C2a / I-4): ``applied`` — the keys that landed — and ``skipped`` —
            ``{"key", "reason"}`` rows for every requested key that did not.
            See ``_apply_params_unlocked`` for the reason taxonomy.

        Raises:
            ValueError: If no network exists.
            RuntimeError: If a snapshot replay session is active (CAN-015c /
                REPLAYING rejects meta-param mutations).
            Exception: Re-raises whatever setattr raised, after rolling back
                any partially-applied updates.
        """
        with self._lock:
            # CAN-015c (B-3): REPLAYING rejects training commands AND
            # meta-param mutations. Enforce here so REST PATCH and WS
            # ``set_params`` share one fail-closed gate (FSM docstring
            # alone is not enough — neither surface consulted it).
            if self.state_machine.is_replaying():
                raise RuntimeError("Cannot update training parameters while replaying a snapshot — invoke /v1/snapshots/{id}/replay/control with action='stop' first")

            ########################################################################################
            # Do NOT remove this commented out code block until explicit approval has been granted
            ########################################################################################
            # if self.network is None:
            #     raise ValueError("No network exists — create a network first")
            # updatable_keys = {
            #     "learning_rate",
            #     "candidate_learning_rate",
            #     "correlation_threshold",
            #     "candidate_pool_size",
            #     "max_hidden_units",
            #     "epochs_max",
            #     "max_iterations",
            #     "patience",
            #     "convergence_threshold",
            #     "candidate_convergence_threshold",
            #     "candidate_patience",
            #     "candidate_epochs",
            #     "output_epochs",  # CAS-002 (Phase 6E Sprint A-1)
            #     "init_output_weights",
            #     "optimizer_type",  # CAN-010 / ENH-006 (Phase 6E Sprint A-2) — nested setter
            #     "activation_function_name",  # CAN-011 (Phase 6E Sprint A-3) — re-init on swap
            # }
            # # Plain setattr targets — keys that map directly to network attributes.
            # # ``optimizer_type`` and ``activation_function_name`` go through
            # # special-cased setters that touch nested config / re-init paths.
            # nested_keys = {"optimizer_type", "activation_function_name"}
            # simple_keys = updatable_keys - nested_keys
            # applicable = {k: v for k, v in params.items() if k in simple_keys and hasattr(self.network, k)}
            # old_values = {k: getattr(self.network, k) for k in applicable}

            # # CAN-010 / ENH-006: ``optimizer_type`` lives at
            # # ``self.network.config.optimizer_config.optimizer_type``, not on
            # # the network directly. Treated separately so the rollback path
            # # below still works through the same revert mechanism.
            # optimizer_pending = "optimizer_type" in params and params["optimizer_type"] is not None
            # old_optimizer_type = _read_optimizer_type(self.network) if optimizer_pending else None

            # # CAN-011 (A-3): ``activation_function_name`` requires re-running
            # # ``_init_activation_function`` so ``activation_fn`` picks up the
            # # new mapping. Same revert pattern as optimizer_type.
            # activation_pending = "activation_function_name" in params and params["activation_function_name"] is not None
            # old_activation_function_name = _read_activation_function_name(self.network) if activation_pending else None

            # applied: list[str] = []
            # try:
            #     for key, value in applicable.items():
            #         setattr(self.network, key, value)
            #         applied.append(key)
            #     if optimizer_pending:
            #         _write_optimizer_type(self.network, params["optimizer_type"])
            #         applied.append("optimizer_type")
            #     if activation_pending:
            #         _write_activation_function_name(self.network, params["activation_function_name"])
            #         applied.append("activation_function_name")
            # except Exception:
            #     # GAP-WS-28: revert any partial application before propagating.
            #     # CAN-010 / ENH-006 (A-2) + CAN-011 (A-3): nested setters
            #     # need their own revert path — mirror the apply branch.
            #     for key in reversed(applied):
            #         try:
            #             if key == "optimizer_type":
            #                 _write_optimizer_type(self.network, old_optimizer_type)
            #             elif key == "activation_function_name":
            #                 _write_activation_function_name(self.network, old_activation_function_name)
            #             else:
            #                 setattr(self.network, key, old_values[key])
            #         except Exception:
            #             # If revert itself raises, log and continue rolling
            #             # back the rest — best-effort consistency.
            #             self.logger.exception("update_params rollback: revert of %s failed", key)
            #     raise
            ########################################################################################

            return self._apply_params_unlocked(params)

    def _apply_params_unlocked(self, params: Dict[str, Any]) -> Dict[str, Any]:  # noqa: C901
        """Apply runtime params assuming the caller already holds ``_lock``.

        Internal helper extracted from ``update_params`` so that
        ``start_training`` can route TrainingParams body fields through the
        same whitelist + atomic-rollback path without re-entering the
        non-reentrant ``_lock`` (see CASCOR_FIT_KWARGS_LATENT_BUG.md
        for the full rationale of the split).

        Three storage flavors are supported:

        - **simple_keys** — plain attributes on ``self.network`` set via
          ``setattr``. The bulk of ``updatable_keys``.
        - **nested_keys** — fields that live on ``network.config`` or in
          a sub-config and need a special-cased setter (``optimizer_type``
          via ``_write_optimizer_type``; ``activation_function_name`` via
          ``_write_activation_function_name``, which also re-runs
          ``_init_activation_function`` so ``activation_fn`` actually
          refreshes from the registry).
        - **lifecycle_keys** — flags that live on the lifecycle (``self``)
          rather than the network (``auto_snap_*``).

        All three flavors share the same GAP-WS-28 atomic-rollback contract:
        if any setter raises, every previously-applied key is reverted to
        its pre-call value before re-raising.

        C2a (I-4 / T3): the success return is ``get_training_params()`` plus two
        additive reporting keys that account for EVERY requested key:
        ``applied`` (keys that landed, in application order) and ``skipped``
        (``{"key", "reason"}`` rows for keys that did not). Reasons:
        ``not-updatable`` (key outside ``updatable_keys``), ``no-such-attribute``
        (whitelisted key the live network object lacks — previously a silent
        drop), and ``null-value`` (None nested/lifecycle value from an internal
        caller; boundary callers strip None via ``exclude_none=True``). Bound
        violations never reach this path — pydantic request-model validation
        rejects the whole body 422 upstream (deliberately atomic; no partial
        apply on validation failure).
        """
        if self.network is None:
            raise ValueError("No network exists — create a network first")
        updatable_keys = {
            "learning_rate",
            "candidate_learning_rate",
            "correlation_threshold",
            "candidate_pool_size",
            "max_hidden_units",
            # "epochs_max" removed in C2b (Q1 outcome (c)): derived read-only —
            # submitted values are reported skipped(not-updatable), never applied.
            "max_iterations",
            "patience",
            "convergence_threshold",
            "candidate_convergence_threshold",
            "candidate_patience",
            "candidate_epochs",
            "output_epochs",  # CAS-002 (Phase 6E Sprint A-1)
            "init_output_weights",
            "optimizer_type",  # CAN-010 / ENH-006 (Phase 6E Sprint A-2) — nested setter
            "activation_function_name",  # CAN-011 (Phase 6E Sprint A-3) — re-init on swap
            "auto_snap_best",  # CAS-006 (Phase 6E Sprint A-4) — lifecycle attribute
            "auto_snap_min_epochs",  # CAS-006 (Phase 6E Sprint A-4) — lifecycle attribute
            # FRONTEND_ISSUES_PLAN_2026-05-09 §1.5 C2 — schema-only in PR-4a; selection
            # logic in PR-4b. Plain network attributes (no nested setter required).
            "multi_candidate",
            "candidate_selection",
            "selected_candidates",
            "top_candidates",
            "random_candidates",
        }
        nested_keys = {"optimizer_type", "activation_function_name"}
        lifecycle_keys = {"auto_snap_best", "auto_snap_min_epochs"}
        simple_keys = updatable_keys - nested_keys - lifecycle_keys

        # FRONTEND_ISSUES_PLAN_2026-05-09 §1.5 C2.1 — validate the candidate-pool
        # triple against the *post-merge* state so a multi-key PATCH that's only
        # valid as a unit (e.g. {S=6, T=4, R=2} from a prior {S=2, T=2, R=0})
        # is accepted in one shot. Raises ValueError → 422 if violated; nothing
        # has been applied yet, so no rollback is needed for this branch.
        triple_violation = self._validate_candidate_pool_post_merge(params)
        if triple_violation is not None:
            raise InvalidCandidatePoolError(triple_violation)
        applicable = {k: v for k, v in params.items() if k in simple_keys and hasattr(self.network, k)}
        old_values = {k: getattr(self.network, k) for k in applicable}

        # C2a (I-4 / T3): account for every requested key. ``skipped`` pairs each
        # non-landing key with a reason so the ``hasattr`` filter above can never
        # again silently drop a whitelisted-but-absent attribute (the latent
        # generator behind canopy's applied-yet-error verification divergence).
        # Computed against the same pre-apply view of the network as ``applicable``.
        skipped: list[dict[str, str]] = []
        for key in params:
            if key not in updatable_keys:
                skipped.append({"key": key, "reason": "not-updatable"})
            elif key in simple_keys and not hasattr(self.network, key):
                skipped.append({"key": key, "reason": "no-such-attribute"})
            elif key in nested_keys | lifecycle_keys and params[key] is None:
                # Boundary callers strip None via ``exclude_none=True``; internal
                # callers can pass raw dicts. A None nested/lifecycle value is
                # deliberately not applied (the ``*_pending`` guards below).
                skipped.append({"key": key, "reason": "null-value"})

        # CAN-010 / ENH-006 (A-2): ``optimizer_type`` lives at
        # ``self.network.config.optimizer_config.optimizer_type``.
        optimizer_pending = "optimizer_type" in params and params["optimizer_type"] is not None
        old_optimizer_type = _read_optimizer_type(self.network) if optimizer_pending else None

        # CAN-011 (A-3): ``activation_function_name`` requires re-running
        # ``_init_activation_function`` so ``activation_fn`` picks up the
        # new mapping.
        activation_pending = "activation_function_name" in params and params["activation_function_name"] is not None
        old_activation_function_name = _read_activation_function_name(self.network) if activation_pending else None

        # CAS-006 (A-4): ``auto_snap_*`` live on the lifecycle. Snapshot
        # the old values (plus the best-metric tracker) so the same
        # rollback semantics extend to lifecycle storage.
        auto_snap_pending = {k: params[k] for k in lifecycle_keys if k in params and params[k] is not None}
        old_lifecycle_values: Dict[str, Any] = {}
        old_auto_snap_best_metric: Optional[float] = None
        if auto_snap_pending:
            with self._auto_snap_lock:
                old_lifecycle_values = {k: getattr(self, f"_{k}") for k in auto_snap_pending}
                old_auto_snap_best_metric = self._auto_snap_best_metric

        applied: list[str] = []
        try:
            for key, value in applicable.items():
                setattr(self.network, key, value)
                applied.append(key)
            if optimizer_pending:
                _write_optimizer_type(self.network, params["optimizer_type"])
                applied.append("optimizer_type")
            if activation_pending:
                _write_activation_function_name(self.network, params["activation_function_name"])
                applied.append("activation_function_name")
            if auto_snap_pending:
                with self._auto_snap_lock:
                    for key, value in auto_snap_pending.items():
                        setattr(self, f"_{key}", value)
                        applied.append(key)
                    # Toggling auto_snap_best off-then-on within a run would
                    # otherwise inherit the prior ceiling. Reset the tracker
                    # whenever the toggle flips on so the next epoch is
                    # treated as a fresh baseline.
                    if "auto_snap_best" in auto_snap_pending and auto_snap_pending["auto_snap_best"] and not old_lifecycle_values.get("auto_snap_best", False):
                        self._auto_snap_best_metric = None
        except Exception:
            # GAP-WS-28: revert any partial application before propagating.
            # CAN-010 / ENH-006 (A-2) + CAN-011 (A-3) + CAS-006 (A-4): each
            # flavor has its own revert path — mirror the apply branch.
            for key in reversed(applied):
                try:
                    if key == "optimizer_type":
                        _write_optimizer_type(self.network, old_optimizer_type)
                    elif key == "activation_function_name":
                        _write_activation_function_name(self.network, old_activation_function_name)
                    elif key in lifecycle_keys:
                        with self._auto_snap_lock:
                            setattr(self, f"_{key}", old_lifecycle_values[key])
                            self._auto_snap_best_metric = old_auto_snap_best_metric
                    else:
                        setattr(self.network, key, old_values[key])
                except Exception:
                    # If revert itself raises, log and continue rolling
                    # back the rest — best-effort consistency.
                    self.logger.exception("update_params rollback: revert of %s failed", key)
            raise
        # C2b (I-4 root / I-1c): a successful apply may have changed values the
        # ``/v1/training/status`` projection reports (learning_rate,
        # max_hidden_units, max_iterations — and any granular limit moves the
        # derived ``max_epochs``). Re-project so the status surface stays
        # coherent with the network object after every PATCH, not only at
        # create time. Runs only on the success path (the rollback above
        # restored the pre-call values, so the projection is already correct).
        self._sync_training_state_from_network()
        # C2a: additive per-key reporting — the params-echo keys are unchanged;
        # ``applied``/``skipped`` ride alongside them in the same dict (the REST
        # route's ``data`` and the WS ack's ``result`` carry them through untouched).
        result = self.get_training_params()
        result["applied"] = applied
        result["skipped"] = skipped
        return result

    def _validate_candidate_pool_post_merge(self, params: Dict[str, Any]) -> Optional[str]:
        """Validate the §1.5 C2.1 candidate-pool invariant against the *post-merge*
        triple — i.e. what (S, T, R, P) would be if every key in ``params``
        landed.  Returns ``None`` on success or a violation string on failure.

        Skipped when none of {selected_candidates, top_candidates,
        random_candidates, candidate_pool_size} appears in ``params`` — the
        ambient triple was already valid (it's an invariant of the network),
        and validating an unrelated PATCH would be needlessly expensive.
        """
        triple_keys = {"selected_candidates", "top_candidates", "random_candidates", "candidate_pool_size"}
        if not (triple_keys & params.keys()):
            return None
        s = params.get("selected_candidates", getattr(self.network, "selected_candidates", 1))
        t = params.get("top_candidates", getattr(self.network, "top_candidates", 1))
        r = params.get("random_candidates", getattr(self.network, "random_candidates", 0))
        p = params.get("candidate_pool_size", getattr(self.network, "candidate_pool_size", 1))
        return _validate_candidate_pool_triple(int(s), int(t), int(r), int(p))

    # ------------------------------------------------------------------
    # Topology & statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _activation_name(activation_fn: Any) -> str:
        """Best-effort name extraction for a hidden unit's activation.

        Handles all three shapes that can land in ``unit["activation_fn"]``:

        - ``ActivationWithDerivative`` wrappers (the production path —
          cascade_correlation.py wraps the user's activation in this picklable
          class for multiprocessing). The wrapper stores its name in
          ``_activation_name``; it does **not** expose ``__name__``, so the
          previous ``activation_fn.__name__`` access raised AttributeError
          and silently failed inside ``get_topology``'s except clause —
          collapsing the topology view because the REST endpoint then
          returned 500 and the WS broadcast skipped the frame entirely.
        - Plain torch builtins (e.g. ``torch.sigmoid``, ``torch.tanh``) used
          by manual ``add_hidden_unit_manual`` callers — these expose
          ``__name__``.
        - ``torch.nn.Module`` instances — expose the class name.

        Returns ``"sigmoid"`` for ``None`` to preserve the prior fallback
        behavior (was ``unit.get("activation_fn", torch.sigmoid).__name__``).
        """
        if activation_fn is None:
            return "sigmoid"
        wrapper_name = getattr(activation_fn, "_activation_name", None)
        if isinstance(wrapper_name, str) and wrapper_name:
            return wrapper_name
        builtin_name = getattr(activation_fn, "__name__", None)
        if isinstance(builtin_name, str) and builtin_name:
            return builtin_name
        return type(activation_fn).__name__

    def get_topology(self) -> Optional[Dict[str, Any]]:
        """Extract network topology for visualization (thread-safe)."""
        if self.network is None:
            return None
        try:
            with self._topology_lock, torch.no_grad():
                topology = {
                    "input_size": self.network.input_size,
                    "output_size": self.network.output_size,
                    "hidden_units": [],
                    "output_weights": self.network.output_weights.detach().cpu().tolist(),
                    "output_bias": self.network.output_bias.detach().cpu().tolist(),
                }
                for i, unit in enumerate(self.network.hidden_units):
                    topology["hidden_units"].append(
                        {
                            "id": i,
                            "weights": unit["weights"].detach().cpu().tolist(),
                            "bias": float(unit["bias"]),
                            "activation": TrainingLifecycleManager._activation_name(unit.get("activation_fn")),
                        }
                    )
            return topology
        except Exception as e:
            self.logger.error(f"Failed to extract topology: {e}", exc_info=True)
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get network weight statistics."""
        if self.network is None:
            return {}
        try:
            with self._topology_lock, torch.no_grad():
                output_weights = self.network.output_weights.detach().cpu()
                stats = {
                    "total_hidden_units": len(self.network.hidden_units),
                    "output_weight_mean": float(output_weights.mean()),
                    "output_weight_std": float(output_weights.std()),
                    "output_weight_min": float(output_weights.min()),
                    "output_weight_max": float(output_weights.max()),
                }
            return stats
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}", exc_info=True)
            return {}

    # ------------------------------------------------------------------
    # CAN-015h-1: PATCH /v1/network/weights — surgical weight edit
    # ------------------------------------------------------------------

    # Sentinel values returned to the route layer so the route can map
    # them to the right HTTP status. Plain string codes keep the
    # lifecycle-↔-route boundary serializable for tests without
    # requiring HTTPException to escape into the lifecycle layer.
    _PATCH_OK: str = "ok"
    _PATCH_FSM_REJECTED: str = "fsm_rejected"
    _PATCH_NO_NETWORK: str = "no_network"
    _PATCH_BAD_TARGET: str = "bad_target"
    _PATCH_SHAPE_MISMATCH: str = "shape_mismatch"
    _PATCH_NAN_INF: str = "nan_inf"
    _PATCH_HIDDEN_UNIT_OUT_OF_RANGE: str = "hidden_unit_out_of_range"

    def patch_weights(
        self,
        target: str,
        field: str,
        values: Any,
        hidden_unit_index: Optional[int] = None,
        dtype: str = "float32",
    ) -> Dict[str, Any]:
        """CAN-015h-1: surgically rewrite a single parameter group.

        Returns a dict ``{"status": <code>, "detail": <str>}`` where
        ``status`` is one of the ``_PATCH_*`` sentinels. The route
        layer maps these to HTTP statuses (200 / 400 / 404 / 409 /
        422) — see ``routes/network.py``.

        FSM gate: requires INVESTIGATING (entered via ``/restore``).
        Any other state returns ``_PATCH_FSM_REJECTED``.

        Validation contract per the design plan:

        - ``values`` must match the target tensor's shape exactly.
          Partial updates are rejected with ``_PATCH_SHAPE_MISMATCH``.
          Rationale: forces canopy to be explicit; prevents subtle
          off-by-one bugs at the wire layer.
        - NaN / Inf values are rejected with ``_PATCH_NAN_INF``.
        - ``dtype`` is float32 by default. float64 inputs are auto-
          cast (lossless when fitting in float32 range; the
          plan-time concern about precision-losing casts is
          captured by the NaN check post-cast).

        Side-effects after a successful patch:

        - The touched parameter is reassigned with a fresh tensor
          carrying the new values, with ``requires_grad`` matching
          the pre-patch attribute (so optimizer wiring elsewhere
          doesn't toggle).
        - The output-layer optimizer's state for the touched
          parameter group is **zeroed** (Adam ``m`` and ``v``
          buffers reset). Stale momentum from pre-patch weights is
          meaningless after the rewrite.
        """
        if self.network is None:
            return {"status": self._PATCH_NO_NETWORK, "detail": "No network created"}

        if not self.state_machine.is_investigating():
            return {
                "status": self._PATCH_FSM_REJECTED,
                "detail": f"patch_weights requires INVESTIGATING state (currently {self.state_machine.status.name})",
            }

        if target not in ("output", "hidden_unit"):
            return {"status": self._PATCH_BAD_TARGET, "detail": f"unknown target: {target!r}"}
        if field not in ("weights", "bias"):
            return {"status": self._PATCH_BAD_TARGET, "detail": f"unknown field: {field!r}"}

        # Resolve the tensor we're rewriting + a setter. The setter is
        # a closure so we do not have to special-case the assignment
        # logic at each branch — and the optimizer-state zero-out
        # below uses the same closure to find the parameter group.
        try:
            new_tensor = self._build_patch_tensor(values, dtype)
        except (TypeError, ValueError) as e:
            return {"status": self._PATCH_NAN_INF, "detail": f"invalid tensor data: {e}"}

        # NaN/Inf check after dtype cast (catches values that would
        # otherwise become Inf when promoted to float32).
        if not torch.isfinite(new_tensor).all():
            return {"status": self._PATCH_NAN_INF, "detail": "values contain NaN or Inf"}

        if target == "hidden_unit":
            if hidden_unit_index is None or hidden_unit_index < 0 or hidden_unit_index >= len(self.network.hidden_units):
                return {
                    "status": self._PATCH_HIDDEN_UNIT_OUT_OF_RANGE,
                    "detail": f"hidden_unit_index={hidden_unit_index} out of range (have {len(self.network.hidden_units)} units)",
                }
            current = self.network.hidden_units[hidden_unit_index][field]
            if tuple(current.shape) != tuple(new_tensor.shape):
                return {
                    "status": self._PATCH_SHAPE_MISMATCH,
                    "detail": f"shape mismatch: hidden_unit[{hidden_unit_index}].{field} expects {tuple(current.shape)}, got {tuple(new_tensor.shape)}",
                }
            # Zero the optimizer state BEFORE reassignment because the
            # optimizer's state dict is keyed by tensor identity —
            # ``current`` is the live key, ``new_tensor`` won't be
            # found until a new optimizer is constructed against it.
            # No ``output_field``: ``output_optimizer`` optimizes only the output
            # ``nn.Linear``, so it holds no state for a hidden unit and this is
            # correctly a no-op. Kept as a call rather than dropped so a future
            # hidden-unit optimizer inherits the guard.
            self._zero_optimizer_state_for(current)
            self.network.hidden_units[hidden_unit_index][field] = new_tensor
        else:
            attr = "output_weights" if field == "weights" else "output_bias"
            current = getattr(self.network, attr)
            if tuple(current.shape) != tuple(new_tensor.shape):
                return {
                    "status": self._PATCH_SHAPE_MISMATCH,
                    "detail": f"shape mismatch: {attr} expects {tuple(current.shape)}, got {tuple(new_tensor.shape)}",
                }
            # Zero the optimizer state BEFORE reassignment (see above).
            self._zero_optimizer_state_for(current, output_field=field)
            requires_grad = bool(getattr(current, "requires_grad", False))
            if requires_grad:
                new_tensor.requires_grad_(True)
            setattr(self.network, attr, new_tensor)

        return {"status": self._PATCH_OK, "detail": "ok"}

    @staticmethod
    def _build_patch_tensor(values: Any, dtype: str) -> "torch.Tensor":
        """Coerce a JSON-decoded ``values`` payload into a float32 tensor.

        Raises ``ValueError`` on shape-irregular nested lists. The
        NaN/Inf check is performed by the caller after this returns
        so dtype-cast-induced infinities are also caught.
        """
        torch_dtype = torch.float64 if dtype == "float64" else torch.float32
        try:
            tensor = torch.tensor(values, dtype=torch_dtype)
        except (TypeError, RuntimeError, ValueError) as e:
            raise ValueError(f"could not coerce values to tensor: {e}") from e
        # Force float32 on the wire side so storage and downstream
        # forward-pass dtypes stay uniform with the existing network.
        return tensor.to(dtype=torch.float32)

    # ``train_output_layer`` builds the output layer as a single ``nn.Linear``,
    # whose ``parameters()`` yield ``weight`` then ``bias``, so the output
    # optimizer's flattened parameter list is always ``[weight, bias]`` in that
    # order. That positional correspondence is the ONLY link between a
    # network-level tensor (``output_weights`` / ``output_bias``) and the
    # ``Parameter`` the optimizer keys its state by — the two are never the same
    # object, which is why the identity lookup below can never match on its own.
    _OUTPUT_OPTIMIZER_SLOTS: Dict[str, int] = {"weights": 0, "bias": 1}

    def _resolve_optimizer_parameter(self, optimizer, parameter, output_field: Optional[str]):
        """Find the ``Parameter`` whose optimizer state corresponds to ``parameter``.

        Tries object identity first, so an optimizer that genuinely holds the
        network-level tensor keeps working, then falls back to the positional slot
        described on ``_OUTPUT_OPTIMIZER_SLOTS``. Returns None when no correspondence
        can be established.
        """
        state = optimizer.state
        # Identity, not ``in`` / ``.get`` — dict lookup would fall back to ``==``
        # on a hash collision, and tensor ``==`` returns a tensor whose truth value
        # raises rather than answering the question.
        for key in state:
            if key is parameter:
                return key
        if output_field is None:
            return None
        slot = self._OUTPUT_OPTIMIZER_SLOTS.get(output_field)
        if slot is None:
            return None
        params = [p for group in getattr(optimizer, "param_groups", ()) for p in group.get("params", ())]
        if slot >= len(params):
            return None
        candidate = params[slot]
        # ``nn.Linear.weight`` is the transpose of ``output_weights``, so shapes
        # differ by construction — but element counts must agree. A mismatch means
        # the layout assumption no longer holds; zero nothing rather than the wrong
        # buffer.
        if candidate.numel() != parameter.numel():
            self.logger.debug(f"_zero_optimizer_state_for: slot {slot} numel {candidate.numel()} != target numel {parameter.numel()}; skipping")
            return None
        return candidate

    def _zero_optimizer_state_for(self, parameter, *, output_field: Optional[str] = None) -> None:
        """Zero out the optimizer's momentum/variance buffers for a
        single parameter, if the network exposes a step-LR optimizer
        whose ``state`` covers that parameter.

        ``output_field`` names which output-layer slot ``parameter`` corresponds to
        (``"weights"`` / ``"bias"``); omit it for a tensor the output optimizer does
        not hold. Without it this was a permanent no-op — it was called with
        ``network.output_weights``, never with the ``nn.Linear`` ``Parameter`` the
        optimizer actually keys state by — which was harmless only for as long as
        ``train_output_layer`` threw the optimizer away on every call. Now that a
        resume carries the moments forward (R3), a weight edit that skipped this
        would be stepped with pre-edit Adam moments.

        Best-effort: if the optimizer is None, missing state, or keyed
        unrecognizably, the function is a no-op. The patched weights still take
        effect; only the optimizer's stale momentum survives, and the next training
        step will overwrite it within a few iterations regardless.
        """
        optimizer = getattr(self.network, "output_optimizer", None)
        if optimizer is None:
            return
        state = getattr(optimizer, "state", None)
        if not isinstance(state, dict):
            return
        target = self._resolve_optimizer_parameter(optimizer, parameter, output_field)
        if target is None:
            return
        param_state = state.get(target)
        if not isinstance(param_state, dict):
            return
        for key, val in list(param_state.items()):
            if not isinstance(val, torch.Tensor):
                continue
            # Skip 0-d tensors (Adam's ``step`` counter is stored as a
            # scalar tensor in newer PyTorch versions). Only the
            # running-statistic buffers (``exp_avg``, ``exp_avg_sq``,
            # etc.) carry the bias from pre-patch gradients and need
            # zeroing.
            if val.dim() == 0:
                continue
            param_state[key] = torch.zeros_like(val)

    # ------------------------------------------------------------------
    # CAN-015h-2: POST /v1/network/hidden-units — manual unit append
    # ------------------------------------------------------------------

    _ADD_OK: str = "ok"
    _ADD_FSM_REJECTED: str = "fsm_rejected"
    _ADD_NO_NETWORK: str = "no_network"
    _ADD_AT_CAP: str = "at_cap"
    _ADD_BAD_ACTIVATION: str = "bad_activation"
    _ADD_BAD_SHAPE: str = "bad_shape"
    _ADD_NAN_INF: str = "nan_inf"

    def add_hidden_unit_manual(
        self,
        weights: Any,
        bias: float = 0.0,
        activation: str = "Tanh",
    ) -> Dict[str, Any]:
        """CAN-015h-2: append a hidden unit at the cascade tail.

        Companion to ``add_unit`` (training-loop cascade-grow), but
        invoked manually from the canopy editor while the FSM is
        Investigating. Reuses the h-0 helpers
        (``_install_hidden_unit`` + ``_resize_output_layer_for_new_units``)
        so the cascade-grow code path and the manual-append code
        path can never drift apart on dict shape, history record
        layout, or output-layer rebuild semantics.

        Returns a sentinel-status dict like ``patch_weights``. Per
        the design plan: the new output-layer column is forced to
        **zero** (regardless of ``self.init_output_weights``)
        because the user hasn't trained against this unit yet —
        zero output column means zero contribution to predictions
        until ``PATCH /v1/network/weights`` rewrites it.
        """
        if self.network is None:
            return {"status": self._ADD_NO_NETWORK, "detail": "No network created"}
        if not self.state_machine.is_investigating():
            return {
                "status": self._ADD_FSM_REJECTED,
                "detail": f"add_hidden_unit_manual requires INVESTIGATING state (currently {self.state_machine.status.name})",
            }
        # max_hidden_units cap — same contract as the training-loop
        # cascade-grow path (which raises rather than returning a
        # sentinel; here we surface 409 to the route layer).
        if len(self.network.hidden_units) >= getattr(self.network, "max_hidden_units", 0):
            return {
                "status": self._ADD_AT_CAP,
                "detail": f"network is at max_hidden_units cap ({self.network.max_hidden_units})",
            }
        # Activation resolution against the network's registry. Pydantic
        # at the route boundary already restricts to the supported
        # Literal set, but we re-check here for direct lifecycle calls.
        registry = getattr(self.network, "activation_functions_dict", {}) or {}
        activation_fn = registry.get(activation)
        if activation_fn is None:
            return {
                "status": self._ADD_BAD_ACTIVATION,
                "detail": f"unknown activation: {activation!r} (registry keys: {sorted(registry.keys())})",
            }
        # Tensor coercion + NaN/Inf check (same path as patch_weights).
        try:
            weights_tensor = self._build_patch_tensor(weights, "float32")
        except (TypeError, ValueError) as e:
            return {"status": self._ADD_NAN_INF, "detail": f"invalid weights: {e}"}
        if not torch.isfinite(weights_tensor).all():
            return {"status": self._ADD_NAN_INF, "detail": "weights contain NaN or Inf"}
        if not (isinstance(bias, (int, float)) and torch.isfinite(torch.tensor(float(bias))).item()):
            return {"status": self._ADD_NAN_INF, "detail": "bias must be a finite scalar"}

        # Shape check: weight vector length must match the current
        # cascade-input width (= input_size + num_existing_hidden_units).
        prev_input_size = self.network.output_weights.shape[0]
        if weights_tensor.ndim != 1 or weights_tensor.shape[0] != prev_input_size:
            return {
                "status": self._ADD_BAD_SHAPE,
                "detail": f"weights shape {tuple(weights_tensor.shape)} does not match expected [{prev_input_size}] (input_size + num_existing_hidden_units)",
            }

        # Install via the h-0 helper. Correlation is undefined for
        # manual inserts — record 0.0 as a sentinel so downstream
        # consumers (history viewer, replay) can distinguish manual
        # from training-loop additions if needed.
        self.network._install_hidden_unit(
            weights=weights_tensor,
            bias=torch.tensor([float(bias)], dtype=torch.float32),
            activation_fn=activation_fn,
            correlation=0.0,
        )

        # Resize the output layer with a forced zero-init so the
        # appended unit's output column doesn't contribute to
        # predictions until trained or patched. Save and restore
        # ``init_output_weights`` so the network's persistent config
        # stays whatever the user originally set.
        original_init = self.network.init_output_weights
        try:
            self.network.init_output_weights = "zero"
            self.network._resize_output_layer_for_new_units(
                num_added=1,
                prev_input_size=prev_input_size,
            )
        finally:
            self.network.init_output_weights = original_init

        # Optimizer state for the now-stale output_weights tensor is
        # invalid (the parameter object was replaced by the resize
        # AND the dimension itself changed). Best-effort drop of the
        # optimizer; the next training pass will reconstruct it. We
        # drop rather than zero-out because Adam's state tensors no
        # longer match the new parameter shape.
        if hasattr(self.network, "output_optimizer"):
            self.network.output_optimizer = None

        new_index = len(self.network.hidden_units) - 1
        return {
            "status": self._ADD_OK,
            "detail": "ok",
            "unit_index": new_index,
            "num_hidden_units": len(self.network.hidden_units),
        }

    # ------------------------------------------------------------------
    # CAN-015h-3: DELETE /v1/network/hidden-units/{idx} — manual remove
    # ------------------------------------------------------------------

    _REMOVE_OK: str = "ok"
    _REMOVE_FSM_REJECTED: str = "fsm_rejected"
    _REMOVE_NO_NETWORK: str = "no_network"
    _REMOVE_OUT_OF_RANGE: str = "out_of_range"

    def remove_hidden_unit_manual(self, idx: int) -> Dict[str, Any]:
        """CAN-015h-3: remove the hidden unit at ``idx``.

        Cascade-rebuild semantics:

        - Unit ``idx`` is removed.
        - Each subsequent unit ``j > idx`` had a weight at position
          ``input_size + idx`` that referenced the removed unit's
          output. That single weight is dropped from each subsequent
          unit's weight vector. After the surgery, what was unit
          ``j`` is now unit ``j - 1`` with weight vector length
          ``input_size + (j - 1)`` — consistent with its new
          cascade position.
        - The corresponding column ``input_size + idx`` of
          ``output_weights`` is removed; ``output_weights`` reshapes
          to ``[in + num_units - 1, out]``.
        - The optimizer is dropped (Adam state tensors no longer
          match the new parameter shapes).

        Returns sentinel-status dict. ``idx`` out of range returns
        404 (not 409) per the design plan.
        """
        if self.network is None:
            return {"status": self._REMOVE_NO_NETWORK, "detail": "No network created"}
        if not self.state_machine.is_investigating():
            return {
                "status": self._REMOVE_FSM_REJECTED,
                "detail": f"remove_hidden_unit_manual requires INVESTIGATING state (currently {self.state_machine.status.name})",
            }
        n = len(self.network.hidden_units)
        if not isinstance(idx, int) or idx < 0 or idx >= n:
            return {
                "status": self._REMOVE_OUT_OF_RANGE,
                "detail": f"hidden_unit_index={idx} out of range (have {n} units)",
            }

        input_size = self.network.input_size
        col_to_drop = input_size + idx

        # 1. For each subsequent unit, drop its weight at index col_to_drop.
        #    Unit j > idx has weight vector length input_size + j; after
        #    surgery the new length is input_size + (j - 1) which matches
        #    its new cascade position post-shift.
        for j in range(idx + 1, n):
            unit = self.network.hidden_units[j]
            old_weights = unit["weights"].detach()
            keep_mask = torch.ones(old_weights.shape[0], dtype=torch.bool)
            keep_mask[col_to_drop] = False
            unit["weights"] = old_weights[keep_mask].clone().detach()

        # 2. Remove the unit from the hidden_units list.
        del self.network.hidden_units[idx]

        # 3. Surgically remove the corresponding column from output_weights.
        old_output = self.network.output_weights.detach()
        keep_rows = torch.ones(old_output.shape[0], dtype=torch.bool)
        keep_rows[col_to_drop] = False
        new_output = old_output[keep_rows].clone().detach()
        was_grad = bool(getattr(self.network.output_weights, "requires_grad", False))
        self.network.output_weights = new_output
        if was_grad:
            self.network.output_weights.requires_grad_(True)
        # output_bias is independent of unit count — leave untouched.

        # 4. Drop the optimizer (Adam state shapes invalid after column drop).
        if hasattr(self.network, "output_optimizer"):
            self.network.output_optimizer = None

        return {
            "status": self._REMOVE_OK,
            "detail": "ok",
            "removed_index": idx,
            "num_hidden_units": len(self.network.hidden_units),
        }

    # ------------------------------------------------------------------
    # Decision boundary
    # ------------------------------------------------------------------

    def get_decision_boundary(self, resolution: int = 50) -> Optional[Dict[str, Any]]:
        """Compute decision boundary grid for 2D visualization.

        Args:
            resolution: Number of grid points per axis.

        Returns:
            Dictionary with x_range, y_range, grid predictions, or None on failure.
        """
        if self.network is None or self._train_x is None:
            return None
        if self._train_x.shape[1] != 2:
            return None

        try:
            with self._topology_lock, torch.no_grad():
                x_data = self._train_x.cpu().numpy()
                x_min, x_max = float(x_data[:, 0].min()) - 0.5, float(x_data[:, 0].max()) + 0.5
                y_min, y_max = float(x_data[:, 1].min()) - 0.5, float(x_data[:, 1].max()) + 0.5

                xx = np.linspace(x_min, x_max, resolution)
                yy = np.linspace(y_min, y_max, resolution)
                grid_x, grid_y = np.meshgrid(xx, yy)
                grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

                grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
                predictions = self.network.forward(grid_tensor)
                pred_classes = predictions.argmax(dim=1).cpu().numpy()

            return {
                "x_range": [x_min, x_max],
                "y_range": [y_min, y_max],
                "resolution": resolution,
                "grid_x": grid_x.tolist(),
                "grid_y": grid_y.tolist(),
                "predictions": pred_classes.reshape(resolution, resolution).tolist(),
            }
        except Exception as e:
            self.logger.error(f"Failed to compute decision boundary: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def _get_snapshots_dir(self) -> Path:
        """Return the snapshots directory, creating it if needed.

        W-6 (CLI experimentation plan §11 / H-4): ``JUNIPER_CASCOR_SNAPSHOTS_DIR``
        overrides the default so a per-run launcher can point each cascor
        instance at its own ``RUN_DIR/snapshots`` (per-run config travels as
        process env, H-3). Read at call time — not import time — so tests and
        long-lived processes see the current env. A set-but-blank value is
        treated as unset (the blank-env guard class).

        The default is ``<repo>/cascor-snapshots`` — the REPO ROOT, and the ONE
        snapshot root shared by every stack origin (direct CLI, this service, and
        the container, which bind-mounts this same host directory). Snapshots are
        project assets, not per-origin scratch: a model saved by a container run is
        restored by a CLI run and resumed by a service run.

        Two earlier destinations are superseded and must not come back:

        * ``<repo>/src/snapshots`` — an importable Python package
          (``snapshot_serializer.py``, ``snapshot_cli.py``, ...). Writing ``.h5``
          artifacts into it coupled data to code: a cleanup glob there once deleted
          five snapshot MODULES and broke every boot (cascor#501), and
          ``.gitignore`` had to blanket-ignore the directory, so editing the
          tracked source required ``git add -f``.
        * ``<repo>/snapshots`` — the short-lived root this function used between
          cascor#537 and the storage-convention ruling. It never held a file.

        The **hyphen** in ``cascor-snapshots`` is load-bearing, not cosmetic: it is
        not a valid Python identifier, so setuptools can never discover the
        directory as a package and no cleanup can mistake it for code. That closes
        the cascor#501 class structurally rather than by convention.

        Design of record: juniper-ml
        ``notes/JUNIPER_2026-08-20_JUNIPER-ECOSYSTEM_SNAPSHOT-STORAGE-CONVENTION-DESIGN.md``.
        """
        override = os.environ.get("JUNIPER_CASCOR_SNAPSHOTS_DIR", "").strip()
        if override:
            snapshots_dir = Path(override).expanduser()
        else:
            # parents[3] == the repo root (this file is src/api/lifecycle/manager.py).
            snapshots_dir = Path(__file__).resolve().parents[3] / "cascor-snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        return snapshots_dir

    def save_snapshot(self, description: str = "") -> Optional[Dict[str, Any]]:
        """Save current network state to an HDF5 snapshot.

        P2-3 (Issue #3): the timestamp is second-resolution
        (``YYYYMMDDTHHMMSSZ``) which used to silently overwrite when two
        snapshots fired within the same second — exactly what happens on
        ``swap_dataset_live`` where pre- and post-swap snapshots are taken
        ≤1 s apart for a small network. We now check for an existing file
        and append ``_2``, ``_3``, etc. to the ID until we find a free
        slot. The no-collision path is byte-identical to the prior
        behaviour, so the ``snapshot_YYYYMMDDTHHMMSSZ`` format remains the
        common case for canopy / REST consumers that parse it.

        C1 (I-3): ``None`` is returned only for the no-network case. A
        failed write raises ``SnapshotSaveError`` carrying the serializer's
        reason — pre-C1 both collapsed to ``None`` and the API route mapped
        a disk/HDF5 failure to the same 404 as a missing network. The
        failure-tolerant callers (``auto_snap_best``, the swap pre/post
        snaps) already wrap this call in ``try/except Exception``.
        """
        if self.network is None:
            return None

        from snapshots.snapshot_errors import SnapshotSaveError
        from snapshots.snapshot_serializer import CascadeHDF5Serializer

        serializer = CascadeHDF5Serializer()
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        base_snapshot_id = f"snapshot_{timestamp}"
        snapshots_dir = self._get_snapshots_dir()
        snapshot_id = base_snapshot_id
        filepath = snapshots_dir / f"{snapshot_id}.h5"
        # P2-3 collision suffix. Capped at 1000 so a misbehaving filesystem
        # (e.g., directory not actually writable, every probe returning
        # exists()=True) cannot wedge the lifecycle in a tight loop. If a
        # thousand same-second snapshots really exist on disk, the caller
        # has bigger problems and a hard failure is the right signal.
        suffix = 2
        while filepath.exists() and suffix <= 1000:
            snapshot_id = f"{base_snapshot_id}_{suffix}"
            filepath = snapshots_dir / f"{snapshot_id}.h5"
            suffix += 1

        try:
            success = serializer.save_network(self.network, filepath, include_training_state=True)
        except SnapshotSaveError as exc:
            self.logger.error(f"Failed to save snapshot to {filepath}: {exc}")
            raise
        if not success:
            # Defensive: ``save_network``'s contract is raise-on-failure, but a
            # stubbed / legacy serializer returning falsy must not turn into a
            # phantom "no network" 404 downstream (C1 / I-3).
            self.logger.error(f"Failed to save snapshot to {filepath}")
            raise SnapshotSaveError(f"Snapshot save to {filepath} failed (serializer returned {success!r})")

        self.logger.info(f"Snapshot saved: {snapshot_id}")
        return {
            "id": snapshot_id,
            "path": str(filepath),
            "timestamp": timestamp,
            "description": description,
        }

    # CAN-015 (Phase 6E Sprint B): keys on ``network.history`` whose contents
    # represent per-epoch training metrics. ``restore_for_retrain`` empties
    # each one so a freshly-retrained run starts with a clean curve. Kept as
    # a class-level constant so the four B-sprint endpoints (Restore /
    # Replay / Resume / Retrain) share a single source of truth for what
    # "history" means; B-2 (Resume) and B-3 (Replay) will read from the
    # same set when locking it as read-only.
    _NETWORK_HISTORY_KEYS: tuple = ("train_loss", "value_loss", "train_accuracy", "value_accuracy")

    def _load_snapshot_to_network(self, snapshot_id: str) -> SnapshotLoadResult:
        """Locate the snapshot, deserialize, and install on the lifecycle.

        Internal helper extracted from ``load_snapshot`` so each Phase 6E
        Sprint B operation (Restore / Replay / Resume / Retrain) can share
        the load semantics while diverging on post-load state mutations
        (FSM transitions, history resets, replay-session setup, etc.).

        D-B: returns a :class:`SnapshotLoadResult` rather than a bare ``bool`` so the
        four verbs can tell the caller *why* a load failed. It is falsy on failure, so
        ``if not ok:`` call sites read unchanged. Absent is detected in two places —
        here at the glob, and again inside ``load_network_result`` — which is why the
        classification cannot live in the serializer alone.
        """
        snapshots_dir = self._get_snapshots_dir()
        matches = [f for f in snapshots_dir.glob("*.h5") if f.stem == snapshot_id]
        if not matches:
            self.logger.warning(f"Snapshot not found: {snapshot_id}")
            return snapshot_absent(f"no snapshot with id {snapshot_id!r}")

        from snapshots.snapshot_serializer import CascadeHDF5Serializer

        serializer = CascadeHDF5Serializer()
        result = serializer.load_network_result(matches[0])
        if not result:
            self.logger.error(f"Failed to load snapshot {snapshot_id}: {result.status} — {result.detail}")
            return result
        network = result.network

        # WS-6 B-phase: the HDF5 deserializer yields a *bare* CCN — re-wrap it so the
        # manager keeps holding a CascorModel (a getter-only property would leave
        # self.model stale here; see plan §4.3 H1). PR-B3.3: no monitoring hooks to
        # install/restore — live monitoring rides CascorModel.fit's on_event sink, wired
        # per-fit in _run_training, so a re-wrapped network needs no hook bookkeeping here.
        self.model = CascorModel(network=network)
        if self._worker_coordinator is not None and hasattr(self.network, "set_worker_coordinator"):
            self.network.set_worker_coordinator(self._worker_coordinator)
        # C2b (I-4 root / I-1c): the loaded network's tunables (restored by
        # ``_load_config_to_network``) may differ from whatever the projection
        # last showed — re-project so ``/v1/training/status`` agrees with
        # ``/v1/network`` and ``GET /v1/training/params`` after every
        # Restore / Replay / Resume / Retrain load.
        self._sync_training_state_from_network()
        return result

    def _snapshot_result(self, *, loaded: bool, snapshot_id: str, operation: str, reason: Optional[str] = None, reason_code: Optional[str] = None) -> Dict[str, Any]:
        """Status dict for the snapshot load/restore/resume verbs (WS-6 B2b return
        convergence toward the ServiceLifecycleManager seam).

        ``reason`` is human-readable; ``reason_code`` is the machine-readable
        discriminator the routes map to a status code (D-B) — one of the
        ``snapshot_load_status`` constants, or ``None`` for a non-load failure such as
        an FSM rejection. The routes previously consumed only ``loaded``, so every
        failure became the same 404 no matter what ``reason`` said."""
        return {
            "loaded": loaded,
            "snapshot_id": snapshot_id,
            "operation": operation,
            "fsm_state": self.state_machine.status.name,
            "reason": reason,
            "reason_code": reason_code,
        }

    def load_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """Load a network snapshot by ID (Restore semantics).

        Preserves the full snapshotted state — weights, topology, training
        history, all meta-parameters per A-5 (CAN-014). The FSM transitions
        to ``INVESTIGATING`` so the user can edit meta-params, replace the
        dataset, and re-snapshot, but cannot start training directly. To
        enter a training state, the user must invoke ``restore_for_retrain``
        (clean slate) or ``resume_from_snapshot`` (extend history).

        Rejected when training is currently active (Started / Paused) —
        same FSM-guard contract as Resume / Retrain. Returns False so
        the route layer can map to 409.

        See ``notes/PHASE_6E_SPRINT_B_DESIGN.md`` §2.1.
        """
        # Same pre-flight as resume_from_snapshot: investigating an
        # active training run would race with the running fit() and
        # leave the lifecycle in a confused state. CAN-015c (B-3) adds
        # the REPLAYING rejection so a Restore can't yank the network
        # out from under an active replay thread.
        if self.state_machine.is_started() or self.state_machine.is_paused() or self.state_machine.is_replaying():
            self.logger.warning(f"load_snapshot rejected: lifecycle is {self.state_machine.status.name}")
            return self._snapshot_result(loaded=False, snapshot_id=snapshot_id, operation="restore", reason=f"rejected: lifecycle is {self.state_machine.status.name}")

        outcome = self._load_snapshot_to_network(snapshot_id)
        if not outcome:
            return self._snapshot_result(loaded=False, snapshot_id=snapshot_id, operation="restore", reason=outcome.detail, reason_code=outcome.status)

        # CAN-015d (B-4): transition to Investigating and clear any
        # state from prior snapshot operations. The user explicitly
        # invoked /restore (not /retrain or /resume) so we want the
        # inspection-only contract: training commands rejected, no
        # implicit history reset, no resume marker.
        self.state_machine.mark_investigating()
        self._resume_point_epoch = None
        self.training_state.update_state(status="Stopped", phase="Idle")
        self._broadcast_training_state(force=True)
        self.logger.info(f"Snapshot restored: {snapshot_id} (FSM=Investigating)")
        return self._snapshot_result(loaded=True, snapshot_id=snapshot_id, operation="restore")

    def restore_for_retrain(self, snapshot_id: str) -> Dict[str, Any]:
        """Load a snapshot and reset training history for a fresh run (CAN-015a).

        Phase 6E Sprint B B-1. Loads the snapshot identically to
        ``load_snapshot`` (so weights, topology, and meta-params are
        preserved per A-5) then resets every history-bearing field so the
        next ``start_training`` call starts at epoch 0 with empty metric
        curves. The user benefits from the snapshot's prior training as a
        starting point but the new run is judged on its own merits.

        Reset scope per ``notes/PHASE_6E_SPRINT_B_DESIGN.md`` §9:

        - Network ``history`` arrays (train/value loss + accuracy) — cleared
        - ``training_state`` counters (current_epoch, current_step) — 0
        - ``_auto_snap_best_metric`` — None (so the new run gets fresh
          ratchet baseline; this also happens in ``start_training`` but
          we do it here too so a ``GET /v1/training/params`` between
          retrain and start_training already shows the cleared value)
        - FSM — Stopped / Idle (via ``Command.RESET``)
        - ``monitor.metrics_buffer`` — cleared
        - ``_last_emitted_history_len`` — 0

        CAN-015c (B-3): rejected when a replay session is active (would
        race with the replay thread reading from network.history).
        """
        if self.state_machine.is_started() or self.state_machine.is_paused() or self.state_machine.is_replaying():
            self.logger.warning(f"restore_for_retrain rejected: lifecycle is {self.state_machine.status.name}")
            return self._snapshot_result(loaded=False, snapshot_id=snapshot_id, operation="retrain", reason=f"rejected: lifecycle is {self.state_machine.status.name}")
        outcome = self._load_snapshot_to_network(snapshot_id)
        if not outcome:
            return self._snapshot_result(loaded=False, snapshot_id=snapshot_id, operation="retrain", reason=outcome.detail, reason_code=outcome.status)

        # Clear history arrays on the network. ``getattr`` rather than direct
        # attribute access so a network that doesn't expose ``history`` yet
        # (older snapshots, or a partially-initialized network from a corner
        # case) doesn't crash the retrain — best-effort consistency mirrors
        # the legacy-snapshot tolerance from A-5.
        history = getattr(self.network, "history", None)
        if isinstance(history, dict):
            for key in self._NETWORK_HISTORY_KEYS:
                if key in history:
                    # Preserve the container type (list vs deque vs other) by
                    # replacing with an empty instance of the same type. Falls
                    # back to ``[]`` if the container isn't a known builtin.
                    try:
                        history[key] = type(history[key])()
                    except Exception:
                        history[key] = []

        # Reset lifecycle-level training state. Mirrors ``reset()`` (line ~840)
        # but without the ``_stop_event.set()`` since no training is
        # currently running — Retrain is invoked from a stopped state and
        # ``start_training`` will clear the event itself.
        self._last_emitted_history_len = 0
        self.state_machine.handle_command(Command.RESET)
        self.monitor.clear_metrics()
        self.training_state.update_state(
            status="Stopped",
            phase="Idle",
            current_epoch=0,
            current_step=0,
        )
        with self._auto_snap_lock:
            self._auto_snap_best_metric = None
        # CAN-015b (B-2): a Retrain over a previously-loaded Resume
        # snapshot should not carry forward the resume marker — the
        # whole point of Retrain is the clean slate.
        self._resume_point_epoch = None
        self._broadcast_training_state(force=True)

        self.logger.info(f"Snapshot restored for retrain: {snapshot_id}")
        return self._snapshot_result(loaded=True, snapshot_id=snapshot_id, operation="retrain")

    def resume_from_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """Load a snapshot and prepare to continue training (CAN-015b).

        Phase 6E Sprint B B-2. Loads the snapshot identically to
        ``load_snapshot`` (so weights, topology, meta-params, AND the
        training history are preserved) then transitions the FSM to
        ``RESUME_READY`` and records the snapshot's terminal-epoch count
        as ``_resume_point_epoch`` so canopy can render a visual
        boundary between the pre-resume read-only history and the new
        training that extends past it.

        In contrast to ``restore_for_retrain`` (which clears history,
        counters, and the auto-snap-best ratchet so the new run starts
        fresh), Resume PRESERVES every history-bearing field. The next
        ``start_training`` extends the existing arrays rather than
        starting at epoch 0, and the auto-snap-best ratchet keeps its
        prior accuracy ceiling so a re-snapshot only fires when the new
        training genuinely beats the previous run.

        Resume requires a non-active state (Stopped / Completed /
        Failed / RESUME_READY again). From STARTED or PAUSED the
        underlying ``mark_resume_ready`` call rejects and this method
        returns False.

        See ``notes/PHASE_6E_SPRINT_B_DESIGN.md`` §2.3 for the full
        spec.
        """
        if self.state_machine.is_started() or self.state_machine.is_paused() or self.state_machine.is_replaying():
            self.logger.warning(f"resume_from_snapshot rejected: lifecycle is {self.state_machine.status.name}")
            return self._snapshot_result(loaded=False, snapshot_id=snapshot_id, operation="resume", reason=f"rejected: lifecycle is {self.state_machine.status.name}")

        outcome = self._load_snapshot_to_network(snapshot_id)
        if not outcome:
            return self._snapshot_result(loaded=False, snapshot_id=snapshot_id, operation="resume", reason=outcome.detail, reason_code=outcome.status)

        # Compute the resume-point epoch from the loaded network's
        # history. Use the longest array's length so a snapshot that's
        # missing some keys still produces a sensible marker. Falls back
        # to 0 if no history is present (a network freshly loaded with
        # no training-state included would land here — unusual but
        # tolerated).
        history = getattr(self.network, "history", None)
        resume_point = 0
        if isinstance(history, dict):
            for key in self._NETWORK_HISTORY_KEYS:
                series = history.get(key, ())
                try:
                    resume_point = max(resume_point, len(series))
                except TypeError:
                    # Unexpected type — skip, keep current best.
                    continue

        self._resume_point_epoch = resume_point
        self.state_machine.mark_resume_ready()
        # Surface the resume point in the broadcast so canopy clients
        # that subscribe to state updates pick it up immediately.
        # Mirrors the pattern used by reset() / restore_for_retrain.
        # NOTE: ``training_state.status`` stays "Stopped" rather than
        # "ResumeReady" — canopy reads RESUME_READY from the FSM summary
        # (state_machine.get_state_summary()), not from training_state.
        self.training_state.update_state(
            status="Stopped",
            phase="Idle",
            current_epoch=resume_point,
        )
        self._broadcast_training_state(force=True)

        self.logger.info(f"Snapshot restored for resume: {snapshot_id} (resume_point_epoch={resume_point})")
        return self._snapshot_result(loaded=True, snapshot_id=snapshot_id, operation="resume")

    def start_replay(self, snapshot_id: str) -> Dict[str, Any]:
        """Load a snapshot and start a replay session (CAN-015c).

        Phase 6E Sprint B B-3. Loads the snapshot identically to
        ``load_snapshot`` then transitions the FSM to ``REPLAYING`` and
        spawns a background ``_ReplaySession`` thread that emits
        synthetic ``epoch_end`` events from the loaded network's
        history arrays at a configurable speed.

        V1 scope: metric arrays + topology evolution metadata only.
        Per-epoch weight history (decision-boundary playback) is
        deferred to CAN-015g — would require a snapshot-format
        extension.

        Rejected when training is currently active (Started / Paused).
        Replacing one replay session with another is permitted — the
        old session's thread is stopped and the new session is
        installed.

        See ``notes/PHASE_6E_SPRINT_B_DESIGN.md`` §2.2.

        D-B: returns the same ``_snapshot_result`` dict as its three sibling verbs.
        It previously returned a bare ``bool``, which left it with no channel for a
        failure reason at all — so replay was the one verb that could not distinguish
        an absent snapshot from a corrupt one even in principle.
        """
        if self.state_machine.is_started() or self.state_machine.is_paused():
            self.logger.warning(f"start_replay rejected: training is {self.state_machine.status.name}")
            return self._snapshot_result(loaded=False, snapshot_id=snapshot_id, operation="replay", reason=f"rejected: lifecycle is {self.state_machine.status.name}")

        outcome = self._load_snapshot_to_network(snapshot_id)
        if not outcome:
            return self._snapshot_result(loaded=False, snapshot_id=snapshot_id, operation="replay", reason=outcome.detail, reason_code=outcome.status)

        # If a previous replay session was running, tear it down first
        # so its thread doesn't keep emitting against the new history.
        prev_session = self._replay_session
        if prev_session is not None:
            try:
                prev_session.stop()
            except Exception:
                self.logger.exception("start_replay: failed to stop previous replay session")

        history = getattr(self.network, "history", None)
        history_dict = history if isinstance(history, dict) else {}
        # CAN-015g (g-2): pull the per-sample weight history that the
        # g-1 serializer loaded onto the network. V1 snapshots have
        # no such attribute (or it's None) — the cache then advertises
        # weights_available=false to canopy via state_summary.
        weight_history = getattr(self.network, "weight_history", None)
        session = _ReplaySession(snapshot_id, history_dict, self.monitor, weight_history=weight_history)
        self._replay_session = session
        # Marker fields used by Resume / Restore are not relevant here.
        self._resume_point_epoch = None
        self.state_machine.mark_replaying()
        self.training_state.update_state(status="Stopped", phase="Idle")
        self._broadcast_training_state(force=True)
        # Start the driver thread AFTER the FSM transitions so the
        # initial frame emission lands while subscribers are looking
        # at a Replaying state.
        session.start_thread()

        self.logger.info(f"Snapshot replay started: {snapshot_id} (length={session.length})")
        return self._snapshot_result(loaded=True, snapshot_id=snapshot_id, operation="replay")

    def replay_control(self, action: str, **params: Any) -> Dict[str, Any]:
        """Apply a control action to the active replay session (CAN-015c).

        Supported actions: ``play`` / ``pause`` / ``seek`` (param
        ``time_index``) / ``speed`` (param ``value``) / ``range``
        (params ``start`` and ``end``) / ``stop``. ``stop`` exits
        Replaying — the FSM transitions back to ``STOPPED`` and the
        session thread is joined.

        Returns the post-action session state for the route response.
        Raises ``RuntimeError`` if no session is active.
        """
        session = self._replay_session
        if session is None or not self.state_machine.is_replaying():
            raise RuntimeError("No active replay session")

        action_lower = action.lower() if isinstance(action, str) else ""
        if action_lower == "play":
            session.play()
        elif action_lower == "pause":
            session.pause()
        elif action_lower == "seek":
            target = params.get("time_index")
            if target is None:
                raise ValueError("seek requires a 'time_index' parameter")
            session.seek(int(target))
        elif action_lower == "speed":
            value = params.get("value")
            if value is None:
                raise ValueError("speed requires a 'value' parameter")
            session.set_speed(float(value))
        elif action_lower == "range":
            start = params.get("start")
            end = params.get("end")
            if start is None or end is None:
                raise ValueError("range requires both 'start' and 'end' parameters")
            session.set_range(int(start), int(end))
            # state_summary() already reflects the new range (set_range mutated
            # self.range_start/self.range_end) and any re-clamped time_index.
            # Don't overlay set_range's return value — it includes a
            # ``time_index`` field that breaks the documented
            # ``range == {"start", "end"}`` contract asserted by
            # test_replay_control_seek_speed_range and
            # test_state_summary_includes_all_fields.
            return session.state_summary()
        elif action_lower == "stop":
            return self.stop_replay()
        else:
            raise ValueError(f"Unknown replay action: {action!r}")
        return session.state_summary()

    def stop_replay(self) -> Dict[str, Any]:
        """End the active replay session (CAN-015c).

        Joins the background thread, clears ``_replay_session``,
        transitions the FSM to STOPPED via ``Command.RESET``, and
        broadcasts the resulting state. Idempotent — calling on an
        inactive session returns a minimal "not_active" status.
        """
        session = self._replay_session
        if session is None:
            return {"status": "not_active"}
        try:
            session.stop()
        finally:
            self._replay_session = None
        # RESET is the universal "back to Stopped" transition. The FSM
        # already documents that REPLAYING accepts RESET as the escape
        # hatch alongside the explicit /control stop.
        self.state_machine.handle_command(Command.RESET)
        self.training_state.update_state(status="Stopped", phase="Idle")
        self._broadcast_training_state(force=True)
        self.logger.info(f"Snapshot replay stopped: {session.snapshot_id}")
        return {"status": "stopped", "snapshot_id": session.snapshot_id}

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List available snapshots."""
        snapshots_dir = self._get_snapshots_dir()
        snapshots = []
        for filepath in sorted(snapshots_dir.glob("*.h5")):
            snapshots.append(
                {
                    "id": filepath.stem,
                    "path": str(filepath),
                    "size_bytes": filepath.stat().st_size,
                    "modified": datetime.fromtimestamp(filepath.stat().st_mtime, tz=UTC).isoformat(),
                }
            )
        return snapshots

    def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific snapshot."""
        snapshots_dir = self._get_snapshots_dir()
        matches = [f for f in snapshots_dir.glob("*.h5") if f.stem == snapshot_id]
        if not matches:
            return None
        filepath = matches[0]
        return {
            "id": filepath.stem,
            "path": str(filepath),
            "size_bytes": filepath.stat().st_size,
            "modified": datetime.fromtimestamp(filepath.stat().st_mtime, tz=UTC).isoformat(),
        }

    def get_snapshot_dataset_swaps(self, snapshot_id: str) -> Optional[List[Dict[str, Any]]]:
        """Read the ``dataset_swap`` events from a stored snapshot's HDF5 file.

        P2-7 follow-up (Issue #3): canopy's Replay timeline uses this to
        render swap markers tied to the *loaded snapshot's* own history
        (parent spec §4.4), separate from the live event feed surfaced by
        ``get_dataset_swap_events`` (P2-2 follow-up B).

        Returns ``None`` when the snapshot file is not present (route maps
        to 404). Returns ``[]`` when the snapshot exists but carries no
        swap events — both a pre-P2-2 snapshot and a fresh training run
        with no live swaps reach this branch.
        """
        snapshots_dir = self._get_snapshots_dir()
        matches = [f for f in snapshots_dir.glob("*.h5") if f.stem == snapshot_id]
        if not matches:
            return None

        from snapshots.snapshot_serializer import CascadeHDF5Serializer

        serializer = CascadeHDF5Serializer()
        return serializer.read_dataset_swap_events(matches[0])

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Release the run's resources before the process goes away.

        The FastAPI lifespan's shutdown stanza calls this, and on a SIGTERM stop that stanza
        is the LAST Python code that runs: uvicorn's ``Server.capture_signals`` restores the
        default SIGTERM disposition and re-raises the captured signal the moment ``serve()``
        returns, so the kernel terminates the process a few hundred milliseconds after this
        method returns -- no ``atexit`` hooks, no interpreter finalisation, no thread joins
        (measured 2026-08-25 on uvicorn 0.46: dead 0.28 s after SIGTERM, wait status 143,
        ``atexit`` never fired; SIGINT is the only stop that unwinds normally). Every fleet
        stop tool -- ``juniper_chop_all.bash``, ``experiment_stack.bash``,
        ``isolated_stack.bash``, ``docker stop`` -- sends SIGTERM.

        Before this fix the method set ``_stop_event`` and returned at once
        (``Executor.shutdown(wait=False, cancel_futures=True)`` cancels only QUEUED futures and
        never waits for the running one), so a service stopped mid-training died with the
        training thread live, its forkserver candidate-worker pool orphaned, and the round's
        deferred-unlink ``SharedMemory`` block plus the pool queues' nine semaphores left in
        ``/dev/shm`` -- the ``juniper_train_*`` / ``sem.mp-*`` ledger characterised in
        juniper-ml ``notes/JUNIPER_2026-08-25_JUNIPER-CASCOR_DEV-SHM-LEAK-CHARACTERISATION.md``.
        The same kill-mid-write trigger produced cohort B (273 truncated snapshots) until
        cascor#561 made the snapshot write atomic.

        Order of operations:

        1. Set ``_stop_event`` (and ``_pause_event``, so a paused wait-loop wakes to observe
           the stop) and join the training future, bounded by
           ``_SHUTDOWN_TRAINING_JOIN_TIMEOUT_SECONDS``. The join lets the existing
           ``TrainingInterrupted`` path (``_handle_event`` -> ``_check_for_interrupt``) unwind
           ``fit`` on the training thread, whose ``finally`` releases the candidate pool and
           unlinks the shared-memory blocks (``_release_candidate_worker_pool`` ->
           ``_shutdown_worker_pool``). The interrupt lands within ~25 output epochs; a stop
           that arrives mid-candidate-round has to wait for the round, so the join can time
           out -- that is logged, never raised.
        2. Whether or not the join completed, release the network's candidate pool and
           shared memory explicitly (:meth:`_release_network_resources`; both hooks are
           idempotent). This replaces the ``atexit`` registrations that never fire under
           SIGTERM and is the belt-and-braces path for a timed-out join.
        3. Tear down the replay session and the executor as before. ``wait=False`` is kept on
           purpose: ``Executor.shutdown(wait=True)`` has no timeout, so it would block on a
           timed-out training thread until the fit ends naturally (minutes) and guarantee the
           stop tool's SIGKILL escalation instead of preventing it.

        Synchronous by design (tests call it directly); the lifespan runs it via
        ``asyncio.to_thread`` so a multi-second join never blocks the event loop.
        """
        started = time.monotonic()
        self._stop_event.set()
        self._pause_event.set()  # ensure a paused wait-loop wakes to observe the stop
        training_joined = True
        future = self._training_future
        if future is not None and not future.done():
            try:
                future.result(timeout=_SHUTDOWN_TRAINING_JOIN_TIMEOUT_SECONDS)
            except TimeoutError:
                training_joined = False
                self.logger.warning("shutdown: training thread did not unwind within %.1fs; releasing the candidate pool and shared memory from the shutdown thread", _SHUTDOWN_TRAINING_JOIN_TIMEOUT_SECONDS)
            except Exception as exc:
                # Only "has the thread finished" matters here; ``_run_training`` has already
                # recorded the terminal state, and a failed fit's exception is not shutdown's.
                self.logger.debug("shutdown: training future raised on join: %s", exc)
        self._training_future = None
        self._release_network_resources()
        self.stop_liveness_heartbeat()
        # CAN-015c (B-3): drain any active replay session so the
        # background driver thread doesn't outlive the lifecycle.
        if self._replay_session is not None:
            try:
                self._replay_session.stop()
            except Exception:
                self.logger.exception("shutdown: failed to stop replay session")
            self._replay_session = None
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        elapsed = time.monotonic() - started
        if training_joined:
            self.logger.info("TrainingLifecycleManager shut down (%.2fs)", elapsed)
        else:
            self.logger.warning("TrainingLifecycleManager shut down with the training thread still running (%.2fs); the process exit will abandon it", elapsed)

    def _release_network_resources(self) -> None:
        """Release the wrapped network's candidate-worker pool and SharedMemory blocks.

        Runs on the shutdown path after the bounded training join. After a clean unwind
        ``fit``'s ``finally`` has already emptied both and the calls are no-ops; after a
        timed-out join they do the release the abandoned training thread no longer can.
        Every failure is logged and swallowed -- nothing downstream of shutdown could act on
        an exception, and the process is about to exit either way.
        """
        network = self.network
        if network is None:
            return
        for hook_name in ("_release_candidate_worker_pool", "_cleanup_shared_memory"):
            hook = getattr(network, hook_name, None)
            if not callable(hook):
                continue
            try:
                hook()
            except Exception:
                self.logger.warning("shutdown: %s failed", hook_name, exc_info=True)
