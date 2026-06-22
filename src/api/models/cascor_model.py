"""Production cascor model wrapper: ``CascadeCorrelationNetwork`` -> model-core ``GrowableModel``.

WS-6 B-phase (native model-core adoption). This is the **production** promotion of the test-only
``src/tests/conformance/cascor_model_core_adapter.py``: it lets the lifecycle manager hold and operate
against the abstract ``juniper_model_core.GrowableModel`` interface instead of naming
``CascadeCorrelationNetwork`` directly.

Two ways it deliberately differs from the test adapter (load-bearing — see
``juniper-ml/notes/JUNIPER_CASCOR_WS6_BPHASE_MODEL_CORE_ADOPTION_BUILD_PLAN_2026-06-19.md`` §4.2):

* It **wraps a pre-built network** — the lifecycle manager constructs the ``CascadeCorrelationNetwork``
  in ``create_network`` and hands it here. This wrapper never constructs one.
* It **never re-seeds or re-constructs inside** :meth:`fit`. The manager owns construction, seeding (at
  construction), and the ``_apply_params_unlocked`` hyperparameter split. Re-seeding here would shift
  the RNG sequence the golden trajectory pins.

The wrapped network is exposed via the public :attr:`network` property so the manager's cascor-specific
operations (HDF5 snapshots, live dataset swap, manual weight/unit surgery, hyperparameter reads,
decision-boundary rendering) keep reaching it as ``self.model.network.<attr>``.

Decisions inherited from the conformance wiring plan: ``grow_step`` is a contract-legal no-op (D-C3 —
cascor grows inside ``fit``); :meth:`predict` returns raw class scores, never argmax (RK-6); numpy
crosses the interface boundary (D2).
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
from juniper_model_core.events import TrainingEvent
from juniper_model_core.interfaces import GrowableModel, GrowthOutcome, TaskType, TrainResult

if TYPE_CHECKING:
    from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork


class CascorModel(GrowableModel):
    """Presents a (pre-built) ``CascadeCorrelationNetwork`` as a model-core ``GrowableModel``."""

    task_type: TaskType = "classification"

    def __init__(self, network: CascadeCorrelationNetwork, *, random_seed: int | None = None) -> None:
        if network is None:
            raise ValueError("CascorModel requires a pre-built CascadeCorrelationNetwork instance")
        self._network = network
        self.random_seed = random_seed if random_seed is not None else getattr(network, "random_seed", None)
        self._frozen = False

    # ----- wrapped-network access (cascor-specific surface) --------------------------
    @property
    def network(self) -> CascadeCorrelationNetwork:
        """The wrapped ``CascadeCorrelationNetwork`` — cascor-specific reaches go through here."""
        return self._network

    # ----- TrainableModel ------------------------------------------------------------
    def fit(self, X, y, *, X_val=None, y_val=None, on_event=None, **kw) -> TrainResult:
        """Train the wrapped network in place, streaming ``TrainingEvent``s **live** during fit.

        WS-6 PR-B3.2: events are emitted as training progresses (not reconstructed post-hoc) by wiring
        CCN's *synchronous* native callback hooks to ``on_event`` for the duration of ``net.fit``:

        * ``training_start`` — before the wrapped ``net.fit``;
        * ``epoch_end`` — per output-training epoch (CCN ``on_epoch_callback`` / ``_output_epoch_callback``);
        * ``phase_change`` — per cascade grow-iteration (CCN ``on_grow_iteration_callback`` /
          ``_grow_iteration_callback``). The cascor-specific per-iteration candidate-pool detail is
          carried verbatim under ``payload["detail"]`` — plan §3.3 "extend the event payload": the lossy
          ``candidate_progress -> phase_change`` collapse is **not** used here, and the async 50 Hz
          per-candidate stream is preserved separately by the manager's retained drain side-channel (PR-B3.3);
        * ``unit_added`` — per installed hidden unit, reconstructed from ``history`` after ``net.fit``
          (CCN exposes no per-unit hook — the grow-iteration hook fires *before* the unit is installed);
        * ``training_end`` — after a clean ``net.fit``.

        The five types are model-core's closed vocabulary (``events.py``); ``seq`` is per-run monotonic.
        Does NOT construct or re-seed the network (plan §4.2). Native hooks are saved and restored in
        ``finally`` so a partial fit (e.g. an ``on_event`` sink that raises to interrupt, PR-B3.3) leaves
        no stale binding on the wrapped network.
        """
        x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        early_stopping = kw.pop("early_stopping", False)

        emit = self._make_event_sink(on_event) if on_event is not None else None
        restore_hooks = self._bind_live_event_hooks(emit) if emit is not None else None
        try:
            if emit is not None:
                emit("training_start", {"n_samples": int(x.shape[0])})
            if X_val is not None and y_val is not None:
                x_val = torch.as_tensor(np.asarray(X_val), dtype=torch.float32)
                y_val_t = torch.as_tensor(np.asarray(y_val), dtype=torch.float32)
                self._network.fit(x, y_t, x_val=x_val, y_val=y_val_t, early_stopping=early_stopping, **kw)
            else:
                self._network.fit(x, y_t, early_stopping=early_stopping, **kw)
            if emit is not None:
                self._emit_units_added(emit)
                emit("training_end", {"metrics": self.metrics()})
        finally:
            if restore_hooks is not None:
                restore_hooks()

        history = self._network.history
        per_epoch = [{"loss": float(loss), "accuracy": float(acc)} for loss, acc in zip(history.get("train_loss", []), history.get("train_accuracy", []))]
        return TrainResult(
            final_metrics=self.metrics(),
            n_epochs=max(1, len(history.get("train_loss", []))),
            history=per_epoch or None,
            stopped_reason=getattr(self._network, "_completion_reason", None),
        )

    @staticmethod
    def _make_event_sink(on_event: Callable[[TrainingEvent], None]) -> Callable[[str, dict[str, Any]], None]:
        """Return an ``emit(type, payload)`` that stamps a per-run monotonic ``seq`` and forwards a
        :class:`TrainingEvent` to ``on_event``. One counter per fit keeps ``seq`` non-decreasing across
        the live (during-fit) and post-hoc (unit_added / training_end) emissions."""
        counter = itertools.count()

        def emit(event_type: str, payload: dict[str, Any]) -> None:
            on_event(TrainingEvent(event_type, payload, next(counter)))

        return emit

    def _bind_live_event_hooks(self, emit: Callable[[str, dict[str, Any]], None]) -> Callable[[], None]:
        """Bind CCN's native callback hooks to ``emit`` for the duration of a fit; return a restore fn.

        ``on_epoch_callback`` -> ``epoch_end``; ``on_grow_iteration_callback`` -> ``phase_change`` (the
        full per-iteration candidate-pool detail preserved under ``payload["detail"]``, plan §3.3). The
        prior callback values are saved and restored so this is reentrancy-safe and leaves no binding
        behind once a fit completes (PR-B3.3 routes everything through ``fit(on_event=...)`` — the manager
        no longer binds these directly).
        """
        net = self._network
        prev_epoch_cb = getattr(net, "_output_epoch_callback", None)
        prev_grow_cb = getattr(net, "_grow_iteration_callback", None)

        def _on_epoch(epoch, epochs, loss) -> None:
            emit("epoch_end", {"epoch": int(epoch), "epochs": int(epochs), "metrics": {"loss": float(loss)}})

        def _on_grow_iteration(iteration, max_iterations, best_correlation, candidates_trained, candidates_total, phase_detail, **detail) -> None:
            emit(
                "phase_change",
                {
                    "phase": "candidate",
                    "detail": {
                        "grow_iteration": int(iteration),
                        "max_iterations": int(max_iterations),
                        "best_correlation": float(best_correlation),
                        "candidates_trained": int(candidates_trained),
                        "candidates_total": int(candidates_total),
                        "phase_detail": phase_detail,
                        "best_candidate_id": detail.get("best_candidate_id", -1),
                        "best_candidate_uuid": detail.get("best_candidate_uuid", ""),
                        "second_candidate_id": detail.get("second_candidate_id"),
                        "second_candidate_correlation": detail.get("second_candidate_correlation", 0.0),
                        "all_correlations": list(detail.get("all_correlations", [])),
                    },
                },
            )

        net._output_epoch_callback = _on_epoch
        net._grow_iteration_callback = _on_grow_iteration

        def restore() -> None:
            net._output_epoch_callback = prev_epoch_cb
            net._grow_iteration_callback = prev_grow_cb

        return restore

    def _emit_units_added(self, emit: Callable[[str, dict[str, Any]], None]) -> None:
        """Emit one ``unit_added`` per installed hidden unit, reconstructed from network history.

        CCN installs units *inside* ``grow_network`` and exposes no per-unit hook (the grow-iteration
        hook fires *before* the unit is installed), so unit_added is reconstructed from
        ``history["hidden_units_added"]`` after ``net.fit`` returns. The conformance kit checks event
        *order* (``training_start`` first, ``training_end`` last, ``seq`` non-decreasing), so emitting
        these between the last grow ``phase_change`` and ``training_end`` is legal.
        """
        for entry in self._network.history.get("hidden_units_added", []):
            unit_index = entry.get("unit_index", -1)
            if unit_index < 0:  # skip the fit() sentinel {corr:0.0, shape:(), idx:-1}
                continue
            emit(
                "unit_added",
                {"n_units": unit_index + 1, "unit_id": f"h{unit_index}", "score": float(entry.get("correlation", 0.0))},
            )

    def predict(self, X, **kw) -> np.ndarray:
        """Raw class scores — never argmax (RK-6); numpy at the boundary (D2)."""
        x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        with torch.no_grad():
            out = self._network.predict(x)
        return out.detach().cpu().numpy()

    def metrics(self) -> dict[str, float]:
        history = getattr(self._network, "history", {}) or {}
        accuracy = history.get("train_accuracy") or [0.0]
        loss = history.get("train_loss") or [0.0]
        return {"accuracy": float(accuracy[-1]), "loss": float(loss[-1])}

    def describe_topology(self) -> dict[str, Any]:
        n = self.n_units
        nodes: list[dict[str, Any]] = [{"id": "input", "kind": "input", "frozen": True}]
        for i in range(n):
            nodes.append({"id": f"h{i}", "kind": "hidden", "frozen": True})
        nodes.append({"id": "output", "kind": "output", "frozen": self._frozen})

        edges: list[dict[str, Any]] = [{"src": "input", "dst": "output", "recurrent": False}]
        for i in range(n):
            edges.append({"src": "input", "dst": f"h{i}", "recurrent": False})
            # Cascade architecture: each unit feeds all *later* units.
            for j in range(i + 1, n):
                edges.append({"src": f"h{i}", "dst": f"h{j}", "recurrent": False})
            edges.append({"src": f"h{i}", "dst": "output", "recurrent": False})

        return {
            "model_type": "cascade_correlation",
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "task_type": self.task_type,
                "n_units": n,
                "n_in": int(self._network.input_size),
                "n_out": int(self._network.output_size),
                "completion_reason": getattr(self._network, "_completion_reason", None),
            },
        }

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (int(self._network.input_size),)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (int(self._network.output_size),)

    # ----- GrowableModel -------------------------------------------------------------
    @property
    def n_units(self) -> int:
        return len(getattr(self._network, "hidden_units", []))

    def grow_step(self, **kw) -> GrowthOutcome:
        # D-C3: cascor grows inside fit(); no standalone single-step grow. No-op (contract-legal).
        return GrowthOutcome(added=False, n_units=self.n_units)

    def freeze(self) -> None:
        self._frozen = True
