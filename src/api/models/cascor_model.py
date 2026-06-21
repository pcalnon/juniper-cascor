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
        """Train the wrapped network in place. Does NOT construct or re-seed (plan §4.2)."""
        x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        early_stopping = kw.pop("early_stopping", False)
        if X_val is not None and y_val is not None:
            x_val = torch.as_tensor(np.asarray(X_val), dtype=torch.float32)
            y_val_t = torch.as_tensor(np.asarray(y_val), dtype=torch.float32)
            self._network.fit(x, y_t, x_val=x_val, y_val=y_val_t, early_stopping=early_stopping, **kw)
        else:
            self._network.fit(x, y_t, early_stopping=early_stopping, **kw)

        if on_event is not None:
            self._emit_events(on_event, int(x.shape[0]))

        history = self._network.history
        per_epoch = [{"loss": float(loss), "accuracy": float(acc)} for loss, acc in zip(history.get("train_loss", []), history.get("train_accuracy", []))]
        return TrainResult(
            final_metrics=self.metrics(),
            n_epochs=max(1, len(history.get("train_loss", []))),
            history=per_epoch or None,
            stopped_reason=getattr(self._network, "_completion_reason", None),
        )

    def _emit_events(self, on_event: Callable[[TrainingEvent], None], n_samples: int) -> None:
        """Reconstruct a legal training-event sequence post-hoc from network history.

        The conformance kit checks event *order* (``training_start`` first, ``training_end`` last, ``seq``
        non-decreasing), not timing. NOTE (WS-6 plan H4): this coarse reconstruction discards
        per-candidate-iteration detail; the production on_event migration (PR-B3) must preserve the
        ``/ws/training`` live-progress granularity separately — this event stream is the conformance
        surface, not the live-progress surface.
        """
        seq = 0
        on_event(TrainingEvent("training_start", {"n_samples": n_samples}, seq))
        for entry in self._network.history.get("hidden_units_added", []):
            unit_index = entry.get("unit_index", -1)
            if unit_index < 0:  # skip the fit() sentinel {corr:0.0, shape:(), idx:-1}
                continue
            seq += 1
            on_event(
                TrainingEvent(
                    "unit_added",
                    {"n_units": unit_index + 1, "unit_id": f"h{unit_index}", "score": float(entry.get("correlation", 0.0))},
                    seq,
                )
            )
        seq += 1
        on_event(TrainingEvent("training_end", {"metrics": self.metrics()}, seq))

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
