"""Test-only adapter: CascadeCorrelationNetwork -> juniper-model-core GrowableModel.

OUT-13 (WS-6 trigger-gate, half 2). The model-core ``GrowableModel`` interface was designed
with cascor as a reference implementer, but ``CascadeCorrelationNetwork``'s current surface
does not implement that contract. This adapter bridges the gap so cascor can be run through
the ``juniper_model_core.conformance`` kit — proving (pre-refactor) that cascor *can* conform.
WS-6's interface-adoption phase later replaces this adapter with native conformance.

It is **test-only** (lives under ``src/tests/``, imports nothing into cascor production code).

Design decisions (ratified 2026-06-18, plan
``juniper-ml/notes/JUNIPER_CASCOR_MODEL_CORE_CONFORMANCE_WIRING_PLAN_2026-06-18.md``):

* **D-C3 — ``grow_step`` is a no-op.** cascor grows units *inside* ``fit()`` and exposes no
  standalone "add one frozen unit" call; the conformance suite tolerates ``added=False``
  (its growth test is guarded). cascor's real growth dynamics are pinned by the OUT-12
  trajectory golden — conformance asserts the *interface*, OUT-12 asserts the *behavior*.
* **on_event** — cascor's ``fit`` has no event sink, so events are reconstructed in legal
  order (``training_start`` -> ``unit_added``×k -> ``training_end``) post-hoc from
  ``network.history`` (the kit checks order, not timing).
* **metrics** — ``{"accuracy", "loss"}`` (both valid classification keys).
"""

from __future__ import annotations

from typing import Any

import golden_support as gs  # OUT-12 determinism harness + frozen two-spiral
import numpy as np
import torch
from juniper_model_core.conformance import ConformanceDataset
from juniper_model_core.events import TrainingEvent
from juniper_model_core.interfaces import GrowableModel, GrowthOutcome, TaskType, TrainResult

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork

# The canonical golden config (mirrors golden_support.GOLDEN_NET_CONFIG) so the conformance
# run exercises the same small, fast, deterministic network as the golden suite.
_NET_CONFIG = dict(gs.GOLDEN_NET_CONFIG)


class CascorModelCoreAdapter(GrowableModel):
    """Presents ``CascadeCorrelationNetwork`` as a model-core ``GrowableModel``."""

    task_type: TaskType = "classification"

    def __init__(self, *, random_seed: int = 42, **config: Any) -> None:
        self.random_seed = random_seed
        self._config = {**_NET_CONFIG, **config, "random_seed": random_seed}
        self._net: CascadeCorrelationNetwork | None = None
        self._frozen = False
        self._in_shape: tuple[int, ...] = ()
        self._out_shape: tuple[int, ...] = ()

    # ----- TrainableModel ------------------------------------------------------------
    def fit(self, X, y, *, X_val=None, y_val=None, on_event=None, **kw) -> TrainResult:
        gs.harden_determinism()
        x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        self._in_shape = (int(x.shape[1]),)
        self._out_shape = (int(y_t.shape[1]),)

        self._net = CascadeCorrelationNetwork(**self._config)
        # Reseed post-construction (matches the validated golden sequence).
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        if X_val is not None and y_val is not None:
            x_val = torch.as_tensor(np.asarray(X_val), dtype=torch.float32)
            y_val_t = torch.as_tensor(np.asarray(y_val), dtype=torch.float32)
            self._net.fit(x, y_t, x_val=x_val, y_val=y_val_t, early_stopping=False)
        else:
            self._net.fit(x, y_t, early_stopping=False)

        if on_event is not None:
            self._emit_events(on_event, int(x.shape[0]))

        history = self._net.history
        per_epoch = [{"loss": float(loss), "accuracy": float(acc)} for loss, acc in zip(history.get("train_loss", []), history.get("train_accuracy", []))]
        return TrainResult(
            final_metrics=self.metrics(),
            n_epochs=max(1, len(history.get("train_loss", []))),
            history=per_epoch or None,
            stopped_reason=self._net._completion_reason,
        )

    def _emit_events(self, on_event, n_samples: int) -> None:
        """Reconstruct a legal training-event sequence from the network history."""
        seq = 0
        on_event(TrainingEvent("training_start", {"n_samples": n_samples}, seq))
        seq += 1
        for entry in self._net.history.get("hidden_units_added", []):
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
        if self._net is None:
            raise RuntimeError("CascorModelCoreAdapter.predict called before fit")
        x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        with torch.no_grad():
            out = self._net.predict(x)
        # Raw class scores — never argmax (RK-6: label collapse is classification-only).
        return out.detach().cpu().numpy()

    def metrics(self) -> dict[str, float]:
        history = self._net.history if self._net is not None else {}
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
                "n_in": self._in_shape[0] if self._in_shape else 0,
                "n_out": self._out_shape[0] if self._out_shape else 0,
                "completion_reason": None if self._net is None else self._net._completion_reason,
            },
        }

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._in_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._out_shape

    # ----- GrowableModel -------------------------------------------------------------
    @property
    def n_units(self) -> int:
        return 0 if self._net is None else len(getattr(self._net, "hidden_units", []))

    def grow_step(self, **kw) -> GrowthOutcome:
        # D-C3: cascor grows inside fit(); no standalone step. No-op (contract-legal).
        return GrowthOutcome(added=False, n_units=self.n_units)

    def freeze(self) -> None:
        self._frozen = True


def two_spiral_classification_dataset() -> ConformanceDataset:
    """A classification ``ConformanceDataset`` from OUT-12's frozen two-spiral.

    The kit's built-in fixtures are regression-only (the RK-6 canary), so cascor supplies its
    own classification dataset. Stratified ~80/20 split (the spiral is class-ordered: first
    half class 0, second half class 1) so both train and val carry both classes.
    """
    x, y = gs.load_two_spiral()
    features = x.numpy()
    targets = y.numpy()
    labels = targets.argmax(axis=1)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        n_val = max(1, int(round(len(idx) * 0.2)))
        val_idx.extend(idx[-n_val:].tolist())
        train_idx.extend(idx[:-n_val].tolist())
    train_idx_arr = np.sort(np.asarray(train_idx))
    val_idx_arr = np.sort(np.asarray(val_idx))

    return ConformanceDataset(
        X=features[train_idx_arr],
        y=targets[train_idx_arr],
        X_val=features[val_idx_arr],
        y_val=targets[val_idx_arr],
        task_type="classification",
    )
