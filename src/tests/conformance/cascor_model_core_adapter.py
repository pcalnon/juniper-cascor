"""Conformance fixtures for the cascor ↔ juniper-model-core gate (WS-6).

OUT-13 introduced a **test-only** ``CascorModelCoreAdapter`` to prove cascor could satisfy the
``juniper_model_core.GrowableModel`` contract *before* the refactor. WS-6 PR-B3 promoted that
adapter into the **production** :class:`api.models.cascor_model.CascorModel`, and **PR-B4**
retires the test-only adapter — the conformance suite now runs against the production wrapper
("native conformance"). This module is reduced to the two fixtures the suite still needs:

* :func:`make_cascor_conformance_model` — builds the canonical golden-config
  ``CascadeCorrelationNetwork`` (seeded for determinism) and wraps it in the production
  ``CascorModel``. In production the lifecycle manager owns construction + seeding; the
  wrapper never re-seeds inside ``fit`` (plan §4.2), so this factory reproduces that
  build-then-seed step before wrapping.
* :func:`two_spiral_classification_dataset` — a classification ``ConformanceDataset`` from
  OUT-12's frozen two-spiral (the kit's built-in fixtures are regression-only).

The filename is retained (the production ``CascorModel`` docstring references this path as its
origin); it no longer defines an adapter class.
"""

from __future__ import annotations

import golden_support as gs  # OUT-12 determinism harness + frozen two-spiral
import numpy as np
import torch
from juniper_model_core.conformance import ConformanceDataset

from api.models.cascor_model import CascorModel
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork


def make_cascor_conformance_model() -> CascorModel:
    """The production ``CascorModel`` wrapping a freshly-built, seeded golden-config network.

    Uses the canonical golden config (``golden_support.GOLDEN_NET_CONFIG``, which pins
    ``random_seed=42``) so the conformance run exercises the same small, fast, deterministic
    network as the golden suite. Seeding happens here — not inside ``CascorModel.fit``, which
    never re-seeds (plan §4.2) — so the contract run is reproducible.
    """
    gs.harden_determinism()
    net = CascadeCorrelationNetwork(**gs.GOLDEN_NET_CONFIG)
    torch.manual_seed(gs.GOLDEN_NET_CONFIG["random_seed"])
    np.random.seed(gs.GOLDEN_NET_CONFIG["random_seed"])
    return CascorModel(network=net)


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
