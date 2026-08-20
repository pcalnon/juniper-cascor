"""WS-6 B-phase (PR-B1): manager <-> CascorModel wiring regression tests.

These guard the manager-side half of the B-phase native model-core adoption: the
``TrainingLifecycleManager`` now holds a model-core :class:`CascorModel` in
``self.model`` and exposes the wrapped ``CascadeCorrelationNetwork`` through a
back-compat ``network`` get/set property. Cover the four assignment sites the plan
calls out (§4.3 H1) — ``__init__`` / ``create_network`` / ``delete_network`` /
``_load_snapshot_to_network`` (the HDF5 re-wrap) — plus the property's get/set
behaviour and the write-through identity the ~40 cascor-specific reaches rely on.

The ``CascorModel`` contract itself is covered by ``test_cascor_model.py``; these
tests are strictly about the manager holding/exposing it correctly.
"""

from __future__ import annotations

import pytest

from api.lifecycle.manager import TrainingLifecycleManager
from api.models.cascor_model import CascorModel

pytestmark = pytest.mark.unit


def _make_ccn():
    """A small, fast, deterministic bare ``CascadeCorrelationNetwork``."""
    from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
    from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

    config = CascadeCorrelationConfig.create_simple_config(
        input_size=2,
        output_size=2,
        learning_rate=0.1,
        max_hidden_units=2,
        candidate_pool_size=2,
        candidate_epochs=2,
        output_epochs=2,
        epochs_max=3,
        max_iterations=2,
        random_seed=42,
    )
    return CascadeCorrelationNetwork(config=config)


# ----- assignment site 1: __init__ -------------------------------------------------
def test_fresh_manager_holds_no_model():
    mgr = TrainingLifecycleManager()
    assert mgr.model is None
    assert mgr.network is None  # property returns None when there is no model
    assert mgr.has_model() is False


def test_has_network_is_deprecated_alias_for_has_model():
    """WS-6 B2a: has_network() is kept as a thin back-compat alias for has_model()."""
    mgr = TrainingLifecycleManager()
    assert mgr.has_network() is mgr.has_model() is False
    mgr.network = _make_ccn()
    assert mgr.has_network() is mgr.has_model() is True


# ----- assignment site 2: create_network ------------------------------------------
def test_create_network_wraps_ccn_in_cascor_model():
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    assert isinstance(mgr.model, CascorModel)
    # The property exposes the *underlying* CCN (identity), not the wrapper.
    assert mgr.network is mgr.model.network
    assert type(mgr.network).__name__ == "CascadeCorrelationNetwork"
    assert mgr.has_model() is True


# ----- assignment site 3: delete_network ------------------------------------------
def test_delete_network_clears_the_model():
    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.delete_network()
    assert mgr.model is None
    assert mgr.network is None
    assert mgr.has_model() is False


# ----- assignment site 4: _load_snapshot_to_network (HDF5 re-wrap, H1) ------------
def test_hdf5_load_rewraps_bare_ccn(tmp_path, monkeypatch):
    """The HDF5 deserializer yields a *bare* CCN — the manager must re-wrap it so it
    keeps holding a CascorModel (a getter-only property would leave self.model stale)."""
    import snapshots.snapshot_serializer as snap_ser

    loaded = _make_ccn()
    (tmp_path / "snap.h5").write_bytes(b"")  # presence only; load_network is stubbed
    monkeypatch.setattr(snap_ser.CascadeHDF5Serializer, "load_network", lambda self, path, restore_multiprocessing=True: loaded, raising=True)

    mgr = TrainingLifecycleManager()
    monkeypatch.setattr(mgr, "_get_snapshots_dir", lambda: tmp_path, raising=True)
    assert bool(mgr._load_snapshot_to_network("snap")) is True
    assert isinstance(mgr.model, CascorModel)
    assert mgr.network is loaded  # the bare CCN got wrapped, exposed via the property


# ----- the network get/set property -----------------------------------------------
def test_network_setter_wraps_a_bare_ccn():
    mgr = TrainingLifecycleManager()
    ccn = _make_ccn()
    mgr.network = ccn  # back-compat: assigning a bare CCN wraps it
    assert isinstance(mgr.model, CascorModel)
    assert mgr.network is ccn


def test_network_setter_accepts_a_ready_cascor_model():
    mgr = TrainingLifecycleManager()
    ccn = _make_ccn()
    model = CascorModel(ccn)
    mgr.network = model  # already a CascorModel — stored as-is, not double-wrapped
    assert mgr.model is model
    assert mgr.network is ccn


def test_network_setter_none_clears_the_model():
    mgr = TrainingLifecycleManager()
    mgr.network = _make_ccn()
    assert mgr.model is not None
    mgr.network = None
    assert mgr.model is None
    assert mgr.network is None


# ----- write-through identity (the monkey-patch / surgery contract) ---------------
def test_network_property_is_stable_write_through_reference():
    """The ~40 cascor-specific reaches (monkey-patch reassignment, live-swap weight
    surgery) mutate ``self.network.<attr>`` in place; the property must return the
    *same* CCN object each time so those mutations persist."""
    mgr = TrainingLifecycleManager()
    mgr.network = _make_ccn()
    sentinel = object()
    mgr.network.ws6_write_through_probe = sentinel  # write through the property
    assert mgr.network.ws6_write_through_probe is sentinel  # read back
    assert mgr.model.network.ws6_write_through_probe is sentinel
    first_ref = mgr.network
    second_ref = mgr.network
    assert first_ref is second_ref  # stable identity across accesses
