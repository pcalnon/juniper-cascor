"""Unit tests for the production ``CascorModel`` GrowableModel wrapper (WS-6 B-phase, PR-B1).

Exercises the wrapper against a real (small, fast) ``CascadeCorrelationNetwork`` to prove it satisfies
the ``juniper_model_core.GrowableModel`` contract while wrapping a *pre-built* network without
re-seeding (plan §4.2). The full conformance kit runs against this wrapper in PR-B4; these are the
fast unit-lane guards.
"""

import numpy as np
import pytest
from juniper_model_core.interfaces import GrowableModel, GrowthOutcome, TrainableModel, TrainResult
from juniper_model_core.validation import validate_metrics, validate_topology

from api.models.cascor_model import CascorModel

pytestmark = pytest.mark.unit


def _make_network():
    """A small, fast, deterministic CascadeCorrelationNetwork."""
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


def _toy_dataset():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((24, 2)).astype(np.float32)
    labels = (X[:, 0] > 0.0).astype(int)
    y = np.eye(2, dtype=np.float32)[labels]
    return X[:16], y[:16], X[16:], y[16:]


def test_is_growable_model():
    model = CascorModel(_make_network())
    assert isinstance(model, GrowableModel)
    assert isinstance(model, TrainableModel)
    assert model.task_type == "classification"


def test_requires_a_network():
    with pytest.raises(ValueError):
        CascorModel(None)


def test_network_property_returns_the_wrapped_instance():
    net = _make_network()
    model = CascorModel(net)
    assert model.network is net  # identity — the manager's cascor-specific reaches go through this


def test_shapes_read_live_from_network_before_fit():
    model = CascorModel(_make_network())
    # Available immediately (no fit required) — unlike the test adapter which cached them in fit.
    assert model.input_shape == (2,)
    assert model.output_shape == (2,)


def test_fit_returns_train_result_and_trains_in_place():
    net = _make_network()
    model = CascorModel(net)
    X, y, X_val, y_val = _toy_dataset()
    result = model.fit(X, y, X_val=X_val, y_val=y_val)
    assert isinstance(result, TrainResult)
    assert result.n_epochs >= 1
    assert result.final_metrics  # non-empty
    # fit trained the SAME wrapped instance (no re-construct, no re-seed).
    assert model.network is net


def test_predict_returns_raw_scores_never_argmax():
    model = CascorModel(_make_network())
    X, y, X_val, y_val = _toy_dataset()
    model.fit(X, y, X_val=X_val, y_val=y_val)
    preds = model.predict(X_val)
    assert isinstance(preds, np.ndarray)
    # Per-sample class scores, not collapsed labels (RK-6): shape (n, output_size), not (n,).
    assert preds.shape == (X_val.shape[0], 2)
    assert preds.shape[1:] == model.output_shape


def test_metrics_keys_valid_for_classification():
    model = CascorModel(_make_network())
    X, y, X_val, y_val = _toy_dataset()
    model.fit(X, y, X_val=X_val, y_val=y_val)
    metrics = model.metrics()
    assert set(metrics) == {"accuracy", "loss"}
    validate_metrics("classification", metrics)  # raises on a contract violation


def test_describe_topology_is_renderable():
    model = CascorModel(_make_network())
    X, y, X_val, y_val = _toy_dataset()
    model.fit(X, y, X_val=X_val, y_val=y_val)
    topo = model.describe_topology()
    validate_topology(topo)  # raises on a contract violation
    assert topo["meta"]["task_type"] == "classification"
    assert topo["model_type"] == "cascade_correlation"


def test_event_stream_is_legal_order():
    model = CascorModel(_make_network())
    X, y, X_val, y_val = _toy_dataset()
    events = []
    model.fit(X, y, X_val=X_val, y_val=y_val, on_event=events.append)
    assert events[0].type == "training_start"
    assert events[-1].type == "training_end"
    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs)  # non-decreasing


def test_grow_step_is_noop_and_freeze_holds():
    model = CascorModel(_make_network())
    X, y, X_val, y_val = _toy_dataset()
    model.fit(X, y, X_val=X_val, y_val=y_val)
    before = model.n_units
    outcome = model.grow_step()
    assert isinstance(outcome, GrowthOutcome)
    assert outcome.added is False  # D-C3: cascor grows inside fit(), not via grow_step
    assert model.n_units == before
    model.freeze()
    assert model.grow_step().added is False


def test_n_units_reflects_grown_units():
    net = _make_network()
    model = CascorModel(net)
    assert model.n_units == 0  # fresh network, no hidden units yet
    X, y, X_val, y_val = _toy_dataset()
    model.fit(X, y, X_val=X_val, y_val=y_val)
    assert model.n_units == len(net.hidden_units)


# ----- PR-B3.2: live on_event streaming during fit --------------------------------------


def _collect_events(model, X, y, X_val, y_val):
    events = []
    model.fit(X, y, X_val=X_val, y_val=y_val, on_event=events.append)
    return events


def test_fit_streams_live_epoch_and_phase_events():
    """fit emits ``epoch_end`` (output epochs) and ``phase_change`` (grow iterations) DURING fit, with
    cascor's per-iteration candidate-pool detail preserved under ``payload['detail']`` (plan §3.3
    extend-the-payload), not collapsed away."""
    net = _make_network()
    model = CascorModel(net)
    X, y, X_val, y_val = _toy_dataset()
    events = _collect_events(model, X, y, X_val, y_val)

    by_type: dict[str, list] = {}
    for e in events:
        by_type.setdefault(e.type, []).append(e)

    # Only model-core's closed vocabulary is emitted.
    assert set(by_type) <= {"training_start", "epoch_end", "phase_change", "unit_added", "training_end"}

    # Live output-epoch events carry loss under the documented metrics dict.
    assert by_type.get("epoch_end"), "expected >=1 live epoch_end event during output training"
    assert "loss" in by_type["epoch_end"][0].payload["metrics"]

    # Live grow-iteration events map to phase_change with the FULL candidate-pool detail retained.
    assert by_type.get("phase_change"), "expected >=1 live phase_change event during cascade growth"
    detail = by_type["phase_change"][0].payload["detail"]
    for key in ("grow_iteration", "candidates_trained", "candidates_total", "best_correlation", "all_correlations"):
        assert key in detail, f"phase_change detail dropped {key!r}: {detail}"


def test_fit_unit_added_one_event_per_grown_unit():
    """One ``unit_added`` per installed hidden unit (fit sentinel skipped), each carrying the documented
    ``{n_units, unit_id, score}`` payload."""
    net = _make_network()
    model = CascorModel(net)
    X, y, X_val, y_val = _toy_dataset()
    events = _collect_events(model, X, y, X_val, y_val)

    unit_added = [e for e in events if e.type == "unit_added"]
    assert len(unit_added) == model.n_units == len(net.hidden_units)
    for e in unit_added:
        assert set(e.payload) == {"n_units", "unit_id", "score"}


def test_fit_event_stream_is_monotonic_with_start_first_end_last():
    """Even with the richer live stream, the conformance ordering invariant holds: training_start first,
    training_end last, seq non-decreasing across live + post-hoc emissions."""
    model = CascorModel(_make_network())
    X, y, X_val, y_val = _toy_dataset()
    events = _collect_events(model, X, y, X_val, y_val)
    assert events[0].type == "training_start"
    assert events[-1].type == "training_end"
    assert [e.seq for e in events] == sorted(e.seq for e in events)


def test_fit_binds_native_hooks_only_with_on_event_and_restores_them():
    """The native epoch/grow hooks are bound only for the duration of an ``on_event`` fit and restored
    afterwards; a fit without ``on_event`` never binds them (no stale callback left on the network)."""
    net = _make_network()
    model = CascorModel(net)
    X, y, X_val, y_val = _toy_dataset()

    # No on_event -> hooks never bound.
    model.fit(X, y, X_val=X_val, y_val=y_val)
    assert getattr(net, "_output_epoch_callback", None) is None
    assert getattr(net, "_grow_iteration_callback", None) is None

    # With on_event -> restored to the prior value (None here) after fit returns.
    model.fit(X, y, X_val=X_val, y_val=y_val, on_event=lambda e: None)
    assert getattr(net, "_output_epoch_callback", None) is None
    assert getattr(net, "_grow_iteration_callback", None) is None
