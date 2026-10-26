#!/usr/bin/env python
"""Unit coverage for the final evaluation's ``accuracy`` (juniper-ml APD-ML-002).

``eval_metrics.final`` (cascor#582) scores the finished model once, on the reported
partition. Until this change it carried F1 / precision / recall / ROC-AUC and no
accuracy, so juniper-ml's experiment reader asked for a ``test_accuracy`` that cascor
never computed. This file pins:

* the value: hand-counted on known cases, binary and multi-class, and equal to the
  network's own ``calculate_accuracy`` on the same rows (the call ``fit()`` makes for
  the history's ``value_accuracy``), so the numbers stay comparable;
* the split: accuracy is scored on the rows that the block's ``split`` label names,
  ``test`` when the test pair is loaded and ``validation`` otherwise;
* degradation: an empty or non-finite split gives ``None`` with its reason in
  ``undefined``, as the sibling metrics do, while a single-class split is still scored;
* the surface: ``GET /v1/metrics`` serves it under ``eval_metrics.final``, and the live
  block keeps its exact shape.

All fast unit tests; no network is trained.
"""

import types

import pytest
import torch
from fastapi.testclient import TestClient

from api.app import create_app
from api.lifecycle.manager import TrainingLifecycleManager
from api.settings import Settings

pytestmark = pytest.mark.unit

_SIBLINGS = ("f1", "precision", "recall", "roc_auc")

#: The live ``eval_metrics`` keys before and after this change. Accuracy lives only in ``final``.
_LIVE_EVAL_METRICS_KEYS = {"enabled", "average", "split", "n_samples", "n_classes", "undefined", "final"}

_X_VAL = torch.tensor([[0.2, 0.8], [0.9, 0.1], [0.1, 0.7], [0.8, 0.2]])
_X_TEST = torch.tensor([[0.3, 0.7], [0.7, 0.3], [0.25, 0.75], [0.75, 0.25]])


@pytest.fixture
def mgr():
    m = TrainingLifecycleManager()
    try:
        yield m
    finally:
        m.shutdown()


def _identity_network(mgr):
    """A stub whose forward returns its input, so each split's accuracy is fixed by its own rows."""
    mgr.network = types.SimpleNamespace(forward=lambda t: t.clone())
    mgr._eval_metrics_enabled = True
    return mgr


def _split(n_correct: int, n_rows: int = 4):
    """Rows whose argmax (under the identity stub) matches the one-hot target on ``n_correct`` of ``n_rows``.

    Both classes occur in the target whenever ``0 < n_correct < n_rows``, so the
    sibling metrics stay defined.
    """
    x = torch.tensor([[0.9, 0.1]] * n_rows)
    y = torch.tensor([[1.0, 0.0]] * n_correct + [[0.0, 1.0]] * (n_rows - n_correct))
    return x, y


def _targets(network, x: torch.Tensor, *, agree: bool) -> torch.Tensor:
    """Targets built from a real network's own decoded predictions: all right, or all wrong."""
    with torch.no_grad():
        output = network.forward(x)
    if output.shape[1] == 1:
        predicted = output.reshape(-1) > 0.5
        labels = predicted if agree else ~predicted
        return labels.to(torch.float32).unsqueeze(1)
    predicted = torch.argmax(output, dim=1)
    labels = predicted if agree else (predicted + 1) % output.shape[1]
    return torch.nn.functional.one_hot(labels, output.shape[1]).to(torch.float32)


class TestFinalAccuracyValue:
    def test_multiclass_exact(self, mgr):
        _identity_network(mgr)
        mgr._val_x, mgr._val_y = _split(3)
        result = mgr._compute_final_eval_metrics()
        assert result["accuracy"] == 3 / 4

    def test_binary_exact(self, mgr):
        """One output column: scores [.9,.4,.2,.6,.8] against [1,1,0,0,1] -> rows 0, 2, 4 match."""
        _identity_network(mgr)
        mgr._val_x = torch.tensor([[0.9], [0.4], [0.2], [0.6], [0.8]])
        mgr._val_y = torch.tensor([[1.0], [1.0], [0.0], [0.0], [1.0]])
        result = mgr._compute_final_eval_metrics()
        assert result["accuracy"] == 3 / 5
        assert result["average"] == "binary"
        assert result["n_samples"] == 5, "accuracy and its siblings are scored on the same rows"

    def test_a_defined_accuracy_adds_nothing_to_undefined(self, mgr):
        _identity_network(mgr)
        mgr._val_x, mgr._val_y = _split(3)
        result = mgr._compute_final_eval_metrics()
        assert result["undefined"] == {}

    @pytest.mark.parametrize("output_size", [1, 2])
    def test_equals_the_history_definition_on_a_real_network(self, mgr, output_size):
        """The same number ``fit()`` records as ``value_accuracy`` for the same rows.

        ``CascadeCorrelationNetwork.calculate_accuracy`` is the call behind every
        ``train_accuracy`` / ``value_accuracy`` history entry.
        """
        mgr.create_network(input_size=2, output_size=output_size)
        network = mgr.network
        y_mixed = _targets(network, _X_VAL, agree=True)
        y_mixed[0] = _targets(network, _X_VAL[:1], agree=False)[0]
        mgr._val_x, mgr._val_y = _X_VAL, y_mixed
        result = mgr._compute_final_eval_metrics()
        assert result["accuracy"] == network.calculate_accuracy(_X_VAL, y_mixed)
        assert result["accuracy"] == 3 / 4


class TestFinalAccuracyFollowsTheSplitLabel:
    def test_test_pair_wins_and_is_labelled_test(self, mgr):
        _identity_network(mgr)
        mgr._val_x, mgr._val_y = _split(1)
        mgr._test_x, mgr._test_y = _split(3)
        result = mgr._compute_final_eval_metrics()
        assert result["split"] == "test"
        assert result["accuracy"] == 3 / 4, "scored on the rows the label names, not on the in-loop split (1/4)"
        assert result["final"] is True

    def test_validation_when_no_test_pair(self, mgr):
        _identity_network(mgr)
        mgr._val_x, mgr._val_y = _split(1)
        result = mgr._compute_final_eval_metrics()
        assert result["split"] == "validation"
        assert result["accuracy"] == 1 / 4

    @pytest.mark.parametrize("output_size", [1, 2])
    def test_real_network_scores_the_test_rows(self, mgr, output_size):
        """Every test row right and every validation row wrong, by construction from the network's own output."""
        mgr.create_network(input_size=2, output_size=output_size)
        network = mgr.network
        mgr._val_x, mgr._val_y = _X_VAL, _targets(network, _X_VAL, agree=False)
        mgr._test_x, mgr._test_y = _X_TEST, _targets(network, _X_TEST, agree=True)
        assert network.calculate_accuracy(_X_VAL, mgr._val_y) == 0.0, "precondition: the in-loop rows would score 0.0"
        result = mgr._compute_final_eval_metrics()
        assert result["split"] == "test"
        assert result["accuracy"] == network.calculate_accuracy(_X_TEST, mgr._test_y) == 1.0


class TestFinalAccuracyDegradation:
    def test_single_class_split_is_still_scored(self, mgr):
        """The siblings need a positive/negative contrast and degrade; accuracy needs none."""
        _identity_network(mgr)
        mgr._test_x = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.6, 0.4]])
        mgr._test_y = torch.tensor([[1.0, 0.0]] * 4)
        result = mgr._compute_final_eval_metrics()
        assert result["accuracy"] == 3 / 4
        assert "accuracy" not in result["undefined"]
        for key in _SIBLINGS:
            assert result[key] is None, key
            assert result["undefined"][key] == "single_class", key

    def test_empty_split_is_none_with_reason(self, mgr):
        _identity_network(mgr)
        mgr._test_x, mgr._test_y = torch.empty(0, 2), torch.empty(0, 2)
        result = mgr._compute_final_eval_metrics()
        assert result["split"] == "test"
        assert result["accuracy"] is None
        assert result["undefined"]["accuracy"] == "empty_batch"

    def test_non_finite_output_is_none_with_reason(self, mgr):
        _identity_network(mgr)
        mgr._test_x = torch.tensor([[float("nan"), 0.1], [0.2, 0.8]])
        mgr._test_y = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = mgr._compute_final_eval_metrics()
        assert result["accuracy"] is None
        assert result["undefined"]["accuracy"] == "invalid_output"
        for key in _SIBLINGS:
            assert result["undefined"][key] == "invalid_output", key


class TestFinalAccuracyOnTheMetricsSurface:
    def test_success_path_publishes_it_under_eval_metrics_final(self, mgr):
        """Drives the real ``_run_training`` success path with ``fit`` stubbed out, then reads ``get_metrics``."""
        mgr.create_network(input_size=2, output_size=2)
        network = mgr.network
        mgr._val_x, mgr._val_y = _X_VAL, _targets(network, _X_VAL, agree=False)
        mgr._test_x, mgr._test_y = _X_TEST, _targets(network, _X_TEST, agree=True)
        mgr.model.fit = lambda *args, **kwargs: None
        mgr._run_training(_X_VAL, mgr._val_y, _X_VAL, mgr._val_y)

        snapshot = mgr.get_metrics()
        final = snapshot["eval_metrics"]["final"]
        assert final["final"] is True
        assert final["split"] == "test"
        assert final["accuracy"] == 1.0
        assert set(snapshot["eval_metrics"]) == _LIVE_EVAL_METRICS_KEYS, "the live block gains nothing"
        assert "accuracy" not in snapshot, "no new top-level key; train_accuracy / val_accuracy are unchanged"


class TestFinalAccuracyOverRest:
    @pytest.fixture
    def client(self):
        app = create_app(Settings(auto_start=False))
        with TestClient(app) as test_client:
            yield test_client

    def test_get_metrics_serves_final_accuracy(self, client):
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})
        lifecycle = client.app.state.lifecycle
        network = lifecycle.network
        lifecycle._val_x, lifecycle._val_y = _X_VAL, _targets(network, _X_VAL, agree=False)
        lifecycle._test_x, lifecycle._test_y = _X_TEST, _targets(network, _X_TEST, agree=True)
        lifecycle._final_scalar_metrics = lifecycle._compute_final_eval_metrics()

        response = client.get("/v1/metrics")
        assert response.status_code == 200
        final = response.json()["data"]["eval_metrics"]["final"]
        assert final["split"] == "test"
        assert final["accuracy"] == 1.0

    def test_an_undefined_accuracy_is_served_as_null(self, client):
        """An undefined accuracy reaches the wire as ``null``, with its reason in ``undefined``.

        The reason is the load-bearing half. A NaN accuracy (what ``_accuracy`` returns
        for an empty batch) would also arrive as ``null``, because the response
        serializer converts it, so without the reason a reader could not tell why the
        value is missing.
        """
        client.post("/v1/network", json={"input_size": 2, "output_size": 2})
        lifecycle = client.app.state.lifecycle
        lifecycle._test_x, lifecycle._test_y = torch.empty(0, 2), torch.empty(0, 2)
        lifecycle._final_scalar_metrics = lifecycle._compute_final_eval_metrics()

        response = client.get("/v1/metrics")
        assert response.status_code == 200
        final = response.json()["data"]["eval_metrics"]["final"]
        assert final["accuracy"] is None
        assert final["undefined"]["accuracy"] == "empty_batch"
