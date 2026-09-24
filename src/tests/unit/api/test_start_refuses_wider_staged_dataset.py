#!/usr/bin/env python
"""F1 — a Start that continues the current network refuses a wider staged dataset BEFORE binding it.

A plain Start continues the current network (``start_fresh=False``). ``start_training``
pads a narrower dataset up to that network, but it cannot widen one: only
``swap_dataset_live`` grows a network, so the pad step refuses a wider dataset (409). That
refusal used to fire only AFTER ``_reload_dataset`` had bound the staged dataset and cleared
the staged slot. juniper-ml's A-N2 run observed the result (juniper-ml
``reports/2026-09-23_canopy-a-n2-generate-stage-train-render/README.md``, F1). ``equities``
(15 features) was staged over a 2x2 network left by ``checkerboard``, and Start answered 409.
Every route then said ``current_dataset: equities`` beside checkerboard's network, status and
metrics, and a retried Start refused identically, because nothing was staged any more.

The owner ruled on 2026-09-24 (juniper-ml
``notes/JUNIPER_2026-09-02_JUNIPER-CANOPY_SELECTION-REACHABILITY-DESIGN.md`` §12.4). Detect
the mismatch before the Start commits anything, refuse with a message that points at Start
fresh, stop serving the previous run's results under the new label, and cover ``mnist`` (784
features) too. The ``equities`` seed stays enabled.

``juniper_data_client`` is replaced at the FAR seam (the HTTP client), so the REAL
``_reload_dataset`` runs and is what refuses. ``_run_training`` is patched: only the
synchronous half of a start runs.

#688's validation added two arms: the refusal also leaves which partitions the record
stands on alone, and a start that carried inline tensors is told only that the STAGED
dataset was not loaded -- its own tensors were.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from api.app import create_app
from api.lifecycle.manager import TrainingLifecycleManager
from api.settings import Settings
from cascor_constants.constants_api import _PROJECT_API_START_FRESH_REQUIRED_MARKER

pytestmark = pytest.mark.unit


def _artifact(n_features, n_outputs, rows=6):
    """An NPZ-shaped dict with all three partitions, ``n_features`` wide and ``n_outputs`` deep."""
    labels = np.eye(n_outputs, dtype=np.float32)[np.arange(rows) % n_outputs]
    part = {"X": np.ones((rows, n_features), dtype=np.float32), "y": labels}
    return {f"{k}_{split}": v.copy() for split in ("train", "val", "test") for k, v in part.items()}


@pytest.fixture
def serve(monkeypatch):
    """Offline juniper-data. ``serve(n_features, n_outputs)`` sets the next artifact's shape."""
    instance = MagicMock()
    instance.create_dataset.return_value = {"dataset_id": "fake-id", "meta": {}}
    module = types.ModuleType("juniper_data_client")
    module.JuniperDataClient = MagicMock(return_value=instance)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "juniper_data_client", module)

    def _serve(n_features, n_outputs):
        instance.download_artifact_npz.return_value = _artifact(n_features, n_outputs)

    return _serve


@pytest.fixture
def mgr():
    m = TrainingLifecycleManager()
    try:
        yield m
    finally:
        m.shutdown()


def _start(m, **kwargs):
    with patch.object(m, "_run_training"):
        return m.start_training(**kwargs)


def _snapshot(m):
    """Everything a refused start must leave exactly as it found it."""
    return {
        "network": m.network,
        "tensors": (m._train_x, m._train_y, m._val_x, m._val_y, m._test_x, m._test_y),
        "current": m.get_status()["current_dataset"],
        "shortfall": m._dataset_shortfall,
        # Which partitions the record stands on (owner ruling 2026-09-24). #688's
        # validation: a refusal that rewrote this set to the REFUSED artifact's
        # partitions (mutant NV1) left every other field here unchanged.
        "described": m._described_partitions,
        "warning": m._validation_warning,
    }


def _previous_run(m, serve):
    """The A-N2 state before F1: a 2x2 network that has trained on a staged 2-feature dataset."""
    serve(2, 2)
    m.stage_dataset_config(dataset_type="checkerboard", n_samples=6)
    _start(m)
    assert m.get_network_info()["input_size"] == 2
    return _snapshot(m)


def _assert_nothing_moved(m, before, staged):
    status = m.get_status()
    assert status["current_dataset"] == before["current"]  # the label did NOT move...
    assert status["pending_dataset"] == staged  # ...the staged dataset is still staged...
    assert m.network is before["network"]  # ...and the network, the data and their
    for now, then in zip((m._train_x, m._train_y, m._val_x, m._val_y, m._test_x, m._test_y), before["tensors"]):
        assert now is then  # annotations are the previous run's, untouched
    assert m._dataset_shortfall == before["shortfall"]
    assert m._described_partitions == before["described"]
    assert m._validation_warning == before["warning"]
    assert m.get_dataset()["input_features"] == 2


class TestAStartThatContinuesRefusesAWiderStagedDataset:
    @pytest.mark.parametrize(
        ("dataset_type", "n_features", "n_outputs"),
        [("equities", 15, 2), ("mnist", 784, 10), ("xor", 2, 3)],
        ids=["equities-15-features", "mnist-784-features-10-outputs", "wider-outputs-only"],
    )
    def test_the_refusal_binds_nothing_and_keeps_the_staged_dataset(self, mgr, serve, dataset_type, n_features, n_outputs):
        before = _previous_run(mgr, serve)
        serve(n_features, n_outputs)
        mgr.stage_dataset_config(dataset_type=dataset_type, n_samples=6)
        staged = mgr.get_status()["pending_dataset"]

        with pytest.raises(ValueError) as refused:
            _start(mgr)

        message = str(refused.value)
        assert message.startswith(_PROJECT_API_START_FRESH_REQUIRED_MARKER)
        assert f"'{dataset_type}' ({n_features} features, {n_outputs} outputs)" in message
        assert "(2 inputs, 2 outputs)" in message
        assert "start_fresh" in message
        assert "The staged dataset was not loaded: it is still staged" in message
        _assert_nothing_moved(mgr, before, staged)

    def test_a_retry_refuses_the_same_way_and_start_fresh_then_consumes_it(self, mgr, serve):
        before = _previous_run(mgr, serve)
        serve(15, 2)
        mgr.stage_dataset_config(dataset_type="equities", n_samples=6)
        staged = mgr.get_status()["pending_dataset"]
        for _attempt in range(2):
            with pytest.raises(ValueError, match=r"^\[start_fresh_required\]"):
                _start(mgr)
            _assert_nothing_moved(mgr, before, staged)

        _start(mgr, start_fresh=True)
        status = mgr.get_status()
        assert status["pending_dataset"] is None
        assert status["current_dataset"]["dataset_type"] == "equities"
        assert mgr.get_network_info()["input_size"] == 15

    def test_a_refusal_after_an_inline_train_start_keeps_the_partitions_the_record_stands_on(self, mgr, serve):
        """#688's validation, MEDIUM: nothing tested that the refusal leaves ``_described_partitions`` alone.

        After an inline train-only start, the fetch's record stands on the fetch's
        val and test alone (owner ruling 2026-09-24, "keep while fetched splits
        stay"). The refused dataset has all three partitions, so a refusal that
        recorded ITS partitions -- mutant NV1 moves that assignment above the
        refusal -- changed nothing else any arm looked at. The next inline val+test
        start then kept a record none of whose partitions was still loaded.
        """
        _previous_run(mgr, serve)
        _start(mgr, X=torch.ones(6, 2), y=torch.zeros(6, 2))
        before = _snapshot(mgr)
        assert before["described"] == frozenset({"val", "test"}), "the record does not stand on val and test alone -- the arm proves nothing"
        serve(15, 2)
        mgr.stage_dataset_config(dataset_type="equities", n_samples=6)
        staged = mgr.get_status()["pending_dataset"]
        with pytest.raises(ValueError, match=r"^\[start_fresh_required\]"):
            _start(mgr)
        _assert_nothing_moved(mgr, before, staged)

        # ...and the set still drives the rule: replacing val and test leaves nothing of the record.
        mgr.clear_pending_dataset_config()
        _start(mgr, X_val=torch.ones(4, 2), y_val=torch.zeros(4, 2), X_test=torch.ones(4, 2), y_test=torch.zeros(4, 2))
        assert mgr.get_status()["current_dataset"] == {"dataset_type": None}
        assert mgr._dataset_shortfall is None
        assert mgr._described_partitions == frozenset()

    def test_a_start_that_also_binds_inline_tensors_is_told_only_the_staged_dataset_was_not_loaded(self, mgr, serve):
        """#688's validation, LOW: the refusal said "Nothing was loaded" beside tensors the same start had bound.

        Inline tensors bind BEFORE the staged reload, so that a staged fetch that goes
        ahead replaces them. A start carrying both has therefore bound its own
        tensors by the time the staged dataset is refused, and the refusal must say
        only what is true: the STAGED dataset was not loaded.
        """
        _previous_run(mgr, serve)
        serve(15, 2)
        mgr.stage_dataset_config(dataset_type="equities", n_samples=6)
        inline_x = torch.ones(6, 2)
        with pytest.raises(ValueError, match=r"^\[start_fresh_required\]") as refused:
            _start(mgr, X=inline_x, y=torch.zeros(6, 2))
        assert mgr._train_x is inline_x, "the inline train split was not bound -- the arm proves nothing"
        message = str(refused.value)
        assert "The staged dataset was not loaded: it is still staged" in message
        assert "Nothing was loaded" not in message
        assert mgr.get_status()["pending_dataset"] is not None

    def test_the_token_is_the_value_canopy_matches(self):
        # juniper-canopy recognises this refusal by the token alone; changing it breaks
        # that match silently, so it is pinned here as well as there.
        assert _PROJECT_API_START_FRESH_REQUIRED_MARKER == "[start_fresh_required]"


class TestWhatStillStarts:
    def test_a_narrower_dataset_is_still_padded_and_started(self, mgr, serve):
        mgr.create_network(input_size=15, output_size=2)
        serve(2, 2)
        mgr.stage_dataset_config(dataset_type="checkerboard", n_samples=6)
        _start(mgr)
        assert mgr.get_status()["pending_dataset"] is None
        assert mgr.get_status()["current_dataset"]["dataset_type"] == "checkerboard"
        assert mgr._train_x.shape[1] == 15  # padded up to the network

    def test_an_equally_wide_dataset_starts(self, mgr, serve):
        _previous_run(mgr, serve)
        serve(2, 2)
        mgr.stage_dataset_config(dataset_type="gaussian", n_samples=6)
        _start(mgr)
        assert mgr.get_status()["current_dataset"]["dataset_type"] == "gaussian"

    def test_with_no_network_the_wide_dataset_builds_its_own(self, mgr, serve):
        serve(784, 10)
        mgr.stage_dataset_config(dataset_type="mnist", n_samples=6)
        _start(mgr)
        assert (mgr.get_network_info()["input_size"], mgr.get_network_info()["output_size"]) == (784, 10)

    def test_start_fresh_is_never_refused(self, mgr, serve):
        _previous_run(mgr, serve)
        serve(15, 2)
        mgr.stage_dataset_config(dataset_type="equities", n_samples=6)
        _start(mgr, start_fresh=True)
        assert mgr.get_network_info()["input_size"] == 15


class TestThroughTheRoutes:
    @pytest.fixture
    def client(self, serve):
        with TestClient(create_app(Settings(auto_start=False))) as c:
            yield c

    def test_the_409_names_start_fresh_and_the_status_keeps_the_previous_label(self, client, serve):
        lifecycle = client.app.state.lifecycle
        before = _previous_run(lifecycle, serve)
        serve(15, 2)
        lifecycle.stage_dataset_config(dataset_type="equities", n_samples=6)

        with patch.object(lifecycle, "_run_training"):
            resp = client.post("/v1/training/start", json={})
        assert resp.status_code == 409, resp.text
        # The API-09 envelope carries the HTTPException detail as ``error.message``.
        message = resp.json()["error"]["message"]
        assert message.startswith(f"Training cannot be started: {_PROJECT_API_START_FRESH_REQUIRED_MARKER}")

        status = client.get("/v1/training/status").json()["data"]
        assert status["current_dataset"] == before["current"]
        assert status["pending_dataset"]["dataset_type"] == "equities"
        assert client.get("/v1/dataset").json()["data"]["input_features"] == 2
