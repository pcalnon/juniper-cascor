#!/usr/bin/env python
"""F2 — a start-fresh discards the model, not the operator's applied params.

juniper-canopy's restart modal applies the operator's edited params (``PATCH
/v1/training/params``) and THEN restarts. With **Start fresh** on, that restart is
``start_training(start_fresh=True)``: ``_start_fresh_reset_locked`` set ``model = None``
and create-on-start rebuilt the network from ``create_simple_config``'s defaults. Every
edit was silently replaced by an engine default. Observed through canopy's own routes
(juniper-ml ``reports/2026-09-23_canopy-a-n2-generate-stage-train-render/README.md``, F2):
``max_iterations`` / ``output_epochs`` / ``candidate_epochs`` / ``max_hidden_units`` read
8 / 60 / 40 / 32 before the restart and 1000000 / 10000 / 400 / 10 after it.

The owner ruled on 2026-09-24 that the edits are applied AFTER the fresh rebuild, so they
survive (juniper-ml ``notes/JUNIPER_2026-09-02_JUNIPER-CANOPY_SELECTION-REACHABILITY-DESIGN.md``
§12.4). The reset now captures the discarded network's params and ``start_training``
re-applies them to the vanilla network, through the same whitelist, nested setters and
atomic rollback as a PATCH. The start body's own params still land on top.

``_run_training`` is patched throughout: the network is created and the params applied
synchronously, before the training future is submitted.
"""

from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient

from api.app import create_app
from api.lifecycle.manager import TrainingLifecycleManager
from api.settings import Settings

pytestmark = pytest.mark.unit

#: The four params the A-N2 run observed being reset (F2), at the values it had applied.
OBSERVED_F2 = {"max_iterations": 8, "output_epochs": 60, "candidate_epochs": 40, "max_hidden_units": 32}

#: The rest of the carried surface: plain attributes, the two nested setters, and the
#: candidate-pool triple (applied as one post-merge-validated unit).
OTHER_EDITS = {
    "learning_rate": 0.123,
    "candidate_learning_rate": 0.0456,
    "patience": 7,
    "correlation_threshold": 0.25,
    "convergence_threshold": 0.0042,
    "candidate_patience": 9,
    "candidate_convergence_threshold": 0.0033,
    "init_output_weights": "random",
    "optimizer_type": "SGD",
    "activation_function_name": "ReLU",
    "candidate_pool_size": 6,
    "multi_candidate": True,
    "candidate_selection": "top",
    "selected_candidates": 2,
    "top_candidates": 2,
    "random_candidates": 0,
}

EDITS = {**OBSERVED_F2, **OTHER_EDITS}


@pytest.fixture
def mgr():
    m = TrainingLifecycleManager()
    try:
        yield m
    finally:
        m.shutdown()


def _vanilla_defaults():
    """The params a network built by create-on-start reports -- what a start-fresh used to leave."""
    probe = TrainingLifecycleManager()
    try:
        probe.create_network(input_size=3, output_size=2)
        return probe.get_training_params()
    finally:
        probe.shutdown()


def _start_fresh(m, n_features=3, **body):
    x = torch.zeros(4, n_features)
    y = torch.zeros(4, 2)
    with patch.object(m, "_run_training"):
        m.start_training(X=x, y=y, start_fresh=True, **body)


class TestF2StartFreshKeepsAppliedParams:
    def test_the_edits_differ_from_the_defaults_they_used_to_become(self):
        # Precondition: otherwise every assertion below would pass on the old behaviour too.
        defaults = _vanilla_defaults()
        assert {k: defaults[k] for k in EDITS} != EDITS
        for key, value in EDITS.items():
            if key not in {"candidate_selection", "random_candidates"}:
                assert defaults[key] != value, f"{key}: the default equals the edit, so this key proves nothing"

    def test_every_applied_param_survives_a_start_fresh(self, mgr):
        mgr.create_network(input_size=2, output_size=2)
        old_network = mgr.network
        mgr.update_params(EDITS)

        _start_fresh(mgr)

        assert mgr.network is not old_network  # really discarded and rebuilt...
        assert mgr.get_network_info()["input_size"] == 3
        assert mgr.get_network_info()["hidden_units"] == 0  # ...vanilla and untrained...
        params = mgr.get_training_params()
        assert {k: params[k] for k in EDITS} == EDITS  # ...and still carrying every edit

    def test_the_status_projection_reports_the_carried_values(self, mgr):
        # ``/v1/training/status`` reads a projected copy; the re-apply must re-project it.
        mgr.create_network(input_size=2, output_size=2)
        mgr.update_params(OBSERVED_F2)
        _start_fresh(mgr)
        state = mgr.training_state.get_state()
        assert state["max_hidden_units"] == 32
        assert state["max_iterations"] == 8

    def test_body_params_still_land_on_top(self, mgr):
        mgr.create_network(input_size=2, output_size=2)
        mgr.update_params(OBSERVED_F2)
        _start_fresh(mgr, max_hidden_units=5)
        params = mgr.get_training_params()
        assert params["max_hidden_units"] == 5  # the body wins
        assert params["output_epochs"] == 60  # the rest are carried

    def test_the_derived_cap_follows_the_carried_limits(self, mgr):
        # ``epochs_max`` is not carried (it is read-only); it is RE-DERIVED from the carried
        # granular limits, so it cannot contradict them.
        mgr.create_network(input_size=2, output_size=2)
        mgr.update_params(OBSERVED_F2)
        expected = mgr.get_training_params()["epochs_max"]
        _start_fresh(mgr)
        assert mgr.get_training_params()["epochs_max"] == expected

    def test_a_start_fresh_with_no_network_builds_at_the_defaults(self, mgr):
        assert mgr.network is None
        _start_fresh(mgr)
        params = mgr.get_training_params()
        defaults = _vanilla_defaults()
        assert {k: params[k] for k in EDITS} == {k: defaults[k] for k in EDITS}

    def test_the_reset_returns_the_carry_and_excludes_the_uncarried_keys(self, mgr):
        mgr.create_network(input_size=2, output_size=2)
        mgr.update_params({**OBSERVED_F2, "auto_snap_best": True, "auto_snap_min_epochs": 3})
        with mgr._lock:
            carried = mgr._start_fresh_reset_locked()
        assert {k: carried[k] for k in OBSERVED_F2} == OBSERVED_F2
        # Literal keys, not ``_START_FRESH_UNCARRIED_PARAMS``: a check that reads the constant
        # passes when the constant itself is emptied.
        assert not set(carried) & {"epochs_max", "auto_snap_best", "auto_snap_min_epochs"}
        # The auto-snap flags live on the lifecycle, and the reset leaves them alone.
        assert mgr._auto_snap_best is True and mgr._auto_snap_min_epochs == 3

    def test_the_rebuilt_network_takes_every_carried_param(self, mgr):
        # ``get_training_params`` is documented to return every updatable key, so the carry
        # must land whole. A key it gains that a PATCH cannot set would be reported skipped
        # here -- decide then whether to carry it, rather than let it warn on every restart.
        mgr.create_network(input_size=2, output_size=2)
        mgr.update_params(EDITS)
        with patch.object(mgr.logger, "warning") as warning:
            _start_fresh(mgr)
        assert not [c for c in warning.call_args_list if "did not take" in str(c.args[0])]

    def test_a_plain_start_still_continues_the_same_network(self, mgr):
        mgr.create_network(input_size=2, output_size=2)
        old_network = mgr.network
        mgr.update_params(OBSERVED_F2)
        with patch.object(mgr, "_run_training"):
            mgr.start_training(X=torch.zeros(4, 2), y=torch.zeros(4, 2))
        assert mgr.network is old_network
        assert {k: mgr.get_training_params()[k] for k in OBSERVED_F2} == OBSERVED_F2


class TestF2ThroughTheRoutes:
    """The sequence canopy's restart modal sends: PATCH the params, then a start-fresh start."""

    @pytest.fixture
    def client(self):
        with TestClient(create_app(Settings(auto_start=False))) as c:
            yield c

    def test_patch_then_start_fresh_keeps_the_patch(self, client):
        assert client.post("/v1/network", json={"input_size": 2, "output_size": 2}).status_code in (200, 201)
        resp = client.patch("/v1/training/params", json=OBSERVED_F2)
        assert resp.status_code == 200, resp.text
        lifecycle = client.app.state.lifecycle
        old_network = lifecycle.network
        with patch.object(lifecycle, "_run_training"):
            resp = client.post(
                "/v1/training/start",
                json={
                    "start_fresh": True,
                    "inline_data": {
                        "train_x": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
                        "train_y": [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
                    },
                },
            )
        assert resp.status_code == 200, resp.text
        assert lifecycle.network is not old_network
        params = client.get("/v1/training/params").json()["data"]
        assert {k: params[k] for k in OBSERVED_F2} == OBSERVED_F2
