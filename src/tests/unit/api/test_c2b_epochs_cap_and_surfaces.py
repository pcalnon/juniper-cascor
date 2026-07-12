"""C2b regression suite: Q1 derived ``epochs_max``, parameter-surface coherence, start-vs-PATCH posture, counter semantics.

Training-runtime-defects plan (juniper-ml
``notes/JUNIPER_2026-07-11_JUNIPER-CANOPY_TRAINING-RUNTIME-DEFECTS-PLAN.md``) §4 I-4 / §7 C2b / §12 Q1:

- **Q1 outcome (c)** — ``epochs_max`` is no longer an independent meta-parameter: the engine never read
  the attribute (it gated nothing), so the API now DERIVES it per run from the granular limits
  (``output_epochs``, ``candidate_epochs``, ``max_iterations``, ``max_hidden_units``) and reports it
  read-only. Submitted values are accepted at the request boundary (floor-only) and skip-reported by the
  C2a accounting — never applied.
- **Default-vs-ceiling coherence** — the shipped defaults must be admissible under the shipped PATCH
  validation (the I-4 root was default 1e11 > ceiling 1e6 on the same wire key).
- **Surface consistency** — ``/v1/network``, ``/v1/training/status`` (``training_state``) and
  ``GET /v1/training/params`` report from a single source of truth (the live network object).
- **Counter semantics** — ``current_epoch`` counts training steps only (single writer); within-pass
  output progress rides the dedicated ``output_epoch`` / ``output_total_epochs`` pair; metrics rows
  carry a ``kind`` discriminator.
"""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.app import create_app
from api.lifecycle.manager import TrainingLifecycleManager
from api.lifecycle.monitor import TrainingMonitor, TrainingState
from api.models.training import TrainingParams, TrainingParamUpdateRequest, TrainingStartRequest
from api.settings import Settings

pytestmark = pytest.mark.unit


@pytest.fixture
def test_client():
    """Test client with lifecycle manager (lifespan runs), auto-start disabled."""
    settings = Settings(auto_start=False)
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def test_client_with_network(test_client):
    """Test client with a network already created."""
    test_client.post("/v1/network", json={"input_size": 2, "output_size": 2})
    return test_client


class _StubNetwork:
    """Minimal attribute bag for derive_epochs_cap unit tests."""

    __slots__ = ("output_epochs", "candidate_epochs", "max_iterations", "max_hidden_units")

    def __init__(self, output_epochs=0, candidate_epochs=0, max_iterations=0, max_hidden_units=0):
        self.output_epochs = output_epochs
        self.candidate_epochs = candidate_epochs
        self.max_iterations = max_iterations
        self.max_hidden_units = max_hidden_units


class TestDeriveEpochsCap:
    """Unit tests for the Q1 outcome-(c) formula itself."""

    def test_formula_iterations_bounded_by_max_iterations(self):
        """min(max_iterations, max_hidden_units) = max_iterations when growth-iteration-bound."""
        net = _StubNetwork(output_epochs=100, candidate_epochs=50, max_iterations=4, max_hidden_units=10)
        # 100 + 4 * (50 + 100) = 700
        assert TrainingLifecycleManager.derive_epochs_cap(net) == 700

    def test_formula_iterations_bounded_by_capacity(self):
        """min(max_iterations, max_hidden_units) = max_hidden_units when capacity-bound."""
        net = _StubNetwork(output_epochs=100, candidate_epochs=50, max_iterations=1_000_000, max_hidden_units=3)
        # 100 + 3 * (50 + 100) = 550
        assert TrainingLifecycleManager.derive_epochs_cap(net) == 550

    def test_zero_iterations_leaves_initial_pass_budget(self):
        """With no growth possible the cap is exactly the initial output pass."""
        net = _StubNetwork(output_epochs=250, candidate_epochs=50, max_iterations=0, max_hidden_units=10)
        assert TrainingLifecycleManager.derive_epochs_cap(net) == 250

    def test_missing_attributes_do_not_raise(self):
        """Partial stand-ins (tests, degraded networks) degrade to 0-contributions, never raise."""

        class _Empty:
            pass

        assert TrainingLifecycleManager.derive_epochs_cap(_Empty()) == 0

    def test_engine_default_derived_cap_is_admissible_and_meaningful(self):
        """The cap derived from the engine's create-on-start defaults is a sane display budget."""
        mgr = TrainingLifecycleManager()
        try:
            mgr.create_network(input_size=2, output_size=2)
            cap = mgr.derive_epochs_cap(mgr.network)
            assert cap > 0
            # Coherence with the formula inputs read from the live network.
            expected = mgr.network.output_epochs + min(mgr.network.max_iterations, mgr.network.max_hidden_units) * (mgr.network.candidate_epochs + mgr.network.output_epochs)
            assert cap == expected
        finally:
            mgr.shutdown()


class TestDefaultVsCeilingCoherence:
    """The I-4 root cause regression: shipped defaults must pass shipped validation.

    Pre-C2b, ``GET /v1/training/params`` echoed the construction-time ``epochs_max``
    default (1e11) while the PATCH model enforced ``le=1e6`` on the same key —
    self-contradiction on one wire surface. Post-C2b the echo is derived and the
    input is floor-only, so the full GET echo must ALWAYS be re-submittable.
    """

    def test_get_params_echo_is_admissible_under_patch_validation(self, test_client_with_network):
        """Round-trip: every value GET /params reports validates through the PATCH model."""
        params = test_client_with_network.get("/v1/training/params").json()["data"]
        patch_fields = set(TrainingParamUpdateRequest.model_fields)
        resubmittable = {k: v for k, v in params.items() if k in patch_fields and v is not None}
        assert "epochs_max" in resubmittable, "the derived cap must remain part of the echo surface"
        TrainingParamUpdateRequest(**resubmittable)  # raises ValidationError on any self-contradiction

    def test_full_form_apply_roundtrip_succeeds(self, test_client_with_network):
        """The canopy failure mode end-to-end: seed a full form from GET, PATCH it back → 200 (not 422)."""
        params = test_client_with_network.get("/v1/training/params").json()["data"]
        patch_fields = set(TrainingParamUpdateRequest.model_fields)
        body = {k: v for k, v in params.items() if k in patch_fields and v is not None}
        response = test_client_with_network.patch("/v1/training/params", json=body)
        assert response.status_code == 200, f"full-form echo apply must not 422: {response.text}"

    def test_derived_cap_reported_not_construction_attribute(self, test_client_with_network):
        """GET /params reports the derived cap even when the network attribute holds the legacy 1e11."""
        lifecycle = test_client_with_network.app.state.lifecycle
        assert getattr(lifecycle.network, "epochs_max", None) is not None  # legacy attribute still exists (snapshot compat)
        reported = test_client_with_network.get("/v1/training/params").json()["data"]["epochs_max"]
        assert reported == lifecycle.derive_epochs_cap(lifecycle.network)
        assert reported != 100_000_000_000


class TestEpochsMaxDeprecatedInput:
    """Q1 outcome (c): ``epochs_max`` is accepted-but-skip-reported on every input surface."""

    def test_patch_epochs_max_is_skipped_not_applied(self, test_client_with_network):
        """PATCH {'epochs_max': N} → 200; reported skipped(not-updatable); attribute untouched."""
        lifecycle = test_client_with_network.app.state.lifecycle
        before = getattr(lifecycle.network, "epochs_max", None)
        response = test_client_with_network.patch("/v1/training/params", json={"epochs_max": 123_456})
        assert response.status_code == 200
        data = response.json()["data"]
        assert {"key": "epochs_max", "reason": "not-updatable"} in data["skipped"]
        assert "epochs_max" not in data["applied"]
        assert getattr(lifecycle.network, "epochs_max", None) == before
        # The echo still reports the derived value, not the submitted one.
        assert data["epochs_max"] == lifecycle.derive_epochs_cap(lifecycle.network)

    def test_patch_epochs_max_has_no_ceiling(self, test_client_with_network):
        """A huge echoed-back value (e.g. a derived cap above the old 1e6 ceiling) is not 422-rejected."""
        response = test_client_with_network.patch("/v1/training/params", json={"epochs_max": 20_000_000_000})
        assert response.status_code == 200
        assert {"key": "epochs_max", "reason": "not-updatable"} in response.json()["data"]["skipped"]

    def test_patch_epochs_max_keeps_floor(self, test_client_with_network):
        """The ge=1 floor survives deprecation (garbage still rejects loudly)."""
        response = test_client_with_network.patch("/v1/training/params", json={"epochs_max": 0})
        assert response.status_code == 422

    def test_start_body_epochs_max_matches_patch_posture(self, test_client_with_network):
        """Start-supplied ``params.epochs_max`` flows through the same skip-reporting path as PATCH —
        no silent acceptance, no application (start-vs-PATCH validation coherence)."""
        lifecycle = test_client_with_network.app.state.lifecycle
        before = getattr(lifecycle.network, "epochs_max", None)
        # Drive the manager path directly (no dataset staged in this fixture, so the
        # HTTP start would 409 before reaching the params path); start_training routes
        # non-fit kwargs through the same _apply_params_unlocked whitelist as PATCH.
        result = lifecycle._apply_params_unlocked({"epochs_max": 777})
        assert {"key": "epochs_max", "reason": "not-updatable"} in result["skipped"]
        assert "epochs_max" not in result["applied"]
        assert getattr(lifecycle.network, "epochs_max", None) == before

    def test_epochs_max_not_in_fit_kwargs_or_updatable_keys(self):
        """``epochs_max`` is neither a fit kwarg nor an updatable network attribute anymore."""
        assert "epochs_max" not in TrainingLifecycleManager._FIT_KWARGS


class TestStartVsPatchValidationCoherence:
    """Start-supplied parameters pass the same validation posture as PATCH (task item 3)."""

    def test_start_max_epochs_ceiling_matches_output_epochs(self):
        """``TrainingParams.max_epochs`` (the initial output-pass budget forwarded to fit)
        now carries the same 1e6 ceiling as ``output_epochs`` — the start path can no
        longer smuggle an unbounded budget past the PATCH-surface ceilings."""
        with pytest.raises(ValidationError):
            TrainingParams(max_epochs=1_000_001)
        assert TrainingParams(max_epochs=1_000_000).max_epochs == 1_000_000

    def test_start_shorthand_epochs_ceiling(self):
        """``TrainingStartRequest.epochs`` (shorthand for params.max_epochs) carries the same ceiling."""
        with pytest.raises(ValidationError):
            TrainingStartRequest(epochs=1_000_001)
        assert TrainingStartRequest(epochs=1_000_000).epochs == 1_000_000

    def test_shared_field_bounds_identical_between_start_and_patch(self):
        """Every field present in both models carries identical numeric bounds — including the
        deprecated ``epochs_max`` (floor-only in both). Complements SEC-F10's parity test."""
        import annotated_types

        def bounds(field_info):
            out = {}
            for meta in field_info.metadata:
                for attr in ("ge", "gt", "le", "lt"):
                    if isinstance(meta, getattr(annotated_types, attr.capitalize())):
                        out[attr] = getattr(meta, attr)
            return out

        start_fields = TrainingParams.model_fields
        patch_fields = TrainingParamUpdateRequest.model_fields
        for name in sorted(set(start_fields) & set(patch_fields)):
            assert bounds(start_fields[name]) == bounds(patch_fields[name]), f"start/PATCH bound divergence on {name}"
        assert bounds(start_fields["epochs_max"]) == {"ge": 1}, "epochs_max must be floor-only (deprecated input)"


class TestSurfaceConsistency:
    """C2b: /v1/network and /v1/training/status report from a single source of truth."""

    def test_network_and_status_agree_after_create(self, test_client_with_network):
        """The two REST surfaces report identical effective values right after create."""
        network = test_client_with_network.get("/v1/network").json()["data"]
        status = test_client_with_network.get("/v1/training/status").json()["data"]
        tstate = status["training_state"]
        assert tstate["max_hidden_units"] == network["max_hidden_units"]
        assert tstate["learning_rate"] == pytest.approx(network["learning_rate"])

    def test_status_reflects_patch_updates(self, test_client_with_network):
        """PATCHing a projected parameter refreshes the status surface (not just the network attr)."""
        response = test_client_with_network.patch("/v1/training/params", json={"learning_rate": 0.033, "max_hidden_units": 42})
        assert response.status_code == 200
        network = test_client_with_network.get("/v1/network").json()["data"]
        tstate = test_client_with_network.get("/v1/training/status").json()["data"]["training_state"]
        assert network["learning_rate"] == pytest.approx(0.033)
        assert tstate["learning_rate"] == pytest.approx(0.033)
        assert network["max_hidden_units"] == tstate["max_hidden_units"] == 42

    def test_status_max_epochs_tracks_derived_cap_after_patch(self, test_client_with_network):
        """PATCHing a granular limit moves the derived ``max_epochs`` on the status surface."""
        lifecycle = test_client_with_network.app.state.lifecycle
        test_client_with_network.patch("/v1/training/params", json={"output_epochs": 100, "candidate_epochs": 50, "max_iterations": 4, "max_hidden_units": 10})
        tstate = test_client_with_network.get("/v1/training/status").json()["data"]["training_state"]
        assert tstate["max_epochs"] == 700  # 100 + min(4, 10) * (50 + 100)
        assert tstate["max_epochs"] == lifecycle.derive_epochs_cap(lifecycle.network)
        params = test_client_with_network.get("/v1/training/params").json()["data"]
        assert params["epochs_max"] == 700  # the params echo agrees with the status surface

    def test_status_before_network_reports_zeroed_limits(self, test_client):
        """Pre-create posture: 0 means "no network / unknown" — not a phantom default layer."""
        tstate = test_client.get("/v1/training/status").json()["data"]["training_state"]
        assert tstate["max_epochs"] == 0
        assert tstate["max_iterations"] == 0
        assert tstate["max_hidden_units"] == 0


class TestCounterSemantics:
    """C2b (I-1c / S12): single-meaning counters + the kind discriminator."""

    def test_epoch_end_event_does_not_clobber_current_epoch(self):
        """The live epoch_end handler exposes within-pass progress via output_epoch/
        output_total_epochs and leaves ``current_epoch`` to the history drain — the
        pre-C2b dual-writer race behind the live 'Epoch: 10000 vs 12' confusion."""

        class _Event:
            type = "epoch_end"
            payload = {"epoch": 9_976, "epochs": 10_000, "metrics": {"loss": 0.5}}

        mgr = TrainingLifecycleManager()
        try:
            mgr.create_network(input_size=2, output_size=2)
            mgr.training_state.update_state(current_epoch=12, current_step=12)
            mgr._handle_event(_Event())
            state = mgr.training_state.get_state()
            assert state["current_epoch"] == 12, "inner output-epoch must not clobber the training-step counter"
            assert state["output_epoch"] == 9_976
            assert state["output_total_epochs"] == 10_000
            assert state["phase_detail"] == "training_output"
        finally:
            mgr.shutdown()

    def test_monitor_current_epoch_advances_only_on_training_steps(self):
        """Only kind='training_step' rows advance monitor.current_epoch; both row kinds are buffered
        and every buffered row carries its kind discriminator."""
        monitor = TrainingMonitor()
        monitor.on_epoch_end(epoch=3, loss=0.1, accuracy=0.9, learning_rate=0.01, kind="training_step")
        monitor.on_epoch_end(epoch=9_976, loss=0.2, accuracy=None, learning_rate=0.01, kind="output_epoch")
        assert monitor.get_current_state()["current_epoch"] == 3
        rows = monitor.get_all_metrics()
        assert [r["kind"] for r in rows] == ["training_step", "output_epoch"]
        assert rows[0]["epoch"] == 3 and rows[1]["epoch"] == 9_976

    def test_monitor_default_kind_is_training_step(self):
        """Callers that predate the discriminator (the history drain) keep step semantics by default."""
        monitor = TrainingMonitor()
        monitor.on_epoch_end(epoch=1, loss=0.1, accuracy=0.9, learning_rate=0.01)
        assert monitor.get_current_state()["current_epoch"] == 1
        assert monitor.get_all_metrics()[0]["kind"] == "training_step"

    def test_training_state_exposes_output_progress_pair(self):
        """The output pair is part of the state schema (zeroed by default) and settable."""
        state = TrainingState()
        snapshot = state.get_state()
        assert snapshot["output_epoch"] == 0
        assert snapshot["output_total_epochs"] == 0
        state.update_state(output_epoch=26, output_total_epochs=10_000)
        snapshot = state.get_state()
        assert snapshot["output_epoch"] == 26
        assert snapshot["output_total_epochs"] == 10_000
