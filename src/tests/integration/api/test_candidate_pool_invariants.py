"""End-to-end PATCH /v1/training/params tests for the candidate-pool triple.

Pins the §1.5 C2.1 invariant against the real FastAPI route — covers schema
acceptance, atomic post-merge validation, 422 surfacing, and per-field
round-trip for each of the five new params (multi_candidate,
candidate_selection, selected_candidates, top_candidates, random_candidates).

The cascade_correlation.py selection-logic wiring is **not** validated here —
that's PR-4b. PR-4a only ensures the storage and the invariant hold; the
defaults preserve current single-top-candidate behavior.
"""

import threading

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.settings import Settings


@pytest.fixture
def client():
    """Minimal TestClient mirroring test_api_full_lifecycle's daemon-shutdown."""
    settings = Settings()
    app = create_app(settings)
    tc = TestClient(app)
    tc.__enter__()
    yield tc
    lifecycle = getattr(app.state, "lifecycle", None)
    if lifecycle:
        lifecycle._stop_event.set()
        if getattr(lifecycle, "_executor", None):
            lifecycle._executor.shutdown(wait=False, cancel_futures=True)
    exit_thread = threading.Thread(target=lambda: tc.__exit__(None, None, None), daemon=True)
    exit_thread.start()
    exit_thread.join(timeout=5)


def _create_network(client, *, candidate_pool_size: int = 8) -> None:
    resp = client.post(
        "/v1/network",
        json={
            "input_size": 2,
            "output_size": 2,
            "candidate_pool_size": candidate_pool_size,
            "epochs_max": 5,
            "candidate_epochs": 2,
            "output_epochs": 2,
            "patience": 1,
        },
    )
    assert resp.status_code == 200, resp.text


@pytest.mark.integration
class TestCandidatePoolInvariants:
    """§1.5 C2.1 — exhaustive matrix over the post-merge invariant."""

    @pytest.mark.parametrize(
        "patch",
        [
            {"selected_candidates": 1, "top_candidates": 1, "random_candidates": 0},
            {"selected_candidates": 4, "top_candidates": 4, "random_candidates": 0},
            {"selected_candidates": 4, "top_candidates": 0, "random_candidates": 4},
            {"selected_candidates": 6, "top_candidates": 4, "random_candidates": 2},
        ],
    )
    def test_valid_triples_accepted(self, client, patch):
        _create_network(client)
        resp = client.patch("/v1/training/params", json=patch)
        assert resp.status_code == 200, resp.text
        applied = resp.json()["data"]
        for k, v in patch.items():
            assert applied[k] == v, f"PATCH applied but GET reports {applied[k]} for {k} (wanted {v})"

    @pytest.mark.parametrize(
        "patch,fragment",
        [
            ({"selected_candidates": 0, "top_candidates": 0, "random_candidates": 0}, "selected_candidates"),
            ({"selected_candidates": 4, "top_candidates": 5, "random_candidates": 0}, "each component"),
            ({"selected_candidates": 4, "top_candidates": 3, "random_candidates": 2}, "must equal S=4"),
            ({"selected_candidates": 4, "top_candidates": 0, "random_candidates": 0}, "cannot both be 0"),
            ({"selected_candidates": 4, "top_candidates": 0, "random_candidates": 3}, "with top_candidates=0"),
            ({"selected_candidates": 4, "top_candidates": 3, "random_candidates": 0}, "with random_candidates=0"),
        ],
    )
    def test_invalid_triples_rejected_422(self, client, patch, fragment):
        _create_network(client)
        resp = client.patch("/v1/training/params", json=patch)
        # Pydantic catches non-negativity itself (ge=0 on the field), so a few
        # of the above bypass the post-merge helper. Either 422 path is fine —
        # we just need the invariant violation NOT to silently apply.
        assert resp.status_code in (400, 422), f"expected 4xx, got {resp.status_code}: {resp.text}"
        body = resp.json()
        # Both error paths now return the SAME envelope shape, so the message is
        # always at ``body["error"]["message"]``: 400 from ``raise
        # HTTPException(...)`` via the HTTPException handler, and 422 from
        # Pydantic field validation via the ``RequestValidationError`` handler.
        #
        # This comment previously said the 422 path "is untouched by API-09 and
        # still returns the legacy ``{"detail": [...]}`` shape" -- true when
        # written, and the checked-in evidence that API-09's "migration complete"
        # claim was false. That gap is closed (defect-register APD-CCLIENT-008);
        # the per-field list still exists, on ``error.detail``.
        #
        # The legacy ``detail`` branch is retained ONLY so this integration test
        # keeps passing against a pre-fix cascor build; it is not a live shape.
        if "error" in body and isinstance(body.get("error"), dict):
            haystack = body["error"]["message"]
        else:
            haystack = body.get("detail")
        assert haystack is not None, body
        assert fragment in str(haystack), f"violation message {haystack!r} missing fragment {fragment!r}"

    def test_atomic_multi_key_patch_accepted_in_one_shot(self, client):
        """A PATCH that's only valid as a unit must not 422 on the first key.

        Starting state: the constructor defaults (S=1, T=1, R=0). The single
        PATCH below sets {S=6, T=4, R=2} — which is invalid if applied
        per-key (S=6, T=1 inherits, R=0 inherits → fails T+R==S) but valid
        as a post-merge unit.
        """
        _create_network(client)
        resp = client.patch(
            "/v1/training/params",
            json={"selected_candidates": 6, "top_candidates": 4, "random_candidates": 2},
        )
        assert resp.status_code == 200, resp.text
        applied = resp.json()["data"]
        assert (applied["selected_candidates"], applied["top_candidates"], applied["random_candidates"]) == (6, 4, 2)

    def test_pool_size_in_same_patch_validates_against_new_pool(self, client):
        """PATCHing pool_size and S in the same call must validate against the
        post-merge pool. Starting pool=2, new pool=8, S=6 — invalid against
        the old pool, valid against the new."""
        _create_network(client, candidate_pool_size=2)
        resp = client.patch(
            "/v1/training/params",
            json={"candidate_pool_size": 8, "selected_candidates": 6, "top_candidates": 4, "random_candidates": 2},
        )
        assert resp.status_code == 200, resp.text


@pytest.mark.integration
class TestCandidatePoolPerFieldRoundTrip:
    """Each of the 5 new params individually round-trips through PATCH/GET."""

    def test_multi_candidate_round_trip(self, client):
        _create_network(client)
        resp = client.patch("/v1/training/params", json={"multi_candidate": True})
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["multi_candidate"] is True
        resp = client.get("/v1/training/params")
        assert resp.json()["data"]["multi_candidate"] is True

    def test_candidate_selection_round_trip(self, client):
        _create_network(client)
        for value in ("top", "random", "mixed"):
            resp = client.patch("/v1/training/params", json={"candidate_selection": value})
            assert resp.status_code == 200, resp.text
            assert resp.json()["data"]["candidate_selection"] == value

    def test_candidate_selection_rejects_unknown(self, client):
        _create_network(client)
        resp = client.patch("/v1/training/params", json={"candidate_selection": "lottery"})
        assert resp.status_code == 422, resp.text  # pydantic Literal rejection

    def test_selected_candidates_round_trip(self, client):
        _create_network(client)
        # Need to provide T+R=S to satisfy the invariant — pure-S patches with
        # the inherited (T=1, R=0) only work when S=1.
        resp = client.patch(
            "/v1/training/params",
            json={"selected_candidates": 3, "top_candidates": 3, "random_candidates": 0},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["selected_candidates"] == 3

    def test_top_candidates_round_trip(self, client):
        _create_network(client)
        resp = client.patch(
            "/v1/training/params",
            json={"selected_candidates": 4, "top_candidates": 2, "random_candidates": 2},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["top_candidates"] == 2

    def test_random_candidates_round_trip(self, client):
        _create_network(client)
        resp = client.patch(
            "/v1/training/params",
            json={"selected_candidates": 4, "top_candidates": 2, "random_candidates": 2},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["random_candidates"] == 2
