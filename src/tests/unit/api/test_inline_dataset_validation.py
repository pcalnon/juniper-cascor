"""Unit coverage for InlineDataset cross-field alignment validators.

Mismatched train/val sample counts and half-specified validation splits used
to pass Pydantic (per-field list length only) and fail later inside
``torch.tensor`` / ``fit`` with opaque shape errors. These tests pin the
request-boundary 422 contract.
"""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.app import create_app
from api.models.training import InlineDataset
from api.settings import Settings

pytestmark = pytest.mark.unit


def _train_pair(n: int = 2):
    return ([[0.0, float(i)] for i in range(n)], [[1.0, 0.0] for _ in range(n)])


class TestInlineDatasetModel:
    """Direct model validation — no ASGI stack needed."""

    def test_aligned_train_only_is_accepted(self):
        train_x, train_y = _train_pair(3)
        ds = InlineDataset(train_x=train_x, train_y=train_y)
        assert len(ds.train_x) == 3
        assert ds.val_x is None and ds.val_y is None

    def test_aligned_train_and_val_is_accepted(self):
        train_x, train_y = _train_pair(2)
        ds = InlineDataset(train_x=train_x, train_y=train_y, val_x=[[0.5, 0.5]], val_y=[[0.0, 1.0]])
        assert len(ds.val_x) == 1

    def test_train_length_mismatch_is_rejected(self):
        with pytest.raises(ValidationError, match="train_x/train_y length mismatch"):
            InlineDataset(train_x=[[0.0, 0.0], [1.0, 1.0]], train_y=[[1.0, 0.0]])

    def test_val_x_without_val_y_is_rejected(self):
        train_x, train_y = _train_pair(1)
        with pytest.raises(ValidationError, match="missing val_y"):
            InlineDataset(train_x=train_x, train_y=train_y, val_x=[[0.1, 0.2]])

    def test_val_y_without_val_x_is_rejected(self):
        train_x, train_y = _train_pair(1)
        with pytest.raises(ValidationError, match="missing val_x"):
            InlineDataset(train_x=train_x, train_y=train_y, val_y=[[1.0, 0.0]])

    def test_val_length_mismatch_is_rejected(self):
        train_x, train_y = _train_pair(1)
        with pytest.raises(ValidationError, match="val_x/val_y length mismatch"):
            InlineDataset(
                train_x=train_x,
                train_y=train_y,
                val_x=[[0.1, 0.2], [0.3, 0.4]],
                val_y=[[1.0, 0.0]],
            )


class TestInlineDatasetStartRoute:
    """HTTP boundary: mismatched inline_data must 422 before training starts."""

    @pytest.fixture
    def client(self):
        settings = Settings(auto_start=False)
        app = create_app(settings)
        with TestClient(app) as c:
            yield c

    def test_mismatched_train_lengths_return_422(self, client):
        response = client.post(
            "/v1/training/start",
            json={
                "inline_data": {
                    "train_x": [[0.0, 0.0], [1.0, 1.0]],
                    "train_y": [[1.0, 0.0]],
                },
                "epochs": 1,
            },
        )
        assert response.status_code == 422
        assert "train_x/train_y length mismatch" in response.text

    def test_partial_validation_split_returns_422(self, client):
        response = client.post(
            "/v1/training/start",
            json={
                "inline_data": {
                    "train_x": [[0.0, 0.0]],
                    "train_y": [[1.0, 0.0]],
                    "val_x": [[0.5, 0.5]],
                },
                "epochs": 1,
            },
        )
        assert response.status_code == 422
        assert "missing val_y" in response.text
