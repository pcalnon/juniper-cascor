#!/usr/bin/env python
"""
Unit tests for JuniperDataClient.
"""
import io
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from juniper_data_client.client import JuniperDataClient


@pytest.mark.unit
class TestJuniperDataClientUrlNormalization:
    """Tests for URL normalization in JuniperDataClient."""

    def test_url_with_scheme_unchanged(self):
        """URL with scheme should keep scheme."""
        client = JuniperDataClient(base_url="http://localhost:8100")
        assert client.base_url == "http://localhost:8100"

    def test_url_without_scheme_adds_http(self):
        """URL without scheme should add http://."""
        client = JuniperDataClient(base_url="example.com/api")
        assert client.base_url == "http://example.com/api"

    def test_url_with_trailing_slash_removed(self):
        """Trailing slash should be removed."""
        client = JuniperDataClient(base_url="http://localhost:8100/")
        assert client.base_url == "http://localhost:8100"

    def test_url_with_v1_suffix_removed(self):
        """URL ending with /v1 should have /v1 removed."""
        client = JuniperDataClient(base_url="http://localhost:8100/v1")
        assert client.base_url == "http://localhost:8100"

    def test_url_with_trailing_slash_and_v1_removed(self):
        """URL with both trailing slash and /v1 should normalize."""
        client = JuniperDataClient(base_url="http://localhost:8100/v1/")
        assert client.base_url == "http://localhost:8100"

    def test_url_with_https_scheme(self):
        """HTTPS scheme should be preserved."""
        client = JuniperDataClient(base_url="https://api.example.com:8100")
        assert client.base_url == "https://api.example.com:8100"

    def test_url_with_whitespace_stripped(self):
        """Whitespace should be stripped from URL."""
        client = JuniperDataClient(base_url="  http://localhost:8100  ")
        assert client.base_url == "http://localhost:8100"


@pytest.mark.unit
class TestJuniperDataClientCreateDataset:
    """Tests for create_dataset method."""

    def test_create_dataset_calls_post_with_correct_url(self):
        """create_dataset should POST to /v1/datasets."""
        client = JuniperDataClient(base_url="http://localhost:8100")

        with patch.object(client.session, "request") as mock_request:
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.json.return_value = {"dataset_id": "test-123"}
            mock_request.return_value = mock_response

            result = client.create_dataset(
                generator="spiral",
                params={"n_spirals": 2, "n_points": 100},
            )

            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "http://localhost:8100/v1/datasets"

    def test_create_dataset_sends_correct_json_body(self):
        """create_dataset should send correct JSON payload."""
        client = JuniperDataClient(base_url="http://localhost:8100")

        with patch.object(client.session, "request") as mock_request:
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.json.return_value = {"dataset_id": "test-123"}
            mock_request.return_value = mock_response

            params = {"n_spirals": 3, "n_points": 50}
            client.create_dataset(generator="spiral", params=params, persist=False)

            call_kwargs = mock_request.call_args[1]
            assert call_kwargs["json"] == {
                "generator": "spiral",
                "params": {"n_spirals": 3, "n_points": 50},
                "persist": False,
            }

    def test_create_dataset_returns_parsed_json(self):
        """create_dataset should return parsed JSON response."""
        client = JuniperDataClient(base_url="http://localhost:8100")

        with patch.object(client.session, "request") as mock_request:
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.json.return_value = {
                "dataset_id": "abc-123",
                "status": "created",
            }
            mock_request.return_value = mock_response

            result = client.create_dataset(generator="spiral", params={})

            assert result == {"dataset_id": "abc-123", "status": "created"}


@pytest.mark.unit
class TestJuniperDataClientDownloadArtifactNpz:
    """Tests for download_artifact_npz method."""

    def test_download_artifact_npz_calls_correct_url(self):
        """download_artifact_npz should GET from correct endpoint."""
        client = JuniperDataClient(base_url="http://localhost:8100")

        npz_buffer = io.BytesIO()
        np.savez(npz_buffer, X_train=np.array([1, 2]), y_train=np.array([0, 1]))
        npz_buffer.seek(0)

        with patch.object(client.session, "request") as mock_request:
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.content = npz_buffer.read()
            mock_request.return_value = mock_response

            client.download_artifact_npz("dataset-456")

            call_args = mock_request.call_args
            assert call_args[0][0] == "GET"
            assert call_args[0][1] == "http://localhost:8100/v1/datasets/dataset-456/artifact"

    def test_download_artifact_npz_parses_npz_correctly(self):
        """download_artifact_npz should parse NPZ and return dict of arrays."""
        client = JuniperDataClient(base_url="http://localhost:8100")

        expected_x_train = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected_y_train = np.array([[0.0], [1.0]])
        npz_buffer = io.BytesIO()
        np.savez(npz_buffer, X_train=expected_x_train, y_train=expected_y_train)
        npz_buffer.seek(0)

        with patch.object(client.session, "request") as mock_request:
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.content = npz_buffer.read()
            mock_request.return_value = mock_response

            result = client.download_artifact_npz("dataset-789")

            assert "X_train" in result
            assert "y_train" in result
            np.testing.assert_array_equal(result["X_train"], expected_x_train)
            np.testing.assert_array_equal(result["y_train"], expected_y_train)


@pytest.mark.unit
class TestJuniperDataClientErrorHandling:
    """Tests for error handling in JuniperDataClient."""

    def test_raises_http_error_on_non_2xx_status(self):
        """Should raise HTTPError on non-2xx response."""
        import requests

        client = JuniperDataClient(base_url="http://localhost:8100")

        with patch.object(client.session, "request") as mock_request:
            mock_response = MagicMock()
            mock_response.ok = False
            mock_response.status_code = 500
            mock_response.reason = "Internal Server Error"
            mock_response.text = "Something went wrong"
            mock_request.return_value = mock_response

            with pytest.raises(requests.HTTPError) as exc_info:
                client.create_dataset(generator="spiral", params={})

            assert "500" in str(exc_info.value)
            assert "Internal Server Error" in str(exc_info.value)

    def test_raises_http_error_on_404(self):
        """Should raise HTTPError on 404 response."""
        import requests

        client = JuniperDataClient(base_url="http://localhost:8100")

        with patch.object(client.session, "request") as mock_request:
            mock_response = MagicMock()
            mock_response.ok = False
            mock_response.status_code = 404
            mock_response.reason = "Not Found"
            mock_response.text = "Dataset not found"
            mock_request.return_value = mock_response

            with pytest.raises(requests.HTTPError) as exc_info:
                client.download_artifact_npz("nonexistent-id")

            assert "404" in str(exc_info.value)

    def test_uses_default_timeout(self):
        """Should use default timeout when not specified."""
        client = JuniperDataClient(base_url="http://localhost:8100")
        assert client.timeout == JuniperDataClient.DEFAULT_TIMEOUT

    def test_uses_custom_timeout(self):
        """Should use custom timeout when specified."""
        client = JuniperDataClient(base_url="http://localhost:8100", timeout=60)
        assert client.timeout == 60

    def test_timeout_passed_to_request(self):
        """Timeout should be passed to requests."""
        client = JuniperDataClient(base_url="http://localhost:8100", timeout=45)

        with patch.object(client.session, "request") as mock_request:
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.json.return_value = {}
            mock_request.return_value = mock_response

            client.create_dataset(generator="spiral", params={})

            call_kwargs = mock_request.call_args[1]
            assert call_kwargs["timeout"] == 45
