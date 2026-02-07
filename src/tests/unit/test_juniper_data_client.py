#!/usr/bin/env python
"""
Unit tests for JuniperDataClient.
"""
import io
import os
import sys
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import requests

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
            assert result
            assert isinstance(JuniperDataClient, result)

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
            self._get_artifact_from_url(npz_buffer, mock_request, client)

    def _get_artifact_from_url(self, npz_buffer, mock_request, client) -> None:
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


@pytest.mark.unit
class TestJuniperDataClientAuthentication:
    """Tests for API key authentication."""

    def test_no_api_key_by_default(self):
        """No API key header should be set when not configured."""
        env_without_key = {k: v for k, v in os.environ.items() if k != "JUNIPER_DATA_API_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True):
            client = JuniperDataClient(base_url="http://localhost:8100")
            assert "X-API-Key" not in client.session.headers

    def test_api_key_from_constructor(self):
        """API key from constructor should be set in session headers."""
        client = JuniperDataClient(base_url="http://localhost:8100", api_key="test-key-123")
        assert client.session.headers["X-API-Key"] == "test-key-123"

    def test_api_key_from_env_var(self):
        """API key from JUNIPER_DATA_API_KEY env var should be set."""
        with patch.dict(os.environ, {"JUNIPER_DATA_API_KEY": "env-key-456"}):
            client = JuniperDataClient(base_url="http://localhost:8100")
            assert client.session.headers["X-API-Key"] == "env-key-456"

    def test_explicit_api_key_overrides_env_var(self):
        """Explicit API key should override env var."""
        with patch.dict(os.environ, {"JUNIPER_DATA_API_KEY": "env-key"}):
            client = JuniperDataClient(base_url="http://localhost:8100", api_key="explicit-key")
            assert client.session.headers["X-API-Key"] == "explicit-key"

    def test_api_key_sent_in_requests(self):
        """API key should be included in HTTP requests."""
        client = JuniperDataClient(base_url="http://localhost:8100", api_key="request-key")

        with patch.object(client.session, "request") as mock_request:
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.json.return_value = {"dataset_id": "test-123"}
            mock_request.return_value = mock_response

            client.create_dataset(generator="spiral", params={})

            assert client.session.headers["X-API-Key"] == "request-key"


@pytest.mark.unit
class TestJuniperDataClientValidation:
    """Tests for URL validation and health check in JuniperDataClient."""

    def test_rejects_url_without_scheme(self):
        """Should raise ValueError for URL without a valid scheme."""
        with pytest.raises(ValueError, match="missing a hostname"):
            JuniperDataClient("://no-scheme")

    def test_rejects_url_without_host(self):
        """Should raise ValueError for URL without a hostname."""
        with pytest.raises(ValueError, match="missing a hostname"):
            JuniperDataClient("http://")

    def test_accepts_valid_http_url(self):
        """Should accept a valid http URL without error."""
        client = JuniperDataClient("http://localhost:8100")
        assert client.base_url == "http://localhost:8100"

    def test_accepts_valid_https_url(self):
        """Should accept a valid https URL without error."""
        client = JuniperDataClient("https://api.example.com")
        assert client.base_url == "https://api.example.com"

    def test_rejects_ftp_scheme(self):
        """Should raise ValueError for unsupported ftp scheme."""
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            JuniperDataClient("ftp://server:21")

    def test_health_check_returns_true_on_success(self):
        """health_check should return True when service responds 200."""
        client = JuniperDataClient("http://localhost:8100")

        with patch.object(client.session, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            assert client.health_check() is True
            mock_get.assert_called_once_with("http://localhost:8100/v1/health", timeout=5)

    def test_health_check_returns_false_on_error(self):
        """health_check should return False on connection error."""
        client = JuniperDataClient("http://localhost:8100")

        with patch.object(client.session, "get") as mock_get:
            mock_get.side_effect = requests.ConnectionError("Connection refused")

            assert client.health_check() is False


@pytest.mark.unit
class TestJuniperDataClientRetry:
    """Tests for retry logic with exponential backoff in _request."""

    def _make_response(self, status_code, ok=None, reason="", text=""):
        """Helper to create a mock response."""
        resp = MagicMock()
        resp.status_code = status_code
        resp.ok = ok if ok is not None else (200 <= status_code < 300)
        resp.reason = reason
        resp.text = text
        resp.json.return_value = {}
        return resp

    @patch("juniper_data_client.client.time.sleep")
    def test_retries_on_502_status(self, mock_sleep):
        """Should retry on 502 Bad Gateway and succeed on third attempt."""
        client = JuniperDataClient(base_url="http://localhost:8100")

        resp_502 = self._make_response(502, reason="Bad Gateway")
        resp_200 = self._make_response(200)

        with patch.object(client.session, "request", side_effect=[resp_502, resp_502, resp_200]) as mock_req:
            result = client._request("GET", "/v1/test")
            assert result.status_code == 200
            assert mock_req.call_count == 3

    @patch("juniper_data_client.client.time.sleep")
    def test_retries_on_503_status(self, mock_sleep):
        """Should retry on 503 Service Unavailable and succeed on third attempt."""
        client = JuniperDataClient(base_url="http://localhost:8100")

        resp_503 = self._make_response(503, reason="Service Unavailable")
        resp_200 = self._make_response(200)

        with patch.object(client.session, "request", side_effect=[resp_503, resp_503, resp_200]) as mock_req:
            result = client._request("GET", "/v1/test")
            assert result.status_code == 200
            assert mock_req.call_count == 3

    @patch("juniper_data_client.client.time.sleep")
    def test_retries_on_connection_error(self, mock_sleep):
        """Should retry on ConnectionError and succeed on third attempt."""
        client = JuniperDataClient(base_url="http://localhost:8100")

        resp_200 = self._make_response(200)

        with patch.object(
            client.session,
            "request",
            side_effect=[requests.ConnectionError("refused"), requests.ConnectionError("refused"), resp_200],
        ) as mock_req:
            result = client._request("GET", "/v1/test")
            assert result.status_code == 200
            assert mock_req.call_count == 3

    @patch("juniper_data_client.client.time.sleep")
    def test_retries_on_timeout(self, mock_sleep):
        """Should retry on Timeout and succeed on third attempt."""
        client = JuniperDataClient(base_url="http://localhost:8100")

        resp_200 = self._make_response(200)

        with patch.object(
            client.session,
            "request",
            side_effect=[requests.Timeout("timed out"), requests.Timeout("timed out"), resp_200],
        ) as mock_req:
            result = client._request("GET", "/v1/test")
            assert result.status_code == 200
            assert mock_req.call_count == 3

    @patch("juniper_data_client.client.time.sleep")
    def test_no_retry_on_400(self, mock_sleep):
        """Should NOT retry on 400 Bad Request - fail immediately."""
        client = JuniperDataClient(base_url="http://localhost:8100")

        resp_400 = self._make_response(400, reason="Bad Request", text="Invalid params")

        with patch.object(client.session, "request", return_value=resp_400) as mock_req:
            with pytest.raises(requests.HTTPError) as exc_info:
                client._request("POST", "/v1/test")

            assert mock_req.call_count == 1
            assert "400" in str(exc_info.value)
            mock_sleep.assert_not_called()

    @patch("juniper_data_client.client.time.sleep")
    def test_no_retry_on_404(self, mock_sleep):
        """Should NOT retry on 404 Not Found - fail immediately."""
        client = JuniperDataClient(base_url="http://localhost:8100")

        resp_404 = self._make_response(404, reason="Not Found", text="Not found")

        with patch.object(client.session, "request", return_value=resp_404) as mock_req:
            with pytest.raises(requests.HTTPError):
                client._request("GET", "/v1/test")

            assert mock_req.call_count == 1
            mock_sleep.assert_not_called()

    @patch("juniper_data_client.client.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep):
        """Should raise HTTPError after exhausting MAX_RETRIES+1 total attempts."""
        client = JuniperDataClient(base_url="http://localhost:8100")

        resp_503 = self._make_response(503, reason="Service Unavailable", text="overloaded")

        with patch.object(
            client.session,
            "request",
            return_value=resp_503,
        ) as mock_req:
            with pytest.raises(requests.HTTPError) as exc_info:
                client._request("GET", "/v1/test")

            assert mock_req.call_count == client.MAX_RETRIES + 1
            assert "503" in str(exc_info.value)

    @patch("juniper_data_client.client.time.sleep")
    def test_backoff_delay_increases(self, mock_sleep):
        """Backoff delays should double each attempt: 1.0, 2.0, 4.0."""
        client = JuniperDataClient(base_url="http://localhost:8100")

        resp_503 = self._make_response(503, reason="Service Unavailable", text="overloaded")

        with patch.object(client.session, "request", return_value=resp_503):
            with pytest.raises(requests.HTTPError):
                client._request("GET", "/v1/test")

            assert mock_sleep.call_count == client.MAX_RETRIES
            expected_delays = [client.RETRY_BACKOFF_BASE * (2**i) for i in range(client.MAX_RETRIES)]
            actual_delays = [c[0][0] for c in mock_sleep.call_args_list]
            assert actual_delays == expected_delays
