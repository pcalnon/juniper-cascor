#!/usr/bin/env python
"""
Smoke tests verifying the shared juniper-data-client package is installed
and provides the API surface that JuniperCascor depends on.

The client library's full test suite (41+ tests, 90% coverage) lives in
the juniper-data-client repository. These tests only validate that the
package is importable and the interfaces CasCor uses are present.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# juniper-data-client requires Python >=3.12
_skip_reason = "juniper-data-client requires Python >=3.12"
_requires_312 = pytest.mark.skipif(sys.version_info < (3, 12), reason=_skip_reason)


@_requires_312
@pytest.mark.unit
class TestSharedPackageInstalled:
    """Verify the shared juniper-data-client package is installed and importable."""

    def test_import_juniper_data_client(self):
        """Package should be importable."""
        import juniper_data_client

        assert hasattr(juniper_data_client, "__version__")

    def test_import_client_class(self):
        """JuniperDataClient should be importable from package."""
        from juniper_data_client import JuniperDataClient

        assert callable(JuniperDataClient)

    def test_import_exceptions(self):
        """All exception classes should be importable."""
        from juniper_data_client import JuniperDataClientError, JuniperDataConfigurationError, JuniperDataConnectionError, JuniperDataNotFoundError, JuniperDataTimeoutError, JuniperDataValidationError

        assert issubclass(JuniperDataConnectionError, JuniperDataClientError)
        assert issubclass(JuniperDataTimeoutError, JuniperDataClientError)
        assert issubclass(JuniperDataNotFoundError, JuniperDataClientError)
        assert issubclass(JuniperDataValidationError, JuniperDataClientError)
        assert issubclass(JuniperDataConfigurationError, JuniperDataClientError)

    def test_version_is_shared_package(self):
        """Should be using the shared package, not a local fallback."""
        import juniper_data_client

        assert not juniper_data_client.__version__.endswith("-local"), f"Using local fallback: {juniper_data_client.__version__}"


@_requires_312
@pytest.mark.unit
class TestClientApiSurface:
    """Verify the shared package exposes the API surface CasCor depends on."""

    def test_client_constructor_accepts_base_url(self):
        """Constructor should accept base_url parameter."""
        from juniper_data_client import JuniperDataClient

        client = JuniperDataClient(base_url="http://localhost:8100")
        assert client.base_url == "http://localhost:8100"

    def test_client_constructor_accepts_api_key(self):
        """Constructor should accept api_key parameter."""
        from juniper_data_client import JuniperDataClient

        client = JuniperDataClient(base_url="http://localhost:8100", api_key="test-key")
        assert client.session.headers["X-API-Key"] == "test-key"

    def test_client_has_create_dataset_method(self):
        """Client should have create_dataset method."""
        from juniper_data_client import JuniperDataClient

        client = JuniperDataClient(base_url="http://localhost:8100")
        assert callable(getattr(client, "create_dataset", None))

    def test_client_has_download_artifact_npz_method(self):
        """Client should have download_artifact_npz method."""
        from juniper_data_client import JuniperDataClient

        client = JuniperDataClient(base_url="http://localhost:8100")
        assert callable(getattr(client, "download_artifact_npz", None))

    def test_client_has_health_check_method(self):
        """Client should have health_check method."""
        from juniper_data_client import JuniperDataClient

        client = JuniperDataClient(base_url="http://localhost:8100")
        assert callable(getattr(client, "health_check", None))

    def test_client_has_context_manager(self):
        """Client should support context manager protocol."""
        from juniper_data_client import JuniperDataClient

        client = JuniperDataClient(base_url="http://localhost:8100")
        assert hasattr(client, "__enter__")
        assert hasattr(client, "__exit__")

    def test_url_normalization_strips_trailing_slash(self):
        """URL normalization should strip trailing slash."""
        from juniper_data_client import JuniperDataClient

        client = JuniperDataClient(base_url="http://localhost:8100/")
        assert client.base_url == "http://localhost:8100"

    def test_url_normalization_strips_v1_suffix(self):
        """URL normalization should strip /v1 suffix."""
        from juniper_data_client import JuniperDataClient

        client = JuniperDataClient(base_url="http://localhost:8100/v1")
        assert client.base_url == "http://localhost:8100"
