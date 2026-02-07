#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     client.py
# Author:        Paul Calnon
# Version:       0.3.2 (0.7.3)
#
# Date Created:  2026-01-29
# Last Modified: 2026-01-29
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    REST API client for JuniperData service integration.
#    Provides dataset creation and artifact download functionality for Cascor.
#
#####################################################################################################################################################################################################

import io
import logging
import os
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import numpy as np
import requests

logger = logging.getLogger(__name__)


class JuniperDataClient:
    """
    Client for interacting with the JuniperData REST API.
    Provides methods for dataset creation and artifact retrieval.
    """

    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 1.0
    _RETRYABLE_STATUS_CODES = (502, 503, 504)

    def __init__(self, base_url: str = "http://localhost:8100", timeout: int = DEFAULT_TIMEOUT, api_key: Optional[str] = None):
        """
        Initialize the JuniperData client.

        Args:
            base_url: Base URL for the JuniperData API (default: http://localhost:8100)
            timeout: Request timeout in seconds (default: 30)
            api_key: API key for authentication. If not provided, reads from JUNIPER_DATA_API_KEY env var.
        """
        self.base_url = self._normalize_url(base_url)
        self.validate_url(self.base_url)
        self.timeout = timeout
        self.session = requests.Session()

        resolved_api_key = api_key or os.environ.get("JUNIPER_DATA_API_KEY")
        if resolved_api_key:
            self.session.headers["X-API-Key"] = resolved_api_key

    def _normalize_url(self, url: str) -> str:
        """
        Normalize the base URL for consistent API calls.

        Args:
            url: Raw URL string to normalize

        Returns:
            Normalized URL with scheme, no trailing slash, no /v1 suffix
        """
        url = url.strip()

        parsed = urlparse(url)
        if not parsed.scheme:
            url = f"http://{url}"
            parsed = urlparse(url)

        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        normalized = normalized.rstrip("/")

        if normalized.endswith("/v1"):
            normalized = normalized[:-3]

        return normalized

    @staticmethod
    def validate_url(url: str) -> None:
        """
        Validate that a URL has a valid scheme and hostname.

        Args:
            url: URL string to validate.

        Raises:
            ValueError: If the URL scheme is not http/https or the hostname is missing.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid URL scheme '{parsed.scheme}': only 'http' and 'https' are supported. URL: {url}")
        if not parsed.hostname:
            raise ValueError(f"URL is missing a hostname. URL: {url}")

    def health_check(self) -> bool:
        """
        Check if the JuniperData service is reachable.

        Makes a GET request to /v1/health with a short timeout.

        Returns:
            True if the service responds with a 2xx status, False otherwise.
        """
        url = f"{self.base_url}/v1/health"
        try:
            response = self.session.get(url, timeout=5)
            healthy = response.ok
            logger.debug("Health check %s: %s %s", url, response.status_code, "OK" if healthy else "FAIL")
            return healthy
        except Exception as e:
            logger.debug("Health check %s failed: %s", url, e)
            return False

    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make an HTTP request with error handling and retry logic.

        Retries on transient errors (HTTP 502/503/504, ConnectionError, Timeout)
        with exponential backoff. Non-retryable errors (4xx) fail immediately.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments passed to requests

        Returns:
            Response object

        Raises:
            requests.HTTPError: On non-2xx response status after retries exhausted
            requests.ConnectionError: On connection failure after retries exhausted
            requests.Timeout: On timeout after retries exhausted
        """
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault("timeout", self.timeout)

        last_exception = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                response = self.session.request(method, url, **kwargs)

                if response.ok:
                    return response

                if response.status_code in self._RETRYABLE_STATUS_CODES:
                    last_exception = requests.HTTPError(
                        f"Request failed: {response.status_code} {response.reason} - {response.text}",
                        response=response,
                    )
                    if attempt < self.MAX_RETRIES:
                        delay = self.RETRY_BACKOFF_BASE * (2**attempt)
                        reason = f"{response.status_code} {response.reason}"
                        logger.warning(f"Request to {url} failed ({reason}), retrying in {delay:.1f}s (attempt {attempt + 1}/{self.MAX_RETRIES})")
                        time.sleep(delay)
                        continue
                    raise last_exception

                raise requests.HTTPError(
                    f"Request failed: {response.status_code} {response.reason} - {response.text}",
                    response=response,
                )

            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exception = exc
                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_BACKOFF_BASE * (2**attempt)
                    reason = type(exc).__name__
                    logger.warning(f"Request to {url} failed ({reason}), retrying in {delay:.1f}s (attempt {attempt + 1}/{self.MAX_RETRIES})")
                    time.sleep(delay)
                    continue
                raise

    def create_dataset(self, generator: str, params: Dict[str, Any], persist: bool = True) -> Dict[str, Any]:
        """
        Create a new dataset via the JuniperData API.

        Args:
            generator: Name of the dataset generator to use
            params: Parameters to pass to the generator
            persist: Whether to persist the dataset (default: True)

        Returns:
            Parsed JSON response from the API containing dataset metadata
        """
        payload = {
            "generator": generator,
            "params": params,
            "persist": persist,
        }

        response = self._request("POST", "/v1/datasets", json=payload)
        return response.json()

    def download_artifact_npz(self, dataset_id: str) -> Dict[str, np.ndarray]:
        """
        Download and load an NPZ artifact for a dataset.

        Args:
            dataset_id: ID of the dataset whose artifact to download

        Returns:
            Dictionary mapping array names to numpy arrays
        """
        response = self._request("GET", f"/v1/datasets/{dataset_id}/artifact")

        npz_file = np.load(io.BytesIO(response.content))
        return {key: npz_file[key] for key in npz_file.files}
