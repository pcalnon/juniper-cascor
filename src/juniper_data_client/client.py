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
from typing import Any, Dict
from urllib.parse import urlparse

import numpy as np
import requests


class JuniperDataClient:
    """
    Client for interacting with the JuniperData REST API.
    Provides methods for dataset creation and artifact retrieval.
    """

    DEFAULT_TIMEOUT = 30

    def __init__(self, base_url: str = "http://localhost:8100", timeout: int = DEFAULT_TIMEOUT):
        """
        Initialize the JuniperData client.

        Args:
            base_url: Base URL for the JuniperData API (default: http://localhost:8100)
            timeout: Request timeout in seconds (default: 30)
        """
        self.base_url = self._normalize_url(base_url)
        self.timeout = timeout
        self.session = requests.Session()

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

    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make an HTTP request with error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments passed to requests

        Returns:
            Response object

        Raises:
            requests.HTTPError: On non-2xx response status
        """
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault("timeout", self.timeout)

        response = self.session.request(method, url, **kwargs)

        if not response.ok:
            raise requests.HTTPError(
                f"Request failed: {response.status_code} {response.reason} - {response.text}",
                response=response,
            )

        return response

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
