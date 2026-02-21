#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     data_provider.py
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
#    Data provider for spiral datasets with JuniperData service integration.
#    Provides spiral datasets as PyTorch tensors, fetching from JuniperData when available.
#
#####################################################################################################################################################################################################

import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from juniper_data_client import JuniperDataClient, JuniperDataConnectionError

logger = logging.getLogger(__name__)

SpiralDatasetTuple = Tuple[
    Tuple[torch.Tensor, torch.Tensor],  # (x_train, y_train)
    Tuple[torch.Tensor, torch.Tensor],  # (x_test, y_test)
    Tuple[torch.Tensor, torch.Tensor],  # (x_full, y_full)
]


class SpiralDataProviderError(Exception):
    """Exception raised when spiral data provider encounters an error."""

    pass


class SpiralDataProvider:
    """
    Provides spiral datasets using JuniperData service.

    Fetches spiral datasets from the JuniperData API and converts them
    to PyTorch tensors matching the format expected by SpiralProblem.
    """

    def __init__(self, juniper_data_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the spiral data provider.

        Args:
            juniper_data_url: Optional URL override for JuniperData service.  If not provided, reads from JUNIPER_DATA_URL environment variable.
            api_key: Optional API key for JuniperData authentication. If not provided, JuniperDataClient reads from JUNIPER_DATA_API_KEY env var.
        """
        self._juniper_data_url = juniper_data_url or os.environ.get("JUNIPER_DATA_URL")
        self._api_key = api_key
        self._client: Optional[JuniperDataClient] = None

    @property
    def use_juniper_data(self) -> bool:
        """Check if JuniperData service is configured."""
        return bool(self._juniper_data_url)

    def validate_configuration(self) -> None:
        """
        Validate that the data provider is properly configured.

        Checks that the JuniperData URL is set and structurally valid.
        Optionally performs a health check, logging a warning if the service
        is unreachable (but does not raise).

        Raises:
            SpiralDataProviderError: If the JuniperData URL is not configured.
            ValueError: If the URL is structurally invalid (via JuniperDataClient).
        """
        if not self._juniper_data_url:
            raise SpiralDataProviderError("JuniperData URL not configured. Set JUNIPER_DATA_URL environment variable.")

        client = self._get_client()

        try:
            client.health_check()
        except JuniperDataConnectionError:
            logger.warning("JuniperData service at %s is not reachable; requests may fail.", self._juniper_data_url)

    def _get_client(self) -> JuniperDataClient:
        """Get or create JuniperData client instance."""
        if self._client is None:
            if not self._juniper_data_url:
                raise SpiralDataProviderError("JuniperData URL not configured. Set JUNIPER_DATA_URL environment variable.")
            self._client = JuniperDataClient(base_url=self._juniper_data_url, api_key=self._api_key)
        return self._client

    def get_spiral_dataset(
        self,
        n_spirals: int,
        n_points: int,
        n_rotations: float,
        noise_level: float,
        clockwise: bool,
        train_ratio: float,
        test_ratio: float,
        seed: Optional[int] = None,
        algorithm: Optional[str] = None,
    ) -> SpiralDatasetTuple:
        """
        Fetch spiral dataset from JuniperData service.

        Args:
            n_spirals: Number of spiral arms to generate.
            n_points: Number of points per spiral arm.
            n_rotations: Number of full rotations for each spiral.
            noise_level: Noise level applied to point positions.
            clockwise: Whether spirals rotate clockwise.
            train_ratio: Fraction of data for training set.
            test_ratio: Fraction of data for test set.
            seed: Random seed for reproducibility.
            algorithm: Generation algorithm - 'modern' (default) or 'legacy_cascor' for backward compatibility.

        Returns:
            Tuple containing:
                - (x_train, y_train): Training set features and targets as torch tensors.
                - (x_test, y_test): Test set features and targets as torch tensors.
                - (x_full, y_full): Full dataset features and targets as torch tensors.

        Raises:
            SpiralDataProviderError: If JuniperData service is unreachable or returns an error.
        """
        if not self.use_juniper_data:
            raise SpiralDataProviderError("JuniperData URL not configured. Set JUNIPER_DATA_URL environment variable.")

        params = {
            "n_spirals": n_spirals,
            "n_points_per_spiral": n_points,
            "n_rotations": n_rotations,
            "noise": noise_level,
            "clockwise": clockwise,
            "train_ratio": train_ratio,
            "test_ratio": test_ratio,
        }
        if seed is not None:
            params["seed"] = seed
        if algorithm is not None:
            params["algorithm"] = algorithm

        try:
            return self._build_spiral_dataset(params)
        except Exception as e:
            raise SpiralDataProviderError(f"Failed to fetch spiral dataset from JuniperData service: {e}") from e

    def _build_spiral_dataset(self, params):
        client = self._get_client()
        logger.debug(f"Creating spiral dataset with params: {params}")

        response = client.create_dataset(generator="spiral", params=params)
        dataset_id = response["dataset_id"]
        logger.debug(f"Created dataset with ID: {dataset_id}")

        arrays = client.download_artifact_npz(dataset_id)
        logger.debug(f"Downloaded artifact with arrays: {list(arrays.keys())}")

        return self._convert_arrays_to_tensors(arrays)

    def _convert_arrays_to_tensors(self, arrays: Dict[str, np.ndarray]) -> SpiralDatasetTuple:
        """
        Convert numpy arrays from NPZ to PyTorch tensors.
        Validates the NPZ artifact meets the expected data contract.

        Args:
            arrays: Dictionary of numpy arrays from NPZ file.

        Returns:
            Tuple of tensor pairs for train, test, and full datasets.

        Raises:
            SpiralDataProviderError: If NPZ artifact does not meet the expected contract.
        """
        required_keys = {"X_train", "y_train", "X_test", "y_test", "X_full", "y_full"}
        missing_keys = required_keys - set(arrays.keys())
        if missing_keys:
            raise SpiralDataProviderError(f"NPZ artifact missing required keys: {sorted(missing_keys)}. " f"Expected: {sorted(required_keys)}, got: {sorted(arrays.keys())}")

        for key in required_keys:
            arr = arrays[key]
            if arr.ndim != 2:
                raise SpiralDataProviderError(f"NPZ array '{key}' has {arr.ndim} dimensions, expected 2. Shape: {arr.shape}")

        for key in ["X_train", "X_test", "X_full"]:
            if arrays[key].shape[1] != 2:
                raise SpiralDataProviderError(f"Feature array '{key}' has {arrays[key].shape[1]} columns, expected 2 (x, y coordinates). " f"Shape: {arrays[key].shape}")

        x_train = torch.tensor(arrays["X_train"], dtype=torch.float32)
        y_train = torch.tensor(arrays["y_train"], dtype=torch.float32)
        x_test = torch.tensor(arrays["X_test"], dtype=torch.float32)
        y_test = torch.tensor(arrays["y_test"], dtype=torch.float32)
        x_full = torch.tensor(arrays["X_full"], dtype=torch.float32)
        y_full = torch.tensor(arrays["y_full"], dtype=torch.float32)

        return (x_train, y_train), (x_test, y_test), (x_full, y_full)
