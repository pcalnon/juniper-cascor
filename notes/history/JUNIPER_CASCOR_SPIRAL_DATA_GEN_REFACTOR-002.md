# Juniper Cascor Spiral Data Generator Refactoring Plan

**Document**: JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR.md  
**Created**: 2026-01-29  
**Last Updated**: 2026-01-29  
**Version**: 1.0.0  
**Status**: Draft - Awaiting Review  
**Author**: Development Team

---

## Executive Summary

This document provides a comprehensive plan to extract the Spiral Problem Data Generator code from Juniper Cascor into a new standalone application called **Juniper Data**. The new application will provide a REST API for dataset generation and retrieval, enabling clean separation of concerns between data generation and neural network training.

### Key Objectives

1. **Decouple data generation** from neural network training in Juniper Cascor
2. **Create Juniper Data** as a standalone microservice with REST API
3. **Enable extensibility** for multiple dataset sources and generators
4. **Maintain backward compatibility** with existing Cascor functionality during migration

### Estimated Effort

| Phase | Description | Effort | Priority |
|-------|-------------|--------|----------|
| Phase 0 | Contract & Interface Design | S (1-2 days) | P0 |
| Phase 1 | Extract Generator Library | M (2-3 days) | P0 |
| Phase 2 | Build REST API | M (2-3 days) | P0 |
| Phase 3 | Storage & Caching | M (2-3 days) | P1 |
| Phase 4 | Cascor Integration | M (2-3 days) | P0 |
| Phase 5 | Extended Data Sources | L (1-2 weeks) | P2 |

**Total Initial Delivery (Phases 0-4)**: 1-2 weeks  
**Full Feature Set (All Phases)**: 3-4 weeks

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Dependency Mapping](#2-dependency-mapping)
3. [Phase 0: Contract & Interface Design](#3-phase-0-contract--interface-design)
4. [Phase 1: Extract Generator Library](#4-phase-1-extract-generator-library)
5. [Phase 2: Build REST API](#5-phase-2-build-rest-api)
6. [Phase 3: Storage & Caching](#6-phase-3-storage--caching)
7. [Phase 4: Cascor Integration](#7-phase-4-cascor-integration)
8. [Phase 5: Extended Data Sources](#8-phase-5-extended-data-sources)
9. [Environment & Configuration Management](#9-environment--configuration-management)
10. [Risk Assessment & Mitigation](#10-risk-assessment--mitigation)
11. [Testing Strategy](#11-testing-strategy)
12. [Implementation Checklist](#12-implementation-checklist)

---

## 1. Current State Analysis

### 1.1 Spiral Problem Code Location

The spiral data generation code is currently located in:

| File | Purpose | Lines |
|------|---------|-------|
| `src/spiral_problem/spiral_problem.py` | Main SpiralProblem class with mixed data generation and training | ~1500 |
| `src/spiral_problem/check.py` | Alternative/duplicate implementation | ~400 |
| `src/spiral_problem/__init__.py` | Package initialization | ~20 |
| `src/cascor_constants/constants_problem/constants_problem.py` | Spiral problem constants | ~880 |
| `src/tests/unit/test_data/generators.py` | Test data generators (SpiralDataGenerator) | ~460 |
| `src/tests/conftest.py` | Test fixtures with spiral data generation | ~280 |

### 1.2 Data Generation Methods to Extract

The following methods from `SpiralProblem` class handle pure data generation:

#### Core Generation Pipeline

| Method | Purpose | Extract Priority |
|--------|---------|------------------|
| `generate_n_spiral_dataset()` | Main entry point - orchestrates dataset generation | P0 |
| `generate_raw_spiral_coordinates()` | Generates raw x,y spiral coordinate arrays | P0 |
| `_generate_xy_coordinates()` | Generates x,y for single point with offset | P0 |
| `_make_coords()` | Creates coordinates using trigonometric functions | P0 |
| `_make_noise()` | Adds random noise to spiral points | P0 |

#### Dataset Processing Pipeline

| Method | Purpose | Extract Priority |
|--------|---------|------------------|
| `_create_input_features()` | Stacks x,y coordinates into input feature array | P0 |
| `_create_one_hot_targets()` | Creates one-hot encoded labels | P0 |
| `_create_spiral_dataset()` | Combines features and targets into dataset tuple | P0 |
| `_convert_to_tensors()` | Converts numpy arrays to PyTorch tensors | P1 (Keep in Cascor) |

#### Dataset Splitting Pipeline

| Method | Purpose | Extract Priority |
|--------|---------|------------------|
| `_shuffle_dataset()` | Randomly shuffles dataset | P0 |
| `_partition_dataset()` | Splits into train/test partitions | P0 |
| `_split_dataset()` | Generic dataset splitting utility | P0 |
| `_find_partition_index_end()` | Calculates partition boundaries | P0 |
| `_dataset_split_index_end()` | Index calculation utility | P0 |

### 1.3 Methods to Remain in Cascor

| Method | Purpose | Reason |
|--------|---------|--------|
| `__init__()` | Network initialization | Training-specific |
| `solve_n_spiral_problem()` | Training orchestration | Training-specific |
| `evaluate()` | Full evaluation pipeline | Training-specific |
| `_convert_to_tensors()` | PyTorch tensor conversion | Framework-specific |
| Network-related methods | CascadeCorrelationNetwork interaction | Training-specific |
| Plotting methods | Visualization | UI-specific |

### 1.4 Code References in Codebase

Spiral data generation is referenced in:

| File | Reference Type |
|------|----------------|
| `src/main.py` | SpiralProblem instantiation and evaluation |
| `src/tests/integration/test_spiral_problem.py` | Integration tests |
| `src/tests/unit/test_data/generators.py` | Test data generators |
| `src/tests/conftest.py` | Pytest fixtures |

---

## 2. Dependency Mapping

### 2.1 Current Dependencies (Spiral Problem)

```
SpiralProblem
├── Standard Library
│   ├── logging, logging.config
│   ├── os, sys
│   ├── random
│   ├── uuid
│   └── multiprocessing
├── Third-Party
│   ├── numpy (np) - Core dependency for array operations
│   ├── torch - Tensor operations, neural network
│   └── matplotlib.pyplot - Plotting
├── Local (Juniper Cascor)
│   ├── cascade_correlation.cascade_correlation.CascadeCorrelationNetwork
│   ├── cascade_correlation_config.CascadeCorrelationConfig
│   ├── cascor_constants.constants (55+ constants)
│   ├── log_config.log_config.LogConfig
│   └── log_config.logger.logger.Logger
```

### 2.2 Dependencies for Extracted Generator (Juniper Data)

```
SpiralGenerator (Juniper Data)
├── Standard Library
│   ├── logging (standard Python)
│   ├── hashlib (for dataset ID)
│   ├── json
│   └── typing
├── Third-Party
│   ├── numpy - REQUIRED (core math operations)
│   ├── pydantic - REQUIRED (schema validation)
│   ├── fastapi - REQUIRED (REST API)
│   └── uvicorn - REQUIRED (ASGI server)
├── Optional
│   └── torch - NOT REQUIRED (convert in Cascor client)
```

### 2.3 Dependency Decoupling Strategy

| Current Dependency | Action | New Location |
|-------------------|--------|--------------|
| `cascor_constants.constants_problem` | Copy & localize | `juniper_data.generators.spiral.defaults` |
| `LogConfig`, `Logger` | Replace | Standard Python `logging` |
| `CascadeCorrelationNetwork` | Remove | Not needed for data generation |
| `torch` | Remove | Client-side conversion in Cascor |
| `multiprocessing` | Remove | Not needed for data generation |
| `matplotlib` | Remove | Not needed for data generation |

### 2.4 Constants to Extract

From `constants_problem.py`, the following constants are needed:

```python
# Spiral Geometry
_SPIRAL_PROBLEM_NUM_SPIRALS = 2
_SPIRAL_PROBLEM_NUMBER_POINTS_PER_SPIRAL = 100
_SPIRAL_PROBLEM_NUM_ROTATIONS = 3
_SPIRAL_PROBLEM_CLOCKWISE = True
_SPIRAL_PROBLEM_DISTRIBUTION_FACTOR = 0.80
_SPIRAL_PROBLEM_DEFAULT_ORIGIN = 0.0
_SPIRAL_PROBLEM_DEFAULT_RADIUS = 10.0

# Noise & Randomness
_SPIRAL_PROBLEM_NOISE_FACTOR_DEFAULT = 0.25
_SPIRAL_PROBLEM_RANDOM_VALUE_SCALE = 0.1
_SPIRAL_PROBLEM_RANDOM_SEED = 42

# Dataset Splitting
_SPIRAL_PROBLEM_TRAIN_RATIO = 0.8
_SPIRAL_PROBLEM_TEST_RATIO = 0.2

# Input/Output Sizes
_SPIRAL_PROBLEM_INPUT_SIZE = 2
_SPIRAL_PROBLEM_OUTPUT_SIZE = 2  # Dynamic based on n_spirals
```

---

## 3. Phase 0: Contract & Interface Design

**Priority**: P0 - Critical Path  
**Effort**: S (1-2 days)  
**Goal**: Define stable interfaces before implementation

### Step 0.1: Define Dataset Contract Model

**Name**: Dataset Contract Schema Definition  
**Description**: Create the canonical data model for datasets returned by Juniper Data.

**Detailed Explanation**:

The Dataset Contract defines the structure of data exchanged between Juniper Data and its consumers. This must be framework-agnostic (no PyTorch tensors) to enable broad compatibility.

```python
# Dataset Contract Model
@dataclass
class DatasetContract:
    """Canonical dataset representation."""

    # Identity
    dataset_id: str                    # SHA256-based deterministic ID
    generator: str                     # "spiral", "xor", "circles", etc.
    version: str                       # Generator version for reproducibility

    # Metadata
    meta: DatasetMeta                  # Shape, dtype, params used

    # Data (numpy arrays serialized as lists for JSON)
    x_train: List[List[float]]         # Training features [n_train, input_size]
    y_train: List[List[float]]         # Training targets [n_train, output_size]
    x_test: List[List[float]]          # Test features [n_test, input_size]
    y_test: List[List[float]]          # Test targets [n_test, output_size]

    # Optional full dataset
    x_full: Optional[List[List[float]]] = None
    y_full: Optional[List[List[float]]] = None

@dataclass
class DatasetMeta:
    """Dataset metadata."""
    input_size: int
    output_size: int
    n_samples: int
    n_train: int
    n_test: int
    created_at: str                    # ISO 8601 timestamp
    params_hash: str                   # Hash of generation parameters
```

### Step 0.2: Define Request Parameter Schema

**Name**: Spiral Generator Request Schema  
**Description**: Define the parameter schema for spiral dataset generation requests.

**Detailed Explanation**:

```python
from pydantic import BaseModel, Field, validator

class SpiralGeneratorParams(BaseModel):
    """Request parameters for spiral dataset generation."""

    # Required parameters
    n_spirals: int = Field(default=2, ge=2, le=20,
        description="Number of spirals/classes to generate")
    n_points_per_spiral: int = Field(default=100, ge=10, le=10000,
        description="Points per spiral")

    # Geometry parameters
    n_rotations: float = Field(default=3.0, ge=0.5, le=10.0,
        description="Number of rotations per spiral")
    clockwise: bool = Field(default=True,
        description="Direction of spiral rotation")
    distribution: float = Field(default=0.8, ge=0.1, le=2.0,
        description="Point distribution factor")
    default_origin: float = Field(default=0.0,
        description="Spiral origin point")
    default_radius: float = Field(default=10.0, gt=0,
        description="Maximum spiral radius")

    # Noise parameters
    noise_level: float = Field(default=0.25, ge=0.0, le=1.0,
        description="Noise factor for point perturbation")
    random_seed: Optional[int] = Field(default=42,
        description="Random seed for reproducibility")

    # Dataset splitting
    train_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    shuffle: bool = Field(default=True,
        description="Shuffle dataset before splitting")

    @validator('test_ratio')
    def ratios_sum_to_one(cls, v, values):
        if 'train_ratio' in values:
            if not abs(values['train_ratio'] + v - 1.0) < 1e-6:
                raise ValueError('train_ratio + test_ratio must equal 1.0')
        return v
```

### Step 0.3: Define Dataset ID Generation

**Name**: Deterministic Dataset ID Algorithm  
**Description**: Implement deterministic dataset identification for caching and retrieval.

**Detailed Explanation**:

```python
import hashlib
import json

def generate_dataset_id(generator: str, params: dict, version: str) -> str:
    """
    Generate a deterministic dataset ID from parameters.

    The ID is a 16-character hex string derived from:
    - Generator name
    - Generator version
    - Canonical JSON of parameters

    This enables:
    - Cache lookup without regeneration
    - Reproducible dataset retrieval
    - Idempotent generation requests
    """
    # Create canonical representation
    canonical = {
        "generator": generator,
        "version": version,
        "params": dict(sorted(params.items()))
    }

    # Generate hash
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
    hash_bytes = hashlib.sha256(canonical_json.encode()).digest()

    return hash_bytes[:8].hex()  # 16-character hex ID
```

### Step 0.4: Define REST API Endpoints

**Name**: REST API Endpoint Specification  
**Description**: Define the complete REST API contract for Juniper Data.

**Detailed Explanation**:

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| `POST` | `/v1/datasets/generate` | Generate a new dataset | GenerateRequest | DatasetResponse |
| `GET` | `/v1/datasets/{dataset_id}` | Retrieve stored dataset | - | DatasetResponse |
| `GET` | `/v1/generators` | List available generators | - | GeneratorListResponse |
| `GET` | `/v1/generators/{name}/schema` | Get parameter schema | - | JSONSchema |
| `GET` | `/healthz` | Health check | - | HealthResponse |
| `DELETE` | `/v1/datasets/{dataset_id}` | Delete cached dataset | - | DeleteResponse |

```python
# POST /v1/datasets/generate
class GenerateRequest(BaseModel):
    generator: str = "spiral"
    params: SpiralGeneratorParams
    store: bool = True  # Whether to cache the dataset

# Response for dataset operations
class DatasetResponse(BaseModel):
    dataset_id: str
    generator: str
    version: str
    meta: DatasetMeta
    dataset: Optional[DatasetContract] = None  # None if store=True, fetch via GET
    links: Dict[str, str] = {}  # HATEOAS links
```

---

## 4. Phase 1: Extract Generator Library

**Priority**: P0 - Critical Path  
**Effort**: M (2-3 days)  
**Goal**: Create pure data generation library without Cascor dependencies

### Step 1.1: Create Juniper Data Project Structure

**Name**: Project Skeleton Creation  
**Description**: Establish the directory structure and configuration for Juniper Data.

**Detailed Explanation**:

```
JuniperData/
├── juniper_data/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── datasets.py      # Dataset endpoints
│   │   │   ├── generators.py    # Generator info endpoints
│   │   │   └── health.py        # Health check
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── requests.py      # Request models
│   │   │   └── responses.py     # Response models
│   │   └── main.py              # FastAPI app factory
│   ├── core/
│   │   ├── __init__.py
│   │   ├── models.py            # Domain models (Dataset, Split)
│   │   ├── dataset_id.py        # ID generation
│   │   └── exceptions.py        # Custom exceptions
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── base.py              # BaseGenerator ABC
│   │   ├── spiral/
│   │   │   ├── __init__.py
│   │   │   ├── generator.py     # SpiralGenerator
│   │   │   ├── defaults.py      # Default constants
│   │   │   └── params.py        # Parameter schema
│   │   ├── xor/
│   │   │   └── ...              # XOR generator (future)
│   │   └── circles/
│   │       └── ...              # Circles generator (future)
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── interface.py         # DatasetStore protocol
│   │   ├── memory.py            # InMemoryStore
│   │   └── filesystem.py        # LocalFileStore
│   └── config/
│       ├── __init__.py
│       └── settings.py          # Configuration via Pydantic
├── tests/
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── pyproject.toml
├── README.md
├── CHANGELOG.md
└── .env.example
```

### Step 1.2: Extract Spiral Generator Core Logic

**Name**: SpiralGenerator Class Implementation  
**Description**: Port the data generation methods from SpiralProblem to a clean SpiralGenerator class.

**Detailed Explanation**:

```python
# juniper_data/generators/spiral/generator.py

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

from ..base import BaseGenerator
from .params import SpiralGeneratorParams
from .defaults import SPIRAL_DEFAULTS

logger = logging.getLogger(__name__)


@dataclass
class SpiralDataset:
    """Spiral dataset container."""
    x: np.ndarray          # Features [n_samples, 2]
    y: np.ndarray          # One-hot targets [n_samples, n_spirals]
    labels: np.ndarray     # Integer labels [n_samples]
    params: dict           # Parameters used


class SpiralGenerator(BaseGenerator):
    """
    Generator for N-spiral classification datasets.

    This generator creates spiral datasets commonly used for testing
    neural network classification capabilities. Each spiral represents
    a distinct class, and points are distributed along the spiral arms
    with optional noise.

    Based on the classic 2-spiral problem (Lang & Witbrock, 1988)
    extended to N spirals.
    """

    NAME = "spiral"
    VERSION = "1.0.0"

    def generate(self, params: SpiralGeneratorParams) -> SpiralDataset:
        """
        Generate an N-spiral dataset with the given parameters.

        Args:
            params: Validated spiral generation parameters

        Returns:
            SpiralDataset containing features, targets, and metadata
        """
        logger.debug(f"Generating spiral dataset: {params.n_spirals} spirals, "
                    f"{params.n_points_per_spiral} points each")

        # Set random seed for reproducibility
        if params.random_seed is not None:
            np.random.seed(params.random_seed)

        # Generate raw spiral coordinates
        spiral_x, spiral_y = self._generate_raw_coordinates(
            n_spirals=params.n_spirals,
            n_points=params.n_points_per_spiral,
            n_rotations=params.n_rotations,
            clockwise=params.clockwise,
            distribution=params.distribution,
            radius=params.default_radius,
            origin=params.default_origin,
            noise=params.noise_level
        )

        # Create dataset arrays
        x = self._create_features(spiral_x, spiral_y)
        y, labels = self._create_targets(
            params.n_spirals,
            params.n_points_per_spiral
        )

        return SpiralDataset(
            x=x,
            y=y,
            labels=labels,
            params=params.dict()
        )

    def _generate_raw_coordinates(
        self,
        n_spirals: int,
        n_points: int,
        n_rotations: float,
        clockwise: bool,
        distribution: float,
        radius: float,
        origin: float,
        noise: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate raw x,y coordinates for all spirals."""

        spiral_x_coords = []
        spiral_y_coords = []

        direction = 1 if clockwise else -1
        angular_offset = 2 * np.pi / n_spirals

        for i in range(n_spirals):
            # Generate normalized distance along spiral
            t = np.linspace(0, 1, n_points)
            t_distributed = np.power(t, distribution)

            # Calculate angle for each point
            angle = t_distributed * n_rotations * 2 * np.pi + (angular_offset * i)

            # Calculate radius at each point
            r = origin + (radius - origin) * t_distributed

            # Generate coordinates with noise
            x = direction * r * np.cos(angle) + self._make_noise(n_points, noise)
            y = direction * r * np.sin(angle) + self._make_noise(n_points, noise)

            spiral_x_coords.append(x)
            spiral_y_coords.append(y)

        return np.array(spiral_x_coords), np.array(spiral_y_coords)

    def _make_noise(self, n_points: int, noise_factor: float) -> np.ndarray:
        """Generate random noise array."""
        return np.random.rand(n_points) * noise_factor

    def _create_features(
        self,
        spiral_x: np.ndarray,
        spiral_y: np.ndarray
    ) -> np.ndarray:
        """Stack spiral coordinates into feature array."""
        x_flat = np.hstack(spiral_x)
        y_flat = np.hstack(spiral_y)
        return np.vstack([x_flat, y_flat]).T.astype(np.float32)

    def _create_targets(
        self,
        n_spirals: int,
        n_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create one-hot encoded targets and integer labels."""
        total_points = n_spirals * n_points

        # One-hot encoding
        y = np.zeros((total_points, n_spirals), dtype=np.float32)
        for i in range(n_spirals):
            start_idx = i * n_points
            end_idx = (i + 1) * n_points
            y[start_idx:end_idx, i] = 1.0

        # Integer labels
        labels = np.repeat(np.arange(n_spirals), n_points).astype(np.int32)

        return y, labels
```

### Step 1.3: Implement Dataset Splitting Utilities

**Name**: Dataset Splitting Module  
**Description**: Create reusable dataset splitting functionality.

**Detailed Explanation**:

```python
# juniper_data/core/splitting.py

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class DatasetSplit:
    """Container for split dataset."""
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_full: np.ndarray
    y_full: np.ndarray
    indices: Dict[str, np.ndarray]  # Original indices for each split


def shuffle_dataset(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffle dataset arrays together maintaining correspondence.

    Returns:
        Tuple of (x_shuffled, y_shuffled, indices)
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = x.shape[0]
    indices = np.random.permutation(n_samples)

    return x[indices], y[indices], indices


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    test_ratio: float = 0.2,
    shuffle: bool = True,
    seed: int = None
) -> DatasetSplit:
    """
    Split dataset into train and test sets.

    Args:
        x: Feature array [n_samples, n_features]
        y: Target array [n_samples, n_outputs]
        train_ratio: Proportion for training set
        test_ratio: Proportion for test set
        shuffle: Whether to shuffle before splitting
        seed: Random seed for reproducibility

    Returns:
        DatasetSplit containing all splits
    """
    # Validate ratios
    if not np.isclose(train_ratio + test_ratio, 1.0):
        raise ValueError(
            f"train_ratio ({train_ratio}) + test_ratio ({test_ratio}) "
            f"must equal 1.0"
        )

    n_samples = x.shape[0]

    # Store original for full dataset
    x_full = x.copy()
    y_full = y.copy()

    # Shuffle if requested
    if shuffle:
        x, y, shuffle_indices = shuffle_dataset(x, y, seed)
    else:
        shuffle_indices = np.arange(n_samples)

    # Calculate split point
    train_end = int(n_samples * train_ratio)

    # Split arrays
    x_train = x[:train_end]
    y_train = y[:train_end]
    x_test = x[train_end:]
    y_test = y[train_end:]

    # Track indices
    indices = {
        'train': shuffle_indices[:train_end],
        'test': shuffle_indices[train_end:],
        'shuffle': shuffle_indices
    }

    return DatasetSplit(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_full=x_full,
        y_full=y_full,
        indices=indices
    )
```

### Step 1.4: Define Base Generator Interface

**Name**: BaseGenerator Abstract Class  
**Description**: Create abstract base class for all dataset generators.

**Detailed Explanation**:

```python
# juniper_data/generators/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Type
from pydantic import BaseModel


class BaseGenerator(ABC):
    """
    Abstract base class for dataset generators.

    All generators must implement:
    - NAME: Unique string identifier
    - VERSION: Semantic version string
    - generate(): Method to produce datasets
    - get_params_schema(): Returns Pydantic model class
    """

    NAME: str = NotImplemented
    VERSION: str = NotImplemented

    @abstractmethod
    def generate(self, params: BaseModel) -> Any:
        """
        Generate a dataset with the given parameters.

        Args:
            params: Validated parameters specific to this generator

        Returns:
            Dataset object (generator-specific type)
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_params_schema(cls) -> Type[BaseModel]:
        """Return the Pydantic model class for this generator's parameters."""
        raise NotImplementedError

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Return generator metadata."""
        return {
            "name": cls.NAME,
            "version": cls.VERSION,
            "params_schema": cls.get_params_schema().schema()
        }
```

### Step 1.5: Create Generator Registry

**Name**: Generator Registry Implementation  
**Description**: Implement a registry for managing available generators.

**Detailed Explanation**:

```python
# juniper_data/generators/__init__.py

from typing import Dict, Type, List
from .base import BaseGenerator
from .spiral.generator import SpiralGenerator


class GeneratorRegistry:
    """
    Registry for dataset generators.

    Provides centralized access to all available generators
    and their metadata.
    """

    _generators: Dict[str, Type[BaseGenerator]] = {}

    @classmethod
    def register(cls, generator_class: Type[BaseGenerator]) -> None:
        """Register a generator class."""
        cls._generators[generator_class.NAME] = generator_class

    @classmethod
    def get(cls, name: str) -> Type[BaseGenerator]:
        """Get a generator class by name."""
        if name not in cls._generators:
            raise KeyError(f"Unknown generator: {name}. "
                          f"Available: {list(cls._generators.keys())}")
        return cls._generators[name]

    @classmethod
    def list_generators(cls) -> List[Dict]:
        """List all registered generators with their metadata."""
        return [gen.get_info() for gen in cls._generators.values()]

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if a generator is registered."""
        return name in cls._generators


# Register built-in generators
GeneratorRegistry.register(SpiralGenerator)

# Future generators:
# GeneratorRegistry.register(XORGenerator)
# GeneratorRegistry.register(CirclesGenerator)
```

---

## 5. Phase 2: Build REST API

**Priority**: P0 - Critical Path  
**Effort**: M (2-3 days)  
**Goal**: Expose generators via FastAPI REST endpoints

### Step 2.1: FastAPI Application Factory

**Name**: Application Setup  
**Description**: Create the FastAPI application with proper configuration.

**Detailed Explanation**:

```python
# juniper_data/api/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from .routes import datasets, generators, health
from ..config.settings import Settings

logger = logging.getLogger(__name__)


def create_app(settings: Settings = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        settings: Application settings (uses defaults if None)

    Returns:
        Configured FastAPI application instance
    """
    settings = settings or Settings()

    app = FastAPI(
        title="Juniper Data",
        description="Dataset generation and management service for Juniper ML platform",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(generators.router, prefix="/v1", tags=["Generators"])
    app.include_router(datasets.router, prefix="/v1", tags=["Datasets"])

    # Startup/shutdown events
    @app.on_event("startup")
    async def startup():
        logger.info("Juniper Data service starting...")

    @app.on_event("shutdown")
    async def shutdown():
        logger.info("Juniper Data service shutting down...")

    return app


# Default app instance
app = create_app()
```

### Step 2.2: Dataset Generation Endpoint

**Name**: POST /v1/datasets/generate Implementation  
**Description**: Implement the primary dataset generation endpoint.

**Detailed Explanation**:

```python
# juniper_data/api/routes/datasets.py

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import logging

from ..schemas.requests import GenerateRequest
from ..schemas.responses import DatasetResponse, DatasetMeta
from ...generators import GeneratorRegistry
from ...core.dataset_id import generate_dataset_id
from ...core.splitting import split_dataset
from ...storage.interface import DatasetStore
from ...config.settings import get_store

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/datasets/generate", response_model=DatasetResponse)
async def generate_dataset(
    request: GenerateRequest,
    store: DatasetStore = Depends(get_store)
) -> DatasetResponse:
    """
    Generate a new dataset using the specified generator.

    If store=True, the dataset is cached and can be retrieved
    via GET /v1/datasets/{dataset_id}
    """
    # Get generator
    try:
        generator_class = GeneratorRegistry.get(request.generator)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Generate dataset ID
    generator = generator_class()
    dataset_id = generate_dataset_id(
        generator=request.generator,
        params=request.params.dict(),
        version=generator.VERSION
    )

    # Check cache first
    if request.store and store.exists(dataset_id):
        logger.info(f"Returning cached dataset: {dataset_id}")
        cached = store.get(dataset_id)
        return cached

    # Generate new dataset
    logger.info(f"Generating new dataset: {request.generator}")
    raw_dataset = generator.generate(request.params)

    # Apply splitting
    split = split_dataset(
        x=raw_dataset.x,
        y=raw_dataset.y,
        train_ratio=request.params.train_ratio,
        test_ratio=request.params.test_ratio,
        shuffle=request.params.shuffle,
        seed=request.params.random_seed
    )

    # Build response
    response = DatasetResponse(
        dataset_id=dataset_id,
        generator=request.generator,
        version=generator.VERSION,
        meta=DatasetMeta(
            input_size=raw_dataset.x.shape[1],
            output_size=raw_dataset.y.shape[1],
            n_samples=raw_dataset.x.shape[0],
            n_train=split.x_train.shape[0],
            n_test=split.x_test.shape[0]
        ),
        dataset={
            "x_train": split.x_train.tolist(),
            "y_train": split.y_train.tolist(),
            "x_test": split.x_test.tolist(),
            "y_test": split.y_test.tolist()
        } if not request.store else None,
        links={
            "self": f"/v1/datasets/{dataset_id}",
            "generator": f"/v1/generators/{request.generator}"
        }
    )

    # Store if requested
    if request.store:
        store.put(dataset_id, response)
        logger.info(f"Stored dataset: {dataset_id}")

    return response


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    store: DatasetStore = Depends(get_store)
) -> DatasetResponse:
    """Retrieve a stored dataset by ID."""
    if not store.exists(dataset_id):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {dataset_id}"
        )

    return store.get(dataset_id)


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    store: DatasetStore = Depends(get_store)
):
    """Delete a stored dataset."""
    if not store.exists(dataset_id):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {dataset_id}"
        )

    store.delete(dataset_id)
    return {"status": "deleted", "dataset_id": dataset_id}
```

### Step 2.3: Generator Information Endpoints

**Name**: Generator Discovery Endpoints  
**Description**: Implement endpoints for listing generators and their schemas.

**Detailed Explanation**:

```python
# juniper_data/api/routes/generators.py

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from ...generators import GeneratorRegistry

router = APIRouter()


@router.get("/generators", response_model=List[Dict[str, Any]])
async def list_generators():
    """List all available dataset generators."""
    return GeneratorRegistry.list_generators()


@router.get("/generators/{name}")
async def get_generator_info(name: str) -> Dict[str, Any]:
    """Get detailed information about a specific generator."""
    try:
        generator_class = GeneratorRegistry.get(name)
        return generator_class.get_info()
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Generator not found: {name}"
        )


@router.get("/generators/{name}/schema")
async def get_generator_schema(name: str) -> Dict[str, Any]:
    """Get the JSON schema for a generator's parameters."""
    try:
        generator_class = GeneratorRegistry.get(name)
        return generator_class.get_params_schema().schema()
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Generator not found: {name}"
        )
```

### Step 2.4: Health Check Endpoint

**Name**: Health Check Implementation  
**Description**: Add health check endpoint for service monitoring.

**Detailed Explanation**:

```python
# juniper_data/api/routes/health.py

from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/healthz")
async def health_check():
    """
    Health check endpoint for load balancers and monitoring.

    Returns service status and version information.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "juniper-data",
        "version": "1.0.0"
    }


@router.get("/readyz")
async def readiness_check():
    """
    Readiness check for Kubernetes-style deployments.

    Checks if the service is ready to accept traffic.
    """
    # Add checks for required dependencies here
    # e.g., database connections, cache availability
    return {
        "status": "ready",
        "checks": {
            "generators": "ok",
            "storage": "ok"
        }
    }
```

---

## 6. Phase 3: Storage & Caching

**Priority**: P1 - Important  
**Effort**: M (2-3 days)  
**Goal**: Implement dataset persistence and caching

### Step 3.1: Storage Interface Definition

**Name**: DatasetStore Protocol  
**Description**: Define the abstract interface for dataset storage.

**Detailed Explanation**:

```python
# juniper_data/storage/interface.py

from typing import Protocol, Optional, runtime_checkable
from ..api.schemas.responses import DatasetResponse


@runtime_checkable
class DatasetStore(Protocol):
    """
    Protocol for dataset storage implementations.

    Implementations must provide:
    - get(): Retrieve dataset by ID
    - put(): Store dataset
    - exists(): Check if dataset exists
    - delete(): Remove dataset
    - list_ids(): List all stored dataset IDs
    """

    def get(self, dataset_id: str) -> Optional[DatasetResponse]:
        """Retrieve a dataset by ID. Returns None if not found."""
        ...

    def put(self, dataset_id: str, dataset: DatasetResponse) -> None:
        """Store a dataset."""
        ...

    def exists(self, dataset_id: str) -> bool:
        """Check if a dataset exists in storage."""
        ...

    def delete(self, dataset_id: str) -> bool:
        """Delete a dataset. Returns True if deleted, False if not found."""
        ...

    def list_ids(self) -> list[str]:
        """List all stored dataset IDs."""
        ...

    def clear(self) -> int:
        """Clear all stored datasets. Returns count of deleted items."""
        ...
```

### Step 3.2: In-Memory Store Implementation

**Name**: InMemoryStore  
**Description**: Simple in-memory storage for development and testing.

**Detailed Explanation**:

```python
# juniper_data/storage/memory.py

from typing import Dict, Optional, List
from datetime import datetime
import logging

from .interface import DatasetStore
from ..api.schemas.responses import DatasetResponse

logger = logging.getLogger(__name__)


class InMemoryStore(DatasetStore):
    """
    In-memory dataset storage.

    Suitable for:
    - Development and testing
    - Single-instance deployments with small datasets
    - Temporary caching

    Limitations:
    - Data lost on restart
    - Not suitable for production with large datasets
    - No persistence across instances
    """

    def __init__(self, max_entries: int = 100):
        self._store: Dict[str, DatasetResponse] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._max_entries = max_entries

    def get(self, dataset_id: str) -> Optional[DatasetResponse]:
        return self._store.get(dataset_id)

    def put(self, dataset_id: str, dataset: DatasetResponse) -> None:
        # Evict oldest if at capacity
        if len(self._store) >= self._max_entries:
            self._evict_oldest()

        self._store[dataset_id] = dataset
        self._timestamps[dataset_id] = datetime.utcnow()
        logger.debug(f"Stored dataset in memory: {dataset_id}")

    def exists(self, dataset_id: str) -> bool:
        return dataset_id in self._store

    def delete(self, dataset_id: str) -> bool:
        if dataset_id in self._store:
            del self._store[dataset_id]
            del self._timestamps[dataset_id]
            return True
        return False

    def list_ids(self) -> List[str]:
        return list(self._store.keys())

    def clear(self) -> int:
        count = len(self._store)
        self._store.clear()
        self._timestamps.clear()
        return count

    def _evict_oldest(self) -> None:
        """Remove the oldest entry to make room for new ones."""
        if not self._timestamps:
            return
        oldest_id = min(self._timestamps, key=self._timestamps.get)
        self.delete(oldest_id)
        logger.debug(f"Evicted oldest dataset: {oldest_id}")
```

### Step 3.3: Local Filesystem Store Implementation

**Name**: LocalFileStore  
**Description**: Persistent storage using local filesystem.

**Detailed Explanation**:

```python
# juniper_data/storage/filesystem.py

from typing import Optional, List
from pathlib import Path
import json
import logging

from .interface import DatasetStore
from ..api.schemas.responses import DatasetResponse

logger = logging.getLogger(__name__)


class LocalFileStore(DatasetStore):
    """
    Local filesystem dataset storage.

    Stores datasets as JSON files in a configurable directory.

    Suitable for:
    - Development with persistence
    - Single-instance production deployments
    - Sharing datasets across restarts

    File format: {dataset_id}.json
    """

    def __init__(self, storage_dir: str = "./data/datasets"):
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalFileStore initialized at: {self._storage_dir}")

    def _get_path(self, dataset_id: str) -> Path:
        return self._storage_dir / f"{dataset_id}.json"

    def get(self, dataset_id: str) -> Optional[DatasetResponse]:
        path = self._get_path(dataset_id)
        if not path.exists():
            return None

        with open(path, 'r') as f:
            data = json.load(f)
        return DatasetResponse(**data)

    def put(self, dataset_id: str, dataset: DatasetResponse) -> None:
        path = self._get_path(dataset_id)
        with open(path, 'w') as f:
            json.dump(dataset.dict(), f, indent=2)
        logger.debug(f"Stored dataset to file: {path}")

    def exists(self, dataset_id: str) -> bool:
        return self._get_path(dataset_id).exists()

    def delete(self, dataset_id: str) -> bool:
        path = self._get_path(dataset_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_ids(self) -> List[str]:
        return [
            p.stem for p in self._storage_dir.glob("*.json")
        ]

    def clear(self) -> int:
        count = 0
        for path in self._storage_dir.glob("*.json"):
            path.unlink()
            count += 1
        return count
```

### Step 3.4: Storage Configuration

**Name**: Storage Provider Configuration  
**Description**: Configure storage backend via environment variables.

**Detailed Explanation**:

```python
# juniper_data/config/settings.py

from pydantic import BaseSettings
from typing import List, Optional
from functools import lru_cache

from ..storage.interface import DatasetStore
from ..storage.memory import InMemoryStore
from ..storage.filesystem import LocalFileStore


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # CORS settings
    cors_origins: List[str] = ["*"]

    # Storage settings
    storage_backend: str = "memory"  # "memory" or "filesystem"
    storage_dir: str = "./data/datasets"
    storage_max_entries: int = 100

    # Dataset limits
    max_samples_per_request: int = 100000
    max_payload_mb: int = 50

    class Config:
        env_prefix = "JUNIPER_DATA_"
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_store() -> DatasetStore:
    """Get configured storage backend."""
    settings = get_settings()

    if settings.storage_backend == "filesystem":
        return LocalFileStore(settings.storage_dir)
    else:
        return InMemoryStore(settings.storage_max_entries)
```

---

## 7. Phase 4: Cascor Integration

**Priority**: P0 - Critical Path  
**Effort**: M (2-3 days)  
**Goal**: Integrate Juniper Data with Juniper Cascor

### Step 4.1: Create DatasetProvider Interface in Cascor

**Name**: DatasetProvider Abstraction  
**Description**: Define interface for dataset sources in Cascor.

**Detailed Explanation**:

```python
# src/dataset_providers/interface.py (in Juniper Cascor)

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import torch


@dataclass
class DatasetResult:
    """Container for dataset results from a provider."""
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    x_full: Optional[torch.Tensor] = None
    y_full: Optional[torch.Tensor] = None
    meta: Dict[str, Any] = None
    dataset_id: Optional[str] = None


class DatasetProvider(ABC):
    """
    Abstract base class for dataset providers.

    Providers fetch datasets from various sources:
    - Local generation (legacy behavior)
    - Remote Juniper Data service
    - File system
    - etc.
    """

    @abstractmethod
    def get_spiral_dataset(
        self,
        n_spirals: int = 2,
        n_points_per_spiral: int = 100,
        **kwargs
    ) -> DatasetResult:
        """
        Fetch a spiral dataset with the given parameters.

        Returns:
            DatasetResult with train/test tensors
        """
        raise NotImplementedError
```

### Step 4.2: Implement Local Provider (Backward Compatibility)

**Name**: LocalSpiralProvider  
**Description**: Wrapper for existing local generation for backward compatibility.

**Detailed Explanation**:

```python
# src/dataset_providers/local.py (in Juniper Cascor)

import torch
import numpy as np
from typing import Tuple

from .interface import DatasetProvider, DatasetResult
from cascor_constants.constants import (
    _SPIRAL_PROBLEM_NUM_SPIRALS,
    _SPIRAL_PROBLEM_NUMBER_POINTS_PER_SPIRAL,
    _SPIRAL_PROBLEM_NUM_ROTATIONS,
    _SPIRAL_PROBLEM_CLOCKWISE,
    _SPIRAL_PROBLEM_NOISE_FACTOR_DEFAULT,
    _SPIRAL_PROBLEM_DISTRIBUTION_FACTOR,
    _SPIRAL_PROBLEM_TRAIN_RATIO,
    _SPIRAL_PROBLEM_TEST_RATIO,
)


class LocalSpiralProvider(DatasetProvider):
    """
    Local spiral dataset provider.

    Uses the original generation code from SpiralProblem class
    for backward compatibility during migration.
    """

    def get_spiral_dataset(
        self,
        n_spirals: int = None,
        n_points_per_spiral: int = None,
        n_rotations: float = None,
        clockwise: bool = None,
        noise_level: float = None,
        distribution: float = None,
        train_ratio: float = None,
        test_ratio: float = None,
        random_seed: int = 42,
        **kwargs
    ) -> DatasetResult:
        """Generate spiral dataset locally."""

        # Apply defaults
        n_spirals = n_spirals or _SPIRAL_PROBLEM_NUM_SPIRALS
        n_points = n_points_per_spiral or _SPIRAL_PROBLEM_NUMBER_POINTS_PER_SPIRAL
        n_rotations = n_rotations or _SPIRAL_PROBLEM_NUM_ROTATIONS
        clockwise = clockwise if clockwise is not None else _SPIRAL_PROBLEM_CLOCKWISE
        noise = noise_level or _SPIRAL_PROBLEM_NOISE_FACTOR_DEFAULT
        dist = distribution or _SPIRAL_PROBLEM_DISTRIBUTION_FACTOR
        train_r = train_ratio or _SPIRAL_PROBLEM_TRAIN_RATIO
        test_r = test_ratio or _SPIRAL_PROBLEM_TEST_RATIO

        # Generate using existing logic (simplified)
        np.random.seed(random_seed)

        x, y = self._generate_spiral_data(
            n_spirals, n_points, n_rotations, clockwise, noise, dist
        )

        # Convert to tensors
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Shuffle and split
        indices = torch.randperm(x_tensor.size(0))
        x_shuffled = x_tensor[indices]
        y_shuffled = y_tensor[indices]

        train_end = int(len(x_shuffled) * train_r)

        return DatasetResult(
            x_train=x_shuffled[:train_end],
            y_train=y_shuffled[:train_end],
            x_test=x_shuffled[train_end:],
            y_test=y_shuffled[train_end:],
            x_full=x_tensor,
            y_full=y_tensor,
            meta={
                "n_spirals": n_spirals,
                "n_points": n_points,
                "source": "local"
            }
        )

    def _generate_spiral_data(
        self,
        n_spirals: int,
        n_points: int,
        n_rotations: float,
        clockwise: bool,
        noise: float,
        distribution: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Core spiral generation logic."""
        # ... (existing spiral generation code)
        pass  # Implementation from current SpiralProblem
```

### Step 4.3: Implement Remote Provider (Juniper Data Client)

**Name**: RemoteJuniperDataProvider  
**Description**: HTTP client for fetching datasets from Juniper Data service.

**Detailed Explanation**:

```python
# src/dataset_providers/remote.py (in Juniper Cascor)

import requests
import torch
import numpy as np
from typing import Optional
import logging

from .interface import DatasetProvider, DatasetResult

logger = logging.getLogger(__name__)


class RemoteJuniperDataProvider(DatasetProvider):
    """
    Remote dataset provider using Juniper Data REST API.

    Fetches datasets from a running Juniper Data service
    and converts them to PyTorch tensors.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._session = requests.Session()

    def get_spiral_dataset(
        self,
        n_spirals: int = 2,
        n_points_per_spiral: int = 100,
        n_rotations: float = 3.0,
        clockwise: bool = True,
        noise_level: float = 0.25,
        distribution: float = 0.8,
        train_ratio: float = 0.8,
        test_ratio: float = 0.2,
        random_seed: int = 42,
        store: bool = True,
        **kwargs
    ) -> DatasetResult:
        """Fetch spiral dataset from Juniper Data service."""

        # Build request
        url = f"{self.base_url}/v1/datasets/generate"
        payload = {
            "generator": "spiral",
            "params": {
                "n_spirals": n_spirals,
                "n_points_per_spiral": n_points_per_spiral,
                "n_rotations": n_rotations,
                "clockwise": clockwise,
                "noise_level": noise_level,
                "distribution": distribution,
                "train_ratio": train_ratio,
                "test_ratio": test_ratio,
                "random_seed": random_seed,
                "shuffle": True
            },
            "store": store
        }

        logger.info(f"Requesting spiral dataset from {url}")

        try:
            response = self._session.post(
                url, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch dataset: {e}")
            raise

        # If store=True, we need to fetch the actual data
        if store and data.get("dataset") is None:
            dataset_id = data["dataset_id"]
            data = self._fetch_stored_dataset(dataset_id)

        # Convert to tensors
        dataset = data["dataset"]
        return DatasetResult(
            x_train=torch.tensor(dataset["x_train"], dtype=torch.float32),
            y_train=torch.tensor(dataset["y_train"], dtype=torch.float32),
            x_test=torch.tensor(dataset["x_test"], dtype=torch.float32),
            y_test=torch.tensor(dataset["y_test"], dtype=torch.float32),
            meta=data.get("meta"),
            dataset_id=data.get("dataset_id")
        )

    def _fetch_stored_dataset(self, dataset_id: str) -> dict:
        """Fetch a stored dataset by ID."""
        url = f"{self.base_url}/v1/datasets/{dataset_id}"
        response = self._session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def health_check(self) -> bool:
        """Check if Juniper Data service is available."""
        try:
            response = self._session.get(
                f"{self.base_url}/healthz",
                timeout=5
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
```

### Step 4.4: Configuration for Provider Selection

**Name**: Provider Configuration  
**Description**: Environment-based provider selection in Cascor.

**Detailed Explanation**:

```python
# src/dataset_providers/__init__.py (in Juniper Cascor)

import os
from typing import Optional
import logging

from .interface import DatasetProvider
from .local import LocalSpiralProvider
from .remote import RemoteJuniperDataProvider

logger = logging.getLogger(__name__)


def get_dataset_provider(
    provider_type: Optional[str] = None,
    base_url: Optional[str] = None
) -> DatasetProvider:
    """
    Get configured dataset provider.

    Configuration via environment variables:
    - CASCOR_DATA_PROVIDER: "local" or "juniper_data"
    - CASCOR_JUNIPER_DATA_URL: Base URL for remote service

    Args:
        provider_type: Override for provider type
        base_url: Override for remote URL

    Returns:
        Configured DatasetProvider instance
    """
    provider = provider_type or os.getenv("CASCOR_DATA_PROVIDER", "local")

    if provider == "juniper_data":
        url = base_url or os.getenv(
            "CASCOR_JUNIPER_DATA_URL",
            "http://localhost:8000"
        )
        logger.info(f"Using remote Juniper Data provider: {url}")
        return RemoteJuniperDataProvider(base_url=url)
    else:
        logger.info("Using local spiral provider")
        return LocalSpiralProvider()
```

### Step 4.5: Update SpiralProblem to Use Provider

**Name**: SpiralProblem Refactoring  
**Description**: Modify SpiralProblem to use DatasetProvider instead of internal generation.

**Detailed Explanation**:

```python
# Changes to src/spiral_problem/spiral_problem.py

# Add import at top
from dataset_providers import get_dataset_provider, DatasetProvider

class SpiralProblem(object):

    def __init__(
        self,
        # ... existing parameters ...
        _SpiralProblem__dataset_provider: DatasetProvider = None,
        **kwargs
    ):
        # ... existing initialization ...

        # Initialize dataset provider
        self.dataset_provider = _SpiralProblem__dataset_provider or get_dataset_provider()

    def solve_n_spiral_problem(
        self,
        n_points=None,
        n_spirals=None,
        # ... other params ...
    ):
        """Solve the N spiral problem using external dataset provider."""

        # Get dataset from provider instead of generating internally
        dataset_result = self.dataset_provider.get_spiral_dataset(
            n_spirals=n_spirals or self.n_spirals,
            n_points_per_spiral=n_points or self.n_points,
            n_rotations=self.n_rotations,
            clockwise=self.clockwise,
            noise_level=self.noise,
            distribution=self.distribution,
            train_ratio=self.train_ratio,
            test_ratio=self.test_ratio,
            random_seed=self.random_seed
        )

        # Use results from provider
        self.x_train = dataset_result.x_train
        self.y_train = dataset_result.y_train
        self.x_test = dataset_result.x_test
        self.y_test = dataset_result.y_test
        self.x_full = dataset_result.x_full
        self.y_full = dataset_result.y_full

        # ... continue with training ...
```

---

## 8. Phase 5: Extended Data Sources

**Priority**: P2 - Enhancement  
**Effort**: L (1-2 weeks)  
**Goal**: Add support for multiple data sources

### Step 5.1: DataSource Abstract Interface

**Name**: DataSource Protocol  
**Description**: Define unified interface for all data sources.

**Detailed Explanation**:

```python
# juniper_data/sources/interface.py

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from enum import Enum


class SourceType(Enum):
    """Types of data sources."""
    GENERATOR = "generator"      # Local synthetic generation
    FILESYSTEM = "filesystem"    # Local file storage
    OBJECT_STORE = "object_store"  # S3, GCS, etc.
    DATABASE = "database"        # SQL or NoSQL
    REMOTE_WEB = "remote_web"    # HuggingFace, Kaggle, etc.


class DataSource(ABC):
    """
    Abstract base class for data sources.

    Data sources can be:
    - Generators (synthetic data creation)
    - Storage (retrieve existing datasets)
    - Remote (download from external services)
    """

    source_type: SourceType

    @abstractmethod
    def get_dataset(
        self,
        identifier: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Get a dataset from this source.

        Args:
            identifier: Dataset name/id/path
            params: Source-specific parameters

        Returns:
            Dataset in canonical format
        """
        raise NotImplementedError

    @abstractmethod
    def list_available(self) -> list:
        """List available datasets from this source."""
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this source is currently available."""
        raise NotImplementedError
```

### Step 5.2: File System Source

**Name**: FileSystemSource  
**Description**: Retrieve datasets from local file system.

**Detailed Explanation**:

```python
# juniper_data/sources/filesystem.py

from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import numpy as np
import logging

from .interface import DataSource, SourceType

logger = logging.getLogger(__name__)


class FileSystemSource(DataSource):
    """
    File system data source.

    Supports:
    - JSON files with nested arrays
    - NPZ files (numpy compressed)
    - CSV files
    - HDF5 files (with h5py)
    """

    source_type = SourceType.FILESYSTEM

    def __init__(self, base_path: str = "./data"):
        self.base_path = Path(base_path)
        self._supported_extensions = ['.json', '.npz', '.csv', '.h5']

    def get_dataset(
        self,
        identifier: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, np.ndarray]:
        """Load dataset from file."""
        path = self.base_path / identifier

        if path.suffix == '.json':
            return self._load_json(path)
        elif path.suffix == '.npz':
            return self._load_npz(path)
        elif path.suffix == '.csv':
            return self._load_csv(path, params)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def list_available(self) -> List[str]:
        """List all available dataset files."""
        datasets = []
        for ext in self._supported_extensions:
            datasets.extend([
                str(p.relative_to(self.base_path))
                for p in self.base_path.rglob(f"*{ext}")
            ])
        return datasets

    def is_available(self) -> bool:
        return self.base_path.exists()

    def _load_json(self, path: Path) -> Dict[str, np.ndarray]:
        with open(path) as f:
            data = json.load(f)
        return {k: np.array(v) for k, v in data.items()}

    def _load_npz(self, path: Path) -> Dict[str, np.ndarray]:
        return dict(np.load(path))

    def _load_csv(
        self,
        path: Path,
        params: Optional[Dict] = None
    ) -> Dict[str, np.ndarray]:
        import csv
        # Implementation for CSV loading
        pass
```

### Step 5.3: Object Store Source (S3/GCS)

**Name**: ObjectStoreSource  
**Description**: Retrieve datasets from cloud object storage.

**Detailed Explanation**:

```python
# juniper_data/sources/object_store.py

from typing import Optional, Dict, Any, List
import logging
from abc import abstractmethod

from .interface import DataSource, SourceType

logger = logging.getLogger(__name__)


class ObjectStoreSource(DataSource):
    """
    Base class for cloud object storage sources.

    Implementations:
    - S3Source (AWS S3)
    - GCSSource (Google Cloud Storage)
    - AzureBlobSource (Azure Blob Storage)
    """

    source_type = SourceType.OBJECT_STORE

    @abstractmethod
    def _get_client(self):
        """Get storage client."""
        raise NotImplementedError


class S3Source(ObjectStoreSource):
    """AWS S3 data source."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str = "us-east-1",
        credentials: Optional[Dict] = None
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client('s3', region_name=self.region)
        return self._client

    def get_dataset(
        self,
        identifier: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """Download and load dataset from S3."""
        import tempfile
        import numpy as np

        key = f"{self.prefix}/{identifier}" if self.prefix else identifier

        with tempfile.NamedTemporaryFile(suffix='.npz') as tmp:
            self._get_client().download_file(self.bucket, key, tmp.name)
            return dict(np.load(tmp.name))

    def list_available(self) -> List[str]:
        """List datasets in bucket."""
        response = self._get_client().list_objects_v2(
            Bucket=self.bucket,
            Prefix=self.prefix
        )
        return [obj['Key'] for obj in response.get('Contents', [])]

    def is_available(self) -> bool:
        try:
            self._get_client().head_bucket(Bucket=self.bucket)
            return True
        except Exception:
            return False
```

### Step 5.4: Database Source

**Name**: DatabaseSource  
**Description**: Retrieve datasets from SQL/NoSQL databases.

**Detailed Explanation**:

```python
# juniper_data/sources/database.py

from typing import Optional, Dict, Any, List
import numpy as np
import logging

from .interface import DataSource, SourceType

logger = logging.getLogger(__name__)


class SQLDatabaseSource(DataSource):
    """
    SQL database data source.

    Supports PostgreSQL, MySQL, SQLite via SQLAlchemy.
    """

    source_type = SourceType.DATABASE

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from sqlalchemy import create_engine
            self._engine = create_engine(self.connection_string)
        return self._engine

    def get_dataset(
        self,
        identifier: str,  # Table name or SQL query
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, np.ndarray]:
        """Load dataset from database table/query."""
        import pandas as pd

        # Determine if identifier is a table or query
        if identifier.lower().startswith('select'):
            query = identifier
        else:
            query = f"SELECT * FROM {identifier}"

        df = pd.read_sql(query, self._get_engine())

        # Convert to numpy arrays based on params
        x_cols = params.get('x_columns', df.columns[:-1])
        y_cols = params.get('y_columns', [df.columns[-1]])

        return {
            'x': df[x_cols].values.astype(np.float32),
            'y': df[y_cols].values.astype(np.float32)
        }

    def list_available(self) -> List[str]:
        """List available tables."""
        from sqlalchemy import inspect
        inspector = inspect(self._get_engine())
        return inspector.get_table_names()

    def is_available(self) -> bool:
        try:
            self._get_engine().connect()
            return True
        except Exception:
            return False
```

### Step 5.5: Remote Web Source (Hugging Face, Kaggle)

**Name**: RemoteWebSource  
**Description**: Download datasets from external web services.

**Detailed Explanation**:

```python
# juniper_data/sources/remote_web.py

from typing import Optional, Dict, Any, List
import numpy as np
import logging
from abc import abstractmethod

from .interface import DataSource, SourceType

logger = logging.getLogger(__name__)


class RemoteWebSource(DataSource):
    """Base class for remote web data sources."""

    source_type = SourceType.REMOTE_WEB

    @abstractmethod
    def _authenticate(self):
        """Authenticate with the remote service."""
        pass


class HuggingFaceSource(RemoteWebSource):
    """
    Hugging Face Datasets source.

    Uses the datasets library to download and cache datasets.
    """

    def __init__(self, token: Optional[str] = None, cache_dir: str = None):
        self.token = token
        self.cache_dir = cache_dir

    def _authenticate(self):
        if self.token:
            from huggingface_hub import login
            login(token=self.token)

    def get_dataset(
        self,
        identifier: str,  # Dataset name, e.g., "mnist"
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, np.ndarray]:
        """Download dataset from Hugging Face."""
        from datasets import load_dataset

        params = params or {}
        split = params.get('split', 'train')

        dataset = load_dataset(
            identifier,
            split=split,
            cache_dir=self.cache_dir
        )

        # Convert to numpy based on schema
        # This is dataset-specific
        return self._to_numpy(dataset, params)

    def _to_numpy(self, dataset, params: Dict) -> Dict[str, np.ndarray]:
        """Convert HF dataset to numpy arrays."""
        x_key = params.get('x_key', 'image')
        y_key = params.get('y_key', 'label')

        return {
            'x': np.array(dataset[x_key]),
            'y': np.array(dataset[y_key])
        }

    def list_available(self) -> List[str]:
        """List popular available datasets."""
        # Return commonly used datasets
        return [
            "mnist",
            "cifar10",
            "cifar100",
            "fashion_mnist",
            "imdb"
        ]

    def is_available(self) -> bool:
        try:
            import datasets
            return True
        except ImportError:
            return False


class KaggleSource(RemoteWebSource):
    """
    Kaggle Datasets source.

    Uses the Kaggle API to download competition/dataset data.
    """

    def __init__(self, username: str = None, key: str = None):
        self.username = username
        self.key = key

    def _authenticate(self):
        import os
        if self.username and self.key:
            os.environ['KAGGLE_USERNAME'] = self.username
            os.environ['KAGGLE_KEY'] = self.key

    def get_dataset(
        self,
        identifier: str,  # "owner/dataset-name"
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, np.ndarray]:
        """Download dataset from Kaggle."""
        import kaggle
        import tempfile
        import pandas as pd

        self._authenticate()

        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle.api.dataset_download_files(
                identifier,
                path=tmpdir,
                unzip=True
            )

            # Load first CSV found
            import glob
            csv_files = glob.glob(f"{tmpdir}/*.csv")
            if csv_files:
                df = pd.read_csv(csv_files[0])
                return {
                    'x': df.iloc[:, :-1].values.astype(np.float32),
                    'y': df.iloc[:, -1].values.astype(np.float32)
                }

        raise FileNotFoundError("No CSV files found in dataset")

    def list_available(self) -> List[str]:
        """List datasets - requires search."""
        return []  # Kaggle requires search

    def is_available(self) -> bool:
        try:
            import kaggle
            kaggle.api.authenticate()
            return True
        except Exception:
            return False
```

---

## 9. Environment & Configuration Management

### 9.1 Juniper Data Environment Variables

```bash
# .env.example for Juniper Data

# Server Configuration
JUNIPER_DATA_HOST=0.0.0.0
JUNIPER_DATA_PORT=8000
JUNIPER_DATA_DEBUG=false

# Storage Configuration
JUNIPER_DATA_STORAGE_BACKEND=memory  # memory, filesystem, s3
JUNIPER_DATA_STORAGE_DIR=./data/datasets
JUNIPER_DATA_STORAGE_MAX_ENTRIES=100

# S3 Configuration (when storage_backend=s3)
JUNIPER_DATA_S3_BUCKET=juniper-datasets
JUNIPER_DATA_S3_REGION=us-east-1
JUNIPER_DATA_S3_PREFIX=datasets

# Dataset Limits
JUNIPER_DATA_MAX_SAMPLES_PER_REQUEST=100000
JUNIPER_DATA_MAX_PAYLOAD_MB=50

# CORS
JUNIPER_DATA_CORS_ORIGINS=["http://localhost:3000"]

# Logging
JUNIPER_DATA_LOG_LEVEL=INFO
```

### 9.2 Juniper Cascor Environment Variables

```bash
# Add to existing Cascor .env

# Dataset Provider Configuration
CASCOR_DATA_PROVIDER=local  # local, juniper_data
CASCOR_JUNIPER_DATA_URL=http://localhost:8000
CASCOR_JUNIPER_DATA_TIMEOUT=30
```

### 9.3 Dependencies Configuration

**Juniper Data pyproject.toml:**

```toml
[project]
name = "juniper-data"
version = "1.0.0"
description = "Dataset generation and management service for Juniper ML platform"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [{name = "Paul Calnon"}]

dependencies = [
    "numpy>=1.24.0",
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-asyncio>=0.21.0",
    "httpx>=0.24.0",  # For testing async endpoints
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.0.0",
]
storage = [
    "boto3>=1.28.0",        # S3 support
    "google-cloud-storage>=2.10.0",  # GCS support
    "sqlalchemy>=2.0.0",    # Database support
]
remote = [
    "datasets>=2.14.0",     # HuggingFace
    "kaggle>=1.5.0",        # Kaggle
]
all = [
    "juniper-data[dev,storage,remote]"
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

---

## 10. Risk Assessment & Mitigation

### 10.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Payload size causing latency | High | Medium | Add size limits, implement NPZ format option |
| RNG differences breaking reproducibility | Medium | High | Use numpy only, document RNG version in dataset ID |
| Service availability during migration | Medium | High | Keep local fallback provider as default |
| Coupling via constants reintroduced | Medium | Medium | Keep Juniper Data defaults local |
| Test failures during refactoring | Low | Medium | Implement both providers in parallel |

### 10.2 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Deployment complexity increase | High | Low | Keep local provider fallback |
| Service discovery issues | Medium | Medium | Use explicit configuration |
| Version mismatch between services | Low | High | Include version in dataset ID |

### 10.3 Migration Risk Mitigation

1. **Parallel Implementation**: Run both local and remote providers simultaneously during migration
2. **Feature Flags**: Use environment variables to toggle provider
3. **Gradual Rollout**: Start with CI using local, then dev, then production
4. **Rollback Plan**: Keep local provider fully functional

---

## 11. Testing Strategy

### 11.1 Unit Tests for Juniper Data

```python
# tests/unit/test_spiral_generator.py

import pytest
import numpy as np
from juniper_data.generators.spiral.generator import SpiralGenerator
from juniper_data.generators.spiral.params import SpiralGeneratorParams


class TestSpiralGenerator:

    def test_generate_basic(self):
        """Test basic spiral generation."""
        generator = SpiralGenerator()
        params = SpiralGeneratorParams(n_spirals=2, n_points_per_spiral=50)

        result = generator.generate(params)

        assert result.x.shape == (100, 2)  # 2 spirals * 50 points
        assert result.y.shape == (100, 2)  # One-hot for 2 classes

    def test_generate_reproducibility(self):
        """Test that same seed produces same dataset."""
        generator = SpiralGenerator()
        params = SpiralGeneratorParams(random_seed=42)

        result1 = generator.generate(params)
        result2 = generator.generate(params)

        np.testing.assert_array_equal(result1.x, result2.x)
        np.testing.assert_array_equal(result1.y, result2.y)

    def test_generate_n_spirals(self):
        """Test N-spiral generation."""
        generator = SpiralGenerator()
        params = SpiralGeneratorParams(n_spirals=5, n_points_per_spiral=20)

        result = generator.generate(params)

        assert result.x.shape == (100, 2)  # 5 * 20
        assert result.y.shape == (100, 5)  # 5 classes
        assert result.labels.max() == 4    # 0-indexed labels
```

### 11.2 Integration Tests

```python
# tests/integration/test_api.py

import pytest
from fastapi.testclient import TestClient
from juniper_data.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestDatasetEndpoints:

    def test_generate_spiral(self, client):
        """Test POST /v1/datasets/generate."""
        response = client.post("/v1/datasets/generate", json={
            "generator": "spiral",
            "params": {
                "n_spirals": 2,
                "n_points_per_spiral": 50
            },
            "store": False
        })

        assert response.status_code == 200
        data = response.json()
        assert "dataset_id" in data
        assert "dataset" in data
        assert len(data["dataset"]["x_train"]) > 0

    def test_store_and_retrieve(self, client):
        """Test storing and retrieving datasets."""
        # Generate and store
        gen_response = client.post("/v1/datasets/generate", json={
            "generator": "spiral",
            "params": {"n_spirals": 2},
            "store": True
        })
        dataset_id = gen_response.json()["dataset_id"]

        # Retrieve
        get_response = client.get(f"/v1/datasets/{dataset_id}")

        assert get_response.status_code == 200
        assert get_response.json()["dataset_id"] == dataset_id
```

### 11.3 Cascor Integration Tests

```python
# src/tests/integration/test_dataset_provider.py

import pytest
import torch
from dataset_providers import get_dataset_provider
from dataset_providers.local import LocalSpiralProvider
from dataset_providers.remote import RemoteJuniperDataProvider


class TestDatasetProviders:

    def test_local_provider(self):
        """Test local provider generates valid data."""
        provider = LocalSpiralProvider()
        result = provider.get_spiral_dataset(n_spirals=2)

        assert isinstance(result.x_train, torch.Tensor)
        assert result.x_train.shape[1] == 2
        assert result.y_train.shape[1] == 2

    @pytest.mark.integration
    def test_remote_provider(self):
        """Test remote provider (requires running Juniper Data)."""
        provider = RemoteJuniperDataProvider()

        if not provider.health_check():
            pytest.skip("Juniper Data service not available")

        result = provider.get_spiral_dataset(n_spirals=2)

        assert isinstance(result.x_train, torch.Tensor)
```

---

## 12. Implementation Checklist

### Phase 0: Contract & Interface Design ☐

- [ ] **0.1** Define DatasetContract model
- [ ] **0.2** Define SpiralGeneratorParams schema
- [ ] **0.3** Implement dataset ID generation algorithm
- [ ] **0.4** Document REST API specification

### Phase 1: Extract Generator Library ☐

- [ ] **1.1** Create Juniper Data project skeleton
- [ ] **1.2** Extract SpiralGenerator class
- [ ] **1.3** Implement dataset splitting utilities
- [ ] **1.4** Create BaseGenerator interface
- [ ] **1.5** Implement GeneratorRegistry
- [ ] **1.6** Write unit tests for generators

### Phase 2: Build REST API ☐

- [ ] **2.1** Create FastAPI application
- [ ] **2.2** Implement POST /v1/datasets/generate
- [ ] **2.3** Implement GET /v1/datasets/{id}
- [ ] **2.4** Implement generator info endpoints
- [ ] **2.5** Add health check endpoints
- [ ] **2.6** Write API integration tests

### Phase 3: Storage & Caching ☐

- [ ] **3.1** Define DatasetStore interface
- [ ] **3.2** Implement InMemoryStore
- [ ] **3.3** Implement LocalFileStore
- [ ] **3.4** Add storage configuration
- [ ] **3.5** Write storage tests

### Phase 4: Cascor Integration ☐

- [ ] **4.1** Create DatasetProvider interface
- [ ] **4.2** Implement LocalSpiralProvider
- [ ] **4.3** Implement RemoteJuniperDataProvider
- [ ] **4.4** Add provider configuration
- [ ] **4.5** Update SpiralProblem to use provider
- [ ] **4.6** Write integration tests
- [ ] **4.7** Update documentation

### Phase 5: Extended Data Sources (Future) ☐

- [ ] **5.1** Define DataSource interface
- [ ] **5.2** Implement FileSystemSource
- [ ] **5.3** Implement S3Source
- [ ] **5.4** Implement SQLDatabaseSource
- [ ] **5.5** Implement HuggingFaceSource
- [ ] **5.6** Implement KaggleSource

---

## Appendix A: API Examples

### Generate Spiral Dataset

```bash
curl -X POST http://localhost:8000/v1/datasets/generate \
  -H "Content-Type: application/json" \
  -d '{
    "generator": "spiral",
    "params": {
      "n_spirals": 2,
      "n_points_per_spiral": 100,
      "noise_level": 0.25,
      "random_seed": 42
    },
    "store": true
  }'
```

### Retrieve Stored Dataset

```bash
curl http://localhost:8000/v1/datasets/a1b2c3d4e5f6a7b8
```

### List Available Generators

```bash
curl http://localhost:8000/v1/generators
```

---

## Appendix B: Related Documents

- [AGENTS.md](../AGENTS.md) - Project guide and conventions
- [PRE-DEPLOYMENT_ROADMAP-2.md](./PRE-DEPLOYMENT_ROADMAP-2.md) - Integration roadmap
- [INTEGRATION_ROADMAP.md](./INTEGRATION_ROADMAP.md) - Cascor-Canopy integration
- [CHANGELOG.md](../CHANGELOG.md) - Version history

---

**Document Status**: Draft  
**Review Required**: Yes  
**Approval Required**: Before implementation begins
