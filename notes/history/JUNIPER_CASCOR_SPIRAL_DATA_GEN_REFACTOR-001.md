# Juniper Data: Spiral Data Generator Extraction Plan

**Document**: JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR.md  
**Created**: 2026-01-29  
**Last Updated**: 2026-01-29  
**Version**: 1.0.0  
**Status**: Draft - Pending Approval  
**Author**: Development Team

---

## Executive Summary

This document outlines the comprehensive plan to extract the Spiral Problem Data Generator code from **Juniper Cascor** and **Juniper Canopy** into a new standalone application called **Juniper Data**. The new application will provide a REST API for dataset generation, retrieval, and management, serving as the unified data service for the entire Juniper ecosystem.

### Key Objectives

1. **Extract** spiral data generation logic from Cascor and Canopy into Juniper Data
2. **Decouple** data generation from neural network training infrastructure
3. **Create** a REST API with POST (generate dataset) and GET (retrieve dataset) endpoints
4. **Design** an extensible architecture supporting multiple data sources:
   - Generated datasets (spiral, XOR, circles, etc.)
   - Local filesystem storage (previously generated/downloaded)
   - Remote cloud storage (S3, Google Cloud Storage)
   - Database retrieval (SQL/NoSQL)
   - Remote web sources (Hugging Face, Kaggle, USPS, etc.)

### Effort Estimation

| Phase                              | Effort | Duration  |
| ---------------------------------- | ------ | --------- |
| Phase 0: Inventory & Contract      | S      | 2-4 hours |
| Phase 1: Core Generator Extraction | M      | 4-8 hours |
| Phase 2: REST API Service          | M      | 4-8 hours |
| Phase 3: Canopy/Cascor Integration | L      | 1-2 days  |
| Phase 4: Provider Interfaces       | L      | 1-2 days  |
| Phase 5: Remote/DB Backends        | XL     | Staged    |

**Total Initial Deployment (Phases 0-3)**: 2-4 days

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Dependency Analysis](#2-dependency-analysis)
3. [Environment Configuration Analysis](#3-environment-configuration-analysis)
4. [Juniper Data Architecture](#4-juniper-data-architecture)
5. [REST API Specification](#5-rest-api-specification)
6. [Implementation Plan](#6-implementation-plan)
7. [Integration Strategy](#7-integration-strategy)
8. [Risk Assessment & Mitigations](#8-risk-assessment--mitigations)
9. [Appendices](#9-appendices)

---

## 1. Current State Analysis

### 1.1 Spiral Data Generator References

#### Juniper Cascor References

| File Path                                                     | Class/Function        | Purpose                                                                         | Lines |
| ------------------------------------------------------------- | --------------------- | ------------------------------------------------------------------------------- | ----- |
| `src/spiral_problem/spiral_problem.py`                        | `SpiralProblem`       | **Primary implementation** - Full N-spiral generation with training integration | ~1000 |
| `src/spiral_problem/check.py`                                 | `SpiralProblem`       | Legacy/secondary duplicate                                                      | ~200  |
| `src/main.py`                                                 | Import & run          | Entry point that uses SpiralProblem                                             | N/A   |
| `src/cascor_constants/constants_problem/constants_problem.py` | Constants             | Spiral-specific default values                                                  | ~50   |
| `src/tests/unit/test_data/generators.py`                      | `SpiralDataGenerator` | **Clean implementation** - Decoupled test data generator                        | 142   |

#### Juniper Canopy References

| File Path                           | Class/Function                                       | Purpose                        | Lines |
| ----------------------------------- | ---------------------------------------------------- | ------------------------------ | ----- |
| `src/demo_mode.py`                  | `DemoMode._generate_spiral_dataset()`                | Simple 2-spiral mock generator | 44    |
| `src/backend/cascor_integration.py` | `CascorIntegration._generate_missing_dataset_info()` | Fallback mock spiral dataset   | 35    |
|                                     |                                                      |                                |       |

### 1.2 Current Implementation Comparison

#### Cascor `SpiralProblem` (Full Implementation)

```python
# Key parameters (from constants_problem.py):
- n_spirals: int = 2
- n_points_per_spiral: int = 97
- n_rotations: int = 2
- noise_factor: float = 0.1
- train_ratio: float = 0.8
- test_ratio: float = 0.2
- clockwise: bool = True
- random_seed: int = 42

# Methods:
- generate_n_spiral_dataset() -> (x_train, y_train, x_test, y_test)
- generate_raw_spiral_coordinates() -> (x_coords, y_coords)
- _create_input_features() -> np.ndarray
- _create_one_hot_targets() -> np.ndarray
- _convert_to_tensors() -> (torch.Tensor, torch.Tensor)
- _shuffle_dataset() -> (torch.Tensor, torch.Tensor)
- _partition_dataset() -> ((train_x, train_y), (test_x, test_y))
```

#### Cascor Test Generator (Clean Pattern)

```python
# Key characteristics:
- Static methods (no class state)
- Returns: (x, y, DatasetInfo) tuple
- Uses torch tensors directly
- Simple parameter interface
- No logging/network dependencies

# Methods:
- generate_2_spiral(n_per_spiral, noise, seed)
- generate_n_spiral(n_spirals, n_per_spiral, noise, seed)
```

#### Canopy `DemoMode._generate_spiral_dataset()` (Simple Mock)

```python
# Key characteristics:
- Fixed 2-class spiral only
- Uses numpy, converts to torch at end
- Returns Dict with 'inputs', 'targets', 'inputs_tensor', 'targets_tensor'
- Fixed seed (42)
- Hardcoded noise (0.1)

# Return schema:
{
    "inputs": np.ndarray,           # [N, 2]
    "targets": np.ndarray,          # [N]
    "inputs_tensor": torch.Tensor,  # [N, 2]
    "targets_tensor": torch.Tensor, # [N, 1]
    "num_samples": int,
    "num_features": int,
    "num_classes": int
}
```

#### Canopy `CascorIntegration._generate_missing_dataset_info()` (Fallback)

```python
# Key characteristics:
- Inline implementation (not reusable)
- Returns JSON-serializable dict (lists, not tensors)
- Used when no dataset is available

# Return schema:
{
    "features": List[List[float]],
    "labels": List[int],
    "num_samples": int,
    "num_features": int,
    "num_classes": int,
    "class_distribution": Dict[int, int],
    "dataset_name": str,
    "mock_mode": bool
}
```

---

## 2. Dependency Analysis

### 2.1 Current Dependencies of SpiralProblem Class

#### Hard Dependencies (Currently Coupled)

| Dependency                   | Type    | Required for Data Gen? | Action                                           |
| ---------------------------- | ------- | ---------------------- | ------------------------------------------------ |
| `torch`                      | Library | Optional               | Keep for tensor conversion, use numpy internally |
| `numpy`                      | Library | **Yes**                | Core math operations                             |
| `CascadeCorrelationNetwork`  | Class   | **No**                 | Remove coupling                                  |
| `CascadeCorrelationConfig`   | Class   | **No**                 | Remove coupling                                  |
| `cascor_constants.constants` | Module  | **Partial**            | Extract relevant defaults only                   |
| `log_config.LogConfig`       | Class   | **No**                 | Use standard logging                             |
| `log_config.logger.Logger`   | Class   | **No**                 | Use standard logging                             |
| `multiprocessing`            | Library | **No**                 | Not needed for data gen                          |
| `matplotlib.pyplot`          | Library | **No**                 | Remove (visualization separate)                  |

#### Minimal Dependencies for Juniper Data

```python
# Core (required)
numpy>=1.24.0
pydantic>=2.0.0      # Request/response validation

# API Layer (required)
fastapi>=0.100.0
uvicorn[standard]>=0.23.0

# Optional (per source type)
torch>=2.0.0         # For tensor output format
h5py>=3.8.0          # For HDF5 format support
boto3>=1.28.0        # For S3 source
google-cloud-storage # For GCS source
sqlalchemy>=2.0.0    # For database source
datasets>=2.14.0     # For Hugging Face source
kaggle>=1.5.16       # For Kaggle source
python-multipart     # For file upload
```

### 2.2 Data Flow Dependencies

```bash
┌─────────────────────────────────────────────────────────────────────┐
│                        CURRENT STATE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐         ┌──────────────────────┐                  │
│  │ Cascor       │         │ Canopy               │                  │
│  │ SpiralProblem├────────►│ Backend Integration  │                  │
│  │ (+ CCN deps) │         │ (demo_mode.py)       │                  │
│  └──────┬───────┘         └──────────┬───────────┘                  │
│         │                            │                              │
│         │                            ▼                              │
│         │                  ┌──────────────────────┐                 │
│         └─────────────────►│ CascadeCorrelation   │                 │
│                            │ Network Training     │                 │
│                            └──────────────────────┘                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        TARGET STATE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────────────────────────────────┐                     │
│  │              JUNIPER DATA                  │                     │
│  │  ┌──────────────────────────────────────┐  │                     │
│  │  │         REST API Layer               │  │                     │
│  │  │  POST /v1/datasets (generate)        │  │                     │
│  │  │  GET  /v1/datasets/{id} (retrieve)   │  │                     │
│  │  └───────────────┬──────────────────────┘  │                     │
│  │                  │                         │                     │
│  │  ┌───────────────▼──────────────────────┐  │                     │
│  │  │         Source Providers             │  │                     │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ │  │                     │
│  │  │  │Generated│ │Local FS │ │ Remote  │ │  │                     │
│  │  │  │(spiral) │ │(cached) │ │(S3/HF)  │ │  │                     │
│  │  │  └─────────┘ └─────────┘ └─────────┘ │  │                     │
│  │  └──────────────────────────────────────┘  │                     │
│  └────────────────────────────────────────────┘                     │
│         │                            │                              │
│         ▼                            ▼                              │
│  ┌──────────────┐         ┌──────────────────────┐                  │
│  │ Cascor       │         │ Canopy               │                  │
│  │ Training     │◄───────►│ Frontend             │                  │
│  │ (consumer)   │         │ (consumer)           │                  │
│  └──────────────┘         └──────────────────────┘                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Environment Configuration Analysis

### 3.1 Current Package Configuration

#### Juniper Cascor (`pyproject.toml`)

```toml
[project]
name = "juniper-cascor"
version = "0.3.19"
requires-python = ">=3.11"

# Note: Runtime dependencies not listed - managed via conda environment
# Key tools configured: black, isort, pytest, mypy, coverage
```

#### Juniper Canopy (`pyproject.toml`)

```toml
[project]
name = "cascor-frontend"
version = "0.2.1"
requires-python = ">=3.11"

# Note: Similar structure - tool configs present, runtime deps not listed
```

### 3.2 Juniper Data Package Configuration (Proposed)

```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "juniper-data"
version = "0.1.0"
description = "Dataset generation and management service for Juniper ecosystem"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [{ name = "Paul Calnon", email = "paul.calnon@gmail.com" }]

dependencies = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    "pydantic>=2.0.0",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
torch = ["torch>=2.0.0"]
s3 = ["boto3>=1.28.0"]
gcs = ["google-cloud-storage>=2.10.0"]
database = ["sqlalchemy>=2.0.0"]
huggingface = ["datasets>=2.14.0"]
kaggle = ["kaggle>=1.5.16"]
all = [
    "juniper-data[torch,s3,gcs,database,huggingface,kaggle]"
]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.0.0",
]
```

### 3.3 Environment Variables

| Variable                      | Description                | Default     |
| ----------------------------- | -------------------------- | ----------- |
| `JUNIPER_DATA_HOST`           | API server host            | `127.0.0.1` |
| `JUNIPER_DATA_PORT`           | API server port            | `8100`      |
| `JUNIPER_DATA_DIR`            | Dataset storage directory  | `./data`    |
| `JUNIPER_DATA_LOG_LEVEL`      | Logging level              | `INFO`      |
| `JUNIPER_DATA_CORS_ORIGINS`   | Allowed CORS origins       | `*`         |
| `JUNIPER_DATA_MAX_CACHE_SIZE` | Max cached datasets        | `100`       |
| `JUNIPER_DATA_ENABLE_S3`      | Enable S3 source           | `false`     |
| `JUNIPER_DATA_ENABLE_GCS`     | Enable GCS source          | `false`     |
| `JUNIPER_DATA_ENABLE_DB`      | Enable database source     | `false`     |
| `JUNIPER_DATA_ENABLE_HF`      | Enable Hugging Face source | `false`     |
| `JUNIPER_DATA_ENABLE_KAGGLE`  | Enable Kaggle source       | `false`     |

---

## 4. Juniper Data Architecture

### 4.1 Project Structure

```bash
juniper_data/
├── src/
│   └── juniper_data/
│       ├── __init__.py
│       ├── main.py                    # Application entry point
│       │
│       ├── api/                       # REST API layer
│       │   ├── __init__.py
│       │   ├── app.py                 # FastAPI application
│       │   ├── dependencies.py        # Dependency injection
│       │   └── routes/
│       │       ├── __init__.py
│       │       ├── generators.py      # Generator discovery routes
│       │       ├── datasets.py        # Dataset CRUD routes
│       │       └── health.py          # Health check routes
│       │
│       ├── core/                      # Core domain models
│       │   ├── __init__.py
│       │   ├── models.py              # DatasetSpec, DatasetMeta, etc.
│       │   ├── registry.py            # Generator/source registry
│       │   └── exceptions.py          # Custom exceptions
│       │
│       ├── generators/                # Dataset generators
│       │   ├── __init__.py
│       │   ├── base.py                # BaseGenerator interface
│       │   ├── spiral.py              # N-spiral generator
│       │   ├── classification.py      # XOR, circles, etc.
│       │   └── regression.py          # Polynomial, sinusoidal
│       │
│       ├── sources/                   # Data source providers
│       │   ├── __init__.py
│       │   ├── base.py                # SourceProvider protocol
│       │   ├── generated.py           # Generated datasets
│       │   ├── local_fs.py            # Local filesystem
│       │   ├── s3.py                  # AWS S3 (optional)
│       │   ├── gcs.py                 # Google Cloud Storage (optional)
│       │   ├── database.py            # SQL/NoSQL (optional)
│       │   ├── huggingface.py         # Hugging Face datasets (optional)
│       │   └── kaggle.py              # Kaggle datasets (optional)
│       │
│       ├── storage/                   # Dataset artifact storage
│       │   ├── __init__.py
│       │   ├── base.py                # StorageBackend protocol
│       │   ├── local.py               # Local filesystem storage
│       │   └── formats.py             # npz, json, hdf5 format handlers
│       │
│       └── config/                    # Configuration
│           ├── __init__.py
│           └── settings.py            # Pydantic settings
│
├── tests/
│   ├── unit/
│   │   ├── test_generators.py
│   │   ├── test_sources.py
│   │   └── test_storage.py
│   └── integration/
│       ├── test_api.py
│       └── test_end_to_end.py
│
├── data/                              # Default dataset storage
│   └── .gitkeep
│
├── pyproject.toml
├── README.md
├── CHANGELOG.md
└── AGENTS.md
```

### 4.2 Core Domain Models

```python
# src/juniper_data/core/models.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import json

class DatasetFormat(str, Enum):
    NPZ = "npz"
    JSON = "json"
    HDF5 = "hdf5"
    TORCH = "torch"  # .pt file

class SourceType(str, Enum):
    GENERATED = "generated"
    LOCAL_FS = "local_fs"
    S3 = "s3"
    GCS = "gcs"
    DATABASE = "database"
    HUGGINGFACE = "huggingface"
    KAGGLE = "kaggle"
    URL = "url"

class SpiralParams(BaseModel):
    """Parameters for spiral dataset generation."""
    n_spirals: int = Field(default=2, ge=2, le=20)
    n_points_per_spiral: int = Field(default=100, ge=10, le=10000)
    n_rotations: float = Field(default=2.0, ge=0.5, le=10.0)
    noise: float = Field(default=0.1, ge=0.0, le=1.0)
    clockwise: bool = Field(default=True)
    seed: Optional[int] = Field(default=42)
    train_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.2, ge=0.0, le=1.0)

class DatasetRequest(BaseModel):
    """Request to create or retrieve a dataset."""
    source: SourceType = SourceType.GENERATED
    generator: str = "spiral"  # For generated source
    params: Dict[str, Any] = Field(default_factory=dict)
    format: DatasetFormat = DatasetFormat.NPZ
    cache: bool = True  # Cache generated dataset

class DatasetMeta(BaseModel):
    """Metadata about a dataset."""
    dataset_id: str
    name: str
    source: SourceType
    generator: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    n_samples: int
    n_features: int
    n_classes: int
    class_distribution: Dict[str, int] = Field(default_factory=dict)
    train_samples: Optional[int] = None
    test_samples: Optional[int] = None
    created_at: str
    format: DatasetFormat
    artifact_size_bytes: int

class DatasetResponse(BaseModel):
    """Response after dataset creation."""
    dataset_id: str
    meta: DatasetMeta
    artifacts: Dict[str, str]  # artifact_type -> URL

@dataclass
class DatasetArtifact:
    """Internal representation of a dataset artifact."""
    X: Any  # numpy array or torch tensor
    y: Any  # numpy array or torch tensor
    X_train: Optional[Any] = None
    y_train: Optional[Any] = None
    X_test: Optional[Any] = None
    y_test: Optional[Any] = None
    meta: Dict[str, Any] = field(default_factory=dict)
```

### 4.3 Generator Interface

```python
# src/juniper_data/generators/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Type
from pydantic import BaseModel
import numpy as np

class BaseGenerator(ABC):
    """Base class for dataset generators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique generator name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        pass

    @property
    @abstractmethod
    def params_model(self) -> Type[BaseModel]:
        """Pydantic model for parameters."""
        pass

    @abstractmethod
    def generate(self, params: BaseModel) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generate dataset.

        Returns:
            (X, y, meta): Features, labels, and metadata
        """
        pass

    def generate_with_split(
        self,
        params: BaseModel
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generate dataset with train/test split.

        Returns:
            (X_train, y_train, X_test, y_test, meta)
        """
        X, y, meta = self.generate(params)
        train_ratio = getattr(params, 'train_ratio', 0.8)

        n_train = int(len(X) * train_ratio)
        indices = np.random.permutation(len(X))

        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        meta['train_samples'] = n_train
        meta['test_samples'] = len(X) - n_train

        return X[train_idx], y[train_idx], X[test_idx], y[test_idx], meta
```

### 4.4 Spiral Generator Implementation

```python
# src/juniper_data/generators/spiral.py

import numpy as np
from typing import Dict, Any, Tuple, Type
from pydantic import BaseModel

from .base import BaseGenerator
from ..core.models import SpiralParams

class SpiralGenerator(BaseGenerator):
    """N-spiral classification dataset generator."""

    @property
    def name(self) -> str:
        return "spiral"

    @property
    def description(self) -> str:
        return "Generates N-spiral classification datasets with configurable noise and rotations"

    @property
    def params_model(self) -> Type[BaseModel]:
        return SpiralParams

    def generate(self, params: SpiralParams) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generate N-spiral dataset.

        The spiral generation algorithm:
        1. For each spiral class, generate points along an Archimedean spiral
        2. Apply angular offset based on spiral index (2π * i / n_spirals)
        3. Add Gaussian noise to introduce classification difficulty
        4. Create one-hot encoded labels

        Args:
            params: SpiralParams configuration

        Returns:
            (X, y, meta): Features [N, 2], labels [N, n_spirals], metadata
        """
        if params.seed is not None:
            np.random.seed(params.seed)

        n_spirals = params.n_spirals
        n_points = params.n_points_per_spiral
        noise = params.noise
        n_rotations = params.n_rotations
        direction = 1 if params.clockwise else -1

        total_points = n_spirals * n_points

        X_list = []
        y_list = []

        for i in range(n_spirals):
            # Radial distance increases linearly
            t = np.linspace(0.5, n_rotations * 2 * np.pi, n_points)

            # Angular offset for this spiral
            angle_offset = 2 * np.pi * i / n_spirals

            # Spiral coordinates with noise
            r = t / (n_rotations * 2 * np.pi)  # Normalize radius to [0, 1]
            x = direction * r * np.cos(t + angle_offset) + np.random.randn(n_points) * noise
            y = direction * r * np.sin(t + angle_offset) + np.random.randn(n_points) * noise

            X_list.append(np.column_stack([x, y]))

            # One-hot encoding for this class
            y_onehot = np.zeros((n_points, n_spirals), dtype=np.float32)
            y_onehot[:, i] = 1.0
            y_list.append(y_onehot)

        X = np.vstack(X_list).astype(np.float32)
        y = np.vstack(y_list).astype(np.float32)

        # Shuffle
        indices = np.random.permutation(total_points)
        X = X[indices]
        y = y[indices]

        meta = {
            "name": f"{n_spirals}-spiral",
            "n_samples": total_points,
            "n_features": 2,
            "n_classes": n_spirals,
            "class_distribution": {str(i): n_points for i in range(n_spirals)},
            "params": params.model_dump(),
        }

        return X, y, meta
```

### 4.5 Source Provider Interface

```python
# src/juniper_data/sources/base.py

from abc import ABC, abstractmethod
from typing import Optional
from ..core.models import DatasetRequest, DatasetArtifact, DatasetMeta

class SourceProvider(ABC):
    """Base class for dataset source providers."""

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Source type identifier."""
        pass

    @abstractmethod
    async def create_dataset(
        self,
        request: DatasetRequest
    ) -> DatasetArtifact:
        """
        Create or fetch a dataset based on the request.

        Args:
            request: Dataset creation request

        Returns:
            DatasetArtifact with data and metadata
        """
        pass

    @abstractmethod
    async def get_dataset(
        self,
        dataset_id: str
    ) -> Optional[DatasetArtifact]:
        """
        Retrieve a previously created dataset.

        Args:
            dataset_id: Unique dataset identifier

        Returns:
            DatasetArtifact if found, None otherwise
        """
        pass
```

---

## 5. REST API Specification

### 5.1 API Endpoints Overview

| Method   | Endpoint                       | Description                    |
| -------- | ------------------------------ | ------------------------------ |
| `GET`    | `/v1/health`                   | Health check                   |
| `GET`    | `/v1/generators`               | List available generators      |
| `GET`    | `/v1/generators/{name}/schema` | Get generator parameter schema |
| `POST`   | `/v1/datasets`                 | Create/generate a dataset      |
| `GET`    | `/v1/datasets/{id}`            | Get dataset metadata           |
| `GET`    | `/v1/datasets/{id}/artifact`   | Download dataset artifact      |
| `GET`    | `/v1/datasets/{id}/preview`    | Preview dataset (first N rows) |
| `DELETE` | `/v1/datasets/{id}`            | Delete a cached dataset        |
| `GET`    | `/v1/sources`                  | List available source types    |
| `POST`   | `/v1/datasets/import`          | Import from file/URL           |

### 5.2 Endpoint Specifications

#### POST /v1/datasets - Create Dataset

**Request Body (Generated Dataset):**

```json
{
    "source": "generated",
    "generator": "spiral",
    "params": {
        "n_spirals": 2,
        "n_points_per_spiral": 100,
        "n_rotations": 2.0,
        "noise": 0.1,
        "clockwise": true,
        "seed": 42,
        "train_ratio": 0.8,
        "test_ratio": 0.2
    },
    "format": "npz",
    "cache": true
}
```

**Request Body (Local Filesystem):**

```json
{
    "source": "local_fs",
    "params": {
        "path": "/data/datasets/my_spiral.npz"
    }
}
```

**Request Body (Hugging Face):**

```json
{
    "source": "huggingface",
    "params": {
        "dataset_name": "mnist",
        "split": "train",
        "config": "default"
    },
    "format": "npz"
}
```

**Request Body (URL Download):**

```json
{
    "source": "url",
    "params": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        "format": "csv",
        "has_header": false
    }
}
```

**Response (201 Created):**

```json
{
    "dataset_id": "ds_a1b2c3d4e5f6",
    "meta": {
        "dataset_id": "ds_a1b2c3d4e5f6",
        "name": "2-spiral",
        "source": "generated",
        "generator": "spiral",
        "params": {
            "n_spirals": 2,
            "n_points_per_spiral": 100,
            "noise": 0.1
        },
        "n_samples": 200,
        "n_features": 2,
        "n_classes": 2,
        "class_distribution": {"0": 100, "1": 100},
        "train_samples": 160,
        "test_samples": 40,
        "created_at": "2026-01-29T12:00:00Z",
        "format": "npz",
        "artifact_size_bytes": 12480
    },
    "artifacts": {
        "full": "/v1/datasets/ds_a1b2c3d4e5f6/artifact?format=npz",
        "train": "/v1/datasets/ds_a1b2c3d4e5f6/artifact?split=train&format=npz",
        "test": "/v1/datasets/ds_a1b2c3d4e5f6/artifact?split=test&format=npz",
        "meta": "/v1/datasets/ds_a1b2c3d4e5f6"
    }
}
```

#### GET /v1/datasets/{id}/artifact - Download Artifact

**Query Parameters:**

- `format`: `npz` | `json` | `torch` | `hdf5` (default: original format)
- `split`: `full` | `train` | `test` (default: `full`)

**Response (200 OK):**

- `Content-Type: application/octet-stream` for binary formats
- `Content-Type: application/json` for JSON format
- `Content-Disposition: attachment; filename="ds_xxx_train.npz"`

#### GET /v1/generators - List Generators

**Response (200 OK):**

```json
{
    "generators": [
        {
            "name": "spiral",
            "description": "Generates N-spiral classification datasets",
            "schema_url": "/v1/generators/spiral/schema"
        },
        {
            "name": "xor",
            "description": "Generates XOR classification problem",
            "schema_url": "/v1/generators/xor/schema"
        },
        {
            "name": "circles",
            "description": "Generates concentric circles classification",
            "schema_url": "/v1/generators/circles/schema"
        }
    ]
}
```

---

## 6. Implementation Plan

### Phase 0: Inventory & Contract Definition (S - 2-4 hours)

**Goal:** Establish a clear contract between Juniper Data, Canopy, and Cascor.

#### Step 0.1: Enumerate Canopy Dataset Expectations

**Description:** Run comprehensive search in Canopy codebase to identify all dataset usage patterns.

**Actions:**

```bash
cd juniper_canopy/src
rg -n "spiral|dataset|features|labels|inputs|targets" --type py
rg -n "num_samples|num_features|num_classes" --type py
```

**Deliverable:** Table of expected fields and data types.

#### Step 0.2: Enumerate Cascor Dataset Expectations

**Description:** Verify what the CCN training pipeline expects.

**Actions:**

1. Review `CascadeCorrelationNetwork.fit()` signature
2. Review `SpiralProblem.evaluate()` data flow
3. Document tensor types (torch.float32), shapes, and device expectations

**Deliverable:** Cascor consumer interface specification.

#### Step 0.3: Define Dataset Contract Document

**Description:** Create formal specification document.

**Deliverable:** `DATASET_CONTRACT.md` with:

- Input format specifications
- Output format specifications  
- Metadata schema
- Conversion utilities specification

---

### Phase 1: Core Generator Extraction (M - 4-8 hours)

**Goal:** Extract pure spiral generation logic into standalone, testable module.

#### Step 1.1: Create Juniper Data Project Structure

**Description:** Initialize the new project with proper Python packaging.

**Actions:**

1. Create directory structure (see Section 4.1)
2. Create `pyproject.toml` (see Section 3.2)
3. Create `README.md` with project overview
4. Create `AGENTS.md` with development guidelines

**Deliverable:** Empty but properly configured Python project.

#### Step 1.2: Implement Base Generator Interface

**Description:** Create abstract base class for all generators.

**File:** `src/juniper_data/generators/base.py`

**Actions:**

1. Define `BaseGenerator` abstract class
2. Define parameter validation interface
3. Define output format interface

**Deliverable:** Working base class with type hints.

#### Step 1.3: Implement Spiral Generator

**Description:** Port spiral generation logic from Cascor test generators.

**File:** `src/juniper_data/generators/spiral.py`

**Source Reference:** Use `juniper_cascor/src/tests/unit/test_data/generators.py::SpiralDataGenerator` as primary reference (cleaner than SpiralProblem class).

**Actions:**

1. Implement `SpiralGenerator` class
2. Implement `SpiralParams` Pydantic model
3. Add parameter validation
4. Add comprehensive docstrings

**Key Differences from Cascor `SpiralProblem`:**

- No CCN network dependencies
- No custom logging system
- No multiprocessing configuration
- Returns numpy arrays (not torch tensors by default)
- Pure functions with no side effects

**Deliverable:** Working spiral generator with tests.

#### Step 1.4: Implement Additional Generators

**Description:** Port XOR, circles, and regression generators.

**Files:**

- `src/juniper_data/generators/classification.py`
- `src/juniper_data/generators/regression.py`

**Deliverable:** Complete generator suite.

#### Step 1.5: Add Generator Unit Tests

**Description:** Comprehensive test coverage for generators.

**File:** `tests/unit/test_generators.py`

**Test Cases:**

- Parameter validation (valid/invalid)
- Output shape verification
- Output type verification
- Reproducibility (same seed → same output)
- Edge cases (minimum/maximum values)

**Deliverable:** >90% coverage for generator modules.

---

### Phase 2: REST API Service (M - 4-8 hours)

**Goal:** Build minimal FastAPI service exposing generators via REST.

#### Step 2.1: Create FastAPI Application

**Description:** Set up FastAPI with proper configuration.

**File:** `src/juniper_data/api/app.py`

**Actions:**

1. Create FastAPI app instance
2. Configure CORS
3. Configure OpenAPI documentation
4. Add exception handlers
5. Add logging middleware

**Deliverable:** Running FastAPI application (no routes yet).

#### Step 2.2: Implement Generator Registry

**Description:** Dynamic registration of available generators.

**File:** `src/juniper_data/core/registry.py`

**Actions:**

1. Create singleton registry
2. Implement auto-discovery of generators
3. Add schema generation from Pydantic models

**Deliverable:** Working registry with spiral generator registered.

#### Step 2.3: Implement Local Storage Backend

**Description:** Simple filesystem storage for dataset artifacts.

**File:** `src/juniper_data/storage/local.py`

**Actions:**

1. Implement save/load for npz format
2. Implement save/load for JSON format
3. Add dataset ID generation (hash of params)
4. Add cache directory management

**Deliverable:** Working local storage with configurable path.

#### Step 2.4: Implement API Routes

**Description:** Create all REST endpoints.

**Files:**

- `src/juniper_data/api/routes/health.py`
- `src/juniper_data/api/routes/generators.py`
- `src/juniper_data/api/routes/datasets.py`

**Actions:**

1. `GET /v1/health` - Health check
2. `GET /v1/generators` - List generators
3. `GET /v1/generators/{name}/schema` - Parameter schema
4. `POST /v1/datasets` - Create dataset
5. `GET /v1/datasets/{id}` - Get metadata
6. `GET /v1/datasets/{id}/artifact` - Download artifact
7. `DELETE /v1/datasets/{id}` - Delete dataset

**Deliverable:** Complete API with OpenAPI documentation.

#### Step 2.5: Add API Integration Tests

**Description:** End-to-end tests for API endpoints.

**File:** `tests/integration/test_api.py`

**Test Cases:**

- Generator listing
- Schema retrieval
- Dataset creation (happy path)
- Dataset retrieval
- Error handling (invalid params, not found)

**Deliverable:** Passing integration tests.

#### Step 2.6: Create Dockerfile and Docker Compose

**Description:** Containerize the service.

**Files:**

- `Dockerfile`
- `docker-compose.yml`

**Deliverable:** Runnable Docker container.

---

### Phase 3: Canopy/Cascor Integration (L - 1-2 days)

**Goal:** Replace existing spiral generators with Juniper Data API calls.

#### Step 3.1: Create Juniper Data Client Library

**Description:** Shared client for consuming Juniper Data API.

**Location:** Either in Juniper Data package or separate `juniper-data-client` package.

**File:** `src/juniper_data/client/client.py` (or separate package)

```python
class JuniperDataClient:
    """Client for Juniper Data REST API."""

    def __init__(self, base_url: str = "http://localhost:8100"):
        self.base_url = base_url

    async def generate_spiral(
        self,
        n_spirals: int = 2,
        n_points_per_spiral: int = 100,
        noise: float = 0.1,
        seed: int = 42,
        **kwargs
    ) -> DatasetResponse:
        """Generate spiral dataset."""
        pass

    async def get_dataset(self, dataset_id: str) -> DatasetArtifact:
        """Retrieve dataset by ID."""
        pass

    def get_dataset_sync(self, dataset_id: str) -> DatasetArtifact:
        """Synchronous version for non-async contexts."""
        pass
```

**Deliverable:** Working client library with sync/async support.

#### Step 3.2: Update Canopy DemoMode

**Description:** Replace `_generate_spiral_dataset()` with API call.

**File:** `juniper_canopy/src/demo_mode.py`

**Actions:**

1. Add `JuniperDataClient` import
2. Add configuration for Juniper Data URL
3. Replace `_generate_spiral_dataset()` implementation
4. Add fallback to local generation if API unavailable

**Example:**

```python
def _generate_spiral_dataset(self, n_samples: int = 200) -> Dict[str, Any]:
    """Generate spiral dataset via Juniper Data API."""
    try:
        client = JuniperDataClient(base_url=self.config.juniper_data_url)
        response = client.generate_spiral_sync(
            n_spirals=2,
            n_points_per_spiral=n_samples // 2,
            noise=0.1,
            seed=42
        )
        return self._convert_response_to_dict(response)
    except Exception as e:
        self.logger.warning(f"Juniper Data unavailable, using local fallback: {e}")
        return self._generate_spiral_dataset_local(n_samples)
```

**Deliverable:** Updated DemoMode with API integration.

#### Step 3.3: Update Canopy CascorIntegration

**Description:** Replace `_generate_missing_dataset_info()` with API call.

**File:** `juniper_canopy/src/backend/cascor_integration.py`

**Actions:**

1. Similar pattern to DemoMode
2. Ensure return schema matches existing expectations

**Deliverable:** Updated CascorIntegration with API integration.

#### Step 3.4: Update Cascor SpiralProblem (Optional)

**Description:** Optionally refactor Cascor to use Juniper Data for dataset generation.

**File:** `juniper_cascor/src/spiral_problem/spiral_problem.py`

**Decision Point:** This may not be necessary if Cascor continues to use its own dataset generation for training. The primary benefit is consistency across the ecosystem.

**Deliverable:** Decision document on Cascor integration strategy.

#### Step 3.5: Add Configuration for Juniper Data URL

**Description:** Add environment variable and config file support.

**Canopy Configuration:**

```yaml
# conf/app_config.yaml
juniper_data:
  url: "http://localhost:8100"
  timeout_seconds: 30
  fallback_enabled: true
```

**Environment Variable:**

- `JUNIPER_DATA_URL=http://localhost:8100`

**Deliverable:** Working configuration in both applications.

#### Step 3.6: Integration Testing

**Description:** End-to-end tests with all three applications.

**Test Scenarios:**

1. Canopy starts → calls Juniper Data → displays spiral dataset
2. Juniper Data unavailable → Canopy falls back to local generation
3. Cascor training uses dataset from Juniper Data

**Deliverable:** Passing integration tests.

---

### Phase 4: Provider Interfaces (L - 1-2 days)

**Goal:** Add extensibility for multiple data sources.

#### Step 4.1: Define Source Provider Interface

**Description:** Abstract base class for all source types.

**File:** `src/juniper_data/sources/base.py`

**Deliverable:** Working interface definition.

#### Step 4.2: Implement Generated Source Provider

**Description:** Wrap generators in provider interface.

**File:** `src/juniper_data/sources/generated.py`

**Deliverable:** Working generated source provider.

#### Step 4.3: Implement Local Filesystem Provider

**Description:** Load datasets from local paths.

**File:** `src/juniper_data/sources/local_fs.py`

**Features:**

- Load from npz, JSON, CSV, HDF5
- Validate file format
- Extract metadata from files

**Deliverable:** Working local FS provider.

#### Step 4.4: Implement URL Download Provider

**Description:** Download datasets from URLs.

**File:** `src/juniper_data/sources/url.py`

**Features:**

- HTTP/HTTPS download
- Format detection
- Caching of downloaded files

**Deliverable:** Working URL provider.

---

### Phase 5: Remote/DB Backends (XL - Staged)

**Goal:** Add enterprise data source integrations.

#### Step 5.1: S3 Provider (1-2 days)

**File:** `src/juniper_data/sources/s3.py`

**Features:**

- Boto3 integration
- Credential management (env vars, IAM)
- Bucket/key path resolution

#### Step 5.2: Google Cloud Storage Provider (1-2 days)

**File:** `src/juniper_data/sources/gcs.py`

**Features:**

- google-cloud-storage integration
- Service account authentication

#### Step 5.3: Hugging Face Provider (2-3 days)

**File:** `src/juniper_data/sources/huggingface.py`

**Features:**

- datasets library integration
- Dataset/config/split selection
- Streaming for large datasets

#### Step 5.4: Kaggle Provider (2-3 days)

**File:** `src/juniper_data/sources/kaggle.py`

**Features:**

- Kaggle API authentication
- Dataset/competition download
- File extraction

#### Step 5.5: Database Provider (2-3 days)

**File:** `src/juniper_data/sources/database.py`

**Features:**

- SQLAlchemy integration
- Query-based dataset loading
- Support for PostgreSQL, MySQL, SQLite

---

## 7. Integration Strategy

### 7.1 Deployment Architecture

```bash
┌─────────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    Docker Compose Stack                     │   │
│   │                                                             │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │   │
│   │   │ Juniper Data │  │ Juniper      │  │ Juniper      │      │   │
│   │   │    :8100     │  │ Canopy       │  │ Cascor       │      │   │
│   │   │              │  │    :8050     │  │              │      │   │
│   │   │  FastAPI     │  │  Dash/       │  │  Training    │      │   │
│   │   │  REST API    │  │  FastAPI     │  │  Backend     │      │   │
│   │   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │   │
│   │          │                 │                 │              │   │
│   │          └─────────────────┼─────────────────┘              │   │
│   │                            │                                │   │
│   │   ┌────────────────────────▼────────────────────────────┐   │   │
│   │   │              Shared Data Volume                     │   │   │
│   │   │            /data/juniper_data/                      │   │   │
│   │   └─────────────────────────────────────────────────────┘   │   │
│   │                                                             │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Service Discovery

**Option A: Environment Variables:**

```bash
export JUNIPER_DATA_URL=http://juniper-data:8100
export JUNIPER_CANOPY_URL=http://juniper-canopy:8050
```

**Option B: Docker Compose DNS:**

```yaml
services:
  juniper-data:
    hostname: juniper-data

  juniper-canopy:
    environment:
      - JUNIPER_DATA_URL=http://juniper-data:8100
```

### 7.3 Backward Compatibility

To maintain backward compatibility during migration:

1. **Feature Flag:** Add `JUNIPER_DATA_ENABLED=true/false` to control API usage
2. **Fallback Logic:** If Juniper Data is unavailable, fall back to local generation
3. **Gradual Rollout:** Enable Juniper Data in development first, then staging, then production

---

## 8. Risk Assessment & Mitigations

### 8.1 Technical Risks

| Risk                                          | Impact | Likelihood | Mitigation                               |
| --------------------------------------------- | ------ | ---------- | ---------------------------------------- |
| API latency impacts Canopy performance        | Medium | Medium     | Local caching, async loading             |
| Breaking changes to dataset schema            | High   | Low        | Versioned API, contract tests            |
| Juniper Data service unavailable              | Medium | Medium     | Fallback to local generation             |
| Inconsistent dataset formats across consumers | High   | Medium     | Strict schema validation, contract tests |

### 8.2 Project Risks

| Risk                                   | Impact | Likelihood | Mitigation                               |
| -------------------------------------- | ------ | ---------- | ---------------------------------------- |
| Scope creep (too many sources at once) | Medium | High       | Phase 5 explicitly deferred              |
| Integration complexity underestimated  | Medium | Medium     | Phased approach, integration tests early |
| Documentation lag                      | Low    | High       | Document as you go, AGENTS.md updates    |

### 8.3 Security Considerations

1. **Path Traversal:** Disable arbitrary filesystem path access by default
2. **URL Fetching:** Whitelist allowed domains for URL source
3. **Resource Limits:** Limit max dataset size, request rate limiting
4. **Authentication:** Consider API key for production (future phase)

---

## 9. Appendices

### Appendix A: Cascor Constants Referenced by Spiral Problem

```python
# From cascor_constants/constants_problem/constants_problem.py

_SPIRAL_PROBLEM_ACTIVATION_FUNCTION = "tanh"
_SPIRAL_PROBLEM_CANDIDATE_DISPLAY_FREQUENCY = 10
_SPIRAL_PROBLEM_CANDIDATE_EPOCHS = 50
_SPIRAL_PROBLEM_CANDIDATE_LEARNING_RATE = 0.5
_SPIRAL_PROBLEM_CANDIDATE_POOL_SIZE = 16
_SPIRAL_PROBLEM_CLOCKWISE = True
_SPIRAL_PROBLEM_CORRELATION_THRESHOLD = 0.001
_SPIRAL_PROBLEM_DEFAULT_ORIGIN = 0.0
_SPIRAL_PROBLEM_DEFAULT_RADIUS = 1.0
_SPIRAL_PROBLEM_DISPLAY_FREQUENCY = 10
_SPIRAL_PROBLEM_DISTRIBUTION_FACTOR = 360
_SPIRAL_PROBLEM_EPOCH_DISPLAY_FREQUENCY = 50
_SPIRAL_PROBLEM_EPOCHS_MAX = 200
_SPIRAL_PROBLEM_GENERATE_PLOTS_DEFAULT = False
_SPIRAL_PROBLEM_INPUT_SIZE = 2
_SPIRAL_PROBLEM_LEARNING_RATE = 0.5
_SPIRAL_PROBLEM_MAX_HIDDEN_UNITS = 12
_SPIRAL_PROBLEM_MAX_NEW = 1.0
_SPIRAL_PROBLEM_MAX_ORIG = 1.0
_SPIRAL_PROBLEM_MIN_NEW = 0.0
_SPIRAL_PROBLEM_MIN_ORIG = 0.0
_SPIRAL_PROBLEM_NOISE_FACTOR_DEFAULT = 0.1
_SPIRAL_PROBLEM_NUM_ROTATIONS = 2
_SPIRAL_PROBLEM_NUM_SPIRALS = 2
_SPIRAL_PROBLEM_NUMBER_POINTS_PER_SPIRAL = 97
_SPIRAL_PROBLEM_OUTPUT_EPOCHS = 100
_SPIRAL_PROBLEM_OUTPUT_SIZE = 2
_SPIRAL_PROBLEM_PATIENCE = 10
_SPIRAL_PROBLEM_RANDOM_SEED = 42
_SPIRAL_PROBLEM_RANDOM_VALUE_SCALE = 0.1
_SPIRAL_PROBLEM_STATUS_DISPLAY_FREQUENCY = 50
_SPIRAL_PROBLEM_TEST_RATIO = 0.2
_SPIRAL_PROBLEM_TRAIN_RATIO = 0.8
```

### Appendix B: Dataset Return Schema Comparison

| Field                | DemoMode     | CascorIntegration | Proposed        |
| -------------------- | ------------ | ----------------- | --------------- |
| `inputs`             | np.ndarray   | ❌                | ❌ (use X)      |
| `targets`            | np.ndarray   | ❌                | ❌ (use y)      |
| `inputs_tensor`      | torch.Tensor | ❌                | Optional        |
| `targets_tensor`     | torch.Tensor | ❌                | Optional        |
| `features`           | ❌           | List              | ❌ (use X)      |
| `labels`             | ❌           | List              | ❌ (use y)      |
| `X`                  | ❌           | ❌                | np.ndarray      |
| `y`                  | ❌           | ❌                | np.ndarray      |
| `num_samples`        | ✓            | ✓                 | `n_samples`     |
| `num_features`       | ✓            | ✓                 | `n_features`    |
| `num_classes`        | ✓            | ✓                 | `n_classes`     |
| `class_distribution` | ❌           | Dict              | Dict            |
| `dataset_name`       | ❌           | ✓                 | `name`          |
| `mock_mode`          | ❌           | ✓                 | ❌ (use source) |

### Appendix C: Commands Reference

```bash
# Juniper Data Development
cd juniper_data
pip install -e ".[dev]"
uvicorn juniper_data.main:app --reload --port 8100

# Run Tests
pytest tests/ -v --cov=juniper_data

# Docker Build
docker build -t juniper-data:latest .
docker-compose up -d

# API Testing
curl http://localhost:8100/v1/health
curl http://localhost:8100/v1/generators
curl -X POST http://localhost:8100/v1/datasets \
  -H "Content-Type: application/json" \
  -d '{"source":"generated","generator":"spiral","params":{"n_spirals":2}}'
```

---

## Document History

| Date       | Version | Author           | Changes       |
| ---------- | ------- | ---------------- | ------------- |
| 2026-01-29 | 1.0.0   | Development Team | Initial draft |

---

**Next Steps:**

1. Review and approve this plan
2. Create Juniper Data repository
3. Begin Phase 0: Inventory & Contract Definition
