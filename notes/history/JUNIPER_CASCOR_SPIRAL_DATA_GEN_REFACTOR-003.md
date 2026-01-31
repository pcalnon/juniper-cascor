# Juniper Cascor Spiral Data Generator Extraction Plan

## JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR.md

**Document Version**: 1.0.0  
**Date Created**: 2026-01-29  
**Last Modified**: 2026-01-29  
**Author**: Juniper Development Team  
**Status**: DRAFT - Pending Approval

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Target Architecture](#3-target-architecture)
4. [Phase 1: Analysis and Preparation](#phase-1-analysis-and-preparation)
5. [Phase 2: Core Library Extraction](#phase-2-core-library-extraction)
6. [Phase 3: REST API Implementation](#phase-3-rest-api-implementation)
7. [Phase 4: Integration and Migration](#phase-4-integration-and-migration)
8. [Phase 5: Extended Data Source Support](#phase-5-extended-data-source-support)
9. [Testing Strategy](#testing-strategy)
10. [Risk Assessment and Mitigation](#risk-assessment-and-mitigation)
11. [Timeline and Effort Estimates](#timeline-and-effort-estimates)
12. [Appendices](#appendices)

---

## 1. Executive Summary

### 1.1 Objective

Extract the Spiral Problem Data Generator code from Juniper Cascor into a standalone application called **"Juniper Data"**. This new application will provide a REST API for dataset generation, enabling:

- **POST**: Submit spiral generation parameters and receive generated datasets
- **GET**: Retrieve previously generated or stored datasets
- **Multiple data sources**: Local generation, file storage, remote sources, databases, and external URLs

### 1.2 Scope

| In Scope                                | Out of Scope                          |
| --------------------------------------- | ------------------------------------- |
| Spiral dataset generation logic         | Neural network training code          |
| Dataset partitioning (train/test)       | CascadeCorrelationNetwork integration |
| Data format conversion (NumPy, PyTorch) | Candidate training infrastructure     |
| REST API with FastAPI                   | Real-time training monitoring         |
| Storage abstraction layer               | Model serialization (HDF5 snapshots)  |
| Multi-source data retrieval             | GPU/CUDA integration                  |

### 1.3 Key Deliverables

1. **Juniper Data Core Library**: Pure Python data generation module
2. **Juniper Data REST API**: FastAPI-based service for dataset management
3. **Storage Abstraction Layer**: Pluggable storage backends
4. **Juniper Cascor Integration**: Updated SpiralProblem to consume new library
5. **Comprehensive Test Suite**: Unit and integration tests

### 1.4 Estimated Effort

| Phase                            | Effort            | Duration     |
| -------------------------------- | ----------------- | ------------ |
| Phase 1: Analysis & Preparation  | S (0.5 day)       | 0.5 days     |
| Phase 2: Core Library Extraction | M (1 day)         | 1 day        |
| Phase 3: REST API Implementation | M (1 day)         | 1 day        |
| Phase 4: Integration & Migration | M (1 day)         | 1 day        |
| Phase 5: Extended Data Sources   | L (2 days)        | 2 days       |
| **Total**                        | **XL (5.5 days)** | **5.5 days** |

---

## 2. Current State Analysis

### 2.1 Code References to Spiral Data Generator

#### 2.1.1 Primary Source: `src/spiral_problem/spiral_problem.py`

The `SpiralProblem` class contains **mixed concerns**:

**Data Generation Methods (TO BE EXTRACTED):**

| Method                              | Lines | Description                            |
| ----------------------------------- | ----- | -------------------------------------- |
| `generate_n_spiral_dataset()`       | ~80   | Main public API for dataset generation |
| `generate_raw_spiral_coordinates()` | ~50   | Core spiral coordinate generation      |
| `_generate_xy_coordinates()`        | ~20   | Helper for x/y coordinate pairs        |
| `_make_coords()`                    | ~30   | Trigonometric coordinate calculation   |
| `_make_noise()`                     | ~10   | Noise generation for data points       |
| `_create_input_features()`          | ~15   | Stack coordinates into feature matrix  |
| `_create_one_hot_targets()`         | ~20   | One-hot encoding for class labels      |
| `_create_spiral_dataset()`          | ~15   | Combine features and targets           |
| `_convert_to_tensors()`             | ~15   | NumPy to PyTorch conversion            |
| `_shuffle_dataset()`                | ~20   | Random shuffling of data               |
| `_partition_dataset()`              | ~30   | Split into train/test sets             |
| `_split_dataset()`                  | ~40   | Core splitting logic                   |
| `_find_partition_index_end()`       | ~15   | Index calculation helper               |
| `_dataset_split_index_end()`        | ~15   | Split ratio to index conversion        |

**Training Methods (TO REMAIN in Cascor):**

| Method                     | Reason to Keep                  |
| -------------------------- | ------------------------------- |
| `solve_n_spiral_problem()` | Uses CascadeCorrelationNetwork  |
| `evaluate()`               | Training evaluation logic       |
| `__init__()`               | Network configuration           |
| Plotting methods           | Visualization of trained models |

#### 2.1.2 Secondary Source: `src/tests/unit/test_data/generators.py`

Contains a simplified `SpiralDataGenerator` class for testing:

```python
class SpiralDataGenerator:
    @staticmethod
    def generate_2_spiral(n_per_spiral=100, noise=0.1, seed=42)

    @staticmethod
    def generate_n_spiral(n_spirals=3, n_per_spiral=50, noise=0.05, seed=42)
```

This serves as a **reference implementation** for the new library.

#### 2.1.3 Test Fixtures: `src/tests/conftest.py`

Inline spiral data generation in fixtures:

```python
@pytest.fixture
def spiral_2d_data() -> Tuple[torch.Tensor, torch.Tensor]

@pytest.fixture
def n_spiral_data() -> callable
```

### 2.2 Dependency Analysis

#### 2.2.1 Constants Dependencies

**File**: `src/cascor_constants/constants_problem/constants_problem.py`

| Constant                                   | Current Value | Required for Data Gen |
| ------------------------------------------ | ------------- | --------------------- |
| `_SPIRAL_PROBLEM_NUM_SPIRALS`              | 2             | ✅ Yes                |
| `_SPIRAL_PROBLEM_NUMBER_POINTS_PER_SPIRAL` | 97            | ✅ Yes                |
| `_SPIRAL_PROBLEM_NUM_ROTATIONS`            | 3             | ✅ Yes                |
| `_SPIRAL_PROBLEM_CLOCKWISE`                | True          | ✅ Yes                |
| `_SPIRAL_PROBLEM_NOISE_FACTOR_DEFAULT`     | 0.25          | ✅ Yes                |
| `_SPIRAL_PROBLEM_DISTRIBUTION_FACTOR`      | 0.80          | ✅ Yes                |
| `_SPIRAL_PROBLEM_DEFAULT_ORIGIN`           | 0.0           | ✅ Yes                |
| `_SPIRAL_PROBLEM_DEFAULT_RADIUS`           | 10.0          | ✅ Yes                |
| `_SPIRAL_PROBLEM_TRAIN_RATIO`              | 0.8           | ✅ Yes                |
| `_SPIRAL_PROBLEM_TEST_RATIO`               | 0.2           | ✅ Yes                |
| `_SPIRAL_PROBLEM_RANDOM_VALUE_SCALE`       | 0.1           | ✅ Yes                |
| `_SPIRAL_PROBLEM_ORIG_POINTS`              | 100           | ⚠️ Maybe              |
| `_SPIRAL_PROBLEM_MIN_NEW/MAX_NEW`          | 50/150        | ⚠️ Maybe              |
| `_SPIRAL_PROBLEM_INPUT_SIZE`               | 2             | ❌ Derived            |
| `_SPIRAL_PROBLEM_OUTPUT_SIZE`              | n_spirals     | ❌ Derived            |

**Logging Constants (NOT needed for extraction):**

- `_SPIRAL_PROBLEM_LOG_*` constants
- Custom log levels (TRACE, VERBOSE)
- LogConfig integration

#### 2.2.2 Import Dependencies in `spiral_problem.py`

```python
# Standard Library
import logging, logging.config, numpy, os, matplotlib.pyplot, random, torch, uuid, multiprocessing

# Juniper Cascor (TO BE DECOUPLED)
from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
from cascor_constants.constants import (...)  # 50+ constants
from log_config.log_config import LogConfig
from log_config.logger.logger import Logger
```

#### 2.2.3 Third-Party Dependencies

| Library      | Version | Required      | Purpose                         |
| ------------ | ------- | ------------- | ------------------------------- |
| `numpy`      | ≥1.24   | ✅ Required   | Core array operations           |
| `torch`      | ≥2.0    | ⚠️ Optional   | Tensor conversion               |
| `matplotlib` | ≥3.7    | ❌ Not needed | Visualization (stays in Cascor) |
| `h5py`       | ≥3.9    | ❌ Not needed | Serialization (stays in Cascor) |

### 2.3 Environment Configuration

#### 2.3.1 Current Configuration Management

Juniper Cascor uses a **hierarchical constants system**:

```bash
src/cascor_constants/
├── constants.py                    # Main aggregator (re-exports all)
├── constants_activation/           # Activation functions
├── constants_candidates/           # Candidate training
├── constants_hdf5/                 # Serialization
├── constants_logging/              # Logging configuration
├── constants_model/                # Model architecture
└── constants_problem/              # Spiral problem (TARGET)
    └── constants_problem.py
```

#### 2.3.2 Proposed Configuration for Juniper Data

**Environment Variables:**

| Variable                    | Description               | Default           |
| --------------------------- | ------------------------- | ----------------- |
| `JUNIPER_DATA_LOG_LEVEL`    | Logging level             | `INFO`            |
| `JUNIPER_DATA_STORAGE_PATH` | Dataset storage directory | `./data/datasets` |
| `JUNIPER_DATA_HOST`         | API server host           | `0.0.0.0`         |
| `JUNIPER_DATA_PORT`         | API server port           | `8000`            |
| `JUNIPER_DATA_WORKERS`      | Uvicorn workers           | `1`               |

**Configuration File** (`juniper_data/config.py`):

```python
from dataclasses import dataclass, field
from typing import Optional
import os

@dataclass
class JuniperDataConfig:
    # Spiral Generation Defaults
    default_n_spirals: int = 2
    default_n_points: int = 97
    default_n_rotations: int = 3
    default_clockwise: bool = True
    default_noise: float = 0.25
    default_distribution: float = 0.80
    default_origin: float = 0.0
    default_radius: float = 10.0
    default_train_ratio: float = 0.8
    default_test_ratio: float = 0.2

    # Storage
    storage_path: str = field(default_factory=lambda: os.environ.get(
        "JUNIPER_DATA_STORAGE_PATH", "./data/datasets"
    ))

    # API Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Logging
    log_level: str = field(default_factory=lambda: os.environ.get(
        "JUNIPER_DATA_LOG_LEVEL", "INFO"
    ))
```

---

## 3. Target Architecture

### 3.1 High-Level Architecture Diagram

```bash
┌─────────────────────────────────────────────────────────────────────────┐
│                              Juniper Ecosystem                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────┐              ┌─────────────────────────────────┐  │
│   │  Juniper Cascor │              │         Juniper Data            │  │
│   │  (Training App) │              │      (Dataset Service)          │  │
│   │                 │              │                                 │  │
│   │  ┌───────────┐  │   Library    │  ┌──────────────────────────┐   │  │
│   │  │  Spiral   │◄─┼──────────────┼──│   Generator Library      │   │  │
│   │  │  Problem  │  │   Import     │  │   (spiral, circles, etc) │   │  │
│   │  └───────────┘  │              │  └──────────────────────────┘   │  │
│   │       │         │              │              │                  │  │
│   │       ▼         │              │              ▼                  │  │
│   │  ┌───────────┐  │              │  ┌──────────────────────────┐   │  │
│   │  │  Cascade  │  │   REST API   │  │      REST API Layer      │   │  │
│   │  │Correlation│◄─┼──────────────┼──│  POST /datasets          │   │  │
│   │  │  Network  │  │  (Optional)  │  │  GET  /datasets/{id}     │   │  │
│   │  └───────────┘  │              │  └──────────────────────────┘   │  │
│   │                 │              │              │                  │  │
│   └─────────────────┘              │              ▼                  │  │
│                                    │  ┌──────────────────────────┐   │  │
│   ┌─────────────────┐              │  │    Storage Abstraction   │   │  │
│   │ Juniper Canopy  │              │  │                          │   │  │
│   │  (Frontend UI)  │              │  │  ┌──────┐ ┌──────┐       │   │  │
│   │                 │   REST API   │  │  │Memory│ │ File │       │   │  │
│   │  ┌───────────┐  │◄─────────────┼──│  └──────┘ └──────┘       │   │  │
│   │  │  Dataset  │  │              │  │  ┌──────┐ ┌──────┐       │   │  │
│   │  │  Browser  │  │              │  │  │  DB  │ │Remote│       │   │  │
│   │  └───────────┘  │              │  │  └──────┘ └──────┘       │   │  │
│   │                 │              │  └──────────────────────────┘   │  │
│   └─────────────────┘              └─────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Juniper Data Project Structure

```bash
JuniperData/
├── juniper_data/
│   ├── __init__.py
│   ├── config.py                    # Configuration management
│   ├── defaults.py                  # Default parameter values
│   │
│   ├── generator/                   # Core data generation library
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract base generator
│   │   ├── spiral.py                # Spiral dataset generator
│   │   ├── circles.py               # Concentric circles (future)
│   │   ├── xor.py                   # XOR problem (future)
│   │   └── regression.py            # Regression datasets (future)
│   │
│   ├── models/                      # Data models and schemas
│   │   ├── __init__.py
│   │   ├── dataset.py               # Dataset dataclass
│   │   ├── params.py                # Parameter dataclasses
│   │   └── responses.py             # API response models
│   │
│   ├── storage/                     # Storage abstraction layer
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract storage interface
│   │   ├── memory.py                # In-memory store
│   │   ├── filesystem.py            # File-based storage
│   │   ├── database.py              # SQL/NoSQL (future)
│   │   └── remote.py                # S3/GCS (future)
│   │
│   ├── sources/                     # Data source handlers
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract source interface
│   │   ├── local_generator.py       # Local generation
│   │   ├── file_source.py           # Load from file
│   │   ├── url_source.py            # Download from URL
│   │   └── registry.py              # Source registry
│   │
│   └── api/                         # REST API layer
│       ├── __init__.py
│       ├── main.py                  # FastAPI application
│       ├── routes/
│       │   ├── __init__.py
│       │   ├── datasets.py          # /datasets endpoints
│       │   └── health.py            # /health endpoint
│       └── middleware/
│           ├── __init__.py
│           └── error_handler.py
│
├── tests/
│   ├── unit/
│   │   ├── test_spiral_generator.py
│   │   ├── test_storage.py
│   │   └── test_models.py
│   ├── integration/
│   │   ├── test_api.py
│   │   └── test_sources.py
│   └── conftest.py
│
├── pyproject.toml
├── README.md
├── AGENTS.md
└── CHANGELOG.md
```

### 3.3 Data Flow Diagrams

#### 3.3.1 Dataset Generation Flow

```bash
┌────────────────────────────────────────────────────────────────────────┐
│                    POST /datasets (Generate Spiral)                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Client Request                                                        │
│       │                                                                │
│       ▼                                                                │
│  ┌────────────────┐     ┌─────────────────┐     ┌─────────────────┐    │
│  │  Validate      │     │  Select Source  │     │  Generate Data  │    │
│  │  Parameters    │────►│  (Generator)    │────►│  (NumPy Arrays) │    │
│  └────────────────┘     └─────────────────┘     └─────────────────┘    │
│                                                          │             │
│                                                          ▼             │
│  ┌────────────────┐     ┌─────────────────┐     ┌─────────────────┐    │
│  │  Return        │     │  Store Dataset  │     │  Create Dataset │    │
│  │  Response      │◄────│  (If Persist)   │◄────│  Object         │    │
│  └────────────────┘     └─────────────────┘     └─────────────────┘    │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

#### 3.3.2 Dataset Retrieval Flow

```bash
┌────────────────────────────────────────────────────────────────────────┐
│                       GET /datasets/{id}                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Client Request                                                        │
│       │                                                                │
│       ▼                                                                │
│  ┌────────────────┐     ┌─────────────────┐                            │
│  │  Parse         │     │  Lookup in      │                            │
│  │  Dataset ID    │────►│  Storage        │                            │
│  └────────────────┘     └─────────────────┘                            │
│                                │                                       │
│                    ┌───────────┴───────────┐                           │
│                    ▼                       ▼                           │
│           ┌─────────────┐         ┌──────────────┐                     │
│           │  Not Found  │         │    Found     │                     │
│           │  (404)      │         │              │                     │
│           └─────────────┘         └──────────────┘                     │
│                                           │                            │
│                         ┌─────────────────┼─────────────────┐          │
│                         ▼                 ▼                 ▼          │
│                  ┌───────────┐     ┌───────────┐     ┌───────────┐     │
│                  │  Metadata │     │  JSON     │     │  NPZ/PT   │     │
│                  │  Only     │     │  Data     │     │  Download │     │
│                  └───────────┘     └───────────┘     └───────────┘     │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Analysis and Preparation

**Goal**: Complete codebase analysis and establish extraction boundaries  
**Duration**: 0.5 days  
**Priority**: P0 - Must Complete First

### Step 1.1: Create Extraction Mapping Document

**Description**: Document every method, constant, and dependency that will be extracted, kept, or modified.

**Deliverables**:

- Complete method-by-method mapping
- Dependency graph
- Import statement analysis

**Actions**:

1. Generate call graph for `SpiralProblem` class
2. Identify all internal method calls
3. Map each method to: EXTRACT, KEEP, or MODIFY

### Step 1.2: Define Data Contracts

**Description**: Create formal specifications for dataset structure and API contracts.

**Deliverables**:

- `SpiralDatasetParams` dataclass specification
- `SpiralDataset` dataclass specification
- OpenAPI schema draft

**Example Contracts**:

```python
# params.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class SpiralDatasetParams:
    """Parameters for spiral dataset generation."""
    n_spirals: int = 2
    n_points_per_spiral: int = 97
    n_rotations: int = 3
    clockwise: bool = True
    noise: float = 0.25
    distribution: float = 0.80
    origin: float = 0.0
    radius: float = 10.0
    train_ratio: float = 0.8
    test_ratio: float = 0.2
    random_seed: Optional[int] = 42
```

```python
# dataset.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
import numpy as np

@dataclass
class SpiralDataset:
    """Represents a generated spiral dataset."""
    dataset_id: str
    params: SpiralDatasetParams
    X: np.ndarray                    # Shape: (N, 2)
    y: np.ndarray                    # Shape: (N, n_spirals) one-hot
    train_indices: np.ndarray        # Indices for training split
    test_indices: np.ndarray         # Indices for test split
    created_at: datetime
    metadata: Optional[Dict] = None
```

### Step 1.3: Set Up Juniper Data Repository Structure

**Description**: Create the new JuniperData project with proper structure.

**Actions**:

1. Create directory structure (see Section 3.2)
2. Initialize `pyproject.toml` with dependencies
3. Create README.md and AGENTS.md
4. Set up pytest configuration
5. Configure CI/CD basics

**pyproject.toml (Initial)**:

```toml
[project]
name = "juniper-data"
version = "0.1.0"
description = "Dataset generation and management for the Juniper project"
readme = "README.md"
license = { text = "MIT" }
authors = [{ name = "Paul Calnon" }]
requires-python = ">=3.11"

dependencies = [
    "numpy>=1.24",
    "pydantic>=2.0",
]

[project.optional-dependencies]
api = [
    "fastapi>=0.100",
    "uvicorn>=0.22",
]
torch = [
    "torch>=2.0",
]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "httpx>=0.24",  # For API testing
    "mypy>=1.0",
    "black>=23.0",
    "isort>=5.12",
]
```

### Step 1.4: Create Test Baseline

**Description**: Run existing spiral tests to establish baseline behavior.

**Actions**:

1. Run all spiral-related tests in Juniper Cascor
2. Capture output shapes, value ranges, and determinism
3. Create snapshot tests for reference

**Commands**:

```bash
cd src/tests && bash scripts/run_tests.bash -m "spiral" -v
```

---

## Phase 2: Core Library Extraction

**Goal**: Extract pure data generation logic into standalone library  
**Duration**: 1 day  
**Priority**: P0 - Critical Path

### Step 2.1: Create Base Generator Interface

**Description**: Define abstract base class for all dataset generators.

**File**: `juniper_data/generator/base.py`

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from dataclasses import dataclass
import numpy as np

ParamsT = TypeVar('ParamsT')
DatasetT = TypeVar('DatasetT')

@dataclass
class BaseDataset:
    """Base class for all datasets."""
    X: np.ndarray
    y: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray

class BaseGenerator(ABC, Generic[ParamsT, DatasetT]):
    """Abstract base class for dataset generators."""

    @abstractmethod
    def generate(self, params: ParamsT) -> DatasetT:
        """Generate a dataset from the given parameters."""
        pass

    @abstractmethod
    def validate_params(self, params: ParamsT) -> None:
        """Validate generation parameters."""
        pass
```

### Step 2.2: Implement Spiral Generator

**Description**: Port spiral generation logic from `SpiralProblem` to pure generator.

**File**: `juniper_data/generator/spiral.py`

**Methods to Port** (in order):

| Order | Method                            | Source Lines | New Function Name               |
| ----- | --------------------------------- | ------------ | ------------------------------- |
| 1     | `_make_noise`                     | L770-784     | `_generate_noise()`             |
| 2     | `_make_coords`                    | L742-768     | `_calculate_coordinates()`      |
| 3     | `_generate_xy_coordinates`        | L707-740     | `_generate_spiral_point()`      |
| 4     | `generate_raw_spiral_coordinates` | L600-705     | `_generate_raw_coordinates()`   |
| 5     | `_create_input_features`          | L786-806     | `_create_features()`            |
| 6     | `_create_one_hot_targets`         | L808-835     | `_create_one_hot_labels()`      |
| 7     | `_shuffle_dataset`                | L889-920     | `_shuffle_data()`               |
| 8     | `_dataset_split_index_end`        | L1044-1067   | `_calculate_split_index()`      |
| 9     | `_find_partition_index_end`       | L1021-1042   | Internal to `_partition_data()` |
| 10    | `_split_dataset`                  | L967-1019    | `_partition_data()`             |
| 11    | `generate_n_spiral_dataset`       | Public API   | `generate()`                    |

**Key Implementation Notes**:

1. **Remove logging dependencies**: Replace `self.logger.*` with optional standard logging
2. **Remove class state**: Convert to pure functions where possible
3. **Use NumPy only**: PyTorch conversion is a separate adapter
4. **Add seed management**: Ensure deterministic output

### Step 2.3: Implement Defaults Module

**Description**: Create defaults module with only required constants.

**File**: `juniper_data/defaults.py`

```python
"""Default values for Juniper Data generators."""

# Spiral Dataset Defaults
SPIRAL_DEFAULT_N_SPIRALS = 2
SPIRAL_DEFAULT_N_POINTS = 97
SPIRAL_DEFAULT_N_ROTATIONS = 3
SPIRAL_DEFAULT_CLOCKWISE = True
SPIRAL_DEFAULT_NOISE = 0.25
SPIRAL_DEFAULT_DISTRIBUTION = 0.80
SPIRAL_DEFAULT_ORIGIN = 0.0
SPIRAL_DEFAULT_RADIUS = 10.0
SPIRAL_DEFAULT_TRAIN_RATIO = 0.8
SPIRAL_DEFAULT_TEST_RATIO = 0.2
SPIRAL_DEFAULT_RANDOM_SEED = 42
SPIRAL_DEFAULT_RANDOM_VALUE_SCALE = 0.1

# Data Format Defaults
DEFAULT_FLOAT_DTYPE = "float32"
```

### Step 2.4: Implement PyTorch Adapter

**Description**: Optional adapter for PyTorch tensor conversion.

**File**: `juniper_data/generator/adapters.py`

```python
"""Optional adapters for different ML frameworks."""

import numpy as np
from typing import Tuple

def to_torch_tensors(
    X: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Convert NumPy arrays to PyTorch tensors.

    Requires torch to be installed.
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for tensor conversion. "
            "Install with: pip install juniper-data[torch]"
        )

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(train_indices, dtype=torch.long),
        torch.tensor(test_indices, dtype=torch.long),
    )
```

### Step 2.5: Create Unit Tests for Generator

**Description**: Comprehensive tests for the extracted generator.

**File**: `tests/unit/test_spiral_generator.py`

**Test Cases**:

| Test Name                     | Description                         |
| ----------------------------- | ----------------------------------- |
| `test_generate_basic`         | Basic generation with defaults      |
| `test_output_shapes`          | Verify X, y shapes match parameters |
| `test_one_hot_encoding`       | Verify one-hot labels sum to 1      |
| `test_determinism_with_seed`  | Same seed = same output             |
| `test_train_test_split_ratio` | Split sizes match ratios            |
| `test_class_balance`          | Equal points per spiral             |
| `test_coordinate_range`       | X values within expected bounds     |
| `test_noise_application`      | Noise affects coordinates           |
| `test_n_spirals_variations`   | 2, 3, 4, 5 spirals work             |
| `test_invalid_params`         | Proper validation errors            |

---

## Phase 3: REST API Implementation

**Goal**: Create FastAPI-based REST service for dataset management  
**Duration**: 1 day  
**Priority**: P1 - High

### Step 3.1: Implement Storage Abstraction

**Description**: Create pluggable storage backend system.

**File**: `juniper_data/storage/base.py`

```python
from abc import ABC, abstractmethod
from typing import Optional, List
from ..models.dataset import SpiralDataset

class DatasetStore(ABC):
    """Abstract interface for dataset storage."""

    @abstractmethod
    def save(self, dataset: SpiralDataset) -> str:
        """Save dataset and return its ID."""
        pass

    @abstractmethod
    def get(self, dataset_id: str) -> Optional[SpiralDataset]:
        """Retrieve dataset by ID."""
        pass

    @abstractmethod
    def list(self, limit: int = 100, offset: int = 0) -> List[str]:
        """List dataset IDs."""
        pass

    @abstractmethod
    def delete(self, dataset_id: str) -> bool:
        """Delete dataset by ID."""
        pass

    @abstractmethod
    def exists(self, dataset_id: str) -> bool:
        """Check if dataset exists."""
        pass
```

### Step 3.2: Implement Memory Store

**Description**: In-memory storage for development and testing.

**File**: `juniper_data/storage/memory.py`

```python
from typing import Dict, Optional, List
from .base import DatasetStore
from ..models.dataset import SpiralDataset

class InMemoryDatasetStore(DatasetStore):
    """In-memory dataset storage for development/testing."""

    def __init__(self):
        self._datasets: Dict[str, SpiralDataset] = {}

    def save(self, dataset: SpiralDataset) -> str:
        self._datasets[dataset.dataset_id] = dataset
        return dataset.dataset_id

    def get(self, dataset_id: str) -> Optional[SpiralDataset]:
        return self._datasets.get(dataset_id)

    def list(self, limit: int = 100, offset: int = 0) -> List[str]:
        ids = list(self._datasets.keys())
        return ids[offset:offset + limit]

    def delete(self, dataset_id: str) -> bool:
        if dataset_id in self._datasets:
            del self._datasets[dataset_id]
            return True
        return False

    def exists(self, dataset_id: str) -> bool:
        return dataset_id in self._datasets
```

### Step 3.3: Implement Filesystem Store

**Description**: File-based storage with metadata JSON and NPZ data.

**File**: `juniper_data/storage/filesystem.py`

**Storage Format**:

```bash
data/datasets/
├── {dataset_id}/
│   ├── metadata.json     # Params, created_at, etc.
│   └── data.npz          # X, y, train_indices, test_indices
```

### Step 3.4: Create API Application

**Description**: FastAPI application with dataset endpoints.

**File**: `juniper_data/api/main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import datasets, health

app = FastAPI(
    title="Juniper Data API",
    description="Dataset generation and management service",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
```

### Step 3.5: Implement Dataset Endpoints

**Description**: REST endpoints for dataset operations.

**File**: `juniper_data/api/routes/datasets.py`

**Endpoints**:

| Method | Path                  | Description           |
| ------ | --------------------- | --------------------- |
| POST   | `/datasets`           | Generate new dataset  |
| GET    | `/datasets`           | List datasets         |
| GET    | `/datasets/{id}`      | Get dataset metadata  |
| GET    | `/datasets/{id}/data` | Download dataset data |
| DELETE | `/datasets/{id}`      | Delete dataset        |

**Request/Response Models**:

```python
# POST /datasets
class CreateDatasetRequest(BaseModel):
    source_type: Literal["spiral", "circles", "xor"] = "spiral"
    params: SpiralDatasetParams
    persist: bool = True

class CreateDatasetResponse(BaseModel):
    dataset_id: str
    source_type: str
    params: SpiralDatasetParams
    n_samples: int
    created_at: datetime
    download_url: str

# GET /datasets/{id}
class DatasetMetadataResponse(BaseModel):
    dataset_id: str
    source_type: str
    params: SpiralDatasetParams
    n_samples: int
    created_at: datetime
    data_url: str
```

### Step 3.6: Create API Tests

**Description**: Integration tests for API endpoints.

**File**: `tests/integration/test_api.py`

**Test Cases**:

| Test Name                            | Description                 |
| ------------------------------------ | --------------------------- |
| `test_create_dataset_spiral`         | POST creates spiral dataset |
| `test_create_dataset_invalid_params` | 422 on invalid params       |
| `test_get_dataset_metadata`          | GET returns metadata        |
| `test_get_dataset_not_found`         | 404 for missing dataset     |
| `test_download_dataset_npz`          | Download as NPZ file        |
| `test_download_dataset_json`         | Download as JSON (small)    |
| `test_list_datasets`                 | List pagination works       |
| `test_delete_dataset`                | DELETE removes dataset      |
| `test_health_endpoint`               | Health check returns 200    |

---

## Phase 4: Integration and Migration

**Goal**: Update Juniper Cascor to use extracted library  
**Duration**: 1 day  
**Priority**: P1 - High

### Step 4.1: Add Juniper Data as Dependency

**Description**: Update Juniper Cascor to depend on Juniper Data library.

**File**: `juniper_cascor/pyproject.toml`

```toml
[project.optional-dependencies]
data = [
    "juniper-data>=0.1.0",
]
```

**Alternative (Local Development)**:

```toml
# For development, use editable install
dependencies = [
    # ... existing deps
]

[tool.setuptools]
packages = ["src"]

# In development:
# pip install -e ../JuniperData
```

### Step 4.2: Create Compatibility Layer

**Description**: Adapter to maintain backward compatibility with existing `SpiralProblem` interface.

**File**: `src/spiral_problem/spiral_data_adapter.py`

```python
"""Adapter layer for backward compatibility with SpiralProblem interface."""

from typing import Tuple, Optional
import torch

# Import from Juniper Data
from juniper_data.generator.spiral import SpiralGenerator, SpiralDatasetParams
from juniper_data.generator.adapters import to_torch_tensors

def generate_spiral_data(
    n_spirals: int = 2,
    n_points: int = 97,
    n_rotations: int = 3,
    clockwise: bool = True,
    noise_level: float = 0.25,
    distribution: float = 0.80,
    train_ratio: float = 0.8,
    test_ratio: float = 0.2,
    random_seed: Optional[int] = 42,
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],  # (x_train, y_train)
    Tuple[torch.Tensor, torch.Tensor],  # (x_test, y_test)
    Tuple[torch.Tensor, torch.Tensor],  # (x_full, y_full)
]:
    """Generate spiral dataset with SpiralProblem-compatible interface.

    This function provides backward compatibility with the existing
    SpiralProblem.generate_n_spiral_dataset() method signature.
    """
    params = SpiralDatasetParams(
        n_spirals=n_spirals,
        n_points_per_spiral=n_points,
        n_rotations=n_rotations,
        clockwise=clockwise,
        noise=noise_level,
        distribution=distribution,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )

    generator = SpiralGenerator()
    dataset = generator.generate(params)

    # Convert to PyTorch tensors
    X_t, y_t, train_idx, test_idx = to_torch_tensors(
        dataset.X, dataset.y, dataset.train_indices, dataset.test_indices
    )

    # Create splits
    x_train = X_t[train_idx]
    y_train = y_t[train_idx]
    x_test = X_t[test_idx]
    y_test = y_t[test_idx]

    return (
        (x_train, y_train),
        (x_test, y_test),
        (X_t, y_t),
    )
```

### Step 4.3: Update SpiralProblem to Use Adapter

**Description**: Modify `SpiralProblem.generate_n_spiral_dataset()` to delegate to adapter.

**File**: `src/spiral_problem/spiral_problem.py`

**Changes**:

1. Import adapter function
2. Replace internal generation logic with adapter call
3. Maintain existing logging and error handling
4. Keep all method signatures unchanged

**Example Modification**:

```python
# In spiral_problem.py
from spiral_problem.spiral_data_adapter import generate_spiral_data

def generate_n_spiral_dataset(
    self,
    n_spirals=None,
    n_points=None,
    # ... other params
) -> Tuple[...]:
    """Generate N-spiral dataset.

    Now delegates to Juniper Data library.
    """
    self.logger.trace("Delegating to Juniper Data library")

    # Use new library
    train_partition, test_partition, full_partition = generate_spiral_data(
        n_spirals=n_spirals or self.n_spirals,
        n_points=n_points or self.n_points,
        # ... map other params
    )

    self.logger.debug(f"Generated dataset via Juniper Data")
    return (train_partition, test_partition, full_partition)
```

### Step 4.4: Update Test Fixtures

**Description**: Update test fixtures to use new generator.

**File**: `src/tests/conftest.py`

**Changes**:

1. Import `SpiralGenerator` from Juniper Data
2. Update `spiral_2d_data` fixture
3. Update `n_spiral_data` fixture
4. Maintain backward compatibility

### Step 4.5: Run Integration Tests

**Description**: Verify all existing tests pass with new implementation.

**Commands**:

```bash
# Run all spiral tests
cd src/tests && bash scripts/run_tests.bash -m "spiral" -v

# Run full test suite
cd src/tests && bash scripts/run_tests.bash -v

# Run with coverage
cd src/tests && bash scripts/run_tests.bash -v -c
```

### Step 4.6: Validate Behavioral Equivalence

**Description**: Ensure new implementation produces equivalent results.

**Validation Checks**:

1. **Shape equivalence**: Same output shapes
2. **Determinism**: Same seed = same output
3. **Range equivalence**: Similar value ranges
4. **Split ratios**: Correct train/test sizes
5. **One-hot encoding**: Proper label format

---

## Phase 5: Extended Data Source Support

**Goal**: Implement multi-source data retrieval capabilities  
**Duration**: 2 days  
**Priority**: P2 - Medium

### Step 5.1: Define Source Interface

**Description**: Abstract interface for data sources.

**File**: `juniper_data/sources/base.py`

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..models.dataset import BaseDataset

class DataSource(ABC):
    """Abstract interface for data sources."""

    source_type: str

    @abstractmethod
    def fetch(self, params: Dict[str, Any]) -> BaseDataset:
        """Fetch or generate dataset from this source."""
        pass

    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> None:
        """Validate source-specific parameters."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if source is available."""
        pass
```

### Step 5.2: Implement Local Generator Source

**Description**: Source that wraps local generators (spiral, circles, etc.).

**File**: `juniper_data/sources/local_generator.py`

```python
from typing import Dict, Any
from .base import DataSource
from ..generator.spiral import SpiralGenerator, SpiralDatasetParams
from ..models.dataset import BaseDataset

class LocalGeneratorSource(DataSource):
    """Data source using local generators."""

    source_type = "local_generator"

    def __init__(self):
        self._generators = {
            "spiral": SpiralGenerator(),
            # Add more generators as implemented
        }

    def fetch(self, params: Dict[str, Any]) -> BaseDataset:
        generator_type = params.get("generator_type", "spiral")
        generator = self._generators.get(generator_type)

        if generator is None:
            raise ValueError(f"Unknown generator: {generator_type}")

        # Convert params dict to appropriate dataclass
        generator_params = self._build_params(generator_type, params)
        return generator.generate(generator_params)

    @property
    def is_available(self) -> bool:
        return True
```

### Step 5.3: Implement File Source

**Description**: Source that loads datasets from local files.

**File**: `juniper_data/sources/file_source.py`

**Supported Formats**:

- `.npz` (NumPy compressed)
- `.npy` (NumPy array)
- `.csv` (with header)
- `.json` (structured)
- `.pt` / `.pth` (PyTorch)

### Step 5.4: Implement URL Source

**Description**: Source that downloads datasets from URLs.

**File**: `juniper_data/sources/url_source.py`

**Supported Sources**:

- Direct download URLs
- Hugging Face Datasets Hub
- Kaggle Datasets (with auth)
- USPS Dataset URLs
- S3/GCS presigned URLs

**Example Implementation**:

```python
import httpx
from typing import Dict, Any
from .base import DataSource

class URLSource(DataSource):
    """Data source for downloading from URLs."""

    source_type = "url"

    def __init__(self, timeout: float = 30.0):
        self._client = httpx.Client(timeout=timeout)

    def fetch(self, params: Dict[str, Any]) -> BaseDataset:
        url = params["url"]
        format_hint = params.get("format", "auto")

        response = self._client.get(url)
        response.raise_for_status()

        return self._parse_response(response.content, format_hint)

    def _parse_response(self, content: bytes, format_hint: str):
        # Auto-detect format from content or use hint
        pass
```

### Step 5.5: Implement Database Source

**Description**: Source that retrieves datasets from databases.

**File**: `juniper_data/sources/database.py`

**Supported Databases**:

- SQLite (local)
- PostgreSQL (remote)
- MongoDB (NoSQL)

### Step 5.6: Create Source Registry

**Description**: Registry for managing available data sources.

**File**: `juniper_data/sources/registry.py`

```python
from typing import Dict, Type, Optional
from .base import DataSource

class SourceRegistry:
    """Registry for data sources."""

    _sources: Dict[str, Type[DataSource]] = {}
    _instances: Dict[str, DataSource] = {}

    @classmethod
    def register(cls, source_class: Type[DataSource]) -> None:
        """Register a data source class."""
        cls._sources[source_class.source_type] = source_class

    @classmethod
    def get(cls, source_type: str) -> Optional[DataSource]:
        """Get a source instance by type."""
        if source_type not in cls._instances:
            source_class = cls._sources.get(source_type)
            if source_class:
                cls._instances[source_type] = source_class()
        return cls._instances.get(source_type)

    @classmethod
    def list_available(cls) -> list[str]:
        """List available source types."""
        return [
            st for st, sc in cls._sources.items()
            if cls.get(st).is_available
        ]
```

### Step 5.7: Update API for Multi-Source Support

**Description**: Extend API endpoints to support different sources.

**Updated Request Model**:

```python
class CreateDatasetRequest(BaseModel):
    source_type: Literal[
        "spiral",           # Local generator
        "circles",          # Local generator  
        "xor",              # Local generator
        "file",             # Local file
        "url",              # Remote URL
        "huggingface",      # HF Hub
        "kaggle",           # Kaggle
        "database",         # DB query
    ] = "spiral"

    # Source-specific params
    params: Dict[str, Any]

    # Storage options
    persist: bool = True
    storage_backend: Optional[str] = None
```

---

## Testing Strategy

### Unit Tests

| Module                  | Test File                  | Coverage Target |
| ----------------------- | -------------------------- | --------------- |
| `generator/spiral.py`   | `test_spiral_generator.py` | 90%             |
| `generator/adapters.py` | `test_adapters.py`         | 85%             |
| `storage/memory.py`     | `test_memory_store.py`     | 95%             |
| `storage/filesystem.py` | `test_filesystem_store.py` | 90%             |
| `models/*`              | `test_models.py`           | 95%             |
| `sources/*`             | `test_sources.py`          | 85%             |

### Integration Tests

| Scenario          | Test File                     | Description                 |
| ----------------- | ----------------------------- | --------------------------- |
| API E2E           | `test_api.py`                 | Full request/response cycle |
| Storage Roundtrip | `test_storage_integration.py` | Save/Load cycles            |
| Source to API     | `test_source_api.py`          | Source fetching via API     |

### Behavioral Tests

| Test                   | Description                         |
| ---------------------- | ----------------------------------- |
| Determinism            | Same seed produces identical output |
| Backward Compatibility | New library matches old behavior    |
| Performance            | Generation time within bounds       |

### Test Commands

```bash
# Juniper Data tests
cd JuniperData && pytest tests/ -v --cov=juniper_data

# Juniper Cascor integration tests
cd juniper_cascor/src/tests && bash scripts/run_tests.bash -m "spiral" -v

# Full regression suite
cd juniper_cascor/src/tests && bash scripts/run_tests.bash -v -c
```

---

## Risk Assessment and Mitigation

### Risk Matrix

| Risk                                | Probability | Impact | Mitigation                             |
| ----------------------------------- | ----------- | ------ | -------------------------------------- |
| Behavioral drift in data generation | Medium      | High   | Snapshot tests, determinism validation |
| Training accuracy changes           | Medium      | High   | Baseline comparison tests              |
| Import conflicts between projects   | Low         | Medium | Proper namespacing, virtual envs       |
| Performance regression              | Low         | Medium | Benchmarks, profiling                  |
| API breaking changes                | Low         | High   | Version pinning, deprecation warnings  |
| Storage corruption                  | Low         | High   | Checksums, atomic writes               |

### Mitigation Details

#### Behavioral Drift Mitigation

1. **Snapshot Tests**: Store reference outputs for known parameter sets
2. **Determinism Checks**: Verify seed reproducibility
3. **Value Range Assertions**: Check coordinate bounds
4. **Class Balance Verification**: Ensure equal distribution

#### Training Accuracy Mitigation

1. **Before/After Comparison**: Run same training with old and new generation
2. **Statistical Tests**: Compare accuracy distributions
3. **Gradual Rollout**: Use feature flag in Cascor

---

## Timeline and Effort Estimates

### Gantt Chart View

```bash
Week 1
├── Day 1 (0.5d): Phase 1 - Analysis & Preparation
│   ├── Step 1.1: Create extraction mapping
│   ├── Step 1.2: Define data contracts
│   ├── Step 1.3: Set up repository
│   └── Step 1.4: Create test baseline
│
├── Day 1-2 (1d): Phase 2 - Core Library Extraction
│   ├── Step 2.1: Create base interface
│   ├── Step 2.2: Implement spiral generator
│   ├── Step 2.3: Implement defaults module
│   ├── Step 2.4: Implement PyTorch adapter
│   └── Step 2.5: Create unit tests
│
├── Day 3 (1d): Phase 3 - REST API Implementation
│   ├── Step 3.1: Implement storage abstraction
│   ├── Step 3.2: Implement memory store
│   ├── Step 3.3: Implement filesystem store
│   ├── Step 3.4: Create API application
│   ├── Step 3.5: Implement endpoints
│   └── Step 3.6: Create API tests
│
├── Day 4 (1d): Phase 4 - Integration & Migration
│   ├── Step 4.1: Add dependency
│   ├── Step 4.2: Create compatibility layer
│   ├── Step 4.3: Update SpiralProblem
│   ├── Step 4.4: Update test fixtures
│   ├── Step 4.5: Run integration tests
│   └── Step 4.6: Validate equivalence
│
Week 2
└── Day 5-6 (2d): Phase 5 - Extended Data Sources
    ├── Step 5.1: Define source interface
    ├── Step 5.2: Implement local generator source
    ├── Step 5.3: Implement file source
    ├── Step 5.4: Implement URL source
    ├── Step 5.5: Implement database source
    ├── Step 5.6: Create source registry
    └── Step 5.7: Update API
```

### Resource Requirements

| Resource    | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
| ----------- | ------- | ------- | ------- | ------- | ------- |
| Developer   | 1       | 1       | 1       | 1       | 1-2     |
| Code Review | -       | 1       | 1       | 1       | 1       |
| Testing     | 0.5     | 1       | 1       | 1       | 1       |

---

## Appendices

### Appendix A: Complete Method Extraction Map

| Source Method                                     | Target Function                      | Status  |
| ------------------------------------------------- | ------------------------------------ | ------- |
| `SpiralProblem._make_noise()`                     | `spiral._generate_noise()`           | EXTRACT |
| `SpiralProblem._make_coords()`                    | `spiral._calculate_coordinates()`    | EXTRACT |
| `SpiralProblem._generate_xy_coordinates()`        | `spiral._generate_spiral_point()`    | EXTRACT |
| `SpiralProblem.generate_raw_spiral_coordinates()` | `spiral._generate_raw_coordinates()` | EXTRACT |
| `SpiralProblem._create_input_features()`          | `spiral._create_features()`          | EXTRACT |
| `SpiralProblem._create_one_hot_targets()`         | `spiral._create_one_hot_labels()`    | EXTRACT |
| `SpiralProblem._create_spiral_dataset()`          | Internal to `generate()`             | EXTRACT |
| `SpiralProblem._convert_to_tensors()`             | `adapters.to_torch_tensors()`        | EXTRACT |
| `SpiralProblem._shuffle_dataset()`                | `spiral._shuffle_data()`             | EXTRACT |
| `SpiralProblem._partition_dataset()`              | `spiral._partition_data()`           | EXTRACT |
| `SpiralProblem._split_dataset()`                  | Internal to `_partition_data()`      | EXTRACT |
| `SpiralProblem.generate_n_spiral_dataset()`       | `SpiralGenerator.generate()`         | EXTRACT |
| `SpiralProblem.solve_n_spiral_problem()`          | N/A                                  | KEEP    |
| `SpiralProblem.evaluate()`                        | N/A                                  | KEEP    |
| `SpiralProblem.__init__()`                        | Modified                             | MODIFY  |

### Appendix B: Constants Migration Map

| Cascor Constant                            | Juniper Data Default                | Value |
| ------------------------------------------ | ----------------------------------- | ----- |
| `_SPIRAL_PROBLEM_NUM_SPIRALS`              | `SPIRAL_DEFAULT_N_SPIRALS`          | 2     |
| `_SPIRAL_PROBLEM_NUMBER_POINTS_PER_SPIRAL` | `SPIRAL_DEFAULT_N_POINTS`           | 97    |
| `_SPIRAL_PROBLEM_NUM_ROTATIONS`            | `SPIRAL_DEFAULT_N_ROTATIONS`        | 3     |
| `_SPIRAL_PROBLEM_CLOCKWISE`                | `SPIRAL_DEFAULT_CLOCKWISE`          | True  |
| `_SPIRAL_PROBLEM_NOISE_FACTOR_DEFAULT`     | `SPIRAL_DEFAULT_NOISE`              | 0.25  |
| `_SPIRAL_PROBLEM_DISTRIBUTION_FACTOR`      | `SPIRAL_DEFAULT_DISTRIBUTION`       | 0.80  |
| `_SPIRAL_PROBLEM_DEFAULT_ORIGIN`           | `SPIRAL_DEFAULT_ORIGIN`             | 0.0   |
| `_SPIRAL_PROBLEM_DEFAULT_RADIUS`           | `SPIRAL_DEFAULT_RADIUS`             | 10.0  |
| `_SPIRAL_PROBLEM_TRAIN_RATIO`              | `SPIRAL_DEFAULT_TRAIN_RATIO`        | 0.8   |
| `_SPIRAL_PROBLEM_TEST_RATIO`               | `SPIRAL_DEFAULT_TEST_RATIO`         | 0.2   |
| `_SPIRAL_PROBLEM_RANDOM_VALUE_SCALE`       | `SPIRAL_DEFAULT_RANDOM_VALUE_SCALE` | 0.1   |

### Appendix C: API Schema (OpenAPI)

```yaml
openapi: 3.0.3
info:
  title: Juniper Data API
  version: 0.1.0

paths:
  /datasets:
    post:
      summary: Create a new dataset
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateDatasetRequest'
      responses:
        '201':
          description: Dataset created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CreateDatasetResponse'
    get:
      summary: List datasets
      parameters:
        - name: limit
          in: query
          schema:
            type: integer
            default: 100
        - name: offset
          in: query
          schema:
            type: integer
            default: 0
      responses:
        '200':
          description: List of dataset IDs
          content:
            application/json:
              schema:
                type: array
                items:
                  type: string

  /datasets/{id}:
    get:
      summary: Get dataset metadata
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Dataset metadata
        '404':
          description: Dataset not found

  /datasets/{id}/data:
    get:
      summary: Download dataset data
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
        - name: format
          in: query
          schema:
            type: string
            enum: [npz, json, pt]
            default: npz
      responses:
        '200':
          description: Dataset file
          content:
            application/octet-stream:
              schema:
                type: string
                format: binary

  /health:
    get:
      summary: Health check
      responses:
        '200':
          description: Service is healthy

components:
  schemas:
    CreateDatasetRequest:
      type: object
      properties:
        source_type:
          type: string
          enum: [spiral, circles, xor, file, url]
          default: spiral
        params:
          type: object
        persist:
          type: boolean
          default: true

    CreateDatasetResponse:
      type: object
      properties:
        dataset_id:
          type: string
        source_type:
          type: string
        n_samples:
          type: integer
        created_at:
          type: string
          format: date-time
        download_url:
          type: string
```

### Appendix D: Test Fixture Updates

**Before** (conftest.py):

```python
@pytest.fixture
def spiral_2d_data() -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    n_per_spiral = 100
    t = torch.linspace(0, 4*np.pi, n_per_spiral)
    # ... inline generation
```

**After** (conftest.py):

```python
from juniper_data.generator.spiral import SpiralGenerator, SpiralDatasetParams
from juniper_data.generator.adapters import to_torch_tensors

@pytest.fixture
def spiral_2d_data() -> Tuple[torch.Tensor, torch.Tensor]:
    params = SpiralDatasetParams(
        n_spirals=2,
        n_points_per_spiral=100,
        random_seed=42,
    )
    generator = SpiralGenerator()
    dataset = generator.generate(params)
    X, y, _, _ = to_torch_tensors(
        dataset.X, dataset.y,
        dataset.train_indices, dataset.test_indices
    )
    return X, y
```

---

## Document History

| Version | Date       | Author           | Changes       |
| ------- | ---------- | ---------------- | ------------- |
| 1.0.0   | 2026-01-29 | Juniper Dev Team | Initial draft |

---
