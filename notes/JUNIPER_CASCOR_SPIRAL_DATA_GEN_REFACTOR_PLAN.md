# Juniper Data: Spiral Dataset Generator Extraction Plan

**Document**: JUNIPER_CASCOR_SPIRAL_DATA_GEN_REFACTOR_PLAN.md  
**Created**: 2026-01-29  
**Last Updated**: 2026-01-29  
**Version**: 1.1.0  
**Status**: In Progress - Phases 0-2 Complete  
**Author**: Juniper Development Team

---

## Executive Summary

This document presents the **comprehensive refactoring plan** to extract the Spiral Problem Data Generator code from **Juniper Cascor** into a new standalone application called **JuniperData**. This plan synthesizes the best approaches from three evaluated proposals, prototype implementations, and Oracle-guided analysis of the existing codebase.

### Core Principles

1. **Pure NumPy Generator Core**: Data generation logic uses only NumPy—no PyTorch, logging frameworks, or training infrastructure
2. **Artifact-First API Design**: Return dataset artifacts (NPZ) instead of embedding large arrays in JSON
3. **Minimal Viable Provider Set**: Implement only Generated + LocalFS cache for v1; defer other sources
4. **Deterministic Reproducibility**: Hash-based dataset IDs enable caching and exact reproduction

### Key Objectives

1. **Extract** spiral data generation from Cascor into JuniperData
2. **Decouple** completely from neural network training infrastructure
3. **Create** REST API with artifact-based dataset delivery
4. **Maintain** backward compatibility during migration with feature flags

### Effort Summary

| Phase   | Description                        | Effort | Duration        | Status      |
| ------- | ---------------------------------- | ------ | --------------- | ----------- |
| Phase 0 | Baseline Inventory & Contract Lock | S      | 0.5-1 day       | ✅ Complete |
| Phase 1 | Core Generator Extraction          | M      | 1-2 days        | ✅ Complete |
| Phase 2 | JuniperData REST API v1            | M      | 1-2 days        | ✅ Complete |
| Phase 3 | Cascor Integration                 | M-L    | 1-2 days        | 🔄 Pending  |
| Phase 4 | Canopy Integration                 | M      | 1 day           | 🔄 Pending  |
| Phase 5 | Extended Data Sources              | L-XL   | Staged (future) | ⏳ Deferred |

**Total Initial Delivery (Phases 0-4)**: 4-8 days  
**Current Progress**: Phases 0-2 Complete (76 tests passing)

---

## Table of Contents

1. [Proposal Evaluation Summary](#1-proposal-evaluation-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Method Extraction Specification](#3-method-extraction-specification)
4. [Dependency Decoupling Strategy](#4-dependency-decoupling-strategy)
5. [Target Architecture](#5-target-architecture)
6. [Dataset Contract Specification](#6-dataset-contract-specification)
7. [Implementation Phases](#7-implementation-phases)
8. [Migration Strategy](#8-migration-strategy)
9. [Testing & Validation Requirements](#9-testing--validation-requirements)
10. [Risk Assessment & Guardrails](#10-risk-assessment--guardrails)
11. [Appendices](#11-appendices)

---

## 1. Proposal Evaluation Summary

### 1.1 Evaluated Proposals

| Proposal                               | Strengths Adopted                                                                         | Elements Deferred/Modified                                                      |
| -------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **001** (REST + Provider Architecture) | Service boundary via REST; endpoint conceptual design; provider abstraction as north-star | Full provider zoo (S3/GCS/HF/DB) deferred to Phase 5; avoid large JSON payloads |
| **002** (Dataset Contract Model)       | Dataset Contract definition; deterministic dataset_id; storage abstraction                | Artifact-first approach instead of JSON arrays; torch as optional adapter       |
| **003** (Method Extraction Map)        | Concrete extraction mapping; constants migration; testing strategy                        | Used as primary execution plan                                                  |

### 1.2 Prototype Code Reference

| Prototype                                | Usage in Plan                                                             |
| ---------------------------------------- | ------------------------------------------------------------------------- |
| `cascor_spiral/.../spiral_generator.py`  | Feature parity reference; avoid logging/dirs/plotting patterns            |
| `spiral_dataset/.../spiral_generator.py` | Simpler generation pattern reference                                      |
| `test_data/generators.py`                | **Target clean style**: static methods, dataclass return, no dependencies |

### 1.3 Decision Summary

**Adopt**: Proposal 002's contract design + deterministic IDs  
**Execute with**: Proposal 003's extraction map  
**Future extension**: Proposal 001's provider architecture  

---

## 2. Current State Analysis

### 2.1 Spiral Data Generation References

#### Juniper Cascor

| File Path                                                     | Class/Method          | Purpose                                                      | Lines  |
| ------------------------------------------------------------- | --------------------- | ------------------------------------------------------------ | ------ |
| `src/spiral_problem/spiral_problem.py`                        | `SpiralProblem`       | Primary implementation with mixed data generation + training | ~1000+ |
| `src/tests/unit/test_data/generators.py`                      | `SpiralDataGenerator` | **Clean reference** - static methods, decoupled              | 142    |
| `src/cascor_constants/constants_problem/constants_problem.py` | Constants             | Spiral-specific defaults                                     | ~50    |

#### Juniper Canopy

| File Path                           | Method                                               | Purpose              |
| ----------------------------------- | ---------------------------------------------------- | -------------------- |
| `src/demo_mode.py`                  | `DemoMode._generate_spiral_dataset()`                | Simple 2-spiral mock |
| `src/backend/cascor_integration.py` | `CascorIntegration._generate_missing_dataset_info()` | Fallback mock        |

#### Juniper Prototypes

| Location                               | Implementation                      | Characteristics                         |
| -------------------------------------- | ----------------------------------- | --------------------------------------- |
| `cascor_spiral/src/spiral_generator/`  | Full SpiralGenerator + SpiralConfig | Feature-rich, includes logging/plotting |
| `spiral_dataset/src/spiral_generator/` | Simpler SpiralGenerator             | Cleaner, fewer dependencies             |

### 2.2 Current Dependency Structure

```bash
SpiralProblem (Cascor)
├── Standard Library
│   ├── logging, logging.config
│   ├── os, sys, random, uuid
│   └── multiprocessing
├── Third-Party
│   ├── numpy (np)
│   ├── torch
│   └── matplotlib.pyplot
├── Local (Juniper Cascor) - TO BE DECOUPLED
│   ├── CascadeCorrelationNetwork
│   ├── CascadeCorrelationConfig
│   ├── cascor_constants.constants (55+ constants)
│   ├── LogConfig
│   └── Logger
```

---

## 3. Method Extraction Specification

### 3.1 Methods to Extract (Pure Data Generation)

These methods will be ported to JuniperData as **pure NumPy functions** (no torch, logging frameworks, or CCN dependencies):

#### Core Coordinate Generation

| Source Method                       | Target Function                      | Priority |
| ----------------------------------- | ------------------------------------ | -------- |
| `generate_raw_spiral_coordinates()` | `spiral._generate_raw_coordinates()` | P0       |
| `_generate_xy_coordinates()`        | `spiral._generate_spiral_point()`    | P0       |
| `_make_coords()`                    | `spiral._calculate_coordinates()`    | P0       |
| `_make_noise()`                     | `spiral._generate_noise()`           | P0       |

#### Feature/Label Construction

| Source Method               | Target Function                   | Priority |
| --------------------------- | --------------------------------- | -------- |
| `_create_input_features()`  | `spiral._create_features()`       | P0       |
| `_create_one_hot_targets()` | `spiral._create_one_hot_labels()` | P0       |
| `_create_spiral_dataset()`  | Internal to `generate()`          | P0       |

#### Dataset Ordering & Partitioning

| Source Method                 | Target Function                 | Priority |
| ----------------------------- | ------------------------------- | -------- |
| `_shuffle_dataset()`          | `split._shuffle_data()`         | P0       |
| `_partition_dataset()`        | `split._partition_data()`       | P0       |
| `_split_dataset()`            | Internal to `_partition_data()` | P0       |
| `_find_partition_index_end()` | Internal helper                 | P0       |
| `_dataset_split_index_end()`  | Internal helper                 | P0       |

#### Public Orchestration

| Source Method                 | Target Function              | Priority |
| ----------------------------- | ---------------------------- | -------- |
| `generate_n_spiral_dataset()` | `SpiralGenerator.generate()` | P0       |

### 3.2 Methods to Keep in Cascor (Training-Specific)

These methods remain in Juniper Cascor because they depend on training infrastructure:

| Method                           | Reason to Keep                                          |
| -------------------------------- | ------------------------------------------------------- |
| `__init__()`                     | Creates log config, network, seeds torch, config wiring |
| `solve_n_spiral_problem()`       | Training orchestration using CCN                        |
| `evaluate()`                     | Training evaluation pipeline                            |
| `_convert_to_tensors()`          | PyTorch-specific; move to Cascor client                 |
| Plotting methods                 | Visualization (keep in Cascor or Canopy)                |
| Multiprocessing queue management | Training infrastructure                                 |

### 3.3 Borderline Utilities Decision

| Method                  | Decision           | Rationale                                                                    |
| ----------------------- | ------------------ | ---------------------------------------------------------------------------- |
| `_convert_to_tensors()` | **Keep in Cascor** | Converts numpy arrays from JuniperData to torch tensors; cleanest separation |

---

## 4. Dependency Decoupling Strategy

### 4.1 Dependencies to Remove from Extracted Generator

| Current Dependency           | Action          | Replacement                                  |
| ---------------------------- | --------------- | -------------------------------------------- |
| `cascor_constants.constants` | Copy & localize | `juniper_data/generators/spiral/defaults.py` |
| `LogConfig`, `Logger`        | Remove          | Standard `logging` (optional) or none        |
| `CascadeCorrelationNetwork`  | Remove          | Not needed for data generation               |
| `CascadeCorrelationConfig`   | Remove          | Not needed for data generation               |
| `torch`                      | Remove          | Not required for generation                  |
| `matplotlib`                 | Remove          | Visualization separate concern               |
| `multiprocessing`            | Remove          | Training infrastructure                      |

### 4.2 Minimal Dependencies for JuniperData

```python
# Core (required)
numpy>=1.24.0            # Core array operations
pydantic>=2.0.0          # Request/response validation

# API Layer (required)
fastapi>=0.100.0         # REST API framework
uvicorn[standard]>=0.23.0  # ASGI server

# Optional (future phases)
torch>=2.0.0             # Optional adapter for tensor output
h5py>=3.8.0              # Optional HDF5 format support
```

### 4.3 Constants to Extract

From `constants_problem.py`, extract only spiral generation defaults:

```python
# Spiral Geometry
SPIRAL_DEFAULT_N_SPIRALS = 2
SPIRAL_DEFAULT_N_POINTS = 97
SPIRAL_DEFAULT_N_ROTATIONS = 3
SPIRAL_DEFAULT_CLOCKWISE = True
SPIRAL_DEFAULT_DISTRIBUTION = 0.80
SPIRAL_DEFAULT_ORIGIN = 0.0
SPIRAL_DEFAULT_RADIUS = 10.0

# Noise & Randomness
SPIRAL_DEFAULT_NOISE = 0.25
SPIRAL_DEFAULT_RANDOM_VALUE_SCALE = 0.1
SPIRAL_DEFAULT_SEED = 42

# Dataset Splitting
SPIRAL_DEFAULT_TRAIN_RATIO = 0.8
SPIRAL_DEFAULT_TEST_RATIO = 0.2

# Input/Output (derived)
SPIRAL_INPUT_SIZE = 2
# SPIRAL_OUTPUT_SIZE = n_spirals (dynamic)
```

---

## 5. Target Architecture

### 5.1 Repository Structure

```bash
JuniperData/
├── juniper_data/                     # Python package root
│   ├── __init__.py
│   ├── core/                         # Core utilities
│   │   ├── __init__.py
│   │   ├── models.py                 # DatasetMeta, schemas (Pydantic)
│   │   ├── dataset_id.py             # Deterministic ID generation
│   │   ├── split.py                  # Shuffle/split utilities (NumPy)
│   │   └── artifacts.py              # NPZ encode/decode helpers
│   │
│   ├── generators/                   # Dataset generators
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseGenerator, registry
│   │   └── spiral/                   # Spiral generator module
│   │       ├── __init__.py
│   │       ├── params.py             # SpiralParams (Pydantic)
│   │       ├── generator.py          # SpiralGenerator (NumPy-only)
│   │       └── defaults.py           # Spiral defaults + validation
│   │
│   ├── storage/                      # Dataset storage backends
│   │   ├── __init__.py
│   │   ├── base.py                   # DatasetStore interface
│   │   ├── memory.py                 # InMemoryDatasetStore (tests)
│   │   └── local_fs.py               # LocalFSDatasetStore (v1)
│   │
│   └── api/                          # REST API layer
│       ├── __init__.py
│       ├── app.py                    # FastAPI application factory
│       ├── settings.py               # Environment configuration
│       └── routes/
│           ├── __init__.py
│           ├── health.py             # Health check endpoints
│           ├── generators.py         # Generator listing/schema
│           └── datasets.py           # Dataset CRUD operations
│
├── tests/                            # Test suite
│   ├── unit/
│   │   ├── test_spiral_generator.py
│   │   ├── test_split.py
│   │   └── test_dataset_id.py
│   ├── integration/
│   │   └── test_api.py
│   └── fixtures/
│       └── golden_datasets/          # Reference datasets for parity
│
├── pyproject.toml                    # Package configuration
├── Dockerfile
├── docker-compose.yml
├── AGENTS.md
├── CHANGELOG.md
└── README.md
```

### 5.2 Component Architecture Diagram

```bash
┌─────────────────────────────────────────────────────────────────────┐
│                          JUNIPER DATA                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌───────────────────────────────────────────────────────────────┐ │
│   │                      REST API Layer                           │ │
│   │                                                               │ │
│   │  GET  /v1/health                                              │ │
│   │  GET  /v1/generators                                          │ │
│   │  GET  /v1/generators/{name}/schema                            │ │
│   │  POST /v1/datasets                                            │ │
│   │  GET  /v1/datasets/{id}                                       │ │
│   │  GET  /v1/datasets/{id}/artifact                              │ │
│   │  GET  /v1/datasets/{id}/preview                               │ │
│   │  DELETE /v1/datasets/{id}                                     │ │
│   └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│   ┌──────────────────────────▼────────────────────────────────────┐ │
│   │                    Generator Registry                         │ │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │ │
│   │  │   Spiral    │  │    XOR      │  │   Circles   │  ...       │ │
│   │  │ Generator   │  │ Generator   │  │  Generator  │            │ │
│   │  │ (NumPy)     │  │ (NumPy)     │  │  (NumPy)    │            │ │
│   │  └─────────────┘  └─────────────┘  └─────────────┘            │ │
│   └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│   ┌──────────────────────────▼────────────────────────────────────┐ │
│   │                   Storage Abstraction                         │ │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │ │
│   │  │   Memory    │  │  LocalFS    │  │   (Future)  │            │ │
│   │  │   Store     │  │   Store     │  │  S3/DB/etc  │            │ │
│   │  └─────────────┘  └─────────────┘  └─────────────┘            │ │
│   └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
┌──────────────────┐          ┌──────────────────────┐
│  Juniper Cascor  │          │   Juniper Canopy     │
│  (Training)      │          │   (Frontend)         │
│                  │          │                      │
│  - CCN Training  │          │  - Dataset Preview   │
│  - Evaluation    │          │  - Training Monitor  │
│  - torch tensors │          │  - Visualization     │
└──────────────────┘          └──────────────────────┘
```

### 5.3 Data Flow

```bash
┌──────────────────────────────────────────────────────────────────┐
│                        DATA FLOW                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. CLIENT REQUEST                                               │
│     POST /v1/datasets                                            │
│     {                                                            │
│       "generator": "spiral",                                     │
│       "params": {                                                │
│         "n_spirals": 2,                                          │
│         "n_points_per_spiral": 100,                              │
│         "noise": 0.1,                                            │
│         "seed": 42                                               │
│       }                                                          │
│     }                                                            │
│                         │                                        │
│                         ▼                                        │
│  2. GENERATE + COMPUTE ID                                        │
│     dataset_id = hash(generator + version + canonical_params)    │
│     X, y = SpiralGenerator.generate(params)                      │
│     X_train, y_train, X_test, y_test = split(X, y, ratios)       │
│                         │                                        │
│                         ▼                                        │
│  3. STORE ARTIFACT                                               │
│     /data/datasets/{dataset_id}.npz                              │
│     /data/datasets/{dataset_id}.meta.json                        │
│                         │                                        │
│                         ▼                                        │
│  4. RESPONSE                                                     │
│     {                                                            │
│       "dataset_id": "spiral-v1.0.0-abc123def456",                │
│       "generator": "spiral",                                     │
│       "meta": { ... },                                           │
│       "artifact_url": "/v1/datasets/{id}/artifact"               │
│     }                                                            │
│                         │                                        │
│                         ▼                                        │
│  5. CLIENT DOWNLOAD                                              │
│     GET /v1/datasets/{id}/artifact                               │
│     → Returns: .npz file with X_train, y_train, X_test, y_test   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. Dataset Contract Specification

### 6.1 Dataset ID Generation

Dataset IDs are deterministic hashes ensuring reproducibility:

```python
import hashlib
import json

def generate_dataset_id(
    generator: str,
    version: str,
    params: dict
) -> str:
    """Generate deterministic dataset ID from generator + params."""
    canonical = json.dumps({
        "generator": generator,
        "version": version,
        "params": dict(sorted(params.items()))
    }, sort_keys=True, separators=(',', ':'))

    hash_digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return f"{generator}-{version}-{hash_digest}"

# Example: "spiral-v1.0.0-a3f8e12b4c567890"
```

### 6.2 Metadata Schema

```python
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class DatasetMeta(BaseModel):
    """Dataset metadata (always small, JSON-safe)."""

    # Identity
    dataset_id: str
    generator: str
    generator_version: str

    # Generation Parameters
    params: Dict[str, any]

    # Shape Information
    n_samples: int
    n_features: int
    n_classes: int
    n_train: int
    n_test: int

    # Class Distribution
    class_distribution: Dict[int, int]

    # Artifacts
    artifact_formats: List[str]  # ["npz"]

    # Timestamps
    created_at: datetime

    # Optional
    checksum: Optional[str] = None
```

### 6.3 Artifact Schema (NPZ)

```python
# NPZ file contents
{
    "X_train": np.ndarray,  # Shape: (n_train, n_features), dtype: float32
    "y_train": np.ndarray,  # Shape: (n_train, n_classes), dtype: float32 (one-hot)
    "X_test": np.ndarray,   # Shape: (n_test, n_features), dtype: float32
    "y_test": np.ndarray,   # Shape: (n_test, n_classes), dtype: float32 (one-hot)

    # Optional full dataset (if requested)
    "X_full": np.ndarray,   # Shape: (n_samples, n_features)
    "y_full": np.ndarray,   # Shape: (n_samples, n_classes)
}
```

### 6.4 SpiralParams Schema

```python
from pydantic import BaseModel, Field, field_validator

class SpiralParams(BaseModel):
    """Parameters for spiral dataset generation."""

    n_spirals: int = Field(default=2, ge=2, le=10)
    n_points_per_spiral: int = Field(default=100, ge=10, le=10000)
    n_rotations: float = Field(default=3.0, ge=0.5, le=10.0)
    noise: float = Field(default=0.25, ge=0.0, le=2.0)
    clockwise: bool = Field(default=True)
    seed: Optional[int] = Field(default=42)
    train_ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    test_ratio: float = Field(default=0.2, gt=0.0, lt=1.0)
    shuffle: bool = Field(default=True)

    @field_validator('train_ratio', 'test_ratio')
    def validate_ratios_sum(cls, v, values):
        # Ensure ratios sum to <= 1.0
        pass
```

---

## 7. Implementation Phases

### Phase 0: Baseline Inventory & Contract Lock (S, 0.5-1 day)

**Goal**: Freeze v1 contract and capture golden reference datasets.

#### Step 0.1: Generate Golden Reference Datasets

**Description**: Run existing Cascor spiral generation with fixed parameters to create reference datasets for parity testing.

**Tasks**:

1. Run `SpiralProblem.generate_n_spiral_dataset()` with:
   - n_spirals=2, n_points=100, noise=0.1, seed=42
   - n_spirals=3, n_points=50, noise=0.05, seed=42
2. Save outputs as NPZ files in `tests/fixtures/golden_datasets/`
3. Document exact shapes, dtypes, and class distributions

**Deliverables**:

- `golden_2spiral_seed42.npz`
- `golden_3spiral_seed42.npz`
- `golden_datasets.md` (documentation)

#### Step 0.2: Finalize Contract Schema

**Description**: Lock the SpiralParams and DatasetMeta schemas.

**Tasks**:

1. Define final Pydantic models
2. Define generator version scheme (`spiral@1.0.0`)
3. Define deterministic ID algorithm

**Deliverables**:

- `juniper_data/core/models.py` (initial)
- `CONTRACT.md` (specification document)

---

### Phase 1: Core Generator Extraction (M, 1-2 days)

**Goal**: Extract spiral generation logic into pure NumPy implementation.

#### Step 1.1: Create Package Structure

**Description**: Set up JuniperData package with core modules.

**Tasks**:

1. Create `juniper_data/` package structure
2. Set up `pyproject.toml` with minimal dependencies
3. Create `__init__.py` files with version

**Deliverables**:

- Complete package structure
- `pyproject.toml`

#### Step 1.2: Implement SpiralParams

**File**: `juniper_data/generators/spiral/params.py`

**Tasks**:

1. Create `SpiralParams` Pydantic model
2. Add validation constraints
3. Add ratio validation

#### Step 1.3: Implement SpiralGenerator

**File**: `juniper_data/generators/spiral/generator.py`

**Description**: Port generation methods from `spiral_problem.py`.

**Implementation Pattern**:

```python
import numpy as np
from typing import Tuple, Dict, Any

class SpiralGenerator:
    """NumPy-only N-spiral dataset generator."""

    VERSION = "1.0.0"

    @staticmethod
    def generate(params: SpiralParams) -> Dict[str, np.ndarray]:
        """
        Generate N-spiral classification dataset.

        Returns:
            Dictionary with X_train, y_train, X_test, y_test arrays
        """
        # Use local RNG for reproducibility
        rng = np.random.default_rng(params.seed)

        # Generate raw coordinates
        X, y = SpiralGenerator._generate_raw(params, rng)

        # Shuffle if requested
        if params.shuffle:
            X, y = SpiralGenerator._shuffle(X, y, rng)

        # Split into train/test
        result = SpiralGenerator._split(X, y, params)

        return result

    @staticmethod
    def _generate_raw(
        params: SpiralParams,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate raw spiral coordinates (NumPy only)."""
        # Implementation ported from spiral_problem.py
        # Uses np.cos, np.sin, np.linspace
        pass

    @staticmethod
    def _make_noise(
        n_points: int,
        noise: float,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Generate noise using local RNG."""
        return rng.standard_normal(n_points) * noise
```

**Extraction Mapping**:

| Source (`spiral_problem.py`)        | Target (`generator.py`)    |
| ----------------------------------- | -------------------------- |
| `generate_raw_spiral_coordinates()` | `_generate_raw()`          |
| `_generate_xy_coordinates()`        | `_generate_spiral_point()` |
| `_make_coords()`                    | `_calculate_coordinates()` |
| `_make_noise()`                     | `_make_noise()`            |
| `_create_input_features()`          | `_create_features()`       |
| `_create_one_hot_targets()`         | `_create_one_hot_labels()` |

#### Step 1.4: Implement Split Utilities

**File**: `juniper_data/core/split.py`

**Tasks**:

1. Implement `shuffle_data(X, y, rng)` (NumPy-only)
2. Implement `split_data(X, y, train_ratio, test_ratio)`

#### Step 1.5: Implement Defaults Module

**File**: `juniper_data/generators/spiral/defaults.py`

**Tasks**:

1. Define all spiral default constants
2. Define validation bounds

#### Step 1.6: Create Unit Tests

**File**: `tests/unit/test_spiral_generator.py`

**Tests**:

- `test_spiral_shapes`: Verify output dimensions
- `test_one_hot_encoding`: Verify class encoding
- `test_determinism`: Same seed → identical output
- `test_split_sizes`: Train/test sizes match ratios
- `test_param_validation`: Invalid params raise errors

---

### Phase 2: JuniperData REST API v1 (M, 1-2 days)

**Goal**: Build FastAPI service with minimal endpoint set.

#### Step 2.1: Implement Storage Interface

**File**: `juniper_data/storage/base.py`

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict

class DatasetStore(ABC):
    """Abstract dataset storage interface."""

    @abstractmethod
    def save(self, dataset_id: str, meta: dict, arrays: dict) -> None:
        """Save dataset metadata and arrays."""
        pass

    @abstractmethod
    def get_meta(self, dataset_id: str) -> Optional[dict]:
        """Get dataset metadata."""
        pass

    @abstractmethod
    def get_artifact(self, dataset_id: str) -> Optional[bytes]:
        """Get dataset artifact (NPZ bytes)."""
        pass

    @abstractmethod
    def exists(self, dataset_id: str) -> bool:
        """Check if dataset exists."""
        pass

    @abstractmethod
    def delete(self, dataset_id: str) -> bool:
        """Delete dataset."""
        pass

    @abstractmethod
    def list_datasets(self, limit: int = 100, offset: int = 0) -> list:
        """List dataset IDs."""
        pass
```

#### Step 2.2: Implement Memory Store

**File**: `juniper_data/storage/memory.py`

For testing and development.

#### Step 2.3: Implement LocalFS Store

**File**: `juniper_data/storage/local_fs.py`

**Storage Layout**:

```bash
{storage_path}/
├── {dataset_id}.meta.json
└── {dataset_id}.npz
```

#### Step 2.4: Create FastAPI Application

**File**: `juniper_data/api/app.py`

```python
from fastapi import FastAPI
from juniper_data.api.routes import health, generators, datasets

def create_app() -> FastAPI:
    app = FastAPI(
        title="Juniper Data",
        description="Dataset generation and management service",
        version="1.0.0"
    )

    app.include_router(health.router, prefix="/v1", tags=["health"])
    app.include_router(generators.router, prefix="/v1", tags=["generators"])
    app.include_router(datasets.router, prefix="/v1", tags=["datasets"])

    return app
```

#### Step 2.5: Implement Endpoints

**Endpoints (v1)**:

| Method | Path                           | Description                    |
| ------ | ------------------------------ | ------------------------------ |
| GET    | `/v1/health`                   | Health check                   |
| GET    | `/v1/generators`               | List available generators      |
| GET    | `/v1/generators/{name}/schema` | Get generator parameter schema |
| POST   | `/v1/datasets`                 | Create/generate dataset        |
| GET    | `/v1/datasets`                 | List datasets                  |
| GET    | `/v1/datasets/{id}`            | Get dataset metadata           |
| GET    | `/v1/datasets/{id}/artifact`   | Download NPZ artifact          |
| GET    | `/v1/datasets/{id}/preview`    | Preview first N samples (JSON) |
| DELETE | `/v1/datasets/{id}`            | Delete dataset                 |

#### Step 2.6: Create API Tests

**File**: `tests/integration/test_api.py`

**Tests**:

- `test_health_endpoint`
- `test_list_generators`
- `test_create_dataset`
- `test_get_dataset_meta`
- `test_download_artifact`
- `test_preview_endpoint`
- `test_dataset_caching` (same params → same ID → no regeneration)

---

### Phase 3: Cascor Integration (M-L, 1-2 days)

**Goal**: Update Cascor to consume datasets from JuniperData.

#### Step 3.1: Add JuniperData Client

**File**: `juniper_cascor/src/juniper_data_client/client.py`

```python
import requests
import numpy as np
from typing import Dict, Optional

class JuniperDataClient:
    """Client for JuniperData service."""

    def __init__(self, base_url: str = "http://localhost:8100"):
        self.base_url = base_url.rstrip("/")

    def create_dataset(
        self,
        generator: str,
        params: dict
    ) -> dict:
        """Create dataset and return metadata."""
        response = requests.post(
            f"{self.base_url}/v1/datasets",
            json={"generator": generator, "params": params}
        )
        response.raise_for_status()
        return response.json()

    def download_arrays(
        self,
        dataset_id: str
    ) -> Dict[str, np.ndarray]:
        """Download and parse NPZ artifact."""
        response = requests.get(
            f"{self.base_url}/v1/datasets/{dataset_id}/artifact"
        )
        response.raise_for_status()

        import io
        return dict(np.load(io.BytesIO(response.content)))
```

#### Step 3.2: Create Compatibility Layer

**File**: `juniper_cascor/src/spiral_problem/data_provider.py`

```python
import os
from typing import Tuple
import torch

class SpiralDataProvider:
    """Provides spiral datasets, using JuniperData when available."""

    def __init__(self):
        self.juniper_data_url = os.environ.get("JUNIPER_DATA_URL")
        self._client = None

    @property
    def use_juniper_data(self) -> bool:
        return bool(self.juniper_data_url)

    def get_dataset(
        self,
        n_spirals: int,
        n_points: int,
        noise: float,
        seed: int,
        train_ratio: float,
        test_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get spiral dataset as torch tensors."""

        if self.use_juniper_data:
            return self._get_from_service(...)
        else:
            return self._get_legacy(...)

    def _get_from_service(self, ...) -> Tuple[...]:
        """Fetch from JuniperData and convert to torch."""
        from juniper_data_client.client import JuniperDataClient

        client = JuniperDataClient(self.juniper_data_url)
        meta = client.create_dataset("spiral", {...params...})
        arrays = client.download_arrays(meta["dataset_id"])

        # Convert to torch tensors
        return (
            torch.tensor(arrays["X_train"], dtype=torch.float32),
            torch.tensor(arrays["y_train"], dtype=torch.float32),
            torch.tensor(arrays["X_test"], dtype=torch.float32),
            torch.tensor(arrays["y_test"], dtype=torch.float32),
        )

    def _get_legacy(self, ...) -> Tuple[...]:
        """Use existing SpiralProblem methods (temporary)."""
        pass
```

#### Step 3.3: Update SpiralProblem

**Description**: Modify `generate_n_spiral_dataset()` to use data provider.

**Changes**:

1. Add `SpiralDataProvider` import
2. Check for `JUNIPER_DATA_URL` environment variable
3. Route to JuniperData or legacy implementation

#### Step 3.4: Run Parity Tests

**Tests**:

- Compare Cascor training results with legacy vs JuniperData datasets
- Verify same seed produces compatible training behavior
- Document any acceptable tolerances

---

### Phase 4: Canopy Integration (M, 1 day)

**Goal**: Replace Canopy mock generators with JuniperData calls.

#### Step 4.1: Update DemoMode

**File**: `juniper_canopy/src/demo_mode.py`

**Changes**:

1. Replace `_generate_spiral_dataset()` with JuniperData client call
2. Use `/v1/datasets/{id}/preview` for quick display data

#### Step 4.2: Update CascorIntegration Fallback

**File**: `juniper_canopy/src/backend/cascor_integration.py`

**Changes**:

1. Replace `_generate_missing_dataset_info()` with JuniperData call
2. Handle service unavailable gracefully

---

### Phase 5: Extended Data Sources (L-XL, Staged - Future)

**Goal**: Add additional data sources when needed.

**Trigger Conditions** (implement only when one is true):

- ≥2 real dataset sources in use
- Need dataset lineage/versioning
- Datasets exceed LocalFS practical limits
- Need multi-user access control

**Potential Extensions**:

| Source            | Priority | Trigger                                |
| ----------------- | -------- | -------------------------------------- |
| Local file import | P1       | User wants to upload existing datasets |
| URL download      | P1       | User wants to fetch from web           |
| Database          | P2       | Integration with data warehouse        |
| Hugging Face      | P2       | ML dataset ecosystem access            |
| S3/GCS            | P3       | Cloud storage requirements             |

---

## 8. Migration Strategy

### 8.1 Deployment Sequence

```bash
Week 1:
├── Day 1-2: Phase 1 (Core Generator)
│   └── JuniperData generator core tested and ready
├── Day 3-4: Phase 2 (REST API)
│   └── JuniperData service deployed (dev/staging)
└── Day 5: Phase 0 (Golden datasets captured if not done earlier)

Week 2:
├── Day 1-2: Phase 3 (Cascor Integration)
│   ├── Feature flag: JUNIPER_DATA_URL
│   ├── Cascor updated with data provider
│   └── Parity testing complete
├── Day 3: Phase 4 (Canopy Integration)
│   └── Canopy updated to use JuniperData
└── Day 4-5: Stabilization & documentation
```

### 8.2 Feature Flag Strategy

**Environment Variables**:

```bash
# Enable JuniperData integration (set to empty to use legacy)
export JUNIPER_DATA_URL=http://juniper-data:8100

# Development/testing
export JUNIPER_DATA_URL=""  # Falls back to legacy generator
```

### 8.3 Backward Compatibility Window

| Phase                  | Cascor Behavior                | Canopy Behavior              |
| ---------------------- | ------------------------------ | ---------------------------- |
| Pre-migration          | Legacy generator               | Mock generators              |
| Dual-path (2 releases) | JuniperData or legacy via flag | JuniperData or mock via flag |
| Post-migration         | JuniperData only               | JuniperData only             |

### 8.4 Rollback Plan

If issues arise:

1. Set `JUNIPER_DATA_URL=""` to disable JuniperData
2. Legacy generators remain functional during dual-path period
3. No data loss (datasets stored independently)

---

## 9. Testing & Validation Requirements

### 9.1 Unit Tests (JuniperData)

| Test                      | Description                    | Assertion                         |
| ------------------------- | ------------------------------ | --------------------------------- |
| `test_spiral_shapes`      | Verify X, y dimensions         | X: (N, 2), y: (N, n_spirals)      |
| `test_one_hot_encoding`   | Each row sums to 1             | `y.sum(axis=1) == 1` for all rows |
| `test_determinism`        | Same seed → identical arrays   | Bitwise equality                  |
| `test_split_sizes`        | Train/test counts match ratios | Within ±1 due to rounding         |
| `test_param_validation`   | Invalid params raise errors    | ValidationError raised            |
| `test_class_distribution` | Classes balanced               | Each class ≈ n_points_per_spiral  |

### 9.2 Contract Tests (API)

| Test                         | Description                                    |
| ---------------------------- | ---------------------------------------------- |
| `test_create_returns_schema` | POST returns correct meta structure            |
| `test_artifact_npz_keys`     | NPZ contains X_train, y_train, X_test, y_test  |
| `test_artifact_dtypes`       | All arrays float32                             |
| `test_caching_same_id`       | Same params → same dataset_id, no regeneration |
| `test_preview_format`        | Preview returns JSON with sample data          |

### 9.3 Parity Tests (Cascor ↔ JuniperData)

| Test                        | Description                   |
| --------------------------- | ----------------------------- |
| `test_shape_parity`         | Shapes match legacy generator |
| `test_distribution_parity`  | Class distribution matches    |
| `test_coordinate_range`     | X values in expected range    |
| `test_training_equivalence` | Training convergence similar  |

### 9.4 E2E Smoke Tests

| Test                       | Description                                |
| -------------------------- | ------------------------------------------ |
| `test_cascor_training_e2e` | Full training run with JuniperData dataset |
| `test_canopy_preview_e2e`  | Preview loads in UI                        |
| `test_service_health`      | JuniperData responds to health check       |

### 9.5 Test Commands

```bash
# JuniperData tests
cd JuniperData
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/ -v --cov=juniper_data

# Cascor integration tests
cd JuniperCascor/juniper_cascor/src
JUNIPER_DATA_URL=http://localhost:8100 pytest tests/integration/test_data_provider.py -v

# Parity tests
pytest tests/parity/ -v
```

---

## 10. Risk Assessment & Guardrails

### 10.1 Technical Risks

| Risk                                          | Impact | Likelihood | Mitigation                                     |
| --------------------------------------------- | ------ | ---------- | ---------------------------------------------- |
| Behavior drift in dataset generation          | High   | Medium     | Golden dataset comparison tests                |
| Large JSON payloads causing slowdowns         | Medium | Low        | Artifact-first design (NPZ download)           |
| RNG differences (np.random.seed vs Generator) | Medium | Medium     | Use `np.random.default_rng(seed)` consistently |
| Service unavailability                        | Medium | Medium     | Feature flag fallback to legacy                |

### 10.2 Project Risks

| Risk                                 | Impact | Likelihood | Mitigation                                  |
| ------------------------------------ | ------ | ---------- | ------------------------------------------- |
| Scope creep (too many sources early) | Medium | High       | Phase 5 deferred; explicit triggers defined |
| Integration complexity               | Medium | Medium     | Phased approach; feature flags              |
| Documentation lag                    | Low    | Medium     | Document during implementation              |

### 10.3 Guardrails

1. **Golden Dataset Lock**: Before extraction, capture reference datasets with fixed seeds
2. **Artifact-First API**: Never return large arrays in JSON (use NPZ download)
3. **Provider Minimal Set**: Only Generated + LocalFS until Phase 5 triggers
4. **Feature Flag Required**: Cascor/Canopy integration behind env var during transition
5. **Parity Tests Mandatory**: Must pass before removing legacy generators

### 10.4 RNG Modernization Note

**Important**: The extraction will use `np.random.default_rng(seed)` instead of global `np.random.seed()`. This may cause:

- Different noise patterns for same seed
- Different shuffle orders for same seed

**Mitigation**:

- Document this as intentional modernization
- Golden datasets captured with new RNG
- Accept slight distribution differences in parity tests

---

## 11. Appendices

### Appendix A: Constants Migration Map

| Cascor Constant                            | JuniperData Default                 | Value |
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
| `_SPIRAL_PROBLEM_RANDOM_SEED`              | `SPIRAL_DEFAULT_SEED`               | 42    |

### Appendix B: API Endpoint Reference

```yaml
openapi: 3.0.3
info:
  title: Juniper Data API
  version: 1.0.0

paths:
  /v1/health:
    get:
      summary: Health check
      responses:
        '200':
          description: Service healthy

  /v1/generators:
    get:
      summary: List available generators
      responses:
        '200':
          description: List of generator names and versions

  /v1/generators/{name}/schema:
    get:
      summary: Get generator parameter schema
      parameters:
        - name: name
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: JSON schema for generator params

  /v1/datasets:
    post:
      summary: Create/generate dataset
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                generator:
                  type: string
                params:
                  type: object
      responses:
        '201':
          description: Dataset created
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

  /v1/datasets/{id}:
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
          description: Not found
    delete:
      summary: Delete dataset
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      responses:
        '204':
          description: Deleted
        '404':
          description: Not found

  /v1/datasets/{id}/artifact:
    get:
      summary: Download dataset artifact
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
            enum: [npz]
            default: npz
      responses:
        '200':
          description: NPZ file
          content:
            application/octet-stream:
              schema:
                type: string
                format: binary

  /v1/datasets/{id}/preview:
    get:
      summary: Preview dataset (JSON)
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
        - name: n
          in: query
          schema:
            type: integer
            default: 100
      responses:
        '200':
          description: Sample data as JSON
```

### Appendix C: Docker Compose Configuration

```yaml
version: '3.8'

services:
  juniper-data:
    build:
      context: ./JuniperData
      dockerfile: Dockerfile
    ports:
      - "8100:8100"
    environment:
      - JUNIPER_DATA_STORAGE_PATH=/data/datasets
      - JUNIPER_DATA_LOG_LEVEL=INFO
    volumes:
      - juniper-data-storage:/data/datasets
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8100/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  juniper-cascor:
    build:
      context: ./JuniperCascor
      dockerfile: Dockerfile
    environment:
      - JUNIPER_DATA_URL=http://juniper-data:8100
    depends_on:
      juniper-data:
        condition: service_healthy

  juniper-canopy:
    build:
      context: ./JuniperCanopy
      dockerfile: Dockerfile
    ports:
      - "8050:8050"
    environment:
      - JUNIPER_DATA_URL=http://juniper-data:8100
      - JUNIPER_CASCOR_URL=http://juniper-cascor:8000
    depends_on:
      juniper-data:
        condition: service_healthy

volumes:
  juniper-data-storage:
```

### Appendix D: Example Usage

**Generate dataset via API**:

```bash
# Create spiral dataset
curl -X POST http://localhost:8100/v1/datasets \
  -H "Content-Type: application/json" \
  -d '{
    "generator": "spiral",
    "params": {
      "n_spirals": 2,
      "n_points_per_spiral": 100,
      "noise": 0.1,
      "seed": 42
    }
  }'

# Response:
# {
#   "dataset_id": "spiral-v1.0.0-a3f8e12b4c567890",
#   "generator": "spiral",
#   "meta": {
#     "n_samples": 200,
#     "n_features": 2,
#     "n_classes": 2,
#     ...
#   },
#   "artifact_url": "/v1/datasets/spiral-v1.0.0-a3f8e12b4c567890/artifact"
# }

# Download artifact
curl http://localhost:8100/v1/datasets/spiral-v1.0.0-a3f8e12b4c567890/artifact \
  -o dataset.npz

# Preview
curl http://localhost:8100/v1/datasets/spiral-v1.0.0-a3f8e12b4c567890/preview?n=10
```

**Use in Python (Cascor)**:

```python
from juniper_data_client import JuniperDataClient

client = JuniperDataClient("http://localhost:8100")

# Create dataset
meta = client.create_dataset("spiral", {
    "n_spirals": 2,
    "n_points_per_spiral": 100,
    "noise": 0.1,
    "seed": 42
})

# Download arrays
arrays = client.download_arrays(meta["dataset_id"])

# Convert to torch
import torch
X_train = torch.tensor(arrays["X_train"], dtype=torch.float32)
y_train = torch.tensor(arrays["y_train"], dtype=torch.float32)
```

---

## 12. Implementation Status

### Phase 0: Complete ✅

**Completed 2026-01-29**

- ☑ Created `tests/fixtures/generate_golden_datasets.py` - Script to generate reference datasets from Cascor
- ☑ Created `tests/fixtures/golden_datasets/README.md` - Documentation for golden datasets
- ☑ Defined contract schema (SpiralParams, DatasetMeta models)

### Phase 1: Complete ✅

**Completed 2026-01-29**

- ☑ Created JuniperData package structure at `Juniper/JuniperData/`
- ☑ Created `pyproject.toml` with dependencies and build configuration
- ☑ Implemented `juniper_data/generators/spiral/defaults.py` - All spiral default constants
- ☑ Implemented `juniper_data/generators/spiral/params.py` - SpiralParams Pydantic model
- ☑ Implemented `juniper_data/generators/spiral/generator.py` - SpiralGenerator (NumPy-only)
- ☑ Implemented `juniper_data/core/split.py` - shuffle_data, split_data, shuffle_and_split
- ☑ Implemented `juniper_data/core/dataset_id.py` - Deterministic ID generation
- ☑ Implemented `juniper_data/core/models.py` - DatasetMeta, CreateDatasetRequest/Response
- ☑ Implemented `juniper_data/core/artifacts.py` - NPZ save/load, checksum
- ☑ Created 60 unit tests (all passing)

### Phase 2: Complete ✅

**Completed 2026-01-29**

- ☑ Implemented `juniper_data/storage/base.py` - DatasetStore abstract base class
- ☑ Implemented `juniper_data/storage/memory.py` - InMemoryDatasetStore
- ☑ Implemented `juniper_data/storage/local_fs.py` - LocalFSDatasetStore
- ☑ Implemented `juniper_data/api/settings.py` - Pydantic-settings configuration
- ☑ Implemented `juniper_data/api/routes/health.py` - GET /v1/health
- ☑ Implemented `juniper_data/api/routes/generators.py` - Generator listing and schema
- ☑ Implemented `juniper_data/api/routes/datasets.py` - Full CRUD + artifact + preview
- ☑ Implemented `juniper_data/api/app.py` - FastAPI app factory
- ☑ Implemented `juniper_data/__main__.py` - CLI entry point
- ☑ Created 16 integration tests (all passing)

**Total: 76 tests passing**

### Phase 3: Pending 🔄

- ☐ Add JuniperDataClient to Cascor
- ☐ Create SpiralDataProvider compatibility layer
- ☐ Update SpiralProblem to use data provider
- ☐ Run parity tests

### Phase 4: Pending 🔄

- ☐ Update Canopy DemoMode
- ☐ Update CascorIntegration fallback

### Phase 5: Deferred ⏳

- ☐ Additional data sources (when triggered)

---

## Document History

| Version | Date       | Author           | Changes                                       |
| ------- | ---------- | ---------------- | --------------------------------------------- |
| 1.0.0   | 2026-01-29 | Juniper Dev Team | Initial comprehensive plan                    |
| 1.1.0   | 2026-01-29 | Juniper Dev Team | Phases 0-2 complete, 76 tests passing         |

---

**Status**: 🔄 **In Progress - Phases 0-2 Complete**

**Completed**:

1. ☑ Review and approve this plan
2. ☑ Create JuniperData repository structure
3. ☑ Phase 0: Baseline Inventory & Contract Lock
4. ☑ Phase 1: Core Generator Extraction
5. ☑ Phase 2: JuniperData REST API v1

**Next Steps**:

1. ☐ Phase 3: Cascor Integration (add JuniperDataClient)
2. ☐ Phase 4: Canopy Integration
3. ☐ Run end-to-end validation with Cascor training
