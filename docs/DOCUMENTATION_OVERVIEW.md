# Documentation Overview

## Complete Navigation Guide to Juniper Cascor Documentation

**Version:** 0.6.3  
**Last Updated:** February 1, 2026  
**Project:** Juniper Cascor - Cascade Correlation Neural Network Implementation

---

## Table of Contents

- [Quick Navigation](#quick-navigation)
- [Getting Started](#getting-started)
- [Core Documentation](#core-documentation)
- [Technical Guides](#technical-guides)
- [Development Resources](#development-resources)
- [Historical Documentation](#historical-documentation)
- [Document Index](#document-index)
- [Documentation Standards](#documentation-standards)

---

## Quick Navigation

### I'm New Here - Where Do I Start?

```bash
1. README.md                     → Project overview, what is this?
2. docs/install/QUICK_START.md   → Get running in 5 minutes
3. docs/install/ENVIRONMENT_SETUP.md → Set up your environment
4. AGENTS.md                     → Development conventions and guides
```

### I Want To

| Goal | Document | Location |
|------|----------|----------|
| **Get the app running** | [QUICK_START.md](install/QUICK_START.md) | docs/install/ |
| **Understand the project** | [README.md](../README.md) | Root |
| **Set up my environment** | [ENVIRONMENT_SETUP.md](install/ENVIRONMENT_SETUP.md) | docs/install/ |
| **Learn the API** | [API_REFERENCE.md](api/API_REFERENCE.md) | docs/api/ |
| **Understand HDF5 schemas** | [API_SCHEMAS.md](api/API_SCHEMAS.md) | docs/api/ |
| **Configure constants** | [CONSTANTS_GUIDE.md](overview/CONSTANTS_GUIDE.md) | docs/overview/ |
| **Run tests** | [QUICK_START.md](testing/QUICK_START.md) | docs/testing/ |
| **Set up test environment** | [ENVIRONMENT_SETUP.md](testing/ENVIRONMENT_SETUP.md) | docs/testing/ |
| **Write tests** | [MANUAL.md](testing/MANUAL.md) | docs/testing/ |
| **Run specific tests** | [SELECTIVE_TESTING_GUIDE.md](testing/SELECTIVE_TESTING_GUIDE.md) | docs/testing/ |
| **Understand CI/CD** | [QUICK_START.md](ci_cd/QUICK_START.md) | docs/ci_cd/ |
| **Configure CI/CD** | [MANUAL.md](ci_cd/MANUAL.md) | docs/ci_cd/ |
| **Navigate the source** | [QUICK_START.md](source/QUICK_START.md) | docs/source/ |
| **Understand modules** | [MANUAL.md](source/MANUAL.md) | docs/source/ |
| **See version history** | [CHANGELOG.md](../CHANGELOG.md) | Root |
| **Quick-reference dev tasks** | [DEVELOPER_CHEATSHEET.md](DEVELOPER_CHEATSHEET.md) | docs/ |
| **Contribute code** | [AGENTS.md](../AGENTS.md) | Root |
| **Generate spiral datasets** | JuniperData service | External |
| **Configure JuniperData URL** | [REFERENCE.md](install/REFERENCE.md) | docs/install/ |

---

## Getting Started

### Essential Documents (Read First)

#### 1. [README.md](../README.md)

**Location:** Root directory  
**Purpose:** Project overview, features, quick start  
**Audience:** Everyone  
**Key Sections:**

- What is Juniper Cascor?
- Quick Start installation and running
- Active research components
- Basic usage examples
- Documentation links
- Thread safety warning

**When to Read:** First time visiting the project

---

#### 2. [install/QUICK_START.md](install/QUICK_START.md)

**Location:** docs/install/ directory  
**Purpose:** Get running in 5 minutes  
**Audience:** New users, developers  
**Key Sections:**

- Prerequisites checklist (Python, Conda)
- Step-by-step installation
- Running the spiral problem evaluation (requires JuniperData service)
- Running tests
- Next steps and links

**When to Read:** When you want to run the application immediately

---

#### 3. [install/ENVIRONMENT_SETUP.md](install/ENVIRONMENT_SETUP.md)

**Location:** docs/install/ directory  
**Purpose:** Complete environment configuration guide  
**Audience:** Developers setting up for the first time  
**Key Sections:**

- System requirements (OS, Python, hardware)
- Conda environment setup
- Manual pip installation
- GPU configuration for CUDA
- Development tools (Black, isort, Flake8, MyPy)
- Troubleshooting common issues

**When to Read:** Before starting development, when environment issues occur

---

#### 4. [AGENTS.md](../AGENTS.md)

**Location:** Root directory  
**Purpose:** AI agent development guide and conventions  
**Audience:** Developers, AI assistants  
**Key Sections:**

- Essential commands
- Environment variables (CASCOR_LOG_LEVEL)
- Key entry points
- Directory structure
- Core components
- Programming conventions
- Testing infrastructure
- Profiling commands

**When to Read:** Before writing any code, when debugging issues

---

## Core Documentation

### Project Information

#### [CHANGELOG.md](../CHANGELOG.md)

**Location:** Root directory  
**Purpose:** Version history and release notes  
**Format:** Keep a Changelog standard  
**Audience:** All users  
**Key Sections:**

- Current version (0.4.1)
- Version history (0.4.0, 0.3.21, 0.3.20, etc.)
- Added/Changed/Fixed per version
- Technical notes and SemVer impact
- Breaking changes and migration guides

**When to Read:**

- After updates/upgrades
- When investigating when a feature was added
- When troubleshooting regressions

**Update Frequency:** Every release, every significant change

---

### Architecture & Design

#### Project Structure

```bash
juniper_cascor/
├── README.md                      ← Start here
├── AGENTS.md                      ← Development guide
├── CHANGELOG.md                   ← Version history
├── conf/                          ← Configuration
│   ├── conda_environment.yaml
│   ├── logging_config.yaml
│   └── script_util.cfg
├── docs/                          ← Technical documentation
│   ├── DOCUMENTATION_OVERVIEW.md  ← You are here
│   ├── INDEX.md                   ← Quick navigation
│   ├── DOCUMENTATION_AUDIT.md     ← Audit report
│   ├── api/                       ← API documentation (2 files)
│   │   ├── API_REFERENCE.md
│   │   └── API_SCHEMAS.md
│   ├── ci_cd/                     ← CI/CD documentation (4 files)
│   │   ├── QUICK_START.md
│   │   ├── ENVIRONMENT_SETUP.md
│   │   ├── MANUAL.md
│   │   └── REFERENCE.md
│   ├── install/                   ← Installation guides (4 files)
│   │   ├── QUICK_START.md
│   │   ├── ENVIRONMENT_SETUP.md
│   │   ├── USER_MANUAL.md
│   │   └── REFERENCE.md
│   ├── overview/                  ← Configuration guides (1 file)
│   │   └── CONSTANTS_GUIDE.md
│   ├── source/                    ← Source code guides (4 files)
│   │   ├── QUICK_START.md
│   │   ├── ENVIRONMENT_SETUP.md
│   │   ├── MANUAL.md
│   │   └── REFERENCE.md
│   └── testing/                   ← Testing documentation (5 files)
│       ├── QUICK_START.md
│       ├── ENVIRONMENT_SETUP.md
│       ├── MANUAL.md
│       ├── REFERENCE.md
│       └── SELECTIVE_TESTING_GUIDE.md
├── notes/                         ← Historical documentation
│   ├── API_REFERENCE.md           ← Original API (v0.3.2)
│   ├── FEATURES_GUIDE.md          ← Feature documentation
│   ├── ARCHITECTURE_GUIDE.md      ← Architecture overview
│   ├── PRE-DEPLOYMENT_ROADMAP-2.md ← Integration roadmap
│   └── ...                        ← Other historical docs
├── src/                           ← Source code
│   ├── main.py                    ← Entry point
│   ├── cascade_correlation/       ← Core network implementation
│   ├── candidate_unit/            ← Candidate unit for growth
│   ├── spiral_problem/            ← Two-spiral benchmark
│   ├── juniper_data_client/       ← JuniperData service client
│   ├── cascor_constants/          ← Project constants
│   ├── log_config/                ← Logging infrastructure
│   ├── profiling/                 ← Profiling tools
│   ├── snapshots/                 ← HDF5 serialization
│   ├── remote_client/             ← Remote multiprocessing
│   ├── utils/                     ← Utility functions
│   └── tests/                     ← Test suite
├── util/                          ← Utility scripts
│   ├── juniper_cascor.bash
│   └── profile_training.bash
└── try                            ← Application launcher
```

**Note:** JuniperData is an external dependency providing dataset generation services.

---

## External Services

### JuniperData

**Type:** External sub-project dependency  
**Purpose:** Dataset generation service for spiral problem evaluation  
**Default URL:** `http://localhost:8100`  
**Required For:** Spiral problem evaluation, dataset generation

The JuniperData service must be running before executing spiral problem evaluations. See the [JuniperData project](https://github.com/pcalnon/juniper-data) for installation and setup instructions.

**Configuration:** Set the JuniperData URL via environment variable or configuration. See [install/REFERENCE.md](install/REFERENCE.md) for details.

### juniper-cascor-client

**Type:** Companion client library
**Purpose:** HTTP/WebSocket client for consuming the juniper-cascor REST API
**PyPI:** `pip install juniper-cascor-client`
**Repository:** [pcalnon/juniper-cascor-client](https://github.com/pcalnon/juniper-cascor-client)

The juniper-cascor-client library provides a Python API for interacting with the juniper-cascor service. Used by juniper-canopy and other consumers that need programmatic access to CasCor training and inference.

---

## Technical Guides

### API Documentation

#### [api/API_REFERENCE.md](api/API_REFERENCE.md)

**Lines:** ~600  
**Purpose:** Complete API documentation with examples  
**Audience:** Developers, integrators  
**Key Sections:**

- API Stability table
- CascadeCorrelationNetwork class
  - Constructor parameters
  - fit(), forward(), get_accuracy() methods
  - save_to_hdf5(), load_from_hdf5() serialization
- CascadeCorrelationConfig class
  - All configuration parameters
  - OptimizerConfig nested class
- CandidateUnit class
- SpiralProblem class
- Serialization API (HDF5Utils, CLI)
- Profiling API (ProfileContext, MemoryTracker)
- Logger API
- Data Classes (TrainingResults, CandidateTrainingResult, etc.)
- Exceptions (ConfigurationError, TrainingError, ValidationError)
- Quick reference code examples

**When to Read:**

- Learning the public API
- Integrating Cascor into another application
- Understanding method signatures and return types

---

#### [api/API_SCHEMAS.md](api/API_SCHEMAS.md)

**Lines:** ~400  
**Purpose:** Data schema documentation for serialization  
**Audience:** Developers working with serialization  
**Key Sections:**

- HDF5 Snapshot Schema
  - Top-level structure diagram
  - Metadata group
  - Architecture group
  - Weights group (including hidden units)
  - Training state group
  - Random state group
  - Compression settings
- Training History Schema
- Configuration Schema
- Data Class Schemas
  - TrainingResults fields
  - CandidateTrainingResult fields
  - ValidateTrainingInputs/Results fields
- Backward Compatibility
  - Version handling
  - Compatibility matrix
  - Migration guidance
  - Checksum verification

**When to Read:**

- Working with HDF5 snapshots
- Debugging serialization issues
- Understanding data structures
- Migrating older snapshot files

---

### Configuration

#### [overview/CONSTANTS_GUIDE.md](overview/CONSTANTS_GUIDE.md)

**Lines:** ~400  
**Purpose:** Complete reference for project constants  
**Audience:** Developers, users configuring the system  
**Key Sections:**

- Constants Hierarchy (directory structure)
- Logging Constants
  - Log level names (TRACE, VERBOSE, DEBUG, etc.)
  - CASCOR_LOG_LEVEL environment variable
- Model Constants
  - Network architecture defaults
  - Training defaults
- Candidate Training Constants
  - Pool configuration
  - Multiprocessing settings
- Serialization Constants
  - Default paths
  - Compression settings
  - HDF5 group names
- Problem Constants (spiral problem defaults)
- Activation Function Constants
- Overriding Constants
  - Configuration objects (recommended)
  - Environment variables
  - Constructor parameters

**When to Read:**

- Customizing network behavior
- Understanding default values
- Configuring logging verbosity

---

### Testing Documentation

#### [testing/QUICK_START.md](testing/QUICK_START.md)

**Lines:** ~200  
**Purpose:** Run tests quickly  
**Audience:** Developers  
**Key Sections:**

- Test category overview (unit, integration, slow, GPU)
- Most common commands
- Understanding test output
- Quick troubleshooting

**When to Read:** First-time running tests, quick validation

---

#### [testing/ENVIRONMENT_SETUP.md](testing/ENVIRONMENT_SETUP.md)

**Lines:** ~300  
**Purpose:** Test environment configuration  
**Audience:** Developers  
**Key Sections:**

- Test dependencies (pytest, pytest-cov, pytest-timeout)
- Test configuration (pytest.ini, conftest.py)
- GPU testing setup
- Coverage configuration
- Multiprocessing considerations
- IDE integration (VS Code, PyCharm)

**When to Read:** Setting up test environment, debugging test failures

---

#### [testing/MANUAL.md](testing/MANUAL.md)

**Lines:** ~500  
**Purpose:** Writing and organizing tests  
**Audience:** Developers  
**Key Sections:**

- Test organization (unit/ vs integration/)
- Writing tests (basic structure, markers, AAA pattern)
- Test fixtures (available fixtures, creating new)
- Mock objects
- Testing specific components
- Test data (pre-generated, creating)
- Common testing patterns (random seeds, exceptions, async)

**When to Read:** Writing new tests, understanding test patterns

---

#### [testing/REFERENCE.md](testing/REFERENCE.md)

**Lines:** ~300  
**Purpose:** Complete testing reference  
**Audience:** Developers  
**Key Sections:**

- Complete marker reference table (13 markers)
- Report locations (HTML, XML, JUnit)
- Test command reference
- CI test matrix
- Timeout reference
- Exit codes

**When to Read:** Looking up markers, understanding CI behavior

---

#### [testing/SELECTIVE_TESTING_GUIDE.md](testing/SELECTIVE_TESTING_GUIDE.md)

**Lines:** ~300  
**Purpose:** Run specific test categories  
**Audience:** Developers  
**Key Sections:**

- Running by category (unit, integration, slow, GPU)
- Running by module/file
- Running slow tests safely
- Running with coverage
- Parallel test execution
- Re-running failed tests
- Performance testing
- Common recipes

**When to Read:** Running targeted tests, optimizing test time

---

### CI/CD Documentation

#### [ci_cd/QUICK_START.md](ci_cd/QUICK_START.md)

**Lines:** ~200  
**Purpose:** Understand the CI pipeline  
**Audience:** Developers  
**Key Sections:**

- Pipeline overview (what runs when)
- Triggering CI
- Checking results
- Reproducing CI locally
- Quick fixes for common failures
- Publishing packages to PyPI (Release tags + Trusted Publishing)

**When to Read:** First-time understanding CI, debugging CI failures

---

#### [ci_cd/ENVIRONMENT_SETUP.md](ci_cd/ENVIRONMENT_SETUP.md)

**Lines:** ~300  
**Purpose:** CI environment configuration  
**Audience:** DevOps, maintainers  
**Key Sections:**

- GitHub Actions environment
- Environment creation in CI
- Environment variables
- Local reproduction
- Troubleshooting

**When to Read:** Modifying CI, debugging environment issues

---

#### [ci_cd/MANUAL.md](ci_cd/MANUAL.md)

**Lines:** ~400  
**Purpose:** Pipeline architecture and jobs  
**Audience:** Developers, DevOps  
**Key Sections:**

- Pipeline architecture
- Job details (lint, test, integration, quality-gate, notify)
- Coverage handling
- Slow test handling (CASCOR-TIMEOUT-001)
- Modifying the pipeline
- PyPI publishing (OIDC Trusted Publishing, tag guards, TestPyPI verify)

**When to Read:** Understanding CI deeply, making CI changes

---

#### [ci_cd/REFERENCE.md](ci_cd/REFERENCE.md)

**Lines:** ~300  
**Purpose:** CI configuration reference  
**Audience:** All  
**Key Sections:**

- Workflow configuration reference
- Publish workflows (tag guards, OIDC, verify contract)
- Coverage gates (80% aggregate)
- Artifact reference
- Test marker mapping
- Timeout configuration
- Status badges

**When to Read:** Looking up CI settings, configuring thresholds

---

### Source Code Documentation

#### [source/QUICK_START.md](source/QUICK_START.md)

**Lines:** ~200  
**Purpose:** Developer onboarding  
**Audience:** New contributors  
**Key Sections:**

- Repository tour
- Common dev commands
- Quick code navigation
- Running a minimal training loop

**When to Read:** First-time contributing, getting oriented

---

#### [source/ENVIRONMENT_SETUP.md](source/ENVIRONMENT_SETUP.md)

**Lines:** ~300  
**Purpose:** Development tools setup  
**Audience:** Developers  
**Key Sections:**

- Development tools (Black, isort, Flake8, MyPy)
- IDE configuration
- Pre-commit hooks
- Profiling tools (py-spy, cProfile, tracemalloc)
- Debugging setup

**When to Read:** Setting up dev environment, configuring tools

---

#### [source/MANUAL.md](source/MANUAL.md)

**Lines:** ~500  
**Purpose:** Module-by-module guide  
**Audience:** Contributors  
**Key Sections:**

- Module-by-module overview
  - cascade_correlation/ (core network)
  - candidate_unit/ (candidate units)
  - spiral_problem/ (benchmark)
  - snapshots/ (serialization)
  - cascor_constants/ (constants)
  - log_config/ (logging)
  - profiling/ (profiling tools)
  - remote_client/ (distributed training)
  - utils/ (utilities)
- Code architecture (interaction diagram, data flow)
- Extension points (new problems, activations, serializers)
- Coding conventions (headers, imports, type hints, logging)

**When to Read:** Understanding codebase, adding new features

---

#### [source/REFERENCE.md](source/REFERENCE.md)

**Lines:** ~400  
**Purpose:** Internal conventions  
**Audience:** Contributors  
**Key Sections:**

- Key internal invariants
- Data structures
- Error handling conventions
- Pickling and serialization
- Multiprocessing details
- Performance considerations

**When to Read:** Deep implementation work, debugging internals

---

## Development Resources

### User Manual

#### [install/USER_MANUAL.md](install/USER_MANUAL.md)

**Lines:** ~500  
**Purpose:** Comprehensive usage guide  
**Audience:** Users, researchers  
**Key Sections:**

- Overview of Cascade Correlation Algorithm
- Basic workflow (config → network → train → evaluate)
- Configuration options (CascadeCorrelationConfig, OptimizerConfig)
- Working with data (tensor formats, one-hot encoding)
- Saving and loading networks (HDF5 serialization)
- Visualization (plots)
- Deterministic training (random seeds)
- Example workflows

**When to Read:** Learning to use Cascor, understanding training

---

### Configuration Reference

#### [install/REFERENCE.md](install/REFERENCE.md)

**Lines:** ~300  
**Purpose:** CLI arguments and environment variables  
**Audience:** Users, operators  
**Key Sections:**

- Environment variables (CASCOR_LOG_LEVEL)
- CLI arguments (--profile, --profile-memory, etc.)
- Configuration files
- Directory conventions
- Logging configuration

**When to Read:** Configuring runtime behavior, troubleshooting

---

## Historical Documentation

### notes/ Directory

Contains historical development documentation, implementation notes, and research references. These documents capture the project's evolution and design decisions.

**Location:** [notes/](../notes/)

**Key Historical Documents:**

| Document | Purpose | Status |
|----------|---------|--------|
| `API_REFERENCE.md` | Original API reference (v0.3.2) | Superseded by docs/api/ |
| `FEATURES_GUIDE.md` | Feature documentation | Partially incorporated |
| `ARCHITECTURE_GUIDE.md` | Architecture overview | Historical reference |
| `PRE-DEPLOYMENT_ROADMAP-2.md` | Integration roadmap | Active |
| `IMPLEMENTATION_SUMMARY.md` | Implementation status | Historical |
| `CASCOR_ENHANCEMENTS_ROADMAP.md` | Enhancement roadmap | Historical |

**Purpose:** Historical reference for:

- Original design decisions
- Implementation progress
- Research references
- Bug fix documentation

---

## Document Index

### Root Directory

| File | Lines | Type | Audience | Status |
|------|-------|------|----------|--------|
| **README.md** | ~115 | Overview | All | ✅ **Active** |
| **AGENTS.md** | ~300 | Reference | Developers, AI | ✅ **Active** |
| **CHANGELOG.md** | ~950 | History | All | ✅ **Active** |

### docs/ Directory

| File | Lines | Type | Audience | Status |
|------|-------|------|----------|--------|
| **DOCUMENTATION_OVERVIEW.md** | ~800 | Overview | All | ✅ **Active** |
| **INDEX.md** | ~110 | Navigation | All | ✅ **Active** |
| **DOCUMENTATION_AUDIT.md** | ~200 | Report | Maintainers | ✅ **Active** |
| **DEVELOPER_CHEATSHEET.md** | ~100 | Cheatsheet | Developers | ✅ **Active** |
| **ENVIRONMENT_SETUP.md** | ~10 | Redirect | All | ✅ **Active** — Redirect stub to install/ENVIRONMENT_SETUP.md |
| **USER_MANUAL.md** | ~10 | Redirect | All | ✅ **Active** — Redirect stub to install/USER_MANUAL.md |

### docs/api/ Directory

| File | Lines | Type | Audience | Status |
|------|-------|------|----------|--------|
| **API_REFERENCE.md** | ~600 | Reference | Developers | ✅ **Active** |
| **API_SCHEMAS.md** | ~400 | Reference | Developers | ✅ **Active** |

### docs/install/ Directory

| File | Lines | Type | Audience | Status |
|------|-------|------|----------|--------|
| **QUICK_START.md** | ~150 | Tutorial | New users | ✅ **Active** |
| **ENVIRONMENT_SETUP.md** | ~350 | Guide | Developers | ✅ **Active** |
| **USER_MANUAL.md** | ~500 | Guide | Users | ✅ **Active** |
| **REFERENCE.md** | ~300 | Reference | All | ✅ **Active** |

### docs/overview/ Directory

| File | Lines | Type | Audience | Status |
|------|-------|------|----------|--------|
| **CONSTANTS_GUIDE.md** | ~400 | Reference | Developers | ✅ **Active** |

### docs/testing/ Directory

| File | Lines | Type | Audience | Status |
|------|-------|------|----------|--------|
| **QUICK_START.md** | ~200 | Tutorial | Developers | ✅ **Active** |
| **ENVIRONMENT_SETUP.md** | ~300 | Guide | Developers | ✅ **Active** |
| **MANUAL.md** | ~500 | Guide | Developers | ✅ **Active** |
| **REFERENCE.md** | ~300 | Reference | Developers | ✅ **Active** |
| **SELECTIVE_TESTING_GUIDE.md** | ~300 | Guide | Developers | ✅ **Active** |

### docs/ci_cd/ Directory

| File | Lines | Type | Audience | Status |
|------|-------|------|----------|--------|
| **QUICK_START.md** | ~200 | Tutorial | Developers | ✅ **Active** |
| **ENVIRONMENT_SETUP.md** | ~300 | Guide | DevOps | ✅ **Active** |
| **MANUAL.md** | ~400 | Guide | Developers | ✅ **Active** |
| **REFERENCE.md** | ~300 | Reference | All | ✅ **Active** |
| **BRANCH_PROTECTION.md** | ~100 | Reference | DevOps | ✅ **Active** — GitHub branch protection rules |

### docs/source/ Directory

| File | Lines | Type | Audience | Status |
|------|-------|------|----------|--------|
| **QUICK_START.md** | ~200 | Tutorial | Contributors | ✅ **Active** |
| **ENVIRONMENT_SETUP.md** | ~300 | Guide | Developers | ✅ **Active** |
| **MANUAL.md** | ~500 | Guide | Contributors | ✅ **Active** |
| **REFERENCE.md** | ~400 | Reference | Contributors | ✅ **Active** |

---

## Documentation Standards

### File Naming Conventions

**Active Documentation:**

- Use UPPER_SNAKE_CASE for all documentation files: `QUICK_START.md`, `ENVIRONMENT_SETUP.md`
- Descriptive names indicating content: `API_REFERENCE.md`, `CONSTANTS_GUIDE.md`
- Root-level major files follow the same convention: `README.md`, `CHANGELOG.md`, `AGENTS.md`

**Historical Documentation:**

- Preserved in `notes/` directory
- Original naming conventions maintained
- Include dates for time-sensitive docs

---

### Markdown Formatting

**Required Elements:**

- Title (# heading)
- Version and date metadata
- Table of contents (for docs >200 lines)
- Clear section headings (##, ###)
- Code blocks with language specification
- Links to related documents

**Example:**

```markdown
# Document Title

**Version**: 0.4.1  
**Last Updated**: 2026-01-29  
**Purpose**: Brief description

---

## Table of Contents

- [Section 1](#section-1)
- [Section 2](#section-2)

---

## Section 1

Content...

## Section 2

Content...
```

---

### Cross-Referencing

**Internal Links:**

- Use relative paths: `[README.md](../README.md)`, `[API_REFERENCE.md](api/API_REFERENCE.md)`
- Include section anchors: `[Testing](#testing-documentation)`, `[Quick Start](install/QUICK_START.md#prerequisites)`

**External Links:**

- Use descriptive text: `[PyTorch Documentation](https://pytorch.org/docs/)`

---

### Update Requirements

**On Every Change:**

1. **CHANGELOG.md** - Summarize changes and impact
2. **README.md** - Update if run/test instructions change
3. **Relevant technical docs** - Update affected guides

**Version Bumps:**

- Update version numbers in README.md, CHANGELOG.md
- Add release notes to CHANGELOG.md
- Update "Last Updated" dates

---

## Finding Information

### Search Strategies

**By Topic:**

1. Check this overview's "I Want To..." table
2. Search AGENTS.md for development topics
3. Search docs/ for technical guides
4. Search notes/ for historical context

**By Keyword:**

```bash
# Search all markdown files
grep -r "keyword" *.md docs/**/*.md notes/*.md

# Search with context
grep -r -C 3 "keyword" docs/

# Search specific directory
grep -r "keyword" docs/api/
```

**By Recent Changes:**

1. Check CHANGELOG.md for version history
2. Review git log for recent commits
3. Check DOCUMENTATION_AUDIT.md for audit status

---

## Quick Reference Card

### Essential Commands

```bash
# Get running
cd src && python main.py

# Run fast tests
cd src/tests && bash scripts/run_tests.bash

# Run specific test
pytest unit/test_forward_pass.py -v

# Run with coverage
bash scripts/run_tests.bash -c

# Type check
cd src && python -m mypy cascade_correlation/ candidate_unit/

# Lint
cd src && python -m flake8 . --max-line-length=512

# Format
cd src && python -m black . && python -m isort .

# Profile
cd src && python main.py --profile
```

### Essential Files

```bash
# Start here
README.md                        # What is this?
docs/install/QUICK_START.md      # Get running now
docs/install/ENVIRONMENT_SETUP.md # Set up environment

# Development
AGENTS.md                        # Development guide
docs/api/API_REFERENCE.md        # API documentation
docs/source/MANUAL.md            # Source code guide

# Reference
CHANGELOG.md                     # Version history
docs/overview/CONSTANTS_GUIDE.md # Constants reference
```

---

## Contact & Support

- **Author:** Paul Calnon
- **Project:** Juniper
- **Sub-Project:** juniper_cascor (Juniper Cascor)

**For Documentation Issues:**

1. Check this overview first
2. Search existing docs
3. Consult AGENTS.md for conventions
4. Check CHANGELOG.md for recent changes

---

**Last Updated:** January 29, 2026
**Version:** 0.4.1
**Maintainer:** Paul Calnon

> See the [Juniper Ecosystem Guide](https://github.com/pcalnon/juniper-ml/blob/main/CLAUDE.md) for the full project map and dependency graph.

---

## Recent Updates

### 2026-01-29: Documentation Suite Created

- **Created:** 22 documentation files in `docs/` directory
- **Categories:**
  - Installation & Configuration (4 files)
  - API Documentation (2 files)
  - Testing Documentation (5 files)
  - CI/CD Documentation (4 files)
  - Source Code Documentation (4 files)
  - Overview & Navigation (3 files)
- **Updated:** README.md with Quick Start and documentation links
- **Updated:** CHANGELOG.md with v0.4.1 documentation release
- **API Reference:** Updated from v0.3.2 to v0.3.21
