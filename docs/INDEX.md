# Juniper Cascor Documentation

**Version**: 0.6.3 | [Changelog](../CHANGELOG.md) | [Full Documentation Overview](DOCUMENTATION_OVERVIEW.md)

Juniper Cascor is an AI/ML research platform implementing the **Cascade Correlation Neural Network** algorithm from foundational research (Fahlman & Lebiere, 1990). The project emphasizes ground-up implementations from primary literature, enabling transparent exploration of constructive learning algorithms.

> **📚 New to the docs?** Start with the [Documentation Overview](DOCUMENTATION_OVERVIEW.md) for a complete navigation guide.

---

## Audience

This documentation serves two audiences:

- **Users**: Researchers and practitioners who want to train and evaluate Cascade Correlation networks
- **Contributors**: Developers extending the platform with new features, tests, or optimizations

---

## Documentation

### Getting Started

| Document | Description |
|----------|-------------|
| [Quick Start](install/QUICK_START.md) | Get up and running in minutes |
| [Environment Setup](install/ENVIRONMENT_SETUP.md) | Detailed environment configuration |
| [User Manual](install/USER_MANUAL.md) | Comprehensive usage instructions |
| [Configuration Reference](install/REFERENCE.md) | CLI arguments and environment variables |

### API Documentation

| Document | Description |
|----------|-------------|
| [REST and WebSocket API Reference](api/JUNIPER_CASCOR_API_REFERENCE.md) | Maintained service API documentation with REST endpoints, WebSocket protocols, close codes, and operational constraints |
| [Python API Reference](api/API_REFERENCE.md) | In-process Python API documentation with examples |
| [API Schemas](api/API_SCHEMAS.md) | HDF5, lifecycle, WebSocket, and data structure schemas |

### Configuration

| Document | Description |
|----------|-------------|
| [Constants Guide](overview/CONSTANTS_GUIDE.md) | Project constants and override methods |

### Testing

| Document | Description |
|----------|-------------|
| [Testing Quick Start](testing/QUICK_START.md) | Run tests quickly |
| [Testing Environment](testing/ENVIRONMENT_SETUP.md) | Test environment configuration |
| [Testing Manual](testing/MANUAL.md) | Writing and organizing tests |
| [Testing Reference](testing/REFERENCE.md) | Markers, reports, and CI mapping |
| [Selective Testing](testing/SELECTIVE_TESTING_GUIDE.md) | Run specific test categories |

### CI/CD

| Document | Description |
|----------|-------------|
| [CI/CD Quick Start](ci_cd/QUICK_START.md) | Understand the CI pipeline |
| [CI Environment](ci_cd/ENVIRONMENT_SETUP.md) | GitHub Actions environment |
| [CI/CD Manual](ci_cd/MANUAL.md) | Pipeline architecture and jobs |
| [CI/CD Reference](ci_cd/REFERENCE.md) | Configuration reference |

### Source Code

| Document | Description |
|----------|-------------|
| [Source Quick Start](source/QUICK_START.md) | Developer onboarding |
| [Source Environment](source/ENVIRONMENT_SETUP.md) | Development tools setup |
| [Source Manual](source/MANUAL.md) | Module-by-module guide |
| [Source Reference](source/REFERENCE.md) | Internal conventions |

---

## Key Modules

| Module | Purpose |
|--------|---------|
| `cascade_correlation/` | Core neural network implementation |
| `candidate_unit/` | Candidate unit for network growth |
| `spiral_problem/` | Two-spiral benchmark (requires JuniperData service) |
| `juniper_data_client/` | REST API client for JuniperData service |
| `cascor_constants/` | Project-wide constants |
| `log_config/` | Logging configuration and custom logger |
| `profiling/` | Performance profiling infrastructure |
| `snapshots/` | HDF5 serialization system |
| `remote_client/` | Remote multiprocessing client |
| `utils/` | Utility functions |

---

## External Dependencies

The **JuniperData** service is required for spiral dataset generation. The spiral problem module connects to this REST API service to fetch training and test datasets.

- **Default URL**: `http://localhost:8100`
- **Project**: [JuniperData](https://github.com/pcalnon/juniper-data)

---

## Quick Links

- **Run the application**: `cd src && python main.py`
- **Run tests**: `cd src/tests && bash scripts/run_tests.bash`
- **View coverage**: `open src/tests/reports/htmlcov/index.html`
- **Start JuniperData service for spiral datasets**: See [JuniperData](https://github.com/pcalnon/juniper-data)

---

## Historical Documentation

The `notes/` directory contains historical development documentation, implementation notes, and research references. These documents capture the project's evolution and design decisions.

Key historical documents:

- `notes/API_REFERENCE.md` - Original API reference (v0.3.2)
- `notes/FEATURES_GUIDE.md` - Feature documentation
- `notes/ARCHITECTURE_GUIDE.md` - Architecture overview
- `notes/PRE-DEPLOYMENT_ROADMAP-2.md` - Integration roadmap

---

## License

MIT License — Copyright (c) 2024, 2025, 2026 Paul Calnon
