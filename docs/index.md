# Juniper Cascor Documentation

**Version**: 0.5.2 | [Changelog](../CHANGELOG.md) | [Full Documentation Overview](DOCUMENTATION_OVERVIEW.md)

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
| [Quick Start](install/quick-start.md) | Get up and running in minutes |
| [Environment Setup](install/environment-setup.md) | Detailed environment configuration |
| [User Manual](install/user-manual.md) | Comprehensive usage instructions |
| [Configuration Reference](install/reference.md) | CLI arguments and environment variables |

### API Documentation

| Document | Description |
|----------|-------------|
| [API Reference](api/api-reference.md) | Complete API documentation with examples |
| [API Schemas](api/api-schemas.md) | HDF5 schemas and data structures |

### Configuration

| Document | Description |
|----------|-------------|
| [Constants Guide](overview/constants-guide.md) | Project constants and override methods |

### Testing

| Document | Description |
|----------|-------------|
| [Testing Quick Start](testing/quick-start.md) | Run tests quickly |
| [Testing Environment](testing/environment-setup.md) | Test environment configuration |
| [Testing Manual](testing/manual.md) | Writing and organizing tests |
| [Testing Reference](testing/reference.md) | Markers, reports, and CI mapping |
| [Selective Testing](testing/selective-testing-guide.md) | Run specific test categories |

### CI/CD

| Document | Description |
|----------|-------------|
| [CI/CD Quick Start](ci/quick-start.md) | Understand the CI pipeline |
| [CI Environment](ci/environment-setup.md) | GitHub Actions environment |
| [CI/CD Manual](ci/manual.md) | Pipeline architecture and jobs |
| [CI/CD Reference](ci/reference.md) | Configuration reference |

### Source Code

| Document | Description |
|----------|-------------|
| [Source Quick Start](source/quick-start.md) | Developer onboarding |
| [Source Environment](source/environment-setup.md) | Development tools setup |
| [Source Manual](source/manual.md) | Module-by-module guide |
| [Source Reference](source/reference.md) | Internal conventions |

---

## Key Modules

| Module | Purpose |
|--------|---------|
| `cascade_correlation/` | Core neural network implementation |
| `candidate_unit/` | Candidate unit for network growth |
| `spiral_problem/` | Two-spiral classification benchmark |
| `cascor_constants/` | Project-wide constants |
| `log_config/` | Logging configuration and custom logger |
| `profiling/` | Performance profiling infrastructure |
| `snapshots/` | HDF5 serialization system |
| `remote_client/` | Remote multiprocessing client |
| `utils/` | Utility functions |

---

## Quick Links

- **Run the application**: `cd src && python main.py`
- **Run tests**: `cd src/tests && bash scripts/run_tests.bash`
- **View coverage**: `open src/tests/reports/htmlcov/index.html`

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
